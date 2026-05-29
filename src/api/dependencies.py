"""
FastAPI dependency injection functions.

Provides shared singletons and per-request auth dependencies.
Uses lazy imports for heavy RAG components to avoid triggering
the entire langchain/transformers import chain at module-load time.
"""

import os
import asyncio
import threading
from types import SimpleNamespace

import jwt
from jwt import PyJWKClient
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.components.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------
# Per-user rate limiter
# -----------------------------------------
# Keys on the Bearer token (first 50 chars) for authenticated endpoints.
# This is per-session, not per-IP, so it works correctly behind proxies
# and cannot be bypassed by rotating IPs.
# Falls back to remote IP for unauthenticated routes (/auth/login, etc.).

def get_user_key(request: Request) -> str:
    """Rate-limit by Bearer token for authed routes, IP for unauthed."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        # First 50 chars is enough for a unique, non-spoofable key.
        # Full token validation still happens in get_current_user().
        return f"token:{auth[7:57]}"
    return get_remote_address(request)  # fallback: IP for /auth routes


limiter = Limiter(key_func=get_user_key)


# -----------------------------------------
# Shared singletons (created once at startup)
# -----------------------------------------

_config: Config = None


def init_config():
    """Called during app startup to initialize the shared Config."""
    global _config
    _config = Config()


def get_config() -> Config:
    """Return the shared Config singleton."""
    if _config is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _config


# -----------------------------------------
# Auth dependencies
# -----------------------------------------

security = HTTPBearer()


def get_supabase():
    """Create an unauthenticated SupabaseManager (anon key, no RLS user context)."""
    from src.components.db import SupabaseManager
    return SupabaseManager()


# B1: JWKS client for asymmetric (ES256/RS256) verification. Supabase's modern
# signing keys are asymmetric — the public keys are published at the project's
# JWKS endpoint and only the project URL is needed (no secret). PyJWKClient
# fetches once and caches keys (refetching only when an unknown `kid` appears),
# so verification is local after the first request.
_jwks_client: PyJWKClient | None = None
_jwks_lock = threading.Lock()


def _get_jwks_client() -> PyJWKClient | None:
    global _jwks_client
    if _jwks_client is None:
        url = os.getenv("SUPABASE_URL", "").rstrip("/")
        if not url:
            return None
        with _jwks_lock:
            if _jwks_client is None:
                _jwks_client = PyJWKClient(
                    f"{url}/auth/v1/.well-known/jwks.json", cache_keys=True
                )
    return _jwks_client


def _verify_jwt_local(token: str):
    """B1: Verify a Supabase access token locally, avoiding the auth.get_user()
    network round-trip on every request.

    Asymmetric tokens (ES256/RS256 — Supabase's current signing keys) are verified
    against the published JWKS public key. Legacy HS256 tokens are verified with
    SUPABASE_JWT_SECRET if it's configured. Returns a lightweight user object
    (.id/.email) on success, or None when local verification isn't possible (so the
    caller falls back to the network path). An expired token raises 401 directly —
    a clean rejection, never a fallback that would mask it.
    """
    try:
        alg = jwt.get_unverified_header(token).get("alg")
    except jwt.InvalidTokenError:
        return None

    try:
        if alg in ("ES256", "RS256", "EdDSA"):
            client = _get_jwks_client()
            if client is None:
                return None
            signing_key = client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token, signing_key, algorithms=[alg], audience="authenticated"
            )
        elif alg == "HS256":
            secret = os.getenv("SUPABASE_JWT_SECRET", "")
            if not secret:
                return None
            claims = jwt.decode(
                token, secret, algorithms=["HS256"], audience="authenticated"
            )
        else:
            return None
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except HTTPException:
        raise
    except Exception as e:
        # Unknown key / signature mismatch / JWKS fetch failure / malformed —
        # fall back to the network path so verification stays correct.
        logger.debug("Local JWT verify failed, falling back to network: %s", e)
        return None

    sub = claims.get("sub")
    if not sub:
        return None
    return SimpleNamespace(id=sub, email=claims.get("email"))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Validate the Bearer token and return a SupabaseManager backed by the SERVICE
    ROLE key so all subsequent Storage / PostgREST calls bypass RLS while still
    being scoped to the validated user via sb._user.

    B1: tokens are verified locally (HS256) when SUPABASE_JWT_SECRET is set,
    avoiding an ~80–150ms auth.get_user() round-trip on every request. Falls back
    to the network path when local verification isn't available.
    """
    from src.components.db import SupabaseManager, get_supabase_client

    token = credentials.credentials

    # Step 1: Validate the token. Prefer local verification (no per-request hop).
    # Run off the event loop — the JWKS fetch (first request only) is blocking.
    user = await asyncio.to_thread(_verify_jwt_local, token)
    if user is None:
        anon_client = get_supabase_client(use_service_role=False)
        try:
            res = anon_client.auth.get_user(token)
            if not res or not res.user:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            user = res.user
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

    # Step 2: Return a service-role SupabaseManager for all actual operations.
    # The service role key bypasses RLS so Storage, PostgREST etc. all work
    # without needing to thread the JWT through the Supabase client internals.
    # User identity is confirmed above; _user is set here.
    sb = SupabaseManager(use_service_role=True)
    sb._user = user
    return sb


# -----------------------------------------
# User-scoped RAG dependencies (lazy imports + singleton cache)
# -----------------------------------------

# L1 FIX: Cache heavy objects per Pinecone namespace so they aren't
# re-created on every request. The CrossEncoder model (loaded by
# RetrievalManager -> Reranker) takes 1-3s — caching eliminates that.
_retrieval_cache: dict[str, object] = {}
_generator_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def get_user_config(
    sb=Depends(get_current_user),
    config: Config = Depends(get_config),
) -> Config:
    """Create a user-scoped Config with isolated Pinecone namespace."""
    # B5: guard — user_id must always be set before we use it as a Pinecone namespace
    if not sb.user_id:
        raise HTTPException(status_code=401, detail="User ID not available — cannot scope vector namespace.")
    user_config = Config()
    # Namespace uses the user's UUID for strict isolation in Pinecone
    user_config.PINECONE_NAMESPACE = sb.user_id
    return user_config


def get_retrieval_mgr(
    user_config: Config = Depends(get_user_config),
):
    """Return a RetrievalManager scoped to the current user (cached per namespace)."""
    ns = user_config.PINECONE_NAMESPACE
    with _cache_lock:
        if ns not in _retrieval_cache:
            from src.components.retrieval import RetrievalManager
            _retrieval_cache[ns] = RetrievalManager(user_config)
        return _retrieval_cache[ns]


def get_generator(
    user_config: Config = Depends(get_user_config),
):
    """Return an AnswerGenration instance scoped to the current user (cached per namespace)."""
    ns = user_config.PINECONE_NAMESPACE
    with _cache_lock:
        if ns not in _generator_cache:
            from src.components.generation import AnswerGenration
            _generator_cache[ns] = AnswerGenration(user_config)
        return _generator_cache[ns]
