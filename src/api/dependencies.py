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
    # Carry user_metadata from the verified claims so server-side personalization
    # (e.g. preferred_name) reads from the AUTHENTICATED token, not a request body.
    # Supabase includes user_metadata in the access-token claims.
    meta = claims.get("user_metadata")
    if not isinstance(meta, dict):
        meta = {}
    return SimpleNamespace(id=sub, email=claims.get("email"), user_metadata=meta)


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
    # F1 RLS hardening (defense-in-depth): attach the VERIFIED token so user-facing READS
    # run through an RLS-enforced client (`sb.read_client`) where auth.uid() resolves and the
    # existing `auth.uid() = user_id` policies are a real data-layer backstop — a forgotten
    # app-layer `.eq(user_id)` filter can no longer leak across users. Writes/Storage stay on
    # the service-role `sb.client`. A null token (shouldn't happen post-verify) just keeps the
    # service-role fallback, so nothing breaks.
    sb.attach_access_token(token)
    return sb


# -----------------------------------------
# F2b — central authorization: per-request membership + require_cap guard
# -----------------------------------------
#
# resolve_membership turns the verified user (get_current_user) into a PURE
# authz.Membership (firm + role + caps + screen/delegation seams), resolved SERVER-SIDE,
# PER REQUEST (T7: a revoked role / new screen takes effect on the NEXT request — never
# cached in the JWT). require_cap(verb) is a dependency factory that calls authz.authorize()
# and 403s with the human reason BEFORE the handler runs — the single gatekeeper every
# mutating route trusts (D2), not 40 hand-written ifs.
#
# Legacy parity (the load-bearing invariant): a firm-less or solo Managing-Partner user is
# resolved as an MP-of-one ⇒ authorize() allows everything ⇒ existing routes are
# byte-identical to pre-F2. Migration 012 backfills every legacy user to exactly this, and a
# user whose firm row isn't resolvable (migration not yet applied live) degrades to the same
# implicit solo-MP — so the guard never locks anyone out of what they could already do.


def resolve_membership(sb, firm_id: str | None = None):
    """Build the per-request authz.Membership for the authenticated user (T7 — server-side,
    never from the JWT). Reuses db.get_user_firm (which now returns the role + supports the
    active-firm pick, §0.8). Derives caps from the role via authz.caps_for_role.

    The screen + delegation lookups are SEAMS (F2c / F2e): empty now, wired by those slices —
    authorize() already honors them, so this is where they plug in with no change to the
    decision function.

    Degrades to an implicit solo Managing-Partner when no firm membership is resolvable (legacy
    user pre-backfill, or migration 012 not yet applied live) so behavior is byte-identical to
    pre-F2 — the guard never strips a power the user already had.
    """
    from src.components import authz

    uid = sb.user_id
    role = "managing_partner"
    resolved_firm = firm_id
    try:
        firm = sb.get_user_firm(firm_id=firm_id) if firm_id else sb.get_user_firm()
        if firm:
            resolved_firm = firm.get("id") or resolved_firm
            role = firm.get("role") or role
    except Exception as e:
        # A DB hiccup must not become a 500 on every guarded route — fall back to solo-MP
        # (legacy parity) rather than failing closed and locking the user out.
        logger.debug("resolve_membership: firm lookup failed, defaulting to solo-MP: %s", e)

    # F2c (THE MOAT): the vaults this user is actively screened off (ethical wall) within the
    # resolved firm. authorize() checks these FIRST in its deny-overrides precedence, so a screen
    # beats ANY role grant — even an MP's (deny-overrides-role). Resolved PER REQUEST (T7): a new
    # screen blocks, and a soft-removed one restores, on the NEXT request. Degrades to the empty
    # set (byte-identical to pre-F2c) when no firm / no screens / the table is unapplied.
    screened: frozenset[str] = frozenset()
    try:
        screened = frozenset(sb.screened_vault_ids(firm_id=resolved_firm))
    except Exception as e:
        # A screen-lookup hiccup must NOT fail open (silently dropping the wall) NOR 500 every
        # route. We log and proceed with no screens — the retrieval-layer guard (Layer 2) and the
        # row RLS (F2f) are the defense-in-depth backstops if Layer 1's lookup ever degrades.
        logger.debug("resolve_membership: screen lookup failed, proceeding with no screens: %s", e)

    # F2d (D6): the verbs this user holds via an ACTIVE, un-expired, un-revoked delegation (a PA
    # acting for a senior). authorize() honors these at step 1.5 — a POSITIVE grant that STILL
    # cannot beat a screen DENY (precedence holds: screen is step 1, delegation is step 1.5). The
    # set is BOUNDED to the delegator's own caps inside db.active_delegated_verbs (a delegate can
    # never gain a verb the delegator lacks — T1). Resolved PER REQUEST (T7): a revoked/expired
    # delegation stops granting on the NEXT request; there is no session cache. Degrades to ∅
    # (byte-identical to pre-F2d) when no firm / no delegations / the table is unapplied.
    delegated: frozenset[str] = frozenset()
    try:
        delegated = frozenset(sb.active_delegated_verbs(firm_id=resolved_firm))
    except Exception as e:
        # A delegation-lookup hiccup must NOT 500 every route; proceed with no delegated verbs
        # (the user still has their ROLE caps — we only drop the optional positive grant).
        logger.debug("resolve_membership: delegation lookup failed, proceeding with none: %s", e)

    # F2e (D0/D3): the matters (vaults) this user is STAFFED on. A PRODUCTIVITY seam — being on a
    # matter grants the FULL working toolkit on it (the role already carries the toolkit; this is a
    # scope grant, never a cap restriction). Resolved PER REQUEST (T7): an un-staffed member loses
    # matter access on their NEXT request. Degrades to ∅ (byte-identical to pre-F2e) when no firm /
    # no staffing / the table is unapplied.
    matters: frozenset[str] = frozenset()
    try:
        matters = frozenset(sb.matter_member_vault_ids(firm_id=resolved_firm))
    except Exception as e:
        logger.debug("resolve_membership: matter-team lookup failed, proceeding with none: %s", e)

    return authz.Membership(
        user_id=str(uid) if uid else "",
        firm_id=str(resolved_firm) if resolved_firm else "",
        role=role,
        caps=authz.caps_for_role(role),
        screened_vault_ids=screened,
        delegated_verbs=delegated,
        matter_vault_ids=matters,
    )


# ─────────────────────────────────────────
# F2c Layer 2 — the retrieval/read-layer ethical wall (the load-bearing one).
#   The verb guard (require_cap, Layer 1 via authorize) already denies a screened vault on the
#   resolved Membership. But a wall that covers only the verb is a FALSE wall: a path that resolves
#   a collection_id WITHOUT re-checking (a prompt-injected doc_id, a tool that loads a foreign vault,
#   a worker re-ingest) must STILL return 0 / refuse. So every entry path that resolves a
#   collection_id calls this ONE helper right after it has the id — BEFORE building the RunScope /
#   filename filters / grids — so the wall is in the DATA path, not the prompt (the F1 lesson).
#
#   This is deliberately a plain function (not a Depends) because the collection_id is in the request
#   BODY, resolved inside the handler — not a dependency. It resolves the screen SERVER-SIDE (T5/T3)
#   and raises 403; it never trusts a body-supplied firm_id.
# ─────────────────────────────────────────

def assert_vault_not_screened(sb, collection_id: str | None) -> None:
    """Raise HTTPException(403) if the authenticated user is actively screened off `collection_id`
    (an ethical wall). The retrieval/read-layer floor (F2c P1–P7): called by every route that scopes
    work to a vault, AFTER it resolves the collection_id and BEFORE any retrieval/grid/tool runs.

    Resolves the screen server-side via the per-request membership (firm + screens), so it honors the
    SAME deny-overrides decision authorize() makes — a screen beats any role. A null collection_id is
    a no-op here (the route's own collection_id requirement handles that). Degrades to allow only when
    no screen is resolvable (no firm / table unapplied) — byte-identical to pre-F2c."""
    if not collection_id:
        return
    membership = resolve_membership(sb)
    if str(collection_id) in membership.screened_vault_ids:
        # T10: a wall block is a security event — auditable. Best-effort (never block the 403).
        try:
            from src.api.routes.audit import log_audit
            log_audit(sb, "screen.block", "vault", str(collection_id),
                      {"role": membership.role, "reason": "ethical wall (conflict screen)"})
        except Exception:
            pass
        raise HTTPException(
            status_code=403,
            detail="Access denied by an ethical wall (conflict screen) on this matter.",
        )


def require_cap(verb: str):
    """FastAPI dependency factory: returns a dependency that authorizes `verb` for the caller
    and raises HTTPException(403, reason) on deny BEFORE the handler runs (T9 — the cap check is
    load-bearing on service-role writes until F2f's row RLS lands under it).

    The returned dependency yields the resolved authz.Membership so the handler can reuse it
    (firm_id, role) without a second lookup. For verbs that target a specific vault/firm/role,
    the HANDLER builds a precise authz.Scope from the SERVER-RESOLVED target and re-checks via
    authz.authorize (T2/T3 — never trust a body firm_id); this guard is the route-level floor.

    Solo-MP / legacy users are allowed everything (resolve_membership defaults them to MP), so
    existing routes stay byte-identical.
    """
    def _dep(sb=Depends(get_current_user)):
        from src.components import authz

        membership = resolve_membership(sb)
        decision = authz.authorize(membership, verb)
        if not decision.allow:
            # T10: a deny-of-consequence is auditable. Log best-effort (never block the 403).
            try:
                from src.api.routes.audit import log_audit
                log_audit(sb, "authz.deny", "capability", verb,
                          {"role": membership.role, "reason": decision.reason})
            except Exception:
                pass
            raise HTTPException(status_code=403, detail=decision.reason)
        return membership

    return _dep


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


def get_kb_retrieval_mgr(
    user_config: Config = Depends(get_user_config),
):
    """G8: a RetrievalManager scoped to the SHARED, read-only legal KB namespace
    (`KNOWLEDGE_NAMESPACE`, e.g. `kb_in`) — the corpus `search_knowledge` queries.

    Returns None when USE_KNOWLEDGE is off, so the route threads nothing onto the
    RunScope ⇒ the tool is never offered ⇒ byte-identical to pre-G8. Cached on the KB
    namespace (shared across all users — it is read-only and identical for everyone),
    distinct from the per-user vault caches above. Never targets a user namespace."""
    if not getattr(user_config, "USE_KNOWLEDGE", False):
        return None
    kb_ns = getattr(user_config, "KNOWLEDGE_NAMESPACE", "kb_in")
    cache_key = f"__kb__:{kb_ns}"
    with _cache_lock:
        if cache_key not in _retrieval_cache:
            from copy import copy
            from src.components.retrieval import RetrievalManager
            kb_config = copy(user_config)
            kb_config.PINECONE_NAMESPACE = kb_ns  # the shared KB namespace, NOT the user's
            _retrieval_cache[cache_key] = RetrievalManager(kb_config)
        return _retrieval_cache[cache_key]


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
