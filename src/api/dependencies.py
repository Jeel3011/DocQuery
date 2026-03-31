"""
FastAPI dependency injection functions.

Provides shared singletons and per-request auth dependencies.
Uses lazy imports for heavy RAG components to avoid triggering
the entire langchain/transformers import chain at module-load time.
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.components.config import Config

# ─────────────────────────────────────────
# Shared singletons (created once at startup)
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# Auth dependencies
# ─────────────────────────────────────────

security = HTTPBearer()


def get_supabase():
    """Create an unauthenticated SupabaseManager (anon key, no RLS user context)."""
    from src.components.db import SupabaseManager
    return SupabaseManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Validate the Bearer token via Supabase auth (uses anon key \u2014 required for
    auth.get_user()). Returns a SupabaseManager backed by the SERVICE ROLE key
    so all subsequent Storage / PostgREST calls bypass RLS while still being
    scoped to the validated user via sb._user.
    """
    from src.components.db import SupabaseManager, get_supabase_client

    token = credentials.credentials

    # Step 1: Validate token using anon-key client (get_user requires anon key)
    anon_client = get_supabase_client(use_service_role=False)
    try:
        res = anon_client.auth.get_user(token)
        if not res or not res.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

    # Step 2: Return a service-role SupabaseManager for all actual operations.
    # The service role key bypasses RLS so Storage, PostgREST etc. all work
    # without needing to thread the JWT through the Supabase client internals.
    # User identity is confirmed above via get_user(); _user is set here.
    sb = SupabaseManager(use_service_role=True)
    sb._user = res.user
    return sb


# ─────────────────────────────────────────
# User-scoped RAG dependencies (lazy imports)
# ─────────────────────────────────────────

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
    """Return a RetrievalManager scoped to the current user."""
    from src.components.retrieval import RetrievalManager
    return RetrievalManager(user_config)


def get_generator(
    user_config: Config = Depends(get_user_config),
):
    """Return an AnswerGenration instance scoped to the current user."""
    from src.components.genration import AnswerGenration
    return AnswerGenration(user_config)
