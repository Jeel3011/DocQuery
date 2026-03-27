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
    """Create a fresh SupabaseManager per request."""
    from src.components.db import SupabaseManager
    return SupabaseManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Extract Bearer token from Authorization header,
    validate it against Supabase, and return an authenticated SupabaseManager.
    """
    from src.components.db import SupabaseManager

    token = credentials.credentials
    sb = SupabaseManager()

    try:
        res = sb.client.auth.get_user(token)
        if not res or not res.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        sb._user = res.user
        return sb
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


# ─────────────────────────────────────────
# User-scoped RAG dependencies (lazy imports)
# ─────────────────────────────────────────

def get_user_config(
    sb=Depends(get_current_user),
    config: Config = Depends(get_config),
) -> Config:
    """Create a user-scoped Config with isolated ChromaDB collection."""
    user_config = Config()
    user_config.VECTOR_DB_PATH = f"{config.VECTOR_DB_PATH}_{sb.user_id}"
    user_config.COLLECTION_NAME = f"docquery_{sb.user_id[:8]}"
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
