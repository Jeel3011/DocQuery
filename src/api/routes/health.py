"""
Health check endpoint - verifies API liveness and dependency reachability.
"""

import asyncio
import os
import logging
from fastapi import APIRouter
from src.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _ping_supabase() -> bool:
    """Return True if Supabase PostgREST is reachable."""
    try:
        from src.components.db import get_supabase_client
        client = get_supabase_client(use_service_role=False)
        client.table("documents").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.warning("Supabase health check failed: %s", e)
        return False


def _ping_pinecone() -> bool:
    """Return True if Pinecone index is reachable."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index_name = os.getenv("PINECONE_INDEX_NAME", "docquery")
        pc.describe_index(index_name)
        return True
    except Exception as e:
        logger.warning("Pinecone health check failed: %s", e)
        return False


@router.get("/health")
async def health_check():
    """Check API health and verify all dependency connections.

    Runs sync dependency pings in a thread pool so they don't block
    the async event loop.
    """
    supabase_ok, pinecone_ok = await asyncio.gather(
        asyncio.to_thread(_ping_supabase),
        asyncio.to_thread(_ping_pinecone),
    )

    all_ok = supabase_ok and pinecone_ok
    status = "ok" if all_ok else "degraded"

    from src.components.circuit_breaker import get_openai_breaker, get_pinecone_breaker
    openai_breaker = get_openai_breaker()
    pinecone_breaker = get_pinecone_breaker()

    return {
        "status": status,
        "version": "0.1.0",
        "dependencies": {
            "supabase": "ok" if supabase_ok else "unreachable",
            "pinecone": "ok" if pinecone_ok else "unreachable",
        },
        "circuit_breakers": {
            "openai": openai_breaker.status,
            "pinecone": pinecone_breaker.status,
        },
    }
