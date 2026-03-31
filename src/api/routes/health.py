"""
Health check endpoint \u2014 verifies API liveness and dependency reachability.
"""

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
        # Lightweight query \u2014 just checks connectivity, no data returned
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
    """Check API health and verify all dependency connections."""
    supabase_ok = _ping_supabase()
    pinecone_ok = _ping_pinecone()

    all_ok = supabase_ok and pinecone_ok
    status = "ok" if all_ok else "degraded"

    return {
        "status": status,
        "version": "0.1.0",
        "dependencies": {
            "supabase": "ok" if supabase_ok else "unreachable",
            "pinecone": "ok" if pinecone_ok else "unreachable",
        },
    }
