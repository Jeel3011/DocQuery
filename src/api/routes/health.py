"""
Health check endpoint.
"""

from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and version."""
    return HealthResponse(status="ok", version="0.1.0")
