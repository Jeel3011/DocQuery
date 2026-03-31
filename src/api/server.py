"""
DocQuery — FastAPI Backend Server

Production-grade REST API wrapping the RAG pipeline.
Run with: uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.dependencies import init_config
from src.api.routes import health, auth, documents, chat

# S9: disable interactive API docs in production
_IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"

# P1: shared rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    init_config()
    yield


app = FastAPI(
    title="DocQuery API",
    description="Intelligent Document Q&A System — REST API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _IS_PROD else "/docs",    # S9: hidden in prod
    redoc_url=None if _IS_PROD else "/redoc",  # S9: hidden in prod
)

# P1: attach rate-limiter state + 429 Too Many Requests handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dev server
        "http://localhost:3000",   # React / other dev frontend
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
        # Add production domain, e.g.:
        # "https://docquery.yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
app.include_router(auth.router,   prefix=API_PREFIX, tags=["Auth"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])
app.include_router(chat.router,   prefix=API_PREFIX, tags=["Chat"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "DocQuery API",
        "version": "0.1.0",
        "docs": None if _IS_PROD else "/docs",
    }
