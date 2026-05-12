"""
DocQuery — FastAPI Backend Server

Production-grade REST API wrapping the RAG pipeline.
Run with: uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import init_config, limiter
from src.api.routes import health, auth, documents, chat
from src.api.middleware import CorrelationIDMiddleware, SecurityHeadersMiddleware

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# S9: disable interactive API docs in production
_IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"


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

# -- CORS -------------------------------------------------------------------------
# Set FRONTEND_URL to your deployed frontend's origin.
# Works with any platform (Vercel, Netlify, etc.) — just set the env var.
_frontend_url = os.getenv("FRONTEND_URL", "")

_cors_origins = [
    "http://localhost:3000",    # React dev server
    "http://127.0.0.1:3000",
    "http://localhost:8501",    # Streamlit (local dev only)
    "http://127.0.0.1:8501",
]
if _frontend_url:
    _cors_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Correlation-ID"],
)

# -- Correlation ID middleware (traces requests across FastAPI -> Celery -> logs) --
app.add_middleware(CorrelationIDMiddleware)

# -- Security headers (X-Content-Type-Options, X-Frame-Options, etc.) --
app.add_middleware(SecurityHeadersMiddleware)

# -- Routes --
API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
app.include_router(auth.router,   prefix=API_PREFIX, tags=["Auth"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])
app.include_router(chat.router,   prefix=API_PREFIX, tags=["Chat"])


# -- Prometheus Metrics --
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics"],   # don't track the metrics endpoint itself
).instrument(app).expose(app, endpoint="/metrics")


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "DocQuery API",
        "version": "0.1.0",
        "docs": None if _IS_PROD else "/docs",
    }
