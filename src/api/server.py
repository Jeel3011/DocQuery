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
from src.api.routes import admin as admin_routes
from src.api.routes import collections as collections_routes
from src.api.routes import export as export_routes
from src.api.routes import analytics as analytics_routes
from src.api.routes import document_comparison as comparison_routes
from src.api.routes import audit as audit_routes
from src.api.routes import agent_core as agent_core_routes  # A4: /query/agentcore/stream (flag-gated)
from src.api.middleware import CorrelationIDMiddleware, SecurityHeadersMiddleware

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# Phase 5: Sentry error tracking + performance tracing
# Initialise before anything else so all exceptions are captured.
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        integrations=[FastApiIntegration(), CeleryIntegration()],
        traces_sample_rate=0.1,   # 10% of requests traced (cost control)
        environment="production" if os.getenv("IS_PROD") == "true" else "development",
    )


# S9: disable interactive API docs in production
_IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    init_config()

    # Pre-warm the cross-encoder reranker so the FIRST query doesn't pay the
    # 1-3s model-load (and the Hub network checks) inline. Best-effort: a failure
    # here must never block API startup — the model lazy-loads on first use anyway.
    try:
        from src.api.dependencies import get_config
        cfg = get_config()
        if getattr(cfg, "USE_RERANKER", False):
            import asyncio
            from src.components.reranker import _get_model
            await asyncio.to_thread(_get_model, cfg.RERANKER_MODEL)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Reranker pre-warm failed (non-fatal, will lazy-load on first query): %s", exc
        )

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
    "http://localhost:3001",    # Next.js fallback
    "http://127.0.0.1:3001",
    "http://localhost:3002",    # Next.js fallback when 3000/3001 taken
    "http://127.0.0.1:3002",
    "http://localhost:8501",    # Streamlit (local dev only)
    "http://127.0.0.1:8501",
]
if _frontend_url:
    _cors_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Correlation-ID"],
)

# -- Correlation ID middleware (traces requests across FastAPI -> Celery -> logs) --
app.add_middleware(CorrelationIDMiddleware)

# -- Security headers (X-Content-Type-Options, X-Frame-Options, etc.) --
app.add_middleware(SecurityHeadersMiddleware)

# -- Routes --
API_PREFIX = "/api/v1"

app.include_router(health.router,    prefix=API_PREFIX, tags=["Health"])
app.include_router(auth.router,      prefix=API_PREFIX, tags=["Auth"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])
app.include_router(chat.router,      prefix=API_PREFIX, tags=["Chat"])
app.include_router(admin_routes.router, prefix=API_PREFIX, tags=["Admin"])   # Phase 5
app.include_router(collections_routes.router, prefix=API_PREFIX, tags=["Collections"])  # Phase 1
app.include_router(export_routes.router, prefix=API_PREFIX, tags=["Export"])  # Phase 3
app.include_router(analytics_routes.router, prefix=API_PREFIX, tags=["Analytics"])  # Phase 4
app.include_router(comparison_routes.router, prefix=API_PREFIX, tags=["Comparison"])  # Phase 5
app.include_router(audit_routes.router, prefix=API_PREFIX, tags=["Audit"])  # Phase 6
app.include_router(agent_core_routes.router, prefix=API_PREFIX, tags=["AgentCore"])  # A4 (flag-gated)


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
