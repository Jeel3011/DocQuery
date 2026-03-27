"""
DocQuery — FastAPI Backend Server

Production-grade REST API wrapping the RAG pipeline.
Run with: uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import init_config
from src.api.routes import health, auth, documents, chat


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
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
app.include_router(auth.router, prefix=API_PREFIX, tags=["Auth"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])
app.include_router(chat.router, prefix=API_PREFIX, tags=["Chat"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "DocQuery API",
        "version": "0.1.0",
        "docs": "/docs",
    }
