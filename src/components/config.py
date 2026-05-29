from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Project root = two levels up from this file (src/components/config.py -> project root)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

@dataclass
class Config:
    """Centralized configuration"""

    # Embeddings and LLM
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # Chunking params
    CHUNK_SIZE: int = 3000
    NEW_AFTER_N_CHARS: int = 2400
    COMBINE_TEXT_UNDER_N_CHARS: int = 500
    CHUNK_OVERLAP: int = 500

    # ── Document Processing ──
    # PDF_STRATEGY: used for explicit overrides or non-PDF files.
    # For PDFs, _detect_strategy() in data_ingestion.py auto-selects based on page count.
    #   "fast"   — no OCR/layout model, 5-10x faster, text-heavy PDFs ≤ PDF_FAST_THRESHOLD_PAGES
    #   "auto"   — unstructured auto-detects, medium PDFs
    #   "hi_res" — full layout + OCR, scanned/large PDFs > PDF_MEDIUM_THRESHOLD_PAGES
    PDF_STRATEGY: str = "auto"          # fallback for non-PDFs and force-override
    EXTRACT_IMAGES: bool = True          # False = skip image extraction (faster)
    PARALLEL_PDF_PAGES: bool = True      # Process PDF page-ranges in parallel (3-5x faster)
    # A1: match parallelism to the box's vCPUs to avoid oversubscription thrashing.
    # 4 unstructured/YOLOX processes on a 2-vCPU worker fight over cores. Default to
    # the container's CPU count; override per-deploy via PDF_PARALLEL_WORKERS env var.
    PDF_PARALLEL_WORKERS: int = int(os.getenv("PDF_PARALLEL_WORKERS", os.cpu_count() or 2))

    # ── Phase 3: Tiered PDF strategy thresholds ──
    # ≤ PDF_FAST_THRESHOLD_PAGES   → strategy="fast" (no OCR, ~0.5s/doc)
    # ≤ PDF_MEDIUM_THRESHOLD_PAGES → strategy="auto" (~2-5s/doc)
    # > PDF_MEDIUM_THRESHOLD_PAGES → strategy="hi_res" (full OCR, ~15-30s/doc)
    PDF_FAST_THRESHOLD_PAGES: int = 10   # Raised from 5 — more PDFs avoid model loading entirely
    PDF_MEDIUM_THRESHOLD_PAGES: int = 30
    # A5: above MEDIUM, only OCR (hi_res) when the avg extractable text/page is below
    # this — i.e. the PDF is genuinely scanned. Born-digital long PDFs use "auto".
    PDF_TEXT_LAYER_MIN_CHARS_PER_PAGE: int = 100

    # Retrieval params
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30   # Noise floor — reranker handles precision; 0.45 was too aggressive for text-embedding-3-small

    # Hybrid search: BM25 (local) + Dense (Pinecone) merged via Reciprocal Rank Fusion
    # Set to True to activate HybridRetriever in retrieval.py.
    USE_HYBRID_SEARCH: bool = False
    # How many candidates to over-fetch from Pinecone for BM25 to rank over.
    # Larger = better recall for BM25 at the cost of slightly more Pinecone latency.
    HYBRID_FETCH_K: int = 25

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY")
    # B1: local JWT verification avoids an auth.get_user() round-trip per request.
    # Modern Supabase tokens (ES256/RS256) are verified via the project's JWKS
    # endpoint using SUPABASE_URL alone — no secret needed. This is ONLY the legacy
    # HS256 shared secret (Dashboard → Settings → API → JWT Keys → Legacy JWT Secret),
    # used to verify any still-valid legacy tokens. Optional.
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "docquery")
    PINECONE_NAMESPACE: str = ""  # B5: Must be set per-user — empty default prevents silent fallback

    # UPLOAD_DIR is used as a temp directory when downloading files from Supabase
    # for processing. Files are NOT stored here permanently anymore.
    UPLOAD_DIR: str = os.path.join(_PROJECT_ROOT, "tmp_uploads")
    SUPPORTED_FILE_TYPES: tuple = (
        "pdf",
        "docx",
        "pptx",
        "txt",
        "xlsx",
    )

    # ── Reranker ──
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_INITIAL_K: int = 10   # Lowered from 15 — 10 gives good recall with less Pinecone/reranker work
    RERANK_TOP_K: int = 5        # Final docs after reranking

    # ── Multi-Query Retrieval ──
    USE_MULTI_QUERY: bool = True
    MULTI_QUERY_COUNT: int = 2   # Lowered from 3 — 2 variants + original = 3 Pinecone calls (was 4)

    # ── Input limits ──
    MAX_QUERY_LENGTH: int = 2000   # Max characters for user question

    # ── Phase 2: Web Search Fallback ──
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    USE_WEB_FALLBACK: bool = True    # Enable web search when docs return no results
    WEB_SEARCH_MAX_RESULTS: int = 3  # Max web results to merge with doc results