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
    PDF_PARALLEL_WORKERS: int = 4        # Max parallel workers for page processing

    # ── Phase 3: Tiered PDF strategy thresholds ──
    # ≤ PDF_FAST_THRESHOLD_PAGES   → strategy="fast" (no OCR, ~0.5s/doc)
    # ≤ PDF_MEDIUM_THRESHOLD_PAGES → strategy="auto" (~2-5s/doc)
    # > PDF_MEDIUM_THRESHOLD_PAGES → strategy="hi_res" (full OCR, ~15-30s/doc)
    PDF_FAST_THRESHOLD_PAGES: int = 5
    PDF_MEDIUM_THRESHOLD_PAGES: int = 30

    # Retrieval params
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30

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
    RERANK_INITIAL_K: int = 15   # Over-fetch from Pinecone
    RERANK_TOP_K: int = 5        # Final docs after reranking

    # ── Multi-Query Retrieval ──
    USE_MULTI_QUERY: bool = True
    MULTI_QUERY_COUNT: int = 3   # Number of query variants to generate

    # ── Input limits ──
    MAX_QUERY_LENGTH: int = 2000   # Max characters for user question