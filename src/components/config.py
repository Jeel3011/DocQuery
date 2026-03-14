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

    # Retrieval params
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30

    # Hybrid search (wired up later)
    USE_HYBRID_SEARCH: bool = False

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY")

    # Local paths (ChromaDB stays local for now — Pinecone migration later)
    VECTOR_DB_PATH: str = os.path.join(_PROJECT_ROOT, "vector_db")

    # UPLOAD_DIR is used as a temp directory when downloading files from Supabase
    # for processing. Files are NOT stored here permanently anymore.
    UPLOAD_DIR: str = os.path.join(_PROJECT_ROOT, "tmp_uploads")

    COLLECTION_NAME: str = "docquery_v1"
    SUPPORTED_FILE_TYPES: tuple = (
        "pdf",
        "docx",
        "pptx",
        "txt",
        "xlsx",
    )