from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # Load .env file

# Project root = two levels up from this file (src/components/config.py -> project root)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

@dataclass
class Config:
    """Centralized configuration"""

    
    # for embedings and llm
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    
    # chunking para 
    CHUNK_SIZE: int = 3000
    NEW_AFTER_N_CHARS: int = 2400
    COMBINE_TEXT_UNDER_N_CHARS: int = 500
    CHUNK_OVERLAP: int = 500
    
    #retrieval para
    TOP_K: int = 5
    SIMILARITY_THRESHOLD:float = 0.30

    # hybrid search
    USE_HYBRID_SEARCH: bool = True

    #API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    #vector db and its paths (relative to project root, not CWD)
    VECTOR_DB_PATH: str = os.path.join(_PROJECT_ROOT, "vector_db")
    UPLOAD_DIR: str = os.path.join(_PROJECT_ROOT, "docs")
    COLLECTION_NAME : str = "docquery_v1"
    SUPPORTED_FILE_TYPES: Literal[
        "pdf",
        "docx",
        "pptx",
        "txt",
        "xlsx"
    ] = (
        "pdf",
        "docx",
        "pptx",
        "txt",
        "xlsx"
    )



