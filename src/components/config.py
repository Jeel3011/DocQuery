from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

@dataclass
class Config:
    """Centralized configuration"""

    
    # for embedings and llm
    EMBEDDING_MODEL_NAME: str = "openai-embedding-3-small"
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"
    
    # chunking para 
    CHUNK_SIZE: int = 3000
    NEW_AFTER_N_CHARS: int = 2400
    COMBINE_TEXT_UNDER_N_CHARS: int = 500
    CHUNK_OVERLAP: int = 500
    
    #retrieval para
    TOP_K: int = 5
    SIMILARITY_THRESHOLD:float = 0.70

    #API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    #vector db and its paths
    VECTOR_DB_PATH:str="./vector_db"
    UPLOAD_DIR:str = "./uploaded_docs"
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

USE_HYBRID_SEARCH:bool = True


