from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

@dataclass
class config:
    """Centralized configuration"""

    # for embedings and llm
    EMBEDDING_MODEL_NAME: str = "openai-embedding-3-small"
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"
    
    # chunking para 
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    #retrieval para
    TOP_K: int = 5
    SIMILARITY_THRESHOLD:float = 0.70

    #API keys
    OPENAI_API_KEY: str = load_dotenv("OPENAI_API_KEY")

    #vector db and its paths
    VECTOR_DB_PATH:str="./vector_db"
    OPLOAD_DIR:str = "./oploaded_docs"
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


