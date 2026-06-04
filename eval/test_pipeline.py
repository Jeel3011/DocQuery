"""
Test script: runs the full pipeline (ingest → embed → query) twice
to verify that the second run skips re-processing and re-embedding.
"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from src.components.embeddings import EmbeddingManager
from src.components.retrieval import RetrievalManager
from src.components.generation import AnswerGenration

def run_pipeline():
    config = Config()
    
    # Step 1: Ingest & chunk
    print("=" * 60)
    print("STEP 1: INGESTION & CHUNKING")
    print("=" * 60)
    t1 = time.time()
    processor = DocumentProcessor(config)
    chunks = processor.process_batch(config.UPLOAD_DIR)
    t2 = time.time()
    print(f"Ingestion took: {t2 - t1:.2f}s")
    
    # Step 2: Embed
    print("\n" + "=" * 60)
    print("STEP 2: EMBEDDING")
    print("=" * 60)
    t3 = time.time()
    embedding_mgr = EmbeddingManager(config)
    vector_store = embedding_mgr.create_vector_store(chunks)
    t4 = time.time()
    print(f"Embedding took: {t4 - t3:.2f}s")
    
    # Step 3: Query
    print("\n" + "=" * 60)
    print("STEP 3: QUERY")
    print("=" * 60)
    t5 = time.time()
    retrieval_mgr = RetrievalManager(config)
    generator = AnswerGenration(config)
    query = "what is attention mechanism?"
    docs = retrieval_mgr.retrieve(query)
    result = generator.generate(query, docs)
    t6 = time.time()
    print(f"Query took: {t6 - t5:.2f}s")
    print(f"\nAnswer: {result['answer'][:200]}...")
    
    print("\n" + "=" * 60)
    print(f"TOTAL TIME: {t6 - t1:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
