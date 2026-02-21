from src.components.config import Config
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.components.data_ingestion import DocumentProcessor
from src.components.embeddings import EmbeddingManager
from src.components.retrieval import RetrievalManager
from src.components.genration import AnswerGenration

def run_pipeline(query: str):
    config = Config()

    # ingestion adn chunking

    processor = DocumentProcessor(config=config)
    chunks = processor.process_batch(config.UPLOAD_DIR)

    # embedding

    embed_mng = EmbeddingManager(config=config)
    vector_store = embed_mng.create_vector_store(chunks,config.VECTOR_DB_PATH)

    # retrieval 

    retrieval_mng = RetrievalManager(config=config)

    genrator = AnswerGenration(config=config)

    docs = retrieval_mng.retrieve(query)

    result = genrator.generate(query=query,retrieved_docs=docs)

    return {
        "answer": result,
        "num_sources_used": len(docs),
        "sources": docs
    }

if __name__ == "__main__":
    query = "what is attention mechanism?"
    result = run_pipeline(query)
    print(f"Answer: {result['answer']['answer']}")

    print(f"Sources used: {result['num_sources_used']}")

    