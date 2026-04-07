"""
DocQuery — Cross-Encoder Reranker

Reranks initial vector-search candidates using a cross-encoder model
for higher precision.  The cross-encoder scores each (query, document)
pair and returns the top-k by relevance.
"""

import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.logger import get_logger
from src.components.metrics import rerank_latency

logger = get_logger(__name__)


class Reranker:
    """Thin wrapper around a SentenceTransformers CrossEncoder."""

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded: %s", model_name)

    def rerank(
        self, query: str, docs: list[Document], top_k: int = 5
    ) -> list[Document]:
        """Score every (query, doc) pair and return the top-k by relevance."""
        if not docs:
            return []

        start = time.perf_counter()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        elapsed = time.perf_counter() - start
        rerank_latency.observe(elapsed)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, _score in ranked[:top_k]]

        logger.info("Reranked %d→%d docs in %.3fs", len(docs), len(result), elapsed)
        return result
