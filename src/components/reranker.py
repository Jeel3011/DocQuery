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

# ── Singleton model cache ──
# CrossEncoder takes 1-3s to load from disk. Cache it so the cost is paid
# once at first use, not on every request.
_model_cache: dict[str, CrossEncoder] = {}


def _best_device() -> str:
    """Auto-detect the best available device for the cross-encoder.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU.
    Falls back to CPU if torch is not available or device check fails.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _model_cache:
        device = _best_device()
        # Try loading from local HuggingFace cache first (skips ~15s of Hub checks).
        # Falls back to online download if the model hasn't been cached yet.
        import os
        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _model_cache[model_name] = CrossEncoder(model_name, device=device)
            logger.info(
                "Reranker model loaded (local cache): %s on device=%s", model_name, device
            )
        except Exception:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _model_cache[model_name] = CrossEncoder(model_name, device=device)
            logger.info(
                "Reranker model downloaded: %s on device=%s", model_name, device
            )
        finally:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return _model_cache[model_name]


class Reranker:
    """Thin wrapper around a SentenceTransformers CrossEncoder."""

    def __init__(self, model_name: str):
        self.model = _get_model(model_name)

    def rerank(
        self, query: str, docs: list[Document], top_k: int = 5
    ) -> list[Document]:
        """Score every (query, doc) pair and return the top-k by relevance."""
        if not docs:
            return []

        # Short-circuit: if we already have <= top_k docs, no need to rerank —
        # they'd all be returned anyway. Skip the model inference entirely.
        if len(docs) <= top_k:
            return docs

        start = time.perf_counter()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        elapsed = time.perf_counter() - start
        rerank_latency.observe(elapsed)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, _score in ranked[:top_k]]

        logger.info("Reranked %d→%d docs in %.3fs", len(docs), len(result), elapsed)
        return result
