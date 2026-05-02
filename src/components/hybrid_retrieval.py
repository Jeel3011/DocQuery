"""
DocQuery — Hybrid Retrieval (BM25 + Dense via Reciprocal Rank Fusion)

Architecture
------------
Since the Pinecone index is serverless / dense-only (cosine metric, 1536-dim),
true sparse-dense hybrid at the index level is not available without migrating
to a pod-based index.

Instead we implement the "client-side hybrid" pattern that is both Pinecone-agnostic
and well-studied in the IR literature:

  1. Dense retrieval  → top-K candidates from Pinecone (same as before)
  2. BM25 retrieval   → top-K candidates from an in-memory BM25 index built
                        from the same candidates already fetched from Pinecone.
                        BM25 scores keyword/exact-match quality.
  3. Reciprocal Rank Fusion (RRF) → merges both ranked lists without
                        needing to normalise scores across different scales.
  4. Optional reranker → cross-encoder final pass on the RRF result.

Why RRF?
  RRF(d) = Σ  1 / (k + rank_i(d))   where k=60 (de-facto standard)
  It is robust, parameter-free, and consistently outperforms weighted
  score fusion when the score scales differ (dense cosine vs BM25 TF-IDF).

Usage
-----
Set USE_HYBRID_SEARCH=True in Config to activate.  The RetrievalManager
will transparently use HybridRetrieval instead of pure dense search.
"""

from __future__ import annotations

import math
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.logger import get_logger

logger = get_logger(__name__)

# RRF constant — 60 is the canonical value from Cormack et al. 2009
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = _RRF_K,
) -> list[Document]:
    """
    Fuse multiple ranked lists of Documents using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is already ranked best→worst.
        k:            RRF constant (default 60).

    Returns:
        Documents sorted by descending RRF score, deduplicated by content hash.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata.get("content_hash", doc.page_content[:120])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map.setdefault(key, doc)

    fused = sorted(doc_map.keys(), key=lambda k_: scores[k_], reverse=True)
    return [doc_map[k_] for k_ in fused]


class HybridRetriever:
    """
    Runs BM25 over a pre-fetched dense candidate pool then fuses the
    two ranked lists via RRF.

    This avoids re-fetching from Pinecone and keeps latency low:
    we over-fetch a large candidate pool from Pinecone once (RERANK_INITIAL_K
    or a configurable HYBRID_FETCH_K), rank it with BM25, fuse, then optionally
    rerank with the cross-encoder.
    """

    def __init__(
        self,
        top_k: int = 5,
        rrf_k: int = _RRF_K,
    ):
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_docs: list[Document],
    ) -> list[Document]:
        """
        Given a dense-ranked candidate pool, apply BM25 re-ranking and fuse.

        Args:
            query:      The user's search query.
            dense_docs: Documents already ranked by dense (cosine) similarity.

        Returns:
            Fused and truncated list of Documents (length ≤ self.top_k).
        """
        if not dense_docs:
            return []

        if len(dense_docs) == 1:
            return dense_docs

        # ── BM25 rank ──────────────────────────────────────────────────────
        corpus = [_tokenize(doc.page_content) for doc in dense_docs]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))

        # Sort by BM25 score descending → bm25_ranked is a ranked list
        bm25_indexed = sorted(
            enumerate(dense_docs),
            key=lambda x: bm25_scores[x[0]],
            reverse=True,
        )
        bm25_ranked = [doc for _, doc in bm25_indexed]

        logger.info(
            "HybridRetriever: dense=%d docs, bm25 top token='%s'",
            len(dense_docs),
            _tokenize(query)[:3],
        )

        # ── Reciprocal Rank Fusion ──────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            [dense_docs, bm25_ranked],
            k=self.rrf_k,
        )

        result = fused[: self.top_k]
        logger.info(
            "HybridRetriever: RRF fused %d → returning top %d",
            len(fused),
            len(result),
        )
        return result
