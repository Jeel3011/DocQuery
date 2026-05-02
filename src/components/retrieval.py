from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.components.config import Config
from src.logger import get_logger
import os

logger = get_logger(__name__)

# Lazy import — only needed when USE_HYBRID_SEARCH is True
_HybridRetriever = None


def _get_hybrid_retriever_class():
    global _HybridRetriever
    if _HybridRetriever is None:
        from src.components.hybrid_retrieval import HybridRetriever
        _HybridRetriever = HybridRetriever
    return _HybridRetriever


class RetrievalManager:

    def __init__(self, config: Config):
        self.config = config
        self.logger = logger

        if self.config.PINECONE_API_KEY:
            os.environ["PINECONE_API_KEY"] = self.config.PINECONE_API_KEY

        self.vectorstore = PineconeVectorStore(
            index_name=self.config.PINECONE_INDEX_NAME,
            embedding=OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL_NAME,
                openai_api_key=self.config.OPENAI_API_KEY,
            ),
            namespace=self.config.PINECONE_NAMESPACE,
        )

        # ── Optional reranker ──
        self._reranker = None
        if self.config.USE_RERANKER:
            from src.components.reranker import Reranker
            self._reranker = Reranker(self.config.RERANKER_MODEL)

        # ── Optional hybrid retriever (BM25 + Dense → RRF) ──
        self._hybrid = None
        if self.config.USE_HYBRID_SEARCH:
            hybrid_retriever_cls = _get_hybrid_retriever_class()
            self._hybrid = hybrid_retriever_cls(
                top_k=self.config.RERANK_TOP_K if self._reranker else self.config.TOP_K,
            )
            logger.info("HybridRetriever enabled (fetch_k=%d)", self.config.HYBRID_FETCH_K)

    # ── Private helper: raw vector search (shared by retrieve & retrieve_multi_query) ──

    def _raw_retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
    ) -> list[Document]:
        """Run similarity search against Pinecone and return docs above threshold.

        When USE_HYBRID_SEARCH is True the caller should pass a larger fetch_k
        so the BM25 step has enough candidates to re-rank from.
        """
        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        if self._hybrid:
            # Over-fetch a big candidate pool for BM25 to rank over
            fetch_k = self.config.HYBRID_FETCH_K
        elif self._reranker:
            fetch_k = self.config.RERANK_INITIAL_K
        else:
            fetch_k = self.config.TOP_K

        try:
            filter_dict = {}
            if filename_filter:
                filter_dict["filename"] = filename_filter
            if page_filter:
                filter_dict["page_number"] = page_filter

            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter_dict if filter_dict else None,
            )

            docs = [
                doc for doc, score in docs_and_scores if score >= similarity_threshold
            ]
            self.logger.info(
                "Retrieved %d/%d docs above threshold", len(docs), len(docs_and_scores)
            )
            return docs

        except Exception as e:
            self.logger.error("retrieval failed: %s", e)
            return []

    # ── Public: single-query retrieve ──

    def retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
    ) -> list[Document]:
        """Retrieve relevant docs with optional hybrid BM25+RRF fusion and/or reranking."""
        docs = self._raw_retrieve(query, filename_filter, page_filter)

        # Step 1: Hybrid BM25 + RRF fusion
        if self._hybrid and docs:
            docs = self._hybrid.retrieve(query, docs)

        # Step 2: Cross-encoder reranker
        if self._reranker and docs:
            top_k = self.config.RERANK_TOP_K
            docs = self._reranker.rerank(query, docs, top_k=top_k)

        return docs

    # ── Public: multi-query retrieve (Feature 3) ──

    def retrieve_multi_query(
        self,
        queries: list[str],
        filename_filter: str = None,
        page_filter: str = None,
    ) -> list[Document]:
        """Retrieve docs for multiple query variants, deduplicate, optional rerank.

        Uses a thread pool to run Pinecone calls in parallel (fixes L3).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_docs: dict[str, Document] = {}

        def _fetch(q: str) -> list[Document]:
            return self._raw_retrieve(q, filename_filter, page_filter)

        # Run all Pinecone queries in parallel (I/O-bound)
        max_workers = min(len(queries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, q): q for q in queries}
            for future in as_completed(futures):
                for doc in future.result():
                    key = doc.metadata.get("content_hash", doc.page_content[:100])
                    if key not in all_docs:
                        all_docs[key] = doc

        merged = list(all_docs.values())
        self.logger.info(
            "Multi-query: %d variants -> %d unique docs (parallel)", len(queries), len(merged)
        )

        # Hybrid BM25 + RRF fusion (same as single-query path)
        if self._hybrid and merged:
            merged = self._hybrid.retrieve(queries[0], merged)

        if self._reranker and merged:
            merged = self._reranker.rerank(
                queries[0], merged, top_k=self.config.RERANK_TOP_K
            )
        elif len(merged) > self.config.TOP_K:
            merged = merged[: self.config.TOP_K]  # simple truncation fallback

        return merged

    # ── Delete helper ──

    def delete_document_by_filename(self, filename: str):
        try:
            self.vectorstore.delete(filter={"filename": filename})
            self.logger.info("Deleted documents with filename %s", filename)
        except Exception as e:
            self.logger.error(
                "Failed to delete documents with filename %s: %s", filename, e
            )
