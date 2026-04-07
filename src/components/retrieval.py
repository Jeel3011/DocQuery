from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.components.config import Config
from src.logger import get_logger
import os

logger = get_logger(__name__)


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

        # ── Optional reranker (Feature 2) ──
        self._reranker = None
        if self.config.USE_RERANKER:
            from src.components.reranker import Reranker

            self._reranker = Reranker(self.config.RERANKER_MODEL)

    # ── Private helper: raw vector search (shared by retrieve & retrieve_multi_query) ──

    def _raw_retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
    ) -> list[Document]:
        """Run similarity search against Pinecone and return docs above threshold."""
        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        fetch_k = (
            self.config.RERANK_INITIAL_K if self._reranker else self.config.TOP_K
        )

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
        """Retrieve relevant docs, optionally reranked."""
        docs = self._raw_retrieve(query, filename_filter, page_filter)

        if self._reranker and docs:
            docs = self._reranker.rerank(
                query, docs, top_k=self.config.RERANK_TOP_K
            )

        return docs

    # ── Public: multi-query retrieve (Feature 3) ──

    def retrieve_multi_query(
        self,
        queries: list[str],
        filename_filter: str = None,
        page_filter: str = None,
    ) -> list[Document]:
        """Retrieve docs for multiple query variants, deduplicate, optional rerank."""
        all_docs: dict[str, Document] = {}

        for q in queries:
            docs = self._raw_retrieve(q, filename_filter, page_filter)
            for doc in docs:
                # Dedupe by content hash (already in metadata from embedding time)
                key = doc.metadata.get("content_hash", doc.page_content[:100])
                if key not in all_docs:
                    all_docs[key] = doc

        merged = list(all_docs.values())
        self.logger.info(
            "Multi-query: %d variants → %d unique docs", len(queries), len(merged)
        )

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
