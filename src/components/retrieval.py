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

    # ── Phase 4.3: semantic table-chunk retrieval (generic table selection) ──
    def retrieve_table_chunks(
        self,
        query: str,
        *,
        collection_id: str = None,
        filename_filters: list[str] = None,
        doc_id: str = None,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
        k: int = 8,
    ) -> list:
        """Semantically retrieve TABLE chunks for a numeric/tabular query.

        Filters Pinecone to ``chunk_type="table"`` and ranks by the embedded table
        CAPTION (set at ingest), so "AWS operating margin" matches the segment
        statement, not a same-keyword mix table — generically, by MEANING, for any
        document the Brain encounters. Returns the table chunks (metadata carries
        ``table_json`` for the Analyst). Never raises; returns [] on failure so the
        Analyst path degrades to prose-only synthesis.

        Scope priority (G3): ``doc_id``/``doc_ids`` (the stable, ingest-stamped axis —
        vault isolation as a data property) win over the legacy filename fallback. The
        per-USER Pinecone namespace already isolates users; this filter isolates VAULTS.
        ``metadata_filter`` (G3 Step B) is a conjunctive narrowing merged on top — it
        NEVER replaces the scope (a bug there would be a cross-vault leak).
        """
        f: dict = {"chunk_type": "table"}
        if doc_id:
            f["doc_id"] = doc_id
        elif doc_ids:
            f["doc_id"] = {"$in": doc_ids}
        elif collection_id:
            f["collection_id"] = collection_id
        elif filename_filters:
            f["filename"] = {"$in": filename_filters}
        if metadata_filter:
            # Conjunctive narrowing on top of vault scope; never replaces scope keys.
            for mk, mv in metadata_filter.items():
                if mk not in ("doc_id", "collection_id", "filename", "chunk_type"):
                    f[mk] = mv
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=f,
            )
            docs = [doc for doc, _score in docs_and_scores]
            self.logger.info("Retrieved %d table chunks for numeric query", len(docs))
            return docs
        except Exception as e:
            self.logger.warning("table-chunk retrieval failed: %s", e)
            return []

    # ── Private helper: raw vector search (shared by retrieve & retrieve_multi_query) ──

    # Scope keys we never let a metadata_filter overwrite (it NARROWS, never REPLACES,
    # the vault scope — overwriting one would be a cross-vault leak; G3 §5 risk #4).
    _SCOPE_KEYS = frozenset({"doc_id", "collection_id", "filename"})

    @classmethod
    def _build_filter(
        cls,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
        collection_id: str = None,
        doc_id: str = None,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
    ) -> dict | None:
        """Build a Pinecone metadata filter dict from the supplied scope parameters.

        Scope priority (most selective wins — G3 makes doc_id the stable axis):
          doc_id        → single-doc scalar filter (most efficient)
          doc_ids       → $in over the vault's docs (vault isolation as a DATA property)
          collection_id → scalar filter on the stamped collection_id field (Phase 1+)
          filename_filters → $in list (legacy fallback for un-stamped vectors)
          filename_filter  → single filename scalar

        ``metadata_filter`` (G3 Step B: doc_type / fiscal_year / …) is merged CONJUNCTIVELY
        on top of the scope — it can only NARROW, never replace a scope key (that would be
        a cross-vault leak). Scope keys present in metadata_filter are dropped.
        """
        f: dict = {}
        if doc_id:
            f["doc_id"] = doc_id
        elif doc_ids:
            f["doc_id"] = {"$in": doc_ids}
        elif collection_id:
            f["collection_id"] = collection_id
        elif filename_filters:
            f["filename"] = {"$in": filename_filters}
        elif filename_filter:
            f["filename"] = filename_filter
        if page_filter:
            f["page_number"] = page_filter
        if metadata_filter:
            for mk, mv in metadata_filter.items():
                if mk not in cls._SCOPE_KEYS:
                    f[mk] = mv
        return f if f else None

    def _raw_retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
        apply_threshold: bool = True,
        collection_id: str = None,
        doc_id: str = None,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
        fetch_k_override: int = None,
    ) -> list[Document]:
        """Run similarity search against Pinecone and return docs above threshold.

        When USE_HYBRID_SEARCH is True the caller should pass a larger fetch_k
        so the BM25 step has enough candidates to re-rank from.

        fetch_k_override raises the candidate pool (used by the Brain, which reads
        many chunks per document so it can find needles in large filings).

        apply_threshold=False keeps the top-k regardless of absolute score — used by
        per-file collection retrieval, where we want each file's best chunks even if
        modestly scored (the reranker provides precision).
        """
        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        if fetch_k_override:
            fetch_k = fetch_k_override
        elif self._hybrid:
            fetch_k = self.config.HYBRID_FETCH_K
        elif self._reranker:
            fetch_k = self.config.RERANK_INITIAL_K
        else:
            fetch_k = self.config.TOP_K

        try:
            filter_dict = self._build_filter(
                filename_filter=filename_filter,
                page_filter=page_filter,
                filename_filters=filename_filters,
                collection_id=collection_id,
                doc_id=doc_id,
                doc_ids=doc_ids,
                metadata_filter=metadata_filter,
            )

            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter_dict if filter_dict else None,
            )

            docs = [
                doc for doc, score in docs_and_scores
                if not apply_threshold or score >= similarity_threshold
            ]
            self.logger.info(
                "Retrieved %d/%d docs above threshold", len(docs), len(docs_and_scores)
            )
            return docs

        except Exception as e:
            self.logger.error("retrieval failed: %s", e)
            return []

    def _raw_retrieve_by_vector(
        self,
        query_embedding: list,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
        apply_threshold: bool = True,
        collection_id: str = None,
        doc_id: str = None,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
    ) -> list[Document]:
        """Run similarity search using a pre-computed embedding vector.

        Avoids a redundant OpenAI embedding API call when the caller already has
        the embedding (e.g., computed for the semantic cache lookup).
        Saves ~150ms per cache-miss query.

        apply_threshold=False keeps the top-k regardless of absolute score — used by
        per-file collection retrieval (see retrieve_across_files).
        """
        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        if self._hybrid:
            fetch_k = self.config.HYBRID_FETCH_K
        elif self._reranker:
            fetch_k = self.config.RERANK_INITIAL_K
        else:
            fetch_k = self.config.TOP_K

        try:
            filter_dict = self._build_filter(
                filename_filter=filename_filter,
                page_filter=page_filter,
                filename_filters=filename_filters,
                collection_id=collection_id,
                doc_id=doc_id,
                doc_ids=doc_ids,
                metadata_filter=metadata_filter,
            )

            docs_and_scores = self.vectorstore.similarity_search_by_vector_with_score(
                query_embedding,
                k=fetch_k,
                filter=filter_dict if filter_dict else None,
            )

            docs = [
                doc for doc, score in docs_and_scores
                if not apply_threshold or score >= similarity_threshold
            ]
            self.logger.info(
                "Retrieved %d/%d docs above threshold (by vector)", len(docs), len(docs_and_scores)
            )
            return docs

        except Exception as e:
            self.logger.error("retrieval by vector failed: %s", e)
            return []

    # ── Public: single-query retrieve ──


    def retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
        top_k: int = None,
        apply_threshold: bool = True,
        use_reranker: bool = True,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
    ) -> list[Document]:
        """Retrieve relevant docs with optional hybrid BM25+RRF fusion and/or reranking.

        top_k overrides the final number of chunks returned (and widens the candidate
        pool accordingly). Used by the Brain to read many chunks per document.

        apply_threshold: when False, keeps top-k chunks regardless of absolute score.
        The Brain endpoint passes False because its MAP+VERIFY pipeline handles
        precision — the similarity threshold is unnecessary and actively harmful for
        per-doc retrieval where the query is cross-document in nature.

        use_reranker: when False, skip the local CrossEncoder rerank and return the
        top_k by vector similarity directly. The Brain passes False: its MAP+VERIFY
        already does the precision work, and the local CPU CrossEncoder (~18s/batch on
        a laptop under concurrent load) was causing per-doc retrieval TIMEOUTS that
        silently dropped whole documents from the answer (e.g. AWS net-sales never
        reaching MAP). Skipping it removes the timeout source; on the cloud (GPU/hosted
        rerank) it can be re-enabled via env.
        """
        # Vault scope spanning multiple docs → guarantee each doc is represented.
        # G3: prefer the stable doc_id axis; fall back to the legacy filename balance.
        if doc_ids and len(doc_ids) > 1:
            return self.retrieve_across_files(
                query, page_filter=page_filter, doc_ids=doc_ids,
                metadata_filter=metadata_filter,
            )
        if filename_filters and len(filename_filters) > 1:
            return self.retrieve_across_files(
                query, filename_filters, page_filter=page_filter,
                metadata_filter=metadata_filter,
            )

        rerank_on = use_reranker and self._reranker is not None
        # Widen the candidate pool when a large top_k is requested so rerank has enough
        # to choose from (≈4× the final count). When the reranker is skipped we don't
        # need the extra pool — fetch exactly top_k so we don't over-pull from Pinecone.
        if top_k:
            fetch_override = max(top_k * 4, self.config.RERANK_INITIAL_K) if rerank_on else top_k
        else:
            fetch_override = None
        docs = self._raw_retrieve(
            query, filename_filter, page_filter,
            filename_filters=filename_filters, fetch_k_override=fetch_override,
            apply_threshold=apply_threshold,
            doc_ids=doc_ids, metadata_filter=metadata_filter,
        )

        # Step 1: Hybrid BM25 + RRF fusion
        if self._hybrid and docs:
            docs = self._hybrid.retrieve(query, docs)

        # Step 2: Cross-encoder reranker (skippable — see use_reranker above)
        if rerank_on and docs:
            docs = self._reranker.rerank(query, docs, top_k=top_k or self.config.RERANK_TOP_K)
        elif top_k:
            docs = docs[:top_k]

        return docs

    def retrieve_by_vector(
        self,
        query_embedding: list,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
    ) -> list[Document]:
        """Retrieve using a pre-computed embedding — skips the OpenAI embed API call.

        Use this when the caller already has the query embedding (e.g., computed for
        the semantic cache lookup). Saves ~150ms per cache-miss query.

        Args:
            query_embedding: Pre-computed embedding vector from OpenAI.
            query: Original query string (used for reranking, hybrid search).
            filename_filter: Optional Pinecone metadata filter.
            page_filter: Optional page number filter.
            filename_filters: Optional list of filenames for collection-scoped search.
        """
        # Collection scope spanning multiple files → guarantee each file is represented.
        if filename_filters and len(filename_filters) > 1:
            return self.retrieve_across_files(
                query, filename_filters, page_filter=page_filter, query_embedding=query_embedding
            )

        docs = self._raw_retrieve_by_vector(query_embedding, filename_filter, page_filter, filename_filters=filename_filters)

        # Step 1: Hybrid BM25 + RRF fusion (needs string query for BM25)
        if self._hybrid and docs:
            docs = self._hybrid.retrieve(query, docs)

        # Step 2: Cross-encoder reranker (needs string query for pair scoring)
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
        filename_filters: list[str] = None,
        primary_embedding: list = None,
    ) -> list[Document]:
        """Retrieve docs for multiple query variants, deduplicate, optional rerank.

        Uses a thread pool to run Pinecone calls in parallel (fixes L3).

        B5: if primary_embedding is supplied it is reused for queries[0] (skipping
        a redundant OpenAI embed of the primary query); callers must only pass it
        when it actually corresponds to queries[0].
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Collection scope spanning multiple files → guarantee each file is represented
        # (the primary query carries the intent; per-file balancing matters more here
        # than variant recall within a single global pool).
        if filename_filters and len(filename_filters) > 1:
            return self.retrieve_across_files(
                queries[0], filename_filters, page_filter=page_filter, query_embedding=primary_embedding
            )

        all_docs: dict[str, Document] = {}

        def _fetch(q: str) -> list[Document]:
            return self._raw_retrieve(q, filename_filter, page_filter, filename_filters=filename_filters)

        def _fetch_primary() -> list[Document]:
            return self._raw_retrieve_by_vector(primary_embedding, filename_filter, page_filter, filename_filters=filename_filters)

        # Run all Pinecone queries in parallel (I/O-bound)
        max_workers = min(len(queries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i, q in enumerate(queries):
                if i == 0 and primary_embedding:
                    futures[pool.submit(_fetch_primary)] = q
                else:
                    futures[pool.submit(_fetch, q)] = q
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

    # ── Collection-scoped balanced retrieval ──

    def retrieve_across_files(
        self,
        query: str,
        filenames: list[str] = None,
        page_filter: str = None,
        per_file_k: int = None,
        query_embedding: list = None,
        doc_ids: list[str] = None,
        metadata_filter: dict = None,
    ) -> list[Document]:
        """Collection retrieval that GUARANTEES every file is represented.

        G3: when ``doc_ids`` is supplied it is the balance axis (the stable, ingest-
        stamped vault scope); otherwise we balance over ``filenames`` (legacy). Either
        way each in-scope doc keeps its best chunks so no document silently vanishes.

        The normal path reranks one merged pool and keeps the global top-k, which for
        a cross-file question ("compare A and B", "difference between both") can
        collapse entirely onto one file. Here we fetch each file's best chunks
        INDEPENDENTLY (parallel I/O), rerank within each file, then merge — so the LLM
        always sees content from every document in the collection. The global
        similarity threshold is skipped on purpose: a file's best chunks are kept even
        if their absolute score is modest (the cross-encoder reranker gives precision).

        Invariant R1: fan-out is capped at ROUTING_MAX_FANOUT.  Callers that pass a
        larger list must run Stage-1 routing first to reduce the corpus; if they don't,
        we warn and truncate to the cap so the request never melts down.  Once the
        Stage-1 router exists (Phase 3) it will pre-filter before calling this method.
        """
        from concurrent.futures import ThreadPoolExecutor

        # G3: balance over the stable doc_id axis when given, else legacy filenames.
        # `by_doc_id` decides which scalar scope key each per-doc fetch uses.
        by_doc_id = bool(doc_ids)
        scope_items = list(doc_ids) if by_doc_id else list(filenames or [])

        max_fanout = self.config.ROUTING_MAX_FANOUT
        if len(scope_items) > max_fanout:
            self.logger.warning(
                "retrieve_across_files: %d docs exceeds ROUTING_MAX_FANOUT=%d — "
                "truncating to first %d. Deploy Stage-1 router (Phase 3) to fix this properly.",
                len(scope_items), max_fanout, max_fanout,
            )
            scope_items = scope_items[:max_fanout]

        n = len(scope_items)
        if per_file_k is None:
            # Keep total context bounded (~8-12 chunks) regardless of collection size.
            per_file_k = 4 if n <= 2 else (3 if n <= 4 else 2)

        def _fetch_raw(item: str) -> list[Document]:
            # Scope this single fetch to ONE doc — by doc_id (G3) or filename (legacy).
            scalar = {"doc_id": item} if by_doc_id else {"filename_filter": item}
            if query_embedding is not None:
                return self._raw_retrieve_by_vector(
                    query_embedding, **scalar,
                    page_filter=page_filter, apply_threshold=False,
                    metadata_filter=metadata_filter,
                )
            return self._raw_retrieve(
                query, **scalar,
                page_filter=page_filter, apply_threshold=False,
                metadata_filter=metadata_filter,
            )

        # Parallel Pinecone fetch (I/O-bound). Rerank sequentially afterwards — the
        # cross-encoder model isn't safe to call from multiple threads at once.
        with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            raw_per_file = list(pool.map(_fetch_raw, scope_items))

        merged: list[Document] = []
        for docs in raw_per_file:
            if self._reranker and docs:
                docs = self._reranker.rerank(query, docs, top_k=per_file_k)
            else:
                docs = docs[:per_file_k]
            merged.extend(docs)

        self.logger.info(
            "Cross-file retrieve: %d %s x ~%d/doc -> %d chunks",
            n, "doc_ids" if by_doc_id else "files", per_file_k, len(merged)
        )
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
