from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.components.config import Config
from src.logger import get_logger
from typing import Optional
import os

logger = get_logger(__name__)

# Lazy import — only needed when USE_HYBRID_SEARCH is True
_HybridRetriever = None


def _doc_key(doc: Document) -> str:
    """Stable dedup key for a retrieved chunk. Prefer chunk_id, then content_hash, then content prefix."""
    md = doc.metadata or {}
    return md.get("chunk_id") or md.get("content_hash") or (doc.page_content or "")[:120]


def _doc_file(doc: Document) -> str:
    """Filename a chunk belongs to (the field we group by for coverage)."""
    return (doc.metadata or {}).get("filename") or "unknown"


def _jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


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
            # BUG-2 fix: when a reranker follows, hybrid must keep ENOUGH candidates for it to
            # rerank (RERANK_INITIAL_K), not truncate to RERANK_TOP_K (which made rerank a no-op
            # via its len<=top_k short-circuit). The reranker does the final cut.
            self._hybrid = hybrid_retriever_cls(
                top_k=self.config.RERANK_INITIAL_K if self._reranker else self.config.TOP_K,
            )
            logger.info("HybridRetriever enabled (fetch_k=%d)", self.config.HYBRID_FETCH_K)

    # ── Private helper: raw vector search (shared by retrieve & retrieve_multi_query) ──

    def _raw_retrieve(
        self,
        query: str,
        filename_filter: str = None,
        page_filter: str = None,
        filename_filters: list[str] = None,
        apply_threshold: bool = True,
        fetch_k: int = None,
    ) -> list[Document]:
        """Run similarity search against Pinecone and return docs above threshold.

        When USE_HYBRID_SEARCH is True the caller should pass a larger fetch_k
        so the BM25 step has enough candidates to re-rank from.

        apply_threshold=False keeps the top-k regardless of absolute score — used by
        per-file collection retrieval, where we want each file's best chunks even if
        modestly scored (the reranker provides precision).
        """
        # BUG-1 fix: distinguish "no scope" (None -> search all) from "empty collection"
        # ([] -> zero results). An empty $in must NOT fall through to an all-docs search.
        if filename_filters is not None and len(filename_filters) == 0:
            return []

        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        if fetch_k is None:
            if self._hybrid:
                fetch_k = self.config.HYBRID_FETCH_K
            elif self._reranker:
                fetch_k = self.config.RERANK_INITIAL_K
            else:
                fetch_k = self.config.TOP_K

        try:
            filter_dict = {}
            if filename_filters:
                filter_dict["filename"] = {"$in": filename_filters}
            elif filename_filter:
                filter_dict["filename"] = filename_filter
            if page_filter:
                filter_dict["page_number"] = page_filter

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
        fetch_k: int = None,
    ) -> list[Document]:
        """Run similarity search using a pre-computed embedding vector.

        Avoids a redundant OpenAI embedding API call when the caller already has
        the embedding (e.g., computed for the semantic cache lookup).
        Saves ~150ms per cache-miss query.

        apply_threshold=False keeps the top-k regardless of absolute score — used by
        per-file collection retrieval (see retrieve_across_files).
        """
        # BUG-1 fix: distinguish "no scope" (None -> search all) from "empty collection"
        # ([] -> zero results). An empty $in must NOT fall through to an all-docs search.
        if filename_filters is not None and len(filename_filters) == 0:
            return []

        similarity_threshold = self.config.SIMILARITY_THRESHOLD
        if fetch_k is None:
            if self._hybrid:
                fetch_k = self.config.HYBRID_FETCH_K
            elif self._reranker:
                fetch_k = self.config.RERANK_INITIAL_K
            else:
                fetch_k = self.config.TOP_K

        try:
            filter_dict = {}
            if filename_filters:
                filter_dict["filename"] = {"$in": filename_filters}
            elif filename_filter:
                filter_dict["filename"] = filename_filter
            if page_filter:
                filter_dict["page_number"] = page_filter

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
    ) -> list[Document]:
        """Retrieve relevant docs with optional hybrid BM25+RRF fusion and/or reranking."""
        # Collection scope spanning multiple files → guarantee each file is represented.
        if filename_filters and len(filename_filters) > 1:
            return self.retrieve_across_files(query, filename_filters, page_filter=page_filter)

        docs = self._raw_retrieve(query, filename_filter, page_filter, filename_filters=filename_filters)

        # Step 1: Hybrid BM25 + RRF fusion
        if self._hybrid and docs:
            docs = self._hybrid.retrieve(query, docs)

        # Step 2: Cross-encoder reranker
        if self._reranker and docs:
            top_k = self.config.RERANK_TOP_K
            docs = self._reranker.rerank(query, docs, top_k=top_k)

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
                    key = _doc_key(doc)
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
        filenames: list[str],
        page_filter: str = None,
        per_file_k: int = None,
        query_embedding: list = None,
    ) -> list[Document]:
        """Collection retrieval that GUARANTEES every file is represented.

        The normal path reranks one merged pool and keeps the global top-k, which for
        a cross-file question ("compare A and B", "difference between both") can
        collapse entirely onto one file. Here we fetch each file's best chunks
        INDEPENDENTLY (parallel I/O), rerank within each file, then merge — so the LLM
        always sees content from every document in the collection. The global
        similarity threshold is skipped on purpose: a file's best chunks are kept even
        if their absolute score is modest (the cross-encoder reranker gives precision).
        """
        from concurrent.futures import ThreadPoolExecutor

        n = len(filenames)
        if per_file_k is None:
            # Keep total context bounded (~8-12 chunks) regardless of collection size.
            per_file_k = 4 if n <= 2 else (3 if n <= 4 else 2)

        def _fetch_raw(fname: str) -> list[Document]:
            if query_embedding is not None:
                return self._raw_retrieve_by_vector(
                    query_embedding, filename_filter=fname,
                    page_filter=page_filter, apply_threshold=False,
                )
            return self._raw_retrieve(
                query, filename_filter=fname,
                page_filter=page_filter, apply_threshold=False,
            )

        # Parallel Pinecone fetch (I/O-bound). Rerank sequentially afterwards — the
        # cross-encoder model isn't safe to call from multiple threads at once.
        with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            raw_per_file = list(pool.map(_fetch_raw, filenames))

        merged: list[Document] = []
        for docs in raw_per_file:
            if self._reranker and docs:
                docs = self._reranker.rerank(query, docs, top_k=per_file_k)
            else:
                docs = docs[:per_file_k]
            merged.extend(docs)

        self.logger.info(
            "Cross-file retrieve: %d files x ~%d/file -> %d chunks", n, per_file_k, len(merged)
        )
        return merged

    # ── Coverage-aware collection retrieval ──

    def _select_with_coverage(
        self,
        ranked_docs: list[Document],
        k_final: int,
        min_per_file: int,
        jaccard_thresh: float,
    ) -> list[Document]:
        """Pick up to k_final docs guaranteeing per-file coverage, then fill by relevance,
        dropping near-duplicate chunks (lexical MMR). Deterministic, no embeddings/network."""
        from src.components.hybrid_retrieval import _tokenize

        if not ranked_docs:
            return []

        by_file: dict[str, list[Document]] = {}
        for d in ranked_docs:
            by_file.setdefault(_doc_file(d), []).append(d)

        selected: list[Document] = []
        selected_tok: list[set] = []
        used_keys: set[str] = set()

        def _try_add(doc: Document) -> bool:
            key = _doc_key(doc)
            if key in used_keys:
                return False
            toks = set(_tokenize(doc.page_content or ""))
            for st in selected_tok:
                if _jaccard(toks, st) > jaccard_thresh:
                    return False
            selected.append(doc)
            selected_tok.append(toks)
            used_keys.add(key)
            return True

        # Pass 1 — coverage: round-robin each file's best unused chunk.
        for _round in range(max(1, min_per_file)):
            if len(selected) >= k_final:
                break
            for fname, docs in by_file.items():
                if len(selected) >= k_final:
                    break
                for cand in docs:
                    if _doc_key(cand) in used_keys:
                        continue
                    _try_add(cand)
                    break

        # Pass 2 — fill remaining budget by global relevance order.
        if len(selected) < k_final:
            for cand in ranked_docs:
                if len(selected) >= k_final:
                    break
                _try_add(cand)

        return selected[:k_final]

    def _fanout_per_file(
        self,
        query: str,
        files: list[str],
        per_file_k: int,
        page_filter: str = None,
    ) -> list[Document]:
        """Retrieve top per_file_k chunks from EACH file in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out: dict[str, Document] = {}
        max_workers = min(len(files), self.config.COLLECTION_FANOUT_WORKERS)

        def _one(fname: str) -> list[Document]:
            return self._raw_retrieve(
                query, filename_filter=fname, page_filter=page_filter, fetch_k=per_file_k
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, f): f for f in files}
            for fut in as_completed(futures):
                try:
                    for d in fut.result():
                        out.setdefault(_doc_key(d), d)
                except Exception as exc:
                    self.logger.warning("fan-out retrieve failed for %s: %s", futures[fut], exc)
        return list(out.values())

    def retrieve_collection(
        self,
        query: str,
        filename_filters: list[str],
        page_filter: str = None,
        query_embedding: list = None,
    ) -> list[Document]:
        """Coverage-aware retrieval across many files (accuracy-tuned, decision 2a).

        - 0 files  -> [] (empty collection)
        - 1 file   -> normal single-file retrieve (no behavior change)
        - >=2 files-> per-file coverage path: pool -> (hybrid) -> rerank whole pool ->
                      coverage-aware final selection (round-robin per file + lexical MMR).
        """
        cfg = self.config
        files = filename_filters or []
        n = len(files)

        if n == 0:
            return []
        if n == 1:
            if query_embedding:
                return self.retrieve_by_vector(query_embedding, query, filename_filter=files[0], page_filter=page_filter)
            return self.retrieve(query, filename_filter=files[0], page_filter=page_filter)

        # 1) Candidate pool with per-file representation
        if n <= cfg.COLLECTION_FANOUT_MAX_FILES:
            pool = self._fanout_per_file(query, files, cfg.COLLECTION_PER_FILE_K, page_filter)
        else:
            pool_k = min(n * cfg.COLLECTION_POOL_PER_FILE, cfg.COLLECTION_POOL_CAP)
            pool = self._raw_retrieve(query, filename_filters=files, page_filter=page_filter, fetch_k=pool_k)

        if not pool:
            return []

        # 2) Optional hybrid BM25+RRF over the pool (keyword precision across files).
        if cfg.COLLECTION_USE_HYBRID and len(pool) > 1:
            try:
                hybrid_cls = _get_hybrid_retriever_class()
                hybrid = hybrid_cls(top_k=len(pool))
                pool = hybrid.retrieve(query, pool)
            except Exception as exc:
                self.logger.warning("collection hybrid fusion failed (non-fatal): %s", exc)

        # 3) Rerank the WHOLE pool so every candidate gets a relevance score.
        if self._reranker and len(pool) > 1:
            pool = self._reranker.rerank_scores(query, pool)

        # 4) Coverage-aware final selection. k_final scales with #files.
        k_final = min(
            cfg.COLLECTION_RERANK_TOP_K_BASE + cfg.COLLECTION_RERANK_TOP_K_PER_FILE * n,
            cfg.COLLECTION_RERANK_TOP_K_CAP,
        )
        result = self._select_with_coverage(
            pool, k_final=k_final,
            min_per_file=cfg.COLLECTION_MIN_PER_FILE,
            jaccard_thresh=cfg.COLLECTION_MMR_JACCARD,
        )
        files_covered = len({_doc_file(d) for d in result})
        self.logger.info(
            "retrieve_collection: %d files -> pool=%d -> %d docs covering %d/%d files (k_final=%d)",
            n, len(pool), len(result), files_covered, n, k_final,
        )
        return result

    # ── Delete helper ──

    def delete_document_by_filename(self, filename: str):
        try:
            self.vectorstore.delete(filter={"filename": filename})
            self.logger.info("Deleted documents with filename %s", filename)
        except Exception as e:
            self.logger.error(
                "Failed to delete documents with filename %s: %s", filename, e
            )
