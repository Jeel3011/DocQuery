"""
DocQuery — Agentic Retrieval (Phase 6)

Implements Harvey AI's agentic pattern: decompose → multi-retrieve → deduplicate.
"""

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.components.config import Config
from src.components.retrieval import _doc_key
from src.logger import get_logger

logger = get_logger(__name__)


class AgenticRetriever:
    """
    Multi-step agentic retriever:
      1. decompose_query — break complex query into 2-4 atomic sub-queries
      2. retrieve_and_synthesize — retrieve per sub-query, deduplicate, return merged docs

    Inspired by Harvey's agent decomposition pattern. Falls back to single-pass
    retrieval gracefully if decomposition fails.
    """

    MAX_SUB_QUERIES = 4

    def __init__(self, config: Config, retrieval_mgr):
        self.config = config
        self.retrieval_mgr = retrieval_mgr
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            temperature=0.0,
            api_key=config.OPENAI_API_KEY,
            request_timeout=20,
        )

    def decompose_query(self, query: str) -> List[str]:
        """Break a complex query into atomic sub-questions. Falls back to [query] on error."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a document search query decomposer. Your job is to break
the user's question into {self.MAX_SUB_QUERIES} or fewer specific search queries
that can be used to find relevant information in a document collection.

Rules:
- Each sub-query MUST be a search query, NOT a clarifying question.
- Never generate questions like "What topic are you interested in?" — those are useless for search.
- If the query is vague (e.g. "tell me more about"), generate broad search queries that cover the main topics likely in the documents.
- If already atomic and specific, return it unchanged.
- Return ONLY the search queries, one per line, no numbering.

Examples:
  Input: "tell me more about"
  Output:
  main topics and overview
  key projects and experience
  summary of document contents

  Input: "explain the architecture and performance"
  Output:
  system architecture and design
  performance benchmarks and results"""),
            ("user", "{query}"),
        ])
        try:
            raw = (prompt | self.llm | StrOutputParser()).invoke({"query": query})
            subs = [q.strip() for q in raw.strip().split("\n") if q.strip()]
            subs = subs[:self.MAX_SUB_QUERIES] or [query]
            logger.info("Decomposed into %d sub-queries", len(subs))
            return subs
        except Exception as exc:
            logger.warning("Decomposition failed (%s), using original.", exc)
            return [query]

    def retrieve_and_synthesize(
        self,
        query: str,
        filename_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        filename_filters: Optional[list[str]] = None,
    ) -> Dict:
        """Decompose → retrieve per sub-query in parallel → deduplicate → return merged docs.

        Sub-queries are retrieved concurrently using a thread pool (I/O-bound Pinecone calls).
        With 4 sub-queries running in parallel, retrieval time drops from ~1.2-3.2s → ~400ms.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sub_queries = self.decompose_query(query)
        seen: dict[str, Document] = {}

        def _fetch(sq: str) -> list:
            try:
                return self.retrieval_mgr.retrieve(sq, filename_filter, page_filter, filename_filters=filename_filters)
            except Exception as exc:
                logger.warning("Retrieval failed for sub-query '%s': %s", sq, exc)
                return []

        # Run all sub-query retrievals in parallel — Pinecone calls are I/O-bound
        max_workers = min(len(sub_queries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, sq): sq for sq in sub_queries}
            for future in as_completed(futures):
                for doc in future.result():
                    key = _doc_key(doc)
                    if key not in seen:
                        seen[key] = doc

        merged = list(seen.values())

        # Fallback: if all sub-queries returned nothing, try the original query directly
        if not merged:
            logger.info("Agentic: sub-queries returned 0 docs, falling back to original query")
            try:
                merged = self.retrieval_mgr.retrieve(query, filename_filter, page_filter, filename_filters=filename_filters)
            except Exception as exc:
                logger.warning("Fallback retrieval failed: %s", exc)

        # Phase 2: Web search fallback — if documents returned nothing, search the web
        web_search_used = False
        if not merged and self.config.USE_WEB_FALLBACK:
            try:
                from src.components.web_search import WebSearcher
                searcher = WebSearcher(self.config)
                if searcher.is_available():
                    web_docs = searcher.search(query, max_results=self.config.WEB_SEARCH_MAX_RESULTS)
                    if web_docs:
                        merged.extend(web_docs)
                        web_search_used = True
                        logger.info("Agentic: web fallback added %d results", len(web_docs))
            except Exception as exc:
                logger.warning("Web search fallback failed (non-fatal): %s", exc)

        logger.info(
            "Agentic: %d sub-queries (parallel) → %d unique docs",
            len(sub_queries), len(merged)
        )
        return {
            "docs": merged,
            "sub_queries": sub_queries,
            "unique_docs": len(merged),
            "web_search_used": web_search_used,
        }
