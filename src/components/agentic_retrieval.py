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
            ("system", f"""Break the user's question into {self.MAX_SUB_QUERIES} or fewer
atomic sub-questions that together fully answer the original. Return ONLY the
sub-questions, one per line, no numbering. If already atomic, return it unchanged."""),
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
    ) -> Dict:
        """Decompose → retrieve per sub-query → deduplicate → return merged docs."""
        sub_queries = self.decompose_query(query)
        seen: dict[str, Document] = {}

        for sq in sub_queries:
            try:
                for doc in self.retrieval_mgr.retrieve(sq, filename_filter, page_filter):
                    key = doc.metadata.get("chunk_id") or doc.page_content[:100]
                    if key not in seen:
                        seen[key] = doc
            except Exception as exc:
                logger.warning("Retrieval failed for sub-query '%s': %s", sq, exc)

        merged = list(seen.values())
        logger.info("Agentic: %d sub-queries → %d unique docs", len(sub_queries), len(merged))
        return {"docs": merged, "sub_queries": sub_queries, "unique_docs": len(merged)}
