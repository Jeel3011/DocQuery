"""
DocQuery — Web Search Fallback (Phase 2)

Provides web search capability when document retrieval returns
insufficient or zero results. Uses Tavily API for LLM-optimized
search results.

Usage:
    searcher = WebSearcher(config)
    docs = searcher.search("What is transformer architecture?", max_results=3)
"""

import os
from typing import List, Optional

from langchain_core.documents import Document

from src.components.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class WebSearcher:
    """Web search fallback using Tavily API.

    Returns LangChain Document objects with metadata.source = "web" so the
    generator can cite them distinctly from document-sourced chunks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.api_key = getattr(config, "TAVILY_API_KEY", None) or os.getenv("TAVILY_API_KEY", "")
        self.enabled = bool(self.api_key) and getattr(config, "USE_WEB_FALLBACK", True)
        self._client = None

    def _get_client(self):
        """Lazy-init the Tavily client to avoid import cost at module load."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("tavily-python not installed. Web search disabled.")
                self.enabled = False
                return None
        return self._client

    def search(
        self,
        query: str,
        max_results: int = None,
    ) -> List[Document]:
        """Search the web and return results as LangChain Documents.

        Args:
            query: Search query string.
            max_results: Max results to return (default from config).

        Returns:
            List of Document objects with source="web" metadata.
            Empty list if web search is disabled or fails.
        """
        if not self.enabled:
            logger.debug("Web search disabled — no API key or USE_WEB_FALLBACK=False")
            return []

        max_results = max_results or getattr(self.config, "WEB_SEARCH_MAX_RESULTS", 3)

        client = self._get_client()
        if client is None:
            return []

        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
            )

            docs = []
            for result in response.get("results", []):
                content = result.get("content", "").strip()
                if not content:
                    continue

                title = result.get("title", "Web Result")
                url = result.get("url", "")

                doc = Document(
                    page_content=f"[{title}]\n{content}",
                    metadata={
                        "source": "web",
                        "filename": f"🌐 {title}",
                        "url": url,
                        "chunk_type": "web",
                        "page_number": None,
                        "relevance_score": result.get("score", 0),
                    },
                )
                docs.append(doc)

            logger.info("Web search returned %d results for: %.60s", len(docs), query)
            return docs

        except Exception as exc:
            logger.warning("Web search failed (non-fatal): %s", exc)
            return []

    def is_available(self) -> bool:
        """Check if web search is configured and available."""
        return self.enabled and bool(self.api_key)
