"""
DocQuery — Multi-Hop Retrieval (Phase 4.5 / T1.5)

Replaces the independent-sub-query AgenticRetriever (decompose → retrieve all
sub-queries in parallel → dedup) with a SEQUENTIAL ReAct/Self-RAG loop:

    retrieve → reason over accumulated evidence → detect the missing link →
    issue ONE follow-up query INFORMED BY findings → repeat, bounded by a hop budget.

The difference that matters: the parallel decomposer fires all its sub-queries at
once, so a sub-query can't use what an earlier one found. That's fine for "explain
the architecture AND performance" (two independent facets) but wrong for a *bridge*
question — "which subsidiary did the company that acquired X spin off?" — where you
must first find the acquirer, THEN search for its spin-offs. The loop does exactly
that: it goes back for one more file once it knows what's missing. This is the first
genuinely human reasoning behavior (plan §4.5).

Discipline carried from AgenticRetriever:
  - Same return contract (docs / sub_queries / unique_docs / web_search_used) so it's
    a drop-in for the chat agent endpoints.
  - Never raises: any LLM/retrieval failure degrades to single-pass retrieval, never
    worse than the non-agentic path.
  - Opt-in via Config.USE_MULTIHOP; the proven parallel path stays the default.
"""

import re
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.components.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


# The gap detector reads the running evidence and either stops or proposes ONE
# follow-up. A plain System/Human invoke (no ChatPromptTemplate) avoids the {}-brace
# mangling that bites on document text (same lesson as table_extraction.summarize_table).
_GAP_SYSTEM = (
    "You control retrieval for a document-Q&A system answering a MULTI-HOP question — "
    "one whose answer is reached by chaining facts: find fact A, then USE A to find B.\n\n"
    "You are given the question and the evidence gathered so far. Decide the next step.\n\n"
    "Reply in EXACTLY this format, two lines:\n"
    "  FOUND: <the concrete fact(s) the evidence now establishes — a specific year, "
    "name, entity, or figure; or 'nothing relevant yet'>\n"
    "  NEXT: SUFFICIENT   (if FOUND already contains everything needed to answer)\n"
    "         — or —\n"
    "  NEXT: <one search query that uses a fact from FOUND to retrieve the STILL-MISSING piece>\n\n"
    "Hard rules:\n"
    "- The NEXT query MUST substitute the concrete fact you just learned. It must NOT be a "
    "reworded copy of the original question — if you only rephrased the question, you failed.\n"
    "- It targets the ONE missing link the evidence references but does not yet contain.\n"
    "- It is search keywords a retriever can match, never a question to the user.\n"
    "- Output NEXT: SUFFICIENT the moment the evidence holds BOTH halves of the chain.\n\n"
    "Example — Question: 'In the year AWS operating income first exceeded $20B, what was "
    "Amazon total net sales?'\n"
    "  Evidence shows AWS operating income hit $22.6B in 2021.\n"
    "  FOUND: AWS operating income first exceeded $20B in fiscal 2021\n"
    "  NEXT: Amazon total net sales fiscal year 2021\n"
    "(Note the query carries '2021' — the learned fact — not a rephrase of the question.)"
)


class MultiHopRetriever:
    """Sequential, finding-informed retrieval loop (ReAct/Self-RAG, bounded hops).

    Mirrors AgenticRetriever's public surface (``retrieve_and_synthesize``) so the
    chat endpoints can swap it in behind ``Config.USE_MULTIHOP`` with no other change.
    """

    def __init__(self, config: Config, retrieval_mgr):
        self.config = config
        self.retrieval_mgr = retrieval_mgr
        self.max_hops = max(1, getattr(config, "MULTIHOP_MAX_HOPS", 3))
        self.per_hop_k = max(1, getattr(config, "MULTIHOP_PER_HOP_K", 5))
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            temperature=0.0,
            api_key=config.OPENAI_API_KEY,
            request_timeout=20,
        )

    # ── one retrieval round ─────────────────────────────────────────────────
    def _retrieve(
        self,
        query: str,
        filename_filter: Optional[str],
        page_filter: Optional[int],
        filename_filters: Optional[List[str]],
    ) -> List[Document]:
        try:
            return self.retrieval_mgr.retrieve(
                query,
                filename_filter,
                page_filter,
                filename_filters=filename_filters,
                top_k=self.per_hop_k,
            )
        except Exception as exc:
            logger.warning("Multi-hop: retrieval failed for '%s': %s", query, exc)
            return []

    # ── the gap detector: stop, or propose the next informed query ──────────
    def _next_query(self, question: str, docs: List[Document]) -> Optional[str]:
        """Inspect accumulated evidence. Return a follow-up query, or None to stop.

        Returns None on SUFFICIENT, on any failure, or on a degenerate/repeat query —
        so a broken detector simply ends the loop with what we already have.
        """
        # Compact evidence digest — labels + a content snippet, capped so the prompt
        # stays cheap as the running set grows across hops.
        lines = []
        for d in docs[:12]:
            fn = (d.metadata or {}).get("filename") or "?"
            pg = (d.metadata or {}).get("page_number")
            head = (d.page_content or "")[:300].replace("\n", " ")
            lines.append(f"[{fn}{f' p{pg}' if pg else ''}] {head}")
        evidence = "\n".join(lines) if lines else "(no evidence retrieved yet)"

        user = f"Question: {question}\n\nEvidence so far:\n{evidence}"
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_GAP_SYSTEM), HumanMessage(content=user)]
            )
            text = (resp.content or "").strip()
        except Exception as exc:
            logger.warning("Multi-hop: gap detector failed (%s); stopping loop", exc)
            return None

        follow = self._parse_next(text)
        if follow is None:
            return None  # SUFFICIENT, empty, or unparseable → stop
        # Guard against a no-op hop: an empty/echo query, or one that just rephrases the
        # original (the reformulation failure mode) — none of those make progress.
        if not follow or self._too_similar(follow, question):
            logger.info("Multi-hop: follow-up '%s' rephrases the question; stopping", follow)
            return None
        return follow

    @staticmethod
    def _parse_next(text: str) -> Optional[str]:
        """Extract the NEXT query from the FOUND/NEXT reply. None = stop.

        Tolerant: handles the two-line format, a bare 'SUFFICIENT', a legacy
        'SEARCH: x' line, or a single query line — so a model that drifts from the
        exact format still drives the loop instead of silently aborting it.
        """
        if not text:
            return None
        nxt = None
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith("NEXT:"):
                nxt = s.split(":", 1)[1].strip()
                break
        # Fall back to the whole reply if there was no NEXT: line (drifted format).
        candidate = nxt if nxt is not None else text.strip()
        if not candidate or candidate.upper().startswith("SUFFICIENT"):
            return None
        # Strip a legacy/leading "SEARCH" verb and surrounding quotes.
        if candidate.upper().startswith("SEARCH"):
            candidate = candidate[len("SEARCH"):].lstrip(": ").strip()
        return candidate.strip().strip('"').strip() or None

    @staticmethod
    def _too_similar(follow: str, question: str, threshold: float = 0.7) -> bool:
        """True if the follow-up makes NO progress over the original question.

        The right test for a multi-hop bridge is NOT word-overlap (a real bridge query
        reuses the question's nouns — "Amazon total net sales" — and just ADDS the fact
        it learned, e.g. "2021"; that high overlap is correct, not a rephrase). The test
        is: does the follow-up introduce at least one CONTENT WORD the question did not
        already contain? A learned fact (a year, name, or figure) is a new token →
        progress. Zero new tokens → the detector only reworded the question → no progress,
        stop. ``threshold`` is unused now (kept for signature stability).
        """
        def words(s):
            return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}
        fw, qw = words(follow), words(question)
        if not fw:
            return True
        if follow.strip().lower() == question.strip().lower():
            return True
        # New content words the question lacked = the learned fact(s). None ⇒ rephrase.
        return len(fw - qw) == 0

    # ── public entry: same contract as AgenticRetriever ─────────────────────
    def retrieve_and_synthesize(
        self,
        query: str,
        filename_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        filename_filters: Optional[List[str]] = None,
    ) -> Dict:
        """Run the bounded hop loop and return the merged, deduplicated evidence.

        ``sub_queries`` carries the actual hop TRAIL (original question first, then each
        follow-up), so the UI's existing "decomposition" panel shows the reasoning path.
        """
        seen: Dict[str, Document] = {}
        hop_trail: List[str] = []

        next_q: Optional[str] = query
        for hop in range(self.max_hops):
            if not next_q:
                break
            hop_trail.append(next_q)
            fresh = self._retrieve(next_q, filename_filter, page_filter, filename_filters)
            for doc in fresh:
                key = (doc.metadata or {}).get("chunk_id") or doc.page_content[:100]
                if key not in seen:
                    seen[key] = doc

            # Last allowed hop → no point asking for another query we can't run.
            if hop == self.max_hops - 1:
                break
            next_q = self._next_query(query, list(seen.values()))
            # Guard against a repeated hop: the detector sometimes re-proposes a query
            # it already ran (seen live: hops 2 and 3 identical). Re-running it retrieves
            # the same docs and burns the budget — stop instead.
            if next_q and any(
                next_q.strip().lower() == prev.strip().lower() for prev in hop_trail
            ):
                logger.info("Multi-hop: follow-up '%s' repeats an earlier hop; stopping", next_q)
                next_q = None

        merged = list(seen.values())

        # Fallback: nothing matched any hop → try the raw query once more directly
        # (mirrors AgenticRetriever, covers the degenerate "every hop empty" case).
        if not merged:
            logger.info("Multi-hop: 0 docs after %d hop(s); retrying original query", len(hop_trail))
            merged = self._retrieve(query, filename_filter, page_filter, filename_filters)

        # Web-search fallback — identical policy/shape to the parallel path.
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
                        logger.info("Multi-hop: web fallback added %d results", len(web_docs))
            except Exception as exc:
                logger.warning("Multi-hop: web search fallback failed (non-fatal): %s", exc)

        logger.info(
            "Multi-hop: %d hop(s) %s → %d unique docs",
            len(hop_trail), hop_trail, len(merged),
        )
        return {
            "docs": merged,
            "sub_queries": hop_trail,
            "unique_docs": len(merged),
            "web_search_used": web_search_used,
        }
