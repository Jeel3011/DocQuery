"""
Phase 4.3 — table intent detection + grid loading (the retrieval-side glue).

Two jobs, both cheap and side-effect-free:

1. ``has_numeric_intent(question)`` — a fast keyword/regex gate (NO LLM, so it
   adds ~microseconds). The Analyst path only runs when this returns True, so a
   normal prose / single-doc question pays ZERO added latency and the proven
   fast path is never touched (PRIME DIRECTIVE).

2. ``load_grids_for_docs(...)`` — pull the structured table grids we stored at
   ingest (``chunk_type="table"`` rows, ``metadata.table_json``) for a set of
   doc_ids, so the Analyst can compute on them. Reads Supabase JSONB; no Pinecone
   round-trip needed (the grid is bookkeeping, not a vector).
"""

from __future__ import annotations

import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Verbs/nouns that signal the user wants a COMPUTED or tabular numeric answer.
# Deliberately broad on the positive side (a false positive just loads grids the
# Analyst may not use); the Analyst + verifier are the real gates downstream.
_NUMERIC_CUES = re.compile(
    r"\b("
    r"growth|grew|increase|decrease|decline|change|delta|"
    r"margin|ratio|percent|percentage|%|rate|"
    r"sum|total|aggregate|average|mean|cagr|compound|"
    r"compare|comparison|versus|vs\.?|difference|differ|"
    r"how much|how many|trend|year-over-year|yoy|"
    r"revenue|sales|income|profit|loss|expense|cost|"
    r"margin|ebitda|cash flow|assets|liabilities|equity"
    r")\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(r"\b\d{4}\b|\$|\bQ[1-4]\b")  # a year, currency, or quarter


def has_numeric_intent(question: str) -> bool:
    """Fast gate: does this question plausibly want a computed/tabular number?

    Requires at least one numeric/financial cue. This is intentionally cheap and
    err-on-include: the cost of a false positive is loading some grids; the cost
    of a false negative is missing a table answer, so we lean inclusive but still
    skip obviously non-numeric questions (the zero-latency fast-path case).
    """
    if not question:
        return False
    return bool(_NUMERIC_CUES.search(question)) or bool(_HAS_NUMBER.search(question))


def _relevance(question: str, grid_tj: Dict[str, Any], caption: str) -> float:
    """Cheap lexical relevance of a table to the question (no embeddings/LLM).

    Scores overlap between question terms and the table's captions + row labels +
    periods. Used to keep only the most relevant tables so the Analyst's LLM picks
    among a SMALL, on-topic set (not all ~45 tables in a doc), which keeps the
    spec-write prompt small/fast and the table-selection accurate.
    """
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not q:
        return 0.0
    hay_terms = set(re.findall(r"[a-z0-9]+", (caption or "").lower()))
    for r in grid_tj.get("rows", []):
        hay_terms |= set(re.findall(r"[a-z0-9]+", (r.get("label", "") + " " + r.get("section", "")).lower()))
    hay_terms |= {str(p).lower() for p in grid_tj.get("periods", [])}
    if not hay_terms:
        return 0.0
    return len(q & hay_terms) / len(q)


def grids_from_table_chunks(table_chunks) -> list:
    """Rehydrate analyst.Grid objects from retrieved table chunks (semantic path).

    Each chunk's metadata carries ``table_json`` (the normalized grid) + filename +
    page. This is the PRODUCTION table-selection path: the chunks come from semantic
    retrieval over embedded captions, so the grid set is already the on-topic
    tables — generic across any document, by meaning not keywords. Never raises.
    """
    from src.components.brain.analyst import Grid

    grids = []
    for ch in table_chunks or []:
        md = getattr(ch, "metadata", None) or {}
        raw = md.get("table_json")
        if not raw:
            continue
        try:
            tj = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        grids.append(Grid(tj, doc=md.get("filename"), page=md.get("page_number")))
    return grids


def load_grids_for_docs(
    db_client,
    doc_ids: List[str],
    *,
    question: Optional[str] = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    top_grids: int = 8,
    max_tables_per_doc: int = 60,
):
    """Load structured table grids (analyst.Grid) for doc_ids, ranked by relevance.

    Pulls ``chunk_type="table"`` chunks from Supabase, rehydrates each
    ``metadata.table_json`` into a Grid with provenance, and — when ``question``
    is given — keeps only the ``top_grids`` most lexically relevant tables. This
    is the "table-aware retrieval" step: a numeric question gets the handful of
    on-topic tables, so the Analyst selects among a small, accurate set (small
    prompt, low latency, right table). Never raises.
    """
    from src.components.brain.analyst import Grid

    scored = []  # (relevance, Grid)
    for did in doc_ids:
        # Pull ONLY table chunks from the DB (metadata is JSONB), not every chunk.
        # On an 8-doc collection the naive "select all chunks, filter in Python"
        # loaded ~2600 rows of text per numeric query — needless memory pressure
        # that contributed to the local OOM. The JSONB filter keeps it to the few
        # hundred table rows. Falls back to the full pull if the filter is
        # unsupported, so it can never break.
        rows = None
        try:
            rows = (
                db_client.client.table("document_chunks")
                .select("content,metadata")
                .eq("document_id", did)
                .eq("metadata->>chunk_type", "table")
                .execute()
                .data
            )
        except Exception:
            rows = None
        if rows is None:
            try:
                rows = (
                    db_client.client.table("document_chunks")
                    .select("content,metadata")
                    .eq("document_id", did)
                    .execute()
                    .data
                    or []
                )
            except Exception as exc:
                logger.warning("[table_intent] grid load failed for doc %s: %s", did, exc)
                continue

        n = 0
        for r in rows:
            md = r.get("metadata") or {}
            if md.get("chunk_type") != "table":
                continue
            raw = md.get("table_json")
            if not raw:
                continue
            try:
                tj = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            doc_name = (filename_by_doc or {}).get(did) or md.get("filename")
            grid = Grid(tj, doc=doc_name, page=md.get("page_number"))
            rel = _relevance(question, tj, r.get("content", "")) if question else 1.0
            scored.append((rel, grid))
            n += 1
            if n >= max_tables_per_doc:
                break

    if question:
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = [s for s in scored if s[0] > 0][:top_grids] or scored[:top_grids]

    return [g for _rel, g in scored]
