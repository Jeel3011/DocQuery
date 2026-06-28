"""`search_knowledge` tool — the agent's hand into the shared legal Knowledge Base
(G8_KNOWLEDGE_BASE_PLAN §G8.2).

When the user's OWN documents don't contain the law the agent needs (a statute, a
constitutional Article, a precedent), it searches the shared, read-only KB corpus
(the `kb_in` namespace) through this tool — and cites what it finds through the SAME
cite-or-abstain output gate as `search_vault`. The moat — "every claim traces to a
retrieved passage, or we abstain" — extended from the user's cells to the law itself.

The discipline (§G8.2): this tool returns EVIDENCE — cited passages, each carrying a
citation id + source + as_of — NEVER a written legal conclusion. The agent reasons,
the output gate binds it. Same contract as every tool.

Modeled on `tools/search.py`: a thin adapter over a KB-scoped `RetrievalManager`
(built from a Config whose `PINECONE_NAMESPACE = KNOWLEDGE_NAMESPACE`). The retriever
ranks; we shape into the uniform §3.3 envelope so a knowledge span is INDISTINGUISHABLE
to the ledger and the gate from a `search_vault` span.

  - `source` / `instrument_type` → conjunctive metadata filters (the chips, §G8.7).
  - `as_of`                      → the version-in-force filter (§G8.5): drop provisions
                                   not vouchably in force on that date; a span past the
                                   snapshot horizon is WITHHELD, never cited as current.

Behind `config.USE_KNOWLEDGE` at the loop layer (off ⇒ never offered ⇒ byte-identical).
`@safe_tool` — never raises. Gate: `eval/test_tools.py` with a stub KB manager ($0).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "search_knowledge",
    "description": (
        "Search the shared legal Knowledge Base — Indian statutes, regulations, and "
        "case law — for authority relevant to a query, when the user's OWN documents do "
        "not contain the law you need. Returns cited passages, each with its citation "
        "(e.g. 'Constitution Art.21', 'Companies Act 2013 s.149'), source, and the date "
        "through which it is known in force — NOT a written legal conclusion. Use "
        "'as_of' to get the version of the law in force on a given date; a provision we "
        "cannot vouch for on that date is withheld, never cited. Quote what this returns; "
        "never state law from memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The legal question or topic to find authority for."},
            "jurisdiction": {
                "type": "string",
                "description": "Jurisdiction to restrict to (default 'IN' — India).",
            },
            "source": {
                "type": "string",
                "description": (
                    "Restrict to one source key (e.g. 'constitution_of_india', "
                    "'companies_act_2013'). Omit to search all in-scope sources."
                ),
            },
            "instrument_type": {
                "type": "string",
                "enum": ["act", "regulation", "circular", "judgment", "article", "schedule", "rule"],
                "description": "Restrict to one instrument type (e.g. 'article', 'judgment').",
            },
            "as_of": {
                "type": "string",
                "description": (
                    "ISO date (YYYY-MM-DD). Return only the law in force on this date; "
                    "withhold provisions we cannot vouch for as-of then. Use the matter/"
                    "transaction date so the law cited matches the facts' date."
                ),
            },
            "k": {"type": "integer", "description": "How many passages to return (default 8)."},
        },
        "required": ["query"],
    },
}


def _kb_span_to_dict(doc: Any) -> Dict[str, Any]:
    """Serialize a retrieved KB chunk into the uniform §3.3 span shape — but with the
    legal addressing (`citation`, `source`, `as_of`) the gate's `[...]` markers need.

    `doc` is the filename-anchored marker `search_vault` spans carry; here the gate's
    `[Constitution Art.21]` marker is the CITATION, so we set `doc` to the citation and
    keep the structured fields beside it. Never raises; missing fields degrade to None.
    """
    md = getattr(doc, "metadata", None) or {}
    citation = md.get("citation")
    return {
        "kind": "knowledge",
        # `doc` is what the gate's bracket-marker references — the legal citation here.
        "doc": citation or md.get("source_key") or md.get("source"),
        "citation": citation,
        "source": md.get("source_key") or md.get("source"),
        "instrument_type": md.get("instrument_type"),
        "section_or_article_id": md.get("section_or_article_id"),
        "jurisdiction": md.get("jurisdiction"),
        "as_of": md.get("as_of_date"),
        "enacted": md.get("enacted_date"),
        "repealed": bool(md.get("repealed")),
        "superseded_by": md.get("superseded_by"),
        "page": md.get("page_number"),
        "chunk_id": md.get("chunk_id") or md.get("id"),
        "score": md.get("score") or md.get("relevance_score"),
        "snippet": (getattr(doc, "page_content", "") or "")[:500],
    }


def _vouchable_on(span_md: Dict[str, Any], as_of: Optional[str]) -> bool:
    """Version-in-force filter (§G8.5), applied to a retrieved span's metadata.

    A span is dropped when it is repealed/superseded (a 'wrong cell' — never current),
    or when we cannot vouch it was in force on `as_of`: before its enacted date, or
    AFTER our snapshot horizon (`as_of_date`) — past which the law may have changed and
    we will not guess. Mirrors `Provision.is_in_force_on`, working off the vector's
    stamped metadata so no relational round-trip is needed at query time.
    """
    # Repealed / superseded → never current.
    if span_md.get("repealed") or span_md.get("superseded_by"):
        return False
    if not as_of:
        return True  # no date asked → current-snapshot semantics (already not repealed)
    enacted = span_md.get("enacted_date")
    horizon = span_md.get("as_of_date")
    if enacted and as_of < enacted:
        return False  # not yet in force
    if horizon and as_of > horizon:
        return False  # past our vouchable snapshot — withhold, never guess
    return True


# ── S-D: KB retrieval quality grading (T1 pattern, external signal — T7 rule) ──
# Applied to the KB's score distribution exactly as _grade_spans does for search_vault.
# A "weak" statute match must signal weak in the envelope, not pass as authority.
# Import is deferred to avoid a circular dependency at module load time.

def _grade_kb_spans(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade KB retrieval quality from the span score distribution.

    Mirrors search.py's _grade_spans: reads scores already in the KB spans — $0, no LLM.
    Returns {"grade": strong|weak|empty, "top_score": float|None}.
    """
    if not spans:
        return {"grade": "empty", "top_score": None}
    scores = [s.get("score") for s in spans if s.get("score") is not None]
    if not scores:
        return {"grade": "strong", "top_score": None}  # no scores → treat as strong (no alarm)
    _STRONG = 0.55
    _GAP = 0.10
    scores_sorted = sorted(scores, reverse=True)
    top = scores_sorted[0]
    if top < _STRONG:
        return {"grade": "weak", "top_score": round(top, 4)}
    if len(scores_sorted) == 1:
        return {"grade": "strong", "top_score": round(top, 4)}
    median = scores_sorted[len(scores_sorted) // 2]
    grade = "strong" if (top - median) >= _GAP else "weak"
    return {"grade": grade, "top_score": round(top, 4)}


def _section_exists_in_span(span: Dict[str, Any]) -> bool:
    """S-D: verify the cited §/article actually appears in the retrieved text.

    The citation field encodes the reference (e.g. "Companies Act 2013 s.149" or
    "Constitution Art.21"). We check that the section/article identifier embedded in
    `section_or_article_id` (e.g. "149" or "21") is present in either the citation
    string or the snippet — a basic hallucination guard: if the retriever returned
    the wrong chunk, the cited provision number will be absent.

    Returns True when the check passes (or cannot be applied — we don't false-alarm
    when section_or_article_id is absent).
    """
    sec_id = span.get("section_or_article_id")
    if not sec_id:
        return True  # no id to check → pass (can't verify, don't block)
    sec_str = str(sec_id).strip()
    citation = str(span.get("citation") or "")
    snippet = str(span.get("snippet") or "")
    return sec_str in citation or sec_str in snippet


@safe_tool
def search_knowledge(
    query: str,
    kb_retrieval_manager: Any,
    *,
    jurisdiction: str = "IN",
    source: Optional[str] = None,
    instrument_type: Optional[str] = None,
    as_of: Optional[str] = None,
    k: int = 8,
    allowed_instrument_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search the KB for `query`; return the §3.3 envelope with cited legal spans.

    `kb_retrieval_manager` is a live `RetrievalManager` scoped to the `kb_in` namespace
    (built from a KB Config). The metadata filters (`source`/`instrument_type`/
    `jurisdiction`) narrow conjunctively; `as_of` post-filters the retrieved spans to
    those vouchably in force. Returns EVIDENCE — never a conclusion.

    S-D: the envelope now carries a `kb_retrieval_quality` grade (strong|weak|empty)
    mirroring T1's signal for search_vault — a weak statute match signals weak, not
    authority.  Spans whose cited §/article id is absent from the snippet are flagged
    with `section_verified: false` (a hallucinated-cite guard).
    """
    if not query:
        return error_result(
            "search_knowledge requires a non-empty 'query'. "
            "Provide a specific legal topic or question "
            "(e.g. 'right to privacy under Article 21', 'director duties Companies Act 2013')."
        )
    if kb_retrieval_manager is None:
        return error_result(
            "search_knowledge requires a kb_retrieval_manager (internal: USE_KNOWLEDGE may be "
            "off or the KB namespace was not threaded into this run — routing bug, not a model error)."
        )

    # G8.7 source-chip allow-list (server-side gate): if the run restricts which instrument
    # types are allowed, a model-requested type OUTSIDE the allow-list is dropped to "no
    # results" (the user turned that source off), and an unspecified request is narrowed to
    # the allowed set. None ⇒ no restriction. The post-filter below is the hard enforcement
    # (a retriever filter alone could be bypassed by metadata gaps).
    allow = set(allowed_instrument_types) if allowed_instrument_types else None
    if allow is not None and instrument_type and instrument_type not in allow:
        return ok_result(
            summary=f"search_knowledge {query!r}: 0 passages (instrument_type "
                    f"{instrument_type!r} is not an enabled source)",
            data={"passages": [], "as_of": as_of, "withheld": 0,
                  "blocked_by_source_filter": True},
            provenance=[],
        )

    # Conjunctive metadata narrowing (the chips + jurisdiction). Only set keys that were
    # asked — an absent filter searches all in-scope authority.
    metadata_filter: Dict[str, Any] = {}
    if jurisdiction:
        metadata_filter["jurisdiction"] = jurisdiction
    if source:
        metadata_filter["source_key"] = source
    if instrument_type:
        metadata_filter["instrument_type"] = instrument_type

    # Over-fetch when an as_of OR a source-chip allow-list will prune, so we still return ~k.
    fetch_k = max(k * 2, k) if (as_of or allow is not None) else k
    docs = kb_retrieval_manager.retrieve(
        query,
        metadata_filter=metadata_filter or None,
        top_k=fetch_k,
        apply_threshold=False,
        use_reranker=True,
    )
    spans_all = [_kb_span_to_dict(d) for d in (docs or [])]

    # §G8.5 version-in-force + G8.7 source-chip allow-list: drop spans we cannot vouch for
    # as-of the asked date, AND drop any span whose instrument type the user turned off.
    withheld = 0
    blocked = 0
    spans: List[Dict[str, Any]] = []
    for s in spans_all:
        if allow is not None and s.get("instrument_type") not in allow:
            blocked += 1
            continue
        # Re-derive the raw md keys the filter needs from the serialized span.
        md = {
            "repealed": s.get("repealed"),
            "superseded_by": s.get("superseded_by"),
            "enacted_date": s.get("enacted"),
            "as_of_date": s.get("as_of"),
        }
        if _vouchable_on(md, as_of):
            spans.append(s)
        else:
            withheld += 1
    spans = spans[: max(k, 1)]

    # S-D: verify the cited section/article id is present in each span's snippet.
    # A span that fails this check carries section_verified=False — the agent sees it
    # and knows the citation may not match the retrieved text (hallucinated-cite signal).
    for sp in spans:
        sp["section_verified"] = _section_exists_in_span(sp)

    # S-D: grade the KB retrieval quality (T1 pattern, external signal — T7 rule).
    # A weak grade (low scores, clustered, or empty) signals the result is uncertain;
    # the agent must not treat a weak KB match as authoritative.
    kb_rq = _grade_kb_spans(spans)
    rq_payload: Dict[str, Any] = {"grade": kb_rq["grade"], "top_score": kb_rq["top_score"]}
    if kb_rq["grade"] != "strong":
        rq_payload["repair"] = (
            f"KB retrieval grade={kb_rq['grade']}; consider broadening the query, "
            "specifying a different source, or checking whether the KB contains this provision."
        )

    note = f" [{kb_rq['grade']}]"
    if as_of and withheld:
        note += f" ({withheld} withheld as not-in-force on {as_of})"
    if blocked:
        note += f" ({blocked} hidden by source filter)"
    return ok_result(
        summary=f"search_knowledge {query!r}: {len(spans)} cited passage(s){note}",
        data={"passages": spans, "as_of": as_of, "withheld": withheld, "blocked": blocked,
              "kb_retrieval_quality": rq_payload},
        provenance=spans,
    )
