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
    """
    if not query:
        return error_result("search_knowledge requires a non-empty 'query'")
    if kb_retrieval_manager is None:
        return error_result("search_knowledge requires a kb_retrieval_manager")

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

    note = f" ({withheld} withheld as not-in-force on {as_of})" if (as_of and withheld) else ""
    if blocked:
        note += f" ({blocked} hidden by source filter)"
    return ok_result(
        summary=f"search_knowledge {query!r}: {len(spans)} cited passage(s){note}",
        data={"passages": spans, "as_of": as_of, "withheld": withheld, "blocked": blocked},
        provenance=spans,
    )
