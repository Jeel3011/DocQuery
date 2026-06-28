"""`search_vault` tool — thin adapter over retrieval (AGENT_CORE_PLAN §3.3).

Wraps `RetrievalManager.retrieve` (text) / `retrieve_table_chunks` (tables), with the
`DocumentRouter` invoked for large (>20-doc) scopes per §3b. Returns the top chunks as
spans with source addressing + score. No new intelligence: the retriever ranks, we
shape. Requires a live `RetrievalManager` (vector index), so its primary gate is a
schema/shape unit with a stub manager — the live recall fixture set is a Phase-B item.

T1 (CRAG — Corrective Retrieval-Augmented Generation):
  After retrieval, the score distribution is read to assign a deterministic
  retrieval-quality grade (`strong | weak | empty`).  This is an EXTERNAL signal
  (T7 rule): purely from the scores the retriever already returns, no LLM self-check.
  On `weak` or `empty` the tool attempts ONE query expansion (flag-gated via
  `_EXPAND_ON_WEAK`, default True): the query is broadened by trimming the longest
  quoted/parenthetical clause, then re-retrieval runs and spans are merged.  The
  envelope `data` carries a `retrieval_quality` sub-dict so the model knows to
  refine or abstract, and the loop summary names the repair taken.  Flag off ⇒
  byte-identical to before T1.

  LIVE gate owed: `eval/routing_recall.py` must pass (weak queries flagged, strong
  queries pass through); a live Pinecone fixture that deliberately feeds a
  semantically-distant query and asserts grade==`weak`.

Gate: `eval/test_tools.py` (envelope + never-raise + T1 grade fixtures).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool, span_to_dict

_ROUTER_DOC_THRESHOLD = 20  # >this many docs in scope → route first (§3b / CDB §7.3)

# ── T1: retrieval quality grading (CRAG, external signal — T7 rule) ──────────────
# Thresholds calibrated for text-embedding-3-small + Pinecone cosine scores (0–1).
# STRONG_THRESHOLD: a span at or above this is a genuine hit (not noise).
# GAP_THRESHOLD: top-vs-median gap that confirms the top result stands out.
# Flag: when False, the grading block is skipped and the old path is byte-identical.
_STRONG_THRESHOLD: float = 0.55   # cosine ≥ 0.55 = relevant hit for this embedding model
_GAP_THRESHOLD: float = 0.10      # top score must be ≥ median + this to be "strong"
_EXPAND_ON_WEAK: bool = True       # auto-expand query once on weak/empty (flag-gated)


def _grade_spans(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade retrieval quality from the span score distribution.

    Returns a dict with keys `grade` (strong|weak|empty), `top_score`, and
    `above_threshold` — the loop/model can use this to decide whether to refine.
    Purely structural: reads scores already in spans, no LLM, $0.
    """
    if not spans:
        return {"grade": "empty", "top_score": None, "above_threshold": 0}

    scores = [s.get("score") for s in spans if s.get("score") is not None]
    if not scores:
        # Scores missing (stub/table path) — treat as strong so we don't false-alarm.
        return {"grade": "strong", "top_score": None, "above_threshold": len(spans)}

    scores_sorted = sorted(scores, reverse=True)
    top = scores_sorted[0]
    above = sum(1 for sc in scores if sc >= _STRONG_THRESHOLD)

    if top < _STRONG_THRESHOLD:
        grade = "weak"
    elif len(scores_sorted) == 1:
        grade = "strong"
    else:
        median = scores_sorted[len(scores_sorted) // 2]
        grade = "strong" if (top - median) >= _GAP_THRESHOLD else "weak"

    return {"grade": grade, "top_score": round(top, 4), "above_threshold": above}


def _expand_query(query: str) -> Optional[str]:
    """Broaden a query by removing the most specific clause (parenthetical or trailing
    prepositional phrase).  Returns None if no broadening is possible.

    Examples:
      "governing law (dispute resolution)" → "governing law"
      "net sales in fiscal year 2023 Q2"   → "net sales"
    """
    # Strip trailing parenthetical.
    expanded = re.sub(r"\s*\([^)]*\)\s*$", "", query).strip()
    if expanded and expanded != query:
        return expanded
    # Strip the longest trailing prepositional phrase (in/for/of/during/at/under + rest).
    expanded = re.sub(
        r"\s+(in|for|of|during|at|under|within|between)\s+\S.*$", "", query,
        flags=re.IGNORECASE,
    ).strip()
    if expanded and expanded != query:
        return expanded
    return None


SCHEMA: Dict[str, Any] = {
    "name": "search_vault",
    "description": (
        "Semantically search the user's documents for passages relevant to a query. "
        "Returns the top matching chunks with their document, page, and a snippet. Use "
        "kind='table' to find structured financial tables, 'text' for prose, 'both' to "
        "mix. This finds WHERE to look; read_document/table_lookup then read the cells."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "scope": {
                "type": "object",
                "description": "Where to search.",
                "properties": {
                    "collection_id": {"type": "string"},
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                    "filenames": {"type": "array", "items": {"type": "string"}},
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional metadata narrowing (e.g. {\"doc_type\": "
                            "\"legal_contract\", \"fiscal_year\": {\"$in\": [2023]}}). "
                            "Conjunctive — narrows the vault scope, never replaces it."
                        ),
                    },
                },
            },
            "k": {"type": "integer", "description": "How many chunks to return (default 8)."},
            "kind": {
                "type": "string",
                "enum": ["text", "table", "both"],
                "description": "Search prose, table chunks, or both (default 'both').",
            },
        },
        "required": ["query"],
    },
}

@safe_tool
def search_vault(
    query: str,
    retrieval_manager: Any,
    *,
    scope: Optional[Dict[str, Any]] = None,
    k: int = 8,
    kind: str = "both",
    fidelity_by_doc: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Search the vault for `query`; return the §3.3 envelope with chunk spans.

    `retrieval_manager` is a live `RetrievalManager`. `scope` narrows the search
    (collection_id / doc_ids / filenames). `kind` selects text/table/both.

    S-E: `fidelity_by_doc` is an optional {doc_id: fidelity_grade} dict (e.g.
    {"uuid-1": "partial"}).  When a returned span's doc_id maps to a grade other
    than "good", the span carries `fidelity_warning: true` so the agent (and
    eventually the user) sees "this doc ingested poorly — results may be incomplete"
    instead of silently abstaining on it.  Flag-off: pass None → byte-identical.
    """
    if not query:
        return error_result(
            "search_vault requires a non-empty 'query'. "
            "Provide a specific phrase describing what you are looking for "
            "(e.g. 'governing law clause', 'total revenue 2022')."
        )
    if retrieval_manager is None:
        return error_result(
            "search_vault requires a retrieval_manager (internal: the run context did not "
            "inject one; this is a routing bug, not a model error — report it)."
        )

    scope = scope or {}
    collection_id = scope.get("collection_id")
    # G3: doc_id is the stable, ingest-stamped scope axis (vault isolation as a DATA
    # property). The retriever filters `doc_id $in` when given; the per-USER namespace
    # already isolates users, this isolates VAULTS. `filenames` stays as the legacy
    # fallback for any un-stamped vector — the retriever's scope cascade prefers doc_ids.
    doc_ids: Optional[List[str]] = scope.get("doc_ids")
    filenames: Optional[List[str]] = scope.get("filenames")
    # G3 Step B: doc_type / fiscal_year / … narrowing — CONJUNCTIVE on top of scope,
    # never a replacement (a bug there would be a cross-vault leak).
    metadata_filter: Optional[Dict[str, Any]] = scope.get("filters") or None

    # ── F1b: the VAULT-SCOPE FLOOR (the cross-vault leak guard) ──────────────────────────
    # Within a user, multiple vaults share the per-user Pinecone namespace, so an UNSCOPED
    # query (no doc_id / collection_id / filename filter) would fan out across ALL the user's
    # vaults — a cross-vault leak. The agent-core route ALWAYS runs inside a vault (it requires
    # collection_id; routes/agent_core.py), so the run carries `vault_active=True`. F1b makes
    # the vault scope a HARD FLOOR: when vault_active is set, the query MUST resolve to at
    # least one vault-scoping key (doc_ids / collection_id / filenames) — else we ERROR rather
    # than run unscoped. `collection_id` ALONE is a valid floor (the retriever filters on the
    # stamped field — vectors carry doc_id + collection_id from ingest, so NO re-ingest), which
    # binds a vault whose doc_ids weren't preloaded. The flag is separate from the scope keys
    # precisely so that "active but unscoped" is detectable (a key, not a falsy collection_id).
    vault_active = bool(scope.get("vault_active") or collection_id)
    has_vault_scope = bool(doc_ids or collection_id or filenames)
    if vault_active and not has_vault_scope:
        # A vault is active but nothing scopes the query → refuse (never fan out).
        return error_result(
            "search_vault refused an unscoped query inside an active vault "
            "(cross-vault leak guard). Scope to the vault's documents."
        )

    docs: List[Any] = []

    if kind in ("text", "both"):
        # `retrieve` handles the doc_ids axis end-to-end: len>1 balances per-doc, len==1
        # becomes a scalar doc_id filter. Filenames are the legacy fallback only when no
        # doc_ids are present (un-stamped vectors).
        text_docs = retrieval_manager.retrieve(
            query,
            doc_ids=doc_ids or None,
            filename_filters=(filenames if filenames and len(filenames) > 1 else None) if not doc_ids else None,
            filename_filter=(filenames[0] if filenames and len(filenames) == 1 else None) if not doc_ids else None,
            metadata_filter=metadata_filter,
            # F1b: bind the text query to the active vault when doc_ids/filenames aren't the
            # scope axis — the floor's collection_id-only case (the retriever filters on the
            # stamped `collection_id` field via _build_filter's scope cascade).
            collection_id=(collection_id if not doc_ids and not filenames else None),
            top_k=k,
            apply_threshold=False,
            use_reranker=True,
        )
        docs.extend(text_docs or [])

    if kind in ("table", "both"):
        # G3: scope table chunks by doc_id (stable, ingest-stamped) when present, else
        # fall back to FILENAME. F1b: ingest NOW stamps collection_id on every chunk
        # (text AND table — worker/tasks.py:157-161, after build_langchain_documents), so
        # the collection_id-only floor binds table chunks too. (The BUG-F 2026-06-11 note
        # predates that stamping; collection_id is a valid table scope now.)
        table_docs = retrieval_manager.retrieve_table_chunks(
            query,
            doc_ids=doc_ids if doc_ids else None,
            collection_id=(collection_id if not doc_ids and not filenames else None),
            # Normalize an empty list to None here (consistent with the text path's
            # `doc_ids or None`): the floor above already refused an unscoped active vault,
            # so reaching here with no scope is a non-vault run, not an empty vault.
            filename_filters=(filenames or None),
            metadata_filter=metadata_filter,
            k=k,
        )
        docs.extend(table_docs or [])

    spans = [span_to_dict(d) for d in docs][: max(k, 1) if kind != "both" else max(k * 2, 1)]

    # ── S-E: stamp fidelity_warning on spans from low-fidelity docs ──────────────
    # A doc that ingested with fidelity != "good" already has that flag in the DB.
    # When fidelity_by_doc is provided, each span's doc_id is looked up: any grade
    # other than "good" (e.g. "partial") stamps fidelity_warning=True on that span
    # so the agent and the trust UI see "this doc ingested poorly — results may be
    # incomplete" rather than silently abstaining. Flag off: fidelity_by_doc=None →
    # no stamp, byte-identical to before S-E.
    if fidelity_by_doc:
        for sp in spans:
            did = sp.get("doc_id")
            if did and fidelity_by_doc.get(did, "good") != "good":
                sp["fidelity_warning"] = True

    # ── T1: grade the retrieval quality, optionally auto-expand on weak/empty ─────
    rq = _grade_spans(spans)
    repair: Optional[str] = None

    if _EXPAND_ON_WEAK and rq["grade"] in ("weak", "empty"):
        expanded_query = _expand_query(query)
        if expanded_query:
            # One expansion attempt: re-retrieve with the broader query and merge.
            expand_docs: List[Any] = []
            if kind in ("text", "both"):
                expand_docs.extend(retrieval_manager.retrieve(
                    expanded_query,
                    doc_ids=doc_ids or None,
                    filename_filters=(filenames if filenames and len(filenames) > 1 else None) if not doc_ids else None,
                    filename_filter=(filenames[0] if filenames and len(filenames) == 1 else None) if not doc_ids else None,
                    metadata_filter=metadata_filter,
                    collection_id=(collection_id if not doc_ids and not filenames else None),
                    top_k=k,
                    apply_threshold=False,
                    use_reranker=True,
                ) or [])
            if kind in ("table", "both"):
                expand_docs.extend(retrieval_manager.retrieve_table_chunks(
                    expanded_query,
                    doc_ids=doc_ids if doc_ids else None,
                    collection_id=(collection_id if not doc_ids and not filenames else None),
                    filename_filters=(filenames or None),
                    metadata_filter=metadata_filter,
                    k=k,
                ) or [])

            if expand_docs:
                # Merge: deduplicate by chunk_id (original spans take precedence).
                seen_ids = {s.get("chunk_id") for s in spans if s.get("chunk_id")}
                new_spans = [span_to_dict(d) for d in expand_docs
                             if span_to_dict(d).get("chunk_id") not in seen_ids]
                # S-E: stamp fidelity_warning on the newly-expanded spans too.
                if fidelity_by_doc:
                    for sp in new_spans:
                        did = sp.get("doc_id")
                        if did and fidelity_by_doc.get(did, "good") != "good":
                            sp["fidelity_warning"] = True
                merged = (spans + new_spans)[: max(k, 1) if kind != "both" else max(k * 2, 1)]
                rq_after = _grade_spans(merged)
                repair = (
                    f"query expanded to {expanded_query!r}; "
                    f"grade {rq['grade']}→{rq_after['grade']} "
                    f"({len(spans)}→{len(merged)} spans)"
                )
                spans = merged
                rq = rq_after
            else:
                repair = f"query expanded to {expanded_query!r} but yielded no additional chunks"

    rq_payload: Dict[str, Any] = {"grade": rq["grade"], "top_score": rq["top_score"]}
    if rq["grade"] != "strong":
        rq_payload["repair"] = repair or (
            f"retrieval grade={rq['grade']}; consider refining the query or reading the document directly"
        )

    summary = f"search_vault {query!r} ({kind}): {len(spans)} chunk(s) [{rq['grade']}]"
    if repair:
        summary += f"; {repair}"

    return ok_result(
        summary=summary,
        data={"chunks": spans, "retrieval_quality": rq_payload},
        provenance=spans,
    )
