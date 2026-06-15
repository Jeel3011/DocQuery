"""`survey_collection` tool — the broad whole-vault pass (GRAND_PLAN §G5, G5 plan §2.A).

Deep Analysis needs ONE tool the other modes don't: a breadth pass that, in a single
call, sweeps EVERY document in the vault and returns the lay of the land — ranked,
cited evidence clusters grouped by document/sub-topic — so the agent can then drill
with `read_document` / `table_lookup` / `compute` for the exact cells it will cite.

**This is the demoted `brain/map_reduce.py`.** We reuse the Brain's proven MAP step
(`Brain._map_all_docs` → per-doc cited-claim extraction, run in bounded parallel) but
STOP BEFORE its free-text REDUCE: the agent does the reasoning, the output gate binds
it. map_reduce stops being a parallel ANSWER path and becomes a retrieval-breadth TOOL.
There is NO new model path and NO fork of the loop — survey_collection orchestrates the
existing map machinery and shapes its output into the uniform tool envelope.

Contract (like every agent-core tool, §3.2/§3.3):
- `@safe_tool` — NEVER raises; any failure becomes an `{ok:false, error}` result.
- Standard envelope; `provenance` = the clusters' evidence spans, so the survey's
  citations flow through the ledger and the citation gate identically to `search_vault`.
- Returns EVIDENCE (clusters of claims with verbatim spans + source addressing), never a
  written conclusion.

Registered in `_MODE_TOOLS["deep"]` ONLY — it is the breadth tool for long deep runs,
not a standard-mode tool.

Gate: `eval/test_tools.py` (envelope + never-raise with a stub retrieval manager;
returns clusters carrying provenance). The live whole-vault sweep is exercised only in
the one end-to-end deep run (Jeel's go).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "survey_collection",
    "description": (
        "Sweep EVERY document in the vault at once and return the lay of the land: "
        "ranked evidence clusters (passages grouped by document/sub-topic) with their "
        "source addressing and a quoted span — NOT a written answer. Call this ONCE "
        "early in a deep analysis to discover what each document contains across the "
        "whole vault, then drill into the specific pages/cells you will cite with "
        "read_document, table_lookup, and compute. This is breadth; the other tools are "
        "depth."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The analysis topic to survey the vault for (e.g. 'key commercial "
                    "terms and risks', 'revenue and margin trends across filings')."
                ),
            },
            "k_docs": {
                "type": "integer",
                "description": (
                    "Max documents to survey (default: all in scope). Lower it only to "
                    "focus a very large vault."
                ),
            },
            "per_doc_k": {
                "type": "integer",
                "description": "Chunks to retrieve per document before extraction (default 8).",
            },
        },
        "required": ["query"],
    },
}


def _span_from_evidence(ev: Any, filename_by_doc: Dict[str, str]) -> Dict[str, Any]:
    """Serialize a Brain `EvidenceSpan` into the uniform §3.3 span-provenance dict.

    Mirrors `_envelope.span_to_dict`'s shape (kind/doc/page/chunk_id/snippet) so a
    survey citation is INDISTINGUISHABLE to the ledger and the gate from a `search_vault`
    span — same downstream binding, no special-casing. `doc` resolves to the filename
    (what the gate's `[doc p.N]` markers reference) when we can map the doc_id."""
    doc_id = getattr(ev, "doc_id", None)
    return {
        "kind": "span",
        "doc": filename_by_doc.get(doc_id, doc_id),
        "doc_id": doc_id,
        "page": None,  # MAP works over chunk text; page lives on the chunk, surfaced on read_document
        "chunk_id": getattr(ev, "chunk_id", None),
        "snippet": (getattr(ev, "verbatim_span", "") or "")[:500],
    }


def _retrieve_per_doc(
    query: str,
    retrieval_manager: Any,
    filenames: List[str],
    per_doc_k: int,
) -> Dict[str, tuple]:
    """Broad per-document retrieval → `{doc_id: (filename, [chunks])}` for the MAP step.

    Mirrors the breadth retrieval the Brain path uses (chat.py): per-file retrieve with
    `apply_threshold=False` (MAP+the gate handle precision; the 0.30 similarity floor
    drops valid cross-doc chunks) and `use_reranker=False` (the local CrossEncoder
    timed out under concurrent load and silently dropped whole docs). A per-file failure
    is non-fatal — that doc simply contributes nothing, like a Brain MAP miss."""
    doc_chunks: Dict[str, tuple] = {}
    for fname in filenames:
        try:
            chunks = retrieval_manager.retrieve(
                query,
                fname,        # filename_filter = single file
                None,         # page_filter
                None,         # filename_filters
                per_doc_k,    # top_k
                False,        # apply_threshold=False
                False,        # use_reranker=False
            )
        except Exception:  # noqa: BLE001 — a dead doc is non-fatal (quorum logic)
            chunks = []
        if chunks:
            doc_id = chunks[0].metadata.get("doc_id", fname)
            doc_chunks[doc_id] = (fname, chunks)
    return doc_chunks


@safe_tool
def survey_collection(
    query: str,
    retrieval_manager: Any,
    config: Any,
    *,
    filenames: Optional[List[str]] = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    k_docs: Optional[int] = None,
    per_doc_k: int = 8,
) -> Dict[str, Any]:
    """Survey the whole vault; return §3.3 envelope with cited evidence CLUSTERS.

    `retrieval_manager` is the live `RetrievalManager`; `config` is the user Config the
    demoted Brain MAP step runs on. `filenames` is the vault's routed doc set (one
    cluster per relevant doc). Returns evidence, never a written answer; never raises.
    """
    if not query:
        return error_result("survey_collection requires a non-empty 'query'")
    if retrieval_manager is None:
        return error_result("survey_collection requires a retrieval_manager")
    if config is None:
        return error_result("survey_collection requires a config (for the broad MAP pass)")

    filenames = list(filenames or [])
    filename_by_doc = filename_by_doc or {}
    if not filenames:
        # Fall back to the filenames implied by the doc map (vault scope), else nothing.
        filenames = list(filename_by_doc.values())
    if not filenames:
        return error_result(
            "survey_collection has no documents in scope to survey",
            summary="survey_collection: empty vault scope",
        )
    if k_docs and k_docs > 0:
        filenames = filenames[:k_docs]

    per_doc_k = max(1, int(per_doc_k or 8))

    # ── Broad pass: per-doc retrieval → the demoted Brain MAP step (NO reduce) ───────
    doc_chunks = _retrieve_per_doc(query, retrieval_manager, filenames, per_doc_k)
    if not doc_chunks:
        # No relevant content anywhere — a clean, honest survey result (not an error).
        return ok_result(
            summary=f"survey_collection {query!r}: no relevant passages across {len(filenames)} doc(s)",
            data={"clusters": [], "docs_surveyed": len(filenames), "docs_with_evidence": 0},
            provenance=[],
        )

    from src.components.brain.map_reduce import Brain

    brain = Brain(config)
    extracts = brain._map_all_docs(query, doc_chunks)  # the demoted map step — evidence only

    # ── Shape per-doc extracts into ranked, cited clusters (one per doc) ─────────────
    # A "cluster" is the natural survey unit: a document's relevant claims, each carrying
    # its verbatim span. We rank clusters by how much grounded evidence they hold so the
    # agent reads the densest documents first. We STOP HERE — no synthesis, no answer.
    clusters: List[Dict[str, Any]] = []
    all_provenance: List[Dict[str, Any]] = []
    for ext in extracts:
        if getattr(ext, "error", None) or getattr(ext, "nothing_relevant", False):
            continue
        claims = getattr(ext, "claims", []) or []
        if not claims:
            continue
        cluster_claims: List[Dict[str, Any]] = []
        cluster_spans: List[Dict[str, Any]] = []
        for cl in claims:
            ev_list = getattr(cl, "evidence", []) or []
            spans = [_span_from_evidence(ev, filename_by_doc) for ev in ev_list]
            cluster_spans.extend(spans)
            cluster_claims.append({
                "claim": getattr(cl, "text", ""),
                "verbatim_span": (ev_list[0].verbatim_span if ev_list else ""),
                "confidence": getattr(cl, "confidence", 0.0),
                "doc": getattr(ext, "filename", ""),
                "doc_id": getattr(ext, "doc_id", ""),
            })
        if not cluster_claims:
            continue
        all_provenance.extend(cluster_spans)
        clusters.append({
            "doc": getattr(ext, "filename", ""),
            "doc_id": getattr(ext, "doc_id", ""),
            "claims": cluster_claims,
            "evidence_count": len(cluster_claims),
        })

    # Densest-evidence-first: the agent drills the richest documents before the thin ones.
    clusters.sort(key=lambda c: c["evidence_count"], reverse=True)

    docs_with_evidence = len(clusters)
    n_claims = sum(c["evidence_count"] for c in clusters)
    return ok_result(
        summary=(
            f"survey_collection {query!r}: {n_claims} cited point(s) across "
            f"{docs_with_evidence}/{len(filenames)} doc(s)"
        ),
        data={
            "clusters": clusters,
            "docs_surveyed": len(filenames),
            "docs_with_evidence": docs_with_evidence,
        },
        provenance=all_provenance,
    )
