"""`read_document` tool — thin adapter over grid loading + chunk fetch (AGENT_CORE_PLAN §3.3).

Wraps `table_intent.load_grids_for_docs` (the structured-grid loader the spine already
uses) plus a page-scoped text-chunk fetch. Returns the page text and the grid JSONs
(headers/rows/periods) for the requested document so the model can read structure
before it claims anything. No new intelligence: it loads what ingest stored.

Two dependency shapes, both honored so the tool is testable offline and live:
  - LIVE: pass `db_client` (a `db.SupabaseClient`-like). Grids come from
    `load_grids_for_docs`; text chunks from a doc/page-scoped select.
  - OFFLINE / pre-loaded: pass `grids=[...]` (already-built `analyst.Grid`s, e.g. from
    `extract_tables_from_pdf` in a gate). The DB is never touched.

Gate: extraction benchmark (the grids themselves, exists, 100%); `eval/test_tools.py`
(this adapter's envelope + never-raise on real grids).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "read_document",
    "description": (
        "Read a document's structured table grids (and optionally its page text) so "
        "you can see the actual rows, sections, and periods before making any claim. "
        "Returns grid JSONs (headers, rows, periods) with source addressing. Use this "
        "before table_lookup/compute when you are unsure what rows or periods exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "The document id to read."},
            "page_range": {
                "type": "string",
                "description": "Optional 'start-end' page range to scope text, e.g. '40-45'.",
            },
            "table_grids": {
                "type": "boolean",
                "description": "Include structured table grids (default true).",
            },
        },
        "required": ["doc_id"],
    },
}


def _grid_to_json(g: Any) -> Dict[str, Any]:
    """Serialize an analyst.Grid into a compact, model-readable JSON view."""
    return {
        "doc": getattr(g, "doc", None),
        "page": getattr(g, "page", None),
        "table_id": getattr(g, "table_id", None),
        "summary": getattr(g, "summary", "") or "",
        "headers": getattr(g, "headers", []),
        "periods": getattr(g, "periods", []),
        "units": getattr(g, "units", None),
        "rows": getattr(g, "rows", []),
    }


def _parse_page_range(page_range: Optional[str]):
    if not page_range:
        return None
    try:
        if "-" in page_range:
            a, b = page_range.split("-", 1)
            return int(a), int(b)
        p = int(page_range)
        return p, p
    except Exception:  # noqa: BLE001
        return None


@safe_tool
def read_document(
    doc_id: str,
    *,
    db_client: Any = None,
    grids: Optional[List[Any]] = None,
    question: Optional[str] = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    page_range: Optional[str] = None,
    table_grids: bool = True,
) -> Dict[str, Any]:
    """Read grids (+ optional page text) for `doc_id`; return the §3.3 envelope.

    Provenance lists one span per grid (doc/page) so the ledger records what was read.
    """
    if not doc_id and not grids:
        return error_result("read_document requires a 'doc_id' (or pre-loaded grids)")

    loaded: List[Any] = []
    if grids is not None:
        # Offline / pre-loaded path: caller already built the grids. Honor the model's
        # doc_id when it matches the grids' doc labels (the model knows docs by the
        # filename it saw in search results); if nothing matches, fall back to all —
        # showing extra structure beats silently returning nothing.
        loaded = list(grids)
        if doc_id:
            scoped = [g for g in loaded if getattr(g, "doc", None) == doc_id]
            if scoped:
                loaded = scoped
    elif table_grids and db_client is not None:
        from src.components.brain.table_intent import load_grids_for_docs

        loaded = load_grids_for_docs(
            db_client,
            [doc_id],
            question=question,
            filename_by_doc=filename_by_doc,
        )

    grid_jsons = [_grid_to_json(g) for g in loaded] if table_grids else []

    # Page text: only available with a live db_client; scoped to the page range.
    page_text: List[Dict[str, Any]] = []
    pr = _parse_page_range(page_range)
    if db_client is not None and pr is not None:
        try:
            rows = (
                db_client.client.table("document_chunks")
                .select("content,metadata")
                .eq("document_id", doc_id)
                .execute()
                .data
                or []
            )
            lo, hi = pr
            for r in rows:
                md = r.get("metadata") or {}
                if md.get("chunk_type") == "table":
                    continue
                pg = md.get("page_number")
                if isinstance(pg, int) and lo <= pg <= hi:
                    page_text.append({"page": pg, "text": r.get("content", "")})
        except Exception:  # noqa: BLE001 — degrade to grids-only, never raise
            page_text = []

    provenance = [
        {"kind": "span", "doc": gj["doc"], "page": gj["page"], "table_id": gj["table_id"]}
        for gj in grid_jsons
    ]
    summary = f"read_document {doc_id}: {len(grid_jsons)} grid(s), {len(page_text)} text page(s)"
    return ok_result(
        summary=summary,
        data={"doc_id": doc_id, "grids": grid_jsons, "page_text": page_text},
        provenance=provenance,
    )
