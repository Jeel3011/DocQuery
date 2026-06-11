"""`list_metrics` tool — the line-item INDEX of a document (AGENT_CORE_PLAN §precision).

The 2026-06-11 R&D-ratio failure showed the model GUESSING labels it couldn't know by
heart ("Technology and content" instead of Amazon's actual "Technology and
infrastructure"; the wrong section for a "Total"). Even a perfect kernel can't resolve a
reference the model never knew to write. This tool closes that gap: it returns the
COMPACT, clean list of {section, label, periods} line-items actually present in a
document's grids — so the model PICKS an exact reference instead of guessing, then feeds
it straight into `compute`/`table_lookup`.

Narrow + read-only + no new intelligence: it lists what extraction already stored,
filtered to real line-items (the `is_lineitem_label` prose guard drops narrative
fragments) and de-duplicated. NOT a retrieval tool (no ranking, no LLM) — a directory.

Gate: `eval/test_tools.py` (envelope + the AMZN/GOOG real-label assertions).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "list_metrics",
    "description": (
        "List the actual line-items (section + label + which periods) available in a "
        "document's tables, so you can pick the EXACT reference to pass to compute / "
        "table_lookup instead of guessing a label. Use this the moment a compute / "
        "table_lookup call fails to resolve a metric, or before computing over a "
        "document whose exact row labels you don't already know (different filings name "
        "the same metric differently — e.g. R&D may be 'Research and development' or "
        "'Technology and infrastructure'). Optionally filter by a substring (e.g. "
        "'revenue', 'research') to narrow the list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string",
                       "description": "The document to index (filename or a distinctive substring)."},
            "contains": {"type": "string",
                         "description": "Optional: only list labels containing this substring "
                                        "(case-insensitive), e.g. 'revenue' or 'research'."},
            "period": {"type": "string",
                       "description": "Optional: only list line-items that have a value for this "
                                      "period, e.g. '2023'."},
        },
        "required": ["doc_id"],
    },
}


@safe_tool
def list_metrics(
    doc_id: str,
    grids: List[Any],
    *,
    contains: Optional[str] = None,
    period: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the clean {section,label,periods} line-item index for `doc_id`'s grids.

    `grids` is the run's loaded `analyst.Grid` list (the registry injects it). We scope to
    the requested doc (kernel-style substring match), keep only real line-items, and
    de-duplicate by (section,label) so the model sees a short, pickable directory.
    """
    from src.components.brain.analyst import is_lineitem_label
    try:
        from src.components.brain.analyst import _doc_match
    except Exception:  # noqa: BLE001
        _doc_match = None

    if not grids:
        return error_result("list_metrics: no documents are loaded in this run's scope")
    if not doc_id:
        return error_result("list_metrics requires a 'doc_id'")

    def _match(gdoc: Any) -> bool:
        if _doc_match is not None:
            try:
                return _doc_match(gdoc, doc_id)
            except Exception:  # noqa: BLE001
                pass
        g = str(gdoc or "").lower(); w = doc_id.lower()
        return bool(w) and (g == w or w in g or g in w)

    scoped = [g for g in grids if _match(getattr(g, "doc", None))]
    if not scoped:
        have = sorted({str(getattr(g, "doc", "")) for g in grids if getattr(g, "doc", None)})
        return error_result(
            f"list_metrics: document {doc_id!r} not in scope (loaded: {have})")

    want = (contains or "").strip().lower()
    seen = set()
    items: List[Dict[str, Any]] = []
    for g in scoped:
        periods = list(getattr(g, "periods", []) or [])
        for r in getattr(g, "rows", []) or []:
            label = (r.get("label") or "").strip()
            section = (r.get("section") or "").strip()
            if not label or not is_lineitem_label(label):
                continue  # drop prose fragments / narrative rows
            if want and want not in label.lower() and want not in section.lower():
                continue
            # which of this grid's periods this row actually carries a value for
            has = [p for p in periods if str(r.get(p, "")).strip()]
            if period and period not in has:
                continue
            key = (section.lower(), label.lower())
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "section": section, "label": label,
                "periods": has or periods,
                "page": getattr(g, "page", None),
            })

    # Provenance: one span per distinct page we indexed (so the ledger records what we read).
    pages = sorted({it["page"] for it in items if it["page"] is not None})
    provenance = [{"kind": "span", "doc": getattr(scoped[0], "doc", doc_id), "page": p}
                  for p in pages]
    filt = f" containing {contains!r}" if contains else ""
    return ok_result(
        summary=f"list_metrics {doc_id}{filt}: {len(items)} line-item(s)",
        data={"doc_id": doc_id, "metrics": items},
        provenance=provenance,
    )
