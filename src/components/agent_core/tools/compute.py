"""`compute` tool — thin adapter over the deterministic kernel (AGENT_CORE_PLAN §3.3).

Wraps `analyst.compute(spec, grids)` — the 15-op whitelisted kernel with CellRef
provenance and selection trails. The agent model writes the spec (op + cell
references) directly; we do NOT call the LLM spec-writer (`write_specs`/`analyze`),
which retires in B5. There is NO new intelligence here: the kernel decides values
and placement, the kernel abstains on ambiguity, we only shape the envelope.

Gate: the existing kernel gates (`test_analyst_compute.py`, `test_selection.py`,
`test_descending_pivot.py`) cover `analyst.compute`; `eval/test_tools.py` covers the
envelope/abstain/never-raise contract of this adapter on real grids.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._envelope import abstain_result, cellref_to_dict, error_result, ok_result, safe_tool

# JSON-schema for the tool's args (consumed by the registry in A2; here so the tool
# is self-describing and `eval/test_tools.py` can validate calls against it).
SCHEMA: Dict[str, Any] = {
    "name": "compute",
    "description": (
        "Deterministically compute ONE numeric result over the loaded table grids "
        "using a whitelisted op and explicit cell references. Returns the value, the "
        "human-readable formula, and the exact source cells used (provenance). Use "
        "this for ANY figure that is a calculation (growth, margin, sum, ratio) or a "
        "selection over a series (argmax/argmin/first_exceeds/...). Never guesses: a "
        "bad/ambiguous spec returns an error or abstains, it does not invent a number. "
        "When SEVERAL documents are loaded, common metrics (e.g. 'total revenue') exist "
        "in more than one — pass `doc` (the document filename) to scope the references "
        "to one document, or the kernel will report the clash and abstain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "value", "delta", "growth_pct", "sum", "difference", "ratio",
                    "margin_pct", "average", "cagr_pct",
                    "argmax", "argmin", "first_exceeds", "last_below", "rank", "filter",
                ],
                "description": "The whitelisted operation to perform.",
            },
            "table": {
                "type": "integer",
                "description": "Optional grid index to try first; the kernel falls back to all grids.",
            },
            "doc": {
                "type": "string",
                "description": (
                    "Optional document filename (e.g. 'msft-10k_20220630.htm.pdf'; a "
                    "distinctive substring is enough) scoping ALL cell references in this "
                    "spec to that document. REQUIRED in practice when several documents "
                    "are loaded and the metric exists in more than one (every issuer has "
                    "a 'total revenue'). A per-reference `doc` inside row/numerator/... "
                    "overrides it for cross-document comparisons."
                ),
            },
            "row": {
                "type": "object",
                "description": "Cell reference {section?, label, doc?} for single-cell ops.",
                "properties": {"section": {"type": "string"}, "label": {"type": "string"},
                               "doc": {"type": "string"}},
            },
            "from_period": {"type": "string"},
            "to_period": {"type": "string"},
            "period": {"type": "string"},
            "periods": {"type": "array", "items": {"type": "string"}},
            "cells": {"type": "array", "items": {"type": "object"}},
            "numerator": {"type": "object"},
            "denominator": {"type": "object"},
            "threshold": {"type": "number"},
            "over": {
                "type": "string",
                "enum": ["period", "entity"],
                "description": (
                    "Selection axis: 'period' scans ONE row across its years "
                    "(first_exceeds / peak year); 'entity' compares a LIST of rows at a "
                    "FIXED period (which company/segment was highest) — needs `rows` + `period`."
                ),
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "object",
                            "properties": {"section": {"type": "string"},
                                           "label": {"type": "string"},
                                           "doc": {"type": "string"}},
                        },
                    },
                },
                "description": "For over='entity': the row references to compare, one per entity.",
            },
        },
        "required": ["op"],
    },
}


@safe_tool
def compute(spec: Dict[str, Any], grids: List[Any]) -> Dict[str, Any]:
    """Run one kernel spec against `grids`; return the §3.3 envelope.

    `spec` is the same dict shape `analyst.compute` already accepts (op + references).
    `grids` is the list of `analyst.Grid` for the current scope (loaded by
    `read_document` / the loop's grid cache). The kernel never raises; we map its
    `ComputeResult` to ok / error envelopes, carrying every cell + candidate as
    provenance so the gates can trace the figure.
    """
    from src.components.brain.analyst import compute as kernel_compute

    if not isinstance(spec, dict) or not spec.get("op"):
        return error_result("spec must be a dict with an 'op'")

    result = kernel_compute(spec, grids or [])

    cells_prov = [cellref_to_dict(c) for c in (result.cells or [])]
    cand_prov = [cellref_to_dict(c) for c in (result.candidates or [])]

    if not result.ok:
        # A kernel error (bad spec, missing cell) is an error; an AmbiguousCell is
        # surfaced as the error string by the kernel. Either way we carry the
        # candidates the kernel scanned so the model can disambiguate.
        return error_result(
            result.error or "compute failed",
            summary=f"compute {result.op}: {result.error or 'failed'}",
        )

    # A selection op's threshold (e.g. first_exceeds > 500000) is a legitimate figure the
    # answer may restate, and it originated from this VERIFIED spec — so it belongs in
    # provenance as a `param` (distinct from a read cell). The output gate (verify_numbers)
    # traces a stated figure against cells OR params, so "$500B" no longer reads as
    # free-floating — while the model still can't launder an arbitrary number, because the
    # param is only ledgered when the kernel actually executed the op with it.
    extra_prov = []
    threshold = spec.get("threshold")
    if isinstance(threshold, (int, float)):
        extra_prov.append({"kind": "param", "label": "threshold",
                           "value": float(threshold), "op": result.op})

    data = {
        "value": result.value,
        "formula": result.formula,
        "unit": result.unit,
        "display": result.display(),
        "binding": result.binding,           # selection ops: the resolved period/entity
        "cells": cells_prov,
        "candidates": cand_prov,             # the completeness trail (argmax saw all N?)
    }
    return ok_result(
        summary=f"compute {result.op} = {result.display()}",
        data=data,
        provenance=cells_prov + extra_prov,
    )
