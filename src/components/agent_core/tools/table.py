"""`table_lookup` tool — thin adapter over grounding (AGENT_CORE_PLAN §3.3).

Wraps `perception.grounding.ground_metric(intent, grids)` — the correct-or-abstain
metric resolver (the 0-confidently-wrong gate). On a confident resolution it returns
the value + the fully-addressed source cell. On low confidence / ambiguity it does
NOT guess: it returns an ABSTAIN envelope carrying the abstain reason AND the
candidate cells, so the model can disambiguate (pick a section, a period) or surface
the abstention to the user — exactly the §3b directive ("abstain reasons + candidates
returned TO the model").

Gate: `test_grounding.py` (the resolver itself); `eval/test_tools.py` (this adapter's
envelope + abstain semantics + never-raise, incl. the MSFT total→component trap).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import abstain_result, cellref_to_dict, error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "table_lookup",
    "description": (
        "Resolve ONE metric to ONE source cell over the loaded table grids "
        "(e.g. 'total revenue' for 2022 in the 'AWS' section). Returns the value and "
        "the exact cell it came from, with confidence. If the metric is ambiguous or "
        "no row matches confidently, it ABSTAINS and returns the candidate cells it "
        "considered — use those to disambiguate (specify a section or period) or tell "
        "the user it can't be verified. NEVER returns a wrong cell as if it were right."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "The row/line-item label to resolve."},
            "period": {"type": "string", "description": "The period column, e.g. '2022'."},
            "entity_or_section": {
                "type": "string",
                "description": "Optional section/entity scope, e.g. 'AWS', 'Revenue'.",
            },
            "aggregation": {
                "type": "string",
                "enum": ["total", "component", "any"],
                "description": "Must resolve to a total/parent row, a child line, or no preference.",
            },
        },
        "required": ["metric", "period"],
    },
}

_AGG_MAP = {"total": "total", "component": "component", "any": "any"}


@safe_tool
def table_lookup(
    metric: str,
    period: str,
    grids: List[Any],
    *,
    entity_or_section: str = "",
    aggregation: str = "any",
) -> Dict[str, Any]:
    """Ground one metric to one cell over `grids`; return the §3.3 envelope.

    `grids` is the list of `analyst.Grid` for the current scope. Maps the args to a
    `MetricIntent`, calls `ground_metric`, and:
      - confident resolution → ok envelope with the cell as provenance;
      - abstain → abstain envelope carrying reason + candidate cells (NOT an error).
    """
    from src.components.brain.perception.grounding import MetricIntent, ground_metric

    if not metric or not period:
        return error_result("table_lookup requires both 'metric' and 'period'")

    intent = MetricIntent(
        metric=metric,
        period=period,
        section=entity_or_section or "",
        aggregation_level=_AGG_MAP.get(aggregation, "any"),
    )
    grounding = ground_metric(intent, grids or [])

    candidates = [
        {**cellref_to_dict(c.cell), "score": c.score, "why": c.why}
        for c in (grounding.candidates or [])
    ]

    if not grounding.ok:
        return abstain_result(
            grounding.reason or "no confident match",
            summary=f"table_lookup {metric!r}[{period}]: abstained ({grounding.reason or 'low confidence'})",
            data={
                "value": None,
                "confidence": grounding.confidence,
                "candidates": candidates,
            },
            provenance=[],
        )

    best = cellref_to_dict(grounding.best)
    return ok_result(
        summary=(
            f"table_lookup {metric!r}[{period}] = {grounding.best.raw} "
            f"({grounding.best.doc} p{grounding.best.page})"
        ),
        data={
            "value": grounding.best.value,
            "cell": best,
            "confidence": grounding.confidence,
            "candidates": candidates,
        },
        provenance=[best],
    )
