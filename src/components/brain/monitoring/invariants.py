"""Structural invariants — BRAIN_REASONING_PLAN §5.5.

Generic, domain-agnostic structural checks the self-monitoring organ uses as the
§4a-safe backstop: a row used as a TOTAL must ≈ Σ of its children; a figure stated in an
answer must TRACE to a real source cell (no free-floating number). These read STRUCTURE
and VALUES, never a label list — the same value-based discipline as `analyst.looks_like_total`
(which this module reuses), so the same invariant guards a balance sheet and (later) a
contract schedule.

Pure functions over grids/cells; no LLM, no retrieval. Each returns a small verdict the
reasoning verifier folds into its binary decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..analyst import Grid, CellRef, looks_like_total, parse_cell


@dataclass
class InvariantCheck:
    """One invariant's result. `ok=False` is a HARD signal to abstain (a violated
    structural law); `ok=True` with `decided=False` means 'undecidable, no evidence
    either way' — which must NOT be read as a pass (absence of evidence ≠ evidence)."""
    name: str
    ok: bool
    decided: bool          # was there enough structure to actually evaluate it?
    detail: str = ""


def total_consistency(grid: Grid, row: dict, period: str) -> InvariantCheck:
    """A row USED AS a total must ≈ Σ of its children (or dominate by magnitude). Delegates
    to the value-based `looks_like_total` (§5.4): True→consistent, False→VIOLATION (a
    component masquerading as a total), None→undecidable. This is the check that powers
    grounding's aggregation-level filter and backstops a 'total revenue' answer that
    silently bound a child line."""
    verdict = looks_like_total(grid, row, period)
    if verdict is True:
        return InvariantCheck("total≈Σchildren", ok=True, decided=True,
                              detail="row is a consistent total/parent")
    if verdict is False:
        return InvariantCheck("total≈Σchildren", ok=False, decided=True,
                              detail="row used as a total is a COMPONENT (smaller than a sibling)")
    return InvariantCheck("total≈Σchildren", ok=True, decided=False,
                          detail="undecidable (too few numeric siblings)")


# numbers in a stated answer: $574,785M / 574,785 / 22.8 / 88,523 etc.
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _canon(tok: str) -> Optional[float]:
    try:
        return abs(parse_cell(tok))
    except Exception:
        return None


def figure_traces_to_cells(answer: str, cells: List[CellRef],
                           rel_tol: float = 0.01) -> InvariantCheck:
    """Every SUBSTANTIVE figure stated in the answer must trace to a source cell (or be a
    simple round/scaled restatement of one). A number in the prose that matches NO cell is
    a free-floating figure → abstain (the §4a 'a stated figure must trace to a cell' law).

    Scaling-aware: a cell of 513,983 ($M) legitimately surfaces as "513,983", "$514B",
    "514", or "~$513.98 billion". We accept a stated number if it equals a cell value, the
    cell scaled by 1e-3/1e-6 (millions→billions), or vice-versa, within rel_tol. Bare
    small integers (years like 2022, list indices) are ignored — they're not figures.
    """
    cell_vals = [abs(c.value) for c in cells if c.value is not None]
    if not cell_vals:
        # nothing to trace against — undecidable, NOT a pass
        return InvariantCheck("figure→cell", ok=True, decided=False,
                              detail="no source cells to trace against")

    def matches_a_cell(x: float) -> bool:
        for v in cell_vals:
            for scale in (1.0, 1e-3, 1e3, 1e-6, 1e6):
                if v == 0:
                    continue
                if abs(x - v * scale) <= rel_tol * max(abs(v * scale), 1.0):
                    return True
        return False

    untraced: List[str] = []
    for tok in _NUM_RE.findall(answer or ""):
        x = _canon(tok)
        if x is None:
            continue
        # ignore bare year-like / tiny integers (not substantive figures)
        if "," not in tok and "." not in tok and "$" not in tok and x < 10000:
            continue
        if not matches_a_cell(x):
            untraced.append(tok)

    if untraced:
        return InvariantCheck("figure→cell", ok=False, decided=True,
                              detail=f"stated figure(s) trace to no source cell: {untraced[:3]}")
    return InvariantCheck("figure→cell", ok=True, decided=True,
                          detail="all stated figures trace to cells")
