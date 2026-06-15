"""Review grid (AGENT_CORE_PLAN Phase B2) — the law-first head-to-head feature.

A review grid is **N documents × M columns**, where each column is a legal/structured
fact to extract (e.g. "Governing law", "Termination notice period", "Indemnity cap",
"Change-of-control", "Auto-renewal"). Each (document, column) pair = ONE bounded agent
run over THAT document only, producing a single `GridCell`.

THE DISCIPLINE (the moat, applied to text):
  The numeric agent's rule is "every figure traces to a cell, or abstain." For clause
  columns the values are text, not numbers — so the rule becomes **"every cell quotes
  the exact source span and is clickable to it, or the cell ABSTAINS (status=missing /
  not-found)."** The agent NEVER invents a clause it cannot ground. That cite-or-abstain
  rule is what we publish against Harvey's advertised "96%": every cell is verified to a
  source span, or honestly flagged — there are no silent wrong cells.

This module defines the DATA CONTRACT only (schemas). The engine (per-cell run_agent
fan-out) and the route come next; they fill these structures. Pure dataclasses — no
intelligence, no I/O, never raises on construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Column definitions — what a grid extracts.
# ──────────────────────────────────────────────────────────────────────────────

class ColumnKind(str, Enum):
    """How a column's value is produced and therefore how its cell is VERIFIED.

    - CLAUSE: a legal term extracted as text, verified by a quoted source span
      (cite-or-abstain). The law-first default.
    - NUMERIC: a figure computed by the kernel, verified by a CellRef (the numeric
      moat). Not the B2 first target, but the schema admits it so the same grid can
      mix clause + numeric columns later (a contract grid that also pulls a $ cap).
    """
    CLAUSE = "clause"
    NUMERIC = "numeric"


@dataclass
class GridColumn:
    """One column = one fact to extract from every document in the grid.

    `key` is a stable id (snake_case); `label` is the human header; `prompt` is the
    extraction instruction handed to the per-cell agent ("Find the governing-law
    clause. Quote it exactly."). `risk_rubric`, when set, tells the agent how to
    classify the finding (e.g. "standard if New York/Delaware/England; else
    non-standard") — this drives the cell's `risk` flag.
    """
    key: str
    label: str
    prompt: str
    kind: ColumnKind = ColumnKind.CLAUSE
    risk_rubric: Optional[str] = None


@dataclass
class GridSpec:
    """The full request: which documents, which columns, what scope/budget.

    `doc_ids` are the rows; `columns` are the columns. `collection_id` scopes
    retrieval. `title` is for display/export. The engine fans out one bounded agent
    run per (doc_id × column).
    """
    title: str
    collection_id: str
    doc_ids: List[str]
    columns: List[GridColumn]

    @property
    def cell_count(self) -> int:
        return len(self.doc_ids) * len(self.columns)


# ──────────────────────────────────────────────────────────────────────────────
# Cell results — what the agent produces per (doc, column).
# ──────────────────────────────────────────────────────────────────────────────

class CellStatus(str, Enum):
    """The honest state of one cell. NEVER a silent wrong value.

    - FOUND:    a grounded value with a clickable source span (or CellRef for numeric).
    - MISSING:  the agent searched and the document does not contain this term
                (a legitimate finding — e.g. "no indemnity cap clause"). Abstain, flagged.
    - ABSTAIN:  the agent could not ground a confident answer (ambiguous / low evidence).
                Distinct from MISSING: "I couldn't determine," not "it isn't there."
    - ERROR:    the run failed (budget/exception). Surfaced, never hidden.
    """
    FOUND = "found"
    MISSING = "missing"
    ABSTAIN = "abstain"
    ERROR = "error"


class RiskFlag(str, Enum):
    """Per-cell risk classification from the column's rubric. NONE = no rubric / N/A."""
    STANDARD = "standard"
    NON_STANDARD = "non_standard"
    MISSING = "missing"          # the term is absent and its absence is itself a risk
    NONE = "none"


@dataclass
class GridCell:
    """One (document, column) result. The unit the UI renders and the export emits.

    The value is ONLY trustworthy when `status == FOUND`; for every other status the
    `value` is None and `status`/`note` explain why — that is the cite-or-abstain
    contract made structural. `provenance` carries the span/CellRef dicts (same shape
    the EvidenceLedger emits → the UI click-to-source reuses the existing `sources`
    rendering). `quote` is the exact text the agent grounded on (for a clause), shown
    on hover / in the cell detail.
    """
    doc_id: str
    column_key: str
    status: CellStatus
    value: Optional[str] = None            # the extracted clause text / formatted figure
    quote: Optional[str] = None            # exact grounding span text (clause columns)
    risk: RiskFlag = RiskFlag.NONE
    provenance: List[Dict[str, Any]] = field(default_factory=list)  # span/CellRef dicts
    note: Optional[str] = None             # why ABSTAIN/MISSING/ERROR; or a short rationale
    doc_name: Optional[str] = None         # filename, for display/export convenience
    abstain_reason: Optional[str] = None   # WHY this cell abstained, distinguishably:
    #   "unparsed"    → our bug: the agent answered (often grounded) but in a shape the
    #                   envelope parser couldn't read. NEVER let this look like no_evidence
    #                   (that conflation is the G4 loop — see G4_REVIEW_GRID_PLAN §0).
    #   "no_evidence" → genuine: the agent searched and could not ground an answer.
    #   "ambiguous"   → genuine: conflicting/unclear evidence the agent declined to pick.
    #   None for FOUND/MISSING/ERROR. Additive + optional — existing consumers ignore it.

    @property
    def is_verified(self) -> bool:
        """A cell is 'verified' iff it has a grounded value AND at least one provenance
        entry. This is the metric we publish: verified-or-abstained, never wrong-silent."""
        return self.status == CellStatus.FOUND and bool(self.provenance)


@dataclass
class GridResult:
    """The assembled grid + the headline metric. Returned by the engine, served by the
    route, rendered by the UI.

    `coverage` is the publishable number: of all cells, how many are verified vs
    abstained vs missing — the honest accuracy story (no silent wrong cells).
    """
    spec: GridSpec
    cells: List[GridCell]

    def coverage(self) -> Dict[str, int]:
        """Tally cells by status — the 'every cell verified or flagged' headline."""
        tally = {s.value: 0 for s in CellStatus}
        for c in self.cells:
            tally[c.status.value] = tally.get(c.status.value, 0) + 1
        tally["verified"] = sum(1 for c in self.cells if c.is_verified)
        tally["total"] = len(self.cells)
        return tally
