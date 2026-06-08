"""Grounding (sensory cortex) — BRAIN_REASONING_PLAN §5.1 ⭐ THE BOTTLENECK.

Turn an *intent* ("the total revenue line for FY2022") into **the right cell**, with
**confidence** and the **alternatives considered**, or a clean **abstain** — replacing
the loose one-shot LLM row pick that causes the §2.1 failure (asking for a *total* and
silently getting a *component* line like "Gross margin", computed perfectly on the wrong
cell). This is the single highest-leverage organ.

Design discipline (the bright lines, §8):
  • The DETERMINISTIC resolver decides; the LLM only proposes the *intent* (which
    metric/entity/period/level), never which cell holds the number.
  • Aggregation-level is HONORED: a request for a TOTAL never resolves to a COMPONENT.
    Enforced by the value-based `looks_like_total` structural check (kernel §5.4),
    NOT a hardcoded label list.
  • Ambiguity ABSTAINS: close candidates, a failed level-check, or cross-grid value
    disagreement → abstain with a reason. Correct-or-abstain (§4a). No hardcoded
    labels/values.

It resolves against STRUCTURE (section tree, period axis, totals-vs-children), so the
same organ grounds a balance sheet and (later) a contract clause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from ..analyst import Grid, CellRef, CellError, parse_cell, looks_like_total, _norm_section_match


# ── Intent: what the caller wants grounded (the LLM proposes this, as DATA) ─────

AggLevel = Literal["total", "component", "any"]


@dataclass
class MetricIntent:
    """A request to ground one metric to one cell. Proposed by the planner/LLM; the
    resolver decides the cell. `aggregation_level`:
      "total"     — must resolve to a total/parent row (e.g. "total revenue").
      "component" — a specific child line is wanted (e.g. "cost of revenue").
      "any"       — caller has no level preference (default).
    """
    metric: str                       # the row label to resolve, e.g. "total revenue"
    period: str                       # the period column, e.g. "2022"
    section: str = ""                 # optional section/entity scope, e.g. "AWS"
    aggregation_level: AggLevel = "any"


# ── Outputs ────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    cell: CellRef
    score: float                      # structural+lexical match strength [0,1]
    why: str                          # human-readable reason (Trust UI + audit)


@dataclass
class Grounding:
    best: Optional[CellRef]
    candidates: List[Candidate] = field(default_factory=list)
    confidence: float = 0.0           # calibrated; low → caller abstains
    abstained: bool = True
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.best is not None and not self.abstained


# ── Lexical match strength (structure first; this only RANKS, never decides) ────

def _norm(s: str) -> str:
    s = re.sub(r"[\[\]\"'`()]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _label_score(query_label: str, row_label: str) -> float:
    """How well `row_label` matches the requested `query_label`. Exact=1.0,
    full-containment=0.8, token-overlap proportional. Deterministic, no embeddings.
    """
    q, r = _norm(query_label), _norm(row_label)
    if not q or not r:
        return 0.0
    if q == r:
        return 1.0
    if q in r or r in q:
        return 0.8
    qt, rt = set(q.split()), set(r.split())
    if not qt:
        return 0.0
    return 0.6 * len(qt & rt) / len(qt)


# the "this row is a total" lexical hint is ADVISORY only — the value-based
# looks_like_total() is authoritative. We use it solely to break ties when the
# structural check is undecidable (None), never to override a False.
_TOTAL_HINT = re.compile(r"\b(total|consolidated|net (sales|revenue|income)|grand total)\b")


# ── The resolver ───────────────────────────────────────────────────────────────

# scoring/abstention thresholds (calibrated; tracked by the gate, not magic numbers
# tuned to one question — they encode "be conservative when the structure is unsure")
_MIN_SCORE = 0.55          # below this, no candidate is a confident match → abstain
_CLOSE_MARGIN = 0.08       # top-2 within this AND different values → ambiguous → abstain


def _collect(intent: MetricIntent, grids: List[Grid]) -> List[Candidate]:
    """Every plausible (cell, score, why) across grids for this intent.

    Structural resolve: honor the section (find_row already makes section binding),
    require the period column to exist and parse, then score the label match. The
    aggregation-level check is applied here as a HARD filter for level="total".
    """
    out: List[Candidate] = []
    for g in grids:
        row = g.find_row(intent.metric, intent.section)
        if row is None:
            continue
        if intent.section and not _norm_section_match(row, intent.section):
            continue
        if intent.period not in g.value_columns():
            continue
        try:
            cell = g.cell(intent.metric, intent.period, intent.section)
        except CellError:
            continue

        score = _label_score(intent.metric, row.get("label", ""))
        why = f"label≈{score:.2f}"

        # Aggregation-level: a TOTAL request must resolve to a total/parent row.
        if intent.aggregation_level == "total":
            is_total = looks_like_total(g, row, intent.period)
            if is_total is False:
                # structurally a component → this candidate is DISQUALIFIED. This is
                # the §2.1 structural fix: "total revenue" can never bind a child line.
                continue
            if is_total is True:
                score = min(1.0, score + 0.15)
                why += " · total✓"
            else:  # None (undecidable) → fall back to the advisory lexical hint
                if _TOTAL_HINT.search(_norm(row.get("label", ""))):
                    why += " · total?(label)"
                else:
                    score *= 0.85       # unproven total → less confident, not excluded
                    why += " · total?(unproven)"
        elif intent.aggregation_level == "component":
            if looks_like_total(g, row, intent.period) is True:
                score *= 0.85           # asked for a component, got a total → demote
                why += " · is-total"

        out.append(Candidate(cell=cell, score=round(score, 4), why=why))
    return out


def ground_metric(intent: MetricIntent, grids: List[Grid]) -> Grounding:
    """Resolve an intent to the right cell, or abstain. Deterministic; §5.1.

    Resolution order:
      1) structural resolve + aggregation-level filter (in `_collect`)
      2) rank candidates; the best wins ONLY if it clears _MIN_SCORE and is not in a
         near-tie with a DIFFERENT value (cross-grid disagreement / genuine ambiguity)
      3) otherwise ABSTAIN with a reason (correct-or-abstain, §4a)
    """
    cands = _collect(intent, grids)
    if not cands:
        lvl = "" if intent.aggregation_level == "any" else f" at level={intent.aggregation_level}"
        return Grounding(
            best=None, candidates=[], confidence=0.0, abstained=True,
            reason=f"no row matched {intent.metric!r}"
                   f"{(' in section ' + intent.section) if intent.section else ''}"
                   f"[{intent.period}]{lvl}",
        )

    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[0]

    # below the confidence floor → abstain (we don't ship a weak structural guess)
    if top.score < _MIN_SCORE:
        return Grounding(
            best=None, candidates=cands, confidence=top.score, abstained=True,
            reason=f"best match for {intent.metric!r} too weak (score {top.score:.2f} "
                   f"< {_MIN_SCORE}); abstaining rather than guess a row",
        )

    # near-tie with a DIFFERENT value → genuine ambiguity → abstain (§4a). Same value
    # across grids is fine (it's the same fact restated); only disagreement abstains.
    if len(cands) > 1:
        second = cands[1]
        same_value = abs(top.cell.value - second.cell.value) <= 1e-6 * max(abs(top.cell.value), 1.0)
        if (top.score - second.score) <= _CLOSE_MARGIN and not same_value:
            return Grounding(
                best=None, candidates=cands, confidence=top.score, abstained=True,
                reason=f"ambiguous: {intent.metric!r}[{intent.period}] resolves to "
                       f"{top.cell.value:g} vs {second.cell.value:g} with near-equal "
                       f"scores ({top.score:.2f}/{second.score:.2f}) — abstaining",
            )

    return Grounding(
        best=top.cell, candidates=cands, confidence=top.score,
        abstained=False, reason=top.why,
    )
