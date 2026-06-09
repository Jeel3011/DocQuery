"""Reasoning verifier (anterior cingulate) — BRAIN_REASONING_PLAN §5.5 ⭐ the headline organ.

Verify the *reasoning*, not just the facts — the organ that converts a confident **WRONG
→ ABSTAIN**. It is DISTINCT from the atomic `verifier.py` (claim↔span entailment): that
one passes a wrong-pivot answer because every atomic claim is individually true (the §2
confident-wrong failure — "in 2021 AWS op income was $18.5B" is a true sentence; it's just
the wrong year). This organ checks the *bridge*: did the chosen pivot satisfy the
condition, did the op see all candidates, is the binding consistent, do the invariants
hold. When any check fails, the answer is withheld (binary abstain), never hedged.

It runs over the executive spine's deterministic output (`QueryPlan` + `ExecResult`), using
the kernel's selection trail (`Binding.op/threshold/candidates`) the executor now carries
forward — so the predicate/completeness checks are DETERMINISTIC, not an LLM judgement.
LLM-free; pure structure over what the kernel already resolved.

The contract (§5.5 / §4a): `monitor()` returns a `ReasoningVerdict`. `abstain=True` means
"withhold the figure and refuse cleanly". A degraded plan or a claims-only plan is NOT
abstained here — those are routed elsewhere; this organ only judges what the deterministic
spine actually computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ..executive.planner import QueryPlan, SubGoal
from ..executive.executor import ExecResult
from ..executive.workspace import Binding
from .invariants import InvariantCheck, figure_traces_to_cells


@dataclass
class ReasoningVerdict:
    """The self-monitor's binary decision plus its trail (Trust UI / audit). `abstain=True`
    → withhold the figure and refuse cleanly. `failures` lists the checks that fired."""
    abstain: bool
    reason: str = ""
    checks: List[str] = field(default_factory=list)      # human-readable per-check log
    failures: List[str] = field(default_factory=list)    # names of checks that failed

    @property
    def ok(self) -> bool:
        return not self.abstain


import re as _re
_PERIODISH = _re.compile(r"(?:19|20)\d{2}|FY\s?\d{2,4}|Q[1-4]", _re.I)


def _check_period_pivot(b: Binding) -> Optional[InvariantCheck]:
    """A threshold pivot (first_exceeds/last_below) resolves a PERIOD — so its value must
    be a real period token (a year / FY / quarter). A binding like 'col_1' means the kernel
    scanned a grid whose header was never recovered (extraction failure) and the 'pivot' is
    a positional placeholder, not a year. That can never be trusted → ABSTAIN. This is the
    deterministic backstop that converts an extraction-corrupted pivot into a clean refusal
    instead of a confident-wrong figure (§5.5 WRONG→ABSTAIN)."""
    if b.op not in ("first_exceeds", "last_below"):
        return None
    if not _PERIODISH.search(str(b.value)):
        return InvariantCheck("period_pivot", ok=False, decided=True,
                              detail=f"{b.op} resolved a NON-period pivot {b.value!r} — the grid's "
                                     f"columns were not real years (extraction failure); cannot trust it")
    return InvariantCheck("period_pivot", ok=True, decided=True,
                          detail=f"pivot {b.value!r} is a valid period")


def _check_predicate(b: Binding) -> Optional[InvariantCheck]:
    """Did the chosen pivot SATISFY the threshold predicate? For `first_exceeds` the
    winner must be > threshold AND the immediately-prior scanned period must be ≤ threshold
    (the 2021-below / 2022-above structure). Deterministic over the kernel's candidates
    trail. Returns None when the binding isn't a threshold selection (nothing to check)."""
    if b.op not in ("first_exceeds", "last_below") or b.threshold is None:
        return None
    cands = b.candidates or []
    if len(cands) < 1:
        return InvariantCheck("predicate", ok=False, decided=True,
                              detail=f"{b.op}: no candidates scanned — cannot confirm the pivot")
    thr = b.threshold
    # locate the winning candidate (its value is the binding's resolved cell value)
    win = next((c for c in cands if str(c.period) == str(b.value)), None)
    if win is None:
        return InvariantCheck("predicate", ok=False, decided=True,
                              detail=f"{b.op}: resolved pivot {b.value!r} not among scanned candidates")

    if b.op == "first_exceeds":
        if not (win.value > thr):
            return InvariantCheck("predicate", ok=False, decided=True,
                                  detail=f"first_exceeds: winner {win.value:g} is NOT > {thr:g}")
        # the period just before the winner (in scan order) must be ≤ threshold
        idx = cands.index(win)
        if idx > 0 and cands[idx - 1].value > thr:
            return InvariantCheck("predicate", ok=False, decided=True,
                                  detail=f"first_exceeds: a PRIOR period ({cands[idx-1].period}="
                                         f"{cands[idx-1].value:g}) already exceeded {thr:g} — "
                                         f"{b.value} is not the FIRST")
        return InvariantCheck("predicate", ok=True, decided=True,
                              detail=f"first_exceeds: {b.value}={win.value:g} > {thr:g}, prior ≤ {thr:g}")
    else:  # last_below
        if not (win.value < thr):
            return InvariantCheck("predicate", ok=False, decided=True,
                                  detail=f"last_below: winner {win.value:g} is NOT < {thr:g}")
        idx = cands.index(win)
        if idx < len(cands) - 1 and cands[idx + 1].value < thr:
            return InvariantCheck("predicate", ok=False, decided=True,
                                  detail=f"last_below: a LATER period ({cands[idx+1].period}) is also "
                                         f"< {thr:g} — {b.value} is not the LAST")
        return InvariantCheck("predicate", ok=True, decided=True,
                              detail=f"last_below: {b.value}={win.value:g} < {thr:g}, next ≥ {thr:g}")


def _check_completeness(b: Binding, goal: SubGoal) -> Optional[InvariantCheck]:
    """Did the op see ALL the candidates it was supposed to? For an entity argmax/argmin,
    the number of scanned candidates must match the number of entities the plan listed
    (argmax over 3 companies must have seen 3). A short scan = an incomplete comparison =
    a possibly-wrong extremum → abstain. None when there's no expected count to check."""
    if b.op not in ("argmax", "argmin"):
        return None
    sel = goal.selector or {}
    rows = sel.get("rows")
    if not isinstance(rows, list) or not rows:
        return None  # period-axis or unspecified — nothing to count against
    expected = len(rows)
    seen = len(b.candidates or [])
    if seen < expected:
        return InvariantCheck("completeness", ok=False, decided=True,
                              detail=f"{b.op} saw {seen}/{expected} candidates — "
                                     f"{expected - seen} entity(ies) didn't resolve; "
                                     f"the extremum may be wrong")
    return InvariantCheck("completeness", ok=True, decided=True,
                          detail=f"{b.op} saw all {seen}/{expected} candidates")


def _check_binding(plan: QueryPlan, res: ExecResult) -> Optional[InvariantCheck]:
    """Is every `$var` a downstream goal references actually BOUND (and non-abstained) in
    the workspace? A bridge that reads a follow-on value for a pivot that never resolved is
    the classic confident-wrong; the executor abstains it, but we re-assert it here as the
    §5.5 binding check. None when the plan has no cross-goal references."""
    refs: List[str] = []
    for g in plan.goals:
        for v in g.inputs.values():
            if isinstance(v, str) and v.startswith("$"):
                refs.append(v[1:])
    if not refs:
        return None
    for name in refs:
        bound = res.workspace.get(name)
        if bound is None or not bound.ok:
            return InvariantCheck("binding", ok=False, decided=True,
                                  detail=f"pivot ${name} is unbound/abstained — a downstream "
                                         f"read for it cannot be trusted")
    return InvariantCheck("binding", ok=True, decided=True,
                          detail=f"all pivot bindings resolved: {refs}")


def monitor(plan: QueryPlan, res: ExecResult, answer_text: str = "") -> ReasoningVerdict:
    """Run the §5.5 checks over the spine output → a binary abstain verdict.

    Checks (any HARD failure ⇒ abstain): predicate (threshold pivot satisfies its
    condition), completeness (extremum saw all candidates), binding (every $pivot is
    resolved), figure→cell (a stated number traces to a source cell). Undecidable checks
    (no structure to evaluate) are recorded but do NOT abstain — absence of evidence isn't
    a violation. A plan whose deterministic answer already abstained stays abstained (we
    don't resurrect it).
    """
    checks: List[str] = []
    failures: List[str] = []

    def fold(c: Optional[InvariantCheck]):
        if c is None:
            return
        status = "ok" if c.ok else ("undecided" if not c.decided else "FAIL")
        checks.append(f"{c.name}: {status} — {c.detail}")
        if c.decided and not c.ok:
            failures.append(c.name)

    # the spine already abstained (unresolved pivot, weak grounding) → propagate, don't
    # second-guess into a pass. The deterministic abstain is itself a correct §4a outcome.
    if not res.ok:
        return ReasoningVerdict(abstain=True,
                                reason=res.reason or "spine produced no resolved answer",
                                checks=["spine: abstained upstream"], failures=["spine"])

    by_id: Dict[str, SubGoal] = {g.id: g for g in plan.goals}
    # predicate + completeness over each selection binding the workspace holds
    for g in plan.goals:
        if g.type not in ("select", "compute"):
            continue
        b = res.workspace.get(g.binds or g.id)
        if b is None or not b.ok:
            continue
        fold(_check_period_pivot(b))
        fold(_check_predicate(b))
        fold(_check_completeness(b, by_id.get(g.id, g)))

    fold(_check_binding(plan, res))

    # figure→cell over the final answer prose, if one was rendered (the trace law)
    if answer_text:
        fold(figure_traces_to_cells(answer_text, res.cells))

    if failures:
        return ReasoningVerdict(
            abstain=True,
            reason="reasoning check(s) failed: " + "; ".join(failures)
                   + " — withholding the figure rather than state it (correct-or-abstain)",
            checks=checks, failures=failures,
        )
    return ReasoningVerdict(abstain=False, reason="all reasoning checks passed",
                            checks=checks, failures=[])
