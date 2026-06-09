"""Executor (prefrontal cortex, part 2) — BRAIN_REASONING_PLAN §5.3.

Walk the plan DAG, BIND each pivot, ROUTE each sub-goal to the right organ, and write a
provenance-carrying `Binding` to the workspace — turning the planner's static DAG into a
sequenced computation. This is the half that makes the brain actually *sequence and
compose* instead of collapsing a bridge into one free-text LLM pass (the §2 failure).

Routing (§5.3):
  • select / compute → the KERNEL (`analyst.compute`), which resolves the cell + computes
    deterministically and returns the resolved `binding` (the pivot) + `candidates` trail.
  • lookup           → GROUNDING (`perception.ground_metric`), which resolves the right
    cell or abstains (the §5.1 bottleneck organ). Numbers are READ, never generated.
  • exists / qualitative → deferred to the CLAIMS+VERIFIER path (these need an LLM and
    narrative evidence; the executor marks them `needs_claims` so the orchestrator routes
    them, keeping this module LLM-free and deterministic).

Invariants (§5.3): every binding carries provenance; an unmet dependency or a failed
resolve ABSTAINS that goal (and any goal depending on it), never guesses; a degraded plan
runs its single lookup down the same path. Correct-or-abstain end to end (§4a).

This module is deterministic and LLM-free: it consumes a `QueryPlan` (from the planner)
and a `List[Grid]` (the directed-retrieval arm hands it the complete series/entity set the
plan needs) and produces an `ExecResult`. The LLM-bound organs (comprehension, claims) sit
OUTSIDE it, exactly as §5.4's spec-not-code line keeps arithmetic out of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analyst import Grid, CellRef, compute, ComputeResult
from ..perception.grounding import MetricIntent, ground_metric
from .planner import QueryPlan, SubGoal
from .workspace import Workspace, Binding


@dataclass
class ExecResult:
    """The executor's output for one plan. `answer_binding` is the final binding (the
    goal the plan named `final`); `ok` is true only when it resolved without abstaining.
    `needs_claims` carries any exists/qualitative goals the orchestrator must run through
    the claims+verifier path (the executor itself stays deterministic)."""
    ok: bool
    answer_binding: Optional[Binding]
    workspace: Workspace
    needs_claims: List[SubGoal] = field(default_factory=list)
    trail: List[str] = field(default_factory=list)   # per-goal hop log (Trust UI)
    reason: str = ""                                  # why it abstained (when not ok)

    @property
    def cells(self) -> List[CellRef]:
        return self.workspace.all_cells()


def _toposort(goals: List[SubGoal]) -> List[SubGoal]:
    """Order goals so every goal comes after its `depends_on`. The plans are tiny DAGs
    (1-2 goals) so a simple Kahn pass suffices; a cycle (shouldn't happen) degrades to the
    input order rather than raising — the executor never crashes the spine."""
    by_id = {g.id: g for g in goals}
    done: List[SubGoal] = []
    placed: set = set()
    # iterate until fixpoint; bounded by len(goals) passes
    for _ in range(len(goals) + 1):
        progressed = False
        for g in goals:
            if g.id in placed:
                continue
            if all((d in placed) or (d not in by_id) for d in g.depends_on):
                done.append(g)
                placed.add(g.id)
                progressed = True
        if len(placed) == len(goals):
            break
        if not progressed:           # unsatisfiable dep / cycle → append the rest as-is
            for g in goals:
                if g.id not in placed:
                    done.append(g)
                    placed.add(g.id)
            break
    return done


def _intent_from(goal: SubGoal, ws: Workspace) -> Optional[MetricIntent]:
    """Build a grounding MetricIntent for a `lookup` goal, substituting `$var` inputs from
    the workspace. Returns None when a required input (the period) is an unmet `$var` — the
    caller abstains that goal rather than reading a cell for an unknown period."""
    sel = goal.selector or {}
    row = sel.get("row", {})
    metric = row.get("label", "")
    section = row.get("section", "")

    # period may be a literal (from the IR) or a $var bound by an upstream pivot goal
    raw_period = goal.inputs.get("period")
    period = ws.resolve_ref(raw_period) if raw_period is not None else None
    if raw_period is not None and period is None:
        return None                  # the pivot didn't resolve → don't guess a period

    # an entity pivot ($pivot_entity) can override the section (which company won)
    raw_section = goal.inputs.get("section")
    if raw_section is not None:
        bound_section = ws.resolve_ref(raw_section)
        if bound_section is None:
            return None
        section = str(bound_section)

    if not metric or period is None:
        return None
    # the planner may pin an aggregation level (a "total"/"consolidated" follow-on must
    # bind the parent total, not a same-named segment component — the §2.1 fix); "any" else.
    agg = (goal.selector or {}).get("aggregation_level", "any")
    if agg not in ("total", "component", "any"):
        agg = "any"
    return MetricIntent(metric=metric, period=str(period), section=section,
                        aggregation_level=agg)


def _exec_select(goal: SubGoal, ws: Workspace, grids: List[Grid]) -> Binding:
    """Route a select/compute goal to the KERNEL. The kernel resolves the cell + computes,
    returning `binding` (the resolved pivot) for selection ops or `value` for arithmetic.
    A kernel error or empty result → an abstained Binding (never a guess)."""
    spec = dict(goal.selector or {})

    # substitute any $var in the spec's period (an entity-scan at a bound pivot year)
    if isinstance(spec.get("period"), str) and spec["period"].startswith("$"):
        bound = ws.resolve_ref(spec["period"])
        if bound is None:
            return Binding(goal.binds or goal.id, None, abstained=True,
                           reason=f"{goal.id}: pivot period {spec['period']} unresolved")
        spec["period"] = str(bound)

    res: ComputeResult = compute(spec, grids)
    if not res.ok and res.binding is None:
        return Binding(goal.binds or goal.id, None, abstained=True,
                       reason=f"{goal.id}: kernel {spec.get('op','?')} → {res.error or 'no result'}")

    # selection op → the pivot is the binding; arithmetic op → the number is the value
    value = res.binding if res.binding is not None else res.value
    thr = spec.get("threshold")
    return Binding(
        name=goal.binds or goal.id, value=value,
        cells=list(res.cells), confidence=1.0,           # deterministic kernel result
        derivation=res.formula or res.display(),
        op=res.op, threshold=(float(thr) if thr is not None else None),
        candidates=list(res.candidates),                 # completeness trail for §5.5
    )


def _exec_lookup(goal: SubGoal, ws: Workspace, grids: List[Grid]) -> Binding:
    """Route a lookup goal to GROUNDING. Resolves the right cell (honoring aggregation
    level, abstaining on weak/ambiguous matches) or abstains. The number is READ from the
    grid — the executor never generates it (§5.4)."""
    intent = _intent_from(goal, ws)
    if intent is None:
        return Binding(goal.binds or goal.id, None, abstained=True,
                       reason=f"{goal.id}: lookup has an unmet dependency or no period")

    g = ground_metric(intent, grids)
    if not g.ok:
        return Binding(goal.binds or goal.id, None, abstained=True,
                       reason=f"{goal.id}: grounding abstained — {g.reason}")
    return Binding(
        name=goal.binds or goal.id, value=g.best.value,
        cells=[g.best], confidence=round(g.confidence, 4),
        derivation=f"{intent.metric}[{intent.period}]"
                   f"{(' @'+intent.section) if intent.section else ''} = {g.best.raw} ({g.reason})",
    )


def execute(plan: QueryPlan, grids: List[Grid]) -> ExecResult:
    """Walk the plan DAG over `grids`, binding + routing each goal (§5.3). Deterministic,
    LLM-free. exists/qualitative goals are collected into `needs_claims` for the
    orchestrator's claims+verifier path; everything else resolves here or abstains.

    The final answer is the binding the plan named `final`. If that binding abstained (or
    any goal it transitively depends on did), the whole result abstains with a reason —
    correct-or-abstain, no confident wrong (§4a).
    """
    ws = Workspace()
    needs_claims: List[SubGoal] = []
    trail: List[str] = []

    for goal in _toposort(plan.goals):
        if goal.type in ("exists", "qualitative"):
            # narrative/yes-no — needs an LLM + evidence spans; defer to claims+verifier.
            needs_claims.append(goal)
            trail.append(f"{goal.id}[{goal.type}] → claims+verifier (deferred)")
            continue

        if goal.type in ("select", "compute"):
            b = _exec_select(goal, ws, grids)
        elif goal.type == "lookup":
            b = _exec_lookup(goal, ws, grids)
        elif goal.type == "compare":
            # compare resolves each operand via the kernel's entity series, then the
            # comparison itself is a claims/render step; surface the series + defer.
            b = _exec_select(goal, ws, grids)
            needs_claims.append(goal)
        else:
            b = Binding(goal.binds or goal.id, None, abstained=True,
                        reason=f"{goal.id}: unroutable goal type {goal.type!r}")

        # index the binding by its variable NAME (so $pivot_year resolves) AND by the
        # GOAL ID (so plan.final, which references a goal id, resolves) — the two key
        # spaces the planner uses interchangeably.
        ws.write(b)
        if goal.id != b.name:
            ws.write_alias(goal.id, b)
        mark = "✓" if b.ok else "✗"
        trail.append(f"{goal.id}[{goal.type}] {mark} "
                     f"{(b.name+'='+str(b.value)) if b.ok else b.reason}")

    final = ws.get(plan.final)

    # a plan whose final goal is itself a claims goal (pure exists/qualitative) is not an
    # abstain — it's "the deterministic spine has nothing to compute, route to claims".
    if final is None:
        deferred_final = any(g.id == plan.final for g in needs_claims)
        return ExecResult(
            ok=False, answer_binding=None, workspace=ws,
            needs_claims=needs_claims, trail=trail,
            reason=("routed to claims+verifier" if deferred_final
                    else f"final goal {plan.final!r} produced no binding"),
        )

    return ExecResult(
        ok=final.ok, answer_binding=final, workspace=ws,
        needs_claims=needs_claims, trail=trail,
        reason="" if final.ok else final.reason,
    )
