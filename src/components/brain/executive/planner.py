"""Planner (prefrontal cortex, part 1) — BRAIN_REASONING_PLAN §5.3 / Stage C3.

`plan(ir: QueryIR) -> QueryPlan` — decompose a typed question (the §5.2 QueryIR) into a
**DAG of typed sub-goals** over the basis `lookup/select/compute/compare/exists/
qualitative`. This is the *decompose → bind → route* spine: the planner only DECOMPOSES
and names the variable bindings; the executor (C4) walks the DAG, binds the pivot, and
routes each sub-goal to grounding/kernel/claims. C3 is gated INDEPENDENTLY of execution
(test_planner.py) — a plan is judged on its structure (right types, right deps, a pivot
binding where one is needed), not on the number it would eventually produce.

Why deterministic (no LLM here). The hard understanding already happened upstream in
comprehension (§5.2): the QueryIR carries question_type + metrics + entities + periods +
constraints. Turning that typed IR into a sub-goal DAG is a STRUCTURAL transform over a
small basis — the most verifiable, lowest-latency arm, exactly the §5.3 framing ("the
planner is a structural task, far easier + more verifiable than end-to-end reasoning").
Anything we can't map structurally degrades to a single `lookup` goal, so the fast path
is never worse than today (the §5.2/§5.3 degrade invariant).

The selector dict a `select`/`compute` sub-goal carries is the SAME shape the kernel's
`analyst.compute()` consumes (`{op, over, row|rows, period, threshold, cmp, ...}`), so C4
hands it straight to the kernel with no translation. The bridge tail emits
`lookup(inputs={"period": "$pivot_year"}, depends_on=[pivot_goal])` — the hippocampal
variable binding (§5.3) that today's single-pass path collapses and gets wrong (§2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ..comprehension import QueryIR


# The sub-goal basis (§1, §5.3). A small set that RECOMBINES into thousands of question
# shapes — NOT a handler per question. Each routes (in C4) to one organ:
#   lookup      — read one stated value          → grounding + claims
#   select      — argmax/first_exceeds/…  binds a pivot → kernel (selection ops)
#   compute     — arithmetic (ratio/growth/margin)      → kernel (arithmetic ops)
#   compare     — side-by-side of named entities/periods → kernel/claims
#   exists      — yes/no predicate over a series         → claims+verifier
#   qualitative — narrative/explanatory                  → claims+verifier (never kernel)
SUBGOAL_TYPES = ("lookup", "select", "compute", "compare", "exists", "qualitative")

# QueryIR predicate → kernel selection op. The predicate already decided the IR was an
# extremum_pivot (the §5.2 constraint-determines-type rule); here it picks the kernel op.
_PREDICATE_OP = {
    "first_exceeds": "first_exceeds",
    "last_below": "last_below",
    "argmax": "argmax",
    "argmin": "argmin",
}


@dataclass
class SubGoal:
    """One node in the plan DAG. Mirrors the §5.3 signature.

    `binds` names the variable this goal RESOLVES (e.g. "pivot_year") so a later goal can
    reference it as "$pivot_year" in `inputs`. `selector` is a kernel-ready selection/
    compute spec for select/compute goals. `depends_on` lists upstream goal ids.
    """
    id: str
    type: str                                  # one of SUBGOAL_TYPES
    description: str
    binds: Optional[str] = None                # variable resolved here, e.g. "pivot_year"
    inputs: Dict[str, Any] = field(default_factory=dict)   # refs, e.g. {"period":"$pivot_year"}
    selector: Optional[Dict[str, Any]] = None  # kernel spec for select/compute goals
    depends_on: List[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    """A DAG of typed sub-goals + which binding is the final answer.

    `degraded=True` marks the safe single-`lookup` fallback (unrecognized shape) — the
    executor still runs it down the plain path, never worse than baseline (§5.3).
    """
    goals: List[SubGoal]
    final: str                                 # the goal id whose binding is the answer
    raw: str = ""                              # the original question (provenance)
    degraded: bool = False

    @property
    def is_bridge(self) -> bool:
        """A bridge plan = a pivot goal feeds a downstream read (the §2 failure shape)."""
        return any(g.depends_on for g in self.goals)


# ── selector builders (planner → kernel spec, no translation needed by C4) ──────────

def _row_ref(metric: str, section: str = "") -> Dict[str, Any]:
    """A {section,label} row reference the kernel's resolve() understands."""
    ref: Dict[str, Any] = {"label": metric}
    if section:
        ref["section"] = section
    return ref


def _period_selector(ir: QueryIR, metric: str, section: str) -> Dict[str, Any]:
    """select OVER PERIODS: scan one row across its years (first_exceeds/last_below).
    The COMPLETE series — the kernel uses every value column when `periods` is omitted,
    which is the whole point of a 'first exceeded' read (§3.3)."""
    op = _PREDICATE_OP[ir.constraints["predicate"]]
    sel: Dict[str, Any] = {"op": op, "over": "period", "row": _row_ref(metric, section)}
    thr = ir.constraints.get("threshold")
    if thr is not None:
        sel["threshold"] = thr
    return sel


def _entity_selector(ir: QueryIR, metric: str) -> Dict[str, Any]:
    """select OVER ENTITIES: compare the SAME row across the named entities at one period
    (argmax/argmin). Lists EVERY entity so the completeness check (§5.5) can verify the op
    saw them all. Period left to the executor when the IR fixes one; else the kernel's
    single-period entity scan still needs it, so we pass the IR's period if present."""
    op = _PREDICATE_OP[ir.constraints["predicate"]]
    rows = [{"row": _row_ref(metric, section=e)} for e in ir.entities]
    sel: Dict[str, Any] = {"op": op, "over": "entity", "rows": rows}
    if ir.periods:
        sel["period"] = ir.periods[0]
    return sel


def _is_entity_extremum(ir: QueryIR) -> bool:
    """argmax/argmin over a SET of named entities (≥2) ⇒ scan over entities; otherwise
    (first_exceeds/last_below, or a single series) ⇒ scan over periods. Structural, off
    the predicate + entity count the IR already carries."""
    pred = ir.constraints.get("predicate")
    return pred in ("argmax", "argmin") and len(ir.entities) >= 2


# ── plan builders per IR question_type ──────────────────────────────────────────────

def _plan_lookup(ir: QueryIR) -> QueryPlan:
    """One stated value, no pivot. A single grounding read."""
    metric = ir.metrics[0] if ir.metrics else ir.raw
    section = ir.entities[0] if ir.entities else ""
    inputs: Dict[str, Any] = {}
    if ir.periods:
        inputs["period"] = ir.periods[0]
    g = SubGoal(id="A", type="lookup", description=f"read {metric}",
                binds="answer", inputs=inputs,
                selector={"row": _row_ref(metric, section)})
    return QueryPlan(goals=[g], final="A", raw=ir.raw)


def _plan_extremum_pivot(ir: QueryIR) -> QueryPlan:
    """An extremum binds a pivot (year OR entity), then — if a follow-on metric is asked —
    a second goal reads that value FOR the resolved pivot. The bridge shape.

      A = select(extremum) → binds pivot
      B = lookup(metric, period=$pivot) depends_on A   [only when a 2nd metric is asked]
    """
    metric = ir.metrics[0] if ir.metrics else ir.raw
    over_entity = _is_entity_extremum(ir)

    if over_entity:
        selA = _entity_selector(ir, metric)
        binds = "pivot_entity"
        descA = f"{ir.constraints['predicate']} of {metric} over {ir.entities}"
    else:
        section = ir.entities[0] if ir.entities else ""
        selA = _period_selector(ir, metric, section)
        binds = "pivot_year"
        descA = f"{ir.constraints['predicate']} year for {metric}"

    goalA = SubGoal(id="A", type="select", description=descA, binds=binds, selector=selA)

    # a follow-on read exists when a SECOND, different metric is named (e.g. "…first
    # exceeded $20B, what was TOTAL NET SALES that year"). One metric ⇒ the pivot itself
    # is the answer (e.g. "which YEAR did X first exceed …").
    follow_metric = ir.metrics[1] if len(ir.metrics) > 1 else None
    if follow_metric is None:
        return QueryPlan(goals=[goalA], final="A", raw=ir.raw)

    pivot_var = "$" + binds
    inputs = {"period": pivot_var} if not over_entity else {"section": pivot_var}
    follow_section = ir.entities[0] if (not over_entity and ir.entities) else ""
    goalB = SubGoal(
        id="B", type="lookup",
        description=f"read {follow_metric} for {binds}",
        binds="answer", inputs=inputs, depends_on=["A"],
        selector={"row": _row_ref(follow_metric, follow_section)},
    )
    return QueryPlan(goals=[goalA, goalB], final="B", raw=ir.raw)


def _plan_lookup_pivot(ir: QueryIR) -> QueryPlan:
    """A stated NON-extremum condition picks the pivot, then read another value. Same
    bridge SHAPE as extremum_pivot but the pivot goal is a `select` whose condition is a
    match/closest/equality (the kernel resolves it as a filter/closest selection in C4).

      A = select(condition) → binds pivot_year
      B = lookup(follow metric, period=$pivot_year) depends_on A
    """
    pivot_metric = ir.metrics[0] if ir.metrics else ir.raw
    section = ir.entities[0] if ir.entities else ""
    goalA = SubGoal(
        id="A", type="select",
        description=f"year where {pivot_metric} meets the stated condition",
        binds="pivot_year",
        selector={"over": "period", "row": _row_ref(pivot_metric, section)},
    )
    follow_metric = ir.metrics[1] if len(ir.metrics) > 1 else pivot_metric
    follow_section = ir.entities[-1] if ir.entities else ""
    goalB = SubGoal(
        id="B", type="lookup",
        description=f"read {follow_metric} for the pivot year",
        binds="answer", inputs={"period": "$pivot_year"}, depends_on=["A"],
        selector={"row": _row_ref(follow_metric, follow_section)},
    )
    return QueryPlan(goals=[goalA, goalB], final="B", raw=ir.raw)


def _plan_compare(ir: QueryIR) -> QueryPlan:
    """A direct side-by-side of named entities/periods, no pivot. One compare goal that
    lists every operand (the executor reads each, the kernel/claims compares)."""
    metric = ir.metrics[0] if ir.metrics else ir.raw
    rows = [{"row": _row_ref(metric, section=e)} for e in ir.entities] or [{"row": _row_ref(metric)}]
    sel: Dict[str, Any] = {"over": "entity", "rows": rows}
    if ir.periods:
        sel["period"] = ir.periods[0]
    g = SubGoal(id="A", type="compare", description=f"compare {metric} across {ir.entities or ir.periods}",
                binds="answer", selector=sel)
    return QueryPlan(goals=[g], final="A", raw=ir.raw)


def _plan_exists(ir: QueryIR) -> QueryPlan:
    """A yes/no predicate (profit-or-loss, increased-every-year, reached-profitability).
    Routes to claims+verifier in C4 — NEVER the kernel as a number (§5.2)."""
    metric = ir.metrics[0] if ir.metrics else ir.raw
    section = ir.entities[0] if ir.entities else ""
    inputs: Dict[str, Any] = {}
    if ir.periods:
        inputs["period"] = ir.periods[0]
    g = SubGoal(id="A", type="exists", description=f"does the predicate hold for {metric}",
                binds="answer", inputs=inputs, selector={"row": _row_ref(metric, section)})
    return QueryPlan(goals=[g], final="A", raw=ir.raw)


def _plan_qualitative(ir: QueryIR) -> QueryPlan:
    """Narrative/explanatory. A single claims+verifier goal; never touches the kernel."""
    topic = ir.metrics[0] if ir.metrics else ir.raw
    g = SubGoal(id="A", type="qualitative", description=f"explain: {topic}",
                binds="answer", inputs={"query": ir.raw})
    return QueryPlan(goals=[g], final="A", raw=ir.raw)


def _degraded(ir: QueryIR) -> QueryPlan:
    """The safe fallback for an unmapped/compound shape: a single plain lookup down the
    fast path. Never worse than baseline (§5.3 degrade invariant)."""
    g = SubGoal(id="A", type="lookup", description="answer the question (degraded plan)",
                binds="answer", inputs={"query": ir.raw})
    return QueryPlan(goals=[g], final="A", raw=ir.raw, degraded=True)


_BUILDERS = {
    "lookup": _plan_lookup,
    "extremum_pivot": _plan_extremum_pivot,
    "lookup_pivot": _plan_lookup_pivot,
    "compare": _plan_compare,
    "exists": _plan_exists,
    "qualitative": _plan_qualitative,
}


def plan(ir: QueryIR) -> QueryPlan:
    """QueryIR → QueryPlan. Deterministic, structural, no LLM (§5.3 / Stage C3).

    Dispatch on the IR's question_type to a structural builder over the sub-goal basis.
    `compound` and any unrecognized/degraded IR fall back to a single lookup goal so the
    executor (C4) never has a worse path than today's single pass. Every plan carries
    explicit deps + named pivot bindings — the structure the C3 gate scores and the C4
    executor walks.
    """
    if ir is None:
        return _degraded(QueryIR(raw=""))
    builder = _BUILDERS.get(ir.question_type)
    if builder is None or ir.degraded:
        return _degraded(ir)
    try:
        return builder(ir)
    except Exception:
        # a malformed IR (e.g. empty everything) must never crash the spine → degrade
        return _degraded(ir)
