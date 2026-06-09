"""Executive organ (prefrontal cortex) — BRAIN_REASONING_PLAN §5.3.

The spine: *decompose → bind → route*.
  • C3 planner   — `plan(QueryIR) → QueryPlan`, a DAG of typed sub-goals with named pivot
    bindings (deterministic, structural, gated independently of execution).
  • C4 executor  — `execute(QueryPlan, grids) → ExecResult`: walk the DAG, BIND each pivot
    into the `Workspace` (provenance-carrying `Binding`s), ROUTE each goal to the right
    organ (select/compute → kernel; lookup → grounding; exists/qualitative → claims), and
    resolve the final answer or abstain. Deterministic + LLM-free.

Together they make the brain SEQUENCE and COMPOSE instead of collapsing a bridge question
into one free-text pass (the §2 failure that produces a confident wrong).
"""

from .planner import plan, QueryPlan, SubGoal, SUBGOAL_TYPES
from .workspace import Workspace, Binding
from .executor import execute, ExecResult

__all__ = [
    "plan", "QueryPlan", "SubGoal", "SUBGOAL_TYPES",
    "Workspace", "Binding", "execute", "ExecResult",
]
