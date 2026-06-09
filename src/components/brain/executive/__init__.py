"""Executive organ (prefrontal cortex) — BRAIN_REASONING_PLAN §5.3.

The spine: *decompose → bind → route*. C3 ships the planner (QueryIR → QueryPlan, a DAG
of typed sub-goals with named pivot bindings); the workspace (variable bindings) and
executor (DAG walk + per-type routing) arrive in C4. The planner is deterministic and
gated independently of execution — a structural transform over the small sub-goal basis,
which is the most verifiable place to make the brain SEQUENCE instead of collapsing a
bridge question into one free-text pass (the §2 failure that produces a confident wrong).
"""

from .planner import plan, QueryPlan, SubGoal, SUBGOAL_TYPES

__all__ = ["plan", "QueryPlan", "SubGoal", "SUBGOAL_TYPES"]
