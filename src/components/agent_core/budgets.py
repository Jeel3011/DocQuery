"""Per-mode budgets for the agent loop (AGENT_CORE_PLAN §3.1, §3.2).

A budget is a HARD ceiling enforced in code by the loop runner — not advisory. When
any limit is hit the loop stops cleanly and the run wraps up with whatever is already
gated plus an explicit "ran out of budget" abstain for the rest (§3.2). Values come
from config (env-overridable); this module only shapes them per mode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    """The ceiling for one run. `model` is the orchestrator model id for this mode."""
    mode: str
    model: str
    max_steps: int
    wall_clock_s: float
    token_budget: int

    # mutable counters the loop updates as it runs
    steps_used: int = 0
    tokens_used: int = 0

    def step_exhausted(self) -> bool:
        return self.steps_used >= self.max_steps

    def tokens_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_used >= self.token_budget

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)


def budget_for(mode: str, config) -> Budget:
    """Build the Budget for `mode` ('fast' | 'standard' | 'deep') from config.

    fast has no tool steps (it's the existing direct path); standard/deep get the
    config's step/wall/token ceilings and the configured Opus model. Unknown modes
    fall back to standard (safe — bounded).
    """
    if mode == "deep":
        return Budget(
            mode="deep",
            model=config.AGENT_MODEL_DEEP,
            max_steps=config.AGENT_DEEP_MAX_STEPS,
            wall_clock_s=config.AGENT_DEEP_WALL_S,
            token_budget=config.AGENT_DEEP_TOKEN_BUDGET,
        )
    if mode == "fast":
        return Budget(
            mode="fast",
            model=config.AGENT_MODEL_CLASSIFIER,  # fast path doesn't loop; placeholder model
            max_steps=0,
            wall_clock_s=3.0,
            token_budget=4000,
        )
    # standard (default)
    return Budget(
        mode="standard",
        model=config.AGENT_MODEL_STANDARD,
        max_steps=config.AGENT_STD_MAX_STEPS,
        wall_clock_s=config.AGENT_STD_WALL_S,
        token_budget=config.AGENT_STD_TOKEN_BUDGET,
    )
