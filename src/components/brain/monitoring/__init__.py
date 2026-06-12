"""Structural invariants for the output gates.

`invariants.py` holds the generic value-based structural laws (total ≈ Σ children,
figure→cell) used by the agent-core `verify_numbers` output gate. LLM-free.

(The Spine-coupled `reasoning_verifier.monitor` — a verdict over the retired executive
planner/executor — was removed 2026-06-12 with the rest of the fixed-route cognition.
The agent core verifies numbers at the OUTPUT via `figure_traces_to_cells`, not by
monitoring a plan.)
"""

from .invariants import InvariantCheck, total_consistency, figure_traces_to_cells

__all__ = [
    "InvariantCheck", "total_consistency", "figure_traces_to_cells",
]
