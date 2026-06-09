"""Self-monitoring organ (anterior cingulate) — BRAIN_REASONING_PLAN §5.5.

Verifies the REASONING, not just the facts — the organ that converts a confident WRONG
into a clean ABSTAIN. `reasoning_verifier.monitor(plan, exec_result)` runs the §5.5 checks
(predicate / completeness / binding / structural invariants) over the executive spine's
deterministic output and returns a BINARY verdict; `invariants.py` holds the generic
value-based structural laws (total ≈ Σ children, figure→cell). LLM-free.

Distinct from the atomic `verifier.py` (claim↔span entailment), which cannot catch a
wrong-pivot answer because each atomic claim is individually true. The two work together:
the atomic verifier grounds each sentence; this one validates the bridge between them.
"""

from .reasoning_verifier import monitor, ReasoningVerdict
from .invariants import InvariantCheck, total_consistency, figure_traces_to_cells

__all__ = [
    "monitor", "ReasoningVerdict",
    "InvariantCheck", "total_consistency", "figure_traces_to_cells",
]
