"""Perception organ (sensory cortex) — BRAIN_REASONING_PLAN §5.1.

Turns raw documents into faithful structure (Layer 0 extraction, which currently
lives in `table_extraction.py`) and turns an *intent* into the right cell/span with
confidence — the grounding bottleneck. This subpackage is the NEW home for the
grounding organ; the existing extraction code stays in place (§3 reorg escape hatch:
logical architecture over a high-blast-radius physical move).
"""

from .grounding import Grounding, Candidate, MetricIntent, ground_metric

__all__ = ["Grounding", "Candidate", "MetricIntent", "ground_metric"]
