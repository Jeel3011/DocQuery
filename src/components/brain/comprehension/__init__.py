"""Comprehension organ (Wernicke's area) — BRAIN_REASONING_PLAN §5.2.

Parses the *question* (natural language) into a typed `QueryIR` the rest of the brain
consumes — pure structural understanding, no retrieval and no numbers. This is the
organ that lets the executive layer SEQUENCE and BIND instead of collapsing everything
into one free-text LLM pass (the §2 failure). It is also what structurally kills the
`aws_margin` class: once "AWS operating margin" is a typed intent (metric+entity+level),
the denominator is DERIVED from the entity, never free-picked by the LLM.
"""

from .query_ir import QueryIR, comprehend, QUESTION_TYPES

__all__ = ["QueryIR", "comprehend", "QUESTION_TYPES"]
