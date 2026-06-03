"""
Structured Claim objects — Phase 4 / §4a.2 of the Brain plan.

A claim is the first-class, verifiable unit of knowledge the Brain produces.
Prose is rendered FROM verified claims, never emitted directly and hoped to be grounded.

Every claim carries:
  text         — the assertion itself (one sentence)
  evidence     — list of {doc_id, chunk_id, verbatim_span} supporting it
  confidence   — float [0, 1]
  verified     — True if the entailment verifier confirmed it
  derivation   — "extracted" | "computed" | "synthesized" | "argument"

A claim with no evidence span, or one whose span doesn't entail it, cannot
appear in the final answer — the verifier drops or flags it (§4a.3 step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvidenceSpan:
    """One piece of supporting evidence for a claim."""
    doc_id: str
    chunk_id: str
    verbatim_span: str          # quoted text from the source chunk


@dataclass
class Claim:
    text: str
    evidence: list[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False
    derivation: Literal["extracted", "computed", "synthesized", "argument"] = "extracted"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "evidence": [
                {"doc_id": e.doc_id, "chunk_id": e.chunk_id, "verbatim_span": e.verbatim_span}
                for e in self.evidence
            ],
            "confidence": self.confidence,
            "verified": self.verified,
            "derivation": self.derivation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(
            text=d["text"],
            evidence=[EvidenceSpan(**e) for e in d.get("evidence", [])],
            confidence=d.get("confidence", 0.0),
            verified=d.get("verified", False),
            derivation=d.get("derivation", "extracted"),
        )


@dataclass
class PerDocExtract:
    """Result of the MAP step for one document."""
    doc_id: str
    filename: str
    claims: list[Claim] = field(default_factory=list)
    nothing_relevant: bool = False   # True when the doc had no relevant content
    error: str | None = None         # set if MAP failed for this doc (non-fatal)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "claims": [c.to_dict() for c in self.claims],
            "nothing_relevant": self.nothing_relevant,
            "error": self.error,
        }


@dataclass
class BrainResult:
    """Final output of the Brain (REDUCE + VERIFY step)."""
    answer: str
    claims: list[Claim] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False          # True when confidence < abstention threshold
    abstain_reason: str | None = None
    sources: list[dict] = field(default_factory=list)  # same shape as generate() sources
    # Coverage ledger
    docs_routed: int = 0
    docs_read: int = 0
    docs_relevant: int = 0
    docs_failed: int = 0
    # Per-doc extracts (for audit/debugging)
    per_doc_extracts: list[PerDocExtract] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "confidence": self.confidence,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "sources": self.sources,
            "coverage": {
                "docs_routed": self.docs_routed,
                "docs_read": self.docs_read,
                "docs_relevant": self.docs_relevant,
                "docs_failed": self.docs_failed,
            },
        }
