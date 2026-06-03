"""
Claim-level entailment verifier — Phase 4 / §4a.3 step 2.

For each claim produced by REDUCE, this verifier asks an independent LLM
(different model than the one that generated the claim) whether the claim is
entailed by its cited evidence spans.

Design choices:
  - Uses a separate, explicit verification prompt (not the generation model's
    own self-assessment) to de-correlate errors per §4a.3.
  - Returns a confidence score in [0, 1]; below ABSTAIN_THRESHOLD the claim
    is dropped from the answer and flagged.
  - Numbers / computations are NOT sent through this verifier — they go through
    the deterministic Analyst tool (§4b, Phase 4.3). This verifier handles
    prose claims only.
  - Failure is non-fatal: if the verifier LLM call fails, the claim is marked
    unverified (confidence = 0.5) rather than crashing the whole answer.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.components.brain.claims import Claim, EvidenceSpan
from src.logger import get_logger

logger = get_logger(__name__)

# Confidence below this → claim is dropped from the answer (abstain/flag)
ABSTAIN_THRESHOLD = 0.5

# Verification prompt  ──────────────────────────────────────────────────────────
_VERIFY_SYSTEM = """You are a strict fact-checker verifying whether a CLAIM is entailed by SOURCE TEXT.

Rules:
- Answer ONLY with a JSON object: {"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0, "reason": "..."}
- "supported"     → the source text directly and explicitly supports the claim (confidence ≥ 0.8)
- "partial"       → the source partially supports it but the claim over-states or adds details not present (confidence 0.3-0.79)
- "not_supported" → the source does not support or contradicts the claim (confidence 0.0-0.29)
- confidence is your certainty about the verdict, not about the claim itself.
- Do NOT use world knowledge — only the provided source text matters.
- Be strict: if the claim adds ANY information not in the source, it is at most "partial".
"""

_VERIFY_USER = """CLAIM: {claim}

SOURCE TEXT:
{evidence}

Is the claim entailed by the source text? Respond with JSON only."""


def verify_claim(
    claim: Claim,
    llm,
    abstain_threshold: float = ABSTAIN_THRESHOLD,
) -> Claim:
    """Run entailment check on a single claim.  Mutates and returns the claim.

    Args:
        claim:              The claim to verify.
        llm:                A LangChain-compatible LLM (should differ from the generator).
        abstain_threshold:  Drop claim if confidence below this.

    Returns:
        The claim with .verified and .confidence updated.
    """
    if not claim.evidence:
        claim.verified = False
        claim.confidence = 0.0
        return claim

    evidence_text = "\n---\n".join(
        f"[{e.doc_id} / {e.chunk_id}]: {e.verbatim_span}"
        for e in claim.evidence
    )

    try:
        # Build messages directly — _VERIFY_SYSTEM contains a literal JSON example
        # ({"verdict": ...}) which ChatPromptTemplate would mis-parse as variables.
        from langchain_core.messages import SystemMessage, HumanMessage

        user_msg = _VERIFY_USER.format(claim=claim.text, evidence=evidence_text)
        raw = llm.invoke(
            [SystemMessage(content=_VERIFY_SYSTEM), HumanMessage(content=user_msg)]
        ).content

        # Parse JSON response
        raw_stripped = (raw or "").strip()
        if raw_stripped.startswith("```"):
            raw_stripped = raw_stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw_stripped)

        verdict = result.get("verdict", "not_supported")
        conf = float(result.get("confidence", 0.5))

        claim.confidence = conf
        claim.verified = conf >= abstain_threshold

        if not claim.verified:
            logger.info(
                "Verifier: claim dropped (conf=%.2f < %.2f): %s",
                conf, abstain_threshold, claim.text[:80],
            )

    except Exception as exc:
        logger.warning("Verifier LLM call failed (non-fatal): %s", exc)
        claim.confidence = 0.5  # uncertain — don't drop, but flag
        claim.verified = False

    return claim


def verify_claims(
    claims: list[Claim],
    llm,
    abstain_threshold: float = ABSTAIN_THRESHOLD,
    max_workers: int = 8,
) -> tuple[list[Claim], list[Claim]]:
    """Verify a list of claims (in parallel, bounded concurrency).

    Each claim is an independent entailment check, so we run them in a small
    thread pool to keep latency reasonable when REDUCE feeds in dozens of claims.
    Order is preserved in the returned lists.

    Returns:
        (verified_claims, dropped_claims)
        verified_claims: confidence ≥ abstain_threshold
        dropped_claims:  confidence < abstain_threshold (for audit log)
    """
    if not claims:
        return [], []

    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(max_workers, len(claims)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map preserves input order
        checked = list(pool.map(lambda c: verify_claim(c, llm, abstain_threshold), claims))

    verified = [c for c in checked if c.verified]
    dropped = [c for c in checked if not c.verified]
    return verified, dropped
