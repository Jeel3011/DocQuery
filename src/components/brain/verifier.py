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
    max_workers: int = 3,
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


# ── REDUCE-output verification (§4a.3 step 2) ───────────────────────────────────
# The per-claim verifier above checks the MAP-extracted claims. But REDUCE then
# *synthesizes* prose from those claims, and synthesis can introduce an
# unsupported "connection" that no individual claim asserts (§4a.1 row 3). This
# second pass closes that hole: every substantive sentence of the synthesized
# answer must be entailed by the pool of already-verified claims. Sentences that
# are not are flagged, and the answer's groundedness ratio downgrades confidence.

# Sentences shorter than this (chars) are treated as connective/structural and
# skipped (headings, "In summary:", list bullets with no assertion, etc.).
_MIN_SENTENCE_CHARS = 25

_REDUCE_VERIFY_SYSTEM = """You are a strict fact-checker. You are given a SENTENCE from an AI-generated answer and a set of VERIFIED FACTS that were extracted from source documents. Decide whether the sentence is supported by the verified facts.

Rules:
- Answer ONLY with a JSON object: {"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}
- "supported"     → every assertion in the sentence is backed by the verified facts (confidence ≥ 0.8)
- "partial"       → the sentence is mostly supported but adds a detail or connection not in the facts (confidence 0.3-0.79)
- "not_supported" → the sentence asserts something the verified facts do not contain or contradict (confidence 0.0-0.29)
- A sentence that only restates the question, gives structure ("Here is a summary"), or hedges ("I could not find X") is "supported" with confidence 1.0 — it asserts no new fact.
- Do NOT use world knowledge — only the verified facts matter. A fluent, plausible-sounding connection that the facts do not state is "not_supported".
"""

_REDUCE_VERIFY_USER = """VERIFIED FACTS:
{facts}

SENTENCE FROM THE ANSWER:
{sentence}

Is the sentence supported by the verified facts? Respond with JSON only."""


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no nltk dependency).

    Splits on sentence-final punctuation followed by whitespace, but keeps
    markdown structure (lines/bullets) from being glued together.
    """
    import re

    # Split on newlines first (preserves markdown structure), then on sentence enders.
    pieces: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading markdown bullet/heading markers for the assertion check.
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            if sent:
                pieces.append(sent)
    return pieces


def verify_reduce_output(
    answer: str,
    verified_claims: list[Claim],
    llm,
    support_threshold: float = ABSTAIN_THRESHOLD,
) -> tuple[float, list[str]]:
    """Check that the synthesized REDUCE answer is entailed by verified claims.

    This is §4a.3 step 2: independent entailment over the *synthesized* sentences,
    not just the MAP-extracted claims. Catches REDUCE-introduced hallucinations.

    Args:
        answer:            The synthesized prose from REDUCE.
        verified_claims:   The pool of already-verified claims REDUCE was given.
        llm:               Independent verifier LLM (same one used for claims).
        support_threshold: A sentence at/above this confidence counts as grounded.

    Returns:
        (groundedness, unsupported_sentences)
        groundedness:         fraction of substantive sentences that are supported
                              [0, 1]. 1.0 if there are no substantive sentences.
        unsupported_sentences: the sentences that failed the check (for flagging).
    """
    if not answer or not answer.strip():
        return 1.0, []
    if not verified_claims:
        # Nothing to ground against — any substantive prose is unsupported.
        substantive = [s for s in _split_sentences(answer) if len(s) >= _MIN_SENTENCE_CHARS]
        return (1.0 if not substantive else 0.0), substantive

    facts_text = "\n".join(f"- {c.text}" for c in verified_claims)

    sentences = [s for s in _split_sentences(answer) if len(s) >= _MIN_SENTENCE_CHARS]
    if not sentences:
        return 1.0, []

    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.messages import SystemMessage, HumanMessage

    def _check(sentence: str) -> tuple[str, bool]:
        try:
            user_msg = _REDUCE_VERIFY_USER.format(facts=facts_text, sentence=sentence)
            raw = llm.invoke(
                [SystemMessage(content=_REDUCE_VERIFY_SYSTEM), HumanMessage(content=user_msg)]
            ).content
            raw_stripped = (raw or "").strip()
            if raw_stripped.startswith("```"):
                raw_stripped = raw_stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw_stripped)
            conf = float(result.get("confidence", 0.5))
            return sentence, conf >= support_threshold
        except Exception as exc:  # non-fatal: treat as unsupported (conservative)
            logger.warning("REDUCE verifier failed for a sentence (non-fatal): %s", exc)
            return sentence, False

    workers = max(1, min(3, len(sentences)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_check, sentences))

    unsupported = [s for s, ok in results if not ok]
    supported_count = len(sentences) - len(unsupported)
    groundedness = supported_count / len(sentences)

    if unsupported:
        logger.info(
            "REDUCE verifier: %d/%d sentences unsupported (groundedness=%.2f)",
            len(unsupported), len(sentences), groundedness,
        )
    return groundedness, unsupported
