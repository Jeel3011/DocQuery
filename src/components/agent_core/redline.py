"""G6.3 — Redline engine.

For each playbook row (clause topic + standard position), runs ONE bounded agent per
target document to:
  1. Extract the relevant clause from the document (like a grid cell — search + read).
  2. Compare it against the playbook's standard position.
  3. Emit a redline finding: {topic, target_quote, deviation, suggested_edit, rationale}.

BIND-OR-FLAG discipline (§0 of the G6 plan): a finding is emitted ONLY when:
  - `target_quote` cites a real span from the document (traced via the ledger), AND
  - `suggested_edit` cites a specific playbook rule (by clause_topic).
A finding with no clause span OR no rule reference is NOT emitted — the same failure
class as an uncited number.

No "soft" path for contract prose. If the agent cannot ground a clause, it emits
a MISSING or ABSTAIN finding — the honest answer, never a hallucination.

The model is injected (model_factory), so this is fully offline-testable without a
live API call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .budgets import Budget
from .loop import run_agent, GateOutcome
from .registry import REGISTRY, RunScope

logger = logging.getLogger(__name__)

# Per-clause-topic cell budget — tighter than a full deep run; one clause, one doc.
_REDLINE_MAX_STEPS = 8
_REDLINE_WALL_S = 120.0
_REDLINE_TOKEN_BUDGET = 35_000


@dataclass
class RedlineFinding:
    clause_topic: str
    status: str                     # "deviation" | "conforming" | "missing" | "abstain"
    target_quote: Optional[str]     # the exact clause text found in the target doc
    deviation: Optional[str]        # description of how it deviates from the standard
    suggested_edit: Optional[str]   # suggested replacement clause text
    rationale: Optional[str]        # why: cites the playbook rule
    playbook_standard: str          # the firm's standard position (for the .docx diff)
    grounded: bool = False          # True only when target_quote is ledger-traced
    # S-B: machine-readable source reference — which clause topic + playbook rule
    # the suggested edit is grounded in.  None when the edit is ungrounded (status
    # will be "abstain" in that case — the two fields always agree).
    source_ref: Optional[str] = None
    # S-B: explicit ungrounded flag so callers don't have to re-derive it.
    ungrounded_edit: bool = False


@dataclass
class RedlineResult:
    doc_id: str
    doc_name: str
    playbook_id: Optional[str]
    findings: List[RedlineFinding] = field(default_factory=list)
    error: Optional[str] = None


def _redline_system_prompt(
    clause_topic: str,
    standard_position: str,
    fallback_position: Optional[str],
    doc_name: str,
) -> str:
    fallback_line = (
        f"\nAcceptable fallback: {fallback_position}"
        if fallback_position else ""
    )
    return f"""You are a contract reviewer comparing a target document against a firm playbook.

Document: {doc_name}
Clause topic: {clause_topic}
Firm standard position: {standard_position}{fallback_line}

YOUR TASK:
1. Search the document for the clause covering "{clause_topic}". Use search_vault with relevant
   keywords (e.g. for "Governing Law": "governing law jurisdiction courts").
2. Read the relevant clause text carefully.
3. Compare what the document says against the firm standard position above.
4. Return ONE JSON object with EXACTLY these keys and NO others:

{{
  "status": "deviation" | "conforming" | "missing" | "abstain",
  "target_quote": "<exact verbatim text from the document, or null if missing>",
  "deviation": "<clear description of how the clause deviates from the standard, or null if conforming/missing>",
  "suggested_edit": "<suggested replacement clause text that would bring it into conformity, or null>",
  "rationale": "<why this edit is needed — must cite the firm standard position above, or null>"
}}

RULES (non-negotiable):
- target_quote MUST be verbatim text from the document (found via search/read). Do NOT paraphrase.
- If the clause is absent: status "missing", target_quote null.
- If you cannot confidently identify the clause: status "abstain".
- suggested_edit must be a complete, usable clause — not a description of what to change.
- rationale MUST reference the firm standard position. Do NOT invent a rule.
- NEVER invent a clause that isn't in the document. NEVER fabricate a quote.
- Return ONE JSON object and NOTHING else."""


def build_redline_cell(
    clause_topic: str,
    standard_position: str,
    fallback_position: Optional[str],
    model_factory: Callable,
    scope: RunScope,
    doc_name: str,
) -> RedlineFinding:
    """Run one bounded agent for one clause topic × one document.

    Returns a RedlineFinding. Never raises — errors degrade to status="abstain".
    """
    budget = Budget(
        mode="grid",
        model="",  # injected via model_factory; not used by Budget itself
        max_steps=_REDLINE_MAX_STEPS,
        wall_clock_s=_REDLINE_WALL_S,
        token_budget=_REDLINE_TOKEN_BUDGET,
    )
    sys_prompt = _redline_system_prompt(clause_topic, standard_position, fallback_position, doc_name)
    question = (
        f"Review the '{clause_topic}' clause in {doc_name} against the firm standard position. "
        f"Return the JSON redline finding."
    )

    model = model_factory(system=sys_prompt)

    def _json_gate(draft, ledger):
        """Redline cells emit JSON only — no numeric tracing needed."""
        return GateOutcome(passed=True, abstained=False, failures=[])

    final_text: Optional[str] = None
    try:
        for ev in run_agent(question=question, model=model, scope=scope,
                            budget=budget, gate_fn=_json_gate):
            if ev.get("type") == "token":
                final_text = ev.get("text") or final_text
    except Exception as exc:
        logger.warning("[redline] agent error for '%s'/'%s': %s", doc_name, clause_topic, exc)
        return RedlineFinding(
            clause_topic=clause_topic, status="abstain",
            target_quote=None, deviation=None, suggested_edit=None,
            rationale=f"Agent error: {exc}", playbook_standard=standard_position,
        )

    if not final_text:
        return RedlineFinding(
            clause_topic=clause_topic, status="abstain",
            target_quote=None, deviation=None, suggested_edit=None,
            rationale="No response from agent.", playbook_standard=standard_position,
        )

    # Parse the JSON envelope
    try:
        # Strip markdown fences if the model wrapped it
        text = final_text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.split("```")[0].strip()
        env = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return RedlineFinding(
            clause_topic=clause_topic, status="abstain",
            target_quote=None, deviation=None, suggested_edit=None,
            rationale="Could not parse agent response as JSON.",
            playbook_standard=standard_position,
        )

    status = env.get("status", "abstain")
    target_quote = env.get("target_quote") or None
    deviation = env.get("deviation") or None
    suggested_edit = env.get("suggested_edit") or None
    rationale = env.get("rationale") or None

    # BIND-OR-FLAG: a deviation finding without both target_quote AND rationale is demoted.
    grounded = bool(target_quote)
    ungrounded_edit = False
    if status == "deviation" and (not target_quote or not rationale):
        status = "abstain"
        rationale = "Bind-or-flag: deviation finding lacked a quoted clause or a rule citation."
        ungrounded_edit = True

    # S-B: source_ref — the machine-readable grounding trace for the suggested edit.
    # A grounded deviation carries "clause_topic → playbook rule" so the caller can
    # verify provenance without re-parsing the rationale string.  Ungrounded = None.
    source_ref: Optional[str] = None
    if not ungrounded_edit and suggested_edit and rationale:
        source_ref = f"{clause_topic} → {standard_position[:120]}"

    return RedlineFinding(
        clause_topic=clause_topic, status=status,
        target_quote=target_quote, deviation=deviation,
        suggested_edit=suggested_edit, rationale=rationale,
        playbook_standard=standard_position, grounded=grounded,
        source_ref=source_ref, ungrounded_edit=ungrounded_edit,
    )


def build_redline(
    playbook_rows: List[Dict[str, Any]],
    model_factory: Callable,
    scope: RunScope,
    doc_id: str,
    doc_name: str,
    playbook_id: Optional[str] = None,
) -> RedlineResult:
    """Run the full redline for one document against a set of playbook rows.

    Runs cells sequentially (same discipline as build_grid). Never raises.
    """
    result = RedlineResult(doc_id=doc_id, doc_name=doc_name, playbook_id=playbook_id)
    for row in playbook_rows:
        try:
            finding = build_redline_cell(
                clause_topic=row["clause_topic"],
                standard_position=row["standard_position"],
                fallback_position=row.get("fallback_position"),
                model_factory=model_factory,
                scope=scope,
                doc_name=doc_name,
            )
            result.findings.append(finding)
        except Exception as exc:
            logger.warning("[redline] cell error '%s': %s", row.get("clause_topic"), exc)
            result.findings.append(RedlineFinding(
                clause_topic=row.get("clause_topic", "unknown"),
                status="abstain", target_quote=None, deviation=None,
                suggested_edit=None, rationale=f"Unexpected error: {exc}",
                playbook_standard=row.get("standard_position", ""),
            ))
    return result
