"""Coordinator / meta-reasoner (brain stem) — BRAIN_REASONING_PLAN §5.6.

"Knows when to do what." Classifies a question and, for the numeric/pivot/bridge shapes the
executive spine is built for, runs the full deterministic chain — comprehend → plan →
execute → self-monitor — over the already-loaded table grids, returning a rendered,
provenance-carrying, **abstain-aware** block. For everything else (simple lookup, narrative,
degraded comprehension, or a monitor abstain) it returns `None`, so the caller falls through
to today's path BYTE-IDENTICALLY. This is the single dispatch seam (§5.6): the spine is
opt-in, the fast path is the default, and the prime directive — never regress single-doc /
normal-chat latency or quality — holds because a non-spine question never enters here.

What the spine adds over the bare Analyst: it BINDS the pivot (the year/entity) as a
first-class object and then reads the follow-on metric FOR that bound pivot, and it runs the
reasoning verifier (§5.5) that converts a wrong-pivot/incomplete/untraceable result into a
clean ABSTAIN instead of a confident wrong number. The kernel still reads every cell; no LLM
emits a figure (§5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .analyst import Grid
from .comprehension import comprehend, QueryIR
from .executive import plan, execute
from .monitoring import monitor


# The question shapes the deterministic spine handles end-to-end. Currently ONLY
# `extremum_pivot` — its `select` goals carry a whitelisted kernel op (argmax/argmin/
# first_exceeds/last_below), so plan→execute resolves a real pivot.
#
# `lookup_pivot` and `compare` are DELIBERATELY excluded (SPINE_TRUST_BREAKS H1): their
# planner builders emit a `select` goal with NO `op` (the kernel has no "closest/matches"
# or entity-compare selection op yet), so they could only ever ABSTAIN and fall through —
# listing them here just burned a plan/execute round-trip and pretended to a capability we
# don't have. They fall through to the existing Analyst/Brain path EXACTLY as before; the
# only change is the spine no longer feigns coverage. Re-add a type here the moment its
# kernel op lands (the executability gate, eval/test_spine_executability.py, enforces that
# every type listed here produces an executable plan). A plain `lookup` is likewise left to
# the existing path; `exists`/`qualitative` route to claims+verifier elsewhere.
_SPINE_TYPES = ("extremum_pivot",)


@dataclass
class SpineOutcome:
    """The coordinator's result. `block` is a markdown block to hand the Brain as its
    `analyst_block` (the monitored answer + provenance). `abstained` marks a monitor/spine
    abstain — the block then states the clean refusal (withholding the figure, §4a).
    `applied=False` means the spine didn't run (non-pivot / degraded) → caller falls
    through unchanged."""
    applied: bool
    block: Optional[str] = None
    abstained: bool = False
    binding: Optional[str] = None      # the resolved pivot, for logging/telemetry
    reason: str = ""


def _render_answer(plan_obj, res, verdict) -> str:
    """Render the monitored spine result as the markdown block the Brain prepends to its
    REDUCE context (same role as the Analyst block). Shows the bound pivot, the answer
    value, and the source cells — auditable back to the grid (§4b)."""
    b = res.answer_binding
    lines = ["**Executive reasoning (deterministic, self-monitored):**", ""]
    # the per-goal hop trail (bind the pivot → read the follow-on) — the Trust UI signal
    for step in res.trail:
        lines.append(f"- {step}")
    lines.append("")
    val = b.value
    lines.append(f"**Resolved answer:** {val}")
    if b.cells:
        lines.append("")
        lines.append("| Value | Source cell |")
        lines.append("| --- | --- |")
        for c in b.cells:
            lines.append(f"| {c.raw} | {c.trace()} |")
    lines.append("")
    lines.append(f"_Reasoning checks: {'; '.join(verdict.checks) or 'n/a'}_")
    return "\n".join(lines)


def run_executive_spine(question: str, grids: List[Grid], llm) -> SpineOutcome:
    """Coordinator entry point (§5.6). Returns a `SpineOutcome`.

    Steps: comprehend (one cheap LLM call) → if the type is a spine shape, plan (no LLM) →
    execute over the grids (no LLM) → monitor (no LLM). The LLM is used ONLY to comprehend
    the question's structure; every number is read by the kernel and every pivot is checked
    by the monitor. Any failure degrades to `applied=False` so the caller is never worse
    off than the existing path.
    """
    if not grids or not (question or "").strip():
        return SpineOutcome(applied=False, reason="no grids or empty question")

    try:
        ir: QueryIR = comprehend(question, llm)
    except Exception as exc:
        return SpineOutcome(applied=False, reason=f"comprehend failed: {exc}")

    # only the pivot/bridge/compare shapes use the spine; everything else falls through to
    # the existing Analyst/Brain/claims paths unchanged (the §5.6 dispatch).
    if ir.degraded or ir.question_type not in _SPINE_TYPES:
        return SpineOutcome(applied=False, reason=f"not a spine type ({ir.question_type})")

    try:
        plan_obj = plan(ir)
        res = execute(plan_obj, grids)
        verdict = monitor(plan_obj, res)
    except Exception as exc:
        # the spine must never crash the request — degrade to the existing path
        return SpineOutcome(applied=False, reason=f"spine error: {exc}")

    # Dispatch (§5.6):
    #   • spine did NOT resolve (a capability gap — e.g. a lookup_pivot the kernel has no op
    #     for, or grounding abstained) → FALL THROUGH (applied=False, abstained=False). The
    #     spine simply has nothing to compute; a prose answer the Brain can give is not
    #     suppressed.
    #   • spine RESOLVED a figure but the self-monitor FLAGGED the reasoning → return
    #     abstained=True. The caller (chat.py / brain.run) reads this as a binary WITHHOLD
    #     and refuses cleanly INSTEAD of letting the old prose path state the rejected number
    #     — the C4 enforcement (§5.5). This is the deliberate over-abstain trade vs. shipping
    #     a confident wrong: correct-or-abstain is the prime directive (§4a).
    #   • spine resolved AND verified → inject the authoritative figure block (the win).
    if not res.ok:
        return SpineOutcome(applied=False, reason=f"spine did not resolve ({res.reason[:50]})")
    if verdict.abstain:
        return SpineOutcome(applied=False, abstained=True, reason=verdict.reason[:140])

    b = res.answer_binding
    return SpineOutcome(
        applied=True,
        block=_render_answer(plan_obj, res, verdict),
        abstained=False,
        binding=str(b.value) if b else None,
        reason="verified",
    )
