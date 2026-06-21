"""The agent loop (AGENT_CORE_PLAN §3.2).

`run_agent(...)` is a generator of SSE-shaped events (§3.6). The model IS the
orchestrator: it decides which tools to call; there is NO question-type routing
anywhere (Harvey's lesson, verbatim). What this file owns is the hard part no
framework ships: the evidence ledger, budget enforcement, graceful degradation, and
the hook where the non-bypassable output gates (A3) bind the final answer to the
ledger.

Event shapes (yielded dicts), per §3.6:
    {"type": "agent_step", "n": int}
    {"type": "agent_thought", "text": str}
    {"type": "tool_call", "name": str, "args_summary": str}
    {"type": "tool_result", "name": str, "ok": bool, "summary": str, "n_provenance": int}
    {"type": "gate", "name": str, "pass": bool, "detail": str}
    {"type": "sources", "sources": [...]}              # ledger payloads
    {"type": "token", "text": str}                     # final answer (whole, in A2)
    {"type": "meta", "mode": str, "steps": int, "tokens": int, "abstained": bool, ...}

A2 wires everything EXCEPT the real gate logic, which A3 fills into
`run_output_gates`. The stub here passes any draft through (clearly marked) so the
loop is end-to-end testable now; turning the stub into the real gate is A3's whole job
and does not touch this file's control flow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .budgets import Budget
from .ledger import EvidenceLedger
from .model import BaseModel, ModelResponse, ToolCall
from .registry import REGISTRY, RunScope

logger = logging.getLogger(__name__)


# ── Output-gate hook (A3 replaces the body; A2 ships a passthrough stub) ─────────

@dataclass
class GateOutcome:
    passed: bool
    redacted_draft: Optional[str] = None
    failures: List[Dict[str, Any]] = None  # [{name, detail}]
    abstained: bool = False


_GATE_UNAVAILABLE_TEXT = (
    "I can't verify this answer right now (the verification layer was unavailable), so "
    "I'm not stating it. Please retry."
)


def _default_gate(draft: str, ledger: EvidenceLedger) -> GateOutcome:
    """Resolve the real A3 output gates lazily (avoids a gates↔loop import cycle:
    gates.py imports GateOutcome from this module). If gates.py is somehow unavailable
    the loop fails CLOSED — "non-bypassable" (§3.4) means an unverifiable draft is
    withheld, never shipped ungated. Failing open here would be the product failing."""
    try:
        from .gates import run_output_gates as _gates
        return _gates(draft, ledger)
    except Exception as exc:  # noqa: BLE001
        logger.error("[agent_core.loop] output gates unavailable — failing CLOSED: %s", exc)
        return GateOutcome(
            passed=False, abstained=True,
            failures=[{"name": "gates_unavailable", "detail": str(exc)}],
            redacted_draft=_GATE_UNAVAILABLE_TEXT,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _args_summary(args: Dict[str, Any]) -> str:
    """One-line, log-safe summary of tool args for the timeline."""
    try:
        s = json.dumps(args, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = str(args)
    return s[:160]


def _tool_result_message(call: ToolCall, result: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic tool_result content block referencing the tool_use id."""
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(result, default=str),
            "is_error": not result.get("ok", False),
        }],
    }


def _assistant_message(resp: ModelResponse) -> Dict[str, Any]:
    """Reconstruct the assistant turn (text + tool_use blocks) for the message history."""
    content: List[Dict[str, Any]] = []
    if resp.text:
        content.append({"type": "text", "text": resp.text})
    for tc in resp.tool_calls:
        content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
    return {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}


# ── The loop ──────────────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    *,
    model: BaseModel,
    scope: RunScope,
    budget: Budget,
    system_prompt: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    registry=REGISTRY,
    gate_fn: Callable[[str, EvidenceLedger], GateOutcome] = _default_gate,
    tools: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Drive one agent run, yielding §3.6 events. `model` is injected (live or scripted).

    Termination: the model returns a text answer with no tool calls AND the gates pass
    (or a single repair has been spent → the redacted draft ships). Budget exhaustion
    wraps up with whatever is gated + an explicit abstain. A model API error is retried
    once by the caller's model wrapper; an unrecoverable one degrades (the route falls
    back to Brain — that's A4). This function never raises into the generator consumer.
    """
    import time

    ledger = EvidenceLedger()
    # G7: a workflow template may restrict the run to its own `tools` subset (validated
    # against SCHEMAS by the registry); `tools=None` (every other caller) falls back to the
    # mode map — byte-identical to before.
    # G8: offer `search_knowledge` only when the run threaded a KB retrieval manager
    # (USE_KNOWLEDGE on + route wired it). Off ⇒ the tool isn't in the schema list ⇒
    # byte-identical to pre-G8.
    tool_schemas = registry.schemas(
        budget.mode, tools=tools,
        include_knowledge=scope.kb_retrieval_manager is not None,
    )

    messages: List[Dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": question})

    repair_attempted = False
    final_text: Optional[str] = None
    abstained = False
    started = time.monotonic()
    low_budget_warned = False  # inject the "wrap up" nudge once, near the wall

    def _wall_exhausted() -> bool:
        return budget.wall_clock_s > 0 and (time.monotonic() - started) >= budget.wall_clock_s

    while True:
        # Budget gate BEFORE each model call (§3.2): no silent overspend — steps,
        # tokens, AND wall clock are all hard ceilings.
        if budget.step_exhausted() or budget.tokens_exhausted() or _wall_exhausted():
            why = ("step" if budget.step_exhausted()
                   else "token" if budget.tokens_exhausted() else "wall-clock")
            yield {"type": "gate", "name": "budget", "pass": False,
                   "detail": f"{why} budget exhausted at step {budget.steps_used}"}
            # Don't discard verified work. The ledger holds every figure already traced
            # to a source cell; render it so the user sees what we DID confirm (e.g.
            # Google 14.8% + Amazon 14.9% computed before the wall) instead of an empty
            # "ran out of budget". Deterministic, $0, no extra latency.
            verified = ledger.partial_answer()
            final_text = (
                (verified + "\n\n_I stopped here (analysis budget reached) and did not "
                 "verify the remaining parts._")
                if verified else
                "I ran out of my analysis budget before I could verify any figure for "
                "this question. Please try narrowing it (one company / one metric)."
            )
            abstained = True
            break

        # BUDGET AWARENESS (the harness gap): the model must not fly toward the wall
        # blind. When it's near the ceiling, inject ONE message telling it to commit its
        # best VERIFIED answer now — this turns "found the answer at step 7, guillotined
        # at step 9 mid-retry" into "found it, delivered it". A frontier model wraps up
        # cleanly when told the runway is short; it can't when it doesn't know.
        steps_left = budget.remaining_steps()
        tokens_left = (budget.token_budget - budget.tokens_used) if budget.token_budget else 10**9
        if not low_budget_warned and (steps_left <= 3 or tokens_left <= 25000):
            low_budget_warned = True
            messages.append({
                "role": "user",
                "content": (
                    f"[budget notice] You have ~{steps_left} tool-steps left. STOP exploring "
                    f"and DELIVER now: state every figure you have ALREADY verified through a "
                    f"tool (with its [doc p.N] citation), and for anything still unresolved say "
                    f"plainly you couldn't verify it. Do not start a new line of investigation."
                ),
            })

        budget.steps_used += 1
        step = budget.steps_used
        yield {"type": "agent_step", "n": step}

        # Stream the model so text surfaces live (the UX fix: no freeze; the draft appears
        # token-by-token). Deltas are emitted as `token_delta` events; the loop still gets a
        # final ModelResponse (text + tool_calls + usage) to drive the tool/gate logic exactly
        # as before. A stream error inside model.stream() falls back to invoke-once there, so
        # this layer just retries the whole stream once on a hard failure, then degrades.
        def _run_stream():
            """Drive one model.stream(), forwarding ('delta', text) up and returning the
            final ModelResponse. Raises on a hard failure (caught below for retry/degrade)."""
            final_resp = None
            for kind, payload in model.stream(messages, tool_schemas):
                if kind == "delta":
                    yield ("delta", payload)
                elif kind == "done":
                    final_resp = payload
            return final_resp

        resp = None
        try:
            try:
                gen = _run_stream()
                while True:
                    try:
                        kind, payload = next(gen)
                    except StopIteration as si:
                        resp = si.value
                        break
                    if kind == "delta" and payload:
                        yield {"type": "token_delta", "text": payload}
            except Exception as first_exc:  # noqa: BLE001 — one retry for transient API blips
                logger.warning("[agent_core.loop] model.stream failed at step %d (retrying once): %s",
                               step, first_exc)
                resp = model.invoke(messages, tool_schemas)
        except Exception as exc:  # noqa: BLE001 — surfaced as a degrade signal to the caller
            logger.warning("[agent_core.loop] model call failed at step %d: %s", step, exc)
            yield {"type": "gate", "name": "model_error", "pass": False, "detail": str(exc)}
            final_text = None
            abstained = True
            # Signal degrade: yield meta with an error marker; A4's route falls back.
            yield {"type": "meta", "mode": budget.mode, "steps": step,
                   "tokens": budget.tokens_used, "abstained": True,
                   "error": f"model_error: {exc}", "degrade": True}
            return

        if resp is None:  # stream produced no 'done' — treat as a model error / degrade
            resp = model.invoke(messages, tool_schemas)

        budget.tokens_used += (resp.usage or {}).get("in", 0) + (resp.usage or {}).get("out", 0)
        if resp.text:
            yield {"type": "agent_thought", "text": resp.text[:500]}

        # Record the assistant turn in history.
        messages.append(_assistant_message(resp))

        if resp.wants_tools:
            for call in resp.tool_calls:
                yield {"type": "tool_call", "name": call.name,
                       "args_summary": _args_summary(call.args)}
                result = registry.execute(call, scope)
                n_prov = ledger.record(call.name, step, result.get("provenance"))
                yield {"type": "tool_result", "name": call.name,
                       "ok": bool(result.get("ok")), "summary": result.get("summary", ""),
                       "n_provenance": n_prov}
                messages.append(_tool_result_message(call, result))
            continue  # let the model see the results and decide the next move

        # No tool calls → the model thinks it's done. Run the output gates (A3).
        # A raising gate_fn fails CLOSED (withhold) — same contract as _default_gate.
        draft = resp.text or ""
        try:
            outcome = gate_fn(draft, ledger)
        except Exception as exc:  # noqa: BLE001
            logger.error("[agent_core.loop] gate_fn raised — failing CLOSED: %s", exc)
            outcome = GateOutcome(
                passed=False, abstained=True,
                failures=[{"name": "gates_unavailable", "detail": str(exc)}],
                redacted_draft=_GATE_UNAVAILABLE_TEXT,
            )
        for f in (outcome.failures or []):
            yield {"type": "gate", "name": f.get("name", "gate"), "pass": False,
                   "detail": f.get("detail", "")}

        if outcome.passed:
            yield {"type": "gate", "name": "output", "pass": True, "detail": "all claims traced"}
            final_text = draft
            break

        if repair_attempted:
            # Second failure → redact: ship the gated content, withhold the rest (§3.4).
            final_text = outcome.redacted_draft or draft
            abstained = outcome.abstained or True
            yield {"type": "gate", "name": "output", "pass": False,
                   "detail": "redacted ungated claims after one repair"}
            break

        # First failure → ONE repair turn: feed the gate failures back to the model.
        repair_attempted = True
        fb = "; ".join(f.get("detail", "") for f in (outcome.failures or [])) or "untraced claims"
        messages.append({
            "role": "user",
            "content": (
                f"Some claims didn't pass verification ({fb}). Fix or remove them: every "
                f"number must come from a tool result and every factual sentence must cite "
                f"a source. Re-answer."
            ),
        })
        # loop continues → model re-answers

    # Wrap up: emit sources from the ledger, the final answer, and run meta.
    yield {"type": "sources", "sources": ledger.to_sources()}
    if final_text is not None:
        yield {"type": "token", "text": final_text}
    yield {"type": "meta", "mode": budget.mode, "steps": budget.steps_used,
           "tokens": budget.tokens_used, "abstained": abstained,
           "n_evidence": len(ledger.entries)}
