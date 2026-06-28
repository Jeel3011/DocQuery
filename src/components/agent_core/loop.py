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

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

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


def _make_gate(question: str, sectioned: bool = False) -> Callable[..., GateOutcome]:
    """Return a gate_fn closure that threads `question` into the output gates for T2
    (completeness). The gate_fn interface is (draft, ledger) — question is captured from
    the run context so the completeness check knows which entities were asked for.

    `sectioned=False` (default) wraps `run_output_gates` (the simple/standard whole-answer
    path). `sectioned=True` wraps `gate_sectioned` (the deep/draft/report per-section path)
    — both gates accept `question` for the completeness check; without this wrapper the
    deep path runs `gate_sectioned(draft, ledger)` with `question=""`, silently disabling T2
    on exactly the multi-entity reports that need it most.

    Fails CLOSED on import error — byte-identical to the old _default_gate behavior."""
    def _gate(draft: str, ledger: EvidenceLedger) -> GateOutcome:
        try:
            from .gates import run_output_gates, gate_sectioned
            _gates = gate_sectioned if sectioned else run_output_gates
            return _gates(draft, ledger, question=question)
        except Exception as exc:  # noqa: BLE001
            logger.error("[agent_core.loop] output gates unavailable — failing CLOSED: %s", exc)
            return GateOutcome(
                passed=False, abstained=True,
                failures=[{"name": "gates_unavailable", "detail": str(exc)}],
                redacted_draft=_GATE_UNAVAILABLE_TEXT,
            )
    return _gate


def make_question_gate(question: str, sectioned: bool = False) -> Callable[..., GateOutcome]:
    """Public alias of `_make_gate` for the route layer: build a question-bound gate_fn so
    T2 completeness fires on the deep/draft/report paths (which inject a custom gate_fn and
    therefore bypass the loop's own `_make_gate(question)` default). `sectioned=True` →
    per-section gate (`gate_sectioned`); `sectioned=False` → whole-answer (`run_output_gates`)."""
    return _make_gate(question, sectioned=sectioned)


def _default_gate(draft: str, ledger: EvidenceLedger) -> GateOutcome:
    """Legacy no-question gate (used by tests that inject gate_fn directly).
    For the live loop, `_make_gate(question)` is used instead."""
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
    gate_fn: Optional[Callable[[str, EvidenceLedger], GateOutcome]] = None,
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

    # T2: thread the question into the gate so verify_completeness knows which entities
    # were asked for. Callers that inject a custom gate_fn keep their own behavior.
    if gate_fn is None:
        gate_fn = _make_gate(question)

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

    # ── T3: anti-loop circuit-breaker ────────────────────────────────────────────
    # Track (tool, normalized-args-hash) for calls that returned ok=False or 0 provenance.
    # On a repeat of a previously-dead call we inject a structured redirect and skip
    # execution — the model burned the budget by looping; this stops it.
    # On _REPEAT_CAP identical failed sigs, fire the hard block (redirect, no execute).
    _failed_sigs: Dict[str, int] = {}    # sig → count of failures on this sig
    _REPEAT_CAP = 2                       # hard ceiling: after 2 failures, block the 3rd

    # ── T5: budget-aware self-heal ────────────────────────────────────────────────
    # (a) dead-scope: tools whose query returned ok+0-prov (likely empty-ingested doc).
    # (b) consecutive model errors: 2+ in a row → degrade early rather than burning the wall.
    _dead_scopes: Set[str] = set()       # query-keys that already returned nothing
    _consecutive_model_errors: int = 0   # resets on any successful model response

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
            _consecutive_model_errors += 1
            yield {"type": "gate", "name": "model_error", "pass": False, "detail": str(exc)}
            final_text = None
            abstained = True
            # T5(b): an unrecoverable model error → emit a named t5_early_degrade gate so the
            # tracer/UI can distinguish "API died" from "budget ran out" or "gate redaction".
            # This is the "signal → action" promotion: the degrade is explicit + labelled, not
            # just a meta with degrade=True. Signal degrade to A4's route via the meta.
            yield {"type": "gate", "name": "t5_early_degrade", "pass": False,
                   "detail": f"model error at step {step} — stopping (consecutive={_consecutive_model_errors})"}
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
                # T3: compute a normalized signature for this call to detect repeats.
                _raw_sig = json.dumps({"n": call.name, "a": call.args}, sort_keys=True,
                                      default=str)
                _call_sig = hashlib.md5(_raw_sig.encode()).hexdigest()[:12]  # noqa: S324 — not security

                # T3: if this exact call already failed _REPEAT_CAP times, redirect instead
                # of executing again — the model is stuck in a loop and will burn the budget.
                if _failed_sigs.get(_call_sig, 0) >= _REPEAT_CAP:
                    logger.warning("[agent_core.loop] T3: repeated-failing call blocked "
                                   "(%s, sig=%s, count=%d)", call.name, _call_sig,
                                   _failed_sigs[_call_sig])
                    redirect_msg = (
                        f"[circuit-breaker] You already tried `{call.name}` with these same "
                        f"arguments {_failed_sigs[_call_sig]} time(s) and it has not returned "
                        f"useful results. Do NOT retry it again. Instead: (a) change the section, "
                        f"document, or tool, OR (b) state what you have verified so far and "
                        f"explicitly abstain on what you could not find."
                    )
                    yield {"type": "gate", "name": "t3_circuit_breaker", "pass": False,
                           "detail": f"repeated failing call blocked: {call.name} sig={_call_sig}"}
                    messages.append({"role": "user", "content": redirect_msg})
                    break  # inject the redirect and let the model respond to it

                yield {"type": "tool_call", "name": call.name,
                       "args_summary": _args_summary(call.args)}
                result = registry.execute(call, scope)
                ok = bool(result.get("ok"))
                n_prov = ledger.record(call.name, step, result.get("provenance"))
                yield {"type": "tool_result", "name": call.name,
                       "ok": ok, "summary": result.get("summary", ""),
                       "n_provenance": n_prov}
                messages.append(_tool_result_message(call, result))

                # T3: track failed/zero-prov calls for the circuit-breaker.
                if not ok or n_prov == 0:
                    _failed_sigs[_call_sig] = _failed_sigs.get(_call_sig, 0) + 1
                    # First repeat of a dead call: inject a structured redirect hint (softer
                    # than the hard block above — gives the model one chance to self-correct).
                    if _failed_sigs[_call_sig] == 2:
                        prior_result = result.get("summary") or result.get("error") or "no results"
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[tool-loop notice] You just repeated a `{call.name}` call that "
                                f"already returned: {prior_result!r:.200}. "
                                f"Repeating it will return the same result. "
                                f"Try a different query, section, or document; or abstain on this part."
                            ),
                        })

                # T5(a): if a scope-sensitive tool returned ok+0-provenance, the doc/scope
                # is likely empty-ingested. Remember the scope key and warn the model once.
                if ok and n_prov == 0 and call.name in ("search_vault", "read_document"):
                    scope_key = json.dumps(call.args.get("scope") or call.args.get("doc_id") or "",
                                           sort_keys=True, default=str)
                    if scope_key not in _dead_scopes:
                        _dead_scopes.add(scope_key)
                        logger.info("[agent_core.loop] T5: empty scope at step %d: %s args=%s",
                                    step, call.name, scope_key[:80])
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[empty-scope notice] `{call.name}` returned 0 results for this "
                                f"scope — the document may not be ingested or may be outside this "
                                f"vault. Stop probing this scope. Either try a different document, "
                                f"flag the gap explicitly in your answer, or move on."
                            ),
                        })
            else:
                continue  # all tool calls processed normally — let the model respond
            continue  # a break inside the for-loop (redirect injected) — also let model respond

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

    # F-F (tool_hard.md): the cross-vault-leak invariant on the hot path. Before rendering
    # ANY source, assert the ledger holds no span from a vault other than the active one.
    # A leaked span is dropped (never shown, never cited) and the run is flagged — the answer
    # path must never surface a chunk from outside `scope.collection_id`. Clean run ⇒ no-op
    # (byte-identical). Deterministic, $0, no model call.
    leaked = ledger.drop_foreign_spans(getattr(scope, "collection_id", None))
    if leaked:
        logger.error("[agent_core.loop] VAULT LEAK: dropped %d foreign-vault span(s) from the "
                     "ledger before rendering sources (active vault=%s)",
                     leaked, getattr(scope, "collection_id", None))
        yield {"type": "gate", "name": "vault_leak", "pass": False,
               "detail": f"dropped {leaked} span(s) from another vault — cross-vault leak blocked"}

    # Wrap up: emit sources from the ledger, the final answer, and run meta.
    yield {"type": "sources", "sources": ledger.to_sources()}
    if final_text is not None:
        yield {"type": "token", "text": final_text}
    yield {"type": "meta", "mode": budget.mode, "steps": budget.steps_used,
           "tokens": budget.tokens_used, "abstained": abstained,
           "n_evidence": len(ledger.entries)}
