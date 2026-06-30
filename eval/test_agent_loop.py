"""Agent-loop gate — AGENT_CORE_PLAN §3.2 / A2.

Drives `run_agent` with a MOCKED model (ScriptedModel) so it runs offline with ZERO
API spend and no network. Asserts the A2 DoD:

  1. BRIDGE RESOLVES with ledger provenance — a scripted tool sequence (first_exceeds
     → 2022, then value → 513,983) ends in a final answer; the evidence ledger holds
     the real CellRefs the kernel produced (the loop wired provenance end-to-end).
  2. BUDGET EXHAUSTION → graceful partial — a scripted model that only ever asks for
     tools never gets to finish; the loop stops at max_steps with an abstain meta, no
     crash, no overspend.
  3. TOOL ERROR → the model ADAPTS — a bad tool call returns an error envelope as a
     RESULT (not a raise); the next scripted step uses a good call and the run still
     completes.
  4. MODEL ERROR → DEGRADE signal — a model whose invoke raises yields a meta with
     degrade=True (A4's route falls back to Brain), never an exception to the consumer.
  5. EVENT SHAPE — the stream contains the §3.6 event types in a sane order.

The kernel/grounding correctness is covered by their own gates; here we use a small
focused grid fixture so the scripted calls are deterministic and the test isolates the
LOOP mechanics (ledger, budgets, degradation, event stream).

Run: python -u eval/test_agent_loop.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.brain.analyst import Grid
from src.components.agent_core.budgets import Budget
from src.components.agent_core.loop import run_agent
from src.components.agent_core.model import ModelResponse, ScriptedModel, ToolCall
from src.components.agent_core.registry import RunScope


# ── A small, unambiguous fixture grid (AWS bridge figures) ──────────────────────
_TJ = {
    "headers": ["label", "2021", "2022", "2023"],
    "periods": ["2021", "2022", "2023"],
    "rows": [{"section": "", "label": "Total net sales",
              "2021": "469,822", "2022": "513,983", "2023": "574,785"}],
    "table_id": "amzn-cons",
}
GRIDS = [Grid(_TJ, doc="amzn-2022", page=41)]


def std_budget(max_steps=12):
    return Budget(mode="standard", model="claude-opus-4-8",
                  max_steps=max_steps, wall_clock_s=90, token_budget=60000)


def collect(gen):
    return list(gen)


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


def main() -> int:
    c = Check()
    scope = RunScope(grids=GRIDS)

    # ── 1. Bridge resolves with ledger provenance ───────────────────────────────
    print("── bridge resolves (first_exceeds → 2022 → value 513,983) ───────")
    script = [
        # step 1: model asks which year first exceeds $500B
        ModelResponse(text="Let me find the pivot year.", tool_calls=[
            ToolCall(id="c1", name="compute", args={
                "op": "first_exceeds", "row": {"label": "Total net sales"},
                "threshold": 500000, "periods": ["2021", "2022", "2023"]})]),
        # step 2: model reads the value for that year
        ModelResponse(text="Now the value for 2022.", tool_calls=[
            ToolCall(id="c2", name="compute", args={
                "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        # step 3: final answer (no tools) → gate stub passes
        ModelResponse(text="Amazon first exceeded $500B in net sales in 2022 (513,983) [amzn-2022 p.41].",
                      tool_calls=[]),
    ]
    model = ScriptedModel(script)
    events = collect(run_agent("When did Amazon first exceed $500B net sales?",
                               model=model, scope=scope, budget=std_budget()))
    types = [e["type"] for e in events]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    meta = next(e for e in events if e["type"] == "meta")
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    token_ev = [e for e in events if e["type"] == "token"]

    c.ok(all(t["ok"] for t in tool_results) and len(tool_results) == 2,
         "two tool calls, both ok")
    c.ok(any("513,983" in (e.get("text", "")) for e in token_ev),
         "final answer carries the bridge figure 513,983")
    c.ok(meta["abstained"] is False, "meta: not abstained (clean resolution)")
    c.ok(len(sources) >= 1 and any(s.get("value") == 513983.0 for s in sources),
         "evidence ledger holds the real CellRef (513,983)")
    c.ok(meta["n_evidence"] >= 2, "ledger recorded provenance from both tool calls")

    # ── 2. Budget exhaustion → graceful partial ─────────────────────────────────
    print("\n── budget exhaustion → graceful partial (no crash) ──────────────")
    # A model that ALWAYS asks for a tool, never finishing.
    loopy = ScriptedModel([
        (lambda m: ModelResponse(text="thinking", tool_calls=[
            ToolCall(id=f"x", name="compute", args={"op": "value",
                     "row": {"label": "Total net sales"}, "period": "2022"})]))
        for _ in range(50)
    ])
    events = collect(run_agent("loop forever", model=loopy, scope=scope,
                               budget=std_budget(max_steps=4)))
    meta = next(e for e in events if e["type"] == "meta")
    budget_gate = [e for e in events if e["type"] == "gate" and e["name"] == "budget"]
    c.ok(meta["steps"] <= 4, "stopped at max_steps (no overspend)")
    c.ok(meta["abstained"] is True, "budget-exhausted run abstains")
    c.ok(len(budget_gate) == 1 and budget_gate[0]["pass"] is False, "emitted a failed budget gate")
    # Budget AWARENESS (the harness gap): before the wall, the model must receive a
    # [budget notice] telling it to wrap up — so "found it at step 7, guillotined at
    # step 9" becomes "found it, delivered it". Assert the nudge reached the model.
    all_msgs = [m for callrec in loopy.calls for m in callrec["messages"]]
    notice_seen = any("[budget notice]" in str(m.get("content", "")) for m in all_msgs)
    c.ok(notice_seen, "model received a [budget notice] before the wall (budget awareness)")
    # PARTIAL ANSWER on budget exhaustion (2026-06-11): the verified figures in the
    # ledger must SHIP, not be discarded for an empty "ran out of budget" (live: it had
    # computed Google + Amazon ratios then showed nothing). The scripted compute resolved
    # 'Total net sales'[2022] = 513,983, so the final text must carry that verified value.
    final_tok = "".join(e.get("text", "") for e in events if e["type"] == "token")
    c.ok("513,983" in final_tok or "513983" in final_tok,
         "budget-exhausted run SHOWS verified figures (not an empty abstain)"
         + (f" [got: {final_tok[:80]!r}]" if "513" not in final_tok else ""))

    # ── 3. Tool error → the model adapts ─────────────────────────────────────────
    print("\n── tool error → adapt (error is a RESULT, not a raise) ──────────")
    adapt = ScriptedModel([
        ModelResponse(text="bad call", tool_calls=[
            ToolCall(id="b1", name="compute", args={"op": "not_a_real_op", "row": {}})]),
        ModelResponse(text="recover", tool_calls=[
            ToolCall(id="g1", name="compute", args={
                "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        ModelResponse(text="Net sales 2022 were 513,983 [amzn-2022 p.41].", tool_calls=[]),
    ])
    events = collect(run_agent("net sales 2022", model=adapt, scope=scope, budget=std_budget()))
    tool_results = [e for e in events if e["type"] == "tool_result"]
    meta = next(e for e in events if e["type"] == "meta")
    c.ok(tool_results[0]["ok"] is False and tool_results[1]["ok"] is True,
         "first tool errored, second succeeded — loop kept going")
    c.ok(any(e["type"] == "token" for e in events) and meta["abstained"] is False,
         "run completed after adapting to the tool error")

    # ── 4. Model error → degrade signal ──────────────────────────────────────────
    print("\n── model error → degrade signal (never raises to consumer) ──────")
    class _Boom:
        def invoke(self, messages, tools):
            raise RuntimeError("vendor 529")
    events = collect(run_agent("x", model=_Boom(), scope=scope, budget=std_budget()))
    meta = next(e for e in events if e["type"] == "meta")
    c.ok(meta.get("degrade") is True, "model error → meta degrade=True")
    c.ok(any(e["type"] == "gate" and e["name"] == "model_error" for e in events),
         "emitted a model_error gate")

    # ── 5. Event shape / ordering ────────────────────────────────────────────────
    print("\n── event stream shape ───────────────────────────────────────────")
    events = collect(run_agent("net sales 2022", model=ScriptedModel([
        ModelResponse(text="answer 513,983 [amzn-2022 p.41]", tool_calls=[])]),
        scope=scope, budget=std_budget()))
    types = [e["type"] for e in events]
    c.ok(types[0] == "agent_step", "stream starts with agent_step")
    c.ok(types[-1] == "meta", "stream ends with meta")
    c.ok("sources" in types and "token" in types, "stream has sources + token")

    # ── 6. Gate failure → fail CLOSED (non-bypassable means withhold, not passthrough)
    print("\n── raising gate_fn → fail CLOSED (withhold ships, no crash) ─────")
    def _boom_gate(draft, ledger):
        raise RuntimeError("gates exploded")
    events = collect(run_agent("net sales 2022", model=ScriptedModel([
        ModelResponse(text="Invented figure 123,456 with no evidence.", tool_calls=[]),
        ModelResponse(text="Still invented: 123,456.", tool_calls=[]),
    ]), scope=scope, budget=std_budget(), gate_fn=_boom_gate))
    token = next((e["text"] for e in events if e["type"] == "token"), "")
    meta = next(e for e in events if e["type"] == "meta")
    c.ok("123,456" not in token, "gate crash → invented figure does NOT ship")
    c.ok(meta["abstained"] is True, "gate crash → run abstains (fail closed)")
    c.ok(any(e["type"] == "gate" and e["name"] == "gates_unavailable" for e in events),
         "emitted a gates_unavailable gate event")

    # ── 7. Wall-clock budget enforced ────────────────────────────────────────────
    print("\n── wall-clock budget → graceful abstain ─────────────────────────")
    wall0 = Budget(mode="standard", model="m", max_steps=12, wall_clock_s=1e-9,
                   token_budget=60000)
    events = collect(run_agent("x", model=ScriptedModel([
        ModelResponse(text="never reached", tool_calls=[])]), scope=scope, budget=wall0))
    meta = next(e for e in events if e["type"] == "meta")
    c.ok(meta["steps"] == 0 and meta["abstained"] is True,
         "wall-clock exhausted before step 1 → abstain, zero model calls")
    c.ok(any(e["type"] == "gate" and e["name"] == "budget" and "wall-clock" in e["detail"]
             for e in events), "budget gate names wall-clock as the cause")

    # ── 8. Transient model error → ONE retry covers it (no degrade) ──────────────
    print("\n── transient model error → retry-once recovers ──────────────────")
    def _raise_once(_messages):
        raise RuntimeError("transient 529")
    events = collect(run_agent("net sales 2022", model=ScriptedModel([
        _raise_once,  # first invoke raises → loop retries → next item proceeds
        ModelResponse(text="reading", tool_calls=[ToolCall(id="r1", name="compute", args={
            "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41].", tool_calls=[]),
    ]), scope=scope, budget=std_budget()))
    meta = next(e for e in events if e["type"] == "meta")
    c.ok(meta.get("degrade") is not True and meta["abstained"] is False,
         "single transient model error recovered by the retry (no degrade)")
    c.ok(any("513,983" in e.get("text", "") for e in events if e["type"] == "token"),
         "answer shipped after the retry")

    # ── 9. Conversation memory (G5): prior turns precede the current question ─────
    print("\n── conversation memory: prior turns threaded into the run ───────")
    import copy as _copy

    class _CapturingModel(ScriptedModel):
        """Snapshots a DEEP COPY of the messages on the FIRST invoke (so the assertion
        isn't fooled by the loop later appending the model's own turn to the same list)."""
        def __init__(self, script):
            super().__init__(script)
            self.first_messages = None
        def invoke(self, messages, tools):
            if self.first_messages is None:
                self.first_messages = _copy.deepcopy(messages)
            return super().invoke(messages, tools)

    history = [
        {"role": "user", "content": "What is the governing law?"},
        {"role": "assistant", "content": "It is governed by the laws of India [msa.pdf p.12]."},
    ]
    cap = _CapturingModel([ModelResponse(
        text="The termination notice period is 30 days [msa.pdf p.5].", tool_calls=[])])
    events = collect(run_agent(
        "And the termination notice period?", model=cap, scope=RunScope(grids=GRIDS),
        budget=std_budget(), history=history))
    seen = [(m.get("role"), m.get("content")) for m in (cap.first_messages or [])
            if isinstance(m.get("content"), str)]
    c.ok(any("governing law" in (txt or "").lower() for _, txt in seen)
         and any("laws of India" in (txt or "") for _, txt in seen),
         "memory: BOTH prior turns are present in the model's context")
    # The current question is appended AFTER the history (last string message at invoke #1).
    c.ok(seen and seen[-1][0] == "user" and "termination" in (seen[-1][1] or "").lower(),
         "memory: the current question follows the prior turns (correct order)")
    c.ok(any("30 days" in e.get("text", "") for e in events if e["type"] == "token"),
         "memory: the follow-up answer still ships through the gate")
    # No history → the run still works (memory is additive, never required).
    cap2 = _CapturingModel([ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41].", tool_calls=[])])
    events = collect(run_agent("net sales 2022", model=cap2, scope=RunScope(grids=GRIDS),
                               budget=std_budget()))  # history defaults to None
    only_user = [m for m in (cap2.first_messages or []) if isinstance(m.get("content"), str)]
    c.ok(len(only_user) == 1 and "net sales" in (only_user[0].get("content") or "").lower(),
         "memory: no history → only the current question (additive, not required)")

    # ── 10. T3: anti-loop circuit-breaker ────────────────────────────────────────
    print("\n── T3: anti-loop circuit-breaker ───────────────────────────────────")

    def _failing_compute_tc(idx=0):
        """A compute call that returns ok=False (unsupported op). Each idx gets a unique id."""
        return ToolCall(id=f"bad{idx}", name="compute",
                        args={"op": "definitely_not_an_op", "row": {"label": "x"}})

    # A model that repeats the SAME failing compute 3 times then delivers.
    # _REPEAT_CAP=2: after 2 failures the 3rd attempt is BLOCKED (redirect injected).
    t3_script = [
        # attempt 1: failing compute (count → 1)
        ModelResponse(text="try1", tool_calls=[_failing_compute_tc(1)]),
        # attempt 2: same call (count → 2, soft hint injected + REPEAT_CAP reached)
        ModelResponse(text="try2", tool_calls=[_failing_compute_tc(1)]),
        # attempt 3: same call → BLOCKED before execution, redirect injected
        ModelResponse(text="try3", tool_calls=[_failing_compute_tc(1)]),
        # after redirect: deliver without numbers (no tracing needed)
        ModelResponse(text="I could not verify the figure. The data was not available.",
                      tool_calls=[]),
    ]
    t3_events = collect(run_agent("bad op loop", model=ScriptedModel(t3_script),
                                  scope=scope, budget=std_budget()))
    t3_gate_names = [e.get("name") for e in t3_events if e["type"] == "gate"]
    c.ok("t3_circuit_breaker" in t3_gate_names,
         "T3: t3_circuit_breaker gate fires when identical failing call hits the repeat cap")
    t3_tokens = [e for e in t3_events if e["type"] == "token"]
    c.ok(len(t3_tokens) == 1,
         "T3: run completes with a final answer after the redirect (no hang)")
    c.ok("not available" in t3_tokens[0]["text"],
         "T3: final answer is the model's redirect-response (not empty abstain)")

    # Two DIFFERENT failing calls (different sigs) must NOT trip the breaker.
    diff_script = [
        ModelResponse(text="t1", tool_calls=[
            ToolCall(id="d1", name="compute",
                     args={"op": "definitely_not_an_op", "row": {"label": "x"}})]),
        ModelResponse(text="t2", tool_calls=[
            ToolCall(id="d2", name="compute",
                     args={"op": "another_bad_op", "row": {"label": "y"}})]),
        ModelResponse(text="No data found.", tool_calls=[]),
    ]
    diff_events = collect(run_agent("diff sig test", model=ScriptedModel(diff_script),
                                    scope=scope, budget=std_budget()))
    diff_gates = [e.get("name") for e in diff_events if e["type"] == "gate"]
    c.ok("t3_circuit_breaker" not in diff_gates,
         "T3: different-sig failing calls do NOT trigger the breaker (sig-specific)")

    # A model that calls a failing tool TWICE, then changes to a good call — no block.
    # count=2 triggers the soft hint; on the 3rd call the model uses the good op.
    soft_script = [
        ModelResponse(text="t1", tool_calls=[_failing_compute_tc(2)]),
        ModelResponse(text="t2", tool_calls=[_failing_compute_tc(2)]),
        # Soft hint received — switches to a working call
        ModelResponse(text="t3", tool_calls=[
            ToolCall(id="g1", name="compute",
                     args={"op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41].", tool_calls=[]),
    ]
    soft_events = collect(run_agent("soft hint", model=ScriptedModel(soft_script),
                                    scope=scope, budget=std_budget()))
    soft_gates = [e.get("name") for e in soft_events if e["type"] == "gate"]
    c.ok("t3_circuit_breaker" not in soft_gates,
         "T3: 2 repeats (soft hint) + change does NOT trigger the hard block")
    soft_tokens = [e for e in soft_events if e["type"] == "token"]
    c.ok(soft_tokens and "513,983" in soft_tokens[0]["text"],
         "T3: run delivers correctly after soft-hint + model change (no over-block)")

    # ── 10b. T3 protocol: a BLOCKED call in a MULTI-tool-call turn still gets a ────
    # tool_result, so the next invoke doesn't 400 ("tool_call_ids without responses")
    # and degrade to a BLANK answer. This is the live Phase-2 blank-answer bug: the
    # model emitted two tool_calls in one turn, one tripped the repeat cap, the loop
    # `break`-ed and left the sibling's tool_call_id unanswered → 400 → empty answer.
    print("\n── T3 protocol: multi-tool-call turn answers EVERY tool_call_id ──────")

    class _ProtocolModel(ScriptedModel):
        """Snapshots the messages on EACH invoke so we can assert every assistant
        tool_use block has a matching tool_result before the next request."""
        def __init__(self, script):
            super().__init__(script)
            self.snapshots = []
        def invoke(self, messages, tools):
            self.snapshots.append(_copy.deepcopy(messages))
            return super().invoke(messages, tools)

    # Prime the repeat counter on a failing compute (idx 7), then in a LATER turn issue
    # TWO calls at once: the same blocked compute + a sibling good lookup. The blocked one
    # must still receive a synthetic tool_result so the good sibling's turn is protocol-valid.
    multi_script = [
        ModelResponse(text="p1", tool_calls=[_failing_compute_tc(7)]),   # count→1
        ModelResponse(text="p2", tool_calls=[_failing_compute_tc(7)]),   # count→2 (cap reached)
        # one assistant turn, TWO tool calls: blocked compute + a real lookup
        ModelResponse(text="p3", tool_calls=[
            _failing_compute_tc(7),  # BLOCKED — must still get a tool_result
            ToolCall(id="good9", name="compute",
                     args={"op": "value", "row": {"label": "Total net sales"},
                           "period": "2022"})]),
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41].", tool_calls=[]),
    ]
    pm = _ProtocolModel(multi_script)
    multi_events = collect(run_agent("multi tool block", model=pm,
                                     scope=scope, budget=std_budget()))

    def _tool_use_ids(msg):
        ids = []
        if isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    ids.append(b.get("id"))
        return ids

    def _tool_result_ids(msg):
        ids = []
        if isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    ids.append(b.get("tool_use_id"))
        return ids

    # The LAST snapshot (the delivery invoke) must contain a tool_result for EVERY tool_use
    # id ever declared — including the blocked one. That's the protocol invariant the 400
    # enforced; if it holds, no 400, no blank.
    final_msgs = pm.snapshots[-1] if pm.snapshots else []
    declared, answered = set(), set()
    for m in final_msgs:
        declared.update(_tool_use_ids(m))
        answered.update(_tool_result_ids(m))
    c.ok(declared and declared <= answered,
         "T3 protocol: every declared tool_use id (incl. the blocked call) has a tool_result")
    multi_tokens = [e for e in multi_events if e["type"] == "token"]
    c.ok(multi_tokens and multi_tokens[0]["text"].strip(),
         "T3 protocol: run delivers a NON-EMPTY answer (no 400-degrade blank)")
    c.ok(multi_tokens and "513,983" in multi_tokens[0]["text"],
         "T3 protocol: the good sibling call still resolves and is delivered")

    # ── 11. T5(a): empty-scope notice ─────────────────────────────────────────────
    print("\n── T5(a): empty-scope notice ───────────────────────────────────────")

    # search_vault with a stub retrieval_manager that returns [] → ok=True, 0 spans → T5(a).
    class _EmptyRM:
        def retrieve(self, *a, **kw): return []
        def retrieve_table_chunks(self, *a, **kw): return []

    t5a_scope = RunScope(
        grids=GRIDS,
        collection_id="vault-1",
        retrieval_manager=_EmptyRM(),
    )
    t5a_script = [
        # search_vault returns ok+0-prov → T5(a) fires (empty-scope notice injected)
        ModelResponse(text="searching", tool_calls=[
            ToolCall(id="sv1", name="search_vault",
                     args={"query": "net sales", "scope": {"collection_id": "vault-1"}})]),
        ModelResponse(text="No results found in this vault.", tool_calls=[]),
    ]
    t5a_events = collect(run_agent("empty vault search", model=ScriptedModel(t5a_script),
                                   scope=t5a_scope, budget=std_budget()))
    t5a_sv_results = [e for e in t5a_events if e["type"] == "tool_result"
                      and e["name"] == "search_vault"]
    c.ok(t5a_sv_results and t5a_sv_results[0]["ok"] is True
         and t5a_sv_results[0]["n_provenance"] == 0,
         "T5(a): search_vault with empty RM → ok=True, n_prov=0 (triggers empty-scope path)")
    c.ok(any(e["type"] == "token" for e in t5a_events),
         "T5(a): loop still completes after empty-scope notice (no crash)")

    # Second search for the SAME scope key must NOT inject a second notice (dedup).
    t5a_dedup_script = [
        ModelResponse(text="s1", tool_calls=[
            ToolCall(id="sv2", name="search_vault",
                     args={"query": "governing law", "scope": {"collection_id": "vault-1"}})]),
        ModelResponse(text="s2", tool_calls=[
            ToolCall(id="sv3", name="search_vault",
                     args={"query": "governing law", "scope": {"collection_id": "vault-1"}})]),
        ModelResponse(text="Done.", tool_calls=[]),
    ]
    t5a_dedup_events = collect(run_agent("dedup", model=ScriptedModel(t5a_dedup_script),
                                          scope=t5a_scope, budget=std_budget()))
    c.ok(any(e["type"] == "token" for e in t5a_dedup_events),
         "T5(a): dedup — run completes without crash on repeated empty scope")

    # ── 12. T5(b): model error → named early-degrade gate ────────────────────────
    print("\n── T5(b): model error → named t5_early_degrade gate ────────────────")

    # A model whose stream AND invoke both raise → outer except fires → t5_early_degrade.
    class _AlwaysErrors:
        """Raises on every stream/invoke call."""
        def stream(self, messages, schemas):
            raise RuntimeError("simulated API error")
        def invoke(self, messages, schemas):
            raise RuntimeError("simulated API error — retry also fails")

    t5b_events = list(run_agent("t5b test", model=_AlwaysErrors(),
                                scope=scope, budget=std_budget()))
    t5b_gates = [e.get("name") for e in t5b_events if e["type"] == "gate"]
    t5b_meta = [e for e in t5b_events if e["type"] == "meta"]
    c.ok("t5_early_degrade" in t5b_gates,
         "T5(b): t5_early_degrade gate fires on an unrecoverable model error")
    c.ok(t5b_meta and any(e.get("degrade") for e in t5b_meta),
         "T5(b): meta carries degrade=True (A4 falls back to Brain)")
    c.ok(t5b_meta and any(e.get("abstained") for e in t5b_meta),
         "T5(b): meta carries abstained=True (run did not produce an answer)")

    # A transient error followed by a successful retry must NOT emit t5_early_degrade.
    # stream() raises → invoke() succeeds (the retry path).
    class _TransientThenOk:
        def __init__(self): self._stream_called = 0
        def stream(self, messages, schemas):
            self._stream_called += 1
            if self._stream_called == 1:
                raise RuntimeError("transient stream failure")
            # Second stream call succeeds.
            yield ("done", ModelResponse(text="Net sales 513,983 [amzn-2022 p.41].", tool_calls=[]))
        def invoke(self, messages, schemas):
            # invoke is the retry after stream failure — must succeed.
            return ModelResponse(text="Net sales 513,983 [amzn-2022 p.41].", tool_calls=[])

    trans_events = list(run_agent("transient", model=_TransientThenOk(),
                                  scope=scope, budget=std_budget()))
    trans_gates = [e.get("name") for e in trans_events if e["type"] == "gate"]
    c.ok("t5_early_degrade" not in trans_gates,
         "T5(b): stream-error-then-invoke-success (one retry) does NOT emit t5_early_degrade")

    # ── 6. Phase 3.1: loop compaction (the anti-quit spine — DOCUMENT_HARNESS §7.1) ──
    # A run that fills the window mid-task: OFF ⇒ dies at the token wall (abstain); ON ⇒
    # compacts the verbose history and CONTINUES to a clean, verified answer. The ledger
    # is NEVER summarized, so the verified figure survives compaction (gate stays sound).
    import os as _os

    _BIG = "padding " * 300  # ~2.4k chars per turn → the history grows fast toward the wall

    from src.components.agent_core.model import BaseModel as _BaseModel

    class _WallModel(_BaseModel):
        """Drives tokens toward the wall with big per-step usage + bulky assistant text.
        A compaction call (tools == []) returns a short summary and is FREE of the script
        cursor, so it never desyncs the tool sequence. Normal steps: 3 tool calls, then the
        final answer. Each normal step reports ~350 input tokens of usage."""
        def __init__(self):
            self._i = 0
            self._script = [
                ModelResponse(text="Reading the table. " + _BIG, tool_calls=[
                    ToolCall(id="w1", name="compute", args={
                        "op": "value", "row": {"label": "Total net sales"}, "period": "2021"})],
                    usage={"in": 350, "out": 0}),
                ModelResponse(text="Still reading. " + _BIG, tool_calls=[
                    ToolCall(id="w2", name="compute", args={
                        "op": "value", "row": {"label": "Total net sales"}, "period": "2023"})],
                    usage={"in": 350, "out": 0}),
                ModelResponse(text="One more. " + _BIG, tool_calls=[
                    ToolCall(id="w3", name="compute", args={
                        "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})],
                    usage={"in": 350, "out": 0}),
                ModelResponse(
                    text="Amazon net sales were 513,983 in 2022 [amzn-2022 p.41].",
                    tool_calls=[], usage={"in": 50, "out": 30}),
            ]

        def invoke(self, messages, tools):
            # Compaction summary request: no tools, ends with the compaction instruction.
            if not tools:
                return ModelResponse(
                    text=("QUESTION: net sales 2022. VERIFIED FINDINGS: Total net sales (2022): "
                          "513,983 [amzn-2022 p.41]. OPEN SUBGOALS: deliver the answer."),
                    tool_calls=[], usage={"in": 40, "out": 20})
            item = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
            return item

    # token_budget=1000, frac=0.8 → compaction trigger at 800; each step adds ~350 → wall ~step 3.
    def _wall_budget():
        return Budget(mode="standard", model="claude-opus-4-8",
                      max_steps=20, wall_clock_s=90, token_budget=1000)

    print("\n── 3.1 compaction OFF: run dies at the token wall (abstain) ─────")
    _os.environ.pop("USE_LOOP_COMPACTION", None)  # OFF
    off_events = collect(run_agent("net sales 2022", model=_WallModel(),
                                   scope=scope, budget=_wall_budget()))
    off_meta = next(e for e in off_events if e["type"] == "meta")
    off_compaction = [e for e in off_events if e["type"] == "gate" and e["name"] == "compaction"]
    off_budget = [e for e in off_events if e["type"] == "gate" and e["name"] == "budget"]
    c.ok(len(off_compaction) == 0, "OFF: no compaction gate ever fires (flag-OFF identity)")
    c.ok(len(off_budget) == 1 and off_meta["abstained"] is True,
         "OFF: run abstains at the token wall (the false-abstain Phase 3.1 fixes)")

    print("\n── 3.1 compaction ON: run survives the wall and completes ───────")
    _os.environ["USE_LOOP_COMPACTION"] = "true"
    _os.environ["COMPACT_AT_FRACTION"] = "0.8"
    _os.environ["COMPACT_KEEP_LAST_K"] = "2"
    try:
        on_events = collect(run_agent("net sales 2022", model=_WallModel(),
                                      scope=scope, budget=_wall_budget()))
    finally:
        _os.environ.pop("USE_LOOP_COMPACTION", None)
        _os.environ.pop("COMPACT_AT_FRACTION", None)
        _os.environ.pop("COMPACT_KEEP_LAST_K", None)
    on_meta = next(e for e in on_events if e["type"] == "meta")
    on_compaction = [e for e in on_events if e["type"] == "gate" and e["name"] == "compaction"]
    on_final = "".join(e.get("text", "") for e in on_events if e["type"] == "token")
    c.ok(len(on_compaction) >= 1 and on_compaction[0]["pass"] is True,
         "ON: a compaction gate fired (summarized + continued instead of abstaining)")
    c.ok(on_meta["abstained"] is False,
         "ON: the run COMPLETES — wall-death converted to a real answer")
    c.ok("513,983" in on_final,
         "ON: the verified figure 513,983 SURVIVES compaction and ships (ledger never lost)")

    print("\n── 3.1 compaction-correctness: the ledger survives compaction ───")
    # The verified cell (513,983) must be in the run's sources AFTER a compaction happened —
    # proving the EvidenceLedger was NOT summarized away.
    on_sources = next((e for e in on_events if e["type"] == "sources"), {"sources": []})["sources"]
    c.ok(any(s.get("value") == 513983.0 for s in on_sources),
         "ledger holds the real CellRef (513,983) post-compaction — never summarized")

    print("\n── 3.1 boundary safety: _safe_tail never orphans a tool_result ──")
    from src.components.agent_core.loop import _safe_tail, _is_tool_result_msg
    _msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    # keep_last_k=2 would start at the bare tool_result (index 2) → must shift forward off it.
    _tail = _safe_tail(_msgs, 2)
    c.ok(not _is_tool_result_msg(_tail[0]) if _tail else True,
         "_safe_tail does not start the kept tail with an orphaned tool_result")
    c.ok(_safe_tail(_msgs, 99) == _msgs, "_safe_tail keeps everything when K ≥ len(messages)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ A2+T3+T5+3.1 loop gate GREEN (bridge · budget · tool-error · degrade · stream · memory · circuit-breaker · self-heal · compaction)")
        return 0
    print("  ✗ A2+T3+T5+3.1 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
