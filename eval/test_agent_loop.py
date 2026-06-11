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

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ A2 loop gate GREEN (bridge · budget · tool-error · degrade · stream)")
        return 0
    print("  ✗ A2 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
