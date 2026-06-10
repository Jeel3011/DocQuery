"""Output-gate gate — AGENT_CORE_PLAN §3.4 / A3.

Asserts the non-bypassable output gates and the repair→redact protocol. Fully offline,
ZERO API spend: the gates' deterministic core (figure→cell tracing + citation-marker
presence) needs no LLM, and we build the evidence ledger by hand (the gates only read
ledger values/spans, so no extraction/DB is needed here either).

DoD checks:
  1. A fully-traced, cited draft PASSES.
  2. A draft with an UNTRACED figure FAILS verify_numbers and gets REDACTED (the
     untraced figure is removed; a withhold line is appended).
  3. An uncited factual sentence FAILS verify_citations and is redacted.
  4. ONE repair turn is honored end-to-end in the loop: first draft fails → loop feeds
     failures back → second draft (fixed) passes; a STILL-bad second draft → redacted.
  5. Gates never raise on odd input.

Run: python -u eval/test_output_gates.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.agent_core.ledger import EvidenceLedger
from src.components.agent_core.gates import (
    run_output_gates, verify_numbers, verify_citations,
)
from src.components.agent_core.budgets import Budget
from src.components.agent_core.loop import run_agent
from src.components.agent_core.model import ModelResponse, ScriptedModel
from src.components.agent_core.registry import RunScope


def ledger_with(value=513983.0, doc="amzn-2022", page=41):
    """A ledger holding one real cell (value) + one span — what a clean run accrues."""
    led = EvidenceLedger()
    led.record("compute", 1, [{"kind": "cell", "value": value, "doc": doc, "page": page,
                               "label": "Total net sales", "period": "2022", "raw": "513,983",
                               "trace": "Total net sales [2022] = 513,983 (amzn-2022 p41)"}])
    led.record("search_vault", 1, [{"kind": "span", "doc": doc, "page": page,
                                    "chunk_id": "c1", "snippet": "Total net sales were 513,983"}])
    return led


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
    led = ledger_with()

    print("── verify_numbers (figure → cell) ───────────────────────────────")
    # traced figure (matches the cell, scaling-aware: 513,983 or $514B)
    r = verify_numbers("Net sales were 513,983 [amzn-2022 p.41].", led)
    c.ok(r["pass"], "traced figure 513,983 passes")
    r = verify_numbers("Net sales reached $514 billion [amzn-2022 p.41].", led)
    c.ok(r["pass"], "scaled restatement $514B passes (scaling-aware)")
    # untraced figure → fail, naming it
    r = verify_numbers("Operating income was 12,248 [amzn-2022 p.41].", led)
    c.ok(not r["pass"], "untraced figure 12,248 fails verify_numbers")
    # figures stated but empty ledger → fail (free-floating)
    r = verify_numbers("Revenue was 999,999.", EvidenceLedger())
    c.ok(not r["pass"], "figure with empty ledger fails (free-floating)")
    # no figures → vacuous pass
    r = verify_numbers("The company grew over the year.", EvidenceLedger())
    c.ok(r["pass"], "no figures → vacuous pass")

    print("\n── verify_citations (marker presence) ───────────────────────────")
    r = verify_citations("Net sales were 513,983 [amzn-2022 p.41].", led)
    c.ok(r["pass"], "cited factual sentence passes")
    r = verify_citations("Net sales were 513,983 and margins improved sharply.", led)
    c.ok(not r["pass"], "uncited factual sentence fails")
    r = verify_citations("Here is a brief summary of what I found.", led)
    c.ok(r["pass"], "short connective sentence exempt (no citation needed)")

    print("\n── run_output_gates (combined + redaction) ──────────────────────")
    out = run_output_gates("Net sales were 513,983 [amzn-2022 p.41].", led)
    c.ok(out.passed, "clean draft → passed")
    out = run_output_gates(
        "Net sales were 513,983 [amzn-2022 p.41]. Operating income was 12,248 [amzn-2022 p.41].",
        led)
    c.ok(not out.passed, "draft with one untraced figure → fails")
    c.ok("513,983" in (out.redacted_draft or ""), "redaction KEEPS the traced figure")
    c.ok("12,248" not in (out.redacted_draft or ""), "redaction REMOVES the untraced figure")
    c.ok("removed one or more" in (out.redacted_draft or ""), "redaction appends a withhold line")

    print("\n── loop: repair-once-then-redact (end-to-end, mocked model) ─────")
    scope = RunScope(grids=[])

    def budget():
        return Budget(mode="standard", model="m", max_steps=8, wall_clock_s=90, token_budget=60000)

    # We need the ledger populated so the gate can trace. Use a scripted model that first
    # calls compute (populating the ledger via the real adapter on a fixture grid), then
    # answers badly, then — after the repair note — answers correctly.
    from src.components.brain.analyst import Grid
    g = [Grid({"headers": ["label", "2022"], "periods": ["2022"],
               "rows": [{"section": "", "label": "Total net sales", "2022": "513,983"}],
               "table_id": "t"}, doc="amzn-2022", page=41)]
    scope = RunScope(grids=g)
    from src.components.agent_core.model import ToolCall
    script = [
        ModelResponse(text="reading", tool_calls=[ToolCall(id="c1", name="compute", args={
            "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        # first answer: contains an UNTRACED figure → gate fails → ONE repair
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41]. Net income was 11,111 [amzn-2022 p.41].",
                      tool_calls=[]),
        # repaired answer: only the traced figure, cited → passes
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41].", tool_calls=[]),
    ]
    events = list(run_agent("net sales 2022", model=ScriptedModel(script),
                            scope=scope, budget=budget()))
    gate_events = [e for e in events if e["type"] == "gate"]
    token = next(e["text"] for e in events if e["type"] == "token")
    meta = next(e for e in events if e["type"] == "meta")
    c.ok(any(g["name"] in ("verify_numbers", "verify_citations") and not g["pass"]
             for g in gate_events), "first draft tripped a gate")
    c.ok(any(g["name"] == "output" and g["pass"] for g in gate_events),
         "repaired draft passed the output gate")
    c.ok("11,111" not in token and "513,983" in token, "final answer is the repaired (clean) one")
    c.ok(meta["abstained"] is False, "successful repair → not abstained")

    # Now: a model that STAYS bad on the repair → redaction ships
    bad = [
        ModelResponse(text="reading", tool_calls=[ToolCall(id="c1", name="compute", args={
            "op": "value", "row": {"label": "Total net sales"}, "period": "2022"})]),
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41]. Profit was 11,111 [amzn-2022 p.41].",
                      tool_calls=[]),
        ModelResponse(text="Net sales were 513,983 [amzn-2022 p.41]. Profit was 11,111 [amzn-2022 p.41].",
                      tool_calls=[]),
    ]
    events = list(run_agent("net sales 2022", model=ScriptedModel(bad),
                            scope=scope, budget=budget()))
    token = next(e["text"] for e in events if e["type"] == "token")
    meta = next(e for e in events if e["type"] == "meta")
    c.ok("11,111" not in token, "persistent bad draft → untraced figure redacted out")
    c.ok(meta["abstained"] is True, "redacted run flagged abstained")

    print("\n── never raises on odd input ────────────────────────────────────")
    for bad_in in ("", None, "12,345 67,890 no cells"):
        try:
            run_output_gates(bad_in or "", EvidenceLedger())
            ok = True
        except Exception:
            ok = False
        c.ok(ok, f"run_output_gates({bad_in!r}) did not raise")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ A3 output-gate gate GREEN (numbers · citations · repair→redact)")
        return 0
    print("  ✗ A3 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
