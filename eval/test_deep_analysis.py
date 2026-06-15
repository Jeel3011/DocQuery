"""Deep Analysis report-gate — GRAND_PLAN §G5 / G5 plan §2.B + §5.

The moat for the riskiest surface: a long, confident report is exactly where a
hallucinated number or an off-point clause hides best. This gate PROVES, fully offline
and ZERO API spend, that a Deep Analysis report is bound to the evidence ledger
SECTION-BY-SECTION:

  1. A grounded section (every claim cites a real span/cell) PASSES with its citations
     intact.
  2. A section with an UNCITED factual claim is REDACTED to a visible withhold line —
     the unsupported claim never ships, and the redaction is VISIBLE, not a silent drop.
  3. A section the model intentionally withheld ("_Insufficient evidence…_") passes
     through verbatim (honest abstention is a result, not a defect).
  4. Sections are independent: an uncited claim in ONE section does not redact a
     grounded NEIGHBOR.
  5. End-to-end through `run_agent` (deep mode + a ScriptedModel + gate_sectioned): a
     report whose one section is unsupported ships with that section redacted and the
     grounded sections + their citations intact.
  6. The deep system prompt carries the breadth-first, outline, and per-section-abstain
     instructions (the steering that makes the model emit a gateable report).

"Real grids": the cited figure in the ledger is a genuinely-extracted cell value from
the finance corpus (re-extracted deterministically, the test_tools.py offline pattern) —
so the gate is proven against a real number, never an invented one. The MODEL is scripted
(network-free); the GRIDS/cells are real. No DB, no API.

Run: python -u eval/test_deep_analysis.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.agent_core.ledger import EvidenceLedger
from src.components.agent_core.gates import gate_sectioned, _split_sections
from src.components.agent_core.prompt import system_prompt
from src.components.agent_core.budgets import Budget
from src.components.agent_core.loop import run_agent
from src.components.agent_core.model import ModelResponse, ToolCall, ScriptedModel
from src.components.agent_core.registry import RunScope


# ── A real extracted cell from the finance corpus → the ledger value the report cites ──
def _real_amzn_net_sales_2022() -> float:
    """Pull a genuine extracted figure so the gate is proven against a REAL number.

    Falls back to the known value (513,983 — AMZN FY22 total net sales) if the corpus
    isn't present, so the gate still runs in a bare checkout. The point is that when the
    corpus IS there, the cited figure is the extractor's own output, not a literal."""
    KNOWN = 513983.0
    try:
        from src.components.table_extraction import extract_tables_from_pdf
        from src.components.brain.analyst import Grid, compute
        grids = [Grid(t.to_metadata(), doc="amzn-20221231.pdf", page=t.page_number)
                 for t in extract_tables_from_pdf("test docs/amzn-20221231.pdf")]
        res = compute({"op": "value", "metric": {"label": "Total net sales"},
                       "period": "2022"}, grids)
        v = getattr(res, "value", None)
        if v:
            return float(v)
    except Exception as exc:  # noqa: BLE001 — corpus optional; the gate is the point
        print(f"  (note: using known AMZN value; corpus extract skipped: {exc})")
    return KNOWN


def ledger_with_real_cells():
    """A ledger as a clean deep run accrues it: one real computed cell + one span.

    The report may cite the figure (513,983) and the governing-law clause; anything else
    a section asserts is UNGROUNDED and must be redacted."""
    val = _real_amzn_net_sales_2022()
    led = EvidenceLedger()
    led.record("compute", 1, [{"kind": "cell", "value": val, "doc": "amzn-20221231.pdf",
                               "page": 41, "label": "Total net sales", "period": "2022",
                               "raw": f"{val:,.0f}",
                               "trace": f"Total net sales [2022] = {val:,.0f}"}])
    led.record("survey_collection", 1, [{"kind": "span", "doc": "msa.pdf", "page": 12,
                                         "chunk_id": "c1",
                                         "snippet": "This Agreement is governed by the laws of India."}])
    return led, val


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
    led, val = ledger_with_real_cells()
    val_str = f"{val:,.0f}"

    print("\n── the deep system prompt steers a GATEABLE report ──────────────")
    dp = system_prompt(mode="deep")
    base = system_prompt(mode="standard")
    c.ok(dp != base and base in dp,
         "deep prompt EXTENDS the shared base contract (one engine, same cite-or-abstain)")
    c.ok("survey_collection" in dp and "outline" in dp.lower(),
         "deep prompt: start with survey_collection, propose an outline")
    c.ok("Insufficient evidence" in dp and "##" in dp,
         "deep prompt: mark a section 'Insufficient evidence' (not pad) + use ## sections")
    c.ok("standard" != "deep" and "survey_collection" not in base,
         "standard prompt does NOT carry the deep overlay (deep is additive)")

    print("\n── gate_sectioned: section splitting ────────────────────────────")
    report = (
        "Executive summary: revenue is strong [amzn-20221231.pdf p.41].\n\n"
        "## Revenue\nTotal net sales were " + val_str + " in FY2022 [amzn-20221231.pdf p.41].\n\n"
        "## Governing Law\nThe agreement is governed by the laws of India [msa.pdf p.12].\n"
    )
    secs = _split_sections(report)
    c.ok(len(secs) == 3, "split: preamble + 2 ## sections = 3 gateable sections")
    c.ok(secs[1].lstrip().startswith("## Revenue") and secs[2].lstrip().startswith("## Governing Law"),
         "split: each ## header starts its own section")

    print("\n── a fully-grounded report PASSES (all sections cite real evidence) ──")
    out = gate_sectioned(report, led)
    c.ok(out.passed, "grounded report passes the per-section gate")

    print("\n── an UNCITED section is redacted; grounded neighbors survive ───")
    mixed = (
        "## Revenue\nTotal net sales were " + val_str + " in FY2022 [amzn-20221231.pdf p.41].\n\n"
        "## Liability\nThe supplier's liability is capped at three months of fees and the "
        "indemnity is mutual.\n\n"  # NO citation — must be redacted
        "## Governing Law\nThe agreement is governed by the laws of India [msa.pdf p.12].\n"
    )
    out = gate_sectioned(mixed, led)
    c.ok(not out.passed and out.abstained, "mixed report fails → abstained=True")
    rd = out.redacted_draft or ""
    c.ok("liability is capped at three months" not in rd,
         "the uncited Liability claim is REMOVED from the report")
    c.ok("could not verify" in rd.lower() or "removed" in rd.lower(),
         "the redaction is VISIBLE (an explicit withhold line, not a silent drop)")
    c.ok(val_str in rd and "[amzn-20221231.pdf p.41]" in rd,
         "the grounded Revenue section SURVIVES intact with its citation (independence)")
    c.ok("laws of India" in rd and "[msa.pdf p.12]" in rd,
         "the grounded Governing-Law section SURVIVES intact (neighbor not collateral-damaged)")
    c.ok("## Revenue" in rd and "## Liability" in rd and "## Governing Law" in rd,
         "the report KEEPS its section structure (withheld section visible in place)")

    print("\n── an UNTRACED figure in a section is redacted too ──────────────")
    badnum = (
        "## Revenue\nTotal net sales were " + val_str + " [amzn-20221231.pdf p.41].\n\n"
        "## Operating Income\nOperating income was 12,248 in FY2022 [amzn-20221231.pdf p.41].\n"
        # 12,248 is NOT in the ledger → an untraced figure, must be redacted
    )
    out = gate_sectioned(badnum, led)
    rd = out.redacted_draft or ""
    c.ok(not out.passed, "section with an untraced figure fails the gate")
    c.ok("12,248" not in rd, "the untraced figure 12,248 is removed")
    c.ok(val_str in rd, "the traced figure in the grounded section survives")

    print("\n── an intentionally-WITHHELD section passes verbatim ────────────")
    withheld = (
        "## Revenue\nTotal net sales were " + val_str + " [amzn-20221231.pdf p.41].\n\n"
        "## Environmental Liabilities\n_Insufficient evidence in the vault to report on this._\n"
    )
    out = gate_sectioned(withheld, led)
    c.ok(out.passed, "a report that honestly abstains a section PASSES (abstention is success)")
    c.ok("Insufficient evidence" in (out.redacted_draft or withheld),
         "the withhold line is preserved (honest abstention shown, not rewritten)")

    print("\n── a single-section draft → identical to the whole-draft gate ───")
    one = "Net sales were " + val_str + " [amzn-20221231.pdf p.41]."
    out = gate_sectioned(one, led)
    c.ok(out.passed, "no ## headers → behaves like run_output_gates (grounded passes)")
    out = gate_sectioned("Net sales were 12,248.", led)  # untraced, uncited
    c.ok(not out.passed, "no ## headers → an ungrounded single section still fails")

    print("\n── gate_sectioned never raises on odd input ─────────────────────")
    for odd in ("", "   ", "##\n##\n", "## Only A Header\n", None):
        try:
            gate_sectioned(odd, led)  # type: ignore[arg-type]
            c.ok(True, f"no raise on {odd!r}")
        except Exception as exc:  # noqa: BLE001
            c.ok(False, f"RAISED on {odd!r}: {exc}")

    print("\n── END-TO-END: run_agent in deep mode + gate_sectioned ──────────")
    # Scripted model + REAL grids (the plan's literal §5 row). The loop's OWN ledger is
    # built from the tool results the scripted run actually produces — so for a grounded
    # section to survive, the cited evidence must REALLY be in the ledger (the gate doesn't
    # trust the prose, it trusts the ledger). The run therefore:
    #   1. surveys the vault (breadth-first), then
    #   2. computes the real AMZN net-sales cell (513,983) — populates a real ledger cell,
    #   3. searches the vault for the governing-law clause — populates a real span,
    #   4. emits a 3-section report whose middle section is UNSUPPORTED by the ledger.
    # The gate fails → one repair turn → the model still can't ground it → the loop's redact
    # path ships the report with the bad section withheld and the grounded sections intact.
    amzn_grids = []
    try:
        from src.components.table_extraction import extract_tables_from_pdf
        from src.components.brain.analyst import Grid
        amzn_grids = [Grid(t.to_metadata(), doc="amzn-20221231.pdf", page=t.page_number)
                      for t in extract_tables_from_pdf("test docs/amzn-20221231.pdf")]
    except Exception as exc:  # noqa: BLE001 — corpus optional
        print(f"  (note: AMZN corpus unavailable, e2e uses a seeded cell: {exc})")

    class _ClauseRM:
        """Returns the governing-law clause for search_vault → a real ledger span."""
        def retrieve(self, query, *a, **k):
            class _D:
                page_content = "This Agreement is governed by the laws of India."
                metadata = {"filename": "msa.pdf", "page_number": 12, "chunk_id": "c1",
                            "doc_id": "id-msa"}
            return [_D()]
        def retrieve_table_chunks(self, *a, **k):
            return []

    e2e_report = (
        "Executive summary: the vault covers one filing and one contract "
        "[amzn-20221231.pdf p.41].\n\n"
        "## Revenue\nTotal net sales were " + val_str + " in FY2022 [amzn-20221231.pdf p.41].\n\n"
        "## Risk Factors\nThe company faces severe undisclosed litigation risk that could "
        "halve its revenue next year.\n\n"  # uncited, unsupported — must be redacted
        "## Governing Law\nThe agreement is governed by the laws of India [msa.pdf p.12].\n"
    )
    survey_call = ToolCall(id="t1", name="survey_collection",
                           args={"query": "key terms and financials"})
    compute_call = ToolCall(id="t2", name="compute",
                            args={"op": "value", "row": {"label": "Total net sales"},
                                  "period": "2022"})
    search_call = ToolCall(id="t3", name="search_vault",
                           args={"query": "governing law", "kind": "text"})
    script = [
        ModelResponse(text="Surveying the vault.", tool_calls=[survey_call]),
        ModelResponse(text="Computing revenue.", tool_calls=[compute_call]),
        ModelResponse(text="Finding the governing-law clause.", tool_calls=[search_call]),
        ModelResponse(text=e2e_report, tool_calls=[]),   # first draft → gate fails → repair
        ModelResponse(text=e2e_report, tool_calls=[]),   # repaired draft still bad → redact
    ]
    model = ScriptedModel(script)

    scope = RunScope(collection_id="c1", filenames=["amzn-20221231.pdf"],
                     filename_by_doc={"amzn-20221231.pdf": "amzn-20221231.pdf"},
                     grids=amzn_grids, retrieval_manager=_ClauseRM(), config=object())
    budget = Budget(mode="deep", model="scripted", max_steps=8, wall_clock_s=60,
                    token_budget=100_000)

    events = list(run_agent("Analyze this vault.", model=model, scope=scope, budget=budget,
                            system_prompt=system_prompt(mode="deep"),
                            gate_fn=gate_sectioned))
    # The loop emits the final answer as a `token` event carrying the (redacted) report.
    final = next((e for e in events if e.get("type") == "token" and e.get("text")), None)
    answer = (final or {}).get("text", "") if final else ""
    c.ok(final is not None, "deep run reaches a final answer")
    c.ok("halve its revenue" not in answer,
         "end-to-end: the unsupported Risk-Factors claim is redacted from the report")
    if amzn_grids:  # the Revenue section can only survive if the real cell entered the ledger
        c.ok(val_str in answer,
             "end-to-end: the grounded Revenue section ships with its real computed figure")
    c.ok("laws of India" in answer,
         "end-to-end: the grounded Governing-Law section ships with its cited clause")
    c.ok(any(e.get("type") == "tool_call" and e.get("name") == "survey_collection"
             for e in events),
         "end-to-end: the deep run opened with survey_collection (breadth-first)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ Deep Analysis report-gate GREEN (per-section cite-or-withhold)")
        return 0
    print("  ✗ Deep Analysis gate FAILED")
    return 1


class _StubRM:
    """Minimal retrieval manager so survey_collection runs without a DB (returns empty)."""
    def retrieve(self, *a, **k):
        return []
    def retrieve_table_chunks(self, *a, **k):
        return []


if __name__ == "__main__":
    raise SystemExit(main())
