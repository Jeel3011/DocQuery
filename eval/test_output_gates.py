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
    run_output_gates, verify_numbers, verify_citations, verify_completeness,
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

    # BUG-D (live, 2026-06-11): a DERIVED compute result (growth %, delta) is not an
    # operand cell — the gate redacted a CORRECT computed "-0.5" because only cells
    # were ledgered. The compute adapter now ledgers the result (raw + display-rounded)
    # as `param` entries; the gate must trace stated figures to them.
    led_growth = ledger_with(value=198270.0)
    led_growth.record("compute", 2, [
        {"kind": "cell", "value": 168088.0, "doc": "msft-fy22", "page": 95,
         "label": "Total revenue", "period": "2021", "trace": "Total revenue [2021]"},
        {"kind": "param", "label": "result", "value": 17.956, "op": "growth_pct"},
        {"kind": "param", "label": "result_rounded", "value": 18.0, "op": "growth_pct"},
    ])
    r = verify_numbers("Total revenue grew 18.0% in FY2022 [msft-fy22 p.95].", led_growth)
    c.ok(r["pass"], "derived growth % traces via the ledgered compute result param")
    led_delta = ledger_with(value=198270.0)
    led_delta.record("compute", 2, [
        {"kind": "param", "label": "result", "value": 30182.0, "op": "delta"}])
    r = verify_numbers("Revenue increased by 30,182 [msft-fy22 p.95].", led_delta)
    c.ok(r["pass"], "derived delta traces via the ledgered compute result param")
    # …but an arbitrary figure still fails (params don't open a laundering hole).
    r = verify_numbers("Revenue increased by 47,500 [msft-fy22 p.95].", led_delta)
    c.ok(not r["pass"], "non-result figure still fails with params present")

    print("\n── verify_citations (marker presence) ───────────────────────────")
    r = verify_citations("Net sales were 513,983 [amzn-2022 p.41].", led)
    c.ok(r["pass"], "cited factual sentence passes")
    r = verify_citations("Net sales were 513,983 and margins improved sharply.", led)
    c.ok(not r["pass"], "uncited factual sentence fails")
    r = verify_citations("Here is a brief summary of what I found.", led)
    c.ok(r["pass"], "short connective sentence exempt (no citation needed)")

    # Draft scaffolding is NOT a factual claim — must not force a draft to over-abstain.
    # (Regression: 2026-06-21 an NDA skeleton was redacted because the gate flagged its
    # title "# Non-Disclosure Agreement" and its [INPUT NEEDED] brackets as uncited claims.)
    print("\n── verify_citations (draft scaffolding exempt) ──────────────────")
    r = verify_citations("# Non-Disclosure Agreement", led)
    c.ok(r["pass"], "a markdown document TITLE needs no citation")
    r = verify_citations("## 8. Governing law and dispute resolution", led)
    c.ok(r["pass"], "a markdown SECTION HEADING needs no citation")
    r = verify_citations('The "Disclosing Party" is [INPUT NEEDED: disclosing_party].', led)
    c.ok(r["pass"], "an [INPUT NEEDED] placeholder sentence needs no citation (honest gap)")
    r = verify_citations("_Insufficient evidence in the vault to draft this section._", led)
    c.ok(r["pass"], "the honest-withhold line needs no citation")
    # …but a real uncited claim — even alongside a placeholder — is STILL caught.
    r = verify_citations("The consideration is [INPUT NEEDED: amount] but the deposit was 50,000.", led)
    c.ok(not r["pass"], "a stated figure beside a placeholder is STILL flagged (moat intact)")

    # BUG-2 (2026-06-26): a QUALITATIVE, non-numeric factual answer grounded in the
    # retrieved document set must NOT be redacted for lacking a per-cell `[p.N]` marker.
    # The live repro: "What companies are in this vault?" gathered 38 spans across 7 docs,
    # then every sentence was redacted to "I could not verify…". The fix: a sentence with
    # NO substantive figure is exempt from the inline-marker rule WHEN the ledger holds
    # supporting evidence. The numeric moat stays intact.
    print("\n── verify_citations (qualitative answer exemption — BUG-2, T8-hardened) ──")
    # BUG-2's intent — a qualitative "what companies are in this vault?" answer ships — is preserved,
    # but T8 re-roots it on EVIDENCE: the live repro gathered 38 spans NAMING those companies, so the
    # fixture's ledger must actually CONTAIN them (the original fixture used an unrelated span and
    # leaned on the blanket flag — exactly the shape-exemption T8 replaces). Grounded → passes.
    led_q = EvidenceLedger()
    led_q.record("search_vault", 1, [
        {"kind": "span", "doc": "msa.pdf", "page": 1, "chunk_id": "c1",
         "snippet": "This agreement names Reliance, Tata, and Infosys as the contracting parties "
                    "and sets out the obligation of each party to the deal."},
    ])
    # The exemption is opt-in (allow_qualitative) — ONLY the simple direct-answer path enables it.
    r = verify_citations(
        "The agreement names Reliance, Tata, and Infosys as the contracting parties.", led_q,
        allow_qualitative=True)
    c.ok(r["pass"], "a non-numeric list-the-companies sentence passes when evidence NAMES them")
    r = verify_citations(
        "This section of the agreement covers the obligation of each party to the deal.", led_q,
        allow_qualitative=True)
    c.ok(r["pass"], "a qualitative 'what's in the vault' sentence passes with grounding evidence")
    # T8: GROUNDED prose now passes on the strict path TOO (marker drift no longer redacts it) —
    # because the evidence echoes it. (Pre-T8 this required a marker; T8 grounds it on the ledger.)
    r = verify_citations(
        "The agreement names Reliance, Tata, and Infosys as the contracting parties.", led_q)
    c.ok(r["pass"], "T8: the strict path SHIPS a span-grounded qualitative claim (no marker needed)")
    # …but a claim whose ENTITIES are absent from the evidence still fails on the strict path.
    r = verify_citations(
        "The agreement names Wipro and HCL as the contracting parties.", led_q)
    c.ok(not r["pass"], "the strict path STILL flags an UNGROUNDED qualitative claim (entities absent)")
    # …and the SAME qualitative sentence with NO evidence gathered still fails (ungrounded).
    r = verify_citations(
        "The agreement binds Reliance and Tata as the contracting party.",
        EvidenceLedger(), allow_qualitative=True)
    c.ok(not r["pass"], "a qualitative claim with an EMPTY ledger still fails (fail closed)")
    # …and a NUMERIC claim is NEVER exempted, evidence or not (the moat).
    r = verify_citations("Reliance reported total revenue of 875,000 last year.", led_q,
                         allow_qualitative=True)
    c.ok(not r["pass"], "a numeric sentence still REQUIRES a marker even with evidence (moat intact)")
    # End-to-end on the SIMPLE answer path: run_output_gates ships the qualitative answer whole.
    out = run_output_gates(
        "The agreement names Reliance, Tata, and Infosys as the contracting parties.", led_q)
    c.ok(out.passed, "run_output_gates ships an evidence-backed qualitative answer (BUG-2 live path)")

    # ── T8 corpus: re-root citation on the LEDGER, not on marker shape ───────────────
    # The principle (tool_hard.md §0.2/T8): a factual sentence is SUPPORTED if the ledger backs it —
    # NOT only if it carries a `[p.N]` substring. These fixtures encode the CLASS of live over-
    # redaction the plan documents (marker-drift on a grounded sentence) and pin that:
    #   (a) a grounded, marker-LESS sentence now passes on BOTH paths (the over-redaction fix + S-A);
    #   (b) the numeric moat is byte-identical; (c) the overlap bar rejects ungrounded/invented prose.
    print("\n── T8: evidence-grounded citation (marker-drift no longer redacts) ──")
    # A ledger whose SPANS verbatim contain the qualitative claim's salient content.
    led_g = EvidenceLedger()
    led_g.record("search_vault", 1, [
        {"kind": "span", "doc": "msa.pdf", "page": 2, "chunk_id": "c1",
         "snippet": "This Master Services Agreement is entered into between Acme Corporation "
                    "and Globex Industries as the contracting parties."},
        {"kind": "span", "doc": "msa.pdf", "page": 5, "chunk_id": "c2",
         "snippet": "The Agreement shall be governed by the laws of the State of Delaware and "
                    "disputes resolved by binding arbitration in Wilmington."},
    ])
    # (a) marker-DRIFT: a grounded sentence with NO [p.N] marker — was redacted to "I could not
    #     verify…"; T8 ships it because the spans echo it. STRICT path (no allow_qualitative).
    r = verify_citations(
        "The Agreement is between Acme Corporation and Globex Industries as the contracting parties.",
        led_g)
    c.ok(r["pass"], "T8: a span-GROUNDED sentence passes WITHOUT a marker, even on the strict path")
    r = verify_citations(
        "The Agreement is governed by the laws of the State of Delaware with disputes by arbitration.",
        led_g)
    c.ok(r["pass"], "T8: a second grounded clause (governing law) passes marker-less (span overlap)")
    # (a') run_output_gates ships such an answer whole — the live 'abstains every time' fix.
    out = run_output_gates(
        "The Agreement is between Acme Corporation and Globex Industries as the contracting parties.",
        led_g)
    c.ok(out.passed, "T8: run_output_gates SHIPS a grounded marker-less answer (no over-redaction)")
    # (b) MOAT intact: a NUMERIC claim is never grounded by span overlap — it still needs a marker
    #     AND a traced cell. Even though 'Acme'/'revenue' words appear, the figure has no cell.
    r = verify_citations("Acme Corporation reported revenue of 4,200,000 last fiscal year.", led_g)
    c.ok(not r["pass"], "T8: a numeric sentence still needs a marker (span overlap can't vouch a figure)")
    # (c) the overlap bar DISCRIMINATES: an invented claim whose entities are NOT in any span fails,
    #     even sharing generic content words ('agreement', 'parties', 'governed').
    r = verify_citations(
        "The Agreement is between Initech and Umbrella Corporation as the governed parties.", led_g)
    c.ok(not r["pass"], "T8: an UNGROUNDED claim (entities absent from spans) still fails (moat)")
    # (c') fail-closed on an EMPTY ledger — no spans means nothing can be grounded.
    r = verify_citations(
        "The Agreement is between Acme Corporation and Globex Industries.", EvidenceLedger())
    c.ok(not r["pass"], "T8: empty ledger → grounding impossible → still fails (fail closed)")
    # (d) gate_sectioned (deep/draft path) inherits the same evidence-grounding — a grounded report
    #     section ships marker-less (S-A), but an ungrounded one is still redacted.
    from src.components.agent_core.gates import gate_sectioned
    report_ok = ("## Parties\n\nThe Agreement is between Acme Corporation and Globex Industries "
                 "as the contracting parties.\n\n"
                 "## Governing law\n\nThe Agreement is governed by the laws of the State of Delaware.")
    out = gate_sectioned(report_ok, led_g)
    c.ok(out.passed, "T8/S-A: a grounded report ships marker-less per section (no marker-drift redaction)")
    report_bad = ("## Parties\n\nThe Agreement is between Acme Corporation and Globex Industries "
                  "as the contracting parties.\n\n"
                  "## Other parties\n\nThe Agreement also binds Initech and Umbrella Corporation "
                  "as additional contracting parties.")
    out = gate_sectioned(report_bad, led_g)
    c.ok(not out.passed and out.abstained,
         "T8/S-A: an UNGROUNDED report section is STILL redacted (deep-path moat intact)")
    c.ok("Initech" not in (out.redacted_draft or "") and "Acme" in (out.redacted_draft or ""),
         "T8/S-A: redaction drops the ungrounded section, keeps the grounded one")

    # ── S-A: single-section heading bypass (the latent moat hole tracked in §T8) ────────
    # A `## Heading\n\n<claim>` draft is ONE section, so gate_sectioned delegates to
    # run_output_gates. Before the fix, _SENT_SPLIT glued the heading onto the body sentence
    # and _is_factual rejected the blob as scaffold → an ungrounded claim under a lone heading
    # shipped UNCHECKED (verified live 2026-06-27). The fix strips the leading heading before
    # gating (mirroring the per-section path). These fixtures pin the hole closed.
    print("\n── S-A: single-section heading bypass (must not ship ungrounded) ──")
    out = run_output_gates(
        "## Governing Law\n\nThe agreement is governed by the laws of the planet Mars exclusively.",
        EvidenceLedger(), question="What is the governing law?")
    c.ok(not out.passed and out.abstained,
         "S-A: an ungrounded claim under a lone `## heading` is REDACTED, not shipped")
    out = run_output_gates(
        "## Parties\n\nThe agreement binds Initech and Umbrella Corporation as the sole "
        "contracting parties.", led_g, question="Who are the parties?")
    c.ok(not out.passed,
         "S-A: the entity gate fires THROUGH a lone heading (invented entities absent from spans)")
    # A GROUNDED claim under a heading still ships (no over-redaction regression), heading kept.
    out = run_output_gates(
        "## Parties\n\nThe Agreement is between Acme Corporation and Globex Industries as the "
        "contracting parties.", led_g, question="Who are the parties?")
    c.ok(out.passed, "S-A: a span-grounded claim under a heading still SHIPS (no false redaction)")
    # The honest withhold line under a heading is scaffold — it must still pass (not peeled).
    out = run_output_gates(
        "## Governing Law\n\n_Insufficient evidence in the vault to answer this._",
        EvidenceLedger(), question="What is the governing law?")
    c.ok(out.passed, "S-A: a heading + honest withhold line still passes (scaffold, no claim)")
    # The numeric moat still fires through a heading: a figure with no marker is redacted.
    out = run_output_gates(
        "## Revenue\n\nAcme reported revenue of 4,200,000 last fiscal year.", led_g,
        question="What was revenue?")
    c.ok(not out.passed, "S-A: numeric moat intact through a heading (untraced figure redacted)")

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
    # EMPTY ledger + invented figures: tracing is UNDECIDED (nothing to trace against) —
    # redaction must fail CLOSED and drop every figure, not pass them on absence of
    # evidence (the fail-open hole found in review: decided=False was treated as keep).
    out = run_output_gates(
        "Revenue was 999,999 [doc p.1]. Profit was 11,111 [doc p.1].", EvidenceLedger())
    c.ok(not out.passed, "empty ledger + figures → fails")
    rd = out.redacted_draft or ""
    c.ok("999,999" not in rd and "11,111" not in rd,
         "EMPTY-ledger redaction drops ALL invented figures (fail closed)")
    c.ok(out.abstained, "empty-ledger redaction flagged abstained")

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

    # ── T2: verify_completeness (multi-entity coverage gate) ──────────────────────
    print("\n── T2: verify_completeness (decompositional multi-entity gate) ──")

    # Single-entity question → no-op (never a false repair).
    r = verify_completeness("What was Amazon revenue in FY2022?",
                            "Amazon revenue in FY2022 was 513,983 [amzn-2022 p.41].")
    c.ok(r["pass"], "T2: single-entity question → always passes (no-op)")

    # Multi-entity "compare X, Y, Z" — all addressed → passes.
    answer_all = (
        "Amazon revenue in FY2022 was 513,983 [amzn-2022 p.41]. "
        "Google revenue was 282,836 [goog-2022 p.55]. "
        "Microsoft revenue was 198,270 [msft-fy22 p.95]."
    )
    r = verify_completeness("Compare Amazon, Google, and Microsoft revenue in FY2022.", answer_all)
    c.ok(r["pass"], "T2: compare question with all entities addressed → passes")

    # Multi-entity "compare X, Y, Z" — one entity silently dropped → fails.
    answer_drop_one = (
        "Amazon revenue in FY2022 was 513,983 [amzn-2022 p.41]. "
        "Google revenue was 282,836 [goog-2022 p.55]."
        # Microsoft silently omitted
    )
    r = verify_completeness("Compare Amazon, Google, and Microsoft revenue in FY2022.",
                            answer_drop_one)
    c.ok(not r["pass"], "T2: compare question with a silently-dropped entity → fails")
    c.ok("Microsoft" in str(r.get("dropped", [])),
         "T2: the dropped entity is named in the gate result")

    # Explicit abstain on one entity → passes (honest partial is correct).
    answer_explicit_abstain = (
        "Amazon revenue in FY2022 was 513,983 [amzn-2022 p.41]. "
        "Google revenue was 282,836 [goog-2022 p.55]. "
        "Microsoft data could not be verified from the vault."
    )
    r = verify_completeness("Compare Amazon, Google, and Microsoft revenue in FY2022.",
                            answer_explicit_abstain)
    c.ok(r["pass"],
         "T2: one entity explicitly abstained ('could not be verified') → passes (honest partial)")

    # "for each of … X, Y, Z" form — all addressed → passes.
    answer_clauses = (
        "Governing law: the agreement is governed by Delaware law. "
        "Payment terms: net-30 as stated in clause 5. "
        "Liability: capped at total fees paid in the prior 12 months."
    )
    r = verify_completeness(
        "For each of the following clauses: governing law, payment terms, liability.",
        answer_clauses)
    c.ok(r["pass"], "T2: 'for each of…' form with all clauses addressed → passes")

    # "for each of …" — one clause silently dropped → fails.
    answer_missing_clause = (
        "Governing law: the agreement is governed by Delaware law. "
        "Payment terms: net-30 as stated in clause 5."
        # liability silently dropped
    )
    r = verify_completeness(
        "For each of the following clauses: governing law, payment terms, liability.",
        answer_missing_clause)
    c.ok(not r["pass"],
         "T2: 'for each of…' with a silently-dropped clause → fails")
    c.ok("liability" in str(r.get("dropped", [])).lower(),
         "T2: the dropped clause is named in the gate result")

    # run_output_gates wires completeness — a dropped entity surfaces as a gate failure.
    out = run_output_gates(
        answer_drop_one, led,
        question="Compare Amazon, Google, and Microsoft revenue in FY2022.")
    comp_fail = any(f.get("name") == "verify_completeness" for f in (out.failures or []))
    c.ok(not out.passed and comp_fail,
         "T2: run_output_gates fails when a multi-entity answer drops an entity")

    # run_output_gates passes when all entities are addressed.
    out = run_output_gates(
        answer_all, led,
        question="Compare Amazon, Google, and Microsoft revenue in FY2022.")
    # NOTE: answer_all has no ledger-traced figures but no numeric claims on the simple
    # path would cause issues — the verify_numbers gate may fail on untraced figures here.
    # Focus the T2 test: completeness itself should NOT be the failure reason.
    comp_fail2 = any(f.get("name") == "verify_completeness" for f in (out.failures or []))
    c.ok(not comp_fail2,
         "T2: run_output_gates completeness gate passes when all entities are addressed")

    # Unrecognized question form → single-entity no-op → never a false repair.
    r = verify_completeness("What is the net revenue trend?", "Revenue has been growing steadily.")
    c.ok(r["pass"], "T2: unrecognized question form → no-op (safe default, no false repair)")

    # ── T2 on the DEEP/DRAFT/REPORT path: make_question_gate threads the question ──────
    # The deep/draft/report routes inject a CUSTOM gate_fn, so they bypass the loop's own
    # _make_gate(question) default. `make_question_gate(q, sectioned=True)` is what threads the
    # question into gate_sectioned — without it the per-section gate runs with question="" and
    # verify_completeness silently no-ops on exactly the multi-entity reports that need it.
    print("\n── T2: question threaded into the deep/draft gate (make_question_gate) ──")
    from src.components.agent_core.loop import make_question_gate
    led_ms = EvidenceLedger()
    led_ms.record("search_vault", 1, [
        {"kind": "span", "doc": "amzn.pdf", "page": 41, "chunk_id": "a",
         "snippet": "Amazon net sales were 513,983 in fiscal 2022."},
        {"kind": "span", "doc": "goog.pdf", "page": 55, "chunk_id": "g",
         "snippet": "Google revenue was 282,836 in fiscal 2022."},
    ])
    q_ms = "Compare Amazon, Google, and Microsoft revenue in FY2022."
    report_drop_ms = ("## Amazon\n\nAmazon revenue in FY2022 was 513,983 [amzn p.41].\n\n"
                      "## Google\n\nGoogle revenue was 282,836 [goog p.55].")  # Microsoft dropped
    # sectioned=True (deep/draft/report): the gate is callable as gate_fn(draft, ledger) and
    # MUST now flag the silently-dropped entity (this was a no-op before the question was threaded).
    g_deep = make_question_gate(q_ms, sectioned=True)
    out = g_deep(report_drop_ms, led_ms)
    comp_deep = any(f.get("name") == "verify_completeness" for f in (out.failures or []))
    c.ok(not out.passed and comp_deep,
         "T2/deep: make_question_gate(sectioned=True) fires completeness on a dropped entity")
    c.ok("Microsoft" in str([f.get("dropped") for f in (out.failures or [])
                             if f.get("name") == "verify_completeness"]),
         "T2/deep: the dropped entity is named in the sectioned gate result")
    # The whole-answer variant (sectioned=False) threads the question too — parity check.
    g_std = make_question_gate(q_ms, sectioned=False)
    out = g_std("Amazon was 513,983 [amzn p.41]. Google was 282,836 [goog p.55].", led_ms)
    c.ok(any(f.get("name") == "verify_completeness" for f in (out.failures or [])),
         "T2/std: make_question_gate(sectioned=False) also threads the question")
    # A complete report (all three addressed, one explicitly abstained) → completeness PASSES.
    report_ok_ms = ("## Amazon\n\nAmazon revenue in FY2022 was 513,983 [amzn p.41].\n\n"
                    "## Google\n\nGoogle revenue was 282,836 [goog p.55].\n\n"
                    "## Microsoft\n\nMicrosoft revenue could not be verified from the vault.")
    out = g_deep(report_ok_ms, led_ms)
    c.ok(not any(f.get("name") == "verify_completeness" for f in (out.failures or [])),
         "T2/deep: a report addressing all entities (one explicit abstain) passes completeness")

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
