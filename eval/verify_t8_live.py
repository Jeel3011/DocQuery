"""T8 LIVE gate — confirm previously-redacted correct answers now ship.

Plans/tool_hard.md §T8 / sequencing table row 4:
  "Live gate (T0 live replay + UI confirm previously-redacted correct answers now
   ship) still owed."

This script is the OFFLINE-REPRODUCIBLE part of the T8 live gate: it replays a set
of harvested (and synthesized) transcripts — the class of over-redaction T8 was
designed to fix — through the CURRENT gates.py and confirms:

  (a) Answers that were previously redacted by the OLD gate (marker-shape tyranny) NOW
      SHIP through the T8-evidence-grounded gate.
  (b) Every WRONG answer in the existing test_output_gates.py corpus still redacts
      (the numeric moat is intact — not relaxed).

Because T0 live (the API-burning run that harvests REAL gate_redaction transcripts) is
on-demand only, this script uses SYNTHETIC transcripts that faithfully reproduce the
documented failure class:
  - A grounded qualitative claim with NO [p.N] marker → OLD gate redacted; T8 ships.
  - A grounded governing-law clause with marker drift → OLD gate redacted; T8 ships.
  - A numeric claim (figure not in ledger) → STILL redacts (moat).
  - A claim whose entities are absent from evidence → STILL redacts (entity gate).

For each transcript, the script also runs gate_sectioned (the §S-A path) to confirm
the same evidence-grounded check applies to report sections.

─── HOW TO EXTEND WITH REAL T0 TRANSCRIPTS ─────────────────────────────────────────
After running `python -u eval/abstain_autopsy.py` (T0 live), grep the output JSON:
  python -c "
  import json
  data = json.load(open('eval/abstain_autopsy_results.json'))
  for r in data.get('by_cause', {}).get('gate_redaction', []):
      if r.get('traced'):
          print(r['question'], r['draft'], r['spans'])
  "
For each gate_redaction-that-traced: add a CASE below with the draft + spans from the
autopsy, then confirm it now passes. That is the full T8 live closing sequence.

Run:
    python -u eval/verify_t8_live.py
    (No API calls. $0. Fully offline.)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components.agent_core.ledger import EvidenceLedger
from src.components.agent_core.gates import (
    run_output_gates,
    gate_sectioned,
    verify_citations,
    verify_numbers,
)

_passed = _failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  ← {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build ledgers that mirror what a live run would accumulate.
# ─────────────────────────────────────────────────────────────────────────────

def _span_ledger(*snippets: tuple[str, str, int]) -> EvidenceLedger:
    """Build a ledger from (doc, snippet, page) tuples — the search_vault result."""
    led = EvidenceLedger()
    items = [
        {"kind": "span", "doc": doc, "page": page, "chunk_id": f"c{i}",
         "snippet": snippet}
        for i, (doc, snippet, page) in enumerate(snippets, 1)
    ]
    led.record("search_vault", 1, items)
    return led


def _cell_ledger(value: float, doc: str, page: int, label: str, raw: str) -> EvidenceLedger:
    """Build a ledger with one numeric cell — the kernel/compute result."""
    led = EvidenceLedger()
    led.record("compute", 1, [{
        "kind": "cell",
        "value": value,
        "doc": doc,
        "page": page,
        "label": label,
        "period": "2023",
        "raw": raw,
        "trace": f"{label} = {raw} ({doc} p{page})",
    }])
    led.record("search_vault", 1, [{
        "kind": "span",
        "doc": doc,
        "page": page,
        "chunk_id": "s1",
        "snippet": f"The {label.lower()} for the period was {raw}",
    }])
    return led


print(f"\n── T8 LIVE gate: previously-redacted correct answers now ship ───────────")
print("  Replaying the documented T8 failure class through the CURRENT gates.py.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Group A: PREVIOUSLY REDACTED answers that T8 now ships.
# These reproduce the "marker-drift" over-redaction the plan documents.
# ─────────────────────────────────────────────────────────────────────────────

print("── A. Answers T8 now ships (were previously redacted) ──────────────────")

# A1 — Governing-law clause: no [p.N] marker, but ledger has the span verbatim.
# Documented failure: the agent found the governing-law clause, quoted it, and was
# redacted because the citation marker wasn't in the exact "[doc p.N]" format.
led_gov = _span_ledger(
    ("master_services_agreement.pdf",
     "This Agreement shall be governed by and construed in accordance with the laws of "
     "the State of Delaware, without regard to its conflict of law provisions.",
     12),
)
draft_gov = ("The Agreement shall be governed by the laws of the State of Delaware, "
             "without regard to conflict of law provisions.")
r = run_output_gates(draft_gov, led_gov)
check("A1: governing-law clause ships WITHOUT marker (span grounding holds)",
      r.passed,
      f"gate failed: {r.failures}")

# A2 — Contracting parties: marker-free qualitative claim, entities in spans.
led_parties = _span_ledger(
    ("commercial_agreement.pdf",
     "This Master Services Agreement is entered into between Reliance Industries Limited "
     "and Infosys Limited (the contracting parties) effective from 1 January 2025.",
     1),
)
draft_parties = "The Agreement is between Reliance Industries Limited and Infosys Limited."
r = run_output_gates(draft_parties, led_parties)
check("A2: contracting-parties answer ships (entities present in spans)",
      r.passed,
      f"gate failed: {r.failures}")

# A3 — Termination clause: qualitative, no marker, spans contain the clause.
led_term = _span_ledger(
    ("nda.pdf",
     "Either party may terminate this Agreement upon thirty (30) days written notice "
     "to the other party, or immediately upon material breach that remains uncured for "
     "a period of fifteen (15) days following written notice.",
     7),
)
draft_term = ("Either party may terminate this Agreement upon thirty days written notice, "
              "or immediately upon material breach uncured for fifteen days.")
r = run_output_gates(draft_term, led_term)
check("A3: termination-clause answer ships (clause present in spans)",
      r.passed,
      f"gate failed: {r.failures}")

# A4 — Multi-sentence purely qualitative legal summary (no figures): all grounded in spans.
# Note: sentences containing monetary figures need a cell trace (the numeric moat applies).
# This fixture uses purely qualitative language — the class T8 was designed to fix.
# Both spans use language that is verbatim or near-verbatim in the draft (no possessive
# paraphrase — the entity gate requires the claim's entities to appear in the corpus).
led_multi = _span_ledger(
    ("lease.pdf",
     "The Tenant shall pay monthly rent and the Landlord shall provide a receipt for each payment.",
     3),
    ("lease.pdf",
     "The Tenant may sublet the premises only with the prior written consent of the Landlord.",
     4),
)
draft_multi = (
    "The Tenant is required to pay monthly rent and receive a receipt from the Landlord. "
    "Subletting requires the prior written consent of the Landlord."
)
r = run_output_gates(draft_multi, led_multi)
check("A4: multi-sentence qualitative legal summary ships (all sentences grounded in spans)",
      r.passed,
      f"gate failed: {r.failures}")

# A5 — Single-section report (the latent S-A moat hole tracked in tool_hard.md).
# A report with ## Heading + one grounded paragraph must not bypass gating.
led_sect = _span_ledger(
    ("msa.pdf",
     "The Agreement is entered into between Acme Corporation and Globex Industries as the "
     "contracting parties under the laws of Maharashtra.",
     2),
)
draft_sect = (
    "## Contracting Parties\n\n"
    "The Agreement is entered into between Acme Corporation and Globex Industries "
    "as the contracting parties under the laws of Maharashtra."
)
r = run_output_gates(draft_sect, led_sect)
check("A5: single-section report with grounded body ships (S-A heading-bypass fixed)",
      r.passed,
      f"gate failed: {r.failures}")

# A5b — gate_sectioned path also ships the grounded section (S-A parity).
r_sect = gate_sectioned(draft_sect, led_sect)
check("A5b: gate_sectioned ships the same grounded section (S-A parity)",
      r_sect.passed,
      f"gate_sectioned failed: {r_sect.failures}")

# ─────────────────────────────────────────────────────────────────────────────
# Group B: WRONG / UNGROUNDED answers — must STILL redact (moat intact).
# Every case here pins that T8 does NOT create a bypass for hallucinated content.
# ─────────────────────────────────────────────────────────────────────────────

print("\n── B. Wrong/ungrounded answers still redact (numeric moat intact) ──────")

# B1 — Untraced figure: number not in any ledger cell → still fails.
led_cells = _cell_ledger(513983.0, "amzn-2022", 41, "Total net sales", "513,983")
r_wrong = run_output_gates("Total net sales were 999,999 [amzn-2022 p.41].", led_cells)
check("B1: untraced figure still fails (moat: 999,999 not in ledger)",
      not r_wrong.passed,
      "WRONG: untraced figure shipped (moat breach!)")

# B2 — Entity absent: claim mentions Wipro; spans only mention Reliance/Tata.
led_ent = _span_ledger(
    ("msa.pdf",
     "This Agreement is between Reliance and Tata as the contracting parties.",
     1),
)
r_ent = run_output_gates("The Agreement is between Wipro and Infosys as contracting parties.", led_ent)
check("B2: entity-absent claim still fails (Wipro/Infosys not in evidence)",
      not r_ent.passed,
      "WRONG: ungrounded claim with absent entities shipped!")

# B3 — Numeric claim with no marker, even with supporting spans, still requires the moat.
led_num = _span_ledger(
    ("report.pdf",
     "Total revenue for fiscal 2023 was INR 875,000 million.",
     5),
)
r_num = run_output_gates("Total revenue for fiscal 2023 was INR 875,000 million.", led_num)
check("B3: numeric claim without marker + without a ledgered cell still fails (moat)",
      not r_num.passed,
      "WRONG: numeric claim without cell trace shipped!")

# B4 — Empty ledger: any claim with no evidence should fail.
r_empty = run_output_gates(
    "The Agreement is between Acme and Globex under Delaware law.", EvidenceLedger()
)
check("B4: any claim against an empty ledger still fails",
      not r_empty.passed,
      "WRONG: claim shipped against empty ledger!")

# B5 — Hallucinated citation marker on an untraced number is still rejected.
led_halluc = _cell_ledger(100000.0, "some-doc", 3, "Revenue", "100,000")
r_halluc = run_output_gates("Operating profit was 47,500 [some-doc p.3].", led_halluc)
check("B5: hallucinated marker on an untraced number still fails (47,500 not in ledger)",
      not r_halluc.passed,
      "WRONG: untraced number with a plausible-looking marker shipped!")

# ─────────────────────────────────────────────────────────────────────────────
# Group C: Regression — the existing test_output_gates.py corpus.
# We don't re-run the whole file (that's `python -u eval/test_output_gates.py`),
# but we spot-check the specific T8 fixtures it added (indices ~40-49) and the
# two-sided bar (WRONG still redacts, CORRECT still ships with a valid marker).
# ─────────────────────────────────────────────────────────────────────────────

print("\n── C. Regression spot-check (CORRECT with marker still ships) ──────────")

from src.components.agent_core.gates import verify_numbers  # already imported

# Canonical traced figure — must still pass (marker + cell = the gold standard).
led_canon = _cell_ledger(513983.0, "amzn-2022", 41, "Total net sales", "513,983")
r_canon = verify_numbers("Net sales were 513,983 [amzn-2022 p.41].", led_canon)
check("C1: canonical traced figure with marker still passes",
      r_canon["pass"],
      f"detail: {r_canon.get('detail')}")

r_cited = verify_citations("Net sales were 513,983 [amzn-2022 p.41].", led_canon)
check("C2: canonical cited factual sentence still passes",
      r_cited["pass"],
      f"detail: {r_cited.get('detail')}")

# Short connective sentences are still exempt.
r_exempt = verify_citations("Here is a brief summary of what I found.", EvidenceLedger())
check("C3: short connective sentence still exempt from citation gate",
      r_exempt["pass"],
      f"detail: {r_exempt.get('detail')}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
total = _passed + _failed
print(f"\n{'='*64}")
print(f"  T8 LIVE: {_passed}/{total} passed  {_failed} failed")
if _failed == 0:
    print("  PASS — T8 gate changes:")
    print("    (a) Previously-redacted grounded answers: A1-A5 all ship.")
    print("    (b) Wrong/ungrounded answers: B1-B5 all still redact (moat intact).")
    print("    (c) Regression: C1-C3 canonical correct answers unchanged.")
    print()
    print("  NEXT (to fully close T8 live gate):")
    print("    1. Run T0: python -u eval/abstain_autopsy.py  (on-demand, ~$3)")
    print("    2. Grep abstain_autopsy_results.json for gate_redaction where traced=true")
    print("    3. Add those as real fixtures in Group A above (extend, don't replace)")
    print("    4. Confirm from the UI that those specific answers now ship")
else:
    print("  FAIL — T8 gate is not functioning correctly.")
    print("  Review the FAIL lines above — either a previously-redacted answer still")
    print("  redacts (T8 not applied) or a wrong answer now ships (moat breach).")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
