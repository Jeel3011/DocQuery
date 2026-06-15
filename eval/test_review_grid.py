"""B2 gate — the review-grid engine, OFFLINE (scripted model, real grids, ZERO live calls).

Proves the cite-or-abstain contract that is the feature's whole point:
  1. FOUND requires grounding — a value with provenance renders; a 'found' the agent
     claims WITHOUT a source span is downgraded to ABSTAIN (no silent wrong cell).
  2. MISSING ('the term isn't in this doc') is a real, flagged finding, not a failure.
  3. ABSTAIN ('couldn't determine') is distinct from MISSING.
  4. A degraded run → ERROR (surfaced); an unparseable answer → ABSTAIN (never FOUND).
  5. build_cell drives a real run_agent end-to-end with a scripted model.
  6. coverage() tallies the publishable 'verified or flagged' number.

Run: python -u eval/test_review_grid.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from src.components.agent_core.review_grid import (
    CellStatus, ColumnKind, GridCell, GridColumn, GridResult, GridSpec, RiskFlag,
)
from src.components.agent_core.grid_engine import (
    build_cell, build_grid, _cell_from_run, _extract_envelope,
    _recover_custom_shape, _looks_like_envelope, _reads_as_abstention,
)
from src.components.agent_core.registry import RunScope
from src.components.agent_core.budgets import Budget
from src.components.agent_core.model import ModelResponse, ScriptedModel, ToolCall
from src.components.brain.analyst import Grid


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


# A real grid so a scripted compute call produces genuine ledger provenance.
# (Same shape as test_agent_loop's proven fixture — multi-period so compute resolves.)
_TJ = {
    "periods": ["2021", "2022", "2023"],
    "rows": [{"section": "", "label": "Total net sales",
              "2021": "469,822", "2022": "513,983", "2023": "574,785"}],
    "table_id": "amzn-cons",
}
GRIDS = [Grid(_TJ, doc="amzn-2022", page=41)]

GOV = GridColumn(key="gov_law", label="Governing Law",
                 prompt="Find the governing-law clause. Quote it exactly.",
                 risk_rubric="standard if NY/Delaware/England; else non_standard")
CAP = GridColumn(key="cap", label="Indemnity Cap", prompt="Find the indemnification cap.")


def _envelope(status, value=None, quote=None, risk="none", note=None):
    import json
    return json.dumps({"status": status, "value": value, "quote": quote,
                       "risk": risk, "note": note})


def _envelope_raw(obj):
    """Serialize an ARBITRARY dict as the model's final answer — used to script the
    custom (non-envelope) JSON shapes the tolerant parser must recover."""
    import json
    return json.dumps(obj)


def main() -> int:
    c = Check()

    # ── 1. PARSING / CONTRACT (pure _cell_from_run, no model) ────────────────────
    print("── cite-or-abstain contract (parsing layer) ─────────────────────")

    # 1a. FOUND with provenance → verified cell
    ev_found = [
        {"type": "token", "text": _envelope("found", "State of Delaware",
                                            "governed by the laws of the State of Delaware",
                                            "standard")},
        {"type": "sources", "sources": [{"kind": "span", "doc": "c1.pdf", "page": 12,
                                         "snippet": "...State of Delaware..."}]},
        {"type": "meta", "abstained": False},
    ]
    cell = _cell_from_run("d1", "c1.pdf", GOV, ev_found)
    c.ok(cell.status == CellStatus.FOUND and cell.value == "State of Delaware"
         and cell.is_verified and cell.risk == RiskFlag.STANDARD,
         f"FOUND+grounded → verified cell (status={cell.status.value}, verified={cell.is_verified})")

    # 1b. THE KEY ONE: model claims FOUND but NO provenance → downgraded to ABSTAIN
    ev_ungrounded = [
        {"type": "token", "text": _envelope("found", "State of Mars", "(no quote)", "standard")},
        {"type": "sources", "sources": []},   # nothing grounded it
        {"type": "meta", "abstained": False},
    ]
    cell = _cell_from_run("d1", "c1.pdf", GOV, ev_ungrounded)
    c.ok(cell.status == CellStatus.ABSTAIN and cell.value is None and not cell.is_verified,
         f"FOUND but ungrounded → DOWNGRADED to abstain, value dropped (no silent-wrong)")

    # 1c. MISSING → flagged finding, risk=missing
    ev_missing = [
        {"type": "token", "text": _envelope("missing", note="no indemnity cap clause present")},
        {"type": "sources", "sources": []},
        {"type": "meta", "abstained": True},
    ]
    cell = _cell_from_run("d2", "c2.pdf", CAP, ev_missing)
    c.ok(cell.status == CellStatus.MISSING and cell.risk == RiskFlag.MISSING and not cell.is_verified,
         f"MISSING → flagged finding (risk={cell.risk.value})")

    # 1d. ABSTAIN (ambiguous) distinct from missing
    ev_abstain = [
        {"type": "token", "text": _envelope("abstain", note="two conflicting caps; cannot disambiguate")},
        {"type": "sources", "sources": []},
        {"type": "meta", "abstained": True},
    ]
    cell = _cell_from_run("d1", "c1.pdf", CAP, ev_abstain)
    c.ok(cell.status == CellStatus.ABSTAIN and "conflicting" in (cell.note or ""),
         "ABSTAIN (ambiguous) distinct from MISSING")

    # 1e. degraded run → ERROR (surfaced, never hidden)
    ev_degraded = [{"type": "meta", "degrade": True, "error": "model_error: boom"}]
    cell = _cell_from_run("d1", "c1.pdf", GOV, ev_degraded)
    c.ok(cell.status == CellStatus.ERROR, "degraded run → ERROR (surfaced)")

    # 1f. unparseable answer → ABSTAIN (never FOUND), keeps text as note
    ev_garbage = [
        {"type": "token", "text": "Sorry, I think it's probably Delaware but I'm not sure."},
        {"type": "sources", "sources": []},
    ]
    cell = _cell_from_run("d1", "c1.pdf", GOV, ev_garbage)
    c.ok(cell.status == CellStatus.ABSTAIN and cell.note and "Delaware" in cell.note,
         "unparseable answer → ABSTAIN (text kept as note)")
    c.ok(cell.abstain_reason == "unparsed",
         "free-text (no JSON) abstain is flagged abstain_reason='unparsed' (not no_evidence)")

    # ── 1g. THE G4 LOOP-BREAKER: the exact Sezzle failing shape. The agent found the
    #        clause, GROUNDED it (provenance present), the gate passed — but returned a
    #        CUSTOM JSON shape instead of the envelope. Before the fix this was discarded
    #        → ABSTAIN ("unclear" forever). Now the tolerant parser recovers the value and,
    #        because provenance exists, UPGRADES it to FOUND. (Provenance injected directly,
    #        same as 1a — this isolates the parser, no live call.)
    print("── G4 loop-breaker: custom JSON shape recovers to FOUND ──────────")
    custom_shape = _envelope_raw({"governing_law_and_seat": {
        "governing_law": "Minnesota", "seat": "Jackson, Mississippi"}})
    ev_custom_grounded = [
        {"type": "token", "text": custom_shape},
        {"type": "sources", "sources": [{"kind": "span", "doc": "Document.pdf", "page": 5,
                                         "snippet": "...laws of the State of Minnesota..."}]},
        {"type": "meta", "abstained": False},
    ]
    cell = _cell_from_run("d1", "Document.pdf", GOV, ev_custom_grounded)
    c.ok(cell.status == CellStatus.FOUND and cell.value
         and "Minnesota" in cell.value and "Jackson, Mississippi" in cell.value
         and cell.is_verified and cell.abstain_reason is None,
         f"custom shape + provenance → FOUND, value recovered from leaves "
         f"(status={cell.status.value}, value={cell.value!r})")

    # 1h. SAME custom shape but NO provenance → must NOT become FOUND (the grounding
    #     guard is untouched). It abstains, flagged abstain_reason='unparsed' — NOT
    #     no_evidence — so a human/eval can tell "we mangled it" from "agent found nothing".
    ev_custom_ungrounded = [
        {"type": "token", "text": custom_shape},
        {"type": "sources", "sources": []},
        {"type": "meta", "abstained": False},
    ]
    cell = _cell_from_run("d1", "Document.pdf", GOV, ev_custom_ungrounded)
    c.ok(cell.status == CellStatus.ABSTAIN and not cell.is_verified
         and cell.abstain_reason == "unparsed",
         f"custom shape WITHOUT provenance → ABSTAIN unparsed (no silent-wrong; "
         f"abstain_reason={cell.abstain_reason})")

    # 1i. The abstain-reason DISTINCTION is the loop-breaker: a genuine model 'abstain'
    #     is 'ambiguous', and an ungrounded 'found' downgrade is 'no_evidence' — neither
    #     is 'unparsed'. This is what stops re-blaming ingestion for a parser bug.
    cell = _cell_from_run("d1", "c1.pdf", CAP, ev_abstain)
    c.ok(cell.abstain_reason == "ambiguous",
         "genuine model abstain → abstain_reason='ambiguous' (distinct from 'unparsed')")
    cell = _cell_from_run("d1", "c1.pdf", GOV, ev_ungrounded)
    c.ok(cell.abstain_reason == "no_evidence",
         "ungrounded 'found' downgrade → abstain_reason='no_evidence' (distinct from 'unparsed')")

    # 1k. ANTI-MISLABEL: the model declines IN PROSE inside a custom shape ("abstain; I
    #     could not verify a 'Term & Termination' section..."). Even WITH provenance, this
    #     must NOT become a FOUND cell whose value literally reads "abstain..." — it's a
    #     genuine abstain. (Live-observed on EX-10.01 term_termination, 2026-06-15.)
    custom_abstention = _envelope_raw({"term_and_termination":
        "abstain; I could not verify a section titled 'Term & Termination' in this document."})
    ev_custom_abstention = [
        {"type": "token", "text": custom_abstention},
        {"type": "sources", "sources": [{"kind": "span", "doc": "EX-10.01.pdf", "page": 1,
                                         "snippet": "...stock plan..."}]},
        {"type": "meta", "abstained": False},
    ]
    cell = _cell_from_run("d1", "EX-10.01.pdf", CAP, ev_custom_abstention)
    c.ok(cell.status == CellStatus.ABSTAIN and not cell.is_verified
         and cell.abstain_reason == "ambiguous"
         and not (cell.value or "").lower().startswith("abstain"),
         f"custom shape whose VALUE is an abstention → ABSTAIN, not a 'found' saying "
         f"'abstain' (status={cell.status.value}, value={cell.value!r})")
    # and the inverse: a real clause that merely opens normally still recovers to FOUND
    c.ok(not _reads_as_abstention("State of Minnesota; Jackson, Mississippi")
         and _reads_as_abstention("abstain; could not verify")
         and _reads_as_abstention("No indemnity clause is present"),
         "_reads_as_abstention: real clause False; prose-abstention True")

    # ── 1j. recovery helpers (pure) ──────────────────────────────────────────────
    c.ok(not _looks_like_envelope({"governing_law_and_seat": {"x": "y"}})
         and _looks_like_envelope({"status": "found"})
         and _looks_like_envelope({"value": "X"}),
         "_looks_like_envelope: custom shape False, envelope-keyed True")
    c.ok(_recover_custom_shape({"a": {"b": "Minnesota", "c": "Mississippi"}})
         == "Minnesota; Mississippi"
         and _recover_custom_shape({"a": None, "b": ""}) is None,
         "_recover_custom_shape flattens leaves; empty → None")

    # ── 2. _extract_envelope tolerance (fences, leading prose) ───────────────────
    print("── envelope extraction tolerance ────────────────────────────────")
    c.ok(_extract_envelope('```json\n{"status":"missing"}\n```') == {"status": "missing"},
         "extracts JSON from a ```json fence")
    c.ok(_extract_envelope('Here is my answer: {"status":"found","value":"X"} done')
         == {"status": "found", "value": "X"}, "extracts JSON amid prose")
    c.ok(_extract_envelope("no json here at all") is None, "no-JSON → None (→ abstain upstream)")

    # ── 3. build_cell end-to-end with a scripted model (real run_agent) ──────────
    print("── build_cell drives run_agent (scripted model) ─────────────────")

    # 3a. A 'missing' cell: model answers the envelope directly, no tools needed.
    miss_model = ScriptedModel([
        ModelResponse(text=_envelope("missing", note="no governing-law clause"), tool_calls=[]),
    ])
    cell = build_cell("d1", GOV, collection_id="c1", model=miss_model,
                      filename_by_doc={"d1": "c1.pdf"})
    c.ok(cell.status == CellStatus.MISSING and cell.doc_name == "c1.pdf",
         f"build_cell missing-path → {cell.status.value}")

    # 3b. THE GATE IN ACTION end-to-end: a model that claims FOUND but never grounds
    #     it (no tool call) must NOT yield a verified cell — build_cell drives the grid
    #     gate which rejects the ungrounded 'found' → the cell abstains. This is the
    #     'no silent wrong cell' invariant proven through the real loop (not just the
    #     parser). (The FOUND+grounded path is proven at the parsing layer in test 1a,
    #     where provenance is injected directly — grounding compute against a synthetic
    #     grid is a kernel-fixture concern, out of scope for the engine gate.)
    ungrounded_model = ScriptedModel([
        ModelResponse(text=_envelope("found", "State of Atlantis", "(fabricated)", "standard"),
                      tool_calls=[]),
    ])
    cell = build_cell("d1", GOV, collection_id="c1", model=ungrounded_model,
                      filename_by_doc={"d1": "c1.pdf"})
    c.ok(cell.status != CellStatus.FOUND and not cell.is_verified,
         f"build_cell rejects ungrounded 'found' via the grid gate → {cell.status.value} "
         f"(no silent-wrong, end-to-end)")

    # ── 4. build_grid assembles + coverage() reports the headline ────────────────
    print("── grid assembly + coverage headline ────────────────────────────")
    spec = GridSpec(title="t", collection_id="c1", doc_ids=["d1", "d2"], columns=[GOV])
    # model factory returns a fresh scripted model per cell (d1=found-ish missing, d2=missing)
    scripts = iter([
        [ModelResponse(text=_envelope("missing", note="absent"), tool_calls=[])],
        [ModelResponse(text=_envelope("abstain", note="unclear"), tool_calls=[])],
    ])
    res = build_grid(spec, model_factory=lambda: ScriptedModel(next(scripts)),
                     filename_by_doc={"d1": "c1.pdf", "d2": "c2.pdf"})
    cov = res.coverage()
    c.ok(cov["total"] == 2 and len(res.cells) == 2, f"grid filled all cells (total={cov['total']})")
    c.ok(cov["verified"] == 0 and cov["missing"] == 1 and cov["abstain"] == 1,
         f"coverage tally honest: {cov}")
    c.ok(all(not cell.is_verified for cell in res.cells if cell.status != CellStatus.FOUND),
         "NO cell is 'verified' without FOUND+provenance (zero silent-wrong invariant)")

    print()
    print("=" * 64)
    print(f"  {c.passed} passed · {c.failed} failed")
    if c.failed == 0:
        print("  ✓ B2 review-grid gate GREEN (cite-or-abstain · no silent-wrong cells)")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
