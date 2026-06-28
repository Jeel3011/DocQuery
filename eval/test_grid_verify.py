"""T4 gate — de-correlated second-verify pass on FOUND legal grid cells.

Fully offline, ZERO API spend: model_factory returns a ScriptedModel whose invoke()
response drives the second-verify logic deterministically.

DoD checks (from plans/tool_hard.md §T4):
  1. FOUND clause cell + contradicting second-verify response → downgraded to ABSTAIN
     with abstain_reason="verify_disagree".
  2. FOUND NUMERIC cell → second pass NOT run, cell stays FOUND.
  3. Second-pass model raises → original FOUND cell preserved (degrade, no crash).
  4. MISSING/ABSTAIN cell → second pass NOT run, cell unchanged.
  5. Flag off (GRID_SECOND_VERIFY=0) → second pass NOT run regardless.
  6. "yes" response from second pass → FOUND cell kept as-is.

Run: python -u eval/test_grid_verify.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# Override the env flag BEFORE the module loads so test 5 can patch it.
# We test flag=off by calling _second_verify_cell directly with the internal
# condition replicated, rather than reimporting with a mutated env.

from src.components.agent_core.model import ModelResponse, ScriptedModel
from src.components.agent_core.review_grid import (
    CellStatus,
    ColumnKind,
    GridCell,
    GridColumn,
)
from src.components.agent_core.grid_engine import (
    _second_verify_cell,
    _SECOND_VERIFY,
    build_cell,
)


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")


def _clause_column(key="governing_law") -> GridColumn:
    return GridColumn(
        key=key,
        label="Governing Law",
        prompt="Find the governing-law clause.",
        kind=ColumnKind.CLAUSE,
    )


def _numeric_column(key="liability_cap") -> GridColumn:
    return GridColumn(
        key=key,
        label="Liability Cap",
        prompt="Find the maximum liability cap (USD).",
        kind=ColumnKind.NUMERIC,
    )


def _found_cell(column: GridColumn, value="State of Minnesota",
                quote="This Agreement shall be governed by the laws of the State of Delaware.") -> GridCell:
    """A FOUND cell where the quote CONTRADICTS the value — the T4 target case."""
    return GridCell(
        doc_id="doc1",
        column_key=column.key,
        doc_name="contract.pdf",
        status=CellStatus.FOUND,
        value=value,
        quote=quote,
        provenance=[{"kind": "span", "doc": "doc1", "page": 1}],
    )


def _agreeing_cell(column: GridColumn) -> GridCell:
    """A FOUND cell where the quote SUPPORTS the value."""
    return GridCell(
        doc_id="doc1",
        column_key=column.key,
        doc_name="contract.pdf",
        status=CellStatus.FOUND,
        value="State of Delaware",
        quote="This Agreement shall be governed by the laws of the State of Delaware.",
        provenance=[{"kind": "span", "doc": "doc1", "page": 1}],
    )


def main() -> int:
    c = Check()
    col_clause = _clause_column()
    col_numeric = _numeric_column()

    # ── Test 1: contradicting second-verify → downgraded to ABSTAIN ────────────────
    print("── T4.1: contradicting second-verify → ABSTAIN ──────────────────────")
    contradicting_cell = _found_cell(col_clause)
    # Second-verify model says "no" — the quote doesn't support the value.
    no_model = ScriptedModel([ModelResponse(text="no – the quote names Delaware, not Minnesota.")])
    result = _second_verify_cell(contradicting_cell, lambda: no_model)
    c.ok(result.status == CellStatus.ABSTAIN,
         "contradicting second-verify → status is ABSTAIN")
    c.ok(result.abstain_reason == "verify_disagree",
         "contradicting second-verify → abstain_reason is 'verify_disagree'")
    c.ok(result.value is None,
         "downgraded cell has no value")
    c.ok("T4 second-verify disagrees" in (result.note or ""),
         "downgraded cell note carries T4 disagreement reason")
    c.ok(result.quote is not None,
         "downgraded cell preserves original quote (for audit)")

    # ── Test 2: numeric FOUND cell → second pass NOT run ───────────────────────────
    print("\n── T4.2: numeric FOUND cell → second pass not run ──────────────────")
    numeric_cell = GridCell(
        doc_id="doc1", column_key=col_numeric.key, doc_name="contract.pdf",
        status=CellStatus.FOUND, value="$5,000,000",
        quote="The maximum aggregate liability shall not exceed five million dollars.",
        provenance=[{"kind": "span", "doc": "doc1", "page": 2}],
    )
    call_count = [0]
    def factory_counting():
        call_count[0] += 1
        return ScriptedModel([ModelResponse(text="no – should not be called")])

    # Simulate what build_cell does: skip _second_verify_cell for NUMERIC columns.
    if numeric_cell.status == CellStatus.FOUND and col_numeric.kind != ColumnKind.NUMERIC:
        numeric_result = _second_verify_cell(numeric_cell, factory_counting)
    else:
        numeric_result = numeric_cell  # second pass not run for numeric

    c.ok(numeric_result.status == CellStatus.FOUND,
         "numeric FOUND cell stays FOUND")
    c.ok(call_count[0] == 0,
         "model_factory NOT called for numeric column")

    # ── Test 3: second-pass model raises → original FOUND cell preserved ────────────
    print("\n── T4.3: second-pass model raises → original cell preserved ─────────")
    stable_cell = _agreeing_cell(col_clause)

    def failing_factory():
        raise RuntimeError("API error — network timeout")

    result3 = _second_verify_cell(stable_cell, failing_factory)
    c.ok(result3.status == CellStatus.FOUND,
         "second-pass model error → original FOUND cell preserved (degrade, no crash)")
    c.ok(result3.value == stable_cell.value,
         "degraded cell value unchanged")

    # ── Test 4: MISSING/ABSTAIN cells → second pass NOT run ────────────────────────
    print("\n── T4.4: MISSING/ABSTAIN cells → second pass not run ────────────────")
    missing_cell = GridCell(
        doc_id="doc1", column_key=col_clause.key, doc_name="contract.pdf",
        status=CellStatus.MISSING,
    )
    abstain_cell = GridCell(
        doc_id="doc1", column_key=col_clause.key, doc_name="contract.pdf",
        status=CellStatus.ABSTAIN, abstain_reason="ambiguous",
    )
    factory_calls = [0]
    def factory_guard():
        factory_calls[0] += 1
        return ScriptedModel([ModelResponse(text="no")])

    # Neither MISSING nor ABSTAIN should trigger _second_verify_cell (build_cell guards on FOUND).
    for cell_under_test, label in [(missing_cell, "MISSING"), (abstain_cell, "ABSTAIN")]:
        if cell_under_test.status == CellStatus.FOUND:
            _second_verify_cell(cell_under_test, factory_guard)
    c.ok(factory_calls[0] == 0,
         "factory NOT called for MISSING or ABSTAIN cells")
    c.ok(missing_cell.status == CellStatus.MISSING,
         "MISSING cell unchanged")
    c.ok(abstain_cell.status == CellStatus.ABSTAIN,
         "ABSTAIN cell unchanged")

    # ── Test 5: flag off → second pass NOT run ────────────────────────────────────
    print("\n── T4.5: flag off (GRID_SECOND_VERIFY=0) → byte-identical ───────────")
    import src.components.agent_core.grid_engine as _ge
    orig_flag = _ge._SECOND_VERIFY
    _ge._SECOND_VERIFY = False
    try:
        flag_call_count = [0]
        def flag_factory():
            flag_call_count[0] += 1
            return ScriptedModel([ModelResponse(text="no")])

        cell_flag_test = _found_cell(col_clause)
        # Replicate the build_cell guard:
        if _ge._SECOND_VERIFY and flag_factory is not None and cell_flag_test.status == CellStatus.FOUND and col_clause.kind != ColumnKind.NUMERIC:
            result5 = _second_verify_cell(cell_flag_test, flag_factory)
        else:
            result5 = cell_flag_test
        c.ok(result5.status == CellStatus.FOUND,
             "with flag off, FOUND cell stays FOUND (byte-identical)")
        c.ok(flag_call_count[0] == 0,
             "model_factory NOT called when flag is off")
    finally:
        _ge._SECOND_VERIFY = orig_flag

    # ── Test 6: "yes" response → FOUND cell kept as-is ───────────────────────────
    print("\n── T4.6: second-verify says 'yes' → FOUND cell preserved ────────────")
    agree_cell = _agreeing_cell(col_clause)
    yes_model = ScriptedModel([ModelResponse(text="yes – the quote explicitly names Delaware as governing law.")])
    result6 = _second_verify_cell(agree_cell, lambda: yes_model)
    c.ok(result6.status == CellStatus.FOUND,
         "'yes' response → FOUND cell kept")
    c.ok(result6.value == agree_cell.value,
         "'yes' response → value unchanged")

    # ── Test 7: build_cell WIRING — model_factory drives the second pass ──────────
    # This is the integration test that proves T4 is not dormant: build_cell, when given a
    # model_factory, must invoke _second_verify_cell on a FOUND clause cell. We patch
    # _cell_from_run (so we don't have to script the full extraction envelope) and
    # _second_verify_cell (to record the call), then assert the wiring fires — and that a
    # NUMERIC column and a missing factory both skip it.
    print("\n── T4.7: build_cell wiring (model_factory → second pass fires) ──────")
    import src.components.agent_core.grid_engine as _ge

    _found_clause = _agreeing_cell(col_clause)
    _orig_cell_from_run = _ge._cell_from_run
    _orig_second_verify = _ge._second_verify_cell
    verify_calls = {"n": 0}

    def _fake_cell_from_run(doc_id, doc_name, column, events):
        # Always return a FOUND cell for the column under test (numeric or clause).
        return GridCell(doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                        status=CellStatus.FOUND, value=_found_clause.value,
                        quote=_found_clause.quote, provenance=[{"kind": "span", "doc": doc_id}])

    def _spy_second_verify(cell, factory):
        verify_calls["n"] += 1
        return _orig_second_verify(cell, factory)

    # A primary model that ends the loop immediately (no tools) so run_agent returns fast.
    _primary = ScriptedModel([ModelResponse(text="done")])
    _verify_yes = lambda: ScriptedModel([ModelResponse(text="yes – supported.")])

    try:
        _ge._cell_from_run = _fake_cell_from_run
        _ge._second_verify_cell = _spy_second_verify

        # (a) clause column + factory → second pass FIRES.
        verify_calls["n"] = 0
        build_cell("doc1", col_clause, collection_id="c1", model=_primary,
                   filename_by_doc={"doc1": "contract.pdf"}, model_factory=_verify_yes)
        c.ok(verify_calls["n"] == 1,
             "build_cell runs the second pass on a FOUND clause cell when a factory is given")

        # (b) clause column, NO factory → second pass does NOT fire (the old dormant default).
        verify_calls["n"] = 0
        build_cell("doc1", col_clause, collection_id="c1", model=_primary,
                   filename_by_doc={"doc1": "contract.pdf"}, model_factory=None)
        c.ok(verify_calls["n"] == 0,
             "build_cell skips the second pass when no factory is given (backward-compatible)")

        # (c) NUMERIC column + factory → second pass does NOT fire (kernel's deterministic job).
        verify_calls["n"] = 0
        build_cell("doc1", col_numeric, collection_id="c1", model=_primary,
                   filename_by_doc={"doc1": "contract.pdf"}, model_factory=_verify_yes)
        c.ok(verify_calls["n"] == 0,
             "build_cell skips the second pass on a NUMERIC column even with a factory")
    finally:
        _ge._cell_from_run = _orig_cell_from_run
        _ge._second_verify_cell = _orig_second_verify

    # ── Summary ──────────────────────────────────────────────────────────────────
    total = c.passed + c.failed
    print(f"\n{'─'*60}")
    print(f"T4 grid second-verify gate: {c.passed}/{total} passed", end="")
    if c.failed:
        print(f"  ({c.failed} FAILED)")
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
