"""Spine C4 enforcement gate — SPINE_TRUST_BREAKS C4 (the headline fix).

Before this fix the executive spine's self-monitor could convert a confident-WRONG into an
ABSTAIN *in isolation*, but that abstain NEVER reached the shipped answer: the coordinator
returned applied=False and the request fell through to the OLD Brain+Analyst prose path,
which still stated the wrong number. The monitor suppressed nothing live.

The fix threads the monitor's abstain (which the coordinator already computes — `abstained=
True` precisely when the spine RESOLVED a figure but the monitor flagged its reasoning) into
`Brain.run` / `run_stream` as `spine_abstain`, where it BINARY-WITHHOLDS: REDUCE is skipped
entirely (no rejected figure is synthesised) and a clean refusal ships. WRONG→ABSTAIN is now
a SHIPPED outcome (§5.5 / §4a).

Two checks, both OFFLINE (no API):
  1. Coordinator SIGNAL — on the real descending MSFT grid with a units-error threshold (the
     C2 all-exceed case), run_executive_spine returns applied=False AND abstained=True with a
     reason — the withhold signal the caller needs.
  2. Brain WITHHOLD — Brain.run(spine_abstain=...) returns a clean refusal (confidence 0.0,
     abstained=True) and NEVER calls _reduce (so no wrong number is generated). Brain's
     LLM-bound steps are monkeypatched so the test is deterministic and API-free.

Run: python eval/test_spine_enforcement.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid
import src.components.brain.map_reduce as mr
from src.components.brain.map_reduce import Brain
from src.components.brain.meta_reasoner import run_executive_spine

MSFT22 = "test docs/msft-10k_20220630.htm.pdf"
PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append((PASS if cond else FAIL, desc, detail))
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


class _Msg:
    def __init__(self, c): self.content = c


class _FakeComprehend:
    """Returns a fixed comprehend JSON — models C2 for a 'first exceeded $X' question whose
    threshold the model emitted in the WRONG units (190 over $-millions = the C2 trap)."""
    def __init__(self, json_str): self._j = json_str
    def invoke(self, _messages): return _Msg(self._j)


def _income_grid():
    for t in extract_tables_from_pdf(MSFT22):
        g = Grid(t.to_metadata(), doc="msft22", page=t.page_number)
        if g.value_columns()[:3] == ["2022", "2021", "2020"]:
            r = g.find_row("total revenue", "")
            if r and str(r.get("2022", "")).replace(",", "").startswith("198"):
                return g
    return None


def main():
    print("Spine C4 enforcement gate (SPINE_TRUST_BREAKS C4) — abstain becomes a SHIPPED "
          "withhold\n")

    # ── 1. Coordinator produces the withhold signal on a monitor-flagged figure ──
    g = _income_grid()
    if g is None:
        check("locate MSFT descending income grid", False, "not found")
        return _summary()
    units_error_ir = ('{"question_type":"extremum_pivot","metrics":["total revenue"],'
                      '"entities":[],"periods":[],'
                      '"constraints":{"predicate":"first_exceeds","threshold":190}}')
    outcome = run_executive_spine(
        "Which year did Microsoft total revenue first exceed $190 billion?",
        [g], _FakeComprehend(units_error_ir),
    )
    check("coordinator: monitor-flagged figure → applied=False (no wrong block injected)",
          outcome.applied is False, f"applied={outcome.applied}")
    check("coordinator: emits abstained=True WITH a reason (the withhold signal)",
          outcome.abstained is True and bool(outcome.reason),
          f"abstained={outcome.abstained} reason={outcome.reason[:60]!r}")

    # ── 2. Brain.run binary-WITHHOLDS when given spine_abstain; never runs REDUCE ──
    brain = Brain(mr.Config())
    calls = {"reduce": 0}

    def _no_reduce(*a, **k):
        calls["reduce"] += 1
        return ("Microsoft first exceeded $190B in 2023.", 0.9, [])  # the would-be wrong answer

    # neutralise every LLM/DB-bound step so the test is offline + deterministic
    brain._map_all_docs = lambda *a, **k: []
    brain._get_verify_llm = lambda: None
    brain._build_sources = lambda *a, **k: []
    brain._record_ledger = lambda *a, **k: None
    brain._reduce = _no_reduce
    _orig_verify = mr.verify_claims
    mr.verify_claims = lambda claims, llm, *a, **k: ([], [])
    try:
        res = brain.run(
            query="Which year did Microsoft total revenue first exceed $190 billion?",
            doc_chunks={"d": ("msft22", [])},
            spine_abstain=outcome.reason,
        )
    finally:
        mr.verify_claims = _orig_verify

    check("Brain.run: REDUCE is NEVER called under spine_abstain (no wrong number synthesised)",
          calls["reduce"] == 0, f"reduce_calls={calls['reduce']}")
    check("Brain.run: result.abstained is True", res.abstained is True, f"abstained={res.abstained}")
    check("Brain.run: confidence forced to 0.0", res.confidence == 0.0, f"confidence={res.confidence}")
    check("Brain.run: ships the clean withhold message (not the wrong figure)",
          "withholding" in res.answer.lower() and "2023" not in res.answer,
          f"answer={res.answer[:80]!r}")

    # ── guard: with NO spine_abstain the path is unchanged (would reach REDUCE) ──
    calls["reduce"] = 0
    mr.verify_claims = lambda claims, llm, *a, **k: ([], [])
    try:
        res2 = brain.run(query="q", doc_chunks={"d": ("msft22", [])}, spine_abstain=None)
    finally:
        mr.verify_claims = _orig_verify
    # no verified claims here → the 'couldn't verify' branch, NOT the withhold message
    check("guard: spine_abstain=None → withhold message NOT shipped (path unchanged)",
          "withholding the figure" not in res2.answer.lower(),
          f"answer={res2.answer[:60]!r}")

    _summary()


def _summary():
    n = sum(1 for s, _d, _x in results if s == PASS)
    print("\n" + "=" * 68)
    ok = n == len(results)
    print(f"  {n}/{len(results)} checks passed " + ("✓ C4 enforcement holds (abstain is shipped)"
          if ok else "✗ regression — monitor abstain does NOT suppress the live answer"))
    print("=" * 68)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
