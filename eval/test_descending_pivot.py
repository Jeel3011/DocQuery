"""Descending-grid + threshold-units regression gate — SPINE_TRUST_BREAKS C1/C2/M1.

The trust-break analysis (2026-06-10) found the worst confident-wrong: `first_exceeds`/
`last_below` scanned the period series in GRID COLUMN ORDER, which is NOT guaranteed
chronological. Microsoft's statements list the LATEST year first (descending), so a
"first exceeded $X" question bound the wrong (latest) year — and the reasoning verifier
CERTIFIED it, because the winner sat at scan index 0 and the old predicate check had no
prior period to test (`if idx > 0`). A confident-wrong wearing the "deterministic,
self-monitored" badge.

The fix: (A) `build_series` sorts the period series ascending by extracted year before
returning, so a threshold scan walks time forward; (B) `_check_predicate` now requires a
below-threshold WITNESS earlier in the series — so an all-exceed series (too-small
threshold from comprehension, C2) or an unsortable axis abstains instead of certifying.

This gate runs the FULL spine — plan(ir) → execute → monitor — over the REAL descending
MSFT income statement (pg57: Total revenue 2022=198,270 / 2021=168,088 / 2020=143,015),
no LLM, no retrieval. It is the descending-order coverage the executor/selection gates
never had (they load only ascending Amazon grids).

Run: python eval/test_descending_pivot.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, compute
from src.components.brain.comprehension import QueryIR
from src.components.brain.executive import plan, execute
from src.components.brain.monitoring import monitor

MSFT22 = "test docs/msft-10k_20220630.htm.pdf"

PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append((PASS if cond else FAIL, desc, detail))
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


def _income_grid():
    """The MSFT FY2022 income statement grid — descending year columns [2022,2021,2020]
    with the consolidated Total revenue line. Isolated so the test indicts ordering, not
    grid selection."""
    for t in extract_tables_from_pdf(MSFT22):
        g = Grid(t.to_metadata(), doc="msft22", page=t.page_number)
        vc = g.value_columns()
        if vc[:3] == ["2022", "2021", "2020"]:
            r = g.find_row("total revenue", "")
            if r and "2022" in r and str(r.get("2022")).replace(",", "").startswith("198"):
                return g
    return None


def _ir(threshold):
    return QueryIR(raw="", question_type="extremum_pivot", metrics=["total revenue"],
                   entities=[], periods=[],
                   constraints={"predicate": "first_exceeds", "threshold": float(threshold)})


def main():
    print("Descending-pivot gate (SPINE_TRUST_BREAKS C1/C2/M1) — full spine over a real "
          "descending grid\n")
    g = _income_grid()
    if g is None:
        check("locate MSFT descending income statement", False, "grid not found")
        return _summary()
    check("MSFT income grid is DESCENDING", g.value_columns()[:3] == ["2022", "2021", "2020"],
          f"cols={g.value_columns()[:3]}")

    # ── C1: the kernel itself must bind the CHRONOLOGICALLY-first crossing, not the
    #    first column. thr=160,000: 2020=143,015 (below) → 2021=168,088 (FIRST above) →
    #    2022=198,270. Pre-fix (column order) bound 2022; post-fix binds 2021. ──
    spec = {"op": "first_exceeds", "over": "period",
            "row": {"label": "total revenue"}, "threshold": 160000}
    res_k = compute(spec, [g])
    check("C1 kernel: first_exceeds binds the EARLIEST crossing year (2021, not 2022)",
          res_k.ok and res_k.binding == "2021",
          f"binding={res_k.binding!r} (was 2022 before the chronological sort)")

    # ── C1 end-to-end: plan → execute → monitor must resolve 2021 AND the monitor must
    #    CERTIFY it (a real below-witness 2020 exists). Correct + verified, no abstain. ──
    qp = plan(_ir(160000))
    res = execute(qp, [g])
    verdict = monitor(qp, res)
    check("C1 spine: bound pivot is 2021", res.ok and res.answer_binding and res.answer_binding.value == "2021",
          f"value={getattr(res.answer_binding,'value',None)!r}")
    check("C1 spine: monitor CERTIFIES the correct year (below-witness 2020 ≤ thr)",
          res.ok and not verdict.abstain, f"abstain={verdict.abstain} reason={verdict.reason[:60]}")

    # ── C2: a too-small threshold (units error — $190B emitted as 190 over $-millions
    #    cells) makes EVERY year exceed it → the 'first' is just scan position 0, with no
    #    below-witness. The monitor must ABSTAIN, never certify a unit-error pivot. ──
    qp2 = plan(_ir(190))
    res2 = execute(qp2, [g])
    verdict2 = monitor(qp2, res2)
    check("C2 spine: all-exceed (units error) → monitor ABSTAINS, no confident-wrong",
          verdict2.abstain,
          f"abstain={verdict2.abstain} bound={getattr(res2.answer_binding,'value',None)!r} "
          f"reason={verdict2.reason[:70]}")

    # ── No over-abstain guard: a GENUINE crossing with a real below-witness still passes
    #    (thr=190,000 → 2020,2021 below, 2022 first above → 2022 certified). ──
    qp3 = plan(_ir(190000))
    res3 = execute(qp3, [g])
    verdict3 = monitor(qp3, res3)
    check("guard: genuine crossing (thr=190B) → 2022 certified, NOT over-abstained",
          res3.ok and res3.answer_binding.value == "2022" and not verdict3.abstain,
          f"value={getattr(res3.answer_binding,'value',None)!r} abstain={verdict3.abstain}")

    _summary()


def _summary():
    n = sum(1 for s, _d, _x in results if s == PASS)
    print("\n" + "=" * 66)
    ok = n == len(results)
    print(f"  {n}/{len(results)} checks passed " + ("✓ C1/C2/M1 fix holds" if ok
          else "✗ regression — descending/units confident-wrong NOT contained"))
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
