"""Accept-test for the deterministic Analyst (§4b step 3) against KNOWN answers.

Run: python eval/test_analyst_compute.py
Proves: (1) compute is exact on the real AWS FY23 grid, (2) every result traces
to source cells, (3) the whitelist rejects non-arithmetic / injection ops. The
LLM is NOT involved here — this tests the deterministic core in isolation.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, compute, verify_numbers, cells_from_results

AMZN = "test docs/amzn-20231231.pdf"


def aws_segment_grid():
    for t in extract_tables_from_pdf(AMZN):
        if "2023" in t.periods and any(
            r.get("section") == "AWS" and "Net sales" in r.get("label", "") for r in t.rows
        ):
            return Grid(t.to_metadata(), doc="amzn-20231231.pdf", page=t.page_number)
    raise SystemExit("AWS segment grid not found — re-check extraction")


# (description, spec, expected display) — answers verified by hand.
def cases(grid):
    return [
        ("AWS net sales 2023",
         {"op": "value", "row": {"section": "AWS", "label": "Net sales"}, "period": "2023"},
         "90,757"),
        ("AWS net sales growth 22→23",
         {"op": "growth_pct", "row": {"section": "AWS", "label": "Net sales"},
          "from_period": "2022", "to_period": "2023"},
         "13.3%"),
        ("AWS operating margin 2023",
         {"op": "margin_pct",
          "numerator": {"row": {"section": "AWS", "label": "Operating income"}},
          "denominator": {"row": {"section": "AWS", "label": "Net sales"}}, "period": "2023"},
         "27.1%"),
        ("AWS net sales CAGR 21→23",
         {"op": "cagr_pct", "cells": [
             {"row": {"section": "AWS", "label": "Net sales"}, "period": p}
             for p in ("2021", "2022", "2023")]},
         "20.8%"),
        ("AWS delta net sales 22→23",
         {"op": "delta", "row": {"section": "AWS", "label": "Net sales"},
          "from_period": "2022", "to_period": "2023"},
         "10,661"),
    ]


def main():
    grid = aws_segment_grid()
    passed = failed = 0

    for desc, spec, expected in cases(grid):
        r = compute(spec, [grid])
        ok = r.ok and r.display() == expected and len(r.cells) >= 1
        passed += ok; failed += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: {r.display()} (want {expected})  | {r.formula}")

    # security: non-whitelisted / injection ops must be rejected, not run
    for bad in ('__import__("os").system', "eval", "exec", "read_csv", "to_pickle"):
        r = compute({"op": bad, "row": {}, "period": "2023"}, [grid])
        rej = (not r.ok) and "not allowed" in (r.error or "")
        passed += rej; failed += (not rej)
        print(f"  [{'PASS' if rej else 'FAIL'}] reject op {bad!r}: {r.error}")

    # graceful failure: missing row / non-numeric cell must error, never guess
    r = compute({"op": "value", "row": {"label": "Nonexistent line item"}, "period": "2023"}, [grid])
    miss_ok = (not r.ok) and r.value is None
    passed += miss_ok; failed += (not miss_ok)
    print(f"  [{'PASS' if miss_ok else 'FAIL'}] missing row → graceful error: {r.error}")

    # numeric verifier (§4b step 5): grounded answer passes, fabricated figure flagged
    results = [
        compute({"op": "growth_pct", "row": {"section": "AWS", "label": "Net sales"},
                 "from_period": "2022", "to_period": "2023"}, [grid]),
        compute({"op": "value", "row": {"section": "AWS", "label": "Net sales"}, "period": "2023"}, [grid]),
    ]
    cells = cells_from_results(results)
    v_good = verify_numbers("AWS net sales grew 13.3% to $90,757 million from $80,096 million.", cells, results)
    v_bad = verify_numbers("AWS net sales were $95,000 million in 2023.", cells, results)
    ver_ok = v_good.ok and (not v_bad.ok) and "$95,000" in v_bad.ungrounded
    passed += ver_ok; failed += (not ver_ok)
    print(f"  [{'PASS' if ver_ok else 'FAIL'}] numeric verifier: grounded ok={v_good.ok}, "
          f"fabricated flagged={v_bad.ungrounded}")

    print(f"\n  {passed}/{passed+failed} checks passed"
          + ("  ✓" if failed == 0 else "  ✗ BELOW BAR"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
