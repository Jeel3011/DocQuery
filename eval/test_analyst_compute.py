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
MSFT = "test docs/msft-10k_20220630.htm.pdf"
GOOG = "test docs/goog-20221231.pdf"


def aws_segment_grid():
    for t in extract_tables_from_pdf(AMZN):
        if "2023" in t.periods and any(
            r.get("section") == "AWS" and "Net sales" in r.get("label", "") for r in t.rows
        ):
            return Grid(t.to_metadata(), doc="amzn-20231231.pdf", page=t.page_number)
    raise SystemExit("AWS segment grid not found — re-check extraction")


_msft_cache = None


def grids_for_msft():
    """All MSFT FY22 grids in one doc scope (BUG-E: the 'Total' label appears in many
    sections; only section:'Revenue' picks the 198,270 revenue total)."""
    global _msft_cache
    if _msft_cache is None:
        _msft_cache = [Grid(t.to_metadata(), doc="msft-10k_20220630.htm.pdf", page=t.page_number)
                       for t in extract_tables_from_pdf(MSFT)]
    return _msft_cache


def multidoc_grids():
    """Real MSFT FY22 + GOOG FY22 grids in ONE scope — the live BUG-1 shape.

    'Total revenue'[2021] exists in BOTH issuers (MSFT 168,088 / GOOG 257,637), so
    an unscoped reference is GENUINELY ambiguous (the abstain is correct); a
    doc-scoped reference must resolve. Mirrors the agent core's multi-doc scope.
    """
    grids = []
    for path, label in ((MSFT, "msft-10k_20220630.htm.pdf"), (GOOG, "goog-20221231.pdf")):
        for t in extract_tables_from_pdf(path):
            grids.append(Grid(t.to_metadata(), doc=label, page=t.page_number))
    return grids


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

    # ── BUG-1 (live, 2026-06-11): doc scoping in a multi-document scope ──────────
    # In the agent core's 8-doc scope, 'Total revenue'[2021] matched BOTH Microsoft
    # (168,088) and Alphabet ('Total revenues' 257,637) → every cross-doc-common
    # metric was ambiguous → every growth/delta over a totals line abstained.
    md = multidoc_grids()
    tot = {"section": "Revenue", "label": "Total revenue"}

    # (a) unscoped stays a CORRECT abstain (genuine cross-issuer ambiguity)…
    r = compute({"op": "delta", "row": tot, "from_period": "2021", "to_period": "2022"}, md)
    a_ok = (not r.ok) and "ambiguous" in (r.error or "")
    passed += a_ok; failed += (not a_ok)
    print(f"  [{'PASS' if a_ok else 'FAIL'}] multidoc unscoped → abstain: {r.error}")

    # …and the abstain message must NAME the docs + the repair (so the agent can
    # self-disambiguate instead of retrying blind 4 times, as happened live).
    e = r.error or ""
    a2_ok = (not r.ok) and "msft-10k_20220630" in e and "goog-20221231" in e and "doc" in e
    passed += a2_ok; failed += (not a2_ok)
    print(f"  [{'PASS' if a2_ok else 'FAIL'}] ambiguity error names docs + 'doc' repair hint")

    # (b) doc-scoped delta/growth resolve — the exact live MSFT FY22 growth question.
    r = compute({"op": "delta", "row": tot, "doc": "msft-10k_20220630",
                 "from_period": "2021", "to_period": "2022"}, md)
    b_ok = r.ok and r.display() == "30,182"
    passed += b_ok; failed += (not b_ok)
    print(f"  [{'PASS' if b_ok else 'FAIL'}] doc-scoped MSFT delta 21→22: {r.display()} (want 30,182)")

    r = compute({"op": "growth_pct", "row": tot, "doc": "msft-10k_20220630",
                 "from_period": "2021", "to_period": "2022"}, md)
    g_ok = r.ok and r.display() == "18.0%"
    passed += g_ok; failed += (not g_ok)
    print(f"  [{'PASS' if g_ok else 'FAIL'}] doc-scoped MSFT growth 21→22: {r.display()} (want 18.0%)")

    # (c) same question scoped to the OTHER issuer → the other (correct) number.
    # (section included: label-only 'Total revenues' is genuinely ambiguous even
    # WITHIN the GOOG doc — the tax note's 'Current/Total' row contains-matches —
    # and the kernel correctly abstains on that; doc+section is the realistic spec.)
    r = compute({"op": "growth_pct", "row": {"section": "Revenues", "label": "Total revenues"},
                 "doc": "goog-20221231", "from_period": "2021", "to_period": "2022"}, md)
    c_ok = r.ok and r.display() == "9.8%"
    passed += c_ok; failed += (not c_ok)
    print(f"  [{'PASS' if c_ok else 'FAIL'}] doc-scoped GOOG growth 21→22: {r.display()} (want 9.8%)")

    # (d) selection over periods threads doc too (build_series host selection).
    r = compute({"op": "argmax", "over": "period", "row": tot, "doc": "msft-10k_20220630"}, md)
    d_ok = r.ok and r.binding == "2022" and r.value == 198270.0
    passed += d_ok; failed += (not d_ok)
    print(f"  [{'PASS' if d_ok else 'FAIL'}] doc-scoped argmax over periods: {r.binding} ({r.value})")

    # (d2) UNSCOPED selection where the row lives in BOTH issuers must abstain —
    # _find_host returning "first host wins" across docs is a silent wrong-issuer
    # scan (which issuer it binds depends on grid order, i.e. luck).
    r = compute({"op": "argmax", "over": "period", "row": tot}, md)
    d2_ok = (not r.ok) and "doc" in (r.error or "")
    passed += d2_ok; failed += (not d2_ok)
    print(f"  [{'PASS' if d2_ok else 'FAIL'}] unscoped multi-doc selection → abstain: {r.error}")

    # (e) a doc OUTSIDE the scope errors cleanly (never silently widens back out).
    r = compute({"op": "delta", "row": tot, "doc": "nvda-20240128",
                 "from_period": "2021", "to_period": "2022"}, md)
    e_ok = (not r.ok) and "not in scope" in (r.error or "")
    passed += e_ok; failed += (not e_ok)
    print(f"  [{'PASS' if e_ok else 'FAIL'}] unknown doc → clean error: {r.error}")

    # (f) doc scoping must NOT mask genuine ambiguity WITHIN one document.
    twin = {"headers": ["section", "label", "2022"], "periods": ["2022"],
            "rows": [{"section": "X", "label": "Total revenue", "2022": "100"}]}
    twin2 = {"headers": ["section", "label", "2022"], "periods": ["2022"],
             "rows": [{"section": "X", "label": "Total revenue", "2022": "999"}]}
    same_doc = [Grid(twin, doc="one.pdf", page=1), Grid(twin2, doc="one.pdf", page=2)]
    r = compute({"op": "value", "row": {"section": "X", "label": "Total revenue"},
                 "doc": "one.pdf", "period": "2022"}, same_doc)
    f_ok = (not r.ok) and "ambiguous" in (r.error or "")
    passed += f_ok; failed += (not f_ok)
    print(f"  [{'PASS' if f_ok else 'FAIL'}] intra-doc disagreement still abstains: {r.error}")

    # ── BUG-E (live, 2026-06-11): intra-doc ambiguity must hint SECTION, not doc ─────
    # 'Total' appears in 8 sections of the MSFT filing; only section:'Revenue' picks
    # the revenue total (198,270). The old error said 'add doc' (the doc was already
    # scoped) → the model burned its whole step budget guessing sections. The error
    # now NAMES the sections + the 'add section' repair so it self-heals in one step.
    msft = grids_for_msft()
    r = compute({"op": "value", "doc": "msft-10k_20220630.htm.pdf",
                 "row": {"label": "Total"}, "period": "2022"}, msft)
    e_intra = (not r.ok) and "section" in (r.error or "") \
        and "Revenue" in (r.error or "") and 'add "doc"' not in (r.error or "")
    passed += e_intra; failed += (not e_intra)
    print(f"  [{'PASS' if e_intra else 'FAIL'}] intra-doc 'Total' → SECTION hint (not doc): {r.error}")
    # and the hinted repair RESOLVES to the true revenue total.
    r = compute({"op": "value", "doc": "msft-10k_20220630.htm.pdf",
                 "row": {"section": "Revenue", "label": "Total"}, "period": "2022"}, msft)
    e_fix = r.ok and r.value == 198270.0
    passed += e_fix; failed += (not e_fix)
    print(f"  [{'PASS' if e_fix else 'FAIL'}] section:'Revenue' resolves Total[2022]=198,270: {r.display() if r.ok else r.error}")

    # ── BUG-B (live, 2026-06-11): the row ref under "numerator" + empty labels ───────
    # The live MSFT growth call carried its row ref under "numerator" (the ratio
    # shape); resolve() fell through to the WHOLE spec, read an EMPTY label, and
    # find_row('') bound an ARBITRARY row → a kernel-level confident-wrong (-0.5%
    # "growth" computed from an unrelated row's values).
    # (g) "numerator" is now an accepted alias for the single-row ref:
    r_row = compute({"op": "growth_pct", "row": {"section": "AWS", "label": "Net sales"},
                     "from_period": "2022", "to_period": "2023"}, [grid])
    r_num = compute({"op": "growth_pct", "numerator": {"section": "AWS", "label": "Net sales"},
                     "from_period": "2022", "to_period": "2023"}, [grid])
    g_alias_ok = r_row.ok and r_num.ok and r_row.value == r_num.value
    passed += g_alias_ok; failed += (not g_alias_ok)
    print(f"  [{'PASS' if g_alias_ok else 'FAIL'}] numerator-alias growth == row-shape growth "
          f"({r_num.display() if r_num.ok else r_num.error})")

    # (h) an EMPTY label must never resolve — hard error naming the repair, never a value.
    r = compute({"op": "growth_pct", "numerator": {"section": "Revenue", "label": ""},
                 "from_period": "2021", "to_period": "2022"}, [grid])
    h_ok = (not r.ok) and "label" in (r.error or "")
    passed += h_ok; failed += (not h_ok)
    print(f"  [{'PASS' if h_ok else 'FAIL'}] empty label → clean error (no arbitrary bind): {r.error}")

    r = compute({"op": "value", "row": {}, "period": "2022"}, [grid])
    h2_ok = (not r.ok) and "label" in (r.error or "")
    passed += h2_ok; failed += (not h2_ok)
    print(f"  [{'PASS' if h2_ok else 'FAIL'}] empty row ref → clean error: {r.error}")

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
