"""Layer 1 (§3.3) accept-test for the deterministic SELECTION ops against KNOWN
pivots. The executive layer's whole point: the WHICH-year / WHICH-entity answer is
COMPUTED over a complete series, never guessed by a model.

Run: python eval/test_selection.py
Proves:
  (1) first_exceeds / last_below resolve the correct PIVOT YEAR over a complete
      period series (the canonical Q2 "year AWS op income first exceeded $20B"
      → 2022, including the 2021-below-threshold trap);
  (2) argmax / argmin / rank / filter resolve the correct ENTITY over a row series
      (segment with highest/lowest operating income, incl. the negative-loss trap);
  (3) every result carries its winning CellRef AND the full `candidates` trail
      (completeness — the reasoning verifier checks "did it see all of them?");
  (4) the whitelist still rejects injection / non-whitelisted ops;
  (5) selection over a missing row / threshold no one crosses ABSTAINS, never guesses.

The LLM is NOT involved — this tests the deterministic kernel in isolation, exactly
like test_analyst_compute.py. The bar is 100% (a wrong pivot is the §4a-forbidden
confidently-wrong error this layer exists to eliminate).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, compute

AMZN23 = "test docs/amzn-20231231.pdf"
AMZN22 = "test docs/amzn-20221231.pdf"


def _grid(path, predicate):
    for t in extract_tables_from_pdf(path):
        if predicate(t):
            return Grid(t.to_metadata(), doc=path.split("/")[-1], page=t.page_number)
    raise SystemExit(f"grid not found in {path} — re-check extraction")


def aws_segment_grid():
    # AWS segment table (periods 2021-2023 as columns) — one row scanned over years.
    return _grid(AMZN23, lambda t: "2023" in t.periods and any(
        r.get("section") == "AWS" and "Net sales" in r.get("label", "") for r in t.rows))


def amzn22_segment_grid():
    # Amazon FY2022 segment op-income table (NA / International / AWS / Consolidated
    # as ROWS) — the entity series for argmax/argmin.
    def pred(t):
        secs = {r.get("section") for r in t.rows if r.get("section")}
        return {"AWS", "North America", "International"} <= secs and any(
            "Operating income" in (r.get("label") or "") for r in t.rows)
    return _grid(AMZN22, pred)


SEG_ROWS_22 = [
    {"row": {"section": "North America", "label": "Operating income (loss)"}},
    {"row": {"section": "International", "label": "Operating income (loss)"}},
    {"row": {"section": "AWS", "label": "Operating income"}},
]


def period_cases(g):
    # (desc, spec, expected_binding, expected_n_candidates)
    return [
        ("AWS op income first exceeds $20B → 2022 (2021=18.5B is the trap)",
         {"op": "first_exceeds", "over": "period",
          "row": {"section": "AWS", "label": "Operating income"}, "threshold": 20000},
         "2022", 3),
        ("AWS net sales first exceeds $80B → 2022 (just over the line)",
         {"op": "first_exceeds", "over": "period",
          "row": {"section": "AWS", "label": "Net sales"}, "threshold": 80000},
         "2022", 3),
        ("AWS net sales last year below $80B → 2021",
         {"op": "last_below", "over": "period",
          "row": {"section": "AWS", "label": "Net sales"}, "threshold": 80000},
         "2021", 3),
        ("AWS op income last year below $20B → 2021",
         {"op": "last_below", "over": "period",
          "row": {"section": "AWS", "label": "Operating income"}, "threshold": 20000},
         "2021", 3),
    ]


def entity_cases(g):
    return [
        ("argmax segment op income 2022 → AWS (NA & Intl both losses)",
         {"op": "argmax", "over": "entity", "period": "2022", "rows": SEG_ROWS_22},
         "AWS", 3),
        ("argmin segment op income 2022 → International (-7,746, biggest loss)",
         {"op": "argmin", "over": "entity", "period": "2022", "rows": SEG_ROWS_22},
         "International", 3),
        ("argmax segment op income 2021 → AWS (18,532)",
         {"op": "argmax", "over": "entity", "period": "2021", "rows": SEG_ROWS_22},
         "AWS", 3),
        ("rank segment op income 2022 (desc) → AWS first",
         {"op": "rank", "over": "entity", "period": "2022", "rows": SEG_ROWS_22},
         "AWS", 3),
    ]


def main():
    g_per = aws_segment_grid()
    g_ent = amzn22_segment_grid()
    passed = failed = 0

    def check(ok, msg):
        nonlocal passed, failed
        passed += ok; failed += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    # ── period-axis selection: the pivot YEAR ──────────────────────────────────
    for desc, spec, want_bind, want_n in period_cases(g_per):
        r = compute(spec, [g_per])
        ok = r.ok and r.binding == want_bind and len(r.candidates) == want_n and len(r.cells) == 1
        check(ok, f"{desc}: binding={r.binding} (want {want_bind}), "
                  f"scanned={len(r.candidates)} | {r.formula}")

    # ── entity-axis selection: the pivot ENTITY ────────────────────────────────
    for desc, spec, want_bind, want_n in entity_cases(g_ent):
        r = compute(spec, [g_ent])
        ok = r.ok and r.binding == want_bind and len(r.candidates) == want_n
        check(ok, f"{desc}: binding={r.binding} (want {want_bind}), "
                  f"scanned={len(r.candidates)} | {r.formula}")

    # filter: which 2022 segments were PROFITABLE (>0) → only AWS
    r = compute({"op": "filter", "over": "entity", "period": "2022",
                 "rows": SEG_ROWS_22, "cmp": ">", "threshold": 0}, [g_ent])
    check(r.ok and r.binding == "AWS" and len(r.candidates) == 3,
          f"filter profitable segments 2022 → {r.binding} (want AWS) | {r.formula}")

    # ── completeness: the trail must hold every value scanned, not just the winner
    r = compute({"op": "argmax", "over": "entity", "period": "2022", "rows": SEG_ROWS_22}, [g_ent])
    vals = sorted(round(c.value) for c in r.candidates)
    check(vals == sorted([-2847, -7746, 22841]),
          f"completeness: argmax candidates = {vals} (want all 3 segment figures)")

    # ── security: injection / non-whitelisted ops rejected, never run
    for bad in ('__import__("os").system', "sort_values", "argmax; rm -rf"):
        r = compute({"op": bad, "over": "entity", "period": "2022", "rows": SEG_ROWS_22}, [g_ent])
        check((not r.ok) and "not allowed" in (r.error or ""),
              f"reject op {bad!r}: {r.error}")

    # ── correct-or-abstain: no year crosses the threshold → abstain, never guess
    r = compute({"op": "first_exceeds", "over": "period",
                 "row": {"section": "AWS", "label": "Operating income"}, "threshold": 999999}, [g_per])
    check((not r.ok) and r.binding is None,
          f"no period exceeds → abstain: {r.error}")

    # ── correct-or-abstain: a missing row → graceful abstain, never guess
    r = compute({"op": "argmax", "over": "entity", "period": "2022",
                 "rows": [{"row": {"label": "Nonexistent line item"}}]}, [g_ent])
    check((not r.ok) and r.value is None,
          f"missing row → graceful abstain: {r.error}")

    # ── threshold-crossing only over an ORDERED period series (not entities)
    r = compute({"op": "first_exceeds", "over": "entity", "period": "2022",
                 "rows": SEG_ROWS_22, "threshold": 0}, [g_ent])
    check((not r.ok) and "ordered period series" in (r.error or ""),
          f"first_exceeds over entities rejected: {r.error}")

    print(f"\n  {passed}/{passed+failed} checks passed"
          + ("  ✓ Layer 1 selection bar met" if failed == 0 else "  ✗ BELOW BAR"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
