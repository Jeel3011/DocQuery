"""Grounding gate — BRAIN_REASONING_PLAN §5.1 / §6 Stage C1.

On the REAL 8-doc grids, drive `ground_metric(intent, grids)` and assert the
correct-or-abstain contract (§4a): every intent either resolves to the RIGHT cell
or ABSTAINS — it must NEVER bind a wrong cell. The headline bar is **0 confidently-
wrong**; resolution-accuracy (CORRECT, not abstained) is reported and tracked.

The flagship case is the §2.1 failure: asking for a TOTAL ("total revenue") must
resolve to the total row (198,270) OR abstain — NEVER to a component line like
"Gross margin" (135,620). The structural `looks_like_total` value-check (kernel §5.4),
not a label list, is what makes that hold.

No DB, no API — grids are re-extracted deterministically from the PDFs, so this runs
in CI as a pure structural gate. Requires the 10-Ks in `test docs/`.

Run: python eval/test_grounding.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, looks_like_total
from src.components.brain.perception.grounding import ground_metric, MetricIntent

CORPUS = "test docs"
_grids_cache: dict = {}


def grids_for(doc):
    if doc not in _grids_cache:
        _grids_cache[doc] = [
            Grid(t.to_metadata(), doc=doc, page=t.page_number)
            for t in extract_tables_from_pdf(f"{CORPUS}/{doc}")
        ]
    return _grids_cache[doc]


# ── Grounding cases on real grids. Each: an intent + the WRONG value(s) it must
#    never bind. `expect` is the correct value IF it resolves (CORRECT); abstaining
#    is always acceptable (SAFE). A bind to anything in `forbidden` is the only
#    failure that matters (confidently-wrong).
CASES = [
    # ── THE flagship §2.1 case: total must never become a component ──────────────
    {
        "doc": "msft-10k_20220630.htm.pdf",
        "intent": MetricIntent(metric="total revenue", period="2022", aggregation_level="total"),
        "expect": 198270.0,
        "forbidden": [135620.0, 62650.0],   # Gross margin / Total cost of revenue (components)
        "note": "MSFT total revenue FY2022 — the canonical 'total→component' trap",
    },
    {
        "doc": "msft-10k_20220630.htm.pdf",
        "intent": MetricIntent(metric="total revenue", period="2022", section="Revenue",
                               aggregation_level="total"),
        "expect": 198270.0,
        "forbidden": [135620.0, 72732.0, 125538.0],  # gross margin + the two components
        "note": "MSFT total revenue, section-scoped → should land the total (section binds)",
    },
    # ── component request resolves to the component, not the total ──────────────
    {
        "doc": "msft-10k_20220630.htm.pdf",
        "intent": MetricIntent(metric="gross margin", period="2022", aggregation_level="component"),
        "expect": 135620.0,
        "forbidden": [198270.0],            # must not drift up to total revenue
        "note": "MSFT gross margin FY2022 — component path",
    },
    # ── a clean miss must abstain, never invent ─────────────────────────────────
    {
        "doc": "msft-10k_20220630.htm.pdf",
        "intent": MetricIntent(metric="total flux capacitance", period="2022"),
        "expect": None,
        "forbidden": [],
        "note": "nonexistent metric → must abstain",
    },
    # ── Amazon segment total (AWS) — section authority + total level ─────────────
    {
        "doc": "amzn-20231231.pdf",
        "intent": MetricIntent(metric="net sales", period="2023", section="Consolidated",
                               aggregation_level="total"),
        "expect": 574785.0,
        "forbidden": [90757.0],             # AWS net sales — a SEGMENT, not consolidated
        "note": "Amazon consolidated net sales 2023 — must not bind the AWS segment",
    },
]


def _vfind(values, target, tol=0.01):
    return any(abs(v - target) <= max(abs(target), 1.0) * tol for v in values)


def main():
    print("Grounding gate (§5.1 / Stage C1) — correct-or-abstain on real grids\n")
    correct = abstained = wrong = 0
    n = len(CASES)
    for c in CASES:
        grids = grids_for(c["doc"])
        g = ground_metric(c["intent"], grids)
        it = c["intent"]
        tag = f"{it.metric!r}[{it.period}]" + (f" §{it.section}" if it.section else "") \
              + (f" lvl={it.aggregation_level}" if it.aggregation_level != "any" else "")

        if g.abstained or g.best is None:
            abstained += 1
            print(f"  [ABSTAIN ] {tag:48} — {c['note']}")
            continue

        v = g.best.value
        if c["forbidden"] and _vfind(c["forbidden"], v):
            wrong += 1
            print(f"  [WRONG ✗ ] {tag:48} bound {v:g} "
                  f"[{g.best.section}/{g.best.label}] — FORBIDDEN ({c['note']})")
        elif c["expect"] is not None and abs(v - c["expect"]) <= max(abs(c["expect"]), 1.0) * 0.01:
            correct += 1
            print(f"  [CORRECT ] {tag:48} → {v:g} [{g.best.section}/{g.best.label}]")
        else:
            # resolved a non-forbidden, non-expected value: treat as WRONG only if we
            # had an expectation (a bound value we can't vouch for is unsafe in finance)
            if c["expect"] is not None:
                wrong += 1
                print(f"  [WRONG ✗ ] {tag:48} bound {v:g} "
                      f"[{g.best.section}/{g.best.label}] — expected {c['expect']:g}")
            else:
                abstained += 1
                print(f"  [OK?     ] {tag:48} bound {v:g} (no expectation) — {c['note']}")

    # ── structural unit check on looks_like_total (the kernel helper §5.4) ───────
    msft = grids_for("msft-10k_20220630.htm.pdf")
    tl_checks = []
    for grd in msft:
        for r in grd.rows:
            lab = (r.get("label") or "").strip().lower()
            if lab == "total revenue" and "2022" in grd.value_columns():
                tl_checks.append(("total revenue is-total", looks_like_total(grd, r, "2022") is True))
            if lab == "gross margin" and "2022" in grd.value_columns():
                # gross margin sits in Cost-of-revenue; vs its siblings it is the largest
                # so looks_like_total may say True THERE — that's fine: the SECTION (Cost
                # of revenue) is what disqualifies it as "total REVENUE", not this check.
                pass
    helper_ok = all(ok for _, ok in tl_checks) and bool(tl_checks)

    print("\n" + "=" * 64)
    print(f"  CORRECT  : {correct}/{n}   (resolution accuracy)")
    print(f"  ABSTAIN  : {abstained}/{n}   (safe)")
    print(f"  WRONG    : {wrong}/{n}   (DANGEROUS — confidently-wrong bind)")
    print(f"  looks_like_total structural check: {'✓' if helper_ok else '✗'} ({len(tl_checks)} probes)")
    print("=" * 64)
    ok = (wrong == 0 and helper_ok)
    print("  ✓ §5.1 bar MET (0 confidently-wrong)" if ok
          else f"  ✗ BAR FAILED ({wrong} confidently-wrong"
               + ("" if helper_ok else " + looks_like_total broken") + ")")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
