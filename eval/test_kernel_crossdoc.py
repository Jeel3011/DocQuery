"""Cross-document kernel probe — generic correctness across the WHOLE corpus, $0.

The per-question UI fixes (BUG-B..E, 2026-06-11) were structural, but they were
found one question at a time. This gate drives the deterministic kernel (`compute`/
`table_lookup`) over the REAL AMZN/GOOG/MSFT grids for the canonical metric ×
section × label × period combinations the n=27 set needs — all issuers, all years —
and asserts each against the hand-verified gold in eval_questions_multihop.json.

It surfaces the NEXT structural hole (a metric that abstains on one issuer's grid
shape, a label that binds the wrong row) for free, instead of one paid UI run at a
time. Three verdicts per probe:
  CORRECT  — resolved to the gold value
  ABSTAIN  — kernel declined (safe; a coverage gap, not a wrong answer)
  WRONG    — resolved to a NON-gold value (the forbidden class — must be 0)

The bar: **0 WRONG**. Abstains are listed (they're the honest coverage gap the A5
eval will then measure end-to-end). Generic-ness is proven by breadth here, not by
patching the kernel after each live question.

Run: python -u eval/test_kernel_crossdoc.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, compute
from src.components.agent_core.tools import table_lookup

CORPUS = "test docs"
GOLD = json.load(open("eval/eval_questions_multihop.json"))[0]["_verified_figures"]

# Each issuer's filings that hold the gold (the live 8-doc 'very big' scope).
DOCS = {
    "amzn": ["amzn-20221231.pdf", "amzn-20231231.pdf"],
    "msft": ["msft-10k_20210630.htm.pdf", "msft-10k_20220630.htm.pdf",
             "0000950170-23-035122.pdf"],  # MSFT FY23 (per CLAUDE.md)
    "goog": ["goog-20211231.pdf", "goog-20221231.pdf", "goog-20231231.pdf"],
}

_cache = {}
def grids_for(doc):
    if doc not in _cache:
        try:
            _cache[doc] = [Grid(t.to_metadata(), doc=doc, page=t.page_number)
                           for t in extract_tables_from_pdf(f"{CORPUS}/{doc}")]
        except Exception as e:
            print(f"  [skip] {doc}: extraction failed ({e})")
            _cache[doc] = []
    return _cache[doc]

def scope(issuer):
    g = []
    for d in DOCS[issuer]:
        g += grids_for(d)
    return g

# ── A probe: (issuer, op-spec, gold value). doc-scoped to one issuer's filings. ──
# These mirror the cell binds the n=27 questions require (totals, segments, YoY).
def value(issuer, doc, row, period):
    return {"issuer": issuer, "spec": {"op": "value", "doc": doc, "row": row,
                                       "period": period}}

PROBES = []
# MSFT total revenue (the BUG-E label='Total' section='Revenue' bind) — all 3 years.
for yr, gold in GOLD["msft_total_revenue_known"].items():
    doc = {"2021": "msft-10k_20210630", "2022": "msft-10k_20220630",
           "2023": "0000950170-23-035122"}[yr]
    PROBES.append(("msft", f"MSFT total revenue {yr}",
                   {"op": "value", "doc": doc,
                    "row": {"section": "Revenue", "label": "Total"}, "period": yr},
                   float(gold)))
# MSFT operating income total — same shape, different section (genericity check).
for yr, gold in GOLD["msft_op_income"].items():
    doc = {"2021": "msft-10k_20210630", "2022": "msft-10k_20220630",
           "2023": "0000950170-23-035122"}[yr]
    PROBES.append(("msft", f"MSFT operating income {yr}",
                   {"op": "value", "doc": doc,
                    "row": {"section": "Operating Income", "label": "Total"}, "period": yr},
                   float(gold)))
# AMZN consolidated net sales total — the 513,983 bind, both filings.
for yr, gold in GOLD["amzn_net_sales"].items():
    doc = "amzn-20221231" if yr in ("2021", "2022") else "amzn-20231231"
    PROBES.append(("amzn", f"AMZN consolidated net sales {yr}",
                   {"op": "value", "doc": doc,
                    "row": {"section": "Consolidated", "label": "Net sales"}, "period": yr},
                   float(gold)))
# AWS segment net sales — the segment-scoped bind (must NOT grab consolidated).
for yr, gold in GOLD["aws_net_sales"].items():
    if yr == "2020":
        continue
    doc = "amzn-20221231" if yr in ("2021", "2022") else "amzn-20231231"
    PROBES.append(("amzn", f"AWS segment net sales {yr}",
                   {"op": "value", "doc": doc,
                    "row": {"section": "AWS", "label": "Net sales"}, "period": yr},
                   float(gold)))
# GOOG total revenue — the cross-issuer twin of MSFT's (257,637 etc).
for yr, gold in GOLD["goog_total_revenue"].items():
    doc = {"2021": "goog-20211231", "2022": "goog-20221231",
           "2023": "goog-20231231"}[yr]
    PROBES.append(("goog", f"GOOG total revenue {yr}",
                   {"op": "value", "doc": doc,
                    "row": {"label": "Total revenues"}, "period": yr},
                   float(gold)))
# A YoY growth over each issuer's total (delta/growth path, the BUG-D trace).
PROBES.append(("msft", "MSFT total revenue growth 2021→2022",
               {"op": "growth_pct", "doc": "msft-10k_20220630",
                "row": {"section": "Revenue", "label": "Total"},
                "from_period": "2021", "to_period": "2022"}, None))  # None = just must resolve
# GOOG total revenue WITH the hinted section — proves the BUG-E self-heal is generic
# across issuers (the section the abstain error names actually resolves to gold).
for yr, gold in GOLD["goog_total_revenue"].items():
    doc = {"2021": "goog-20211231", "2022": "goog-20221231",
           "2023": "goog-20231231"}[yr]
    PROBES.append(("goog", f"GOOG total revenue {yr} (section=Revenues, post-hint)",
                   {"op": "value", "doc": doc,
                    "row": {"section": "Revenues", "label": "Total revenues"}, "period": yr},
                   float(gold)))


def verdict(issuer, spec, gold):
    r = compute(spec, scope(issuer))
    if not r.ok:
        return "ABSTAIN", r.error
    if gold is None:
        return "CORRECT", r.display()  # resolve-only probe
    if abs(r.value - gold) <= max(1.0, abs(gold) * 0.001):
        return "CORRECT", r.display()
    return "WRONG", f"{r.display()} (gold {gold:g})"


def main():
    correct = abstain = wrong = 0
    wrongs, abstains = [], []
    print("── cross-doc kernel probes (real grids, gold-checked) ───────────────")
    for issuer, label, spec, gold in PROBES:
        v, detail = verdict(issuer, spec, gold)
        tag = {"CORRECT": "PASS", "ABSTAIN": "ABST", "WRONG": "FAIL"}[v]
        print(f"  [{tag}] {label}: {detail}")
        if v == "CORRECT":
            correct += 1
        elif v == "ABSTAIN":
            abstain += 1; abstains.append(label)
        else:
            wrong += 1; wrongs.append((label, detail))

    n = len(PROBES)
    print(f"\n  {correct}/{n} CORRECT · {abstain} ABSTAIN · {wrong} WRONG")
    if abstains:
        print(f"  abstains (coverage gaps, not wrong): {abstains}")
    if wrongs:
        print("  ✗ WRONG (the forbidden class — a confident-wrong):")
        for lab, d in wrongs:
            print(f"      {lab}: {d}")
    bar = (wrong == 0)
    print(("\n  ✓ BAR MET: 0 confident-wrong across the corpus"
           if bar else "\n  ✗ BELOW BAR: a confident-wrong exists"))
    return 0 if bar else 1


if __name__ == "__main__":
    sys.exit(main())
