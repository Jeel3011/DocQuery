"""Extraction-accuracy benchmark against hand-labeled ground truth.

Run: python -u eval/extraction_benchmark.py            # full corpus
     python -u eval/extraction_benchmark.py --doc amzn-20231231.pdf   # one doc
     python -u eval/extraction_benchmark.py --cells     # also list every cell verdict

This is the FOUNDATION metric for the Brain/Spine work: everything downstream
(comprehend -> plan -> kernel -> monitor) sits on the extracted grids, so we
measure the grids honestly FIRST. Ground truth in eval/extraction_ground_truth.json
was transcribed visually from the rendered PDFs (non-circular w.r.t. the
pdfplumber grid pipeline under test). Requires the real 10-Ks in `test docs/`.

Metrics (per-doc, per-statement-kind, and overall):
  * period_detection  - did the extracted table expose ALL the expected year
                        columns?  (this is the "44% have year columns" headline)
  * period_order      - are those year columns in the correct L->R order?
                        (ascending for Amazon/Google, descending for Microsoft)
  * section_coverage  - fraction of expected section headers the extractor kept
  * cell_accuracy     - did each labeled value land in the right (label, period)
                        slot?  (sign/format-normalized, so it tests PLACEMENT)

A table that can't be located at all scores 0 on every metric for that table
(an honest miss, not a skipped row).
"""
import sys, os, json, time, argparse, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.components.table_extraction import extract_tables_from_pdf

CORPUS = "test docs"
GROUND_TRUTH = "eval/extraction_ground_truth.json"
YEAR_RX = re.compile(r"^(19|20)\d{2}$")


# ── normalization ─────────────────────────────────────────────────────────────
def norm_num(s):
    """Sign/format-agnostic numeric key: '$ (1,665 )' -> '-1665', '—' -> ''."""
    s = (s or "").strip().replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("—", "").replace("–", "")        # em/en dash -> empty
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return s


def norm_label(s):
    """Lowercase, collapse whitespace, straighten curly apostrophes, drop a
    trailing colon (extractor stores 'Operating expenses', gt has the printed
    'Operating expenses:')."""
    s = (s or "").replace("’", "'").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1].strip() if s.endswith(":") else s


def years_in_order(headers):
    """The year-like column headers, left-to-right."""
    return [h for h in headers if YEAR_RX.match(str(h).strip())]


def is_subsequence(small, big):
    """True if `small` appears in `big` in the same relative order."""
    it = iter(big)
    return all(x in it for x in small)


# ── locate the extracted table(s) that correspond to a ground-truth statement ──
def locate_pool(tables, gt_table):
    """Return the POOL of extracted fragments making up this statement.

    A single statement is often split into several same-page fragments (e.g. the
    income statement body in one grid and the 'Net income / EPS' tail in another).
    So we pick the PAGE that best matches the statement (most labeled-row hits,
    ties broken by proximity to the expected page) and return every fragment on
    that page. Page-anchored matching stops a shared label like 'Net income'
    from dragging the income query onto the cash-flow table.
    """
    exp_page = gt_table["page"]
    exp_periods = set(gt_table["expected_periods"])

    # Anchor on the EXPECTED page (hand-verified from the rendered PDF), then pool
    # it and the next page (statements read top-down, so a tail like 'Net income
    # / EPS' often spills over) — keeping only fragments whose detected periods
    # match this statement, which excludes a neighbouring statement (e.g. the
    # comprehensive-income or cash-flow page that also has a 'Net income' row).
    # A label like 'Net income' recurs on 4 pages, so we DON'T vote by label.
    pool = [tb for tb in tables
            if tb.page_number in (exp_page, exp_page + 1)
            and exp_periods.issubset(set(map(str, tb.periods)) or {""})]
    if pool:
        return pool
    # fall back: any fragment on the expected page (period detection may have failed)
    return [tb for tb in tables if tb.page_number == exp_page]


def find_row(pool, label):
    """First row across the pool whose label matches (prefer one with values)."""
    nl = norm_label(label)
    fallback = None
    for tb in pool:
        for r in tb.rows:
            if norm_label(r.get("label", "")) == nl:
                if any(v for k, v in r.items() if k not in ("section", "label")):
                    return r
                fallback = fallback or r
    return fallback


# ── scoring ───────────────────────────────────────────────────────────────────
def score_statement(pool, gt, show_cells, doc):
    """Return (period_ok, order_ok, sec_found, sec_total, cells_ok, cells_total)."""
    exp_periods = gt["expected_periods"]
    if not pool:
        n = sum(len(c["values"]) for c in gt["cells"])
        if show_cells:
            print(f"      [MISS] statement not located: {gt['name']}")
        return False, False, 0, len(gt["expected_sections"]), 0, n

    # detection over the union of the pool's detected periods; order from the
    # fragment carrying the most year columns (the table body).
    union_periods = set()
    for tb in pool:
        union_periods |= set(map(str, tb.periods)) or set(years_in_order(tb.headers))
    rep = max(pool, key=lambda tb: len(years_in_order(tb.headers)))
    rep_years = years_in_order(rep.headers) or list(map(str, rep.periods))
    period_ok = all(p in union_periods for p in exp_periods)
    order_ok = period_ok and is_subsequence(exp_periods, rep_years)

    # section coverage: union of section fields across the pool's rows
    row_sections = " || ".join(norm_label(r.get("section", "")) for tb in pool for r in tb.rows)
    sec_found = sum(1 for s in gt["expected_sections"] if norm_label(s) in row_sections)
    sec_total = len(gt["expected_sections"])

    # cells: locate each labeled row anywhere in the pool
    cells_ok = cells_total = 0
    for c in gt["cells"]:
        row = find_row(pool, c["label"])
        for period, expected in c["values"].items():
            cells_total += 1
            got = row.get(period, "") if row else ""
            ok = norm_num(got) == norm_num(expected) and norm_num(expected) != ""
            cells_ok += ok
            if show_cells:
                tag = "ok " if ok else "XX "
                print(f"      [{tag}] {c['label'][:38]:38s} {period}: "
                      f"got={got!r:>12} want={expected!r:>12}")
    if show_cells:
        po = "ok" if period_ok else "XX"
        oo = "ok" if order_ok else "XX"
        ids = ",".join(tb.table_id for tb in pool)
        print(f"      periods {po} {sorted(union_periods)} (want {exp_periods}) | "
              f"order {oo} | sections {sec_found}/{sec_total} | {ids}")
    return period_ok, order_ok, sec_found, sec_total, cells_ok, cells_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="restrict to one doc filename")
    ap.add_argument("--cells", action="store_true", help="print every cell verdict")
    args = ap.parse_args()

    gt = json.load(open(GROUND_TRUTH))["docs"]
    if args.doc:
        gt = [d for d in gt if d["doc"] == args.doc]
        if not gt:
            print(f"no ground truth for {args.doc}"); return 1

    # aggregates
    agg = {"pd_ok": 0, "pd_n": 0, "ord_ok": 0, "sec_f": 0, "sec_t": 0, "c_ok": 0, "c_n": 0}
    by_kind = {}

    for d in gt:
        path = f"{CORPUS}/{d['doc']}"
        if not os.path.exists(path):
            print(f"  SKIP {d['doc']} (not in {CORPUS}/)"); continue
        t0 = time.time()
        tables = extract_tables_from_pdf(path)
        dt = time.time() - t0
        print(f"\n=== {d['doc']}  ({d['issuer']} FY{d['fiscal_year']}) — "
              f"{len(tables)} gated tables in {dt:.1f}s ===")
        for gtab in d["tables"]:
            if args.cells:
                print(f"   • {gtab['kind']}: {gtab['name']}")
            pool = locate_pool(tables, gtab)
            po, oo, sf, st, cok, cn = score_statement(pool, gtab, args.cells, d["doc"])
            agg["pd_ok"] += po; agg["pd_n"] += 1; agg["ord_ok"] += oo
            agg["sec_f"] += sf; agg["sec_t"] += st; agg["c_ok"] += cok; agg["c_n"] += cn
            k = by_kind.setdefault(gtab["kind"],
                                   {"pd_ok": 0, "n": 0, "ord_ok": 0, "c_ok": 0, "c_n": 0})
            k["pd_ok"] += po; k["n"] += 1; k["ord_ok"] += oo; k["c_ok"] += cok; k["c_n"] += cn
            if not args.cells:
                flag = "ok" if (po and cok == cn) else "  "
                print(f"   [{flag}] {gtab['kind']:16s} periods={'Y' if po else 'n'} "
                      f"order={'Y' if oo else 'n'} sections={sf}/{st} "
                      f"cells={cok}/{cn}")

    def pct(a, b):
        return f"{100*a/b:5.1f}%" if b else "  n/a"

    print("\n" + "=" * 60)
    print("BY STATEMENT KIND")
    for kind, k in sorted(by_kind.items()):
        print(f"  {kind:16s}  period-detect {pct(k['pd_ok'], k['n'])}  "
              f"order {pct(k['ord_ok'], k['n'])}  cells {pct(k['c_ok'], k['c_n'])}"
              f"  ({k['pd_ok']}/{k['n']} tables, {k['c_ok']}/{k['c_n']} cells)")

    print("\nOVERALL (the honest current state)")
    print(f"  period detection : {pct(agg['pd_ok'], agg['pd_n'])}   "
          f"({agg['pd_ok']}/{agg['pd_n']} tables expose all expected year columns)")
    print(f"  period order     : {pct(agg['ord_ok'], agg['pd_n'])}   "
          f"({agg['ord_ok']}/{agg['pd_n']} in correct L->R order)")
    print(f"  section coverage : {pct(agg['sec_f'], agg['sec_t'])}   "
          f"({agg['sec_f']}/{agg['sec_t']} expected section headers kept)")
    print(f"  cell accuracy    : {pct(agg['c_ok'], agg['c_n'])}   "
          f"({agg['c_ok']}/{agg['c_n']} labeled cells correctly placed)")
    bar = 90.0
    headline = 100 * agg["c_ok"] / agg["c_n"] if agg["c_n"] else 0
    print(f"\n  {'PASS' if headline >= bar else 'BELOW'} 90% cell bar "
          f"(cell accuracy {headline:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
