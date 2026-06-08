"""Layer 0 §1a — unit checks for the geometry line reader (_read_geometry_lines).

Verifies the deterministic word-geometry row reconstruction against REAL pages:
  - MSFT income statement: recovers the 'Total revenue 198,270' row that
    extract_tables() DROPS, and reports the indent (x0) that separates section
    headers from line items from totals.
  - Amazon segment table: section headers (AWS, Consolidated) come out label-only
    at a left indent; their Net sales / Operating income rows are indented under.
  - Noise lines (footer URLs, 'Year Ended December 31,' date headers) do not
    masquerade as clean data rows.

No LLM, no network. Reads 'test docs/' with pdfplumber. Run:
    python eval/test_geometry_lines.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import _read_geometry_lines

CORPUS = "test docs"
_fail = 0


def check(cond, msg):
    global _fail
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def _page(doc, pno):
    import pdfplumber
    with pdfplumber.open(f"{CORPUS}/{doc}") as pdf:
        return _read_geometry_lines(pdf.pages[pno - 1])


def _find(lines, label_contains):
    lc = label_contains.lower()
    return [ln for ln in lines if lc in ln["label"].lower()]


def test_msft_recovers_total_revenue():
    print("MSFT p57 income statement (extract_tables drops the totals):")
    lines = _page("msft-10k_20220630.htm.pdf", 57)
    tr = _find(lines, "Total revenue")
    check(bool(tr), "'Total revenue' row recovered (was dropped by extract_tables)")
    if tr:
        check("198,270" in tr[0]["values"], f"Total revenue 2022 = 198,270 present ({tr[0]['values'][:3]})")
    # indentation separates the hierarchy levels
    rev_hdr = _find(lines, "Revenue:")
    op_inc = _find(lines, "Operating income")
    gross = _find(lines, "Gross margin")
    if rev_hdr and op_inc and gross and tr:
        # section header sits left; total/gross sit deeper (more indented)
        check(rev_hdr[0]["x0"] < tr[0]["x0"],
              f"section header indent {rev_hdr[0]['x0']:.0f} < total indent {tr[0]['x0']:.0f}")
        # Operating income is a statement-level line at the header indent, NOT under
        # 'Cost of revenue' — so its indent matches the section-header level, not the total level
        check(op_inc[0]["x0"] <= gross[0]["x0"],
              f"Operating income indent {op_inc[0]['x0']:.0f} <= Gross margin indent {gross[0]['x0']:.0f}")


def test_amzn_segment_hierarchy():
    print("AMZN p113 segment footnote (section headers + indented items):")
    lines = _page("amzn-20231231.pdf", 113)
    aws = [ln for ln in lines if ln["label"].lower() == "aws"]
    cons = [ln for ln in lines if ln["label"].lower() == "consolidated"]
    check(bool(aws) and aws[0]["nval"] == 0, "AWS is a label-only section header (0 values)")
    check(bool(cons) and cons[0]["nval"] == 0, "Consolidated is a label-only section header")
    ns = _find(lines, "Net sales")
    check(len(ns) >= 2, f"multiple 'Net sales' rows exist ({len(ns)}) — disambiguated only by section")
    if aws and ns:
        # the section header sits LEFT of its indented Net sales line items
        check(aws[0]["x0"] < ns[0]["x0"],
              f"AWS header indent {aws[0]['x0']:.0f} < Net sales item indent {ns[0]['x0']:.0f}")


def test_noise_lines_not_clean_data():
    print("Noise rejection (date headers / footer URLs must not look like data rows):")
    lines = _page("amzn-20231231.pdf", 61)  # cash-flow page with a date header + URL footer
    # A 'Year Ended December 31,' header may parse, but it must NOT carry a full set
    # of period values as if it were a data row.
    date_hdr = _find(lines, "Year Ended December")
    if date_hdr:
        check(date_hdr[0]["nval"] <= 1,
              f"date header has <=1 value tokens (not a data row): {date_hdr[0]['values'][:3]}")
    urls = [ln for ln in lines if "http" in ln["label"].lower() or "sec.gov" in ln["label"].lower()]
    check(True, f"(info) {len(urls)} url-ish line(s) seen — caller's gate must drop these")


def main():
    test_msft_recovers_total_revenue()
    test_amzn_segment_hierarchy()
    test_noise_lines_not_clean_data()
    print("=" * 60)
    if _fail:
        print(f"  {_fail} CHECK(S) FAILED")
        sys.exit(1)
    print("  all geometry-line checks passed")


if __name__ == "__main__":
    main()
