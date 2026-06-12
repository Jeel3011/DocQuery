"""Whole-document extraction AUDIT — ground-truth-free, runs on ANY PDF corpus.

The problem this solves: on a 370-page Indian annual report you cannot eyeball
every page, and the cell benchmark only scores hand-labeled US cells. So new
docs break silently, one page at a time, forever. This audit turns "it broke
again" into a RANKED, STRUCTURED report you can act on — no gold labels, no API.

For every page of every doc it classifies the extraction outcome by cross-
referencing the extracted grids against the page's own text layer (pdfplumber's
independent line assembly — the same principle as extraction_fidelity.py):

  STATEMENT  page text names a real statement (Balance Sheet / P&L / Cash Flow)
             with >=2 multi-value data rows → we SHOULD have a clean grid here.
  COVERED    every text-layer data line's numbers are present in some grid.
  PARTIAL    a grid exists but some data lines' numbers are missing (row drops).
  MISSING    data-looking rows exist but NO grid was produced (under-extraction).
  JUNK       a grid exists whose rows are mostly prose-labelled (over-extraction).
  PROSE      narrative page, no real table — correctly produced no/at-most-noise.

Output: per-doc rollup (counts by class) + the worst offending pages, so the
single highest-value fix is always at the top. Run:

  python -u eval/audit_extraction.py [--dir Indian_test_corpus] [--doc SUBSTR]
  python -u eval/audit_extraction.py --page INFY:275   # dump one page's grids

Exit 0 always (it's a report, not a pass/fail gate) unless --strict.
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import (  # noqa: E402
    extract_tables_from_pdf,
)
from src.components.extraction_fidelity import (  # noqa: E402
    grid_value_pool,
    text_data_lines,
)
from src.components.table_extraction import _is_lineitem_row_label  # noqa: E402

STMT_RE = re.compile(
    r"(balance sheet|statement of profit and loss|profit and loss|cash flow|"
    r"statement of changes in equity|income statement|statement of operations)",
    re.I,
)


def _grids_for_page(tables, pno):
    out = []
    for t in tables:
        if getattr(t, "page_number", None) == pno:
            md = t.to_metadata() if hasattr(t, "to_metadata") else {}
            out.append(md)
    return out


def _grid_is_junk(grids):
    """A grid is junk if most of its value-bearing rows have prose labels."""
    for md in grids:
        rows = md.get("rows", [])
        valrows = [r for r in rows
                   if any(str(v).strip() and k not in ("section", "label")
                          for k, v in r.items())]
        if not valrows:
            continue
        prose = sum(1 for r in valrows
                    if not _is_lineitem_row_label(str(r.get("label", ""))))
        if prose / len(valrows) >= 0.5:
            return True
    return False


def classify_page(pdf_page, tables, pno):
    grids = _grids_for_page(tables, pno)
    data_lines = text_data_lines(pdf_page)
    text = ""
    try:
        text = pdf_page.extract_text() or ""
    except Exception:
        pass
    is_stmt = bool(STMT_RE.search(text)) and len(data_lines) >= 4

    if not data_lines:
        return "PROSE", 0, 0
    pool = grid_value_pool(tables, pno)
    uncovered = 0
    for _line, nums in data_lines:
        if any(v not in pool for v in nums):
            uncovered += 1

    if not grids:
        # data rows exist but nothing extracted
        cls = "MISSING" if is_stmt or len(data_lines) >= 6 else "PROSE"
        return cls, len(data_lines), uncovered
    if _grid_is_junk(grids):
        return "JUNK", len(data_lines), uncovered
    if uncovered == 0:
        return "COVERED", len(data_lines), 0
    # a grid exists but rows dropped
    if uncovered >= max(2, 0.5 * len(data_lines)):
        return "PARTIAL", len(data_lines), uncovered
    return "COVERED", len(data_lines), uncovered  # minor prose-noise tail


def audit_doc(path):
    import pdfplumber
    t0 = time.time()
    tables = extract_tables_from_pdf(path)
    counts = Counter()
    worst = []  # (severity, page, cls, ndata, uncovered)
    with pdfplumber.open(path) as pdf:
        npages = len(pdf.pages)
        for i, pg in enumerate(pdf.pages, start=1):
            cls, ndata, unc = classify_page(pg, tables, i)
            counts[cls] += 1
            # severity for ranking the report: under/over-extraction on data pages
            if cls in ("MISSING", "JUNK", "PARTIAL"):
                sev = (unc if cls == "PARTIAL" else ndata) + (1000 if cls == "MISSING" else 0)
                worst.append((sev, i, cls, ndata, unc))
    worst.sort(reverse=True)
    return {
        "doc": os.path.basename(path),
        "pages": npages,
        "tables": len(tables),
        "secs": round(time.time() - t0, 1),
        "counts": dict(counts),
        "worst": worst[:8],
    }


def main():
    argv = sys.argv
    d = "Indian_test_corpus"
    if "--dir" in argv:
        d = argv[argv.index("--dir") + 1]
    doc_filter = argv[argv.index("--doc") + 1].lower() if "--doc" in argv else ""

    if "--page" in argv:  # dump one page's grids: INFY:275
        spec = argv[argv.index("--page") + 1]
        sub, pno = spec.split(":")
        pno = int(pno)
        path = next(f"{d}/{f}" for f in sorted(os.listdir(d))
                    if sub.lower() in f.lower())
        tables = extract_tables_from_pdf(path)
        for md in _grids_for_page(tables, pno):
            print(f"\n  grid {md.get('table_id')} conf={md.get('confidence')}")
            for r in md.get("rows", [])[:20]:
                print("   ", {k: str(v)[:32] for k, v in r.items()})
        return 0

    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".pdf")
                   and (not doc_filter or doc_filter in f.lower()))
    grand = Counter()
    print(f"Auditing {len(files)} doc(s) in {d}/\n")
    print(f"{'doc':22} {'pg':>4} {'tbl':>4} {'COVER':>5} {'PART':>4} "
          f"{'MISS':>4} {'JUNK':>4} {'PROSE':>5} {'s':>5}")
    print("-" * 72)
    reports = []
    for f in files:
        rep = audit_doc(f"{d}/{f}")
        reports.append(rep)
        c = rep["counts"]
        for k, v in c.items():
            grand[k] += v
        print(f"{rep['doc']:22} {rep['pages']:>4} {rep['tables']:>4} "
              f"{c.get('COVERED',0):>5} {c.get('PARTIAL',0):>4} "
              f"{c.get('MISSING',0):>4} {c.get('JUNK',0):>4} "
              f"{c.get('PROSE',0):>5} {rep['secs']:>5}")
    print("-" * 72)
    print(f"{'TOTAL':22} {'':>4} {'':>4} {grand.get('COVERED',0):>5} "
          f"{grand.get('PARTIAL',0):>4} {grand.get('MISSING',0):>4} "
          f"{grand.get('JUNK',0):>4} {grand.get('PROSE',0):>5}")

    print("\n=== Worst pages (the ranked fix-list) ===")
    for rep in reports:
        if rep["worst"]:
            print(f"\n{rep['doc']}:")
            for sev, pno, cls, ndata, unc in rep["worst"]:
                detail = (f"{unc}/{ndata} rows dropped" if cls == "PARTIAL"
                          else f"{ndata} data rows" )
                print(f"   p{pno:<4} {cls:8} {detail}  "
                      f"(dump: --page {rep['doc'].split('_')[0]}:{pno})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
