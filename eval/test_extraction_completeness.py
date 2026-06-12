"""Extraction COMPLETENESS gate — no silent row drops on statement pages.

The cell benchmark (extraction_benchmark.py) scores the cells the ground truth
samples; a wholly MISSING row is invisible to it unless the ground truth happens
to include that row. That is exactly how the MSFT FY23 'Research and development'
row (27,195 — label/values split 3.0008pt across y-bands) stayed hidden behind a
"100%" cell score. This gate closes the blind spot with a value-coverage check
that needs no per-row ground truth:

  For every statement page named in extraction_ground_truth.json, every
  "data line" in the page's RAW TEXT layer (pdfplumber's own line assembly —
  independent of the geometry/banding pipeline under test) must have ALL its
  numeric values present somewhere in the extracted grids of that page.

A data line = a text line carrying >=2 value tokens that are not all year/date
tokens (period-header lines like "Year Ended June 30, 2023 2022 2021" are column
headers, not data). Dropping a value line is the failure class; this makes it
loud, generically, for any filing.

No LLM, no network. Run: python -u eval/test_extraction_completeness.py [--doc SUBSTR]
"""
import json
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import (  # noqa: E402
    _FOOTNOTE_TOK,
    _VALUE_TOK,
    extract_tables_from_pdf,
)

CORPUS = "test docs"
GT_PATH = "eval/extraction_ground_truth.json"


from src.components.extraction_fidelity import (  # noqa: E402 — single source of truth
    grid_value_pool as _grid_value_pool,
    text_data_lines as _text_data_lines,
)


def main():
    doc_filter = ""
    if "--doc" in sys.argv:
        doc_filter = sys.argv[sys.argv.index("--doc") + 1].lower()

    gt = json.load(open(GT_PATH))
    pages_checked = lines_checked = 0
    failures = []
    cache = {}

    for d in gt["docs"]:
        doc = d["doc"]
        if doc_filter and doc_filter not in doc.lower():
            continue
        if doc not in cache:
            t0 = time.time()
            cache[doc] = extract_tables_from_pdf(f"{CORPUS}/{doc}")
            print(f"  {doc}: {len(cache[doc])} gated tables in {time.time()-t0:.1f}s")
        tables = cache[doc]

        import pdfplumber
        with pdfplumber.open(f"{CORPUS}/{doc}") as pdf:
            for tspec in d["tables"]:
                page_no = tspec["page"]
                pool = _grid_value_pool(tables, page_no)
                data_lines = _text_data_lines(pdf.pages[page_no - 1])
                pages_checked += 1
                for line, vals in data_lines:
                    lines_checked += 1
                    missing = [v for v in vals if v not in pool]
                    if missing:
                        failures.append((doc, page_no, tspec["name"], line, missing))

    print()
    for doc, page_no, name, line, missing in failures:
        print(f"  MISSING {doc} p{page_no} ({name}):")
        print(f"          line   : {line[:100]!r}")
        print(f"          values absent from extracted grids: {missing}")
    print("=" * 60)
    print(f"  {pages_checked} statement pages · {lines_checked} text-layer data lines checked")
    if failures:
        print(f"  {len(failures)} DATA LINE(S) NOT COVERED by extracted grids — silent row drop(s)")
        return 1
    print("  every text-layer data line is covered by the extracted grids ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
