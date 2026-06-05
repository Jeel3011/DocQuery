"""Accept-test for Phase 4.3 table extraction against KNOWN cell values.

Run: python eval/test_table_extraction.py
Requires the real 10-Ks in `test docs/` (gitignored). Asserts that the
deterministic extractor recovers exact cells from real financial statements —
the 100%-correct bar (§4b). This is a *correctness* test, not a smoke test.

How cells are addressed (GENERIC — no filing-specific logic):
each known answer is (doc, section, label, period, expected). We find the table
that has the requested `period` column and a row matching (section, label), then
compare its value. `section` and `label` come straight from the extractor's own
normalized output, so the same addressing works for any filing's structure.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.components.table_extraction import extract_tables_from_pdf

CORPUS = "test docs"

# (doc, section, label, period, expected) — verified by hand from the filings.
# section="" means a top-level row (no section header above it).
KNOWN = [
    # Amazon FY23 segment results: AWS net sales / operating income.
    ("amzn-20231231.pdf", "AWS", "Net sales", "2023", "90,757"),
    ("amzn-20231231.pdf", "AWS", "Net sales", "2022", "80,096"),
    ("amzn-20231231.pdf", "AWS", "Operating income", "2023", "24,631"),
    # Google FY23 consolidated income statement (top-level rows).
    ("goog-20231231.pdf", "", "Revenues", "2023", "307,394"),
    ("goog-20231231.pdf", "", "Net income", "2023", "73,795"),
    # Microsoft FY22 — DESCENDING period order (2022,2021,2020 L→R); the
    # geometry path must label the 198,270 column as 2022, not 2020.
    ("msft-10k_20220630.htm.pdf", "", "Total", "2022", "198,270"),
    ("msft-10k_20220630.htm.pdf", "", "Total", "2020", "143,015"),
]


def find_cell(tables, section, label, period, expected):
    """Generic: the value at (section, label, period) across the extracted tables.

    A label like "Total" / "Net sales" recurs across many statements, so we
    disambiguate to the table whose (section,label) row actually carries the
    expected figure in *some* period — then read the `period` cell and check it
    landed in the right column. This verifies cell PLACEMENT for a known value;
    it does not invent the value. Falls back to the first structural match.
    """
    fallback = (None, None)
    for tb in tables:
        if period not in tb.headers:
            continue
        for r in tb.rows:
            sec_ok = (section == "") or (section.lower() in r.get("section", "").lower())
            if sec_ok and label.lower() == r["label"].lower():
                row_has_expected = expected in [r.get(p, "") for p in tb.headers if p not in ("section", "label")]
                if row_has_expected:
                    return r.get(period, ""), tb        # the right statement
                if fallback == (None, None):
                    fallback = (r.get(period, ""), tb)   # structural match, keep as backup
    return fallback


def main():
    passed = failed = 0
    cache = {}
    for doc, section, label, period, expected in KNOWN:
        if doc not in cache:
            t = time.time()
            cache[doc] = (extract_tables_from_pdf(f"{CORPUS}/{doc}"), time.time() - t)
        tables, _dt = cache[doc]
        got, tb = find_cell(tables, section, label, period, expected)
        ok = (got == expected)
        passed += ok
        failed += (not ok)
        tag = "PASS" if ok else "FAIL"
        loc = tb.table_id if tb else "—"
        sec = f"{section}/" if section else ""
        print(f"  [{tag}] {doc} {sec}{label} {period}: got={got!r} want={expected!r} ({loc})")
    print()
    for doc, (tables, dt) in cache.items():
        print(f"  {doc}: {len(tables)} gated tables in {dt:.1f}s")
    print(f"\n  {passed}/{passed+failed} known cells correct"
          + ("  ✓ 100% bar met" if failed == 0 else "  ✗ BELOW BAR"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
