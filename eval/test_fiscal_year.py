"""Fiscal-year derivation gate — G3 Step C (structural, $0, NULL when unsure).

`derive_fiscal_year` drives the vault's FY filter chip. It must be STRUCTURAL only —
no LLM, no guess — and return None whenever it isn't certain (a mis-derived FY would
silently hide the right doc; G3 §5 risk #3). This gate asserts:

  1. The 8-doc finance corpus filenames derive the CORRECT fiscal year (issuer period-
     end dates: amzn-20221231 → 2022, goog-20231231 → 2023, msft-10k_20220630 → 2022).
  2. An EDGAR accession-number filename (no period-end) → None from the filename alone
     (it resolves later from period headers — never guessed).
  3. Explicit FY tags (FY2023 / fy23) and bare year tokens derive correctly.
  4. Period-header fallback (grid.periods) takes the LATEST year (the filing's primary FY).
  5. Null-safety + sanity window: None/empty/garbage/out-of-window → None, never raises.

Fully offline — no PDF parse, no DB, no model. Run: python -u eval/test_fiscal_year.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.data_ingestion import derive_fiscal_year


class Checks:
    def __init__(self):
        self.passed = self.failed = 0
    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


def main():
    c = Checks()

    print("\n── 1. finance corpus filenames → correct fiscal year ────────────")
    corpus = {
        "amzn-20221231.pdf": 2022,
        "amzn-20231231.pdf": 2023,
        "goog-20211231.pdf": 2021,
        "goog-20221231.pdf": 2022,
        "goog-20231231.pdf": 2023,
        "msft-10k_20210630.htm.pdf": 2021,  # FY ends June 30 → fiscal year is the YEAR
        "msft-10k_20220630.htm.pdf": 2022,
    }
    for fn, want in corpus.items():
        got = derive_fiscal_year(fn)
        c.ok(got == want, f"{fn} → FY{got} (want {want})")

    print("\n── 2. accession-number filename → None (no guess; periods resolve it) ─")
    # 0000950170-23-035122.pdf IS MSFT FY2023, but the filename carries no period-end —
    # so the filename signal must be None, NOT a guess off the '23' or '035122'.
    c.ok(derive_fiscal_year("0000950170-23-035122.pdf") is None,
         "EDGAR accession filename → None (refuses to guess)")
    # …and it resolves from the statements' period headers at ingest.
    c.ok(derive_fiscal_year("0000950170-23-035122.pdf", ["2021", "2022", "2023"]) == 2023,
         "accession doc + periods → 2023 (latest reported year)")

    print("\n── 3. explicit FY tags + bare year tokens ───────────────────────")
    c.ok(derive_fiscal_year("annual_report_FY2023.pdf") == 2023, "FY2023 tag → 2023")
    c.ok(derive_fiscal_year("acme-fy23-10k.pdf") == 2023, "fy23 (2-digit) → 2023")
    c.ok(derive_fiscal_year("Tesla Annual Report 2022.pdf") == 2022, "bare year 2022 → 2022")

    print("\n── 4. period-header fallback takes the LATEST year ──────────────")
    c.ok(derive_fiscal_year(None, ["2020", "2021", "2022"]) == 2022,
         "periods [2020,2021,2022] → 2022 (primary FY = latest)")
    c.ok(derive_fiscal_year(None, ["Year Ended December 31, 2023", "2022"]) == 2023,
         "messy period header strings → 2023")
    # Filename WINS over periods (it's the precise period-end).
    c.ok(derive_fiscal_year("amzn-20221231.pdf", ["2099"]) == 2022,
         "filename FY wins over period headers")

    print("\n── 5. null-safety + sanity window (never guess, never raise) ────")
    c.ok(derive_fiscal_year(None) is None, "None filename → None")
    c.ok(derive_fiscal_year("") is None, "empty filename → None")
    c.ok(derive_fiscal_year("contract-final-v3.pdf") is None, "no year anywhere → None")
    c.ok(derive_fiscal_year("report-18991231.pdf") is None, "year below window (1899) → None")
    c.ok(derive_fiscal_year(None, ["12345", "abc", "31"]) is None,
         "garbage periods → None (no stray match)")
    c.ok(derive_fiscal_year(12345) is None or isinstance(derive_fiscal_year(12345), int),
         "non-string filename → no raise")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G3 Step C fiscal_year gate GREEN (structural · null-safe · no guess)")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
