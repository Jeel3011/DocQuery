"""G1 regression gate — legal/prose ingestion (clause coverage).

The committed gate for GRAND_PLAN G1 (boilerplate strip + clause-aware chunking +
skip-table-on-prose + doc classifier), mirroring the finance completeness gate. It
asserts STRUCTURAL properties of the prose path on real contracts — no API, no
Pinecone, deterministic. Run:

    PDF_PARALLEL_WORKERS=1 python -u eval/test_legal_extraction.py

It does NOT exercise the review grid (that's G4). It proves the document-understanding
layer: a contract is classified legal, chunked by clause, kept clean of boilerplate,
its clauses survive WHOLE, and NO fake table chunks are emitted — while a financial
filing is still classified financial (so the table moat is never skipped on it).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PDF_PARALLEL_WORKERS", "1")

from src.components.config import Config  # noqa: E402
from src.components.data_ingestion import (  # noqa: E402
    DocumentProcessor,
    DOC_TYPE_LEGAL,
    DOC_TYPE_FINANCIAL,
)

CONTRACT = "test_law/Document.pdf"      # Sezzle EX-10.2 consulting agreement
FINANCE_SAMPLE = "test docs/amzn-20231231.pdf"  # a real 10-K (fence sentinel)

# Clauses the Sezzle contract demonstrably contains; each must be (a) present and
# (b) WHOLE — its heading/label and a body phrase in the SAME chunk (the G1b win).
CONTRACT_CLAUSES = [
    ("Governing Law", "governed by"),
    ("Indemnification", "indemnify"),
    ("Confidentiality", "confidential"),
    ("Arbitration", "arbitration"),
    ("Termination", "terminate"),
]

PAGE_N_OF_M = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
BARE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
PRINT_TS = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M\b", re.IGNORECASE)

_passed = 0
_failed = 0


def _check(cond: bool, label: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {label}")
    else:
        _failed += 1
        print(f"  FAIL {label}")


def _ingest(path: str):
    pipe = DocumentProcessor(Config())
    elements = pipe.process_documents(file_paths=str(Path(path).resolve()))
    chunks = pipe.build_langchain_documents(elements=elements,
                                            pdf_path=str(Path(path).resolve()))
    return pipe, chunks


def test_contract():
    print(f"\n[legal] {CONTRACT}")
    if not Path(CONTRACT).exists():
        print(f"  SKIP — {CONTRACT} not present"); return
    pipe, chunks = _ingest(CONTRACT)
    text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]

    # G1d: classified legal
    _check(getattr(pipe, "_last_doc_type", None) == DOC_TYPE_LEGAL,
           f"classified legal_contract (got {getattr(pipe, '_last_doc_type', None)})")

    # G1c: no fake table chunks on a prose contract
    _check(len(table_chunks) == 0,
           f"no table chunks emitted on prose (got {len(table_chunks)})")

    # G1a: zero boilerplate pollution across text chunks
    joined = "\n".join(c.page_content for c in text_chunks)
    bp = len(PAGE_N_OF_M.findall(joined)) + len(BARE_URL.findall(joined)) + len(PRINT_TS.findall(joined))
    _check(bp == 0, f"no page-chrome in text chunks (page-no/URL/timestamp hits={bp})")

    # G1b: clause-aware chunking produced clause-sized chunks (not 1 giant blob)
    _check(len(text_chunks) >= 8,
           f"clause-aware chunking yields multiple clause chunks (got {len(text_chunks)})")

    # Clause survival: each clause label + body present, and WHOLE (same chunk)
    for label, body in CONTRACT_CLAUSES:
        whole = any(label.lower() in c.page_content.lower()
                    and body.lower() in c.page_content.lower()
                    for c in text_chunks)
        _check(whole, f"clause whole in one chunk: {label}")


def test_finance_fence():
    """The classifier must NEVER call a financial filing 'legal' (that would skip its
    table pass — a catastrophic finance regression). The deep finance gates live in
    extraction_benchmark/test_table_extraction; this is the cheap sentinel here."""
    print(f"\n[fence] {FINANCE_SAMPLE}")
    if not Path(FINANCE_SAMPLE).exists():
        print(f"  SKIP — {FINANCE_SAMPLE} not present"); return
    pipe, chunks = _ingest(FINANCE_SAMPLE)
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    _check(getattr(pipe, "_last_doc_type", None) == DOC_TYPE_FINANCIAL,
           f"10-K classified financial_filing (got {getattr(pipe, '_last_doc_type', None)})")
    _check(len(table_chunks) > 0,
           f"financial-table pass still ran on the 10-K (table chunks={len(table_chunks)})")


def main() -> int:
    print("=" * 60)
    print("G1 legal-extraction gate (clause coverage; prose path)")
    print("=" * 60)
    test_contract()
    test_finance_fence()
    print("=" * 60)
    print(f"  {_passed} passed, {_failed} failed")
    if _failed:
        print("  GATE RED")
        return 1
    print("  GATE GREEN ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
