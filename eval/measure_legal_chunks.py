"""Measure prose/legal ingestion quality on a contract — the G1 instrument.

Runs the REAL ingestion parse+chunk path (`DataIngestionPipeline.process_documents`
+ `build_langchain_documents`) over a contract PDF and reports, with NUMBERS:

  - element categories `partition_pdf` emits (so the boilerplate strip targets real ones)
  - boilerplate pollution per chunk (Page N of M, bare URLs, print timestamps)
  - whether key clauses survive WHOLE vs. truncated mid-sentence
  - a crude "clause retrievability" proxy: for each probe clause, does the chunk that
    contains its body rank its keyword above the boilerplate noise?

This is the measure-before/after harness for GRAND_PLAN G1 (§7: "MEASURE before doing
G1b"). It does NOT touch Pinecone or burn API — pure local parse. Run:

    PDF_PARALLEL_WORKERS=1 python -u eval/measure_legal_chunks.py --doc test_law/Document.pdf

Compare the printed counts before vs. after the G1a boilerplate strip lands.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# OOM discipline (CLAUDE.md): never warm the multi-proc PDF pool locally.
os.environ.setdefault("PDF_PARALLEL_WORKERS", "1")

from src.components.data_ingestion import DocumentProcessor  # noqa: E402
from src.components.config import Config  # noqa: E402
from src.utils import _get_element_type  # noqa: E402

# Boilerplate signatures we EXPECT to find polluting EDGAR .htm→PDF contract chunks.
PAGE_N_OF_M = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
BARE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
PRINT_TS = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M\b", re.IGNORECASE)

# Probe clauses: a label we'd search for + a phrase that proves the body is intact.
PROBE_CLAUSES = [
    ("Governing Law", "governed by"),
    ("Indemnification", "indemnify"),
    ("Confidentiality", "confidential"),
    ("Arbitration", "arbitration"),
    ("Termination", "terminate"),
]


def _count_boilerplate(text: str) -> dict:
    return {
        "page_n_of_m": len(PAGE_N_OF_M.findall(text)),
        "bare_url": len(BARE_URL.findall(text)),
        "print_ts": len(PRINT_TS.findall(text)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="test_law/Document.pdf")
    args = ap.parse_args()

    doc_path = str(Path(args.doc).resolve())
    if not Path(doc_path).exists():
        print(f"ERROR: {doc_path} not found")
        return 2

    print(f"=== measuring legal-chunk quality: {args.doc} ===\n")

    pipe = DocumentProcessor(Config())
    elements = pipe.process_documents(file_paths=doc_path)
    print(f"parsed {len(elements)} elements")

    cats = Counter(_get_element_type(el) for el in elements)
    print("element categories:")
    for cat, n in cats.most_common():
        print(f"    {cat:20s} {n}")

    chunks = pipe.build_langchain_documents(elements=elements, pdf_path=doc_path)
    text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    doc_type = getattr(pipe, "_last_doc_type", "?")
    print(f"\ndoc_type (G1d): {doc_type}")
    print(f"chunks: {len(chunks)} total  ({len(text_chunks)} text, "
          f"{len(table_chunks)} table)")

    # --- boilerplate pollution across text chunks ---
    total_bp = Counter()
    polluted = 0
    for c in text_chunks:
        bp = _count_boilerplate(c.page_content)
        if any(bp.values()):
            polluted += 1
        for k, v in bp.items():
            total_bp[k] += v
    print(f"\nboilerplate in TEXT chunks: {polluted}/{len(text_chunks)} chunks polluted")
    print(f"    Page N of M hits : {total_bp['page_n_of_m']}")
    print(f"    bare URL hits    : {total_bp['bare_url']}")
    print(f"    print-ts hits    : {total_bp['print_ts']}")

    # --- fake-table noise (the footer-hallucinated-table failure) ---
    if table_chunks:
        print(f"\n⚠️  {len(table_chunks)} TABLE chunk(s) emitted on a prose contract "
              f"(G1c should drive this to ~0):")
        for c in table_chunks[:5]:
            print(f"    p{c.metadata.get('page_number')}: "
                  f"{c.page_content[:80]!r}")
    else:
        print("\nno table chunks emitted (good for a prose contract)")

    # --- clause survival: is each probe clause's body present, and whole? ---
    print("\nclause survival (label found · body phrase present · same chunk):")
    joined = "\n".join(c.page_content for c in text_chunks)
    for label, body_phrase in PROBE_CLAUSES:
        label_present = label.lower() in joined.lower()
        body_present = body_phrase.lower() in joined.lower()
        # whole = some single chunk holds BOTH the label and the body phrase
        whole = any(
            label.lower() in c.page_content.lower()
            and body_phrase.lower() in c.page_content.lower()
            for c in text_chunks
        )
        flag = "✓" if whole else ("~" if (label_present and body_present) else "✗")
        print(f"    {flag} {label:18s} label={label_present!s:5s} "
              f"body={body_present!s:5s} whole={whole}")

    print("\n(✓ = label+body in one chunk; ~ = both present but split across chunks; "
          "✗ = missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
