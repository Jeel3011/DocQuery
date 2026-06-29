"""Backfill `metadata.section_path` on existing TEXT chunks (DOCUMENT_HARNESS Phase 1.5).

Re-derives each text chunk's section heading FROM THE STORED CLEAN TEXT (no PDF re-parse,
no PDF_PARALLEL_WORKERS OOM risk) using the SAME detector the runtime heuristic uses
(agent_core.tools._outline._line_heading). New docs already get section_path at ingestion;
this is only for docs ingested before Phase 1.5.

⚠️ SCOPED TO LEGAL DOCS ON PURPOSE (corpus-driven decision, 2026-06-29). A dry-run over
the full corpus showed re-derivation from stored text is NOISY on financial/ESG docs —
table rows ("2021 Acquisition Activity", "12 Months or Greater") and body acronyms
("GDPR", "APAC", "BRSR") trip a generic heading detector and would write WRONG section
labels, sending read_section to the wrong span. The TRUSTWORTHY heading source is
`chunk_legal_prose`, which only runs on `doc_type == 'legal_contract'` and produces clean
clause headings ("19. Governing Law"). So this backfill ONLY touches legal_contract chunks
(use --all to override, NOT recommended). For old NON-legal docs the runtime heuristic
remains the fallback, and a full RE-INGEST is the clean recovery path when you want exact
financial-section labels — re-derivation can't beat capturing the heading at ingest time.

SAFE BY DEFAULT — dry run prints what it WOULD write and changes nothing:
    python -u scripts/backfill_section_path.py                  # dry, legal docs only
    python -u scripts/backfill_section_path.py --apply          # write, legal docs only
    python -u scripts/backfill_section_path.py --apply --doc <document_id>   # one doc
    python -u scripts/backfill_section_path.py --all            # override scope (noisy!)

Idempotent: skips chunks that already have section_path; only writes when a heading is
detected (no key churn for chunks with no heading). A chunk's other metadata is preserved
(read-modify-write of the JSONB blob). Service-role client (server-side maintenance).
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from src.components.db import get_supabase_client  # noqa: E402
from src.components.agent_core.tools._outline import _line_heading  # noqa: E402

_PAGE = 1000


def _derive(content: str):
    """First detected-heading line in the chunk text, or None."""
    if not content:
        return None
    for line in content.split("\n"):
        h = _line_heading(line)
        if h:
            return h
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill metadata.section_path on text chunks.")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--doc", default=None, help="limit to one document_id")
    ap.add_argument("--all", action="store_true",
                    help="override the legal-only scope (NOISY on financial docs — not recommended)")
    args = ap.parse_args()

    client = get_supabase_client(use_service_role=True)

    scope = "ALL doc types (noisy!)" if args.all else "legal_contract docs only"
    print(f"Scope: {scope}{' · doc=' + args.doc if args.doc else ''}\n")

    scanned = derived = already = written = 0
    offset = 0
    while True:
        q = (
            client.table("document_chunks")
            .select("id,content,metadata")
            .range(offset, offset + _PAGE - 1)
        )
        if args.doc:
            q = q.eq("document_id", args.doc)
        elif not args.all:
            # Corpus-driven scope: only the trustworthy clause-heading source.
            q = q.eq("metadata->>doc_type", "legal_contract")
        rows = q.execute().data or []
        if not rows:
            break
        offset += len(rows)

        for r in rows:
            md = r.get("metadata") or {}
            if md.get("chunk_type") not in (None, "text"):
                continue
            scanned += 1
            if md.get("section_path"):
                already += 1
                continue
            heading = _derive(r.get("content", "") or "")
            if not heading:
                continue
            derived += 1
            if args.apply:
                new_md = {**md, "section_path": heading}
                client.table("document_chunks").update({"metadata": new_md}).eq("id", r["id"]).execute()
                written += 1
            else:
                print(f"[dry] chunk {r['id']}: section_path = {heading!r}")

        if len(rows) < _PAGE:
            break

    mode = "APPLIED" if args.apply else "DRY RUN (no writes)"
    print(f"\n{mode}: scanned {scanned} text chunks · {already} already had section_path · "
          f"{derived} headings derived · {written} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
