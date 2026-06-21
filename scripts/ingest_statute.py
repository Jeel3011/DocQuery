"""G8.1b driver — ingest an Indian statute into the Knowledge Base.

The generalisation of `scripts/ingest_constitution.py` to the Tier-0 statute corpus. Same
two modes, the cheap one default (cheap-before-costly, the G8 prime rule):

  --dry-run  (DEFAULT, $0)  parse the text → provisions, ingest into an IN-MEMORY store
                            with a STUB vector writer (no Pinecone, no Supabase, no API),
                            print the receipt + the completeness diff. Proves the parser +
                            the machine on the real text for free.

  --live     ($, NEEDS JEEL'S GO)  the real ingest: embed every provision into the `kb_in`
                            Pinecone namespace (make_kb_vector_writer) AND write the
                            knowledge_provisions / knowledge_sources rows (SupabaseKnowledge-
                            Store + the SERVICE-ROLE client, the only writer the read-only-
                            shared RLS permits). Refuses to run without --as-of.

A statute is identified by a KEY in the Tier-0 registry below (so the metadata — title,
citation prefix, enacted date, source URL — lives in one place and the rest of the corpus
templates off it). The text itself is sourced manually (India Code / regulator portals) and
passed with --text PATH; this driver does not fetch it.

Usage:
    # $0, prove the parser on the real text:
    python -u scripts/ingest_statute.py --act indian_contract_act_1872 --text data/ica1872.txt --dry-run

    # $ live (Jeel's go), vouching the snapshot date:
    python -u scripts/ingest_statute.py --act indian_contract_act_1872 --text data/ica1872.txt \
        --live --as-of 2025-01-01
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

# ── Tier-0 statute registry (the corpus metadata in one place; §G8.1b) ──────────────
# Each entry is the structured metadata the parser stamps onto every provision + the
# source row. The text is sourced separately (India Code / regulator portal) and passed
# with --text. Adding a statute = one entry here + sourcing its text — the parser, the
# gates, and this driver are shared (prove-on-one-then-scale).
TIER0_STATUTES = {
    "indian_contract_act_1872": {
        "title": "The Indian Contract Act, 1872",
        "title_line": "THE INDIAN CONTRACT ACT, 1872",
        "citation_prefix": "Indian Contract Act 1872",
        "enacted_date": "1872-09-01",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2187",
    },
    "companies_act_2013": {
        "title": "The Companies Act, 2013",
        "title_line": "THE COMPANIES ACT, 2013",
        "citation_prefix": "Companies Act 2013",
        "enacted_date": "2013-08-30",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2114",
    },
    "income_tax_act_1961": {
        "title": "The Income-tax Act, 1961",
        "title_line": "THE INCOME-TAX ACT, 1961",
        "citation_prefix": "Income-tax Act 1961",
        "enacted_date": "1961-04-01",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2435",
    },
    "it_act_2000": {
        "title": "The Information Technology Act, 2000",
        "title_line": "THE INFORMATION TECHNOLOGY ACT, 2000",
        "citation_prefix": "IT Act 2000",
        "enacted_date": "2000-10-17",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1999",
    },
    "arbitration_act_1996": {
        "title": "The Arbitration and Conciliation Act, 1996",
        "title_line": "THE ARBITRATION AND CONCILIATION ACT, 1996",
        "citation_prefix": "Arbitration and Conciliation Act 1996",
        "enacted_date": "1996-08-22",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1978",
    },
    "cgst_act_2017": {
        "title": "The Central Goods and Services Tax Act, 2017",
        "title_line": "THE CENTRAL GOODS AND SERVICES TAX ACT, 2017",
        "citation_prefix": "CGST Act 2017",
        "enacted_date": "2017-07-01",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2367",
    },
}


def _print_receipt_and_completeness(receipt, store, source_key, warnings):
    """The ingest receipt + the §G8.4 completeness diff, computed from the relational
    store (the same number the committed gate uses). Shared by both modes."""
    print("\n" + receipt.render())
    if warnings:
        print(f"\n  parser warnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"    - {w}")
        if len(warnings) > 10:
            print(f"    … and {len(warnings) - 10} more")

    src = store.get_source(source_key)
    toc = list((src.toc_ids if src else []) or [])
    ingested = {p.section_or_article_id for p in store.list_provisions(source_key)}
    missing = [i for i in toc if i not in ingested]
    cov = (len(toc) - len(missing)) / len(toc) * 100 if toc else 0.0
    print(f"\n  §G8.4 completeness: {len(toc) - len(missing)}/{len(toc)} ({cov:.1f}%) "
          f"· partial={getattr(src, 'partial', None)}")
    if missing:
        print(f"  missing ids (first 20): {missing[:20]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest an Indian statute into the KB.")
    ap.add_argument("--act", required=True, choices=sorted(TIER0_STATUTES),
                    help="Which Tier-0 statute (its registry key).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to the official India Code statute PDF (preferred: "
                                   "its ARRANGEMENT OF SECTIONS gives the authoritative ToC).")
    src.add_argument("--text", help="Path to a pre-extracted plain-text statute file.")
    ap.add_argument("--dry-run", action="store_true", help="Parse + in-memory ingest, $0 (default if --live absent).")
    ap.add_argument("--live", action="store_true", help="Embed into kb_in + write Supabase rows ($, Jeel's go).")
    ap.add_argument("--as-of", dest="as_of", default=None,
                    help="Snapshot horizon (YYYY-MM-DD) you vouch the text is current through. Required for --live.")
    args = ap.parse_args()

    meta = TIER0_STATUTES[args.act]

    from src.components.knowledge.parse_statute import normalize_statute_text, parse_statute_text
    from src.components.knowledge.source_statute import source_statute_pdf, source_statute_text
    from src.components.knowledge.ingest import ingest_provisions

    # ── source the text: split the Act's own ARRANGEMENT OF SECTIONS (authoritative ToC)
    #    from the enacted body, so completeness is measured against the Act's OWN ToC ──
    path = args.pdf or args.text
    if not os.path.exists(path):
        print(f"error: source file not found: {path}")
        return 2
    title_line = meta["title_line"]
    statute_src = (source_statute_pdf(path, title_line) if args.pdf
                   else source_statute_text(path, title_line))
    if statute_src.warnings:
        print("source warnings:")
        for w in statute_src.warnings:
            print(f"  - {w}")
    print(f"sourced: {len(statute_src.toc_ids)} ToC ids · {len(statute_src.body_text)} body chars")

    norm = normalize_statute_text(statute_src.body_text)
    result = parse_statute_text(
        norm,
        source_key=args.act,
        title=meta["title"],
        citation_prefix=meta["citation_prefix"],
        enacted_date=meta["enacted_date"],
        as_of_date=args.as_of,
        source_url=meta.get("source_url"),
        toc_ids=statute_src.toc_ids or None,
    )
    print(result.summary())
    SOURCE_KEY = result.source.source_key

    # ── DRY RUN (default, $0) ─────────────────────────────────────────────────────
    if not args.live:
        from src.components.knowledge.store import InMemoryKnowledgeStore
        store = InMemoryKnowledgeStore()
        embedded_box = {"n": 0}

        def stub_writer(provisions):
            embedded_box["n"] = len(provisions)
            return len(provisions)

        if args.as_of is None:
            print("\n  ⓘ no --as-of given: provisions have no snapshot horizon, so the "
                  "live ingest WOULD reject them (is_valid). Dry-run still parses to show "
                  "structure; pass --as-of to validate the full path.")
        receipt = ingest_provisions(result.source, result.provisions, store, stub_writer)
        _print_receipt_and_completeness(receipt, store, SOURCE_KEY, result.warnings)
        print(f"\n  [dry-run] {embedded_box['n']} provisions would embed into kb_in. "
              f"No Pinecone/Supabase/API touched. $0.")
        print("  → re-run with --live --as-of YYYY-MM-DD (Jeel's go) to ingest for real.")
        return 0 if receipt.provisions_parsed > 0 else 1

    # ── LIVE ($, Jeel's go) ───────────────────────────────────────────────────────
    if args.as_of is None:
        print("\nerror: --live requires --as-of YYYY-MM-DD (you must vouch the snapshot date). Refusing.")
        return 2

    print("\n  ⚠ LIVE ingest: embeds into Pinecone (kb_in) + writes Supabase rows. This costs.")
    from src.components.config import Config
    from src.components.db import get_supabase_client
    from src.components.knowledge.ingest import make_kb_vector_writer
    from src.components.knowledge.store import SupabaseKnowledgeStore

    config = Config()
    store = SupabaseKnowledgeStore(get_supabase_client(use_service_role=True))
    writer = make_kb_vector_writer(config)

    receipt = ingest_provisions(result.source, result.provisions, store, writer)
    _print_receipt_and_completeness(receipt, store, SOURCE_KEY, result.warnings)
    print(f"\n  [live] ingested into namespace {getattr(config, 'KNOWLEDGE_NAMESPACE', 'kb_in')!r}.")
    print("  → set RUN_LIVE_KB=1 and run the knowledge gates against the live store.")
    return 0 if receipt.provisions_embedded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
