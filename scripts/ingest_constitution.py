"""G8.1a driver — ingest the Constitution of India into the Knowledge Base.

Two modes, the cheap one default (cheap-before-costly, the G8 prime rule):

  --dry-run  (DEFAULT, $0)  parse the text → provisions, ingest into an IN-MEMORY store
                            with a STUB vector writer (no Pinecone, no Supabase, no API),
                            print the receipt + the completeness diff. This proves the
                            parser + the machine on the real text for free.

  --live     ($, NEEDS JEEL'S GO)  the real ingest: embed every provision into the `kb_in`
                            Pinecone namespace (via make_kb_vector_writer) AND write the
                            knowledge_provisions / knowledge_sources rows (via the
                            SupabaseKnowledgeStore + the SERVICE-ROLE client, the only
                            writer the read-only-shared RLS permits). Refuses to run
                            without --as-of (you must consciously vouch the snapshot date).

Source text: pass a normalized plain-text file with --text PATH. (Sourcing + normalizing
the official India Code / legislative.gov.in text is a separate, manual step — this driver
does not fetch it.) The parser is permissive but expects the "N. Title.—Body" convention;
see parse_constitution.py.

Usage:
    # $0, prove the parser + machine on the real text:
    python -u scripts/ingest_constitution.py --text data/constitution.txt --dry-run

    # $ live ingest (Jeel's go), vouching the text current as of a date:
    python -u scripts/ingest_constitution.py --text data/constitution.txt --live --as-of 2025-01-01
"""

import argparse
import os
import sys

sys.path.insert(0, ".")


def _print_receipt_and_completeness(receipt, store, source_key, warnings):
    """Shared reporting for both modes: the ingest receipt + the §G8.4 completeness diff
    computed straight from the relational store (the same number the committed gate uses)."""
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
    ap = argparse.ArgumentParser(description="Ingest the Constitution of India into the KB.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Path to a single normalized plain-text Constitution file.")
    src.add_argument("--from-dir", dest="from_dir",
                     help="Directory of captn3m0/constitution .txt files (Preamble + PART*.txt + "
                          "SCHEDULE*.txt). Assembled in canonical order, encoding-repaired + reflowed.")
    ap.add_argument("--dry-run", action="store_true", help="Parse + in-memory ingest, $0 (default if --live absent).")
    ap.add_argument("--live", action="store_true", help="Embed into kb_in + write Supabase rows ($, Jeel's go).")
    ap.add_argument("--as-of", dest="as_of", default=None,
                    help="Snapshot horizon (YYYY-MM-DD) you vouch the text is current through. Required for --live.")
    args = ap.parse_args()

    from src.components.knowledge.parse_constitution import (
        parse_constitution_text, normalize_constitution_text, repair_encoding,
        make_schedule_provision,
    )
    from src.components.knowledge.ingest import ingest_provisions

    # ── source the text ───────────────────────────────────────────────────────────
    if args.from_dir:
        if not os.path.isdir(args.from_dir):
            print(f"error: directory not found: {args.from_dir}")
            return 2
        # Parse the PART files for ARTICLES (clean numbered units → the completeness gate).
        # SCHEDULE files are added SEPARATELY, one opaque provision each (a Schedule is a
        # list/table cited as a unit — sub-parsing it manufactured fake Articles; dry-run
        # caught that). Preamble is front matter, not a provision.
        part_order = ["PART1", "PART2", "PART3", "PART4", "PART4A", "PART5", "PART6", "PART7",
                      "PART8", "PART9", "PART9A", "PART9B", "PART10", "PART11", "PART12", "PART13",
                      "PART14", "PART14A", "PART15", "PART16", "PART17", "PART18", "PART19", "PART20",
                      "PART21", "PART22"]
        parts, used = [], []
        for name in part_order:
            p = os.path.join(args.from_dir, f"{name}.txt")
            if os.path.exists(p):
                parts.append(repair_encoding(open(p, "rb").read()))
                used.append(name)
        if not parts:
            print(f"error: no PART*.txt files found in {args.from_dir}")
            return 2
        raw = normalize_constitution_text("\n".join(parts))
        result = parse_constitution_text(raw, as_of_date=args.as_of)

        # Append each Schedule as one provision (1..12), in order.
        sched_added = []
        for n in range(1, 13):
            sp = os.path.join(args.from_dir, f"SCHEDULE{n}.txt")
            if os.path.exists(sp):
                stext = normalize_constitution_text(repair_encoding(open(sp, "rb").read()))
                # as_of/enacted are inherited from the source at ingest (ingest_provisions).
                prov = make_schedule_provision(n, stext)
                result.provisions.append(prov)
                result.source.toc_ids.append(prov.section_or_article_id)
                sched_added.append(n)
        print(f"assembled {len(used)} PART files + {len(sched_added)} schedules "
              f"(Preamble excluded — not a provision)")
    else:
        if not os.path.exists(args.text):
            print(f"error: text file not found: {args.text}")
            return 2
        raw = normalize_constitution_text(repair_encoding(open(args.text, "rb").read()))
        result = parse_constitution_text(raw, as_of_date=args.as_of)

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
    print("  → set RUN_LIVE_KB=1 and run the three knowledge gates against the live store.")
    return 0 if receipt.provisions_embedded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
