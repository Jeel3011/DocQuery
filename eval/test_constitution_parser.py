"""G8.1a — Constitution parser gate (offline, $0).

Proves the structured-source parser (`parse_constitution.py`) turns the official
"N. Title.—Body" text into correct, provision-granular `Provision`s BEFORE a rupee is
spent embedding — the G1 lesson ("prove the mechanism on real-format text first") applied
to the corpus. No API, no Pinecone, no Supabase: pure text → dataclasses → in-memory
ingest.

The fixture is a handful of REAL Articles in the real source format (Part header, em-dash
title boundary, a lettered/numbered sub-clause, an `[Art. N omitted ...]` line, and a
Schedule with its own numbered list) — the exact shapes that broke the first parser pass:
  - a Schedule's numbered body item ("1. Defence ...") must NOT become a fake Article 1;
  - the em-dash splits title from body;
  - Part context flows into the ToC path;
  - the parser never raises and reports anything it can't place as a warning.

Run: python -u eval/test_constitution_parser.py
"""

import sys
sys.path.insert(0, ".")

from src.components.knowledge.parse_constitution import parse_constitution_text
from src.components.knowledge.ingest import ingest_provisions
from src.components.knowledge.store import InMemoryKnowledgeStore


class Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")


SAMPLE = """PART III
FUNDAMENTAL RIGHTS

14. Equality before law.—The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

15. Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth.—(1) The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them.

19. Protection of certain rights regarding freedom of speech, etc.—(1) All citizens shall have the right—
(a) to freedom of speech and expression;

21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty except according to procedure established by law.

21A. Right to education.—The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.

[Art. 31 omitted by the Constitution (Forty-fourth Amendment) Act, 1978]

32. Remedies for enforcement of rights conferred by this Part.—(1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.

SEVENTH SCHEDULE
List I—Union List
1. Defence of India and every part thereof.
"""

SNAPSHOT = "2025-01-01"


def main() -> int:
    c = Check()
    r = parse_constitution_text(SAMPLE, as_of_date=SNAPSHOT)
    by_id = {p.section_or_article_id: p for p in r.provisions}

    print("── structure: articles + schedule, no fakes ─────────────────────")
    articles = [p for p in r.provisions if p.instrument_type == "article"]
    schedules = [p for p in r.provisions if p.instrument_type == "schedule"]
    c.ok(len(articles) == 6, "6 articles parsed (14,15,19,21,21A,32)")
    c.ok({p.section_or_article_id for p in articles} == {"14", "15", "19", "21", "21A", "32"},
         "the right article ids (incl. the letter-suffixed 21A)")
    c.ok(len(schedules) == 1 and schedules[0].section_or_article_id == "Schedule.7",
         "the Seventh Schedule parsed as a schedule provision")
    c.ok("1" not in by_id,
         "the Schedule's numbered list item ('1. Defence...') is NOT a fake Article 1")
    c.ok(r.warnings == [], "no parser warnings on clean source text")

    print("\n── title/body split (the em-dash boundary) ──────────────────────")
    c.ok(by_id["14"].title == "Equality before law", "Art.14 title split at the em-dash")
    c.ok(by_id["14"].text.startswith("The State shall not deny"),
         "Art.14 body is the text after the em-dash (no title bleed)")
    c.ok(by_id["21"].title == "Protection of life and personal liberty",
         "Art.21 title correct")
    c.ok("(1)" in by_id["19"].text and "freedom of speech" in by_id["19"].text,
         "Art.19 sub-clauses kept in the body")

    print("\n── citation + ToC path ──────────────────────────────────────────")
    c.ok(by_id["21"].citation == "Constitution Art.21", "citation id is 'Constitution Art.21'")
    c.ok(by_id["21A"].citation == "Constitution Art.21A", "letter-suffixed citation correct")
    c.ok("Part III" in (by_id["14"].toc_path or ""),
         "ToC path carries the Part context (Part III)")
    c.ok(schedules[0].citation == "Constitution Schedule.7", "schedule citation correct")

    print("\n── ToC ids = the completeness ground truth ──────────────────────")
    c.ok(r.source.toc_ids == ["14", "15", "19", "21", "21A", "32", "Schedule.7"],
         "source.toc_ids = every parsed provision id, in order")
    c.ok(r.source.as_of_date == SNAPSHOT and r.source.enacted_date == "1950-01-26",
         "source carries the snapshot horizon + commencement date")

    print("\n── end-to-end: parsed provisions ingest cleanly (in-memory, $0) ─")
    store = InMemoryKnowledgeStore()
    receipt = ingest_provisions(r.source, r.provisions, store, lambda ps: len(ps))
    c.ok(receipt.provisions_parsed == 7, "all 7 provisions ingest (none rejected)")
    c.ok(receipt.complete and receipt.toc_coverage == 1.0,
         "100% ToC coverage (every parsed id ingested) — the §G8.4 number")
    c.ok(all(p.as_of_date == SNAPSHOT for p in store.list_provisions()),
         "every ingested provision inherited the snapshot horizon (live-ingest valid)")

    print("\n── no --as-of ⇒ would be REJECTED at live ingest (the guarantee) ─")
    r2 = parse_constitution_text(SAMPLE, as_of_date=None)
    store2 = InMemoryKnowledgeStore()
    receipt2 = ingest_provisions(r2.source, r2.provisions, store2, lambda ps: len(ps))
    c.ok(receipt2.provisions_parsed == 0 and len(store2.list_provisions()) == 0,
         "without a vouchable as_of, every provision is rejected (version-or-abstain)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.1a parser gate GREEN (structure · title/body · ToC · ingest · $0)")
        print("  ⓘ Live ($): scripts/ingest_constitution.py --text <full> --live --as-of D")
        return 0
    print("  ✗ G8.1a parser gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
