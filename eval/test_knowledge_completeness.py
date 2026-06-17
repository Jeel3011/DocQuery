"""G8.4 — statutory completeness gate: "knows the law" as a NUMBER (G8 §G8.4).

The legal twin of `eval/test_extraction_completeness.py` (which cross-checks finance
grids against the PDF's own text layer). Here we cross-check the INGESTED provisions
against each source's OWN authoritative table of contents — captured into
`knowledge_sources.toc_ids` at ingest — and assert coverage is 100% or the gap is
LISTED and the source flagged `partial`. A silently-missing section is a confident-wrong
waiting to happen (the agent would cite a NEAR section instead).

This is the §G8.4 number: coverage = ingested_ids ∩ toc_ids over toc_ids, computed from
the relational ground truth — **no API, no Pinecone, no embeddings, $0**. It runs now on
the `InMemoryKnowledgeStore` + a fixture; the same diff runs against the live store after
the Constitution ingest (G8.1a), per-source.

Run: python -u eval/test_knowledge_completeness.py
"""

import sys
sys.path.insert(0, ".")

from src.components.knowledge import Provision, SourceMeta
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


SNAPSHOT = "2025-01-01"
COMMENCED = "1950-01-26"


def _article(num, title, text):
    return Provision(
        source_key="constitution_of_india",
        instrument_type="article",
        title=title,
        citation=f"Constitution Art.{num}",
        section_or_article_id=num,
        text=text,
        toc_path=f"Part III > Art.{num}",
    )


# ── the §G8.4 diff, decoupled from the ingest receipt so the GATE owns the number ──
def completeness(store: InMemoryKnowledgeStore, source_key: str):
    """The committed completeness check, computed purely from the relational store.

    Returns (toc_total, covered, missing_ids, extra_ids). `extra_ids` = ingested
    provisions NOT in the source's own ToC (a parser over-segmentation / a wrong id) —
    the finance completeness gate's twin to a hallucinated row. Both directions matter:
    a gap is a silent miss; an extra is a provision we can't vouch is real authority.
    """
    src = store.get_source(source_key)
    toc = list((src.toc_ids if src else []) or [])
    ingested = {p.section_or_article_id for p in store.list_provisions(source_key)}
    missing = [i for i in toc if i not in ingested]
    extra = sorted(ingested - set(toc))
    covered = len(toc) - len(missing)
    return len(toc), covered, missing, extra


def main() -> int:
    c = Check()

    # ── Case 1: a COMPLETE source — every ToC id ingested → 100%, not partial ────────
    print("── complete source: full ToC coverage = the 100% number ─────────")
    store = InMemoryKnowledgeStore()
    full_source = SourceMeta(
        source_key="constitution_of_india",
        title="The Constitution of India",
        instrument_type="article",
        citation_prefix="Constitution",
        enacted_date=COMMENCED, as_of_date=SNAPSHOT,
        toc_ids=["14", "19", "21", "32"],
    )
    full_provisions = [
        _article("14", "Equality before law", "The State shall not deny ... equality before the law."),
        _article("19", "Freedom of speech etc.", "All citizens shall have the right to freedom of speech."),
        _article("21", "Protection of life and personal liberty", "No person shall be deprived of his life or personal liberty except according to procedure established by law."),
        _article("32", "Remedies for enforcement of rights", "The right to move the Supreme Court ... is guaranteed."),
    ]
    ingest_provisions(full_source, list(full_provisions), store, lambda ps: len(ps))
    total, covered, missing, extra = completeness(store, "constitution_of_india")
    c.ok(total == 4 and covered == 4, "complete: 4/4 ToC ids covered")
    c.ok(missing == [], "complete: no missing ids")
    c.ok(extra == [], "complete: no extra (un-ToC'd) provisions")
    c.ok(store.get_source("constitution_of_india").partial is False,
         "complete: source NOT flagged partial")

    # ── Case 2: a GAP — Art.15 in the ToC but not ingested → listed + flagged ────────
    print("\n── gap: a missing section is LISTED, never silent (§G8.4) ───────")
    store2 = InMemoryKnowledgeStore()
    gap_source = SourceMeta(
        source_key="constitution_of_india",
        title="The Constitution of India",
        instrument_type="article",
        citation_prefix="Constitution",
        enacted_date=COMMENCED, as_of_date=SNAPSHOT,
        toc_ids=["14", "15", "19", "21", "32"],   # 5 expected
    )
    ingest_provisions(gap_source, list(full_provisions), store2, lambda ps: len(ps))  # only 4
    total, covered, missing, extra = completeness(store2, "constitution_of_india")
    c.ok(total == 5 and covered == 4, "gap: 4/5 covered (80%)")
    c.ok(missing == ["15"], "gap: the missing id (Art.15) is LISTED")
    c.ok(store2.get_source("constitution_of_india").partial is True,
         "gap: source flagged partial (mirrors finance fidelity self-report)")

    # ── Case 3: an EXTRA — a provision whose id isn't in the source's ToC ────────────
    print("\n── extra: a provision not in the source's own ToC is surfaced ───")
    store3 = InMemoryKnowledgeStore()
    extra_source = SourceMeta(
        source_key="constitution_of_india",
        title="The Constitution of India",
        instrument_type="article",
        citation_prefix="Constitution",
        enacted_date=COMMENCED, as_of_date=SNAPSHOT,
        toc_ids=["14", "19", "21"],   # ToC does NOT list 21A
    )
    provisions_with_extra = full_provisions[:3] + [
        _article("21A", "Right to education", "The State shall provide free and compulsory education to all children of the age of six to fourteen years."),
    ]
    ingest_provisions(extra_source, list(provisions_with_extra), store3, lambda ps: len(ps))
    total, covered, missing, extra = completeness(store3, "constitution_of_india")
    c.ok("21A" in extra, "extra: an un-ToC'd provision (Art.21A) is surfaced as extra")
    c.ok(missing == [], "extra: ToC ids still all covered")

    # ── Case 4: the gate is the HARD LINE — a committed source must be 100% ──────────
    # This asserts the discipline the live G8.1a run will be held to: after ingest, the
    # COMMITTED gate fails the build if any in-scope source is partial. Here we prove the
    # assertion machinery on the in-memory fixture; the live source plugs in at G8.1a.
    print("\n── committed-gate discipline: in-scope sources must be complete ─")
    live_complete = (completeness(store, "constitution_of_india")[2] == [])  # store = Case 1
    c.ok(live_complete, "discipline: a complete source passes the committed bar")
    # And a partial one would FAIL — proven by Case 2's non-empty missing list.
    c.ok(completeness(store2, "constitution_of_india")[2] != [],
         "discipline: a partial source is caught (would fail the committed bar)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.4 completeness gate GREEN (ToC-vs-ingested diff · gap listed · $0)")
        print("  ⓘ Live bar (G8.1a, Jeel's go): the SAME diff over the Supabase store must")
        print("    report 100% for each ingested source, or the gap is listed + source partial.")
        return 0
    print("  ✗ G8.4 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
