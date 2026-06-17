"""G8.0 infra gate — the Knowledge Base machine, before any corpus (G8 §G8.0).

Asserts the foundation the tool + all three knowledge gates stand on, on a tiny
fixture (a handful of Constitution Articles) with an IN-MEMORY store + a STUB vector
writer — $0, offline, no Pinecone, no Supabase, no API. The contracts:

  1. INGEST works end-to-end — provisions land as rows, vectors "embed" (stub count),
     a correct receipt comes back.
  2. EVERY provision carries a citation id + an as_of date (the version-in-force
     guarantee's prerequisite — a provision without as_of is REJECTED at ingest).
  3. The RECEIPT is correct — ToC coverage is the relational diff (§G8.4's shape),
     a gap is listed (never silent), partial flips the source's `partial` flag.
  4. READ-ONLY-SHARED invariant — the ingest path writes ONLY the KB namespace; the
     stub writer is the only thing that ever sees a vector (no user namespace touched).
  5. as_of PRIMITIVE — `is_in_force_on` returns True/False inside the vouchable window
     and **None (→ abstain) past the snapshot horizon** (§G8.5's core).

Run: python -u eval/test_knowledge_infra.py
"""

import sys
sys.path.insert(0, ".")

from src.components.knowledge import IngestReceipt, Provision, SourceMeta
from src.components.knowledge.ingest import ingest_provisions, _stamp_chunk_ids
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


# ── fixture: a handful of real Constitution Articles (Part III, fundamental rights) ──
# as_of = a snapshot horizon we vouch for; enacted = the Constitution's commencement.
SNAPSHOT = "2025-01-01"
COMMENCED = "1950-01-26"

FIXTURE_SOURCE = SourceMeta(
    source_key="constitution_of_india",
    title="The Constitution of India",
    instrument_type="article",
    citation_prefix="Constitution",
    enacted_date=COMMENCED,
    as_of_date=SNAPSHOT,
    # The source's OWN table of contents — the completeness ground truth. We list 5
    # Articles but only ingest 4 below, so the gate must SEE the gap (Art.15).
    toc_ids=["14", "15", "19", "21", "32"],
    source_url="https://legislative.gov.in/constitution-of-india",
)


def _article(num, title, text):
    return Provision(
        source_key="constitution_of_india",
        instrument_type="article",
        title=title,
        citation=f"Constitution Art.{num}",
        section_or_article_id=num,
        text=text,
        toc_path=f"Part III > Art.{num}",
        # as_of/enacted intentionally OMITTED — ingest must inherit them from the source.
    )


FIXTURE_PROVISIONS = [
    _article("14", "Equality before law",
             "The State shall not deny to any person equality before the law or the "
             "equal protection of the laws within the territory of India."),
    _article("19", "Protection of certain rights regarding freedom of speech etc.",
             "All citizens shall have the right to freedom of speech and expression."),
    _article("21", "Protection of life and personal liberty",
             "No person shall be deprived of his life or personal liberty except "
             "according to procedure established by law."),
    _article("32", "Remedies for enforcement of rights",
             "The right to move the Supreme Court by appropriate proceedings for the "
             "enforcement of the rights conferred by this Part is guaranteed."),
    # Note: Article 15 is in the ToC but deliberately NOT ingested → a measured gap.
]


def main() -> int:
    c = Check()
    store = InMemoryKnowledgeStore()

    # A stub vector writer: records what it was asked to "embed" (the KB namespace), so
    # the test proves NO user namespace is ever touched (only this stub sees vectors).
    embedded_box = {"provisions": []}

    def stub_writer(provisions):
        embedded_box["provisions"] = list(provisions)
        return len(provisions)

    print("── ingest (end-to-end on the fixture) ───────────────────────────")
    receipt = ingest_provisions(FIXTURE_SOURCE, list(FIXTURE_PROVISIONS), store, stub_writer)
    c.ok(isinstance(receipt, IngestReceipt), "ingest returns an IngestReceipt")
    c.ok(receipt.provisions_parsed == 4, "receipt: 4 provisions parsed")
    c.ok(receipt.provisions_embedded == 4, "receipt: 4 vectors embedded (stub count)")
    c.ok(len(store.list_provisions("constitution_of_india")) == 4,
         "store: 4 provision rows written")
    c.ok(store.get_source("constitution_of_india") is not None, "store: source row written")

    print("\n── every provision is citable + version-stamped (§0 / §G8.5) ────")
    rows = store.list_provisions("constitution_of_india")
    c.ok(all(r.citation and r.citation.startswith("Constitution Art.") for r in rows),
         "every provision has a citation id")
    c.ok(all(r.section_or_article_id for r in rows),
         "every provision has a section_or_article_id")
    c.ok(all(r.as_of_date == SNAPSHOT for r in rows),
         "every provision inherited the source's as_of_date (version-in-force prereq)")
    c.ok(all(r.enacted_date == COMMENCED for r in rows),
         "every provision inherited the source's enacted_date")
    c.ok(all(r.chunk_id for r in rows),
         "every provision got a stable chunk_id (row↔vector join key)")
    # The vectors carry the same citation + as_of so search_knowledge filters/gate read them.
    vmeta = [p.vector_metadata() for p in embedded_box["provisions"]]
    c.ok(all(m.get("citation") and m.get("as_of_date") and m.get("chunk_id") for m in vmeta),
         "every vector carries citation + as_of + chunk_id metadata")

    print("\n── completeness self-report (§G8.4 shape, at ingest) ────────────")
    c.ok(receipt.toc_total == 5 and receipt.toc_covered == 4,
         "receipt: ToC 4/5 covered (the diff against the source's own ToC)")
    c.ok(receipt.missing_ids == ["15"],
         "receipt: the gap (Art.15) is LISTED, never silent")
    c.ok(not receipt.complete and round(receipt.toc_coverage, 2) == 0.8,
         "receipt: coverage = 80%, flagged not-complete")
    src = store.get_source("constitution_of_india")
    c.ok(src is not None and src.partial is True,
         "store: source flagged partial (mirrors the finance fidelity self-report)")

    print("\n── reject a provision missing as_of (the guarantee's prereq) ────")
    store2 = InMemoryKnowledgeStore()
    bad = Provision(
        source_key="x", instrument_type="act", title="bad",
        citation="X s.1", section_or_article_id="1", text="some text",
        # as_of_date omitted AND the source has none → must be rejected.
    )
    bad_source = SourceMeta(source_key="x", title="X", instrument_type="act")  # no as_of
    r2 = ingest_provisions(bad_source, [bad], store2, lambda ps: len(ps))
    c.ok(r2.provisions_parsed == 0 and len(store2.list_provisions("x")) == 0,
         "a provision with no vouchable as_of is REJECTED (not silently ingested)")
    c.ok(any("as_of" in p for p in r2.problems),
         "receipt records WHY it was dropped (missing as_of)")

    print("\n── as_of primitive: in-force True/False, abstain past horizon ───")
    art21 = next(r for r in rows if r.section_or_article_id == "21")
    c.ok(art21.is_in_force_on("2020-06-01") is True,
         "Art.21 in force on a date inside the window → True")
    c.ok(art21.is_in_force_on("1949-01-01") is False,
         "Art.21 not in force before commencement → False")
    c.ok(art21.is_in_force_on("2030-01-01") is None,
         "a date PAST the snapshot horizon → None (abstain, never a guess)")
    # a repealed/superseded provision is a 'wrong cell' — never current.
    repealed = Provision(
        source_key="constitution_of_india", instrument_type="article", title="old",
        citation="Constitution Art.31", section_or_article_id="31", text="(repealed)",
        as_of_date=SNAPSHOT, enacted_date=COMMENCED, repealed=True,
    )
    c.ok(repealed.is_current() is False and repealed.is_in_force_on("2020-01-01") is False,
         "a repealed provision is never in force (treated like a wrong cell)")

    print("\n── chunk-id determinism (row↔vector join is stable) ─────────────")
    p1 = [_article("14", "t", "same text")]
    p2 = [_article("14", "t", "same text")]
    _stamp_chunk_ids("constitution_of_india", p1)
    _stamp_chunk_ids("constitution_of_india", p2)
    c.ok(p1[0].chunk_id == p2[0].chunk_id,
         "chunk_id is deterministic over (source, index, text)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.0 KB-infra gate GREEN (ingest · cite+version · completeness · as-of)")
        return 0
    print("  ✗ G8.0 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
