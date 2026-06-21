"""G8.1b — statute parser gate (the Constitution parser's twin for Indian Acts).

Proves `parse_statute_text` on a real-format Indian Contract Act 1872 fixture BEFORE any
spend (the same dry-run-first discipline that caught 5 real bugs in the Constitution parse
before a rupee was spent). Asserts the structural deltas a statute adds over the
Constitution:

  - Sections cite "<Act> s.N" (the `s.` joiner), instrument_type "act".
  - CHAPTER / PART headers set the ToC path.
  - A repealed-stub Section ("[Repealed.]") is flagged repealed (a withheld cell).
  - A Schedule is ONE opaque provision, never sub-parsed into fake Sections.
  - The enacting formula / long title before s.1 is front matter (warnings), not a Section.
  - Footnote lines that look like openers ("1. Subs. by Act ...") fold into the body.
  - The per-Act §G8.4 completeness diff (ToC-vs-ingested) is computable and 100% on a
    clean fixture, and LISTS the gap on a deliberately incomplete one.

$0, offline, no API/Pinecone/Supabase. Run: python -u eval/test_statute_parser.py
"""

import sys
sys.path.insert(0, ".")

from src.components.knowledge.parse_statute import (
    normalize_statute_text,
    parse_statute_text,
)
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


# A real-format slice of the Indian Contract Act 1872: an enacting preamble, two Chapters,
# numbered Sections with em-dash marginal-note titles, sub-clauses, a repealed-stub Section,
# a footnote line that mimics a Section opener, and a Schedule. Hard-wrapped on purpose so
# the normalizer is exercised. (Section text is paraphrased/abbreviated; structure faithful.)
ICA_TEXT = """THE INDIAN CONTRACT ACT, 1872
ACT NO. 9 OF 1872
An Act to define and amend certain parts of the law relating to contracts.

CHAPTER I
PRELIMINARY

1. Short title.—This Act may be called the Indian Contract Act, 1872.
Extent, commencement.—It extends to the whole of India and shall come into force
on the first day of September, 1872.

2. Interpretation clause.—In this Act the following words and expressions are used
in the following senses, unless a contrary intention appears from the context:—
(a) When one person signifies to another his willingness to do or to abstain from
doing anything, with a view to obtaining the assent of that other, he is said to
make a proposal;
(b) When the person to whom the proposal is made signifies his assent thereto, the
proposal is said to be accepted.

CHAPTER II
OF CONTRACTS, VOIDABLE CONTRACTS AND VOID AGREEMENTS

10. What agreements are contracts.—All agreements are contracts if they are made by
the free consent of parties competent to contract, for a lawful consideration and
with a lawful object, and are not hereby expressly declared to be void.

11. Who are competent to contract.—Every person is competent to contract who is of
the age of majority according to the law to which he is subject, and who is of sound
mind, and is not disqualified from contracting by any law to which he is subject.

12. [Repealed.] Rep. by the Repealing and Amending Act, 1914 (10 of 1914), s. 3.

1. Subs. by Act 4 of 1882, for the original section.

73. Compensation for loss or damage caused by breach of contract.—When a contract
has been broken, the party who suffers by such breach is entitled to receive, from
the party who has broken the contract, compensation for any loss or damage caused
to him thereby.

THE SCHEDULE
Enactments repealed.—The enactments specified in this Schedule are repealed to the
extent mentioned in the third column thereof.
1. Act 6 of 1871 — The whole.
2. Act 9 of 1872 — Sections 76 to 123.
"""


def main() -> int:
    c = Check()

    norm = normalize_statute_text(ICA_TEXT)
    res = parse_statute_text(
        norm,
        source_key="indian_contract_act_1872",
        title="The Indian Contract Act, 1872",
        citation_prefix="Indian Contract Act 1872",
        enacted_date="1872-09-01",
        as_of_date="2025-01-01",
        source_url="https://www.indiacode.nic.in/handle/123456789/2187",
    )
    print(res.summary())
    by_id = {p.section_or_article_id: p for p in res.provisions}

    # ── Sections parsed with the s. joiner + act instrument type ────────────────────
    print("\n── sections: '<Act> s.N' citation, instrument_type 'act' ────────")
    c.ok({"1", "2", "10", "11", "73"}.issubset(by_id.keys()),
         "all five real Sections (1,2,10,11,73) parsed")
    c.ok(by_id["10"].citation == "Indian Contract Act 1872 s.10",
         "s.10 cites 'Indian Contract Act 1872 s.10' (the s. joiner)")
    c.ok(by_id["10"].instrument_type == "act", "Section instrument_type is 'act'")
    c.ok("free consent" in by_id["10"].text and "lawful consideration" in by_id["10"].text,
         "s.10 body is the verbatim provision text (quotable)")
    c.ok(by_id["10"].title.lower().startswith("what agreements"),
         "s.10 marginal-note title split from the body on the em-dash")

    # ── Chapter headers set the ToC path ────────────────────────────────────────────
    print("\n── chapters: header sets the ToC path ────────────────────────────")
    c.ok("Chapter I" in (by_id["1"].toc_path or ""), "s.1 carries Chapter I in its ToC path")
    c.ok("Chapter II" in (by_id["10"].toc_path or ""), "s.10 carries Chapter II in its ToC path")

    # ── repealed-stub Section → flagged repealed (a withheld cell) ──────────────────
    print("\n── repealed: a '[Repealed.]' Section is flagged, never current ───")
    c.ok("12" in by_id, "the repealed Section 12 is still parsed (citable stub)")
    c.ok(by_id["12"].repealed is True, "s.12 flagged repealed")

    # ── footnote masquerading as an opener folds into the body, not a fake Section ──
    print("\n── footnote discriminator: '1. Subs. by Act ...' is NOT a Section ─")
    # The footnote reuses id "1"; the real s.1 (with the em-dash heading) must win, and
    # the footnote must not create a duplicate/garbage Section.
    c.ok(by_id["1"].title.lower().startswith("short title"),
         "the REAL s.1 (Short title) is kept, not the 'Subs. by Act' footnote")

    # ── Schedule is ONE opaque provision, never sub-parsed ─────────────────────────
    print("\n── schedule: one provision, not fake Sections from its list items ─")
    scheds = [p for p in res.provisions if p.instrument_type == "schedule"]
    c.ok(len(scheds) == 1, "exactly one Schedule provision")
    c.ok(scheds[0].section_or_article_id == "Schedule.1", "Schedule id is 'Schedule.1'")
    c.ok("Act 6 of 1871" in scheds[0].text and "Act 9 of 1872" in scheds[0].text,
         "the Schedule's list items live INSIDE the Schedule provision (not as Sections)")
    # Its numbered items ("1.", "2.") must NOT have become Sections — only the real s.1/s.2.
    c.ok(by_id["2"].title.lower().startswith("interpretation"),
         "s.2 is the real Interpretation clause, not a Schedule item")

    # ── front matter (enacting formula / long title) → warnings, not a provision ────
    print("\n── front matter: enacting formula before s.1 is not a provision ──")
    c.ok(any("before first provision" in w for w in res.warnings),
         "pre-s.1 enacting formula reported as a warning, never a fake provision")
    c.ok(all(p.section_or_article_id not in ("ACT", "An") for p in res.provisions),
         "no garbage provision from the long title / ACT NO. line")

    # ── per-Act §G8.4 completeness: ToC-vs-ingested diff is 100% on a clean parse ────
    print("\n── §G8.4 completeness (per-Act): clean parse = 100%, gap LISTED ──")
    store = InMemoryKnowledgeStore()
    ingest_provisions(res.source, list(res.provisions), store, lambda ps: len(ps))
    src = store.get_source("indian_contract_act_1872")
    ingested_ids = {p.section_or_article_id for p in store.list_provisions("indian_contract_act_1872")}
    missing = [i for i in (src.toc_ids or []) if i not in ingested_ids]
    c.ok(missing == [], "clean parse: every ToC id ingested (100%)")
    c.ok(src.partial is False, "clean parse: source NOT flagged partial")

    # A deliberately incomplete source LISTS the gap + flags partial (the hard line).
    store2 = InMemoryKnowledgeStore()
    gap_source = res.source
    # Pretend the official ToC has an s.74 we failed to parse.
    from copy import deepcopy
    gap_source2 = deepcopy(res.source)
    gap_source2.toc_ids = list(gap_source2.toc_ids) + ["74"]
    ingest_provisions(gap_source2, list(res.provisions), store2, lambda ps: len(ps))
    src2 = store2.get_source("indian_contract_act_1872")
    ingested2 = {p.section_or_article_id for p in store2.list_provisions("indian_contract_act_1872")}
    missing2 = [i for i in (src2.toc_ids or []) if i not in ingested2]
    c.ok(missing2 == ["74"], "gap: the missing Section (s.74) is LISTED")
    c.ok(src2.partial is True, "gap: source flagged partial (the committed bar would fail)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.1b statute parser gate GREEN (sections · chapters · repealed · schedule · completeness)")
        print("  ⓘ Live bar (Jeel's go): the SAME diff over the Supabase store must report")
        print("    100% for each ingested Act, or the gap is listed + source flagged partial.")
        return 0
    print("  ✗ G8.1b statute parser gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
