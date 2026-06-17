"""G8.5 — version-in-force (as-of) gate (G8 §G8.5).

The Harvey-informed guarantee (§1): snapshot + gate, NOT a hand-built citator. A dated
question returns the law in force on that date; a provision we cannot vouch for on that
date — repealed/superseded, not-yet-enacted, or past our snapshot horizon — is WITHHELD,
never cited as current. This is the legal twin of the kernel withholding a wrong cell.

Two layers, both $0/offline:
  1. The PRIMITIVE — `Provision.is_in_force_on(date)`: True/False inside the vouchable
     window, **None (→ abstain) past the snapshot horizon**.
  2. The TOOL FILTER — `search_knowledge`'s `_vouchable_on` post-filter, exercised end-
     to-end through the real tool against a STUB KB manager (no Pinecone, no API): an
     as_of query DROPS the spans not in force and reports how many were withheld.

Run: python -u eval/test_knowledge_asof.py
"""

import sys
sys.path.insert(0, ".")

from src.components.knowledge import Provision

# The tool + its filter, exercised directly (the same code the live agent calls).
from src.components.agent_core.tools.knowledge import search_knowledge, _vouchable_on


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


def _prov(num, *, enacted=COMMENCED, as_of=SNAPSHOT, repealed=False, superseded_by=None):
    return Provision(
        source_key="constitution_of_india", instrument_type="article",
        title=f"Article {num}", citation=f"Constitution Art.{num}",
        section_or_article_id=num, text=f"Text of Article {num}.",
        enacted_date=enacted, as_of_date=as_of,
        repealed=repealed, superseded_by=superseded_by,
    )


# ── a tiny doc + stub retrieval manager (mirrors how test_tools.py stubs the KB) ──
class _StubDoc:
    """A retrieved KB chunk shaped like what RetrievalManager returns: page_content +
    metadata. Built straight from a Provision's vector_metadata so the tool serializes
    it exactly as it would a live span."""

    def __init__(self, provision: Provision):
        self.page_content = provision.text
        self.metadata = provision.vector_metadata()


class _StubKB:
    """A stub KB RetrievalManager — returns its canned docs, no network. Ignores the
    metadata_filter (the as_of post-filter is what this gate tests)."""

    def __init__(self, provisions):
        self._docs = [_StubDoc(p) for p in provisions]

    def retrieve(self, query, **kwargs):
        return list(self._docs)


def main() -> int:
    c = Check()

    # ── Layer 1: the in-force PRIMITIVE ──────────────────────────────────────────────
    print("── primitive: is_in_force_on (True / False / None=abstain) ───────")
    art21 = _prov("21")
    c.ok(art21.is_in_force_on("2020-06-01") is True,
         "in-window date → True (in force)")
    c.ok(art21.is_in_force_on("1949-01-01") is False,
         "before enacted → False (not yet in force)")
    c.ok(art21.is_in_force_on("2030-01-01") is None,
         "past snapshot horizon → None (ABSTAIN, never a guess)")
    c.ok(art21.is_in_force_on(None) is True,
         "no date asked → current-snapshot semantics (True)")

    repealed = _prov("31", repealed=True)
    c.ok(repealed.is_in_force_on("2020-01-01") is False and repealed.is_current() is False,
         "repealed → never in force (treated like a wrong cell)")
    superseded = _prov("19A", superseded_by="Constitution Art.19")
    c.ok(superseded.is_in_force_on("2020-01-01") is False,
         "superseded → never current")

    # ── Layer 2: the TOOL's _vouchable_on filter (off the serialized span) ───────────
    print("\n── filter: _vouchable_on mirrors the primitive on span metadata ─")
    live_md = {"enacted_date": COMMENCED, "as_of_date": SNAPSHOT, "repealed": False, "superseded_by": None}
    c.ok(_vouchable_on(live_md, "2020-06-01") is True, "live span on in-window date → kept")
    c.ok(_vouchable_on(live_md, "1949-01-01") is False, "before enacted → dropped")
    c.ok(_vouchable_on(live_md, "2030-01-01") is False, "past horizon → dropped (withheld, never guessed)")
    c.ok(_vouchable_on(live_md, None) is True, "no as_of → kept (not repealed)")
    c.ok(_vouchable_on({**live_md, "repealed": True}, "2020-06-01") is False, "repealed span → dropped")
    c.ok(_vouchable_on({**live_md, "superseded_by": "X"}, "2020-06-01") is False, "superseded span → dropped")

    # ── Layer 3: END-TO-END through the real tool + a stub KB manager ────────────────
    print("\n── end-to-end: search_knowledge(as_of=...) withholds correctly ──")
    # A mix: a live Article, a repealed one, and one we can't vouch (its own horizon is
    # earlier than the asked date). On an in-window date only the live one survives.
    kb = _StubKB([
        _prov("21"),                                   # live, vouchable through 2025
        _prov("31", repealed=True),                    # repealed → withhold
        _prov("370", as_of="2018-12-31"),              # horizon BEFORE the asked date → abstain/withhold
    ])

    res = search_knowledge("right to life", kb, as_of="2020-06-01")
    c.ok(res.get("ok") is True, "tool returns ok envelope (never raises)")
    passages = res.get("data", {}).get("passages", [])
    cits = {p.get("citation") for p in passages}
    c.ok(cits == {"Constitution Art.21"},
         "only the in-force provision is returned (Art.21); repealed + past-horizon withheld")
    c.ok(res["data"]["withheld"] == 2, "tool reports withheld=2 (the two not-in-force)")
    c.ok("withheld" in res.get("summary", ""), "summary notes the withholding (visible, not silent)")

    # With NO as_of, the repealed one is STILL dropped (a wrong cell is never current),
    # but the past-horizon one is fine under current-snapshot semantics.
    res2 = search_knowledge("right to life", kb)  # no as_of
    cits2 = {p.get("citation") for p in res2["data"]["passages"]}
    c.ok("Constitution Art.31" not in cits2,
         "no-as_of: a repealed provision is STILL withheld (never current)")
    c.ok("Constitution Art.21" in cits2 and "Constitution Art.370" in cits2,
         "no-as_of: live + past-horizon provisions returned (current-snapshot semantics)")

    # The provenance spans are what the citation gate binds — confirm they carry the
    # citation marker so a returned passage flows through the EXISTING cite-or-abstain gate.
    c.ok(all(p.get("doc") == p.get("citation") for p in passages),
         "every returned span's `doc` == its citation (flows through the cite gate)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.5 as-of gate GREEN (in-force True/False/abstain · tool withholds · $0)")
        print("  ⓘ Live bar (G8.1a, Jeel's go): a dated query against the ingested")
        print("    Constitution returns the in-force Article and refuses a superseded one.")
        return 0
    print("  ✗ G8.5 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
