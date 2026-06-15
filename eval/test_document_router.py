"""Router metadata pre-narrow gate — G3 Step D (scale + recall, fully offline).

Step D's job: for a >20-doc vault, the DocumentRouter pre-narrows the candidate set by
the active metadata_filter (doc_type / fiscal_year) BEFORE the vector fan-out, so a
1000-doc vault doesn't embed-search all 1000 (CDB §7.3). This gate proves, with NO real
embeddings / DB / model:

  1. `_doc_matches_filter` is null-safe (unknown value → kept, never guessed away) and
     handles scalar + {"$in": [...]} conditions.
  2. At a SIMULATED 1000 docs, a metadata filter cuts the candidate set to exactly the
     matching subset BEFORE scoring (the summaries fetch sees only the narrowed ids).
  3. recall@k is UNCHANGED vs. the unfiltered-but-correct-scope baseline: the relevant
     fixture docs that should rank top still do (pre-narrow drops noise, not signal).
  4. `route_doc_ids` returns the scoped DOC_IDS (the G3 scope axis), not filenames.
  5. The pre-narrow can never WIDEN scope (it only drops known non-matches from the
     vault-scoped set) and never empties the vault (an all-miss filter keeps the vault).

The DB + embeddings are faked: a stub SupabaseManager returns a synthetic 1000-doc
collection + per-doc topic embeddings, and we inject a deterministic query embedding (no
OpenAI call). Run: python -u eval/test_document_router.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import src.components.document_router as dr
from src.components.document_router import (
    DocumentRouter, _doc_matches_filter, ROUTER_PRENARROW_THRESHOLD,
)


class Checks:
    def __init__(self):
        self.passed = self.failed = 0
    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


# ── fake config (no env / no real Config needed) ────────────────────────────────
class FakeConfig:
    ROUTING_TOP_N = 12
    ROUTING_MMR_LAMBDA = 1.0            # pure relevance → deterministic, testable ordering
    EMBEDDING_MODEL_NAME = "fake-embed"
    OPENAI_API_KEY = "sk-fake"


# ── synthetic 1000-doc collection ───────────────────────────────────────────────
# Embeddings are 4-D so cosine is exact & hand-checkable. The query embedding points at
# axis 0; "relevant" docs lean on axis 0, the rest are noise on other axes.
DIM = 4
N_DOCS = 1000
QUERY_EMB = [1.0, 0.0, 0.0, 0.0]

# Each doc: id, filename, doc_type, fiscal_year, topic_embedding.
# - 20 RELEVANT docs (high axis-0 weight) — the recall target. Among them we vary
#   doc_type / fiscal_year so a filter selects a known subset.
# - 980 NOISE docs (axis-1/2/3) that a correct scope must NOT crowd the relevant out of.
def _make_corpus():
    docs = []
    # 20 relevant docs, all axis-0 dominant (cosine ~1 with the query).
    for i in range(20):
        fy = 2023 if i < 8 else 2022          # 8 are FY2023, 12 FY2022
        dt = "legal_contract" if i % 2 == 0 else "financial_filing"
        # fy=None for two of them → exercise null-safety (kept by any FY filter)
        if i in (18, 19):
            fy = None
        docs.append({
            "id": f"rel-{i:03d}", "filename": f"relevant_{i:03d}.pdf",
            "doc_type": dt, "fiscal_year": fy,
            "emb": [1.0, 0.02 * (i % 3), 0.0, 0.0],
        })
    # 980 noise docs, axis 1/2/3 — never relevant to an axis-0 query.
    for i in range(980):
        axis = 1 + (i % 3)
        emb = [0.0, 0.0, 0.0, 0.0]; emb[axis] = 1.0
        docs.append({
            "id": f"noise-{i:04d}", "filename": f"noise_{i:04d}.pdf",
            "doc_type": "financial_filing", "fiscal_year": 2021,
            "emb": emb,
        })
    return docs

CORPUS = _make_corpus()
EMB_BY_ID = {d["id"]: d["emb"] for d in CORPUS}


# ── stub SupabaseManager (route_ranked does `from src.components.db import …`) ────
# Shared sink: the document_summaries candidate-id set fetched on the LAST route call.
# route_ranked constructs a fresh SupabaseManager() each call, so we record into a
# module-level holder the test reads after the call (not a per-instance attr).
class _Sink:
    last_summary_ids = None

SINK = _Sink()


class _FakeTable:
    """Records the .in_('document_id', ids) filter so we can assert the pre-narrow."""
    def __init__(self):
        self._sel_summaries = False
        self._in_ids = None

    def select(self, cols):
        self._sel_summaries = "topic_embedding" in cols
        return self

    def in_(self, col, ids):
        if col == "document_id":
            self._in_ids = list(ids)
            SINK.last_summary_ids = list(ids)   # ← the candidate set after pre-narrow
        return self

    def eq(self, *a, **k):
        return self

    def execute(self):
        if self._sel_summaries and self._in_ids is not None:
            data = [{"document_id": did, "summary": did,
                     "topic_embedding": EMB_BY_ID[did]}
                    for did in self._in_ids if did in EMB_BY_ID]
            return type("R", (), {"data": data})()
        return type("R", (), {"data": []})()


class _FakeClient:
    def table(self, name):
        return _FakeTable()


class FakeDB:
    """One per route call (route_ranked constructs SupabaseManager() itself)."""
    def __init__(self, *a, **k):
        self.client = _FakeClient()
    def get_collection_documents(self, collection_id):
        return CORPUS  # the full 1000-doc vault membership


def _install_fake_db():
    """Make `from src.components.db import SupabaseManager` resolve to FakeDB."""
    import types
    fake_mod = types.ModuleType("src.components.db")
    fake_mod.SupabaseManager = FakeDB
    sys.modules["src.components.db"] = fake_mod


def main():
    c = Checks()
    _install_fake_db()
    router = DocumentRouter(FakeConfig())

    print("\n── 1. _doc_matches_filter: null-safe + scalar + $in ─────────────")
    c.ok(_doc_matches_filter({"fiscal_year": None}, {"fiscal_year": 2023}),
         "unknown (None) fiscal_year → KEPT (never guessed away)")
    c.ok(_doc_matches_filter({}, {"doc_type": "legal_contract"}),
         "missing key → KEPT (unknown, don't exclude)")
    c.ok(not _doc_matches_filter({"fiscal_year": 2022}, {"fiscal_year": 2023}),
         "known non-match (2022 vs 2023) → DROPPED")
    c.ok(_doc_matches_filter({"fiscal_year": 2023}, {"fiscal_year": {"$in": [2022, 2023]}}),
         "$in match → KEPT")
    c.ok(not _doc_matches_filter({"fiscal_year": 2021}, {"fiscal_year": {"$in": [2022, 2023]}}),
         "$in non-match → DROPPED")
    c.ok(_doc_matches_filter({"doc_type": "legal_contract", "fiscal_year": 2023},
                             {"doc_type": "legal_contract", "fiscal_year": 2023}),
         "multi-key conjunctive match → KEPT")

    print(f"\n── 2. pre-narrow at simulated {N_DOCS} docs (no embed of all 1000) ─")
    c.ok(N_DOCS > ROUTER_PRENARROW_THRESHOLD,
         f"fixture vault ({N_DOCS}) exceeds the pre-narrow threshold ({ROUTER_PRENARROW_THRESHOLD})")
    # Filter: legal_contract — among the 20 relevant, i%2==0 → 10 docs; the 980 noise are
    # financial_filing so they're all dropped by the pre-narrow.
    SINK.last_summary_ids = None
    ranked = router.route_ranked(
        "axis0 query", "coll-1k", "user-1",
        query_embedding=QUERY_EMB, metadata_filter={"doc_type": "legal_contract"},
    )
    candidate_ids = set(SINK.last_summary_ids or [])
    expected_candidates = {d["id"] for d in CORPUS if d["doc_type"] == "legal_contract"}
    c.ok(candidate_ids == expected_candidates,
         f"summaries fetched for EXACTLY the {len(expected_candidates)} legal_contract docs "
         f"(got {len(candidate_ids)}) — the 980 noise never embed-searched")
    c.ok(len(candidate_ids) < N_DOCS,
         f"candidate set cut {N_DOCS} → {len(candidate_ids)} BEFORE the vector fan-out")

    print("\n── 3. recall@k unchanged vs unfiltered-but-correct-scope ────────")
    # Baseline: NO filter, correct (full-vault) scope. The router must surface the relevant
    # axis-0 docs at the top despite 980 noise docs. recall@12 over the 20 relevant docs.
    SINK.last_summary_ids = None
    base = router.route_ranked("axis0 query", "coll-1k", "user-1", query_embedding=QUERY_EMB)
    base_ids = [did for did, _fn in base]
    relevant = {d["id"] for d in CORPUS if d["id"].startswith("rel-")}
    base_hits = sum(1 for did in base_ids if did in relevant)
    c.ok(base_hits == len(base_ids) and base_hits > 0,
         f"baseline (no filter): all {len(base_ids)} routed docs are relevant (recall-clean)")
    # Filtered: legal_contract → the matching relevant docs must still rank top, none lost.
    filt = router.route_ranked(
        "axis0 query", "coll-1k", "user-1",
        query_embedding=QUERY_EMB, metadata_filter={"doc_type": "legal_contract"},
    )
    filt_ids = [did for did, _fn in filt]
    expected_relevant_contracts = {d["id"] for d in CORPUS
                                   if d["id"].startswith("rel-") and d["doc_type"] == "legal_contract"}
    # Every routed doc is a relevant contract (no noise leaked), and we recalled all of
    # the relevant contracts that fit in top_n.
    all_relevant_contracts = all(did in expected_relevant_contracts for did in filt_ids)
    recalled = set(filt_ids) & expected_relevant_contracts
    c.ok(all_relevant_contracts,
         f"filtered route returns ONLY relevant legal_contracts ({len(filt_ids)} docs, no noise)")
    c.ok(recalled == (expected_relevant_contracts if len(expected_relevant_contracts) <= router.config.ROUTING_TOP_N
                      else set(filt_ids)),
         f"recall held: all {len(recalled)} relevant contracts surfaced (vs baseline's same relevant set)")

    print("\n── 4. fiscal_year filter + null-safety end-to-end ───────────────")
    SINK.last_summary_ids = None
    fy = router.route_ranked(
        "axis0 query", "coll-1k", "user-1",
        query_embedding=QUERY_EMB, metadata_filter={"fiscal_year": 2023},
    )
    fy_ids = {did for did, _fn in fy}
    # FY2023 relevant docs are rel-000..007 (8). rel-018/019 have fy=None → KEPT (null-safe).
    # noise docs are fy=2021 → dropped. So candidates = {fy2023} ∪ {fy=None}.
    cand = set(SINK.last_summary_ids or [])
    expected_fy_cand = {f"rel-{i:03d}" for i in list(range(8)) + [18, 19]}
    c.ok(cand == expected_fy_cand,
         f"FY2023 filter keeps the 8 FY2023 docs + 2 unknown-FY docs (null-safe) = {len(expected_fy_cand)}")
    c.ok(all(did.startswith("rel-") for did in fy_ids),
         "no FY2021 noise doc leaked past the FY2023 filter")

    print("\n── 5. route_doc_ids returns the scope axis; never widens/empties ─")
    ids_only = router.route_doc_ids(
        "axis0 query", "coll-1k", "user-1",
        query_embedding=QUERY_EMB, metadata_filter={"doc_type": "legal_contract"},
    )
    c.ok(ids_only and all(i in {d["id"] for d in CORPUS} for i in ids_only),
         "route_doc_ids returns doc_ids (the G3 scope axis), all in-collection (no widening)")
    # An all-miss filter must keep the vault, not empty it (no silent 0-result scope).
    SINK.last_summary_ids = None
    allmiss = router.route_ranked(
        "axis0 query", "coll-1k", "user-1",
        query_embedding=QUERY_EMB, metadata_filter={"doc_type": "nonexistent_type"},
    )
    cand_allmiss = set(SINK.last_summary_ids or [])
    c.ok(len(cand_allmiss) == N_DOCS,
         "an all-miss filter does NOT empty the vault — falls back to the full candidate set")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G3 Step D router pre-narrow gate GREEN (scale + recall, zero model burn)")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
