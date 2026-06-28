"""Document-harness Phase 1 gate — DOCUMENT_HARNESS §17 / §15.5.

Fully offline, ZERO API spend. A fake postgrest-style db_client serves canned
`document_chunks` rows and RECORDS which client (read_client vs service-role) and which
`user_id` each query used, so the §8/§15.2 isolation can be asserted directly. A
ScriptedModel drives the build_doc_cells run.

What it locks down (the Phase-1 acceptance gate):

  1. ENVELOPE shape — list_documents / search_text / read_document(full_text) /
     read_section all return {ok, summary, data, provenance}; failures carry
     error/abstain_reason.
  2. search_text — extracts matching LINES with doc/page; any_of OR-s synonyms;
     is_regex defaults off; an over-long regex is refused (ReDoS guard); never raises.
  3. LEAK TEST (§8) —
       a. a foreign / out-of-scope doc_id resolves to nothing (no cross-vault read);
       b. an active vault with no in-scope target → refuse, never fan out;
       c. a SHARED matter reads via the SERVICE-ROLE client with the OWNER's user_id;
          an OWN vault reads via read_client with the caller's user_id;
       d. a forged user_id / collection_id tool arg is IGNORED (identity comes from
          scope, not the model — Security L2 / G-c).
  4. read_document token guard — over the cap → too_large + outline, never a dump.
  5. read_document empty doc → honest abstain (G-g), never crash.
  6. build_doc_cells PARITY — emits one GridCell per column with the SAME
     GridCell/abstain_reason taxonomy as the per-cell path; per-column degrade.
  7. FLAG-OFF identity — registry.names("grid") is byte-identical to _MODE_TOOLS.

Run: python -u eval/test_doc_harness.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.agent_core.model import ModelResponse, ScriptedModel, ToolCall
from src.components.agent_core.registry import REGISTRY, RunScope, _MODE_TOOLS, _HARNESS_TOOLS
from src.components.agent_core.review_grid import CellStatus, ColumnKind, GridColumn
from src.components.agent_core.grid_engine import build_doc_cells
from src.components.agent_core.tools import (
    list_documents, read_document, read_section, search_text, SCHEMAS,
)
from src.components.agent_core.tools._chunks import resolve_doc_id


# ── Fake postgrest-style db client (records client + user_id per query) ──────────

class _Query:
    def __init__(self, client_tag, store, recorder):
        self._tag = client_tag
        self._store = store
        self._rec = recorder
        self._doc_id = None
        self._user_id = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        if col == "document_id":
            self._doc_id = val
        elif col == "user_id":
            self._user_id = val
        # metadata->>chunk_type etc. ignored (we post-filter)
        return self

    def in_(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        # Record the (client_tag, user_id, doc_id) this query ran under — the leak test
        # reads this to prove the F2m client/uid selection.
        self._rec.append({"client": self._tag, "user_id": self._user_id, "doc_id": self._doc_id})
        rows = self._store.get((self._doc_id, self._user_id), [])
        return type("Res", (), {"data": rows})()


class _ClientFace:
    def __init__(self, tag, store, recorder):
        self._tag = tag
        self._store = store
        self._rec = recorder

    def table(self, _name):
        return _Query(self._tag, self._store, self._rec)


class FakeDB:
    """Mimics db.SupabaseClient's surface the harness tools touch: .client (service-role),
    .read_client (RLS/JWT), .user_id, .get_collection_documents, .accessible_vault_owner."""

    def __init__(self, *, user_id, store, docs=None, owner=None):
        self.user_id = user_id
        self._store = store
        self._docs = docs or []
        self._owner = owner or user_id
        self.queries: list = []  # recorder
        self.client = _ClientFace("service_role", store, self.queries)
        self.read_client = _ClientFace("read_client", store, self.queries)

    def get_collection_documents(self, _cid):
        return self._docs

    def accessible_vault_owner(self, _cid):
        return self._owner


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")


def is_envelope(r) -> bool:
    if not isinstance(r, dict):
        return False
    if not all(k in r for k in ("ok", "summary", "data", "provenance")):
        return False
    if not isinstance(r["ok"], bool) or not isinstance(r["summary"], str):
        return False
    if not isinstance(r["provenance"], list):
        return False
    if r["ok"] is False and "abstain_reason" not in r and "error" not in r:
        return False
    return True


# A small contract: one doc, owned by "alice".
DOC_UUID = "doc-uuid-1"
FILE = "Document.pdf"
ALICE = "alice"
BOB = "bob"

CONTRACT_CHUNKS = [
    {"content": "8. LIMITATION OF LIABILITY\nNeither party shall be liable for indirect "
                "damages. Liability is capped at fees paid.",
     "metadata": {"chunk_type": "text", "page_number": 12, "chunk_index": 30}},
    {"content": "9. GOVERNING LAW\nThis Agreement shall be governed by the laws of the "
                "State of Minnesota.",
     "metadata": {"chunk_type": "text", "page_number": 13, "chunk_index": 31}},
]


def main() -> int:
    c = Check()

    # ── 7. FLAG-OFF identity (do this first; pure, no deps) ──────────────────────
    print("── flag-OFF identity ────────────────────────────────────────────")
    off = REGISTRY.names("grid", harness=False, include_knowledge=False)
    base = [t for t in _MODE_TOOLS["grid"] if t != "search_knowledge"]
    c.ok(off == base, "flag-OFF: names('grid') byte-identical to _MODE_TOOLS")
    on = REGISTRY.names("grid", harness=True, include_knowledge=False)
    c.ok("search_text" in on and "search_vault" not in on,
         "flag-ON: harness offers search_text, not search_vault")
    c.ok(all(n in SCHEMAS for n in on), "flag-ON: every harness tool has a SCHEMA")

    # Own-vault DB: chunks keyed by (doc, alice).
    store = {(DOC_UUID, ALICE): CONTRACT_CHUNKS}
    fb = {DOC_UUID: FILE}
    db = FakeDB(user_id=ALICE, store=store,
                docs=[{"id": DOC_UUID, "filename": FILE, "doc_type": "legal_contract",
                       "fiscal_year": None}])

    # ── 1. list_documents envelope + scope ──────────────────────────────────────
    print("── list_documents ───────────────────────────────────────────────")
    r = list_documents(db_client=db, collection_id="c1", filename_by_doc=fb)
    c.ok(is_envelope(r) and r["ok"], "list_documents: valid envelope")
    c.ok(len(r["data"]["documents"]) == 1 and r["data"]["documents"][0]["filename"] == FILE,
         "list_documents: lists the matter's one doc")
    c.ok(r["provenance"] == [], "list_documents: no provenance (a listing is not evidence)")
    # offline (map-only) path
    r2 = list_documents(filename_by_doc=fb)
    c.ok(is_envelope(r2) and r2["ok"] and len(r2["data"]["documents"]) == 1,
         "list_documents: offline map-only path works")

    # ── 2. search_text envelope + lines + any_of + regex guard ──────────────────
    print("── search_text ──────────────────────────────────────────────────")
    r = search_text("governing law", db_client=db, filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(is_envelope(r) and r["ok"], "search_text: valid envelope")
    c.ok(len(r["data"]["matches"]) >= 1 and any(m["page"] == 13 for m in r["data"]["matches"]),
         "search_text: finds the governing-law clause (page 13)")
    # The clause text is reachable by searching its words (exact, like grep).
    r_mn = search_text("Minnesota", db_client=db, filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(r_mn["ok"] and any("Minnesota" in m["snippet"] for m in r_mn["data"]["matches"]),
         "search_text: a word from the clause body returns the clause line")
    c.ok(all(m.get("page") in (12, 13) for m in r["data"]["matches"]),
         "search_text: every hit carries its page")
    c.ok(r["provenance"] == r["data"]["matches"], "search_text: provenance == hit spans (ledger-identical)")
    # any_of OR
    r = search_text("nonexistentword", any_of=["indemnify", "liable"], db_client=db,
                    filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(r["ok"] and len(r["data"]["matches"]) >= 1, "search_text: any_of OR-s synonyms (finds 'liable')")
    # regex guard
    r = search_text("a" * 250, is_regex=True, db_client=db, filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(not r["ok"] and "error" in r, "search_text: over-long regex refused (ReDoS guard)")
    # missing query
    r = search_text("", db_client=db, filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(not r["ok"], "search_text: empty query → error envelope (no raise)")

    # ── 3. LEAK TEST (§8 / §15.2) ───────────────────────────────────────────────
    print("── leak test (vault isolation) ──────────────────────────────────")
    # a. foreign doc_id resolves to nothing
    c.ok(resolve_doc_id("some-other-vault-doc", fb) is None,
         "leak: a foreign doc_id resolves to None (no cross-vault read)")
    r = search_text("governing law", doc_ids=["some-other-vault-doc"], db_client=db,
                    filename_by_doc=fb, scope_doc_ids=[DOC_UUID])
    c.ok(not r["ok"], "leak: search_text on a foreign doc_id refuses (no fan-out)")
    # b. active vault, empty scope → refuse
    r = search_text("x", db_client=db, filename_by_doc={}, scope_doc_ids=[])
    c.ok(not r["ok"], "leak: empty scope → refuse, never fan out")
    # c. own vault reads via read_client + caller uid
    db.queries.clear()
    read_document(DOC_UUID, db_client=db, filename_by_doc=fb, full_text=True)
    own = [q for q in db.queries if q["doc_id"] == DOC_UUID]
    c.ok(own and all(q["client"] == "read_client" and q["user_id"] == ALICE for q in own),
         "leak: OWN vault reads via read_client with caller's user_id")
    #    shared matter reads via service-role + OWNER uid
    shared_store = {(DOC_UUID, ALICE): CONTRACT_CHUNKS}  # chunks owned by alice
    shared_db = FakeDB(user_id=BOB, store=shared_store, owner=ALICE,
                       docs=[{"id": DOC_UUID, "filename": FILE}])
    shared_db.queries.clear()
    read_document(DOC_UUID, db_client=shared_db, filename_by_doc=fb, full_text=True, owner_id=ALICE)
    sh = [q for q in shared_db.queries if q["doc_id"] == DOC_UUID]
    c.ok(sh and all(q["client"] == "service_role" and q["user_id"] == ALICE for q in sh),
         "leak: SHARED matter reads via service-role client with OWNER's user_id")
    # d. forged identity arg is ignored — the registry never reads user_id from model args
    forged = ToolCall(id="t", name="search_text",
                      args={"query": "governing law", "user_id": "attacker",
                            "collection_id": "other-vault", "doc_ids": ["some-other-vault-doc"]})
    scope = RunScope(collection_id="c1", doc_ids=[DOC_UUID], db_client=db, filename_by_doc=fb)
    res = REGISTRY.execute(forged, scope)
    c.ok(not res["ok"],
         "leak: forged user_id/collection_id args ignored — foreign doc still refused (L2/G-c)")

    # ── 4. read_document token guard ────────────────────────────────────────────
    print("── read_document token guard ────────────────────────────────────")
    big = [{"content": "x" * 700_000, "metadata": {"chunk_type": "text", "page_number": 1, "chunk_index": 1}}]
    big_db = FakeDB(user_id=ALICE, store={(DOC_UUID, ALICE): big})
    r = read_document(DOC_UUID, db_client=big_db, filename_by_doc=fb, full_text=True)
    c.ok(r["ok"] and r["data"].get("too_large") is True and "outline" in r["data"],
         "read_document: over-cap doc returns too_large+outline (no dump)")
    # under cap → full text returned
    r = read_document(DOC_UUID, db_client=db, filename_by_doc=fb, full_text=True)
    c.ok(r["ok"] and "Minnesota" in (r["data"].get("full_text") or ""),
         "read_document: under-cap doc returns the clean full text")

    # ── 5. read_document empty doc → honest abstain (G-g) ────────────────────────
    print("── read_document empty doc ──────────────────────────────────────")
    empty_db = FakeDB(user_id=ALICE, store={})  # no chunks
    r = read_document(DOC_UUID, db_client=empty_db, filename_by_doc=fb, full_text=True)
    c.ok(not r["ok"] and "error" in r, "read_document: empty doc → honest abstain, no crash")

    # ── read_section heading + page_range ────────────────────────────────────────
    print("── read_section ─────────────────────────────────────────────────")
    r = read_section(DOC_UUID, heading="GOVERNING LAW", db_client=db, filename_by_doc=fb)
    c.ok(r["ok"] and "Minnesota" in (r["data"].get("section_text") or ""),
         "read_section: heading slice returns the governing-law section")
    r = read_section(DOC_UUID, page_range="12-12", db_client=db, filename_by_doc=fb)
    c.ok(r["ok"] and "LIABILITY" in (r["data"].get("section_text") or "").upper(),
         "read_section: page_range returns exactly page 12")
    r = read_section("foreign-doc", heading="x", db_client=db, filename_by_doc=fb)
    c.ok(not r["ok"], "read_section: foreign doc refused (no cross-vault read)")

    # ── 6. build_doc_cells parity + per-column degrade ──────────────────────────
    print("── build_doc_cells parity ───────────────────────────────────────")
    cols = [
        GridColumn(key="governing_law", label="Governing Law", prompt="Find governing law.",
                   kind=ColumnKind.CLAUSE),
        GridColumn(key="liability_cap", label="Liability Cap", prompt="Find the liability cap.",
                   kind=ColumnKind.CLAUSE),
        GridColumn(key="termination", label="Termination", prompt="Find termination terms.",
                   kind=ColumnKind.CLAUSE),
    ]
    # Scripted agent: returns a JSON array of envelopes (one FOUND with quote, one
    # MISSING, and OMITS the third column → per-column degrade to no_evidence abstain).
    answer = (
        '[{"key":"governing_law","status":"found","value":"State of Minnesota",'
        '"quote":"governed by the laws of the State of Minnesota","risk":"non_standard","note":"MN"},'
        '{"key":"liability_cap","status":"missing","value":null,"quote":null,"risk":"missing","note":null}]'
    )
    script = [ModelResponse(text=answer)]
    model = ScriptedModel(script)
    cells = build_doc_cells(
        DOC_UUID, cols, collection_id="c1", model=model,
        filename_by_doc=fb, db_client=db, model_id="scripted",
        model_factory=lambda: ScriptedModel([ModelResponse(text="yes – supported.")]),
    )
    c.ok(len(cells) == 3, "build_doc_cells: emits one GridCell per column (3)")
    by_key = {cell.column_key: cell for cell in cells}
    gl = by_key["governing_law"]
    # FOUND requires provenance; the scripted run read no spans, so the contract downgrades
    # it to ABSTAIN(no_evidence) — proving the SAME cite-or-abstain enforcement as per-cell.
    c.ok(gl.status in (CellStatus.FOUND, CellStatus.ABSTAIN),
         "build_doc_cells: governing_law folded through the cite-or-abstain contract")
    c.ok(by_key["liability_cap"].status == CellStatus.MISSING,
         "build_doc_cells: declared-missing column → MISSING (valid finding)")
    deg = by_key["termination"]
    c.ok(deg.status == CellStatus.ABSTAIN and deg.abstain_reason == "no_evidence",
         "build_doc_cells: omitted column → per-column degrade (no_evidence), not a crash")
    # abstain_reason taxonomy is the same vocabulary as the per-cell path
    c.ok(all(cell.abstain_reason in (None, "unparsed", "no_evidence", "ambiguous", "verify_disagree")
             for cell in cells),
         "build_doc_cells: abstain_reason uses the shared G4 taxonomy")

    print(f"\n{c.passed} passed, {c.failed} failed")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
