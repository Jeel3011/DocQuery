"""F1b regression gate — the AUDIT holes H1/H2/H3 (offline, $0, no live DB/Pinecone).

Companion to eval/test_vault_isolation.py (which covers `search_vault`). This gate covers the
OTHER retrieval/document-read paths the F1 isolation audit (plans/F1_ISOLATION_AUDIT.md) found
leaking — the Brain/chat path and the read/grid path. Each test asserts the FIX is in place
(and would have FAILED on the pre-fix code). Run:

    python -u eval/test_isolation_holes.py

Covers:
  H1 — chat `_resolve_collection_filters` NEVER returns "search everything"; a missing
       collection_id is a hard 400 (no unscoped fan-out over the user namespace).
  H3 — the empty-scope sentinel: an explicitly-empty vault (doc_ids=[] / filename_filters=[])
       yields a MATCH-NOTHING filter, never None (which would scan the whole namespace).
  H2 — read/grid chunk reads filter user_id at the DB (RLS is bypassed by the service-role
       client, so this is the ONLY cross-user isolation), and read_document refuses a doc_id
       that is not a member of the active vault (no cross-vault read).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── H1: chat._resolve_collection_filters refuses an unscoped (null-vault) query ───────────
from fastapi import HTTPException  # noqa: E402
from src.api.routes.chat import _resolve_collection_filters  # noqa: E402


class _FakeSB:
    user_id = "user-1"

    def get_collection_document_ids(self, cid):
        return ["d1", "d2"] if cid == "vault-A" else []

    def is_vault_screened(self, cid, user_id=None, firm_id=None):
        return False   # no walls in the isolation-hole scope checks

    def accessible_vault_owner(self, cid):
        # F2m: own-vault → the caller. (These H1 scope checks use the caller's own vault.)
        return self.user_id

    class _Client:
        def table(self, *_a, **_k):
            class _Q:
                def select(self, *a, **k): return self
                def in_(self, *a, **k): return self
                def eq(self, *a, **k): return self
                def execute(self):
                    return type("R", (), {"data": [{"filename": "a1.pdf"}, {"filename": "a2.pdf"}]})()
            return _Q()
    client = _Client()
    # F1 RLS hardening: offline fakes carry the same fallback the real manager has —
    # read_client IS the service-role client when no JWT is attached (worker/test path).
    read_client = client


sb = _FakeSB()

# null collection_id ⇒ HARD 400, never None (the H1 leak was None ⇒ unscoped fan-out).
raised = False
try:
    _resolve_collection_filters(sb, None, query="revenue")
except HTTPException as e:
    raised = e.status_code == 400
check("H1: null collection_id raises 400 (no unscoped 'search everything')", raised)

# a real vault ⇒ scoped filename list (never None).
res = _resolve_collection_filters(sb, "vault-A", query="revenue",
                                  user_config=type("C", (), {"ROUTING_MAX_FANOUT": 8})())
check("H1: a real vault resolves to a scoped filename list", isinstance(res, list) and res,
      str(res))

# an empty vault ⇒ [] (handled by callers as 'no docs'), NEVER None.
res_empty = _resolve_collection_filters(sb, "vault-empty", query="revenue",
                                        user_config=type("C", (), {"ROUTING_MAX_FANOUT": 8})())
check("H1: empty vault resolves to [] (not None ⇒ never unscoped)", res_empty == [])


# ── H3: the empty-scope sentinel in _build_filter / retrieve_table_chunks ─────────────────
from src.components.retrieval import RetrievalManager  # noqa: E402

# explicitly-empty doc_ids ⇒ match-nothing filter, NOT None (the empty-vault leak).
f = RetrievalManager._build_filter(doc_ids=[])
check("H3: _build_filter(doc_ids=[]) ⇒ match-nothing (doc_id $in [])",
      f == {"doc_id": {"$in": []}}, str(f))
f2 = RetrievalManager._build_filter(filename_filters=[])
check("H3: _build_filter(filename_filters=[]) ⇒ match-nothing", f2 == {"doc_id": {"$in": []}}, str(f2))
# None (no axis requested) is still allowed to be 'no filter' (legacy non-vault dev path).
f3 = RetrievalManager._build_filter()
check("H3: _build_filter() with no axis ⇒ None (no scope requested)", f3 is None, str(f3))
# a real doc_ids list still scopes normally (no regression).
f4 = RetrievalManager._build_filter(doc_ids=["d1", "d2"])
check("H3: _build_filter(doc_ids=[d1,d2]) ⇒ scoped $in (no regression)",
      f4 == {"doc_id": {"$in": ["d1", "d2"]}}, str(f4))


# ── H2: read/grid chunk reads filter user_id + reject out-of-vault doc_id ──────────────────
# load_grids_for_docs must add .eq("user_id", uid). We assert via a recording fake client.
from src.components.brain.table_intent import load_grids_for_docs  # noqa: E402


class _RecQuery:
    def __init__(self, rec):
        self._rec = rec

    def select(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def eq(self, col, val):
        self._rec.setdefault("eqs", []).append((col, val))
        return self

    def execute(self):
        return type("R", (), {"data": []})()


class _RecClient:
    def __init__(self, rec):
        self._rec = rec

    def table(self, *_a, **_k):
        return _RecQuery(self._rec)


class _RecDB:
    user_id = "user-1"

    def __init__(self, rec):
        self.client = _RecClient(rec)


rec = {}
load_grids_for_docs(_RecDB(rec), ["some-doc-id"])
eq_cols = {c for c, _ in rec.get("eqs", [])}
check("H2: load_grids_for_docs filters by user_id at the DB (RLS is bypassed)",
      "user_id" in eq_cols, f"eqs={rec.get('eqs')}")
check("H2: load_grids_for_docs also filters document_id", "document_id" in eq_cols)

# read_document: a doc_id NOT in the active vault's filename_by_doc must not load grids.
from src.components.agent_core.tools.read import read_document  # noqa: E402


class _Grid:
    def __init__(self, doc):
        self.doc, self.page, self.table_id = doc, 1, "t0"


# Vault A's docs (filename_by_doc IS the vault membership set). Ask to read a FOREIGN id.
res = read_document(
    "foreign-doc-from-vault-B",
    grids=[_Grid("a1.pdf")],            # preloaded vault-A grids
    table_grids=True,
    db_client=_RecDB({}),
    filename_by_doc={"da1": "a1.pdf", "da2": "a2.pdf"},   # vault A only
    scope_grids=[],
)
# It must NOT return vault-A's grids for a foreign doc, and must not crash — either an error
# envelope or an empty/in-scope result, never the foreign doc's content.
ok = (res.get("ok") is False) or not (res.get("data") or {}).get("grids")
check("H2: read_document refuses a doc_id outside the active vault (no cross-vault read)", ok,
      str(res.get("summary")))


print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
