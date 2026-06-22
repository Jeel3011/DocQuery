"""H6 regression gate — redline scope assembly + vault-membership (offline, $0).

Covers plans/F1_ISOLATION_AUDIT.md §H6. Before the fix, the redline route:
  (1) called `_resolve_collection_filters(sb, cid, None, retrieval_mgr, embed)` — passing a
      RetrievalManager where `query_embedding` belongs and an embedding where `user_config`
      belongs (the signature is (sb, collection_id, query, query_embedding, user_config, ...)); and
  (2) unpacked the result as a TUPLE `filenames, doc_map = ...` though the function returns a
      LIST → `doc_map` became a filename string → `doc_map.get(body.doc_id)` crashed at line 135.
  (3) never REJECTED a doc_id outside the collection — it silently fell back to all vault docs.

This gate drives the real FastAPI route with TestClient (auth/config/retrieval deps overridden,
the agent loop stubbed so NO model is called — fully offline). It asserts:

  H6-1  a valid in-vault doc_id runs end to end (200 SSE stream) — the pre-fix crash is gone,
        and the RunScope handed to the engine is locked to that ONE doc (doc_ids==[target],
        filenames==[that doc's filename], filename_by_doc == the FULL vault map for the H2 guard).
  H6-2  a doc_id NOT in the collection is REJECTED with 404 (no silent fall-back to all docs).
  H6-3  an empty collection is rejected (400) — never a 0-scope run.
  H6-4  the route no longer imports `_resolve_collection_filters` (the broken call is gone).

Run: python -u eval/test_redline_scope.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ── A 2-doc vault fixture, served by a fake SupabaseManager ────────────────────────────────
_VAULT_A = {"da1": "contract-a.pdf", "da2": "side-letter-a.pdf"}


class _FakeQuery:
    """Records .eq() columns and returns the vault's documents for a documents.select()."""
    def __init__(self, rec):
        self._rec = rec

    def select(self, *a, **k):
        return self

    def in_(self, *a, **k):
        return self

    def eq(self, col, val):
        self._rec.setdefault("eqs", []).append((col, val))
        return self

    def execute(self):
        return type("R", (), {"data": [{"id": i, "filename": fn} for i, fn in _VAULT_A.items()]})()


class _FakeClient:
    def __init__(self, rec):
        self._rec = rec

    def table(self, name):
        return _FakeQuery(self._rec)


class _FakeSB:
    user_id = "user-1"

    def __init__(self):
        self.rec = {}
        self.client = _FakeClient(self.rec)
        # F1 RLS hardening: offline fake mirrors the real fallback — no JWT ⇒ read_client IS
        # the service-role client. (The route reads via sb.read_client.)
        self.read_client = self.client

    def get_collection_document_ids(self, cid):
        # Vault A has docs; any other collection is empty (mirrors a foreign/empty vault).
        return list(_VAULT_A.keys()) if cid == "vault-A" else []


# ── Capture the RunScope the redline engine receives (stub the cell builder) ───────────────
_captured = {}


def _fake_build_redline_cell(clause_topic, standard_position, fallback_position,
                             model_factory, scope, doc_name):
    """Stub — records the scope, returns a benign abstain finding (NO model call)."""
    from src.components.agent_core.redline import RedlineFinding
    _captured["scope"] = scope
    _captured["doc_name"] = doc_name
    return RedlineFinding(
        clause_topic=clause_topic, status="abstain", target_quote=None,
        deviation=None, suggested_edit=None, rationale="stub",
        playbook_standard=standard_position,
    )


def _build_app(monkeypatched_cfg):
    """Mount just the redline router with deps overridden."""
    from fastapi import FastAPI
    from src.api.routes import redline as redline_routes
    from src.api.dependencies import get_current_user, get_user_config, get_retrieval_mgr

    app = FastAPI()
    app.include_router(redline_routes.router)

    sb = _FakeSB()
    app.dependency_overrides[get_current_user] = lambda: sb
    app.dependency_overrides[get_user_config] = lambda: monkeypatched_cfg
    app.dependency_overrides[get_retrieval_mgr] = lambda: object()  # never actually called
    return app, sb


def main():
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP (TestClient unavailable: {exc})")
        return 0

    import src.api.routes.redline as redline_routes

    # A minimal config with the flag ON (else the route 404s before scope assembly).
    cfg = type("Cfg", (), {
        "USE_AGENT_CORE": True,
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_API_KEY": "sk-test",
        "ROUTING_MAX_FANOUT": 8,
        # budget_for("grid") falls through to the standard tier — give it the ceilings it reads.
        "AGENT_MODEL_STANDARD": "stub-model",
        "AGENT_STD_MAX_STEPS": 8,
        "AGENT_STD_WALL_S": 120.0,
        "AGENT_STD_TOKEN_BUDGET": 35_000,
    })()

    # Stub the engine + grid preload so the route runs fully offline (no model, no DB grids).
    redline_routes.build_redline_cell = _fake_build_redline_cell  # patched name lookup in _stream
    # build_redline_cell is imported INSIDE _stream via `from ...redline import build_redline_cell`,
    # so patch the source module too.
    import src.components.agent_core.redline as redline_engine
    redline_engine.build_redline_cell = _fake_build_redline_cell
    # Avoid a real model build + grid DB read.
    import src.components.agent_core.model as model_mod
    model_mod.build_model = lambda *a, **k: (lambda *aa, **kk: None)
    import src.components.brain.table_intent as ti
    ti.load_grids_for_docs = lambda *a, **k: []

    app, sb = _build_app(cfg)
    client = TestClient(app)

    playbook = [{"clause_topic": "Governing Law", "standard_position": "Laws of India, courts of Mumbai."}]

    # ── H6-4: the broken import is gone ───────────────────────────────────────────────────
    src = Path(__file__).resolve().parent.parent / "src" / "api" / "routes" / "redline.py"
    text = src.read_text()
    check("H6-4: redline.py no longer imports _resolve_collection_filters (broken call removed)",
          "_resolve_collection_filters" not in text, "still references the broken resolver")

    # ── H6-1: a valid in-vault doc_id runs end to end and scopes to ONE doc ────────────────
    _captured.clear()
    r = client.post("/redline/stream", json={
        "collection_id": "vault-A", "doc_id": "da1", "playbook_rows": playbook,
    })
    check("H6-1: valid in-vault doc_id streams 200 (pre-fix crash gone)", r.status_code == 200,
          f"status={r.status_code} body={r.text[:200]}")
    body = r.text
    # The stub emitted one finding + a redline_done; the stream must not carry a scope-assembly error.
    check("H6-1: stream produced a finding (engine reached, scope assembled)",
          '"type": "finding"' in body, body[:300])
    check("H6-1: no 'Scope assembly failed' error in the stream", "Scope assembly failed" not in body,
          body[:300])

    sc = _captured.get("scope")
    check("H6-1: RunScope was built and handed to the engine", sc is not None)
    if sc is not None:
        check("H6-1: scope locked to the ONE target doc (doc_ids==['da1'])", sc.doc_ids == ["da1"],
              str(sc.doc_ids))
        check("H6-1: scope filenames == the target doc's filename only",
              sc.filenames == ["contract-a.pdf"], str(sc.filenames))
        check("H6-1: filename_by_doc is the FULL vault map (H2 membership guard has the boundary)",
              sc.filename_by_doc == _VAULT_A, str(sc.filename_by_doc))
        check("H6-1: collection_id carried (vault_active floor fires in search.py)",
              sc.collection_id == "vault-A", str(sc.collection_id))
    check("H6-1: doc_name resolved from the vault map (not the raw id)",
          _captured.get("doc_name") == "contract-a.pdf", str(_captured.get("doc_name")))

    # The documents lookup that built the map MUST filter user_id (RLS is bypassed by service role).
    eq_cols = {c for c, _ in sb.rec.get("eqs", [])}
    check("H6-1: vault doc-map lookup filters user_id at the DB (no cross-user read)",
          "user_id" in eq_cols, f"eqs={sb.rec.get('eqs')}")

    # ── H6-2: a doc_id NOT in the collection is rejected with 404 (no silent fall-back) ────
    r2 = client.post("/redline/stream", json={
        "collection_id": "vault-A", "doc_id": "foreign-doc-from-vault-B", "playbook_rows": playbook,
    })
    check("H6-2: foreign doc_id rejected with 404 (not a silent fall-back to all vault docs)",
          r2.status_code == 404, f"status={r2.status_code} body={r2.text[:200]}")

    # ── H6-3: an empty collection is rejected (never a 0-scope run) ────────────────────────
    r3 = client.post("/redline/stream", json={
        "collection_id": "vault-empty", "doc_id": "da1", "playbook_rows": playbook,
    })
    check("H6-3: empty collection rejected with 400", r3.status_code == 400,
          f"status={r3.status_code} body={r3.text[:200]}")

    print()
    print(f"  {_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
