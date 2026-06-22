"""F1 RLS-hardening gate — the data-layer backstop wiring (offline, $0, no live DB).

Companion to the live verification (eval/verify_rls_live.py, run on demand). The live DB
proves the Postgres property: an authenticated JWT'd query is RLS-blocked from another user's
rows EVEN IF the app `.eq(user_id)` filter is dropped. THIS gate proves the WIRING that makes
that property load-bearing on the request path, fully offline:

  RLS-1  attach_access_token(jwt) makes `read_client` an RLS-enforced (anon+JWT) client,
         DISTINCT from the service-role `client`. No token ⇒ read_client IS client (the
         worker/offline fallback — byte-identical, no behaviour change there).
  RLS-2  user-facing READ methods (get_user_documents/get_document/get_collections/
         get_collection_document_ids/get_messages/...) route through `read_client`, so on
         the request path they run RLS-enforced — a forgotten app filter can't leak.
  RLS-3  WRITES + Storage stay on the service-role `client` (they need the RLS-exempt path).
  RLS-4  the H2 chunk reads (read.py / table_intent.py) prefer `read_client` when present,
         and still fall back to `.client` on an offline/test client (no read_client attr).

This asserts the FIX is in place (and would FAIL on the pre-fix code, where every read used
the single service-role client and `read_client` did not exist). Run:

    python -u eval/test_rls_backstop.py
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


# ── A recording fake PostgREST-ish client that tags which CLIENT served a call ──────────────
class _RecQuery:
    def __init__(self, calls, tag, table):
        self._calls, self._tag, self._table = calls, tag, table
        self._op = "select"

    def select(self, *a, **k):
        self._op = "select"
        return self

    def insert(self, *a, **k):
        self._op = "insert"
        return self

    def update(self, *a, **k):
        self._op = "update"
        return self

    def delete(self, *a, **k):
        self._op = "delete"
        return self

    def eq(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def execute(self):
        self._calls.append((self._tag, self._op, self._table))
        return type("R", (), {"data": [], "count": 0})()


class _RecClient:
    """Tagged 'service' or 'rls' so we can see which one a method used."""
    def __init__(self, calls, tag):
        self._calls, self._tag = calls, tag

    def table(self, name):
        return _RecQuery(self._calls, self._tag, name)


def main():
    from src.components.db import SupabaseManager

    # Build a manager but swap its real clients for recording fakes. We DON'T hit the network.
    calls: list = []
    sb = SupabaseManager.__new__(SupabaseManager)
    sb._user = type("U", (), {"id": "user-1", "email": "a@b.c", "user_metadata": {}})()
    sb._access_token = None
    sb._read_client = None
    sb.client = _RecClient(calls, "service")  # service-role client

    # ── RLS-1: no token ⇒ read_client falls back to the service-role client ─────────────────
    check("RLS-1: with no JWT, read_client IS the service-role client (worker/offline fallback)",
          sb.read_client is sb.client)

    # Attach a token ⇒ read_client must become a DISTINCT, RLS-enforced (anon+JWT) client.
    # We monkeypatch the builder so the gate stays offline (no real anon client / network).
    import src.components.db as dbmod
    rls_client = _RecClient(calls, "rls")
    _orig_builder = dbmod.get_rls_read_client
    dbmod.get_rls_read_client = lambda token: rls_client
    try:
        sb.attach_access_token("jwt-abc")
        rc = sb.read_client
        check("RLS-1: with a JWT, read_client is a DISTINCT (RLS-enforced) client, not service-role",
              rc is rls_client and rc is not sb.client)

        # ── RLS-2: user-facing READS route through read_client (the 'rls' tag) ──────────────
        calls.clear()
        sb.get_user_documents()
        sb.get_document("d1")
        sb.get_collections()
        sb.get_collection("c1")
        sb.get_collection_document_ids("c1")
        sb.get_messages("conv1")
        read_tags = {t for (t, op, _tbl) in calls if op == "select"}
        check("RLS-2: every user-facing READ ran on the RLS client (not service-role)",
              read_tags == {"rls"}, f"tags seen: {sorted(read_tags)} calls={calls}")

        # ── RLS-3: WRITES stay on the service-role client (RLS-exempt path) ─────────────────
        calls.clear()
        sb.delete_document_record("d1")
        sb.create_conversation("t")
        sb.delete_conversation("conv1")
        write_tags = {t for (t, op, _tbl) in calls if op in ("insert", "update", "delete")}
        check("RLS-3: writes stay on the service-role client (writes need the RLS-exempt path)",
              write_tags == {"service"}, f"tags seen: {sorted(write_tags)} calls={calls}")
    finally:
        dbmod.get_rls_read_client = _orig_builder

    # ── RLS-4: the H2 chunk reads prefer read_client, and fall back when it's absent ────────
    # The tools read via `getattr(db_client, "read_client", None) or db_client.client`.
    # (a) a manager WITH a read_client → the H2 read should land on 'rls'.
    from src.components.brain.table_intent import load_grids_for_docs
    calls.clear()
    load_grids_for_docs(sb, ["some-doc"])   # sb has the rls read_client attached above
    h2_tags = {t for (t, _op, tbl) in calls if tbl == "document_chunks"}
    check("RLS-4: H2 grid read (table_intent) uses the RLS client when one is attached",
          h2_tags == {"rls"}, f"tags={sorted(h2_tags)} calls={calls}")

    # (b) an OFFLINE/test client with NO read_client attr → falls back to .client (no crash).
    class _OfflineDB:
        user_id = "user-1"

        def __init__(self, c):
            self.client = c
    calls.clear()
    load_grids_for_docs(_OfflineDB(_RecClient(calls, "service")), ["some-doc"])
    off_tags = {t for (t, _op, tbl) in calls if tbl == "document_chunks"}
    check("RLS-4: H2 grid read falls back to .client when no read_client (offline/worker safe)",
          off_tags == {"service"}, f"tags={sorted(off_tags)} calls={calls}")

    print()
    print(f"  {_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
