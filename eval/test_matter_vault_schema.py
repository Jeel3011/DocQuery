"""F1a regression gate — matter/vault schema spine (offline, no API, no Supabase).

The committed gate for F1a (plans/F1_VAULT_PLAN.md §1): promoting a `collection` into a
typed MATTER/VAULT. It asserts the SCHEMA + SERIALIZATION contract end-to-end without a live
DB — deterministic, $0. Run:

    python -u eval/test_matter_vault_schema.py

What it proves:
  1. The matter-kind/status closed sets in schemas.py are IN LOCKSTEP with the CHECK
     constraints in docs/migrations/010_matter_vaults.sql (a drift here = a DB row the API
     would accept but the DB rejects, or vice-versa — a loud, silent-wrong class).
  2. A vault created the OLD way (name only, no matter fields) still validates and serializes
     as status='active' — the legacy create path is byte-identical.
  3. A typed vault round-trips: matter_kind + parties in → out, parties as structured objects.
  4. Invalid matter_kind / status are REJECTED at the Pydantic boundary (before the DB).
  5. db.py only sends matter columns when provided (no hard dependency on migration 010 for
     the legacy path) — verified via a fake client that records the inserted row.

It does NOT touch Pinecone, the kernel, or extraction — F1a is pure data model.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import (  # noqa: E402
    CreateCollectionRequest,
    UpdateCollectionRequest,
    CollectionResponse,
    MatterParty,
    MATTER_KINDS,
    MATTER_STATUSES,
)

MIGRATION = Path(__file__).resolve().parent.parent / "docs/migrations/010_matter_vaults.sql"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. lockstep: schemas constants ↔ migration CHECK constraints ──────────────────────────
def _check_set_in_migration(sql: str, anchor: str, expected: tuple) -> tuple[bool, str]:
    """Extract the IN (...) list following `anchor` and compare to `expected` (as a set)."""
    m = re.search(anchor + r"\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return False, f"could not find `{anchor} IN (...)` in migration"
    found = set(re.findall(r"'([^']+)'", m.group(1)))
    exp = set(expected)
    return found == exp, f"migration={sorted(found)} vs schemas={sorted(exp)}"


sql = MIGRATION.read_text() if MIGRATION.exists() else ""
check("migration 010 exists", bool(sql), str(MIGRATION))
if sql:
    ok, detail = _check_set_in_migration(sql, "matter_kind", MATTER_KINDS)
    check("matter_kind set ↔ migration lockstep", ok, detail)
    ok, detail = _check_set_in_migration(sql, "status", MATTER_STATUSES)
    check("status set ↔ migration lockstep", ok, detail)
    check("migration adds firm_id/matter_kind/status/parties columns",
          all(c in sql for c in ("firm_id", "matter_kind", "status", "parties")))

# ── 2. legacy create path (name only) is byte-identical ──────────────────────────────────
legacy = CreateCollectionRequest(name="My Documents")
check("legacy create validates (no matter fields)", legacy.matter_kind is None)
legacy_resp = CollectionResponse(id="x", name="My Documents")
check("legacy response defaults status='active'", legacy_resp.status == "active")
check("legacy response parties default None/empty", not legacy_resp.parties)

# ── 3. typed vault round-trips ────────────────────────────────────────────────────────────
typed = CreateCollectionRequest(
    name="Acme v. Beta",
    matter_kind="litigation",
    parties=[MatterParty(name="Acme Corp", role="client"),
             MatterParty(name="Beta LLC", role="opposing")],
)
check("typed create accepts valid matter_kind", typed.matter_kind == "litigation")
check("typed create parses parties as objects",
      len(typed.parties) == 2 and typed.parties[0].name == "Acme Corp")
resp = CollectionResponse(
    id="abc", name="Acme v. Beta", matter_kind="litigation", status="active",
    parties=[MatterParty(name="Acme Corp", role="client")], firm_id="firm-1",
)
check("typed response serializes matter fields", resp.matter_kind == "litigation" and resp.firm_id == "firm-1")

# ── 4. invalid values rejected at the Pydantic boundary ───────────────────────────────────
def _rejects(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True

check("invalid matter_kind rejected on create",
      _rejects(lambda: CreateCollectionRequest(name="x", matter_kind="banking")))
check("invalid status rejected on update",
      _rejects(lambda: UpdateCollectionRequest(status="frozen")))
check("invalid matter_kind rejected on update",
      _rejects(lambda: UpdateCollectionRequest(matter_kind="nonsense")))
check("valid lifecycle status accepted on update",
      UpdateCollectionRequest(status="legal_hold").status == "legal_hold")

# ── 5. db.py sends matter columns only when provided (fake client, no live DB) ────────────
class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, recorder):
        self._rec = recorder
        self._pending = None

    def insert(self, row):
        self._rec["inserted"] = row
        self._pending = [dict(row, id="new-id")]
        return self

    def update(self, row):
        self._rec["updated"] = row
        self._pending = [dict(row, id="upd-id")]
        return self

    def eq(self, *a, **k):
        return self

    def execute(self):
        return _FakeResult(self._pending or [])


class _FakeClient:
    def __init__(self, recorder):
        self._rec = recorder

    def table(self, _name):
        return _FakeTable(self._rec)


# Build a SupabaseClient-shaped object without constructing the real one (avoids network):
from src.components.db import SupabaseManager  # noqa: E402

class _User:
    id = "user-1"


rec = {}
fake = SupabaseManager.__new__(SupabaseManager)
fake.client = _FakeClient(rec)
fake._user = _User()  # user_id is a read-only property derived from self._user.id

# legacy create → no matter columns in the inserted row
fake.create_collection("Plain Vault")
legacy_row = rec.get("inserted", {})
check("db.create legacy row omits matter columns",
      "matter_kind" not in legacy_row and "parties" not in legacy_row and "firm_id" not in legacy_row,
      str(legacy_row))

# typed create → matter columns present
rec.clear()
fake.create_collection("Deal Vault", matter_kind="m&a",
                       parties=[{"name": "Acme"}], firm_id="firm-9")
typed_row = rec.get("inserted", {})
check("db.create typed row includes matter columns",
      typed_row.get("matter_kind") == "m&a" and typed_row.get("firm_id") == "firm-9"
      and typed_row.get("parties") == [{"name": "Acme"}], str(typed_row))

# rename-only update → only `name` in the updated row (lifecycle untouched)
rec.clear()
fake.update_collection("cid", name="Renamed")
upd_row = rec.get("updated", {})
check("db.update rename-only omits lifecycle columns",
      upd_row == {"name": "Renamed"}, str(upd_row))

# lifecycle update → status present
rec.clear()
fake.update_collection("cid", status="closed")
upd_row2 = rec.get("updated", {})
check("db.update lifecycle sends status", upd_row2.get("status") == "closed", str(upd_row2))


print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
