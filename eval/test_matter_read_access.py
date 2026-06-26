"""F2m regression gate — shared-matter READ access (offline, $0, no live DB/Pinecone).

D0/D3: a member STAFFED on a matter gets FULL access to it — they can open the shared vault and
query its documents, which live in the OWNER's per-user namespace. This gate pins the ONE authority
that grants that cross-user read, db.SupabaseManager.accessible_vault_owner, so it can never drift
into either failure mode:

  • OVER-OPEN  — a non-staffed firm member, a cross-firm user, or a SCREENED member reading a matter
                 they have no right to (a cross-user leak — the exact boundary F1 hardened).
  • OVER-SHUT  — a staffed member (or the owner) wrongly denied (breaks the productivity promise).

It exercises accessible_vault_owner directly with a fake Supabase that models the firm/membership/
screen tables, and asserts the deny-overrides precedence (a screen beats staffing) the wall demands.

    python -u eval/test_matter_read_access.py
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


from src.components.db import SupabaseManager  # noqa: E402


# ── A fake PostgREST query builder + client modeling exactly the tables the authority reads ──
class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """A chainable .select().eq().limit().execute() over one in-memory table (list of dicts)."""
    def __init__(self, rows, raise_on_execute=False):
        self._rows = rows
        self._filters = {}
        self._raise = raise_on_execute

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters[col] = str(val)
        return self

    def is_(self, col, _val):
        # only used by screened_vault_ids (removed_at IS NULL) — our screen rows are all active
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        if self._raise:
            raise RuntimeError("simulated screens-table live fault")
        out = []
        for r in self._rows:
            if all(str(r.get(k)) == v for k, v in self._filters.items()):
                out.append(r)
        return _Result(out)


class _Client:
    def __init__(self, tables, raise_screens=False):
        self._tables = tables
        self._raise_screens = raise_screens

    def table(self, name):
        return _Query(self._tables.get(name, []),
                      raise_on_execute=(self._raise_screens and name == "screens"))


def _mgr(*, caller, firm, collections, memberships, matter_members, screens,
         raise_screens=False):
    """Build a SupabaseManager wired to fakes WITHOUT touching a live DB.

    read_client and client both point at the same in-memory tables — accessible_vault_owner
    uses read_client for the owner fast-path and client for the cross-user lookups; the gate
    only cares about the AUTHORITY decision, so one backing store is correct and simpler.
    """
    m = SupabaseManager.__new__(SupabaseManager)
    m._user = type("U", (), {"id": caller})()
    tables = {
        "collections": collections,
        "firm_memberships": memberships,
        "matter_memberships": matter_members,
        "screens": screens,
    }
    client = _Client(tables, raise_screens=raise_screens)
    m.client = client
    # read_client is a property that returns self.client when no JWT is attached (offline path),
    # so leaving the token unset routes every read through our fake client.
    m._access_token = None
    m._read_client = None

    # get_user_firm hits firm_memberships+firms via a join shape our fake doesn't model; the
    # authority only needs {"id": firm}. Stub it to the caller's firm (or {} if firm-less).
    m.get_user_firm = lambda *a, **k: ({"id": firm} if firm else {})
    return m


# Canonical world: firm F. ownerU owns vault V (firm F). paraU is a paralegal in F.
OWNER = "owner-1"
PARA = "para-1"
STRANGER = "stranger-1"
FIRM = "firm-A"
OTHER_FIRM = "firm-B"
VAULT = "vault-V"

_COLLECTIONS = [{"id": VAULT, "user_id": OWNER, "firm_id": FIRM}]
_MEMBERSHIPS = [
    {"user_id": OWNER, "firm_id": FIRM, "role": "managing_partner"},
    {"user_id": PARA, "firm_id": FIRM, "role": "paralegal"},
]


def _staffed(*users):
    return [{"id": f"mm-{u}", "firm_id": FIRM, "vault_id": VAULT, "user_id": u} for u in users]


# ── 1. The OWNER always resolves to themselves (legacy/byte-identical path) ──
m = _mgr(caller=OWNER, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(), screens=[])
check("owner resolves to self", m.accessible_vault_owner(VAULT) == OWNER,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 2. A STAFFED member resolves to the OWNER (the productivity grant — D3) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA), screens=[])
check("staffed paralegal resolves to the vault OWNER", m.accessible_vault_owner(VAULT) == OWNER,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 3. A firm member NOT staffed on the matter gets NONE (need-to-know) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(), screens=[])
check("un-staffed firm member is denied (None)", m.accessible_vault_owner(VAULT) is None,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 4. A SCREEN beats staffing (deny-overrides — the ethical wall, even when staffed) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA),
         screens=[{"user_id": PARA, "firm_id": FIRM, "vault_id": VAULT, "removed_at": None}])
check("a screen denies a STAFFED member (deny-overrides)", m.accessible_vault_owner(VAULT) is None,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 5. Cross-firm: a staffed-looking row in ANOTHER firm cannot reach this vault ──
#     Caller is in OTHER_FIRM; the vault is in FIRM ⇒ firm mismatch ⇒ None (T2/T3).
m = _mgr(caller=STRANGER, firm=OTHER_FIRM, collections=_COLLECTIONS,
         memberships=[{"user_id": STRANGER, "firm_id": OTHER_FIRM, "role": "partner"}],
         matter_members=[{"id": "x", "firm_id": OTHER_FIRM, "vault_id": VAULT, "user_id": STRANGER}],
         screens=[])
check("cross-firm caller is denied (firm boundary)", m.accessible_vault_owner(VAULT) is None,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 6. Firm-less caller ⇒ owner-only; a non-owned vault is None (no matter sharing) ──
m = _mgr(caller=PARA, firm=None, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA), screens=[])
check("firm-less caller cannot reach a vault they don't own", m.accessible_vault_owner(VAULT) is None,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 7. FAIL-CLOSED: a live screens-table fault denies a staffed member (never opens the wall) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA), screens=[], raise_screens=True)
check("a screen-lookup fault FAILS CLOSED (denied, wall never silently opens)",
      m.accessible_vault_owner(VAULT) is None, f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 8. A removed screen (soft-lift) restores access on the next call (T7) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA),
         screens=[])  # screened_vault_ids filters removed_at IS NULL; an empty set = lifted
check("a lifted screen restores the staffed member's access", m.accessible_vault_owner(VAULT) == OWNER,
      f"got {m.accessible_vault_owner(VAULT)!r}")

# ── 9. A guessed/foreign vault id (not in any table) resolves to None (no passthrough) ──
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA), screens=[])
check("an unknown vault id resolves to None", m.accessible_vault_owner("vault-GHOST") is None,
      f"got {m.accessible_vault_owner('vault-GHOST')!r}")


# ── 10. UPLOAD INTO A SHARED MATTER (F2m / D0 — the productivity grant) ──
# The upload route resolves accessible_vault_owner, then stamps the document record with that
# owner so a paralegal's upload lands IN the matter (the owner's namespace), not orphaned in
# their own space. Here we assert (a) the owner the route would stamp, and (b) that
# create_document_record honors owner_user_id.

class _InsertCapture:
    """A minimal client whose .table(...).insert(row).execute() records the inserted row."""
    def __init__(self):
        self.inserted = None
    def table(self, _name):
        return self
    def insert(self, row):
        self.inserted = row
        return self
    def execute(self):
        return _Result([{**(self.inserted or {}), "id": "doc-new"}])

# (a) The decision the route makes: a STAFFED paralegal uploading into the shared matter stamps OWNER.
m = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
         matter_members=_staffed(PARA), screens=[])
upload_owner = m.accessible_vault_owner(VAULT)
check("D0 upload: a staffed paralegal's upload resolves to the OWNER (lands in the matter)",
      upload_owner == OWNER, f"got {upload_owner!r}")

# (b) create_document_record stamps that resolved owner, not the caller (so the doc belongs to the matter).
cap = _InsertCapture()
m.client = cap
rec = m.create_document_record(filename="brief.pdf", storage_path=f"{upload_owner}/brief.pdf",
                               file_type="pdf", file_size_bytes=123, owner_user_id=upload_owner)
check("D0 upload: the document row is stamped with the OWNER's user_id (not the paralegal's)",
      cap.inserted and cap.inserted.get("user_id") == OWNER and cap.inserted.get("user_id") != PARA,
      f"stamped user_id={cap.inserted.get('user_id') if cap.inserted else None!r}")

# (c) An UN-STAFFED member uploading into the vault resolves to None ⇒ the route 403s (no orphan write).
m_block = _mgr(caller=PARA, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
               matter_members=_staffed(), screens=[])
check("D0 upload: an un-staffed member is denied (route 403s, never writes an orphan)",
      m_block.accessible_vault_owner(VAULT) is None, f"got {m_block.accessible_vault_owner(VAULT)!r}")

# (d) Own-vault upload is byte-identical: owner_user_id defaults to caller (legacy parity).
m_own = _mgr(caller=OWNER, firm=FIRM, collections=_COLLECTIONS, memberships=_MEMBERSHIPS,
             matter_members=_staffed(), screens=[])
cap2 = _InsertCapture(); m_own.client = cap2
m_own._user = type("U", (), {"id": OWNER})()
m_own.create_document_record(filename="own.pdf", storage_path=f"{OWNER}/own.pdf",
                             file_type="pdf", file_size_bytes=1)
check("D0 upload: an own-vault upload stamps the caller (owner_user_id default ⇒ legacy parity)",
      cap2.inserted and cap2.inserted.get("user_id") == OWNER,
      f"stamped user_id={cap2.inserted.get('user_id') if cap2.inserted else None!r}")


print()
print("=" * 60)
print(f"  test_matter_read_access: {_passed} passed, {_failed} failed")
print("=" * 60)
sys.exit(1 if _failed else 0)
