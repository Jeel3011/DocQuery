"""F2a regression gate — firm population + roles + invites (offline, no API, no Supabase).

The committed gate for F2a (plans/F2_FIRM_CONSOLE_PLAN.md §F2a). It proves the firm-population
DATA MODEL + the invite SECURITY PROPERTIES (T4) without a live DB — deterministic, $0. Run:

    python -u eval/test_firm_roles.py

What it proves (maps to the plan's §3 gate row):
  1. ROLES (schemas.py) ↔ the CHECK constraints in 012_firm_roles.sql are IN LOCKSTEP (a drift
     = a role the API accepts but the DB rejects, or a role with no capability line in F2b).
  2. Signup → firm + Managing-Partner membership (D1): create_firm makes the creator an MP.
  3. Invite round-trips with the INVITED role — the joiner can NEVER self-assign (D1).
  4. T4 invite abuse defeated, each reproduced on the happy path then blocked:
       - EXPIRED token rejected;
       - USED token rejected (replay) — the atomic single-use claim;
       - EMAIL-MISMATCH rejected (email-binding).
  5. The raw token is returned ONCE on create and never stored (only its sha256 hash is).
  6. Legacy parity: a membership defaults to 'managing_partner' (byte-identical power).

It is OFFLINE: a fake Supabase client models the tables + the conditional-UPDATE atomicity
(WHERE accepted_at IS NULL AND expires_at > now()) that makes single-use real. No network,
no Pinecone, no kernel, no extraction (F2a is pure data model + auth wiring).
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import (  # noqa: E402
    ROLES,
    ROLE_RANK,
    InviteRequest,
    AcceptInviteRequest,
)
from src.components.db import SupabaseManager  # noqa: E402

MIGRATION = Path(__file__).resolve().parent.parent / "docs/migrations/012_firm_roles.sql"

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


# ── 1. lockstep: ROLES ↔ migration CHECK constraints ──────────────────────────────────────
def _roles_in_migration(sql: str) -> set:
    # both the membership and invite CHECK list the same 9 roles; grab the first IN (...) list.
    m = re.search(r"role\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE | re.DOTALL)
    return set(re.findall(r"'([^']+)'", m.group(1))) if m else set()


sql = MIGRATION.read_text() if MIGRATION.exists() else ""
check("migration 012 exists", bool(sql), str(MIGRATION))
if sql:
    check("ROLES ↔ migration role-CHECK lockstep",
          _roles_in_migration(sql) == set(ROLES),
          f"migration={sorted(_roles_in_migration(sql))} vs schemas={sorted(ROLES)}")
    check("migration adds role column + firm_invites table",
          "ADD COLUMN IF NOT EXISTS role" in sql and "firm_invites" in sql)
    check("migration default role = managing_partner (legacy parity)",
          "DEFAULT 'managing_partner'" in sql)
    check("invite token stored HASHED, not raw (T4)",
          "token_hash" in sql and re.search(r"\btoken\s+TEXT", sql) is None)
    check("invite is single-use + expiring (accepted_at + expires_at)",
          "accepted_at" in sql and "expires_at" in sql)
    check("backfill block present (firm-less users → solo MP)",
          "NOT EXISTS" in sql and "firm_memberships" in sql and "managing_partner" in sql)

check("ROLES has the 9 expected roles", len(ROLES) == 9 and ROLES[0] == "managing_partner")
check("ROLE_RANK orders MP most-senior (rank 0)",
      ROLE_RANK["managing_partner"] == 0 and ROLE_RANK["guest"] == len(ROLES) - 1)


def _try_reject(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


# invited role is validated at the Pydantic boundary (the joiner can't smuggle an unknown role)
check("InviteRequest rejects an unknown role",
      _try_reject(lambda: InviteRequest(email="x@y.com", role="overlord")))
check("InviteRequest accepts a valid role",
      InviteRequest(email="x@y.com", role="paralegal").role == "paralegal")
check("AcceptInviteRequest requires a non-empty token",
      _try_reject(lambda: AcceptInviteRequest(token="")))


# ── offline fake Supabase modelling firms / firm_memberships / firm_invites ─────────────────
def _now():
    return datetime.now(timezone.utc)


def _parse(ts):
    if ts is None:
        return None
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    """A tiny chainable query over an in-memory list of dict rows, supporting the subset of
    PostgREST ops db.py uses: select/insert/upsert/update/eq/is_/gt/limit/order/execute."""

    def __init__(self, store, name):
        self._store = store
        self._name = name
        self._rows = store[name]
        self._filters = []      # list of (kind, *args)
        self._op = None         # 'select' | 'insert' | 'upsert' | 'update'
        self._payload = None
        self._on_conflict = None
        self._count = None
        self._limit = None

    # --- ops ---
    def select(self, *a, count=None, **k):
        self._op = "select"
        self._count = count
        # mirror PostgREST column projection: select("a, b, c") returns only those columns.
        # "*" (or an embedded-resource select like "firm_id, firms!inner(...)") = all columns.
        cols = a[0] if a else "*"
        if isinstance(cols, str) and cols.strip() != "*" and "(" not in cols:
            self._projection = [c.strip() for c in cols.split(",") if c.strip()]
        else:
            self._projection = None
        return self

    def insert(self, row):
        self._op = "insert"
        self._payload = dict(row)
        return self

    def upsert(self, row, on_conflict=None):
        self._op = "upsert"
        self._payload = dict(row)
        self._on_conflict = on_conflict
        return self

    def update(self, row):
        self._op = "update"
        self._payload = dict(row)
        return self

    # --- filters ---
    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def gt(self, col, val):
        self._filters.append(("gt", col, val))
        return self

    def limit(self, n):
        self._limit = n
        return self

    def order(self, *a, **k):
        return self

    # --- matching ---
    def _match(self, row):
        for f in self._filters:
            kind = f[0]
            if kind == "eq":
                if str(row.get(f[1])) != str(f[2]):
                    return False
            elif kind == "is":
                # is_(col, "null") → col must be None
                want_null = str(f[2]).lower() == "null"
                if want_null and row.get(f[1]) is not None:
                    return False
                if not want_null and row.get(f[1]) is None:
                    return False
            elif kind == "gt":
                rv = _parse(row.get(f[1]))
                cv = _parse(f[2])
                if rv is None or cv is None or not (rv > cv):
                    return False
        return True

    def execute(self):
        if self._op == "insert":
            row = dict(self._payload)
            row.setdefault("id", f"id-{len(self._rows)+1}")
            row.setdefault("created_at", _now().isoformat())
            self._rows.append(row)
            return _Result([dict(row)])
        if self._op == "upsert":
            row = dict(self._payload)
            keys = (self._on_conflict or "").split(",") if self._on_conflict else []
            for existing in self._rows:
                if keys and all(str(existing.get(k.strip())) == str(row.get(k.strip())) for k in keys):
                    existing.update(row)
                    return _Result([dict(existing)])
            row.setdefault("created_at", _now().isoformat())
            self._rows.append(row)
            return _Result([dict(row)])
        if self._op == "update":
            updated = []
            for r in self._rows:
                if self._match(r):
                    r.update(self._payload)
                    updated.append(dict(r))
            return _Result(updated)
        # select
        matched = [dict(r) for r in self._rows if self._match(r)]
        proj = getattr(self, "_projection", None)
        if proj is not None:
            matched = [{c: r.get(c) for c in proj if c in r} for r in matched]
        cnt = len(matched) if self._count else None
        if self._limit is not None:
            matched = matched[: self._limit]
        return _Result(matched, count=cnt)


class _FakeClient:
    def __init__(self):
        self.store = {"firms": [], "firm_memberships": [], "firm_invites": []}

    def table(self, name):
        self.store.setdefault(name, [])
        return _Query(self.store, name)


def _mgr(user_id, email="owner@example.com"):
    sb = SupabaseManager.__new__(SupabaseManager)
    sb.client = _FakeClient()
    sb._access_token = None
    sb._read_client = None

    class _U:
        pass
    u = _U()
    u.id = user_id
    u.email = email
    sb._user = u
    return sb


# ── 2. signup → firm + MP membership (D1) ────────────────────────────────────────────────────
sb = _mgr("user-A", "alice@firm.com")
firm = sb.create_firm("Alice's Firm")
check("create_firm returns a firm with the MP role", firm.get("role") == "managing_partner")
members = sb.list_members(firm["id"])
check("creator is the firm's sole MP member",
      len(members) == 1 and members[0]["user_id"] == "user-A"
      and members[0]["role"] == "managing_partner")
check("get_user_firm returns firm + role", sb.get_user_firm().get("role") == "managing_partner")
check("count_firm_role MP == 1", sb.count_firm_role(firm["id"], "managing_partner") == 1)

# ── 3 + 5. invite round-trips with the INVITED role; raw token returned once, stored hashed ──
invite = sb.create_invite(firm["id"], "bob@firm.com", "paralegal", invited_by="user-A")
raw_token = invite.get("token")
check("create_invite returns a one-time raw token", bool(raw_token))
check("create_invite does NOT leak the hash", "token_hash" not in invite)
stored = sb.client.store["firm_invites"][0]
check("invite is stored HASHED, never raw (T4)",
      stored.get("token_hash") == SupabaseManager._hash_invite_token(raw_token)
      and "token" not in stored)
check("invite carries the invited role, not a self-assigned one", stored.get("role") == "paralegal")

# Bob accepts with his matching verified email → membership created with the INVITED role.
res = sb.accept_invite(raw_token, accepting_user_id="user-B", accepting_email="bob@firm.com")
check("accept_invite creates membership with the INVITED role (D1, no self-assign)",
      res.get("role") == "paralegal" and res.get("firm_id") == firm["id"])
bob_firm = sb.get_user_firm(user_id="user-B")
check("joiner is now a paralegal of the firm (not MP)", bob_firm.get("role") == "paralegal")

# ── 4a. USED token rejected (replay) — the atomic single-use claim ──────────────────────────
check("replay of a used token is rejected (T4)",
      _try_reject(lambda: sb.accept_invite(raw_token, "user-B", "bob@firm.com")))

# ── 4b. EMAIL-MISMATCH rejected (email-binding) ─────────────────────────────────────────────
inv2 = sb.create_invite(firm["id"], "carol@firm.com", "associate", invited_by="user-A")
check("email-mismatch on accept is rejected (T4)",
      _try_reject(lambda: sb.accept_invite(inv2["token"], "user-X", "attacker@evil.com")))
# the rejected invite is STILL pending (not consumed by the failed attempt)
check("a rejected accept does NOT consume the invite",
      sb.get_invite_by_token(inv2["token"]).get("email") == "carol@firm.com")
# correct email still works
ok2 = sb.accept_invite(inv2["token"], "user-C", "carol@firm.com")
check("correct email accepts the still-pending invite", ok2.get("role") == "associate")

# ── 4c. EXPIRED token rejected ──────────────────────────────────────────────────────────────
inv3 = sb.create_invite(firm["id"], "dan@firm.com", "assistant", invited_by="user-A")
# force-expire it in the store
sb.client.store["firm_invites"][-1]["expires_at"] = (_now() - timedelta(hours=1)).isoformat()
check("expired token is rejected (T4)",
      _try_reject(lambda: sb.accept_invite(inv3["token"], "user-D", "dan@firm.com")))
check("get_invite_by_token returns {} for an expired token",
      sb.get_invite_by_token(inv3["token"]) == {})

# ── 6. legacy parity — a bare membership defaults to MP at the DB (CHECK + DEFAULT in 012) ──
check("default role in migration is managing_partner (legacy byte-identical)",
      "DEFAULT 'managing_partner'" in sql)

# list_invites returns the still-pending invite(s) and NEVER the token/hash.
pending = sb.list_invites(firm["id"])
check("list_invites returns pending invites (non-vacuous)", len(pending) >= 1)
check("list_invites omits token + token_hash",
      len(pending) >= 1 and all("token" not in i and "token_hash" not in i for i in pending))


print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
