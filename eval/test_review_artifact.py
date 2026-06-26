"""F2j.1 regression gate — REVIEWABLE ARTIFACTS: a reviewer can SEE & verify the submitted work
(offline, $0, no live DB).

A review you can't read is theatre. F2j.1 lets a reviewer open the actual work behind a review
request (a chat answer in the SUBMITTER's per-user conversation). The cross-user read is granted ONLY
through the review RELATIONSHIP (least privilege) — this gate pins db.review_artifact_authority +
get_review_artifact + get_review_thread so they can never drift into either failure mode:

  • OVER-OPEN — a firm member who is NOT the owner/in-the-chain, a cross-firm user, or a SCREENED
                reviewer reading work they have no right to (a cross-user leak).
  • OVER-SHUT — the legitimate reviewer (current_owner or a chain member) wrongly denied.

It exercises the REAL db methods + the REAL route handlers with a fake PostgREST modeling the
review_requests / messages / conversations / firm_memberships / screens tables, and asserts the
deny-overrides precedence (a screen beats the review grant) the wall demands.

    ./venv/bin/python -u eval/test_review_artifact.py

What it proves (the plan's §4 gate):
  A. THE AUTHORITY GATE — current_owner CAN read; a chain member CAN read; a non-owner/not-in-chain
     firm member CANNOT (None / 404); a cross-firm caller CANNOT (404, no existence leak); a SCREENED
     caller CANNOT (wall beats the relationship; fails CLOSED on a screen-lookup fault).
  B. LEAST PRIVILEGE / SCOPE — the resolved read is scoped to the SUBMITTER + the ONE conversation;
     a guessed request_id for someone else's review touches nothing.
  C. CONTENT — the preview carries title + question (the user msg before the answer) + answer; the
     thread returns the whole conversation oldest-first.
  D. GRACEFUL DEGRADE — a deleted message ⇒ available:False (not a 500); a missing table ⇒ None;
     the route 404s on deny rather than 500.
  E. AUDIT (T10) — a successful artifact/thread view writes the audit row.

OFFLINE: the db methods are driven against an in-memory fake (one backing store for read_client +
client, like test_matter_read_access). See [[run-only-relevant-gates]] — F2j.1 touches db/matters
routes/FE, NOT extraction or the kernel.
"""
from __future__ import annotations

import asyncio
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


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


from src.components.db import SupabaseManager       # noqa: E402
from src.api.routes import matters as mroutes        # noqa: E402
from fastapi import HTTPException                     # noqa: E402


# ── A fake PostgREST query builder covering exactly the ops the resolvers use ──
class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    """Chainable .select().eq().is_().lte().order().limit().in_().execute() over an in-memory table."""
    def __init__(self, rows, raise_missing=False, raise_live=False):
        self._rows = rows
        self._eq = {}
        self._lte = {}
        self._in = {}
        self._is_null = set()
        self._order = None
        self._desc = False
        self._raise_missing = raise_missing
        self._raise_live = raise_live

    def select(self, *_a, **_k): return self

    def eq(self, col, val):
        self._eq[col] = str(val)
        return self

    def is_(self, col, _val):
        self._is_null.add(col)
        return self

    def lte(self, col, val):
        self._lte[col] = str(val)
        return self

    def in_(self, col, vals):
        self._in[col] = {str(v) for v in vals}
        return self

    def order(self, col, desc=False):
        self._order = col
        self._desc = desc
        return self

    def limit(self, *_a, **_k): return self

    def execute(self):
        if self._raise_missing:
            raise RuntimeError('relation "messages" does not exist (PGRST205)')
        if self._raise_live:
            # A LIVE query fault (NOT a missing-table signature) — is_vault_screened must re-raise
            # this so the authority fails CLOSED, never silently degrading the wall to "not screened".
            raise RuntimeError("connection terminated mid-query (live fault)")
        out = []
        for r in self._rows:
            if not all(str(r.get(k)) == v for k, v in self._eq.items()):
                continue
            if not all(str(r.get(k)) <= v for k, v in self._lte.items()):
                continue
            if not all(str(r.get(k)) in vs for k, vs in self._in.items()):
                continue
            if not all(r.get(k) is None for k in self._is_null):
                continue
            out.append(r)
        if self._order:
            out = sorted(out, key=lambda r: r.get(self._order) or "", reverse=self._desc)
        return _Result(out)


class _Client:
    def __init__(self, tables, missing=(), live_fault=()):
        self._tables = tables
        self._missing = set(missing)
        self._live_fault = set(live_fault)

    def table(self, name):
        return _Query(self._tables.get(name, []),
                      raise_missing=(name in self._missing),
                      raise_live=(name in self._live_fault))


def _mgr(*, caller, firm, tables, missing=(), live_fault=()):
    m = SupabaseManager.__new__(SupabaseManager)
    m._user = type("U", (), {"id": caller})()
    client = _Client(tables, missing=missing, live_fault=live_fault)
    m.client = client
    m._access_token = None
    m._read_client = None
    m.get_user_firm = lambda *a, **k: ({"id": firm} if firm else {})
    return m


# ── Canonical world ──
FIRM = "firm-A"
OTHER_FIRM = "firm-B"
OWNER = "user-mp"          # MP, vault owner, the reviewer (current_owner)
SUBMITTER = "user-para"    # paralegal, submitted the work
CHAIN_MEMBER = "user-sa"   # senior associate, appears in the chain (a legit reviewer too)
STRANGER = "user-assoc"    # a firm member NOT on this review
OUTSIDER = "user-out"      # a member of OTHER_FIRM
VAULT = "vault-1"
WALLED_VAULT = "vault-walled"
CONV = "conv-1"
ANSWER_MSG = "msg-answer"
Q_MSG = "msg-question"
REQ = "req-1"
REQ_WALLED = "req-walled"

_MEMBERSHIPS = [
    {"user_id": OWNER, "firm_id": FIRM, "role": "managing_partner"},
    {"user_id": SUBMITTER, "firm_id": FIRM, "role": "paralegal"},
    {"user_id": CHAIN_MEMBER, "firm_id": FIRM, "role": "senior_associate"},
    {"user_id": STRANGER, "firm_id": FIRM, "role": "associate"},
    {"user_id": OUTSIDER, "firm_id": OTHER_FIRM, "role": "associate"},
]
_CONVERSATIONS = [{"id": CONV, "user_id": SUBMITTER, "title": "Indemnity cap question"}]
_MESSAGES = [
    {"id": Q_MSG, "conversation_id": CONV, "user_id": SUBMITTER, "role": "user",
     "content": "What is the indemnity cap?", "sources": None, "created_at": "2026-06-26T05:00:00Z"},
    {"id": ANSWER_MSG, "conversation_id": CONV, "user_id": SUBMITTER, "role": "assistant",
     "content": "The indemnity cap is 12 months of fees.", "sources": [{"filename": "msa.pdf"}],
     "created_at": "2026-06-26T05:00:05Z"},
]


def _requests(owner=OWNER, chain=None, vault=VAULT):
    return [{
        "id": REQ, "firm_id": FIRM, "vault_id": vault, "artifact_ref": ANSWER_MSG,
        "submitted_by": SUBMITTER, "current_owner": owner, "chain": chain or [], "status": "pending",
    }]


def _tables(*, requests=None, screens=None, conversations=None, messages=None):
    return {
        "review_requests": requests if requests is not None else _requests(),
        "firm_memberships": _MEMBERSHIPS,
        "screens": screens or [],
        "conversations": conversations if conversations is not None else _CONVERSATIONS,
        "messages": messages if messages is not None else _MESSAGES,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── A · the authority gate (the security core) ──")

# current_owner CAN.
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables())
check("A1 current_owner is granted", m.review_artifact_authority(REQ) is not None)

# a CHAIN member CAN (a reviewer may see what they're accountable for).
m = _mgr(caller=CHAIN_MEMBER, firm=FIRM, tables=_tables(requests=_requests(owner=OWNER, chain=[CHAIN_MEMBER])))
check("A2 a chain member is granted", m.review_artifact_authority(REQ) is not None)

# a non-owner / not-in-chain firm member CANNOT.
m = _mgr(caller=STRANGER, firm=FIRM, tables=_tables(requests=_requests(owner=OWNER, chain=[CHAIN_MEMBER])))
check("A3 a non-owner/not-in-chain member is denied", m.review_artifact_authority(REQ) is None)

# a CROSS-FIRM caller CANNOT (the request is firm-scoped → not found in their firm).
m = _mgr(caller=OUTSIDER, firm=OTHER_FIRM, tables=_tables())
check("A4 a cross-firm caller is denied (no existence leak)", m.review_artifact_authority(REQ) is None)

# a SCREENED caller CANNOT — the wall beats the review relationship (deny-overrides, T5).
m = _mgr(caller=OWNER, firm=FIRM,
         tables=_tables(requests=_requests(vault=WALLED_VAULT),
                        screens=[{"user_id": OWNER, "vault_id": WALLED_VAULT, "firm_id": FIRM, "removed_at": None}]))
check("A5 a screened reviewer is denied (wall beats the grant)", m.review_artifact_authority(REQ) is None)

# the wall FAILS CLOSED on a live screen-lookup fault (a NON-missing-table error → is_vault_screened
# re-raises → the authority denies). A missing screens table is the legacy "no wall" path (tested via
# A5's absence), distinct from a live fault.
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables(requests=_requests(vault=WALLED_VAULT)), live_fault=("screens",))
check("A6 a screen-lookup fault fails CLOSED (denied)", m.review_artifact_authority(REQ) is None)

# firm-less caller → denied.
m = _mgr(caller=OWNER, firm=None, tables=_tables())
check("A7 a firm-less caller is denied", m.review_artifact_authority(REQ) is None)


print("\n── B · least privilege / scope ──")
# The artifact resolves ONLY the submitter's message — a same-id message owned by someone else is not read.
poisoned = list(_MESSAGES) + [
    {"id": ANSWER_MSG, "conversation_id": "conv-other", "user_id": STRANGER, "role": "assistant",
     "content": "SOMEONE ELSE'S ANSWER", "sources": None, "created_at": "2026-06-26T05:00:05Z"},
]
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables(messages=poisoned))
art = m.get_review_artifact(REQ)
check("B1 reads only the submitter's artifact (not a same-id foreign msg)",
      art and art.get("available") and "SOMEONE ELSE" not in (art.get("answer_full") or ""),
      str(art))
# A guessed request_id that doesn't exist → None.
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables())
check("B2 an unknown request_id touches nothing", m.get_review_artifact("req-nope") is None)


print("\n── C · content correctness ──")
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables())
art = m.get_review_artifact(REQ)
check("C1 preview has the conversation title", art and art.get("title") == "Indemnity cap question")
check("C2 preview has the question (the user msg before the answer)",
      art and art.get("question") == "What is the indemnity cap?")
check("C3 preview has the answer", art and "indemnity cap is 12 months" in (art.get("answer_full") or ""))
thread = m.get_review_thread(REQ)
check("C4 thread returns the whole conversation oldest-first",
      thread and thread.get("available") and [x["role"] for x in thread["messages"]] == ["user", "assistant"],
      str(thread))


print("\n── D · graceful degrade ──")
# The artifact message was deleted (conversation/messages gone) → available:False, not an error.
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables(messages=[], conversations=[]))
art = m.get_review_artifact(REQ)
check("D1 a deleted artifact ⇒ available:False (no error)", art is not None and art.get("available") is False)
th = m.get_review_thread(REQ)
check("D2 a deleted thread ⇒ available:False", th is not None and th.get("available") is False)
# review_requests table missing ⇒ None (dormant).
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables(), missing=("review_requests",))
check("D3 a missing review_requests table ⇒ None", m.review_artifact_authority(REQ) is None)


print("\n── E · routes (404-on-deny, audit on success) ──")
# A granted reviewer → 200 + audit row.
m = _mgr(caller=OWNER, firm=FIRM, tables=_tables())
m.audit = []
import src.api.routes.matters as _mmod
_orig_log = _mmod.log_audit
_mmod.log_audit = lambda sb, action, rtype=None, rid=None, metadata=None, ip_address=None: \
    getattr(sb, "audit", []).append((action, rid))
try:
    resp = run(mroutes.get_review_artifact(REQ, sb=m))
    check("E1 granted reviewer gets the artifact (200)", resp.available and resp.question == "What is the indemnity cap?")
    check("E2 the artifact view is audited (T10)", ("review.artifact.view", REQ) in m.audit)
    # A denied caller → 404 (not 403, no existence leak), no audit.
    m2 = _mgr(caller=STRANGER, firm=FIRM, tables=_tables(requests=_requests(owner=OWNER, chain=[CHAIN_MEMBER])))
    m2.audit = []
    got_404 = False
    try:
        run(mroutes.get_review_artifact(REQ, sb=m2))
    except HTTPException as e:
        got_404 = (e.status_code == 404)
    check("E3 a denied caller gets 404 (no existence leak)", got_404)
    check("E4 a denied view writes no audit row", m2.audit == [])
    # The thread route audits too.
    m3 = _mgr(caller=OWNER, firm=FIRM, tables=_tables())
    m3.audit = []
    tr = run(mroutes.get_review_thread(REQ, sb=m3))
    check("E5 thread route returns the conversation + audits", tr.available and ("review.thread.view", REQ) in m3.audit)
finally:
    _mmod.log_audit = _orig_log


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  F2j.1 REVIEWABLE-ARTIFACTS GATE:  {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(0 if _failed == 0 else 1)
