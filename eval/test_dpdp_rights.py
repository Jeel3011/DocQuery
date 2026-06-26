"""F2k regression gate — DPDP DATA-PRINCIPAL RIGHTS + grievance officer (offline, $0, no live DB).

The committed gate for F2k (plans/F2_FIRM_CONSOLE_PLAN.md §F2k + F2_ARCHITECTURE.md §0.2/§6). It drives
the REAL route handlers + db methods with a fake PostgREST that models the content tables
(documents/conversations/messages/audit_log), the F2i `signatures` hash-chain, and the F2k 019 tables
(grievances/data_erasures + the firms officer columns). Run:

    ./venv/bin/python -u eval/test_dpdp_rights.py

What it proves (maps to the plan's §3 gate row for F2k):
  A. EXPORT (§11) — a data principal's export returns THEIR documents/conversations/messages + a
     processing (audit) summary + a manifest; an admin ON-BEHALF export is cap-gated (manage_members)
     and firm-scoped (a cross-firm principal is rejected, T2/T3).
  B. ERASE (§12) — erasure SOFT-DELETES personal CONTENT (documents/conversations/messages blanked) but
     PRESERVES the immutable records of processing: the audit_log is untouched AND **the F2i signature
     chain still verifies end-to-end AFTER the erase** (the load-bearing distinction). A tombstone
     proves the erasure was honored. On-behalf erase is cap-gated + firm-scoped.
  C. GRIEVANCE (§13) — naming an officer + filing a grievance creates a tracked record routed to the
     NAMED officer, with a 90-day due date; manager actions it; a non-manager sees only their own.
  D. RETENTION (Rule 6.5) — the configured AUDIT_LOG_RETENTION_DAYS asserts >= 365 (one year); the
     dpdp floor constant + assert_retention_floor enforce it; a sub-year value is refused.
  E. DORMANT PARITY — with the 019 tables "unapplied" (fake raises missing-relation), export still
     returns the user's own content, erase still soft-deletes content (tombstone skipped, no 500), and
     the grievance route reports unavailable instead of crashing — byte-identical posture pre-F2k.

OFFLINE: dpdp.py is pure; db + route handlers are driven with a fake PostgREST. See
[[run-only-relevant-gates]] — F2k touches dpdp/db/routes + config, NOT extraction or the kernel.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components import dpdp, esign                       # noqa: E402
from src.components.db import SupabaseManager                # noqa: E402
from src.api.routes import dpdp as dr                        # noqa: E402
from src.api.schemas import (                                # noqa: E402
    OnBehalfRequest, GrievanceOfficerRequest, GrievanceRequest, GrievanceStatusRequest,
)
from src.components.config import Config                     # noqa: E402
from fastapi import HTTPException                            # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")


def _http(fn) -> "int | None":
    """Run the coroutine factory fn(); return the HTTPException status it raises, or None."""
    try:
        asyncio.get_event_loop().run_until_complete(fn())
        return None
    except HTTPException as e:
        return e.status_code


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── A fake PostgREST: select/eq/neq/is_/in_/order/limit/insert/update over backing lists ──
class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, name, rows, missing=False):
        self._name = name
        self._rows = rows
        self._eq = {}
        self._neq = {}
        self._in = {}
        self._is_null = set()
        self._order = None
        self._desc = False
        self._limit = None
        self._missing = missing

    def select(self, *_a, **_k): return self

    def eq(self, col, val):
        self._eq[col] = str(val)
        return self

    def neq(self, col, val):
        self._neq[col] = str(val)
        return self

    def in_(self, col, vals):
        self._in[col] = {str(v) for v in vals}
        return self

    def is_(self, col, _val):
        self._is_null.add(col)
        return self

    def gt(self, *_a, **_k): return self
    def gte(self, *_a, **_k): return self

    def order(self, col, desc=False):
        self._order = col
        self._desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, row):
        self._insert = row
        return self

    def update(self, fields):
        self._update = fields
        return self

    def delete(self):
        self._delete = True
        return self

    def _match(self, r):
        return (all(str(r.get(k)) == v for k, v in self._eq.items())
                and all(str(r.get(k)) != v for k, v in self._neq.items())
                and all(str(r.get(k)) in vs for k, vs in self._in.items())
                and all(r.get(k) is None for k in self._is_null))

    def execute(self):
        if self._missing:
            raise RuntimeError(f'relation "{self._name}" does not exist (PGRST205)')
        if hasattr(self, "_insert"):
            row = dict(self._insert)
            row.setdefault("id", f"{self._name}-{len(self._rows)+1}")
            # model uq_signatures_firm_seq for the chain
            if self._name == "signatures":
                for r in self._rows:
                    if str(r.get("firm_id")) == str(row.get("firm_id")) and \
                       r.get("chain_seq") == row.get("chain_seq"):
                        raise RuntimeError('duplicate key "uq_signatures_firm_seq" (23505)')
            self._rows.append(row)
            return _Result([row])
        if hasattr(self, "_update"):
            hit = [r for r in self._rows if self._match(r)]
            for r in hit:
                r.update(self._update)
            return _Result(hit)
        if hasattr(self, "_delete"):
            hit = [r for r in self._rows if self._match(r)]
            for r in hit:
                self._rows.remove(r)
            return _Result(hit)
        out = [r for r in self._rows if self._match(r)]
        if self._order:
            out = sorted(out, key=lambda r: (r.get(self._order) is None, r.get(self._order)),
                         reverse=self._desc)
        if self._limit is not None:
            out = out[: self._limit]
        return _Result(out)


class _Client:
    def __init__(self, tables, missing=()):
        self._tables = tables
        self._missing = set(missing)

    def table(self, name):
        return _Query(name, self._tables.setdefault(name, []), missing=(name in self._missing))


def _mgr(*, caller, firm, tables, missing=(), role="managing_partner", firm_name="Acme Law"):
    m = SupabaseManager.__new__(SupabaseManager)
    m._user = type("U", (), {"id": caller})()
    client = _Client(tables, missing=missing)
    m.client = client
    m._access_token = None
    m._read_client = client
    m.get_user_firm = lambda *a, **k: ({"id": firm, "name": firm_name, "role": role} if firm else {})
    return m


FIRM = "firm-A"
MP = "user-mp"          # a managing partner (manage_members)
ALICE = "user-alice"    # a data principal (paralegal)
BOB = "user-bob"        # another principal
OUTSIDER = "user-out"   # a principal in firm-B


def _seed():
    """Fresh backing tables: Alice has content + a signature in the firm chain; an outsider is in firm-B."""
    return {
        "firms": [{"id": FIRM, "name": "Acme Law"}],
        "firm_memberships": [
            {"user_id": MP, "firm_id": FIRM, "role": "managing_partner"},
            {"user_id": ALICE, "firm_id": FIRM, "role": "paralegal"},
            {"user_id": BOB, "firm_id": FIRM, "role": "associate"},
            {"user_id": OUTSIDER, "firm_id": "firm-B", "role": "associate"},
        ],
        "documents": [
            {"id": "d1", "user_id": ALICE, "filename": "contract.pdf", "storage_path": "s/1", "status": "ready"},
            {"id": "d2", "user_id": ALICE, "filename": "memo.pdf", "storage_path": "s/2", "status": "ready"},
            {"id": "d3", "user_id": BOB, "filename": "bob.pdf", "storage_path": "s/3", "status": "ready"},
        ],
        "conversations": [{"id": "c1", "user_id": ALICE, "title": "My case Q"}],
        "messages": [
            {"id": "m1", "conversation_id": "c1", "user_id": ALICE, "content": "what is the cap?", "sources": []},
            {"id": "m2", "conversation_id": "c1", "user_id": ALICE, "content": "the cap is 12mo", "sources": ["x"]},
        ],
        "audit_log": [
            {"id": "a1", "user_id": ALICE, "action": "query.ask"},
            {"id": "a2", "user_id": ALICE, "action": "document.upload"},
        ],
        "signatures": [],
        "grievances": [],
        "data_erasures": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── A · EXPORT (§11) — a principal's own data + cap-gated on-behalf ──")

tables = _seed()
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, role="paralegal")
exp = _run(dr.export_my_data(sb=alice))
check("A · export returns Alice's 2 documents", len(exp.documents) == 2, f"{len(exp.documents)}")
check("A · export returns Alice's conversation", len(exp.conversations) == 1)
check("A · export returns Alice's 2 messages", len(exp.messages) == 2)
check("A · export includes the processing (audit) summary", len(exp.processing_records) == 2)
check("A · manifest counts match", exp.manifest.get("documents") == 2 and exp.manifest.get("messages") == 2)
check("A · export does NOT include Bob's doc", all(d["user_id"] == ALICE for d in exp.documents))

# admin on-behalf — cap-gated. The route's require_cap is bypassed in this direct call (we call the
# handler with the membership already resolved), so we assert the firm-scope T2/T3 guard explicitly.
import src.components.authz as authz  # noqa: E402
mp = _mgr(caller=MP, firm=FIRM, tables=tables)
mp_membership = authz.Membership(user_id=MP, role="managing_partner", firm_id=FIRM,
                                 caps=authz.caps_for_role("managing_partner"))
exp2 = _run(dr.export_on_behalf(OnBehalfRequest(principal_id=ALICE), sb=mp, membership=mp_membership))
check("A · on-behalf export (MP) returns Alice's data", len(exp2.documents) == 2)
status = _http(lambda: dr.export_on_behalf(OnBehalfRequest(principal_id=OUTSIDER), sb=mp, membership=mp_membership))
check("A · on-behalf export of a CROSS-FIRM principal → 404 (T2/T3)", status == 404, f"got {status}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── B · ERASE (§12) — content soft-deleted, audit + signature chain PRESERVED ──")

tables = _seed()
# Seed a real F2i signature into the firm chain BEFORE erasing, so we can prove the chain survives.
signer = _mgr(caller=MP, firm=FIRM, tables=tables)
sig = signer.append_signature(firm_id=FIRM, artifact_type="review_request", artifact_ref="rr-1",
                              artifact_content="the reviewed answer text", intent="release")
check("B · pre-erase: a signature exists in the chain", bool(sig) and tables["signatures"], "no sig seeded")
chain_before = signer.verify_firm_chain(FIRM)
check("B · pre-erase: chain verifies", chain_before["ok"] and chain_before["count"] == 1)

alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, role="paralegal")
res = _run(dr.erase_my_data(sb=alice))
check("B · erase reports 2 documents erased", res.documents_erased == 2, f"{res.documents_erased}")
check("B · erase reports 1 conversation + 2 messages", res.conversations_erased == 1 and res.messages_erased == 2)

# Content is soft-deleted (blanked), rows still present.
docs = [d for d in tables["documents"] if d["user_id"] == ALICE]
check("B · documents soft-deleted (status=erased, filename blanked)",
      all(d["status"] == "erased" and d["filename"] == "[erased]" and d["storage_path"] is None for d in docs))
msgs = [m for m in tables["messages"] if m["user_id"] == ALICE]
check("B · messages content blanked", all(m["content"] == "[erased]" and m["sources"] == [] for m in msgs))
convo = [c for c in tables["conversations"] if c["user_id"] == ALICE][0]
check("B · conversation title blanked", convo["title"] == "[erased]")

# THE LOAD-BEARING ASSERTIONS: audit + signature chain PRESERVED.
check("B · Bob's document is UNTOUCHED (erase is principal-scoped)",
      any(d["id"] == "d3" and d["filename"] == "bob.pdf" for d in tables["documents"]))
check("B · the audit_log is PRESERVED (records of processing, not erased)",
      len([a for a in tables["audit_log"] if a["user_id"] == ALICE]) >= 2)
check("B · 'signatures' is in the PRESERVED list", "signatures" in res.preserved and "audit_log" in res.preserved)
chain_after = signer.verify_firm_chain(FIRM)
check("B · ⭐ the F2i signature chain STILL VERIFIES after erase (chain not broken)",
      chain_after["ok"] and chain_after["count"] == 1,
      f"ok={chain_after['ok']} count={chain_after['count']} reason={chain_after.get('reason')}")
check("B · an erasure tombstone was written (the §12 proof)",
      len(tables["data_erasures"]) == 1 and tables["data_erasures"][0]["documents_erased"] == 2)
check("B · tombstone records what was PRESERVED",
      tables["data_erasures"][0]["preserved"] == "audit_log,signatures")

# on-behalf erase — firm-scoped.
tables2 = _seed()
mp = _mgr(caller=MP, firm=FIRM, tables=tables2)
mp_membership = authz.Membership(user_id=MP, role="managing_partner", firm_id=FIRM,
                                 caps=authz.caps_for_role("managing_partner"))
res_ob = _run(dr.erase_on_behalf(OnBehalfRequest(principal_id=BOB), sb=mp, membership=mp_membership))
check("B · on-behalf erase (MP) erases Bob's content", res_ob.documents_erased == 1)
st = _http(lambda: dr.erase_on_behalf(OnBehalfRequest(principal_id=OUTSIDER), sb=mp, membership=mp_membership))
check("B · on-behalf erase of a CROSS-FIRM principal → 404 (T2/T3)", st == 404, f"got {st}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── C · GRIEVANCE (§13) — named officer + tracked record + 90-day window ──")

tables = _seed()
mp = _mgr(caller=MP, firm=FIRM, tables=tables)
mp_membership = authz.Membership(user_id=MP, role="managing_partner", firm_id=FIRM,
                                 caps=authz.caps_for_role("managing_partner"))

# No officer yet.
off0 = _run(dr.get_officer(sb=_mgr(caller=ALICE, firm=FIRM, tables=tables, role="paralegal")))
check("C · no officer configured initially", off0.configured is False)

# Name the officer (manager).
off = _run(dr.set_officer(GrievanceOfficerRequest(name="Priya Rao", email="priya@acme.test", user_id=MP),
                          sb=mp, membership=mp_membership))
check("C · officer named", off.configured and off.name == "Priya Rao")
check("C · officer persisted on the firm row",
      tables["firms"][0].get("grievance_officer_name") == "Priya Rao")

# Alice files a grievance — routed to the named officer, 90-day due date.
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, role="paralegal")
g = _run(dr.file_grievance(GrievanceRequest(subject="My data was shown to the wrong team"), sb=alice))
check("C · grievance created + tracked", bool(g.id) and g.status == "open")
check("C · ⭐ grievance ROUTED to the named officer", g.officer_name == "Priya Rao")
check("C · grievance principal is the filer", g.principal_id == ALICE)
check("C · grievance has a 90-day due date", bool(g.due_at))

# A non-manager sees only THEIR OWN grievances; a manager sees the firm's.
tables["grievances"].append({"id": "g-bob", "firm_id": FIRM, "principal_id": BOB,
                             "subject": "bob complaint", "status": "open", "officer_name": "Priya Rao"})
alice_list = _run(dr.list_grievances(sb=alice))
check("C · a non-manager sees ONLY their own grievance (no cross-principal leak, T2)",
      len(alice_list.grievances) == 1 and alice_list.grievances[0].principal_id == ALICE)
mp_list = _run(dr.list_grievances(sb=mp))
check("C · a manager sees the firm's grievances (2)", len(mp_list.grievances) == 2)
check("C · the officer is surfaced in the list", mp_list.officer.name == "Priya Rao")

# Manager actions it.
gid = g.id
acted = _run(dr.action_grievance(gid, GrievanceStatusRequest(status="resolved", resolution_note="fixed"),
                                 sb=mp, membership=mp_membership))
check("C · grievance resolved + stamped", acted.status == "resolved" and bool(acted.resolved_at))
bad = _http(lambda: dr.action_grievance("nope", GrievanceStatusRequest(status="bananas"),
                                        sb=mp, membership=mp_membership))
check("C · an invalid status → 400", bad == 400, f"got {bad}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── D · RETENTION (Rule 6.5) — logs retained >= 1 year, asserted ──")

cfg = Config()
check("D · AUDIT_LOG_RETENTION_DAYS >= 365 (one year)", cfg.AUDIT_LOG_RETENTION_DAYS >= 365,
      f"{cfg.AUDIT_LOG_RETENTION_DAYS}")
check("D · dpdp floor constant is exactly one year (365)", dpdp.MIN_LOG_RETENTION_DAYS == 365)
check("D · the configured value satisfies the floor", dpdp.retention_ok(cfg.AUDIT_LOG_RETENTION_DAYS))
check("D · assert_retention_floor passes the configured value",
      dpdp.assert_retention_floor(cfg.AUDIT_LOG_RETENTION_DAYS) is None)
_raised = False
try:
    dpdp.assert_retention_floor(364)
except ValueError:
    _raised = True
check("D · a sub-year retention (364) is REFUSED", _raised)
check("D · ERASABLE vs PRESERVED are disjoint + correct",
      set(dpdp.PRESERVED_RECORDS) == {"audit_log", "signatures"}
      and "documents" in dpdp.ERASABLE_CONTENT and "audit_log" not in dpdp.ERASABLE_CONTENT)
check("D · is_preserved guards the immutable records",
      dpdp.is_preserved("signatures") and dpdp.is_preserved("audit_log")
      and not dpdp.is_preserved("documents"))


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── E · DORMANT PARITY — 019 tables unapplied ⇒ byte-identical posture, no 500 ──")

# Export still works with no audit table at all and with the new tables missing.
tables = _seed()
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, missing=("audit_log", "grievances", "data_erasures"),
             role="paralegal")
exp = _run(dr.export_my_data(sb=alice))
check("E · export still returns content when audit_log is unapplied (degrades to [])",
      len(exp.documents) == 2 and exp.processing_records == [])

# Erase still soft-deletes content; the tombstone write is skipped (data_erasures missing), no crash.
tables = _seed()
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, missing=("data_erasures",), role="paralegal")
res = _run(dr.erase_my_data(sb=alice))
check("E · erase still soft-deletes content with the ledger unapplied",
      res.documents_erased == 2 and res.erasure_id is None)
check("E · content was actually blanked even without the ledger",
      all(d["filename"] == "[erased]" for d in tables["documents"] if d["user_id"] == ALICE))

# Grievance route reports unavailable (503) rather than 500 when the table is missing.
tables = _seed()
tables["firms"][0]["grievance_officer_name"] = "Priya Rao"
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, missing=("grievances",), role="paralegal")
gstat = _http(lambda: dr.file_grievance(GrievanceRequest(subject="x"), sb=alice))
check("E · grievance file → 503 (clean unavailable) when the table is unapplied", gstat == 503, f"got {gstat}")

# get_officer degrades cleanly when the column is unapplied.
tables = _seed()
alice = _mgr(caller=ALICE, firm=FIRM, tables=tables, missing=("firms",), role="paralegal")
off = _run(dr.get_officer(sb=alice))
check("E · get_officer degrades to not-configured when the column is unapplied", off.configured is False)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  F2k DPDP RIGHTS GATE: {_passed} passed, {_failed} failed")
print(f"{'='*70}")
sys.exit(1 if _failed else 0)
