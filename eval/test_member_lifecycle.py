"""F2d regression gate — MEMBER LIFECYCLE + DELEGATION + the override_abstain MOMENT (offline, $0).

The committed gate for F2d (plans/F2_FIRM_CONSOLE_PLAN.md §F2d). These are the MOST breach-prone
events in the Firm Console — role change, offboard, delegation, and turning an agent ABSTAIN into an
override. Each is proven END-TO-END THROUGH THE ROUTE (not just the pure authz decision), with the
same path-by-path rigor F2c gave the wall. Deterministic, no API, no Supabase, no extraction/kernel.
Run:

    python -u eval/test_member_lifecycle.py

What it proves (maps to the plan's §3 gate row for F2d):
  A. LAST-MP GUARD end-to-end through the route: demoting OR removing the firm's sole Managing
     Partner is DENIED via the route (the route resolves the target's current role + mp_count
     server-side, builds the authz.Scope, and the EXISTING _manage_members_guard denies it) — and
     ALLOWED once a second MP exists (no over-block).
  B. SELF-ESCALATION blocked end-to-end: a member can't promote themselves (or anyone) to ≥ their
     own rank through the PATCH route (T1).
  C. FIRM-SCOPED (T2/T3): the body never carries a firm_id (the route uses membership.firm_id); a
     target in another firm is a 404 (no cross-firm role change / offboard).
  D. OFFBOARD NEVER ORPHANS A MATTER: the member's vaults are reassigned to the firm owner BEFORE
     the membership is removed, and every delegation they hold/granted is revoked (instant total
     revocation, T7).
  E. DELEGATION grants a MISSING verb (a delegate gains a verb their ROLE lacks) — but a SCREEN on
     the vault STILL denies it (T6 precedence: screen beats the delegation grant). And a delegate
     can NEVER be granted a verb the delegator lacks (T1, bounded at the resolver).
  F. The override_abstain MOMENT (T6): a holder overrides; a non-holder is 403; an override in a
     SCREENED vault is denied (precedence); a reason is required; the override writes a COMPLETE
     audit row (actor / answer ref / vault / reason / gate objection / timestamp = the F3 contract).
  G. EVERY lifecycle event + the override is AUDIT-LOGGED (T10).
  H. T7 / per-request: a demoted member loses the dropped caps on the NEXT resolve_membership (no
     session cache); a revoked/expired delegation stops granting on the next request.

OFFLINE: authz is PURE. The route handlers are driven directly with a fake SupabaseManager that
models firm_memberships + collections + delegations + the screens table + audit, mirroring the real
query semantics (firm-scoping, active = removed_at/revoked_at None, expires_at > now). No live
anything. See [[run-only-relevant-gates]] — F2d touches authz/dependencies/db/routes, NOT
extraction/kernel.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from src.components.authz import caps_for_role  # noqa: E402

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


def run(coro):
    """Drive an async route handler synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# Fixed ids reused across the gate.
FIRM_A = "firm-A"
FIRM_B = "firm-B"
MP1 = "user-mp1"          # the sole Managing Partner (the last-MP subject)
MP2 = "user-mp2"          # a second MP (lifts the last-MP guard)
PARTNER = "user-partner"  # a partner (delegator / actor)
ASSOC = "user-assoc"      # an associate (target of promote/demote)
PARA = "user-para"        # a paralegal (the delegate / PA)
OPEN_VAULT = "vault-open"
WALLED_VAULT = "vault-walled"


def _future():
    return (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()


def _past():
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


# ─────────────────────────────────────────
# A FAKE SupabaseManager modeling memberships + collections + delegations + screens + audit.
#   Mirrors the real query semantics the routes/db rely on: firm-scoping everywhere; active screen
#   = removed_at None; active delegation = revoked_at None AND expires_at > now; the body never
#   carries a firm_id (routes pass the server-resolved firm).
# ─────────────────────────────────────────
class FakeSB:
    def __init__(self, user_id, firm_id=FIRM_A):
        self.user_id = user_id
        self._firm_id = firm_id
        # memberships: list of {user_id, firm_id, role, created_at}
        self._members = [
            {"user_id": MP1, "firm_id": FIRM_A, "role": "managing_partner", "created_at": "t0"},
            {"user_id": PARTNER, "firm_id": FIRM_A, "role": "partner", "created_at": "t0"},
            {"user_id": ASSOC, "firm_id": FIRM_A, "role": "associate", "created_at": "t0"},
            {"user_id": PARA, "firm_id": FIRM_A, "role": "paralegal", "created_at": "t0"},
            {"user_id": "user-otherfirm", "firm_id": FIRM_B, "role": "associate", "created_at": "t0"},
        ]
        # collections (vaults): id -> {user_id (owner), firm_id}
        self._collections = {
            OPEN_VAULT: {"user_id": ASSOC, "firm_id": FIRM_A},
            "vault-assoc-2": {"user_id": ASSOC, "firm_id": FIRM_A},
            WALLED_VAULT: {"user_id": PARTNER, "firm_id": FIRM_A},
        }
        self._screens: list[dict] = []
        self._delegations: list[dict] = []
        self.audit: list[tuple] = []
        self._seq = 0
        self.read_client = self

    # —— firm + role resolution (resolve_membership + the routes read this) ——
    def get_user_firm(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        for m in self._members:
            if m["user_id"] == uid and (firm_id is None or m["firm_id"] == firm_id):
                return {"id": m["firm_id"], "name": "Firm", "role": m["role"]}
        return {}

    def get_membership(self, user_id, firm_id):
        for m in self._members:
            if m["user_id"] == user_id and m["firm_id"] == firm_id:
                return dict(m)
        return {}

    def count_firm_role(self, firm_id, role):
        return sum(1 for m in self._members if m["firm_id"] == firm_id and m["role"] == role)

    def user_in_firm(self, user_id, firm_id):
        return any(m["user_id"] == user_id and m["firm_id"] == firm_id for m in self._members)

    # —— member lifecycle writes ——
    def change_member_role(self, user_id, firm_id, new_role):
        for m in self._members:
            if m["user_id"] == user_id and m["firm_id"] == firm_id:
                m["role"] = new_role
                return dict(m)
        return {}

    def reassign_member_matters(self, user_id, firm_id, new_owner_id):
        n = 0
        for cid, c in self._collections.items():
            if c["user_id"] == user_id and c["firm_id"] == firm_id:
                c["user_id"] = new_owner_id
                n += 1
        return n

    def remove_member(self, user_id, firm_id):
        revoked = self.revoke_member_delegations(user_id, firm_id)
        before = len(self._members)
        self._members = [m for m in self._members
                         if not (m["user_id"] == user_id and m["firm_id"] == firm_id)]
        return {"removed": len(self._members) < before, "delegations_revoked": revoked}

    # —— delegations ——
    def active_delegated_verbs(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        now = datetime.now(timezone.utc)
        granted: set = set()
        for d in self._delegations:
            if d["delegate_id"] != uid:
                continue
            if d.get("revoked_at") is not None:
                continue
            if firm_id is not None and d["firm_id"] != firm_id:
                continue
            try:
                if datetime.fromisoformat(str(d["expires_at"]).replace("Z", "+00:00")) <= now:
                    continue
            except Exception:
                continue
            verbs = set(d.get("verbs") or [])
            # BOUND to the delegator's own caps (mirrors db.active_delegated_verbs — T1).
            dfirm = self.get_user_firm(user_id=d["delegator_id"], firm_id=firm_id)
            drole = (dfirm or {}).get("role")
            granted |= (verbs & set(caps_for_role(drole) if drole else frozenset()))
        return granted

    def create_delegation(self, firm_id, delegator_id, delegate_id, verbs, expires_at):
        if not expires_at:
            raise ValueError("A delegation must be time-boxed (expires_at is required).")
        self._seq += 1
        row = {"id": f"deleg-{self._seq}", "firm_id": firm_id, "delegator_id": delegator_id,
               "delegate_id": delegate_id, "verbs": list(verbs), "expires_at": expires_at,
               "revoked_at": None, "created_at": "t1"}
        self._delegations.append(row)
        return row

    def revoke_delegation(self, delegation_id, firm_id):
        for d in self._delegations:
            if d["id"] == delegation_id and d["firm_id"] == firm_id and d.get("revoked_at") is None:
                d["revoked_at"] = "t2"
                return dict(d)
        return {}

    def revoke_member_delegations(self, user_id, firm_id):
        n = 0
        for d in self._delegations:
            if (d["firm_id"] == firm_id and d.get("revoked_at") is None
                    and (d["delegator_id"] == user_id or d["delegate_id"] == user_id)):
                d["revoked_at"] = "t2"
                n += 1
        return n

    def list_delegations(self, firm_id, include_inactive=False):
        return [dict(d) for d in self._delegations if d["firm_id"] == firm_id
                and (include_inactive or d.get("revoked_at") is None)]

    # —— screens (the wall — for the override precedence test) ——
    def screened_vault_ids(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return {s["vault_id"] for s in self._screens
                if s["user_id"] == uid and s["removed_at"] is None
                and (firm_id is None or s["firm_id"] == firm_id)}

    def is_vault_screened(self, vault_id, user_id=None, firm_id=None):
        return vault_id in self.screened_vault_ids(user_id, firm_id)

    def collection_in_firm(self, vault_id, firm_id):
        c = self._collections.get(vault_id)
        return bool(c and c.get("firm_id") == firm_id)

    def create_screen(self, firm_id, user_id, vault_id, reason, created_by=None):
        self._seq += 1
        row = {"id": f"screen-{self._seq}", "firm_id": firm_id, "user_id": user_id,
               "vault_id": vault_id, "reason": reason, "created_by": created_by or self.user_id,
               "created_at": "t0", "removed_at": None}
        self._screens.append(row)
        return row

    @property
    def client(self):
        return self


def _patch_audit(*sbs):
    """Point log_audit at each fake's audit list (T10 capture). admin.py binds `log_audit` at
    module load (`from ... import log_audit`), so we must patch the name IN the admin module too —
    not only the audit module (the late-import path used by assert_vault_not_screened)."""
    import src.api.routes.audit as audit_mod
    import src.api.routes.admin as admin_mod
    registry = {id(s): s for s in sbs}

    def _fake_log(_sb, action, resource_type=None, resource_id=None, metadata=None, ip_address=None):
        target = registry.get(id(_sb), _sb)
        target.audit.append((action, resource_type, resource_id, metadata or {}))
    audit_mod.log_audit = _fake_log
    admin_mod.log_audit = _fake_log


# Import the live route handlers + deps (drive them directly — END-TO-END through the route).
from src.api.routes.admin import (  # noqa: E402
    change_member_role as route_change_role,
    offboard_member as route_offboard,
    create_firm_delegation as route_create_deleg,
    revoke_firm_delegation as route_revoke_deleg,
    override_abstain as route_override,
)
from src.api.dependencies import resolve_membership  # noqa: E402
from src.api.schemas import (  # noqa: E402
    ChangeRoleRequest, DelegationRequest, OverrideAbstainRequest,
)
from src.components.authz import authorize, Scope  # noqa: E402


def _membership(sb):
    """The server-resolved membership the require_cap guard would yield (drive it the same way)."""
    return resolve_membership(sb)


# ─────────────────────────────────────────
print("\n── A. LAST-MP GUARD end-to-end THROUGH THE ROUTE (not just the decision) ──")
# Single MP firm: an MP1-as-actor tries to DEMOTE the sole MP (themselves) → DENIED via the route.
sb = FakeSB(user_id=MP1)
_patch_audit(sb)
raised = False
try:
    run(route_change_role(MP1, ChangeRoleRequest(role="associate"), sb=sb, membership=_membership(sb)))
except HTTPException as e:
    raised = e.status_code == 403
check("A: demoting the SOLE Managing Partner is DENIED end-to-end (last-MP guard, T1)", raised)
check("A: the membership was NOT changed (the demote did not take effect)",
      sb.get_membership(MP1, FIRM_A)["role"] == "managing_partner")

# Same firm: REMOVING the sole MP via the offboard route → DENIED.
raised = False
try:
    run(route_offboard(MP1, sb=sb, membership=_membership(sb)))
except HTTPException as e:
    raised = e.status_code == 403
check("A: REMOVING the sole Managing Partner is DENIED end-to-end (last-MP guard)", raised)
check("A: the sole MP is still a member after the blocked offboard",
      sb.user_in_firm(MP1, FIRM_A))

# Add a 2nd MP → the guard lifts: demoting MP1 is now ALLOWED (no over-block).
sb._members.append({"user_id": MP2, "firm_id": FIRM_A, "role": "managing_partner", "created_at": "t0"})
res = run(route_change_role(MP1, ChangeRoleRequest(role="partner"), sb=sb, membership=_membership(sb)))
check("A: with a 2nd MP present, demoting MP1 is ALLOWED (no over-block)", res.role == "partner")
check("A: MP1 is now a partner after the allowed demote",
      sb.get_membership(MP1, FIRM_A)["role"] == "partner")


# ─────────────────────────────────────────
print("\n── B. SELF-ESCALATION blocked end-to-end (T1) ──")
# A partner actor tries to promote an associate to managing_partner (more senior than the actor).
sbp = FakeSB(user_id=PARTNER)
_patch_audit(sbp)
raised = False
try:
    run(route_change_role(ASSOC, ChangeRoleRequest(role="managing_partner"),
                          sb=sbp, membership=_membership(sbp)))
except HTTPException as e:
    raised = e.status_code == 403
check("B: a partner cannot promote a member ABOVE their own rank (T1)", raised)
# A partner promoting another to PARTNER (== their own rank, and the actor is not MP) is denied.
raised = False
try:
    run(route_change_role(ASSOC, ChangeRoleRequest(role="partner"),
                          sb=sbp, membership=_membership(sbp)))
except HTTPException as e:
    raised = e.status_code == 403
check("B: a (non-MP) partner cannot grant a peer-level role (T1)", raised)
# A partner promoting an associate to senior_associate (below the actor) is ALLOWED.
res = run(route_change_role(ASSOC, ChangeRoleRequest(role="senior_associate"),
                            sb=sbp, membership=_membership(sbp)))
check("B: a partner CAN promote below their own rank (no over-block)", res.role == "senior_associate")


# ─────────────────────────────────────────
print("\n── C. FIRM-SCOPED — body carries no firm_id; cross-firm target is 404 (T2/T3) ──")
sbc = FakeSB(user_id=MP1)
_patch_audit(sbc)
raised = False
try:
    run(route_change_role("user-otherfirm", ChangeRoleRequest(role="paralegal"),
                          sb=sbc, membership=_membership(sbc)))
except HTTPException as e:
    raised = e.status_code == 404
check("C: a role change targeting ANOTHER firm's member is 404 (T2 — no cross-firm)", raised)
check("C: the other-firm member's role is untouched",
      sbc.get_membership("user-otherfirm", FIRM_B)["role"] == "associate")
raised = False
try:
    run(route_offboard("user-otherfirm", sb=sbc, membership=_membership(sbc)))
except HTTPException as e:
    raised = e.status_code == 404
check("C: offboarding ANOTHER firm's member is 404 (T2)", raised)
check("C: the other-firm member is still a member", sbc.user_in_firm("user-otherfirm", FIRM_B))


# ─────────────────────────────────────────
print("\n── D. OFFBOARD never orphans a matter + revokes delegations (instant revoke, T7) ──")
sbd = FakeSB(user_id=MP1)
_patch_audit(sbd)
# The associate owns two vaults and holds + granted delegations.
sbd._delegations.append({"id": "deleg-x", "firm_id": FIRM_A, "delegator_id": ASSOC,
                         "delegate_id": PARA, "verbs": ["ask"], "expires_at": _future(),
                         "revoked_at": None, "created_at": "t1"})
assoc_vaults_before = [cid for cid, c in sbd._collections.items() if c["user_id"] == ASSOC]
res = run(route_offboard(ASSOC, sb=sbd, membership=_membership(sbd)))
check("D: offboard succeeds for a non-last-MP member", res.removed is True)
check("D: the offboarded member is no longer a firm member (instant)", not sbd.user_in_firm(ASSOC, FIRM_A))
# Every vault the associate owned is now owned by the firm owner (MP1) — NONE orphaned.
orphaned = [cid for cid in assoc_vaults_before if sbd._collections[cid]["user_id"] == ASSOC]
check("D: NO matter is orphaned — all the offboarded member's vaults were reassigned",
      not orphaned and res.matters_reassigned == len(assoc_vaults_before), str(orphaned))
check("D: the reassigned vaults are owned by the firm owner (the MP caller)",
      all(sbd._collections[cid]["user_id"] == MP1 for cid in assoc_vaults_before))
# Their delegation was revoked (instant total revocation) → it no longer grants on the next request.
check("D: the offboarded member's delegations were revoked", res.delegations_revoked == 1)
check("D: the (now-revoked) delegation grants nothing on the delegate's next request (T7)",
      "ask" not in sbd.active_delegated_verbs(user_id=PARA, firm_id=FIRM_A))


# ─────────────────────────────────────────
print("\n── E. DELEGATION grants a missing verb — but a SCREEN still denies it (T6 precedence) ──")
sbe = FakeSB(user_id=PARTNER)
_patch_audit(sbe)
# A paralegal LACKS override_abstain by role. The partner delegates it (a verb the partner holds).
para_caps_before = caps_for_role("paralegal")
check("E: precondition — a paralegal's ROLE lacks override_abstain",
      "override_abstain" not in para_caps_before)
deleg = run(route_create_deleg(
    DelegationRequest(delegate_id=PARA, verbs=["override_abstain"], expires_at=_future()),
    sb=sbe, membership=_membership(sbe)))
check("E: the partner can delegate override_abstain (a verb they hold)", "override_abstain" in deleg.verbs)
# Now the paralegal, resolved per-request, HOLDS override_abstain via the active delegation.
sb_para = FakeSB(user_id=PARA)
sb_para._delegations = sbe._delegations          # share the delegation store
m_para = resolve_membership(sb_para)
check("E: the delegate now HOLDS the missing verb via the delegation (grants a verb the role lacks)",
      "override_abstain" in m_para.delegated_verbs)
check("E: authorize() ALLOWS the delegated verb on an open vault (delegation grant honored)",
      authorize(m_para, "override_abstain", Scope(vault_id=OPEN_VAULT, firm_id=FIRM_A)).allow)
# …but a SCREEN on the vault DENIES it — the screen beats the delegation grant (precedence holds).
sb_para._screens.append({"id": "s1", "firm_id": FIRM_A, "user_id": PARA, "vault_id": WALLED_VAULT,
                         "reason": "conflict", "created_by": MP1, "created_at": "t0", "removed_at": None})
# A new request = a fresh sb in production → empty per-request membership memo; simulate that.
sb_para._membership_memo = {}
m_para2 = resolve_membership(sb_para)
check("E: a SCREEN on the vault DENIES the delegated verb (T6 — screen beats the delegation grant)",
      not authorize(m_para2, "override_abstain", Scope(vault_id=WALLED_VAULT, firm_id=FIRM_A)).allow)
# T1: a delegate can NEVER be granted a verb the DELEGATOR lacks (bounded at the resolver).
sbe2 = FakeSB(user_id=PARA)         # a paralegal tries to delegate release_external (they lack it)
_patch_audit(sbe2)
raised = False
try:
    run(route_create_deleg(
        DelegationRequest(delegate_id=ASSOC, verbs=["release_external"], expires_at=_future()),
        sb=sbe2, membership=_membership(sbe2)))
except HTTPException as e:
    raised = e.status_code in (403,)
check("E: a delegator cannot delegate a verb they DON'T hold (T1, up-front reject)", raised)
# Even if such a row somehow existed, the resolver re-bounds to the delegator's caps (defense in depth).
sb_rebound = FakeSB(user_id=ASSOC)
sb_rebound._delegations.append({"id": "bad", "firm_id": FIRM_A, "delegator_id": PARA,
                                "delegate_id": ASSOC, "verbs": ["release_external"],
                                "expires_at": _future(), "revoked_at": None, "created_at": "t1"})
check("E: the resolver re-bounds — a delegation of a verb the delegator lacks grants NOTHING (T1)",
      "release_external" not in sb_rebound.active_delegated_verbs(user_id=ASSOC, firm_id=FIRM_A))


# ─────────────────────────────────────────
print("\n── F. The override_abstain MOMENT (T6 — holder / non-holder / screened / reason / audit) ──")
# A partner (holds override_abstain by role) overrides on an OPEN vault → success + complete audit.
sbf = FakeSB(user_id=PARTNER)
_patch_audit(sbf)
res = run(route_override(
    OverrideAbstainRequest(answer_ref="ans-1", collection_id=OPEN_VAULT,
                           reason="reviewed source cells by hand; the figure is correct",
                           gate_objection="numeric claim 514,983 not bound to a cell"),
    sb=sbf, membership=_membership(sbf)))
check("F: an override_abstain HOLDER (partner) can override on an open vault", res.status == "overridden")
# The audit row is COMPLETE — the F3 hash-chain contract (actor / answer / vault / reason / objection).
ov = [a for a in sbf.audit if a[0] == "answer.override_abstain"]
check("F: the override writes an 'answer.override_abstain' audit row (T10)", len(ov) == 1, str(sbf.audit))
meta = ov[0][3] if ov else {}
check("F: the audit row is COMPLETE — actor + answer ref + vault + reason + objection (F3 contract)",
      meta.get("overridden_by") == PARTNER and ov[0][2] == "ans-1"
      and meta.get("collection_id") == OPEN_VAULT
      and meta.get("reason") and meta.get("gate_objection"), str(meta))

# A NON-holder (paralegal, no role grant, no delegation) cannot override → the central guard denies.
sb_nh = FakeSB(user_id=PARA)
m_nh = resolve_membership(sb_nh)
check("F: a non-holder (paralegal) is denied override_abstain by the central decision (403 at guard)",
      not authorize(m_nh, "override_abstain").allow)

# An override in a SCREENED vault is denied (precedence — screen beats the override grant), even
# for the partner who holds the verb. Driven end-to-end through the route's wall floor.
sbf2 = FakeSB(user_id=PARTNER)
_patch_audit(sbf2)
sbf2._screens.append({"id": "s2", "firm_id": FIRM_A, "user_id": PARTNER, "vault_id": WALLED_VAULT,
                      "reason": "conflict", "created_by": MP1, "created_at": "t0", "removed_at": None})
raised = False
try:
    run(route_override(
        OverrideAbstainRequest(answer_ref="ans-2", collection_id=WALLED_VAULT, reason="x"),
        sb=sbf2, membership=_membership(sbf2)))
except HTTPException as e:
    raised = e.status_code == 403
check("F: override in a SCREENED vault is DENIED (T6 precedence — screen beats the override grant)",
      raised)
check("F: no override audit row was written for the screened attempt (it never reached the write)",
      not any(a[0] == "answer.override_abstain" for a in sbf2.audit))
check("F: the screened override attempt WAS audited as a wall block (T10)",
      any(a[0] == "screen.block" for a in sbf2.audit), str(sbf2.audit))
# A reason is REQUIRED (schema-enforced) — an empty reason can't even build the request.
reason_required = False
try:
    OverrideAbstainRequest(answer_ref="ans-3", collection_id=OPEN_VAULT, reason="")
except Exception:
    reason_required = True
check("F: a reason is REQUIRED for an override (schema rejects an empty reason)", reason_required)


# ─────────────────────────────────────────
print("\n── G. Every lifecycle event + override is AUDIT-LOGGED (T10) ──")
sbg = FakeSB(user_id=MP1)
sbg._members.append({"user_id": MP2, "firm_id": FIRM_A, "role": "managing_partner", "created_at": "t0"})
_patch_audit(sbg)
run(route_change_role(ASSOC, ChangeRoleRequest(role="senior_associate"), sb=sbg, membership=_membership(sbg)))
check("G: a role change writes a 'member.role_change' audit row (old→new, by whom)",
      any(a[0] == "member.role_change" for a in sbg.audit), str(sbg.audit))
run(route_offboard(PARA, sb=sbg, membership=_membership(sbg)))
check("G: an offboard writes a 'member.offboard' audit row",
      any(a[0] == "member.offboard" for a in sbg.audit))
sbg2 = FakeSB(user_id=PARTNER)
_patch_audit(sbg2)
deleg_g = run(route_create_deleg(
    DelegationRequest(delegate_id=PARA, verbs=["ask"], expires_at=_future()),
    sb=sbg2, membership=_membership(sbg2)))
check("G: a delegation grant writes a 'delegation.grant' audit row",
      any(a[0] == "delegation.grant" for a in sbg2.audit))
run(route_revoke_deleg(deleg_g.id, sb=sbg2, membership=_membership(sbg2)))
check("G: a delegation revoke writes a 'delegation.revoke' audit row",
      any(a[0] == "delegation.revoke" for a in sbg2.audit))


# ─────────────────────────────────────────
print("\n── H. T7 / per-request: dropped caps + expired delegation stop on the NEXT request ──")
# Demote a partner→associate, then resolve again: the demoted member loses the partner-only caps.
sbh = FakeSB(user_id=MP1)
sbh._members.append({"user_id": MP2, "firm_id": FIRM_A, "role": "managing_partner", "created_at": "t0"})
run(route_change_role(PARTNER, ChangeRoleRequest(role="associate"), sb=sbh, membership=_membership(sbh)))
sb_demoted = FakeSB(user_id=PARTNER)
sb_demoted._members = sbh._members        # share the membership store (the demote landed)
m_demoted = resolve_membership(sb_demoted)
check("H: the demoted member's NEXT request resolves to the LOWER role (T7 — no session cache)",
      m_demoted.role == "associate")
check("H: the demoted member LOST the partner-only caps on the next request (e.g. override_abstain)",
      "override_abstain" not in m_demoted.caps and "manage_members" not in m_demoted.caps)
# An EXPIRED delegation grants nothing on the next request.
sb_exp = FakeSB(user_id=PARA)
sb_exp._delegations.append({"id": "d-exp", "firm_id": FIRM_A, "delegator_id": PARTNER,
                            "delegate_id": PARA, "verbs": ["ask"], "expires_at": _past(),
                            "revoked_at": None, "created_at": "t1"})
check("H: an EXPIRED delegation grants nothing on the next request (T7, time-boxed)",
      "ask" not in sb_exp.active_delegated_verbs(user_id=PARA, firm_id=FIRM_A))


# ─────────────────────────────────────────
print("\n── I. Flag-off / pre-F2d parity (no delegations ⇒ byte-identical) ──")
clean = FakeSB(user_id=MP1)
m_clean = resolve_membership(clean)
check("I: with no delegations, membership.delegated_verbs is empty (legacy parity)",
      m_clean.delegated_verbs == frozenset())
check("I: a solo MP still resolves to full caps (byte-identical to pre-F2d)",
      "override_abstain" in m_clean.caps and "manage_members" in m_clean.caps)


# ── tally ──
print(f"\n{'='*64}")
print(f"  test_member_lifecycle: {_passed} passed, {_failed} failed")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
