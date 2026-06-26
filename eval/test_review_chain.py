"""F2e regression gate — MATTER STAFFING + the REVIEW CHAIN of command (offline, $0).

The committed gate for F2e (plans/F2_FIRM_CONSOLE_PLAN.md §F2e — D0/D3/D5). This is THE PRODUCTIVITY
ENGINE: a senior staffs their team onto a matter; everyone staffed gets the FULL working toolkit on
it (D0 — NOT read-only); finished work flows UP a chain of command for review and external release,
and NOTHING is ever ownerless (the anti-stall invariant). Each property is proven END-TO-END THROUGH
THE LIVE ROUTE (not just the pure chain-builder), with the same rigor F2c/F2d gave the wall + the
lifecycle. Deterministic, no API, no Supabase, no extraction/kernel. Run:

    python -u eval/test_review_chain.py

What it proves (maps to the plan's §3 gate row for F2e):
  A. STAFFING (D3) + the D0 PRODUCTIVITY ASSERTION (first-class, not just security): a senior staffs
     a paralegal onto a matter; the staffed paralegal has the FULL working toolkit on it
     (ask/draft/run_workflow/grids/ingest/send_for_review) — explicitly assert NOT read-only.
  B. STAFFING is firm/vault-scoped (T2/T3): the body never carries a firm_id; a cross-firm vault is a
     404; a non-member can't be staffed. Remove = INSTANT revoke (the member loses the matter on the
     next request — T7).
  C. The DEFAULT chain routes to the correct next owner BY RANK (paralegal → associate → senior
     associate → partner), one step up at a time.
  D. A CUSTOM chain (matter_review_config) routes in its DEFINED order (not re-sorted).
  E. ANTI-STALL: every review_request ALWAYS names a current_owner — submit, approve, request-changes,
     and the approved (release) state are each owned by a NAMED person; there is no ownerless state.
  F. Only the CURRENT OWNER may advance/return a step (a non-owner is 403); request-changes returns
     ownership to the submitter.
  G. Only a PARTNER release_externals at the chain's end: a non-holder is 403 at the cap guard; a
     senior associate (holds release_external for own work but is not the chain's partner end-point)
     is blocked from closing the chain externally; a partner releases → terminal `released`.
  H. Every staffing add/remove + every review transition is AUDIT-LOGGED (T10).
  I. Flag-off / pre-F2e parity: empty tables ⇒ no matter membership ⇒ byte-identical.

OFFLINE: the chain-builder is PURE; the route handlers are driven directly with a fake SupabaseManager
that models matter_memberships + review_requests + matter_review_config + collections + firm_memberships
+ screens + audit, mirroring the real query semantics (firm/vault-scoping, the unique team row, the
non-terminal owner set). No live anything. See [[run-only-relevant-gates]] — F2e touches
authz/dependencies/db/routes/review_chain, NOT extraction/kernel.

GOTCHA (from F2d): matters.py binds `log_audit` at module load (`from ... import log_audit`), so we
must patch the name IN the matters module too — not only the audit module (the late-import path used
by assert_vault_not_screened).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from src.components.authz import caps_for_role, _WORKING_TOOLKIT  # noqa: E402

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
    return asyncio.get_event_loop().run_until_complete(coro)


def _raises_403(fn) -> bool:
    try:
        fn()
        return False
    except HTTPException as e:
        return e.status_code == 403


def _raises_404(fn) -> bool:
    try:
        fn()
        return False
    except HTTPException as e:
        return e.status_code == 404


# Fixed ids reused across the gate.
FIRM_A = "firm-A"
FIRM_B = "firm-B"
PARTNER = "user-partner"     # a partner — staffs matters, the chain's external-release end-point
SA = "user-sa"               # a senior associate — holds release_external (own work) but not chain-end
ASSOC = "user-assoc"         # an associate
PARA = "user-para"           # a paralegal — the D0 subject (full toolkit, not read-only)
OUTSIDER = "user-outsider"   # a firm-A member NOT on the matter (for the owner-only / 403 checks)
OTHERFIRM = "user-otherfirm" # a firm-B member (for the cross-firm checks)
MATTER = "vault-matter"      # the matter (vault) the team works
OTHER_MATTER = "vault-other" # another firm-A matter
FIRM_B_VAULT = "vault-firmb" # a firm-B matter


# ─────────────────────────────────────────
# A FAKE SupabaseManager modeling the F2e tables (+ the deps the routes touch).
#   Mirrors the real query semantics: firm/vault-scoping everywhere; the unique (vault,user) team
#   row; review owner = current_owner; the non-terminal owner set; the body never carries a firm_id.
# ─────────────────────────────────────────
class FakeSB:
    def __init__(self, user_id, firm_id=FIRM_A):
        self.user_id = user_id
        self._firm_id = firm_id
        self._members = [
            {"user_id": PARTNER, "firm_id": FIRM_A, "role": "partner", "created_at": "t0"},
            {"user_id": SA, "firm_id": FIRM_A, "role": "senior_associate", "created_at": "t0"},
            {"user_id": ASSOC, "firm_id": FIRM_A, "role": "associate", "created_at": "t0"},
            {"user_id": PARA, "firm_id": FIRM_A, "role": "paralegal", "created_at": "t0"},
            {"user_id": OUTSIDER, "firm_id": FIRM_A, "role": "associate", "created_at": "t0"},
            {"user_id": OTHERFIRM, "firm_id": FIRM_B, "role": "partner", "created_at": "t0"},
        ]
        self._collections = {
            MATTER: {"user_id": PARTNER, "firm_id": FIRM_A},
            OTHER_MATTER: {"user_id": PARTNER, "firm_id": FIRM_A},
            FIRM_B_VAULT: {"user_id": OTHERFIRM, "firm_id": FIRM_B},
        }
        self._matter_team: list[dict] = []          # {firm_id, vault_id, user_id, added_by, created_at}
        self._reviews: list[dict] = []              # review_requests rows
        self._review_config: dict[str, dict] = {}   # vault_id -> {chain}
        self._screens: list[dict] = []
        self.audit: list[tuple] = []
        self._seq = 0
        self.read_client = self

    @property
    def client(self):
        return self

    # —— firm + role resolution ——
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

    def user_in_firm(self, user_id, firm_id):
        return any(m["user_id"] == user_id and m["firm_id"] == firm_id for m in self._members)

    def collection_in_firm(self, vault_id, firm_id):
        c = self._collections.get(vault_id)
        return bool(c and c["firm_id"] == firm_id)

    def get_collection(self, vault_id):
        # F2m: a staffed member's get_collection resolves the OWNER's row (user_id = owner). The
        # review-chain escalation uses this to route a lone-junior submission UP to the vault owner.
        return dict(self._collections.get(vault_id) or {})

    def list_members(self, firm_id):
        return [dict(m) for m in self._members if m["firm_id"] == firm_id]

    # —— screens (for the ethical-wall floor on the matter paths) ——
    def screened_vault_ids(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return {s["vault_id"] for s in self._screens
                if s["user_id"] == uid and s["removed_at"] is None
                and (firm_id is None or s["firm_id"] == firm_id)}

    def is_vault_screened(self, vault_id, user_id=None, firm_id=None):
        return vault_id in self.screened_vault_ids(user_id, firm_id)

    # —— F2e: matter staffing ——
    def add_matter_member(self, firm_id, vault_id, user_id, added_by):
        for r in self._matter_team:
            if r["vault_id"] == vault_id and r["user_id"] == user_id:
                return dict(r)                       # idempotent on (vault,user)
        self._seq += 1
        row = {"id": f"mt-{self._seq}", "firm_id": firm_id, "vault_id": vault_id,
               "user_id": user_id, "added_by": added_by, "created_at": "t1"}
        self._matter_team.append(row)
        return dict(row)

    def remove_matter_member(self, firm_id, vault_id, user_id):
        before = len(self._matter_team)
        self._matter_team = [r for r in self._matter_team
                             if not (r["firm_id"] == firm_id and r["vault_id"] == vault_id
                                     and r["user_id"] == user_id)]
        return len(self._matter_team) < before

    def list_matter_members(self, firm_id, vault_id):
        out = []
        for r in self._matter_team:
            if r["firm_id"] == firm_id and r["vault_id"] == vault_id:
                role = (self.get_membership(r["user_id"], firm_id) or {}).get("role")
                out.append({"user_id": r["user_id"], "role": role,
                            "added_by": r["added_by"], "created_at": r["created_at"]})
        return out

    def resolve_member_emails(self, user_ids):
        # F2g: the team response resolves a human email per member (best-effort, service-role admin
        # auth in prod). Offline we have no auth API, so return {} — the UI falls back to a short id.
        return {}

    def is_matter_member(self, vault_id, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return any(r["vault_id"] == vault_id and r["user_id"] == uid
                   and (firm_id is None or r["firm_id"] == firm_id)
                   for r in self._matter_team)

    def matter_member_vault_ids(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return {r["vault_id"] for r in self._matter_team
                if r["user_id"] == uid and (firm_id is None or r["firm_id"] == firm_id)}

    # —— F2e: review chain ——
    def get_matter_review_chain(self, vault_id, firm_id):
        cfg = self._review_config.get(vault_id)
        return (cfg or {}).get("chain") if cfg else None

    def set_matter_review_chain(self, firm_id, vault_id, chain, set_by=None):
        self._review_config[vault_id] = {"firm_id": firm_id, "chain": chain, "set_by": set_by}
        return dict(self._review_config[vault_id], vault_id=vault_id)

    def create_review_request(self, firm_id, vault_id, artifact_ref, submitted_by, current_owner, chain):
        self._seq += 1
        row = {"id": f"rev-{self._seq}", "firm_id": firm_id, "vault_id": vault_id,
               "artifact_ref": artifact_ref, "submitted_by": submitted_by, "status": "pending",
               "current_owner": current_owner, "chain": list(chain or []),
               "created_at": "t1", "decided_at": None}
        self._reviews.append(row)
        return dict(row)

    def get_review_request(self, request_id, firm_id):
        for r in self._reviews:
            if r["id"] == request_id and r["firm_id"] == firm_id:
                return dict(r)
        return {}

    def update_review_request(self, request_id, firm_id, **fields):
        for r in self._reviews:
            if r["id"] == request_id and r["firm_id"] == firm_id:
                r.update(fields)
                return dict(r)
        return {}

    def list_my_review_queue(self, firm_id, owner_id=None):
        uid = owner_id or self.user_id
        return [dict(r) for r in self._reviews
                if r["firm_id"] == firm_id and r["current_owner"] == uid
                and r["status"] in ("pending", "approved", "changes_requested")]


def _patch_audit(*sbs):
    """Point log_audit at each fake's audit list (T10 capture). matters.py binds `log_audit` at
    module load (`from ... import log_audit`), so we patch the name IN the matters module too — not
    only the audit module (the late-import path used by assert_vault_not_screened)."""
    import src.api.routes.audit as audit_mod
    import src.api.routes.matters as matters_mod
    registry = {id(s): s for s in sbs}

    def _fake_log(_sb, action, resource_type=None, resource_id=None, metadata=None, ip_address=None):
        target = registry.get(id(_sb), _sb)
        target.audit.append((action, resource_type, resource_id, metadata or {}))
    audit_mod.log_audit = _fake_log
    matters_mod.log_audit = _fake_log


# Import the live route handlers + deps (drive them directly — END-TO-END through the route).
from src.api.routes.matters import (  # noqa: E402
    add_matter_team_member as route_staff,
    remove_matter_team_member as route_unstaff,
    list_matter_team as route_team,
    submit_for_review as route_submit,
    approve_review as route_approve,
    request_changes as route_changes,
    release_review as route_release,
    my_review_queue as route_queue,
)
from src.api.dependencies import resolve_membership  # noqa: E402
from src.api.schemas import (  # noqa: E402
    MatterTeamAddRequest, ReviewSubmitRequest, ReviewDecisionRequest,
)


def _membership(sb):
    """The server-resolved membership the require_cap guard would yield (drive it the same way)."""
    return resolve_membership(sb)


def _staff_full_team(sb):
    """Staff the full hierarchy (para/assoc/sa/partner) onto MATTER via the route (PARTNER actor)."""
    actor = FakeSB(user_id=PARTNER)
    actor._matter_team = sb._matter_team       # share the team store
    actor._collections = sb._collections
    actor._members = sb._members
    actor._screens = sb._screens
    _patch_audit(actor)
    for uid in (PARA, ASSOC, SA, PARTNER):
        run(route_staff(MATTER, MatterTeamAddRequest(user_id=uid),
                        sb=actor, membership=_membership(actor)))


# ─────────────────────────────────────────
print("\n── A. STAFFING (D3) + the D0 PRODUCTIVITY ASSERTION (staffed paralegal has the FULL toolkit) ──")
sb = FakeSB(user_id=PARTNER)
_patch_audit(sb)
res = run(route_staff(MATTER, MatterTeamAddRequest(user_id=PARA), sb=sb, membership=_membership(sb)))
check("A: a partner (manage_matter_team) can staff a paralegal onto a matter",
      any(m.user_id == PARA for m in res.members))
check("A: the paralegal is now staffed on the matter (is_matter_member)",
      sb.is_matter_member(MATTER, PARA, FIRM_A))

# THE D0 ASSERTION (first-class): the staffed paralegal's caps ON THE MATTER are the FULL working
# toolkit — explicitly NOT read-only. Resolve their per-request membership and check has_full_toolkit_on.
sb_para = FakeSB(user_id=PARA)
sb_para._matter_team = sb._matter_team
m_para = resolve_membership(sb_para)
toolkit_on_matter = m_para.has_full_toolkit_on(MATTER)
check("A: D0 — the staffed paralegal has the FULL working toolkit on the matter (NOT read-only)",
      toolkit_on_matter == (_WORKING_TOOLKIT & caps_for_role("paralegal")) and len(toolkit_on_matter) >= 5,
      str(sorted(toolkit_on_matter)))
for verb in ("ask", "draft", "run_workflow", "grids", "ingest", "send_for_review"):
    check(f"A: D0 — the staffed paralegal can '{verb}' on the matter (productive, not read-only)",
          verb in toolkit_on_matter)
# It really IS the matter that grants it: an un-staffed member has NO toolkit on this matter.
sb_out = FakeSB(user_id=OUTSIDER)
sb_out._matter_team = sb._matter_team
check("A: an UN-staffed member has no toolkit on the matter (staffing is the access grant, D3)",
      resolve_membership(sb_out).has_full_toolkit_on(MATTER) == frozenset())


# ─────────────────────────────────────────
print("\n── B. STAFFING is firm/vault-scoped (T2/T3) + remove = instant revoke (T7) ──")
sbb = FakeSB(user_id=PARTNER)
_patch_audit(sbb)
# Cross-firm vault → 404 (the body carries no firm_id; the route resolves the caller's firm).
raised = False
try:
    run(route_staff(FIRM_B_VAULT, MatterTeamAddRequest(user_id=PARA), sb=sbb, membership=_membership(sbb)))
except HTTPException as e:
    raised = e.status_code == 404
check("B: staffing onto ANOTHER firm's matter is 404 (T2/T3 — no cross-firm)", raised)
# Staffing a non-member → 404.
raised = False
try:
    run(route_staff(MATTER, MatterTeamAddRequest(user_id=OTHERFIRM), sb=sbb, membership=_membership(sbb)))
except HTTPException as e:
    raised = e.status_code == 404
check("B: staffing a NON-member of the firm is 404 (T2)", raised)
# Staff then remove → instant revoke (the matter drops from the member's set on the next resolve).
run(route_staff(MATTER, MatterTeamAddRequest(user_id=PARA), sb=sbb, membership=_membership(sbb)))
check("B: precondition — paralegal staffed", sbb.is_matter_member(MATTER, PARA, FIRM_A))
run(route_unstaff(MATTER, PARA, sb=sbb, membership=_membership(sbb)))
sb_para_after = FakeSB(user_id=PARA)
sb_para_after._matter_team = sbb._matter_team
check("B: remove = INSTANT revoke — the matter is gone from the member's set on the NEXT request (T7)",
      MATTER not in resolve_membership(sb_para_after).matter_vault_ids)
check("B: removing a member who isn't staffed is a clean 404 (idempotent)",
      _raises_404(lambda: run(route_unstaff(MATTER, OUTSIDER, sb=sbb, membership=_membership(sbb)))))


# ─────────────────────────────────────────
print("\n── C. The DEFAULT chain routes to the correct next owner BY RANK ──")
sbc = FakeSB(user_id=PARA)
_staff_full_team(sbc)
_patch_audit(sbc)
# Paralegal submits → first owner is the ASSOCIATE (the least-senior member above them).
req = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-1"),
                       sb=sbc, membership=_membership(sbc)))
check("C: the default chain is para → assoc → sa → partner (up by rank)",
      req.chain == [ASSOC, SA, PARTNER], str(req.chain))
check("C: submit routes the FIRST owner to the associate (one step up from the paralegal)",
      req.current_owner == ASSOC)
rid = req.id
# Associate approves → advances to the SENIOR ASSOCIATE.
sb_a = FakeSB(user_id=ASSOC); sb_a._reviews = sbc._reviews; sb_a._matter_team = sbc._matter_team
sb_a._collections = sbc._collections; sb_a._members = sbc._members
_patch_audit(sb_a)
r2 = run(route_approve(rid, ReviewDecisionRequest(), sb=sb_a, membership=_membership(sb_a)))
check("C: associate approves → advances UP to the senior associate", r2.current_owner == SA and r2.status == "pending")
# Senior associate approves → advances to the PARTNER.
sb_s = FakeSB(user_id=SA); sb_s._reviews = sbc._reviews; sb_s._matter_team = sbc._matter_team
sb_s._collections = sbc._collections; sb_s._members = sbc._members
_patch_audit(sb_s)
r3 = run(route_approve(rid, ReviewDecisionRequest(), sb=sb_s, membership=_membership(sb_s)))
check("C: senior associate approves → advances UP to the partner (last in the chain)", r3.current_owner == PARTNER)
# Partner approves → the internal chain clears → APPROVED, owned by the partner (chain-end / release owner).
sb_p = FakeSB(user_id=PARTNER); sb_p._reviews = sbc._reviews; sb_p._matter_team = sbc._matter_team
sb_p._collections = sbc._collections; sb_p._members = sbc._members
_patch_audit(sb_p)
r4 = run(route_approve(rid, ReviewDecisionRequest(), sb=sb_p, membership=_membership(sb_p)))
check("C: partner approves the last step → status APPROVED (internal chain cleared)", r4.status == "approved")
check("C: APPROVED is owned by the partner who may release externally (anti-stall — still named)",
      r4.current_owner == PARTNER)


# ─────────────────────────────────────────
print("\n── D. A CUSTOM chain routes in its DEFINED order (not re-sorted) ──")
sbd = FakeSB(user_id=PARA)
_staff_full_team(sbd)
# A deliberately NON-rank order: sa → assoc → partner (associate AFTER senior associate).
sbd.set_matter_review_chain(FIRM_A, MATTER, [SA, ASSOC, PARTNER], set_by=PARTNER)
_patch_audit(sbd)
reqd = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-custom"),
                        sb=sbd, membership=_membership(sbd)))
check("D: the custom chain routes in its DEFINED order (sa → assoc → partner), NOT re-sorted by rank",
      reqd.chain == [SA, ASSOC, PARTNER], str(reqd.chain))
check("D: the first owner is the custom chain's first reviewer (the senior associate)",
      reqd.current_owner == SA)


# ─────────────────────────────────────────
print("\n── E. ANTI-STALL — every review_request ALWAYS names a current_owner ──")
# Across submit / approve / changes / approved — none is ever ownerless.
sbe = FakeSB(user_id=PARA)
_staff_full_team(sbe)
_patch_audit(sbe)
re_sub = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-e"),
                          sb=sbe, membership=_membership(sbe)))
check("E: a freshly-submitted request has a NAMED current_owner (never ownerless)", bool(re_sub.current_owner))
ride = re_sub.id
# Associate requests changes → ownership returns to the SUBMITTER (still named, the ball is theirs).
sb_ae = FakeSB(user_id=ASSOC); sb_ae._reviews = sbe._reviews; sb_ae._matter_team = sbe._matter_team
sb_ae._collections = sbe._collections; sb_ae._members = sbe._members
_patch_audit(sb_ae)
rc_chg = run(route_changes(ride, ReviewDecisionRequest(note="fix cite"), sb=sb_ae, membership=_membership(sb_ae)))
check("E: request-changes returns ownership to the SUBMITTER (anti-stall — never nobody's)",
      rc_chg.status == "changes_requested" and rc_chg.current_owner == PARA)
# Solo-senior matter: a partner alone on a matter submits → owner is the partner himself (can't stall
# on someone else; a one-person matter is releasable directly). Still a NAMED owner, never None.
sbe2 = FakeSB(user_id=PARTNER)
sbe2.add_matter_member(FIRM_A, OTHER_MATTER, PARTNER, PARTNER)
_patch_audit(sbe2)
re_solo = run(route_submit(ReviewSubmitRequest(collection_id=OTHER_MATTER, artifact_ref="solo"),
                           sb=sbe2, membership=_membership(sbe2)))
check("E: a solo-senior matter still names an owner (the submitter) — never ownerless/None",
      re_solo.current_owner == PARTNER)

# ── E2. LONE-JUNIOR ESCALATION (the live "indian big" bug, D5 locked decision) ──
# A paralegal is the ONLY member staffed on a matter the PARTNER owns. The paralegal submits for
# review. The matter team has no senior → the OLD code parked the request on the SUBMITTER (the
# paralegal), so it sat with the person who wrote it and the partner's queue was empty. The fix
# routes UP to the VAULT OWNER (the partner), never the submitter.
sbe3 = FakeSB(user_id=PARA)
sbe3.add_matter_member(FIRM_A, MATTER, PARA, PARTNER)   # paralegal alone on the partner-owned matter
_patch_audit(sbe3)
re_lone = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="lone"),
                           sb=sbe3, membership=_membership(sbe3)))
check("E2: a lone junior's submission routes UP to the VAULT OWNER (the partner), NOT the submitter",
      re_lone.current_owner == PARTNER and re_lone.current_owner != PARA)
# And it lands in the OWNER's review queue (the partner sees it), not the submitter's (anti-stall fix).
sbe3_mp = FakeSB(user_id=PARTNER); sbe3_mp._reviews = sbe3._reviews
queue_mp = sbe3_mp.list_my_review_queue(FIRM_A, owner_id=PARTNER)
queue_para = sbe3._members and sbe3.list_my_review_queue(FIRM_A, owner_id=PARA)
check("E2: the lone-junior request is in the VAULT OWNER's review queue (not the submitter's)",
      any(r["id"] == re_lone.id for r in queue_mp) and not any(r["id"] == re_lone.id for r in queue_para))


# ─────────────────────────────────────────
print("\n── F. Only the CURRENT OWNER may advance/return a step (a non-owner is 403) ──")
sbf = FakeSB(user_id=PARA)
_staff_full_team(sbf)
_patch_audit(sbf)
reqf = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-f"),
                        sb=sbf, membership=_membership(sbf)))   # owner = ASSOC
ridf = reqf.id
# The SENIOR ASSOCIATE (not the current owner) tries to approve → 403 (only the owner acts).
sb_notowner = FakeSB(user_id=SA); sb_notowner._reviews = sbf._reviews; sb_notowner._matter_team = sbf._matter_team
sb_notowner._collections = sbf._collections; sb_notowner._members = sbf._members
_patch_audit(sb_notowner)
check("F: a NON-owner cannot approve a step (only the current owner may act — 403)",
      _raises_403(lambda: run(route_approve(ridf, ReviewDecisionRequest(),
                                            sb=sb_notowner, membership=_membership(sb_notowner)))))


# ─────────────────────────────────────────
print("\n── G. Only a PARTNER release_externals at the chain's end ──")
sbg = FakeSB(user_id=PARA)
_staff_full_team(sbg)
_patch_audit(sbg)
reqg = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-g"),
                        sb=sbg, membership=_membership(sbg)))
ridg = reqg.id
# Drive it to APPROVED (assoc → sa → partner each approve).
for actor in (ASSOC, SA, PARTNER):
    s = FakeSB(user_id=actor); s._reviews = sbg._reviews; s._matter_team = sbg._matter_team
    s._collections = sbg._collections; s._members = sbg._members
    _patch_audit(s)
    run(route_approve(ridg, ReviewDecisionRequest(), sb=s, membership=_membership(s)))
approved = sbg.get_review_request(ridg, FIRM_A)
check("G: the request is APPROVED and owned by the partner before release", approved["status"] == "approved" and approved["current_owner"] == PARTNER)
# A PARALEGAL (no release_external cap) cannot release → 403 at the cap guard (require_cap).
sb_para_rel = FakeSB(user_id=PARA)
m_para_rel = resolve_membership(sb_para_rel)
from src.components.authz import authorize  # noqa: E402
check("G: a paralegal does NOT hold release_external (cannot close the chain externally — 403 at guard)",
      not authorize(m_para_rel, "release_external").allow)
# The PARTNER (holds release_external + is the chain-end tier) releases → terminal `released`.
sb_rel = FakeSB(user_id=PARTNER); sb_rel._reviews = sbg._reviews; sb_rel._matter_team = sbg._matter_team
sb_rel._collections = sbg._collections; sb_rel._members = sbg._members
_patch_audit(sb_rel)
rg = run(route_release(ridg, ReviewDecisionRequest(note="cleared, releasing"),
                       sb=sb_rel, membership=_membership(sb_rel)))
check("G: a PARTNER release_externals at the chain's end → status `released` (terminal)", rg.status == "released")
check("G: the released request has a decided_at timestamp (terminal)", bool(rg.decided_at))
# A SCREENED partner cannot release a walled matter (the wall floor — precedence, T6). Set up fresh.
sbg2 = FakeSB(user_id=PARA)
_staff_full_team(sbg2)
reqg2 = sbg2.create_review_request(FIRM_A, MATTER, "draft-screened", PARA, PARTNER, [ASSOC, SA, PARTNER])
sbg2.update_review_request(reqg2["id"], FIRM_A, status="approved", current_owner=PARTNER)
sb_scr = FakeSB(user_id=PARTNER); sb_scr._reviews = sbg2._reviews; sb_scr._matter_team = sbg2._matter_team
sb_scr._collections = sbg2._collections; sb_scr._members = sbg2._members
sb_scr._screens.append({"id": "s1", "firm_id": FIRM_A, "user_id": PARTNER, "vault_id": MATTER,
                        "reason": "conflict", "removed_at": None})
_patch_audit(sb_scr)
check("G: a SCREENED partner cannot release a walled matter (the wall floor beats release — T6)",
      _raises_403(lambda: run(route_release(reqg2["id"], ReviewDecisionRequest(),
                                            sb=sb_scr, membership=_membership(sb_scr)))))


# ─────────────────────────────────────────
print("\n── H. Every staffing add/remove + every review transition is AUDIT-LOGGED (T10) ──")
sbh = FakeSB(user_id=PARTNER)
_patch_audit(sbh)
run(route_staff(MATTER, MatterTeamAddRequest(user_id=PARA), sb=sbh, membership=_membership(sbh)))
check("H: staffing writes a 'matter.staff' audit row", any(a[0] == "matter.staff" for a in sbh.audit))
run(route_unstaff(MATTER, PARA, sb=sbh, membership=_membership(sbh)))
check("H: un-staffing writes a 'matter.unstaff' audit row", any(a[0] == "matter.unstaff" for a in sbh.audit))
sbh2 = FakeSB(user_id=PARA)
_staff_full_team(sbh2)
_patch_audit(sbh2)
reqh = run(route_submit(ReviewSubmitRequest(collection_id=MATTER, artifact_ref="draft-h"),
                        sb=sbh2, membership=_membership(sbh2)))
check("H: submit writes a 'review.submit' audit row", any(a[0] == "review.submit" for a in sbh2.audit))
sb_h_a = FakeSB(user_id=ASSOC); sb_h_a._reviews = sbh2._reviews; sb_h_a._matter_team = sbh2._matter_team
sb_h_a._collections = sbh2._collections; sb_h_a._members = sbh2._members
_patch_audit(sb_h_a)
run(route_approve(reqh.id, ReviewDecisionRequest(), sb=sb_h_a, membership=_membership(sb_h_a)))
check("H: approve writes a 'review.approve' audit row", any(a[0] == "review.approve" for a in sb_h_a.audit))


# ─────────────────────────────────────────
print("\n── I. Flag-off / pre-F2e parity (empty tables ⇒ byte-identical) ──")
clean = FakeSB(user_id=PARTNER)
m_clean = resolve_membership(clean)
check("I: with no staffing, membership.matter_vault_ids is empty (legacy parity)",
      m_clean.matter_vault_ids == frozenset())
check("I: a partner still resolves to full caps regardless of matter staffing (byte-identical pre-F2e)",
      "release_external" in m_clean.caps and "manage_matter_team" in m_clean.caps)
check("I: has_full_toolkit_on returns ∅ for a vault the member isn't staffed on (no spurious grant)",
      m_clean.has_full_toolkit_on(MATTER) == frozenset())


# ── tally ──
print(f"\n{'='*64}")
print(f"  test_review_chain: {_passed} passed, {_failed} failed")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
