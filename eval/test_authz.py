"""F2b regression gate — central authorization (offline, no API, no Supabase, $0).

The committed gate for F2b (plans/F2_FIRM_CONSOLE_PLAN.md §F2b). It proves THE authorization
slice: the pure decision (authz.authorize) over the full role×verb×scope matrix, the T1
self-escalation + last-MP guards, the deny-overrides precedence (ethical-wall + delegation seams),
and that the require_cap guard 403s BEFORE the handler/DB. Deterministic, offline. Run:

    python -u eval/test_authz.py

What it proves (maps to the plan's §3 gate row for F2b):
  1. ROLE_CAPS ↔ schemas.ROLES lockstep; every cap in ROLE_CAPS is a known CAPABILITY.
  2. The §0.8 default matrix is encoded EXACTLY (every role × every verb, allow/deny).
  3. MP-of-one allowed everything (legacy parity — byte-identical to pre-F2).
  4. D0: paralegal + assistant ARE granted ask/draft/run-agent/grids/ingest — assert NOT
     read-only (the productivity-first correction); external client/guest deny-by-default.
  5. paralegal denied release_external + override_abstain (403 + a human reason).
  6. T1 self-escalation: granting a role ≥ your own is denied; last-MP demote/remove denied.
  7. Deny-overrides precedence (D2): a screen DENY beats a role grant; a delegation grant
     allows a verb the role lacks; cross-firm scope (T2/T3) is denied.
  8. The require_cap guard raises 403 BEFORE the handler/DB — a denied call never touches the
     DB (a tripwire fake-sb proves no mutation happens on deny); a body-supplied firm_id is
     ignored in favor of the server-resolved one.

OFFLINE: authz is PURE (no DB/LLM/network). The guard test uses a fake SupabaseManager whose
DB access is a tripwire — it never runs a real query. No extraction, no kernel (F2b touches
neither — see [[run-only-relevant-gates]]).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import ROLES, ROLE_RANK  # noqa: E402
from src.components.authz import (  # noqa: E402
    CAPABILITIES,
    ROLE_CAPS,
    EXTERNAL_ROLES,
    Decision,
    Membership,
    Scope,
    authorize,
    caps_for_role,
)

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


def _m(role: str, **kw) -> Membership:
    """A resolved membership for `role` with its default caps (the resolver's behavior)."""
    return Membership(user_id="u", firm_id="f1", role=role, caps=caps_for_role(role), **kw)


def _allow(role: str, verb: str, scope: Scope = Scope()) -> bool:
    return authorize(_m(role), verb, scope).allow


# ── 1. lockstep + integrity ────────────────────────────────────────────────────────────────
check("ROLE_CAPS covers exactly the 9 ROLES",
      set(ROLE_CAPS.keys()) == set(ROLES),
      f"role_caps={sorted(ROLE_CAPS)} vs ROLES={sorted(ROLES)}")
_all_granted = set().union(*ROLE_CAPS.values())
check("every cap granted by any role is a known CAPABILITY",
      _all_granted.issubset(CAPABILITIES),
      f"unknown={_all_granted - CAPABILITIES}")
check("CAPABILITIES has the full §0.8 verb set (17)", len(CAPABILITIES) == 17,
      f"got {len(CAPABILITIES)}: {sorted(CAPABILITIES)}")
for required in ("create_vault", "ingest", "ask", "draft", "run_workflow", "grids",
                 "send_for_review", "release_external", "manage_matter_team", "manage_members",
                 "view_billing", "delete", "sign_certificate", "edit_playbooks",
                 "publish_to_firm_brain", "run_sentinel", "override_abstain"):
    check(f"CAPABILITIES includes '{required}'", required in CAPABILITIES)
check("EXTERNAL_ROLES = {client, guest}", EXTERNAL_ROLES == frozenset({"client", "guest"}))


# ── 2. the LOCKED §0.8 default matrix, encoded EXACTLY (role × verb) ──────────────────────────
# Expected allow-sets per role, transcribed from plans/F2_FIRM_CONSOLE_PLAN.md §0.8.
_WORKING = {"create_vault", "ingest", "ask", "draft", "run_workflow", "grids", "send_for_review"}
_PARTNER = _WORKING | {
    "release_external", "manage_matter_team", "delete", "sign_certificate", "override_abstain",
    "run_sentinel", "manage_members", "view_billing", "edit_playbooks", "publish_to_firm_brain",
}
EXPECTED = {
    "managing_partner": _PARTNER,
    "senior_partner":   _PARTNER,
    "partner":          _PARTNER,
    "senior_associate": _WORKING | {"release_external", "manage_matter_team", "delete"},
    "associate":        set(_WORKING),
    "paralegal":        set(_WORKING),
    "assistant":        set(_WORKING),
    "client":           {"ask"},
    "guest":            set(),
}
for role, expected_caps in EXPECTED.items():
    for verb in sorted(CAPABILITIES):
        want = verb in expected_caps
        got = _allow(role, verb)
        check(f"matrix: {role} {'CAN' if want else 'CANNOT'} {verb}", got == want,
              f"want allow={want}, got={got}")


# ── 3. MP-of-one allowed everything (legacy parity) ──────────────────────────────────────────
mp = _m("managing_partner")
check("MP-of-one is allowed every capability (legacy byte-identical)",
      all(authorize(mp, v).allow for v in CAPABILITIES))


# ── 4. D0 — paralegal/assistant are NOT read-only; externals deny-by-default ──────────────────
for role in ("paralegal", "assistant"):
    for verb in ("ask", "draft", "run_workflow", "grids", "ingest", "send_for_review", "create_vault"):
        check(f"D0: {role} CAN {verb} (NOT read-only)", _allow(role, verb))
check("guest is deny-by-default (cannot ask)", not _allow("guest", "ask"))
check("client may ask (the one external grant)", _allow("client", "ask"))
check("client cannot ingest", not _allow("client", "ingest"))
check("client cannot release_external", not _allow("client", "release_external"))


# ── 5. paralegal denied release_external + override (403-shaped: deny + a human reason) ───────
for verb in ("release_external", "override_abstain"):
    d = authorize(_m("paralegal"), verb)
    check(f"paralegal denied {verb} with a reason", (not d.allow) and bool(d.reason), d.reason)


# ── 6. T1 — self-escalation + last-MP guards ─────────────────────────────────────────────────
# An associate who somehow holds manage_members still cannot grant a role >= their own.
assoc_mm = Membership("u", "f1", "associate", caps=caps_for_role("associate") | {"manage_members"})
check("T1: associate cannot grant a MORE-senior role (partner)",
      not authorize(assoc_mm, "manage_members", Scope(target_role="partner")).allow)
check("T1: associate cannot grant a PEER role (associate)",
      not authorize(assoc_mm, "manage_members", Scope(target_role="associate")).allow)
check("T1: associate CAN grant a JUNIOR role (paralegal)",
      authorize(assoc_mm, "manage_members", Scope(target_role="paralegal")).allow)
# A partner cannot grant managing_partner (more senior than partner).
check("T1: partner cannot grant managing_partner (more senior)",
      not authorize(_m("partner"), "manage_members", Scope(target_role="managing_partner")).allow)
# An MP CAN grant a peer MP (the add-a-co-owner flow).
check("T1: MP can grant a co-MP (peer add-owner)",
      authorize(mp, "manage_members", Scope(target_role="managing_partner")).allow)
# Last-MP guard: demoting / removing the sole MP is denied; with 2 MPs it's allowed.
check("T1: demoting the LAST managing_partner is denied",
      not authorize(mp, "manage_members",
                    Scope(target_current_role="managing_partner", target_role="partner",
                          is_self=True, mp_count=1)).allow)
check("T1: removing the LAST managing_partner is denied",
      not authorize(mp, "manage_members",
                    Scope(target_current_role="managing_partner", is_removal=True,
                          mp_count=1)).allow)
check("T1: demoting an MP when 2 exist is allowed",
      authorize(mp, "manage_members",
                Scope(target_current_role="managing_partner", target_role="partner",
                      mp_count=2)).allow)


# ── 7. deny-overrides precedence (D2) + cross-firm (T2/T3) ────────────────────────────────────
# Screen DENY beats a role grant — even for an MP (the ethical wall is a hard boundary).
mp_screened = _m("managing_partner", screened_vault_ids=frozenset({"v9"}))
d = authorize(mp_screened, "ask", Scope(vault_id="v9"))
check("precedence: a screen DENY overrides an MP's role grant (deny-overrides)",
      (not d.allow) and "ethical wall" in d.reason.lower(), d.reason)
check("precedence: an UN-screened vault is still allowed for the same MP",
      authorize(mp_screened, "ask", Scope(vault_id="v-other")).allow)
# Delegation grant (F2e seam): a verb the role lacks is allowed via an active delegation.
pa = _m("assistant", delegated_verbs=frozenset({"release_external"}))
check("precedence: a delegation grants release_external the assistant role lacks",
      authorize(pa, "release_external").allow)
check("precedence: a screen STILL beats a delegation grant",
      not authorize(
          _m("assistant", delegated_verbs=frozenset({"ask"}),
             screened_vault_ids=frozenset({"v9"})),
          "ask", Scope(vault_id="v9")).allow)
# Cross-firm confused-deputy (T2/T3): a resolved scope firm != caller's firm is denied.
d = authorize(mp, "delete", Scope(firm_id="OTHER-FIRM"))
check("T2/T3: a cross-firm scope is denied even for an MP",
      (not d.allow) and "another firm" in d.reason.lower(), d.reason)
check("T2/T3: the caller's OWN firm scope is allowed", authorize(mp, "delete", Scope(firm_id="f1")).allow)
# Unknown verb fails closed.
check("unknown verb fails closed (deny)", not authorize(mp, "fly_to_the_moon").allow)


# ── 8. the require_cap guard 403s BEFORE the handler/DB; body firm_id is ignored ──────────────
from fastapi import HTTPException  # noqa: E402
from src.api.dependencies import require_cap, resolve_membership  # noqa: E402
from src.components.db import SupabaseManager  # noqa: E402


class _Tripwire:
    """A fake DB client that ALLOWS the membership read (firm_memberships SELECT) but TRIPS on any
    business-data write/mutation — so we can prove a denied guard never reaches a handler's DB
    write. An audit_log insert is the EXPECTED deny trace (T10), tracked separately.

    `membership_role` is the role the firm_memberships SELECT returns (None ⇒ firm-less, so
    resolve_membership falls back to solo-MP)."""
    def __init__(self, membership_role):
        self.membership_role = membership_role
        self.mutated = False        # a HANDLER business-data write happened (must stay False on deny)
        self.audited = False        # the deny trace was written to audit_log (T10)

    def table(self, name):
        return _TripQuery(self, name)


class _TripQuery:
    def __init__(self, client, name):
        self._c = client
        self._name = name
        self._op = "select"

    def select(self, *a, **k):
        self._op = "select"
        return self

    # Any mutating op trips the wire (a handler write would call one of these). The audit_log
    # insert is the EXPECTED deny trace (T10) — it is the proof we WANT, not a handler write —
    # so it does not count as a business-data mutation.
    def insert(self, *a, **k):
        if self._name == "audit_log":
            self._c.audited = True
        else:
            self._c.mutated = True
        self._op = "insert"
        return self

    def update(self, *a, **k):
        self._c.mutated = True
        self._op = "update"
        return self

    def upsert(self, *a, **k):
        self._c.mutated = True
        self._op = "upsert"
        return self

    def delete(self, *a, **k):
        self._c.mutated = True
        self._op = "delete"
        return self

    def eq(self, *a, **k):
        return self

    def is_(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def execute(self):
        # The membership read returns the role for resolve_membership; the inner-join shape
        # mirrors get_user_firm's select("firm_id, role, firms!inner(id, name)"). A None role
        # models a firm-less user (no rows) → solo-MP fallback.
        if self._name == "firm_memberships" and self._op == "select" and self._c.membership_role:
            return type("R", (), {"data": [
                {"firm_id": "f1", "role": self._c.membership_role,
                 "firms": {"id": "f1", "name": "Firm"}}
            ], "count": 1})()
        return type("R", (), {"data": [], "count": 0})()


def _fake_sb(role) -> SupabaseManager:
    sb = SupabaseManager.__new__(SupabaseManager)
    tw = _Tripwire(role)
    sb.client = tw
    sb._read_client = None  # _access_token is None ⇒ read_client falls back to .client
    sb._access_token = None

    class _U:
        id = "user-1"
        email = "x@firm.com"
    sb._user = _U()
    return sb


# A paralegal hitting a release_external-gated route: the guard must 403 and the handler's DB
# write must NEVER happen (mutated stays False).
sb_para = _fake_sb("paralegal")
guard = require_cap("release_external")
denied = False
try:
    guard(sb=sb_para)
except HTTPException as e:
    denied = e.status_code == 403
check("guard: paralegal release_external → 403", denied)
check("guard: a DENIED call never mutated business data (403 before the handler/DB)",
      sb_para.client.mutated is False)
check("guard: the deny was audit-logged (T10)", sb_para.client.audited is True)

# An MP hitting the same route: the guard ALLOWS and returns the resolved membership.
sb_mp = _fake_sb("managing_partner")
allowed_membership = guard(sb=sb_mp)
check("guard: MP release_external → allowed (returns the resolved membership)",
      allowed_membership is not None and allowed_membership.role == "managing_partner")

# Legacy/firm-less user (membership_role=None ⇒ no rows) degrades to solo-MP ⇒ allowed
# everything (byte-identical to pre-F2).
legacy_m = resolve_membership(_fake_sb(None))
check("legacy/firm-less user resolves to solo Managing-Partner (byte-identical)",
      legacy_m.role == "managing_partner")
check("legacy user is allowed every capability (no power stripped)",
      all(authorize(legacy_m, v).allow for v in CAPABILITIES))

# T2/T3 at the resolution boundary: the membership firm comes from the SERVER read, never a body.
# resolve_membership takes no body firm_id by default — it resolves the user's own firm.
check("T2/T3: resolve_membership ignores any body firm_id (server-resolved firm only)",
      resolve_membership(_fake_sb("partner")).firm_id == "f1")


print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
