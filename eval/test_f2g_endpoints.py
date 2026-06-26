"""F2g backend gate — the TWO thin additive endpoints (offline, $0).

F2g is UI/UX PLUS exactly two additive backend endpoints (plans/F2_FIRM_CONSOLE_PLAN.md §F2g,
resolved 2026-06-25). This gate proves BOTH — cap-gated, firm-scoped, and (for caps) lockstep with
authz.py ROLE_CAPS so the UI can never drift. Drives the LIVE route handlers directly with a fake
SupabaseManager (END-TO-END, not the pure decision). Deterministic; no API, no Supabase, no
extraction/kernel. Run:

    python -u eval/test_f2g_endpoints.py

What it proves (maps to the §3 F2g gate row, "+2 THIN BACKEND endpoints"):
  A. GET /auth/capabilities returns the resolved cap set from the SAME path require_cap trusts
     (resolve_membership + authz.caps_for_role) — for EVERY role, caps == caps_for_role(role), so the
     UI's render decisions cannot drift from ROLE_CAPS. role/firm_id/is_external are correct; a
     delegated verb appears in the payload (and in delegated_verbs).
  B. A legacy/firm-less user degrades to a solo Managing-Partner cap set (byte-identical to pre-F2)
     so the console renders for everyone.
  C. PUT /matters/{vault}/review-chain is cap-gated (manage_matter_team): a paralegal (no cap) is
     403'd BEFORE any write; a partner sets a custom chain and it round-trips via get_matter_review_chain.
  D. The chain PUT is firm/vault-scoped (T2/T3): the body carries no firm_id; a cross-firm vault is a
     404; a reviewer who is NOT a firm member is rejected (400) — the body can't route work outside.
  E. A null/empty chain CLEARS the custom config (reverts to the rank default).
  F. A screened member cannot set the chain (the ethical-wall floor — screen beats the cap, T6).
  G. Every successful chain set is audit-logged (T10: matter.review_chain.set).

GOTCHA (inherited from F2d/F2e): matters.py binds `log_audit` at module load, so we patch the name
IN the matters module too — not only the audit module.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from src.components.authz import caps_for_role, CAPABILITIES  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _raises(fn, status):
    try:
        fn()
        return False
    except HTTPException as e:
        return e.status_code == status


# Fixed ids.
FIRM_A = "firm-A"
FIRM_B = "firm-B"
PARTNER = "user-partner"
PARA = "user-para"
ASSOC = "user-assoc"
OTHERFIRM = "user-otherfirm"
MATTER = "vault-matter"
FIRM_B_VAULT = "vault-firmb"
LEGACY = "user-legacy"        # a firm-less user (no membership) → solo-MP degrade


class FakeSB:
    def __init__(self, user_id, firm_id=FIRM_A):
        self.user_id = user_id
        self.user_email = "x@firm.com"
        self._members = [
            {"user_id": PARTNER, "firm_id": FIRM_A, "role": "partner"},
            {"user_id": ASSOC, "firm_id": FIRM_A, "role": "associate"},
            {"user_id": PARA, "firm_id": FIRM_A, "role": "paralegal"},
            {"user_id": OTHERFIRM, "firm_id": FIRM_B, "role": "partner"},
        ]
        self._collections = {
            MATTER: {"user_id": PARTNER, "firm_id": FIRM_A},
            FIRM_B_VAULT: {"user_id": OTHERFIRM, "firm_id": FIRM_B},
        }
        self._review_config: dict[str, dict] = {}
        self._screens: list[dict] = []
        self._delegations: list[dict] = []
        self._invites: list[dict] = []     # firm_invites rows {id, firm_id, email, role, token_hash, accepted_at}
        self.audit: list[tuple] = []
        self.read_client = self

    @property
    def client(self):
        return self

    def get_user_firm(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        for m in self._members:
            if m["user_id"] == uid and (firm_id is None or m["firm_id"] == firm_id):
                return {"id": m["firm_id"], "name": "Firm", "role": m["role"]}
        return {}

    def user_in_firm(self, user_id, firm_id):
        return any(m["user_id"] == user_id and m["firm_id"] == firm_id for m in self._members)

    def collection_in_firm(self, vault_id, firm_id):
        c = self._collections.get(vault_id)
        return bool(c and c["firm_id"] == firm_id)

    def screened_vault_ids(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return {s["vault_id"] for s in self._screens
                if s["user_id"] == uid and s["removed_at"] is None
                and (firm_id is None or s["firm_id"] == firm_id)}

    def is_vault_screened(self, vault_id, user_id=None, firm_id=None):
        return vault_id in self.screened_vault_ids(user_id, firm_id)

    def matter_member_vault_ids(self, user_id=None, firm_id=None):
        return set()

    def active_delegated_verbs(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        out: set[str] = set()
        for d in self._delegations:
            if d["delegate_id"] == uid and (firm_id is None or d["firm_id"] == firm_id):
                # bounded to the delegator's caps (mirror the real method)
                delegator_role = next((m["role"] for m in self._members
                                       if m["user_id"] == d["delegator_id"]), None)
                bound = caps_for_role(delegator_role) if delegator_role else frozenset()
                out |= (set(d["verbs"]) & set(bound))
        return out

    def rename_firm(self, firm_id, name):
        self._renamed = {"id": firm_id, "name": name.strip()}
        return dict(self._renamed)

    # —— invite rotation / revoke (mirrors the real conditional-update + firm-scope semantics) ——
    def resend_invite(self, invite_id, firm_id, ttl_hours=168):
        for inv in self._invites:
            if inv["id"] == invite_id and inv["firm_id"] == firm_id and inv.get("accepted_at") is None:
                old = inv["token_hash"]
                inv["token_hash"] = f"hash-{invite_id}-rotated"   # a FRESH hash (token rotates)
                row = {k: v for k, v in inv.items() if k != "token_hash"}
                row["token"] = f"raw-{invite_id}-new"
                row["_prev_hash"] = old
                return row
        return {}

    def revoke_invite(self, invite_id, firm_id):
        before = len(self._invites)
        self._invites = [i for i in self._invites
                         if not (i["id"] == invite_id and i["firm_id"] == firm_id and i.get("accepted_at") is None)]
        return len(self._invites) < before

    def get_matter_review_chain(self, vault_id, firm_id):
        cfg = self._review_config.get(vault_id)
        return (cfg or {}).get("chain") if cfg else None

    def set_matter_review_chain(self, firm_id, vault_id, chain, set_by=None):
        self._review_config[vault_id] = {"firm_id": firm_id, "chain": chain, "set_by": set_by}
        return dict(self._review_config[vault_id], vault_id=vault_id)


def _patch_audit(*sbs):
    import src.api.routes.audit as audit_mod
    import src.api.routes.matters as matters_mod
    registry = {id(s): s for s in sbs}

    def _fake_log(_sb, action, resource_type=None, resource_id=None, metadata=None, ip_address=None):
        target = registry.get(id(_sb), _sb)
        target.audit.append((action, resource_type, resource_id, metadata or {}))
    audit_mod.log_audit = _fake_log
    matters_mod.log_audit = _fake_log


from src.api.routes.auth import my_capabilities as route_caps, rename_my_firm as route_rename  # noqa: E402
from src.api.routes.matters import set_matter_review_chain as route_set_chain  # noqa: E402
from src.api.dependencies import resolve_membership  # noqa: E402
from src.api.schemas import SetReviewChainRequest, RenameFirmRequest  # noqa: E402
from src.components.authz import authorize  # noqa: E402

from src.api.routes.admin import (  # noqa: E402
    resend_firm_invite as route_resend, revoke_firm_invite as route_revoke,
)

# Point the module-bound log_audit at a no-op (auth.py + admin.py bind it at import).
import src.api.routes.auth as auth_mod  # noqa: E402
import src.api.routes.admin as admin_mod  # noqa: E402
auth_mod.log_audit = lambda *a, **k: None
admin_mod.log_audit = lambda *a, **k: None

print("\nF2g backend endpoints — caps source-of-truth + custom review-chain (offline)\n")

# ── A. GET /auth/capabilities — lockstep with authz.py ROLE_CAPS, for every internal role ──
print("A. /auth/capabilities is the caps source of truth (no drift from ROLE_CAPS)")
for role in ("managing_partner", "partner", "senior_associate", "associate", "paralegal", "assistant"):
    sb = FakeSB(PARTNER)
    sb._members = [{"user_id": PARTNER, "firm_id": FIRM_A, "role": role}]
    sb.user_id = PARTNER
    res = run(route_caps(sb))
    check(f"{role}: caps == authz.caps_for_role(role)",
          set(res.caps) == set(caps_for_role(role)))
    check(f"{role}: every cap is a known capability",
          all(c in CAPABILITIES for c in res.caps))
_sb_para_caps = FakeSB(PARA)
_sb_para_caps._members = [{"user_id": PARA, "firm_id": FIRM_A, "role": "paralegal"}]
check("paralegal payload is NOT read-only (D0 — has ask/draft/grids)",
      {"ask", "draft", "grids"}.issubset(set(run(route_caps(_sb_para_caps)).caps)))

# external client → is_external True, deny-by-default surface
sb_ext = FakeSB(PARTNER)
sb_ext._members = [{"user_id": PARTNER, "firm_id": FIRM_A, "role": "client"}]
res_ext = run(route_caps(sb_ext))
check("client: is_external True", res_ext.is_external is True)
check("client: role + firm surfaced", res_ext.role == "client" and res_ext.firm_id == FIRM_A)

# delegated verb shows up in the payload + delegated_verbs
sb_del = FakeSB(ASSOC)
sb_del._delegations = [{"firm_id": FIRM_A, "delegator_id": PARTNER, "delegate_id": ASSOC,
                        "verbs": ["override_abstain"], "revoked_at": None}]
res_del = run(route_caps(sb_del))
check("delegation surfaces override_abstain in caps", "override_abstain" in res_del.caps)
check("delegation surfaces override_abstain in delegated_verbs",
      "override_abstain" in res_del.delegated_verbs)

# ── B. legacy/firm-less user degrades to solo-MP ──
print("\nB. legacy/firm-less degrades to solo Managing-Partner (pre-F2 parity)")
sb_legacy = FakeSB(LEGACY)
sb_legacy._members = []          # no membership at all
res_leg = run(route_caps(sb_legacy))
check("firm-less caps == managing_partner caps", set(res_leg.caps) == set(caps_for_role("managing_partner")))

# ── C. PUT review-chain is cap-gated (manage_matter_team) ──
print("\nC. PUT /matters/{vault}/review-chain is cap-gated")
# The route is guarded by Depends(require_cap("manage_matter_team")) — it 403s BEFORE the handler
# runs. Driving the handler directly bypasses that dep, so we assert the cap at the SAME function
# require_cap trusts (authorize), exactly as test_review_chain asserts release_external.
sb_para = FakeSB(PARA)
m_para = resolve_membership(sb_para)
check("paralegal does NOT hold manage_matter_team (require_cap 403s before any write)",
      not authorize(m_para, "manage_matter_team").allow)
check("partner DOES hold manage_matter_team",
      authorize(resolve_membership(FakeSB(PARTNER)), "manage_matter_team").allow)

sb_partner = FakeSB(PARTNER); _patch_audit(sb_partner)
m_partner = resolve_membership(sb_partner)
run(route_set_chain(MATTER, SetReviewChainRequest(chain=[ASSOC, PARTNER]),
                    sb=sb_partner, membership=m_partner))
check("partner sets a custom chain → round-trips via get_matter_review_chain",
      sb_partner.get_matter_review_chain(MATTER, FIRM_A) == [ASSOC, PARTNER])

# ── D. firm/vault-scoped (T2/T3) ──
print("\nD. firm/vault-scoped (T2/T3) — no body firm_id, cross-firm 404, outsider reviewer 400")
sb_p2 = FakeSB(PARTNER); _patch_audit(sb_p2)
m_p2 = resolve_membership(sb_p2)
check("cross-firm vault → 404",
      _raises(lambda: run(route_set_chain(FIRM_B_VAULT, SetReviewChainRequest(chain=[PARTNER]),
                                          sb=sb_p2, membership=m_p2)), 404))
check("reviewer not in firm → 400 (can't route work to an outsider)",
      _raises(lambda: run(route_set_chain(MATTER, SetReviewChainRequest(chain=[OTHERFIRM]),
                                          sb=sb_p2, membership=m_p2)), 400))
check("SetReviewChainRequest has no firm_id field (T3)",
      "firm_id" not in SetReviewChainRequest.model_fields)

# ── E. null/empty clears the custom config ──
print("\nE. null chain clears the custom config (reverts to rank default)")
sb_clear = FakeSB(PARTNER); _patch_audit(sb_clear)
m_clear = resolve_membership(sb_clear)
sb_clear._review_config[MATTER] = {"firm_id": FIRM_A, "chain": [ASSOC, PARTNER]}
run(route_set_chain(MATTER, SetReviewChainRequest(chain=None), sb=sb_clear, membership=m_clear))
check("null chain → get_matter_review_chain returns None", sb_clear.get_matter_review_chain(MATTER, FIRM_A) is None)

# ── F. screened member can't set the chain (the wall floor) ──
print("\nF. the ethical-wall floor — a screened member can't set the chain (screen beats the cap)")
sb_scr = FakeSB(PARTNER); _patch_audit(sb_scr)
sb_scr._screens = [{"user_id": PARTNER, "vault_id": MATTER, "firm_id": FIRM_A, "removed_at": None}]
m_scr = resolve_membership(sb_scr)
check("screened partner blocked by assert_vault_not_screened (403)",
      _raises(lambda: run(route_set_chain(MATTER, SetReviewChainRequest(chain=[ASSOC, PARTNER]),
                                          sb=sb_scr, membership=m_scr)), 403))

# ── G. audit ──
print("\nG. audit (T10) — every successful chain set writes matter.review_chain.set")
check("matter.review_chain.set audited",
      any(a[0] == "matter.review_chain.set" for a in sb_partner.audit))

# ── H. PATCH /auth/firm rename — onboarding (name the backfilled solo firm) ──
print("\nH. PATCH /auth/firm rename (onboarding) — cap-gated manage_members, firm server-resolved")
sb_rn = FakeSB(PARTNER)
m_rn = resolve_membership(sb_rn)
out = run(route_rename(RenameFirmRequest(name="Mehta & Co"), sb=sb_rn, membership=m_rn))
check("partner (manage_members) renames the firm → new name returned", out.name == "Mehta & Co")
check("RenameFirmRequest has no firm_id field (T3 — server-resolved)",
      "firm_id" not in RenameFirmRequest.model_fields)
check("a paralegal does NOT hold manage_members (require_cap 403s the rename before any write)",
      not authorize(resolve_membership(FakeSB(PARA)), "manage_members").allow)

# ── I. invite resend (rotate) + revoke — the delivery/recovery path (researched) ──
print("\nI. invite resend rotates the token (hash-stored recovery) + revoke, both cap-gated + firm-scoped")
sb_inv = FakeSB(PARTNER)
sb_inv._invites = [{"id": "inv-1", "firm_id": FIRM_A, "email": "bob@firm.com", "role": "paralegal",
                    "token_hash": "hash-inv-1-orig", "accepted_at": None}]
m_inv = resolve_membership(sb_inv)
out_r = run(route_resend("inv-1", sb=sb_inv, membership=m_inv))
check("resend returns a FRESH one-time token", bool(out_r.token))
check("resend ROTATED the stored hash (old token dies)",
      sb_inv._invites[0]["token_hash"] != "hash-inv-1-orig")
check("a paralegal does NOT hold manage_members (require_cap 403s resend before any write)",
      not authorize(resolve_membership(FakeSB(PARA)), "manage_members").allow)
# Cross-firm / missing invite → 404 (firm-scoped, the path id can't reach another firm's invite).
check("resend of a non-existent/cross-firm invite → 404",
      _raises(lambda: run(route_resend("nope", sb=FakeSB(PARTNER), membership=resolve_membership(FakeSB(PARTNER)))), 404))
# Revoke a pending invite → gone; revoke again → 404 (idempotent).
sb_rev = FakeSB(PARTNER)
sb_rev._invites = [{"id": "inv-9", "firm_id": FIRM_A, "email": "z@firm.com", "role": "associate",
                    "token_hash": "h", "accepted_at": None}]
run(route_revoke("inv-9", sb=sb_rev, membership=resolve_membership(sb_rev)))
check("revoke removes the pending invite", not sb_rev._invites)
check("revoke of a missing invite → 404 (idempotent, never 500)",
      _raises(lambda: run(route_revoke("inv-9", sb=sb_rev, membership=resolve_membership(sb_rev))), 404))

print(f"\n{'='*60}\nF2g endpoints gate: {_passed} passed, {_failed} failed\n{'='*60}")
sys.exit(1 if _failed else 0)
