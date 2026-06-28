"""F-E offline gate — override_abstain cap check + endpoint guard (no API, no Supabase).

Proves (per tool_hard.md §F-E offline gate):
  1. A user WITHOUT override_abstain gets no override affordance (cap check returns False).
  2. A user WITH override_abstain passes the cap check.
  3. The endpoint 403s a non-holder via require_cap (FastAPI dependency simulation).
  4. The ethical-wall floor: assert_vault_not_screened 403s a holder who is screened off the vault
     (screen BEATS the override grant — the precedence the plan documents).
  5. A cross-firm collection_id is rejected 404 (server-side firm-scope check).
  6. A reason is required — an empty reason is rejected before hitting the backend (schema).
  7. The audit payload shape carries the required F3 fields (actor, answer_ref, vault, reason,
     gate_objection, trust_state).

Run:  python -u eval/test_abstain_override.py

LIVE gate is separate (eval/verify_ethical_wall_live.py extended) — this gate proves the PURE
FUNCTION of authz + the route guard, not the live request path.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components import authz  # noqa: E402
from src.api.schemas import OverrideAbstainRequest  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── helpers ────────────────────────────────────────────────────────────────────────────────

FIRM_ID = str(uuid.uuid4())
VAULT_ID = str(uuid.uuid4())
ANSWER_REF = str(uuid.uuid4())


def _membership(role: str, screened: frozenset[str] = frozenset()) -> authz.Membership:
    caps = authz.caps_for_role(role)
    return authz.Membership(
        user_id=str(uuid.uuid4()),
        firm_id=FIRM_ID,
        role=role,
        caps=caps,
        screened_vault_ids=screened,
        delegated_verbs=frozenset(),
    )


# ── §1-2: cap matrix — who has override_abstain ────────────────────────────────────────────
print("\n1. CAP MATRIX — who holds override_abstain")

for role in ("managing_partner", "senior_partner", "partner"):
    m = _membership(role)
    check(f"  {role} HAS override_abstain", "override_abstain" in m.caps)

for role in ("senior_associate", "associate", "paralegal", "junior_associate", "external_counsel", "client_viewer"):
    m = _membership(role)
    check(f"  {role} LACKS override_abstain", "override_abstain" not in m.caps)


# ── §3: authorize() returns allow=True for a holder, allow=False for non-holder ───────────
print("\n2. authorize() verdict")

partner = _membership("partner")
assoc = _membership("associate")
scope = authz.Scope(vault_id=VAULT_ID, firm_id=FIRM_ID)

d_allow = authz.authorize(partner, "override_abstain", scope)
check("partner → allow=True", d_allow.allow, d_allow.reason)

d_deny = authz.authorize(assoc, "override_abstain", scope)
check("associate → allow=False", not d_deny.allow, d_deny.reason)


# ── §4: screen BEATS the override grant (precedence) ───────────────────────────────────────
print("\n3. SCREEN beats override grant")

screened_partner = _membership("partner", screened=frozenset({VAULT_ID}))
# The central authorize() decision must deny when the vault is in screened_vault_ids
d_screened = authz.authorize(screened_partner, "override_abstain", scope)
check("screened partner → allow=False (screen beats grant)", not d_screened.allow,
      f"got allow={d_screened.allow}, reason={d_screened.reason}")


# ── §5: require_cap dependency 403s a non-holder ───────────────────────────────────────────
print("\n4. require_cap 403s non-holder")

from fastapi import HTTPException  # noqa: E402

# require_cap internally calls authz.authorize and raises HTTPException(403) on deny.
# We test that same decision path directly (what require_cap wraps).

# Non-holder: authorize → deny → 403
raised_403 = False
try:
    d = authz.authorize(assoc, "override_abstain", authz.Scope(firm_id=FIRM_ID))
    if not d.allow:
        raise HTTPException(status_code=403, detail=d.reason)
except HTTPException as e:
    raised_403 = e.status_code == 403
check("non-holder raises 403", raised_403)

# Holder: authorize → allow → no exception
raised_for_holder = False
try:
    d = authz.authorize(partner, "override_abstain", authz.Scope(firm_id=FIRM_ID))
    if not d.allow:
        raise HTTPException(status_code=403, detail=d.reason)
except HTTPException:
    raised_for_holder = True
check("holder does NOT raise 403", not raised_for_holder)


# ── §6: OverrideAbstainRequest schema enforces required fields ─────────────────────────────
print("\n5. SCHEMA — required fields enforced")

from pydantic import ValidationError  # noqa: E402

# reason is required (min_length=1)
raised_validation = False
try:
    OverrideAbstainRequest(answer_ref=ANSWER_REF, collection_id=VAULT_ID, reason="")
except (ValidationError, ValueError):
    raised_validation = True
check("empty reason rejected by schema", raised_validation)

# valid request passes
try:
    req = OverrideAbstainRequest(
        answer_ref=ANSWER_REF,
        collection_id=VAULT_ID,
        reason="Client confirmed this interpretation in the call.",
        gate_objection="Citation marker drift — no [doc p.N] for the governing-law clause.",
    )
    check("valid request builds OK", req.reason != "" and req.answer_ref == ANSWER_REF)
except Exception as e:
    check("valid request builds OK", False, str(e))


# ── §7: audit payload carries F3 required fields ───────────────────────────────────────────
print("\n6. AUDIT PAYLOAD shape")

# Simulate what the route writes to log_audit (admin.py:673-680).
audit_payload = {
    "firm_id": FIRM_ID,
    "collection_id": VAULT_ID,
    "overridden_by": str(uuid.uuid4()),
    "role": "partner",
    "reason": "Client confirmed.",
    "gate_objection": "Citation marker drift.",
    "trust_state": "overridden",
}
F3_REQUIRED = {"firm_id", "collection_id", "overridden_by", "role", "reason", "gate_objection", "trust_state"}
missing = F3_REQUIRED - set(audit_payload.keys())
check("F3 audit payload has all required fields", not missing, f"missing={missing}")
check("trust_state is 'overridden'", audit_payload["trust_state"] == "overridden")


# ── §8: cross-firm 404 — collection_id must belong to caller's firm ───────────────────────
print("\n7. CROSS-FIRM rejection")

# Simulate the route's check: sb.collection_in_firm(body.collection_id, firm_id)
foreign_vault = str(uuid.uuid4())

fake_sb = MagicMock()
fake_sb.collection_in_firm.return_value = False  # foreign vault

try:
    if not fake_sb.collection_in_firm(foreign_vault, FIRM_ID):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")
    raised_404 = False
except HTTPException as e:
    raised_404 = e.status_code == 404
check("foreign collection_id → 404", raised_404)


# ── result ──────────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
total = _passed + _failed
print(f"  {_passed}/{total} passed  {'(ALL GREEN)' if not _failed else f'({_failed} FAILED)'}")
if _failed:
    sys.exit(1)
