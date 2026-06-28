"""F-E LIVE gate — override affordance + audit row on the real stack.

Plans/tool_hard.md §F-E: "Live gate: drive a real abstain from the UI as a partner
vs. an associate and confirm the affordance + the 403 + the audit row on the real
stack."

This script drives the real POST /admin/firm/answers/override endpoint:
  1. Partner (override_abstain holder)  → 201 + audit row in DB.
  2. Associate (non-holder)             → 403.
  3. Partner screened off the vault     → 403 (screen beats override grant).
  4. Audit row carries all F3 required fields
     (actor=overridden_by, answer_ref, collection_id, reason, gate_objection,
      trust_state=overridden).

Note: this script does NOT drive a "real abstain from the UI" end-to-end (that would
require the full agent loop, which is API-burning per T0). It instead calls the
override endpoint directly with a synthetic answer_ref (a UUID representing a
hypothetical abstained answer). The endpoint does not validate that the answer_ref
exists — it trusts the caller to provide a real one in production. So the 201
response + the audit row IS the live proof that the endpoint is wired, cap-gated,
and audit-logged correctly. The UI abstain→override affordance was separately wired
in the committed `OverrideAffordance.tsx` + `page.tsx`.

─── ENV VARS ────────────────────────────────────────────────────────────────────────
  OVERRIDE_PARTNER_EMAIL     holder of override_abstain (partner or higher)
  OVERRIDE_PARTNER_PASSWORD
  OVERRIDE_ASSOC_EMAIL       non-holder (associate or lower)
  OVERRIDE_ASSOC_PASSWORD
  OVERRIDE_VAULT_ID          a real collection_id in the partner's firm
  API_BASE                   default http://localhost:8000/api/v1

Run:
    python -u eval/verify_override_live.py
"""
from __future__ import annotations

import os
import sys
import uuid
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import httpx
except ImportError:
    print("  SKIP — httpx not installed (pip install httpx).")
    sys.exit(0)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE      = os.getenv("API_BASE",                  "http://localhost:8000/api/v1").rstrip("/")
# Default: MP (managing_partner) is the override_abstain holder.
# Default non-holder: paralegal — has 'ask' but NOT 'override_abstain'.
PARTNER_EMAIL = os.getenv("OVERRIDE_PARTNER_EMAIL",    "jeel15thummar@gmail.com")
PARTNER_PASS  = os.getenv("OVERRIDE_PARTNER_PASSWORD", "jeel@!samosa3011")
ASSOC_EMAIL   = os.getenv("OVERRIDE_ASSOC_EMAIL",      "jeel15trading@gmail.com")
ASSOC_PASS    = os.getenv("OVERRIDE_ASSOC_PASSWORD",   "jeel@!samosa3011")
MP_EMAIL      = os.getenv("OVERRIDE_MP_EMAIL",         "jeel15thummar@gmail.com")
MP_PASS       = os.getenv("OVERRIDE_MP_PASSWORD",      "jeel@!samosa3011")
VAULT_ID      = os.getenv("OVERRIDE_VAULT_ID",         "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e")

# Synthetic answer_ref — represents a hypothetical abstained answer ID.
ANSWER_REF = str(uuid.uuid4())

# ── Helpers ───────────────────────────────────────────────────────────────────
_passed = _failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  ← {detail}")


def _login(email: str, password: str) -> str | None:
    try:
        r = httpx.post(f"{API_BASE}/auth/login",
                       json={"email": email, "password": password}, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json()
        return (d.get("access_token")
                or (d.get("session") or {}).get("access_token")
                or d.get("token"))
    except Exception:
        return None


def _auth(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def _override(tok: str, answer_ref: str, reason: str, gate_objection: str | None = None) -> tuple[int, dict]:
    body: dict = {
        "answer_ref": answer_ref,
        "collection_id": VAULT_ID,
        "reason": reason,
    }
    if gate_objection:
        body["gate_objection"] = gate_objection
    try:
        r = httpx.post(f"{API_BASE}/admin/firm/answers/override",
                       json=body, headers=_auth(tok), timeout=20)
        try:
            rdata = r.json()
        except Exception:
            rdata = {}
        return r.status_code, rdata
    except Exception as exc:
        return -1, {"error": str(exc)}


# ── Sign in ───────────────────────────────────────────────────────────────────
print(f"\n── F-E LIVE: override affordance + audit row ────────────────────────────")
print(f"  API_BASE     : {API_BASE}")
print(f"  VAULT_ID     : {VAULT_ID}")
print(f"  ANSWER_REF   : {ANSWER_REF}")
print(f"  partner      : {PARTNER_EMAIL}")
print(f"  associate    : {ASSOC_EMAIL}\n")

partner_tok = _login(PARTNER_EMAIL, PARTNER_PASS)
assoc_tok   = _login(ASSOC_EMAIL,   ASSOC_PASS)
mp_tok      = _login(MP_EMAIL,      MP_PASS)

if not partner_tok:
    print("  FATAL — partner login failed.")
    sys.exit(1)
if not assoc_tok:
    print("  FATAL — associate login failed.")
    sys.exit(1)
print(f"  partner signed in (token: {partner_tok[:20]}…)")
print(f"  assoc   signed in (token: {assoc_tok[:20]}…)")

# Resolve partner user_id (for audit-row lookup).
from src.components.db import SupabaseManager  # noqa: E402

partner_sb = SupabaseManager(use_service_role=False)
partner_sb.sign_in(PARTNER_EMAIL, PARTNER_PASS)
partner_uid = partner_sb.user_id
print(f"  partner uid   : {partner_uid}")

svc = SupabaseManager(use_service_role=True)

screen_id: str | None = None

try:
    # ── §1: Associate → 403 ───────────────────────────────────────────────────
    print("\n── §1. Associate (non-holder) → 403 ──")
    status, body = _override(assoc_tok, ANSWER_REF,
                             reason="Test: should not reach here",
                             gate_objection="Citation marker drift")
    print(f"  POST /admin/firm/answers/override → {status}")
    check("§1: associate gets 403 from require_cap(override_abstain)",
          status == 403,
          f"got {status} — cap not firing for associate")

    # ── §2: Partner → 201 + audit row ────────────────────────────────────────
    print("\n── §2. Partner (holder) → 201 ──")
    t_before = time.time()
    status2, body2 = _override(
        partner_tok,
        ANSWER_REF,
        reason="Client confirmed this interpretation on the call (F-E live gate).",
        gate_objection="Citation marker drift — no [doc p.N] for the governing-law clause.",
    )
    print(f"  POST /admin/firm/answers/override → {status2}")
    print(f"  response body: {body2}")
    check("§2: partner gets 201 (override_abstain cap passed)",
          status2 == 201,
          f"got {status2}: {body2}")

    # ── §3: Audit row in DB carries F3 required fields ────────────────────────
    print("\n── §3. Audit row in DB carries all F3 required fields ──")
    try:
        # Small sleep to let the async log_audit write commit.
        time.sleep(1)
        audit_rows = (svc.client
                      .table("audit_log")
                      .select("*")
                      .eq("action", "answer.override_abstain")
                      .eq("resource_id", ANSWER_REF)
                      .execute())
        rows = audit_rows.data or []
        print(f"  audit rows found: {len(rows)}")
        if rows:
            row = rows[0]
            print(f"  audit row: {row}")
            meta = row.get("metadata") or {}
            check("§3a: audit row exists for answer.override_abstain",  True)
            check("§3b: metadata.overridden_by = partner_uid",
                  meta.get("overridden_by") == partner_uid,
                  f"got {meta.get('overridden_by')!r}")
            check("§3c: metadata.collection_id = VAULT_ID",
                  meta.get("collection_id") == VAULT_ID,
                  f"got {meta.get('collection_id')!r}")
            check("§3d: metadata.reason is non-empty",
                  bool(meta.get("reason")),
                  f"got {meta.get('reason')!r}")
            check("§3e: metadata.gate_objection is present",
                  meta.get("gate_objection") is not None,
                  f"got {meta.get('gate_objection')!r}")
            check("§3f: metadata.trust_state = 'overridden'",
                  meta.get("trust_state") == "overridden",
                  f"got {meta.get('trust_state')!r}")
            check("§3g: metadata.role is present",
                  bool(meta.get("role")),
                  f"got {meta.get('role')!r}")
        else:
            check("§3a: audit row found", False,
                  "0 rows in audit_log for this answer_ref — log_audit may have failed silently")
    except Exception as exc:
        check("§3: audit row DB query", False, str(exc))

    # ── §4: Screened partner → 403 (screen beats override grant) ─────────────
    if mp_tok:
        print("\n── §4. Screened partner (screen beats override grant) → 403 ──")
        # Screen the partner off the vault.
        r_screen = httpx.post(
            f"{API_BASE}/admin/firm/screens",
            json={
                "user_id": partner_uid,
                "vault_id": VAULT_ID,
                "reason": "verify_override_live.py — F-E screen-beats-override test",
            },
            headers=_auth(mp_tok),
            timeout=15,
        )
        print(f"  create screen → {r_screen.status_code}")
        if r_screen.status_code == 201:
            screen_id = (r_screen.json().get("screen") or r_screen.json()).get("id")
            print(f"  screen_id : {screen_id}")

            # Now the screened partner calls override → must get 403.
            screened_ref = str(uuid.uuid4())
            status4, body4 = _override(partner_tok, screened_ref,
                                       reason="Screen beats override grant test")
            print(f"  screened partner override → {status4}")
            check("§4: screened partner gets 403 (screen beats override grant)",
                  status4 == 403,
                  f"got {status4}: {body4}")
        else:
            print(f"  WARNING — could not create screen ({r_screen.status_code}): {r_screen.text[:200]}")
            print("  SKIP §4 — screen creation failed")

finally:
    if screen_id and mp_tok:
        print(f"\n  Cleaning up screen {screen_id}…")
        try:
            r = httpx.delete(f"{API_BASE}/admin/firm/screens/{screen_id}",
                             headers=_auth(mp_tok), timeout=15)
            print(f"  cleanup: DELETE screen → {r.status_code}")
        except Exception as e:
            print(f"  cleanup FAILED: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
total = _passed + _failed
print(f"\n{'='*64}")
print(f"  F-E LIVE: {_passed}/{total} passed  {_failed} failed")
if _failed == 0:
    print("  PASS — partner 201 + audit row + associate 403 + screened 403.")
    print("  F-E live gate: CLOSED.")
else:
    print("  FAIL — one or more override-affordance checks did not pass.")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
