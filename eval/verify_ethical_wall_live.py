"""F-D LIVE gate — ETHICAL WALL end-to-end on the REAL stack ($0 DB + HTTP probes).

Proves the offline gate (eval/test_ethical_walls.py, P1–P9) holds on the LIVE path:
  - the real SupabaseManager + real `screens` table enforce the wall at the DB layer
  - the real API (agent-core /query/agentcore/stream + brain /query/stream) return 403
    when the requesting user is actively screened off the vault
  - lifting the screen (soft-remove) restores access on the very next request (T7)

The governing rule (tool_hard.md): offline-green ≠ done — every F-item needs a live
verification against the real stack. This IS that verification for F2c.

─── WHAT IT NEEDS ───────────────────────────────────────────────────────────────────
Set these env vars before running (use a test firm with two real accounts):

  WALL_MP_EMAIL       — the Managing Partner who raises/lifts the screen
  WALL_MP_PASSWORD    — their password
  WALL_SCREENED_EMAIL — the user to be screened (must be in the same firm as the MP)
  WALL_SCREENED_PASSWORD
  WALL_VAULT_ID       — a real collection_id (uuid) belonging to the firm
                        (pick one from the Supabase `collections` table)
  API_BASE            — optional; defaults to http://localhost:8000/api/v1

The MP must have the `manage_members` cap (i.e. be a managing_partner or admin role).
The screened user must be a firm member (paralegal / junior / associate etc.).

Both accounts and the vault must already exist — this script does NOT create them.
The screen it creates IS CLEANED UP in the finally block even if an assertion fails,
so the live firm is not left with a stale wall.

Run:
    WALL_MP_EMAIL=mp@firm.test WALL_MP_PASSWORD=... \\
    WALL_SCREENED_EMAIL=para@firm.test WALL_SCREENED_PASSWORD=... \\
    WALL_VAULT_ID=<uuid> \\
    python -u eval/verify_ethical_wall_live.py

─── WHAT IS PROVED ──────────────────────────────────────────────────────────────────
  L1  DB layer — screened_vault_ids returns {VAULT_ID} for the screened user AFTER a
      screen is created.
  L2  DB layer — is_vault_screened returns True for the specific vault.
  L3  HTTP agent-core path (P1) — POST /query/agentcore/stream → 403 for screened user.
  L4  HTTP brain/chat path (P3) — POST /query/stream → 403 for screened user.
  L5  DB layer — screened_vault_ids returns {} AFTER the screen is lifted (T7).
  L6  DB layer — is_vault_screened returns False after the screen is lifted.
  L7  HTTP agent-core path — POST /query/agentcore/stream → NOT 403 after screen lifted
      (the wall is lifted; the route may 200/422/503 for other reasons but must NOT 403
      on the ethical wall — we accept any non-403 response as "wall is open").
  L8  HTTP brain/chat path — POST /query/stream → NOT 403 after screen lifted (same).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import httpx
except ImportError:
    print("  SKIP — httpx not installed (pip install httpx to run live).")
    sys.exit(0)

# ── env ───────────────────────────────────────────────────────────────────────
MP_EMAIL      = os.getenv("WALL_MP_EMAIL", "")
MP_PASSWORD   = os.getenv("WALL_MP_PASSWORD", "")
SC_EMAIL      = os.getenv("WALL_SCREENED_EMAIL", "")
SC_PASSWORD   = os.getenv("WALL_SCREENED_PASSWORD", "")
VAULT_ID      = os.getenv("WALL_VAULT_ID", "")
API_BASE      = os.getenv("API_BASE", "http://localhost:8000/api/v1").rstrip("/")

if not all([MP_EMAIL, MP_PASSWORD, SC_EMAIL, SC_PASSWORD, VAULT_ID]):
    print(
        "\n  SKIP — set WALL_MP_EMAIL / WALL_MP_PASSWORD / WALL_SCREENED_EMAIL /\n"
        "         WALL_SCREENED_PASSWORD / WALL_VAULT_ID to run the live wall proof.\n"
        "\n"
        "  The offline gate (eval/test_ethical_walls.py) models P1–P9 and runs $0.\n"
        "  This script is the live proof that the same wall holds on the real stack.\n"
        "  See the module docstring for setup instructions.\n"
    )
    sys.exit(0)

# ── helpers ───────────────────────────────────────────────────────────────────
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


def _sign_in_http(email: str, password: str) -> str:
    """Sign in via the API and return the access token."""
    r = httpx.post(
        f"{API_BASE}/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Sign-in failed ({r.status_code}): {r.text[:300]}")
    data = r.json()
    token = (
        data.get("access_token")
        or (data.get("session") or {}).get("access_token")
        or data.get("token")
    )
    if not token:
        raise RuntimeError(f"No token in login response: {str(data)[:300]}")
    return token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _probe_agentcore(token: str) -> int:
    """POST to the agent-core stream with the vault; return HTTP status code."""
    r = httpx.post(
        f"{API_BASE}/query/agentcore/stream",
        json={"question": "what is the governing law?", "collection_id": VAULT_ID},
        headers=_auth(token),
        timeout=20,
    )
    return r.status_code


def _probe_brain_stream(token: str) -> int:
    """POST to the brain /query/stream with the vault; return HTTP status code."""
    r = httpx.post(
        f"{API_BASE}/query/stream",
        json={"question": "what is the governing law?", "collection_id": VAULT_ID},
        headers=_auth(token),
        timeout=20,
    )
    return r.status_code


# ── DB layer via SupabaseManager ──────────────────────────────────────────────
from src.components.db import SupabaseManager  # noqa: E402


def _build_anon_sb(email: str, password: str) -> SupabaseManager:
    """Sign in via anon client so self.user_id is set from the session."""
    sb = SupabaseManager(use_service_role=False)
    sb.sign_in(email, password)
    return sb


def _build_service_sb() -> SupabaseManager:
    """Service-role manager for screen DB queries (bypasses RLS so we can read screens table)."""
    return SupabaseManager(use_service_role=True)


# ── main ──────────────────────────────────────────────────────────────────────
print(
    f"\n── F-D: ETHICAL WALL — live path verification ──────────────────────────────\n"
    f"  API_BASE      : {API_BASE}\n"
    f"  VAULT_ID      : {VAULT_ID}\n"
    f"  MP (screener) : {MP_EMAIL}\n"
    f"  Screened user : {SC_EMAIL}\n"
)

screen_id: str | None = None

try:
    # ── 0. Sign in both users ─────────────────────────────────────────────────
    print("── 0. Authenticating both users ──")
    try:
        mp_token = _sign_in_http(MP_EMAIL, MP_PASSWORD)
        print(f"  MP signed in  (token: {mp_token[:20]}…)")
    except Exception as e:
        print(f"  FATAL — MP sign-in failed: {e}")
        sys.exit(1)

    try:
        sc_token = _sign_in_http(SC_EMAIL, SC_PASSWORD)
        print(f"  Screened user signed in (token: {sc_token[:20]}…)")
    except Exception as e:
        print(f"  FATAL — Screened user sign-in failed: {e}")
        sys.exit(1)

    # Sign in both users via anon client to get their user_ids.
    mp_anon = _build_anon_sb(MP_EMAIL, MP_PASSWORD)
    sc_anon = _build_anon_sb(SC_EMAIL, SC_PASSWORD)
    # Service-role manager for screen DB queries (reads the `screens` table without RLS).
    svc_sb  = _build_service_sb()

    mp_firm_info  = mp_anon.get_user_firm()
    sc_user_id    = sc_anon.user_id
    firm_id       = mp_firm_info.get("id") if mp_firm_info else None

    if not firm_id:
        print("  FATAL — could not resolve firm_id from MP membership.")
        sys.exit(1)

    print(f"  firm_id       : {firm_id}")
    print(f"  screened uid  : {sc_user_id}")

    # ── PRE-CHECK: screened_vault_ids should be EMPTY before we create the screen ──
    pre_screens = svc_sb.screened_vault_ids(user_id=sc_user_id, firm_id=firm_id)
    if VAULT_ID in pre_screens:
        print(
            f"\n  ⚠  The vault {VAULT_ID} is ALREADY actively screened for {SC_EMAIL}.\n"
            f"     Run `python -u eval/verify_ethical_wall_live.py` after lifting any\n"
            f"     pre-existing screen, or choose a different WALL_VAULT_ID.\n"
        )
        sys.exit(1)

    # ── BASELINE: both routes should succeed (or at least NOT 403 for the wall) ──
    print("\n── BASELINE (before screen) — both routes should not be wall-403 ──")
    pre_ac = _probe_agentcore(sc_token)
    pre_br = _probe_brain_stream(sc_token)
    print(f"  /query/agentcore/stream → {pre_ac}")
    print(f"  /query/stream           → {pre_br}")
    check(
        "BASELINE: agent-core is not a wall-403 before screen exists",
        pre_ac != 403,
        f"got {pre_ac} — vault may already be screened or the MP firm differs from the vault",
    )
    check(
        "BASELINE: brain stream is not a wall-403 before screen exists",
        pre_br != 403,
        f"got {pre_br}",
    )

    # ── 1. Create the screen via the admin API (as the MP) ────────────────────
    print("\n── 1. Creating the ethical wall via the API (as MP) ──")
    r_create = httpx.post(
        f"{API_BASE}/admin/firm/screens",
        json={
            "user_id": sc_user_id,
            "vault_id": VAULT_ID,
            "reason": "verify_ethical_wall_live.py — automated live gate (F-D)",
        },
        headers=_auth(mp_token),
        timeout=15,
    )
    check(
        "1: POST /admin/firm/screens returns 201",
        r_create.status_code == 201,
        f"got {r_create.status_code}: {r_create.text[:200]}",
    )
    if r_create.status_code != 201:
        print("  FATAL — screen creation failed; aborting (no screen to clean up).")
        sys.exit(1)

    screen_data = r_create.json()
    screen_id = (screen_data.get("screen") or screen_data).get("id")
    print(f"  screen_id: {screen_id}")

    # ── L1/L2: DB layer sees the screen ──────────────────────────────────────
    print("\n── L1/L2. DB layer — screened_vault_ids + is_vault_screened ──")
    # Use service-role manager with explicit user_id (simulates a fresh per-request read).
    live_ids = svc_sb.screened_vault_ids(user_id=sc_user_id, firm_id=firm_id)
    check(
        "L1: screened_vault_ids includes VAULT_ID after screen is created",
        VAULT_ID in live_ids,
        f"returned: {live_ids}",
    )
    live_check = svc_sb.is_vault_screened(VAULT_ID, user_id=sc_user_id, firm_id=firm_id)
    check(
        "L2: is_vault_screened returns True for the screened vault",
        live_check is True,
        f"returned: {live_check}",
    )

    # ── L3/L4: HTTP paths return 403 for the screened user ───────────────────
    print("\n── L3/L4. HTTP paths — 403 on the walled vault (real API) ──")
    ac_status  = _probe_agentcore(sc_token)
    br_status  = _probe_brain_stream(sc_token)
    print(f"  /query/agentcore/stream → {ac_status}  (expected 403)")
    print(f"  /query/stream           → {br_status}  (expected 403)")
    check(
        "L3: agent-core returns 403 for the screened user on the walled vault",
        ac_status == 403,
        f"got {ac_status}",
    )
    check(
        "L4: brain /query/stream returns 403 for the screened user on the walled vault",
        br_status == 403,
        f"got {br_status}",
    )

    # ── 2. Lift the screen (soft-remove, MP only) ─────────────────────────────
    print("\n── 2. Lifting the screen (soft-remove via API) ──")
    r_remove = httpx.delete(
        f"{API_BASE}/admin/firm/screens/{screen_id}",
        headers=_auth(mp_token),
        timeout=15,
    )
    check(
        "2: DELETE /admin/firm/screens/{screen_id} returns 200",
        r_remove.status_code == 200,
        f"got {r_remove.status_code}: {r_remove.text[:200]}",
    )
    # Mark as cleaned up so finally block skips the redundant remove.
    if r_remove.status_code == 200:
        screen_id = None

    # ── L5/L6: DB layer sees the lift ─────────────────────────────────────────
    print("\n── L5/L6. DB layer — screened_vault_ids + is_vault_screened AFTER lift ──")
    # Fresh service-role read — simulates a new per-request DB query (T7).
    post_ids = svc_sb.screened_vault_ids(user_id=sc_user_id, firm_id=firm_id)
    check(
        "L5: screened_vault_ids does NOT include VAULT_ID after screen is lifted (T7)",
        VAULT_ID not in post_ids,
        f"returned: {post_ids}",
    )
    post_check = svc_sb.is_vault_screened(VAULT_ID, user_id=sc_user_id, firm_id=firm_id)
    check(
        "L6: is_vault_screened returns False after screen is lifted",
        post_check is False,
        f"returned: {post_check}",
    )

    # ── L7/L8: HTTP paths accept the vault again ──────────────────────────────
    print("\n── L7/L8. HTTP paths — wall is open after screen lifted (NOT 403) ──")
    ac_post = _probe_agentcore(sc_token)
    br_post = _probe_brain_stream(sc_token)
    print(f"  /query/agentcore/stream → {ac_post}  (must not be 403 on wall)")
    print(f"  /query/stream           → {br_post}  (must not be 403 on wall)")
    # 403 specifically means wall — other codes (422, 404, 5xx) are other issues.
    check(
        "L7: agent-core is no longer wall-403 after screen lifted",
        ac_post != 403,
        f"got {ac_post} — wall may not have been removed correctly",
    )
    check(
        "L8: brain /query/stream is no longer wall-403 after screen lifted",
        br_post != 403,
        f"got {br_post}",
    )

finally:
    # Always clean up — remove the screen if the deletion step failed or was skipped.
    if screen_id:
        print(f"\n  ⚠  Cleaning up screen {screen_id} (deletion step failed or skipped)…")
        try:
            mp_token_cleanup = _sign_in_http(MP_EMAIL, MP_PASSWORD)
            r = httpx.delete(
                f"{API_BASE}/admin/firm/screens/{screen_id}",
                headers=_auth(mp_token_cleanup),
                timeout=15,
            )
            print(f"  cleanup: DELETE screen → {r.status_code}")
        except Exception as e:
            print(f"  cleanup FAILED — remove the screen manually: {e}")
            print(f"  screen_id to remove: {screen_id}  vault: {VAULT_ID}")

# ── tally ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  verify_ethical_wall_live: {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(0 if _failed == 0 else 1)
