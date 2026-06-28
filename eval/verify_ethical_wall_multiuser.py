"""F-D EXTENDED — ETHICAL WALL multi-user combination matrix (live, $0 DB + HTTP).

Creates screens against every role in Firm A, tests cross-firm isolation, tests
un-screened peers, and verifies lift-restores-access for each role. 12 users,
~30 combinations. All screens are cleaned up in the finally block.

Users (all pre-seeded in Supabase, password = WallTest@123):
  FIRM A  (b4762e37-0506-4cc1-b9c1-d4c516d53bc5)  — Jeel's firm
    U0  jeel15thummar@gmail.com        managing_partner   (the MP who raises screens)
    U1  walltest+sp@docquery.test      senior_partner
    U2  walltest+p@docquery.test       partner
    U3  walltest+sa@docquery.test      senior_associate
    U4  walltest+assoc@docquery.test   associate
    U5  jeel15trading@gmail.com        paralegal
    U6  walltest+asst@docquery.test    assistant
    U7  walltest+client@docquery.test  client
    U8  walltest+guest@docquery.test   guest

  FIRM B  (aaaaaaaa-0000-0000-0000-000000000001)
    U9   walltest+firmb_mp@docquery.test    managing_partner
    U10  walltest+firmb_para@docquery.test  paralegal

  OUTSIDER (no firm)
    U11  walltest+outsider@docquery.test    (no membership)

Vault under test:  8ca20f0a-5dfe-40b8-ab7a-26358a512f7e  ("contracts", Firm A)
Firm B vault:      cccccccc-0000-0000-0000-000000000001  ("WallTest FirmB Vault")

Combinations tested:
  C1   Every Firm-A role screened off the Firm-A vault → 403 on /query/stream
  C2   MP screened (deny-overrides-role headline)
  C3   Un-screened peer in same firm is NOT blocked (no collateral damage)
  C4   Screened user's screen lifted → access restored on very next request (T7)
  C5   Firm-B MP hits the Firm-A vault → not a wall 403 (non-member, different error)
  C6   Firm-A screen does NOT bleed to Firm-B members on a Firm-B vault with same id pattern
  C7   Outsider (no firm) hitting the Firm-A vault → not a wall 403

Run:
    python -u eval/verify_ethical_wall_multiuser.py

Needs the API running on localhost:8000. No extra env vars required — all credentials
are baked in below (test accounts only, seeded via migration wall_test_users_firm_b).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    print("  SKIP — httpx not installed (pip install httpx).")
    sys.exit(0)

# ── config ────────────────────────────────────────────────────────────────────
import os
API_BASE   = os.getenv("API_BASE", "http://localhost:8000/api/v1").rstrip("/")
VAULT_A    = "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e"   # "contracts" — Firm A
VAULT_B    = "cccccccc-0000-0000-0000-000000000001"   # WallTest FirmB Vault
FIRM_A     = "b4762e37-0506-4cc1-b9c1-d4c516d53bc5"
FIRM_B     = "aaaaaaaa-0000-0000-0000-000000000001"
WALL_PASS  = "WallTest@123"
MP_PASS    = "jeel@!samosa3011"

# User table: (label, email, password, role, firm)
USERS = {
    "MP":        ("jeel15thummar@gmail.com",              MP_PASS,   "managing_partner", FIRM_A),
    "SP":        ("jeel15thummar+wallsp@gmail.com",       WALL_PASS, "senior_partner",   FIRM_A),
    "P":         ("jeel15thummar+wallp@gmail.com",        WALL_PASS, "partner",          FIRM_A),
    "SA":        ("jeel15thummar+wallsa@gmail.com",       WALL_PASS, "senior_associate", FIRM_A),
    "ASSOC":     ("jeel15thummar+wallassoc@gmail.com",    WALL_PASS, "associate",        FIRM_A),
    "PARA":      ("jeel15trading@gmail.com",              MP_PASS,   "paralegal",        FIRM_A),
    "ASST":      ("jeel15thummar+wallasst@gmail.com",     WALL_PASS, "assistant",        FIRM_A),
    "CLIENT":    ("jeel15thummar+wallclient@gmail.com",   WALL_PASS, "client",           FIRM_A),
    "GUEST":     ("jeel15thummar+wallguest@gmail.com",    WALL_PASS, "guest",            FIRM_A),
    "FIRMB_MP":  ("jeel15thummar+wallfbmp@gmail.com",     WALL_PASS, "managing_partner", FIRM_B),
    "FIRMB_PARA":("jeel15thummar+wallfbpara@gmail.com",   WALL_PASS, "paralegal",        FIRM_B),
    "OUTSIDER":  ("jeel15thummar+wallout@gmail.com",      WALL_PASS, None,               None),
}

# GoTrue UUIDs (from admin API creation — used for DB-layer checks and screen creation)
KNOWN_UIDS = {
    "MP":        "74f022f1-69d8-4f2b-a60c-6b08b79a80ea",
    "SP":        "67c4d43c-9e51-47eb-8955-e75161398254",
    "P":         "bde3aa4b-a1b7-4313-97e2-758ab059cac2",
    "SA":        "ee2fd51e-621f-4ca2-be78-c8f84c0671dd",
    "ASSOC":     "78618ec7-5e6c-4d2e-9c7c-8709b298d993",
    "PARA":      "d6f051fa-522e-4ce2-8e89-5dfbff0ecca6",
    "ASST":      "a78c3a2b-289c-4852-b8db-466800e2489c",
    "CLIENT":    "259e6005-cd2a-472f-a951-1a5516527e8a",
    "GUEST":     "4627f7ea-01a2-4a36-bd6d-488464d66ac6",
    "FIRMB_MP":  "fb47db56-a343-4d98-8684-de6069f0f627",
    "FIRMB_PARA":"583ea1ee-b66f-49c3-9386-78f354016e3c",
    "OUTSIDER":  "9bee960c-58a4-4d41-b997-0417c88c8fe7",
}

# ── helpers ───────────────────────────────────────────────────────────────────
_passed = _failed = 0
_created_screens: list[tuple[str, str, str]] = []   # (screen_id, mp_token_or_None, firm_id)


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    sym = "PASS" if cond else "FAIL"
    if cond:
        _passed += 1
    else:
        _failed += 1
        detail = f"  ← {detail}" if detail else ""
    print(f"  {sym}  {name}{detail}")


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


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _query(token: str, vault_id: str = VAULT_A) -> int:
    """POST /query/stream and return HTTP status. Retries once on connection error."""
    for attempt in range(2):
        try:
            r = httpx.post(f"{API_BASE}/query/stream",
                           json={"question": "what is the governing law?",
                                 "collection_id": vault_id},
                           headers=_auth(token), timeout=25)
            return r.status_code
        except Exception:
            if attempt == 0:
                time.sleep(1)
    return -1


def _create_screen(mp_token: str, user_id: str, vault_id: str, label: str) -> str | None:
    """Create a screen via API, return screen_id or None on failure."""
    r = httpx.post(f"{API_BASE}/admin/firm/screens",
                   json={"user_id": user_id,
                         "vault_id": vault_id,
                         "reason": f"multiuser-wall-test C1 — {label}"},
                   headers=_auth(mp_token), timeout=15)
    if r.status_code == 201:
        d = r.json()
        sid = (d.get("screen") or d).get("id")
        if sid:
            _created_screens.append((sid, mp_token, FIRM_A))
        return sid
    print(f"    ⚠ screen create failed ({r.status_code}): {r.text[:120]}")
    return None


def _remove_screen(screen_id: str, mp_token: str) -> bool:
    r = httpx.delete(f"{API_BASE}/admin/firm/screens/{screen_id}",
                     headers=_auth(mp_token), timeout=15)
    return r.status_code == 200


# ── sign in all users upfront ─────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  F-D EXTENDED — Ethical Wall multi-user combination matrix")
print(f"  API: {API_BASE}   Vault A: {VAULT_A[:8]}…")
print(f"{'='*62}\n")

print("── 0. Signing in all 12 users ──")
tokens: dict[str, str | None] = {}
user_ids: dict[str, str] = {}

from src.components.db import SupabaseManager  # noqa: E402

for label, (email, pw, role, firm) in USERS.items():
    tok = _login(email, pw)
    tokens[label] = tok
    user_ids[label] = KNOWN_UIDS.get(label)
    status = "OK  " if tok else "FAIL"
    note = "" if tok else "  ← login failed"
    print(f"  {status} {label:12} {role or 'no-firm':20} {email}{note}")

mp_token = tokens["MP"]
if not mp_token:
    print("\nFATAL — MP login failed; cannot run any test.")
    sys.exit(1)

# ── use service-role SB for DB checks ────────────────────────────────────────
svc = SupabaseManager(use_service_role=True)

try:
    # ══════════════════════════════════════════════════════════════════════════
    # C1  — EVERY FIRM-A ROLE screened off Vault A → 403
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C1. Every Firm-A role screened → 403 on /query/stream ──")
    # Roles that can actually log in and have the `ask` cap (or any cap that reaches the wall check)
    # client/guest have ask or no verbs — wall is checked BEFORE cap for P3 path
    screened_roles = ["SP", "P", "SA", "ASSOC", "PARA", "ASST", "CLIENT", "GUEST"]
    c1_screens: dict[str, str] = {}

    for label in screened_roles:
        uid = user_ids.get(label)
        tok = tokens.get(label)
        if not uid or not tok:
            print(f"  SKIP {label} — no token/uid")
            continue
        sid = _create_screen(mp_token, uid, VAULT_A, label)
        if sid:
            c1_screens[label] = sid
            status = _query(tok, VAULT_A)
            check(f"C1: {label:12} screened → 403", status == 403, f"got {status}")
        else:
            check(f"C1: {label:12} screen created", False, "create failed")

    # ══════════════════════════════════════════════════════════════════════════
    # C2  — MP screened (deny-overrides-role headline — the hardest case)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C2. MP screened (deny-overrides-role) ──")
    # Need a second MP to raise the wall on the first MP. Promote SP temporarily via DB.
    # Instead: raise the screen directly via service-role DB (simulating a second MP action).
    mp_uid = user_ids["MP"]
    mp_sid = None
    if mp_uid:
        row = svc.create_screen(FIRM_A, mp_uid, VAULT_A,
                                "multiuser-wall-test C2 — MP screened (deny-overrides-role)",
                                created_by=mp_uid)
        mp_sid = row.get("id")
        if mp_sid:
            _created_screens.append((mp_sid, None, FIRM_A))
        # Now the MP's own token should be 403 on Vault A
        status = _query(mp_token, VAULT_A)
        check("C2: MP screened → 403 on their own vault (deny-overrides-role)", status == 403, f"got {status}")
        # Remove immediately (MP needs their token working for C3 screen admin)
        if mp_sid:
            removed = svc.remove_screen(mp_sid, FIRM_A)
            check("C2: MP screen lifted (svc remove)", bool(removed.get("removed_at")))
            _created_screens = [(s, t, f) for s, t, f in _created_screens if s != mp_sid]
            mp_sid = None
        # Confirm MP is unblocked again
        status_after = _query(mp_token, VAULT_A)
        check("C2: MP access restored after screen lifted", status_after != 403, f"got {status_after}")

    # ══════════════════════════════════════════════════════════════════════════
    # C3  — Un-screened peer in same firm is NOT blocked (no collateral damage)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C3. Un-screened peers in same firm are NOT blocked ──")
    # C1 screens are still active for screened_roles. The MP itself is now un-screened.
    # Verify MP (unscreened) can still query Vault A while all those roles are screened.
    status = _query(mp_token, VAULT_A)
    check("C3: MP (un-screened) still gets 200 while others are walled", status == 200, f"got {status}")
    # Also verify a fresh un-screened user: OUTSIDER won't be a firm member so it's
    # a different kind of non-access. Use FIRMB_MP on Vault A as the cleaner test:
    # they're not a firm-A member so the vault is simply not accessible (not a wall 403).
    if tokens.get("FIRMB_MP"):
        s_fb = _query(tokens["FIRMB_MP"], VAULT_A)
        check("C3: Firm-B MP hitting Firm-A vault is NOT a wall-403 (non-member path)",
              s_fb != 403, f"got {s_fb}")

    # ══════════════════════════════════════════════════════════════════════════
    # C4  — Screen lifted → access restored on very next request (T7)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C4. Screen lifted → access restored (T7, per-request resolution) ──")
    # Lift all C1 screens one by one and confirm each role gets un-403'd.
    for label, sid in c1_screens.items():
        tok = tokens.get(label)
        if not tok:
            continue
        ok = _remove_screen(sid, mp_token)
        check(f"C4: {label:12} screen removed (200)", ok)
        if ok:
            _created_screens = [(s, t, f) for s, t, f in _created_screens if s != sid]
            status = _query(tok, VAULT_A)
            # After lift: 200 expected for roles with `ask` cap; client/guest may get
            # 403 from require_cap (not the wall), which is correct — wall is open.
            # We only assert NOT a wall-403 (the wall-403 detail is the ethical wall).
            # We detect the wall block vs cap block by checking DB directly.
            still_screened = svc.is_vault_screened(VAULT_A,
                                                    user_id=user_ids[label],
                                                    firm_id=FIRM_A)
            check(f"C4: {label:12} DB wall gone after lift",
                  not still_screened)
            # HTTP: for roles that have ask cap, expect 200. For client: might be 403
            # from require_cap, not wall. We check the DB proof above; HTTP is bonus.
            if label not in ("GUEST",):
                check(f"C4: {label:12} /query/stream not wall-403 after lift",
                      status != 403, f"got {status}")

    # ══════════════════════════════════════════════════════════════════════════
    # C5  — Firm-B members hitting Firm-A vault (cross-firm non-member)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C5. Firm-B members hitting Firm-A vault ──")
    for label in ("FIRMB_MP", "FIRMB_PARA"):
        tok = tokens.get(label)
        if not tok:
            print(f"  SKIP {label} — no token")
            continue
        status = _query(tok, VAULT_A)
        # Cross-firm non-member: no wall screen exists (not screened — just not a member).
        # accessible_vault_owner returns None → _resolve_collection_filters returns []
        # → /query/stream runs with empty context and returns 200 with no content (no data leaked).
        # The moat is content isolation (no docs returned), NOT a 403 HTTP status.
        # A 403 would only appear if they were an *active member who is screened*.
        wall_active = svc.is_vault_screened(VAULT_A, user_id=user_ids[label], firm_id=FIRM_A)
        check(f"C5: {label:12} has NO ethical wall (non-member, not screened — wall is for members)",
              not wall_active)
        # Verify no documents were returned in the response body (content isolation)
        try:
            r = httpx.post(f"{API_BASE}/query/stream",
                           json={"question": "list all contract parties",
                                 "collection_id": VAULT_A},
                           headers=_auth(tok), timeout=20)
            body_text = r.text
            # The response must not contain actual vault document filenames or chunks.
            # Web search fallback results are OK (no vault data); we check for the
            # vault's own document filenames in the source chips — that would be a real leak.
            # The "contracts" vault belongs to Firm A (user 74f022f1); any chunk from it
            # would carry a filename the Firm-B MP has no right to see.
            # Simplest signal: no "filename" source from the vault owner's documents.
            import re
            # SSE "sources" events carry {"filename":"..."} — check none come from the vault owner.
            vault_owner_id = "74f022f1"  # Jeel's MP, the actual owner of the contracts vault
            has_vault_content = vault_owner_id in body_text
            check(f"C5: {label:12} gets no vault content (content isolation holds, status={status})",
                  not has_vault_content,
                  f"vault content leaked in response: {body_text[:200]}")
        except Exception as e:
            check(f"C5: {label:12} content isolation probe", False, str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # C6  — Firm-A screen does NOT bleed to Firm-B members on Firm-B vault
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C6. Cross-firm screen isolation ──")
    # Create a screen for FIRMB_PARA on Vault A IN FIRM A (simulate a rogue MP trying
    # to wall a foreign user). The DB should reject because user_in_firm check fails —
    # but even if it somehow inserts, the screen is firm-scoped and should not affect
    # FIRMB_PARA's access to Vault B (their own firm's vault).
    fb_para_uid = user_ids.get("FIRMB_PARA")
    if fb_para_uid and tokens.get("FIRMB_PARA"):
        # Attempt cross-firm screen via service-role (bypassing the route's T2 guard)
        # to prove the DB-layer firm_id scoping prevents bleed.
        try:
            cross_screen = svc.create_screen(FIRM_A, fb_para_uid, VAULT_A,
                                             "multiuser-wall-test C6 cross-firm screen",
                                             created_by=user_ids["MP"])
            cross_sid = cross_screen.get("id")
            if cross_sid:
                _created_screens.append((cross_sid, None, FIRM_A))
            # Check: the cross-firm screen exists in Firm A
            in_firm_a = svc.is_vault_screened(VAULT_A, user_id=fb_para_uid, firm_id=FIRM_A)
            check("C6: cross-firm screen visible in Firm A (inserted via svc-role)",
                  in_firm_a)
            # But does it bleed into Firm B? screened_vault_ids with firm_id=FIRM_B must be empty.
            fb_screens = svc.screened_vault_ids(user_id=fb_para_uid, firm_id=FIRM_B)
            check("C6: Firm-B member's screened_vault_ids(firm_B) is EMPTY — no bleed",
                  VAULT_A not in fb_screens, f"bled: {fb_screens}")
            # And they can still query Vault B normally
            status_vb = _query(tokens["FIRMB_PARA"], VAULT_B)
            check(f"C6: FIRMB_PARA can still query their own Vault-B (no bleed, status={status_vb})",
                  status_vb != 403, f"got {status_vb}")
            # Cleanup
            if cross_sid:
                svc.remove_screen(cross_sid, FIRM_A)
                _created_screens = [(s, t, f) for s, t, f in _created_screens if s != cross_sid]
        except Exception as e:
            print(f"  NOTE: cross-firm screen insertion raised: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # C7  — Outsider (no firm) hitting Firm-A vault
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── C7. Outsider (no firm membership) hitting Firm-A vault ──")
    if tokens.get("OUTSIDER"):
        status = _query(tokens["OUTSIDER"], VAULT_A)
        wall_active = svc.is_vault_screened(VAULT_A, user_id=user_ids["OUTSIDER"])
        check("C7: outsider has NO wall screen (correct — not even a firm member)",
              not wall_active)
        check(f"C7: outsider cannot access Firm-A vault (no-firm block, status={status})",
              status != 200, f"got {status}")
    else:
        print("  SKIP outsider — login failed (no firm → may not be able to log in)")

finally:
    # ── Cleanup: remove any leftover screens ──────────────────────────────────
    if _created_screens:
        print(f"\n── Cleanup: removing {len(_created_screens)} leftover screen(s) ──")
        for sid, tok, firm in _created_screens:
            if tok:
                ok = _remove_screen(sid, tok)
            else:
                row = svc.remove_screen(sid, firm)
                ok = bool(row.get("removed_at"))
            print(f"  {'OK' if ok else 'FAILED'} removed screen {sid}")

# ── tally ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  verify_ethical_wall_multiuser: {_passed} passed, {_failed} failed")
print(f"{'='*62}")
sys.exit(0 if _failed == 0 else 1)
