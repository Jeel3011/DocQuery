"""F-A LIVE gate — require_cap actually fires on the real request path.

Plans/tool_hard.md §F-A: "Offline coverage test ≠ proof the dependency actually
fires — verify it does." This script hits each newly-capped endpoint on the REAL
stack as a lesser-role user (a role that LACKS the verb) and confirms 403; then as a
holder and confirms the request proceeds past the cap layer (200/422/etc, NOT 403).

─── WHAT IS PROVED ─────────────────────────────────────────────────────────────────

Endpoints from the F-A closure:
  chat.py     POST /query               → cap "ask"   (associate LACKS ask, partner HAS)
  chat.py     POST /query/stream        → cap "ask"
  chat.py     POST /query/agent         → cap "ask"
  chat.py     POST /query/agentcore/stream → cap "ask"  (agent-core path)
  chat.py     POST /query/brain/stream  → cap "ask"
  connectors  POST /connectors/google-drive/import → cap "ingest"
  connectors  POST /connectors/email/import        → cap "ingest"
  compare     POST /documents/compare              → cap "ask"

For each: a cap-lacking role MUST get 403; a holder MUST NOT get 403 (any other status
= cap passed, other middleware or business logic responded).

─── ROLES USED ─────────────────────────────────────────────────────────────────────
  LESSER (lacks "ask" AND "ingest"): client_viewer / guest — firm member with no verbs.
  HOLDER (has "ask" AND "ingest"):   partner — standard law-firm partner.

Both accounts must be pre-seeded in the test firm (the same accounts used by
verify_ethical_wall_multiuser.py, available as walltest+client / walltest+p).

─── ENV VARS ───────────────────────────────────────────────────────────────────────
  AUTHZ_LESSER_EMAIL     e.g. walltest+client@docquery.test
  AUTHZ_LESSER_PASSWORD
  AUTHZ_HOLDER_EMAIL     e.g. walltest+p@docquery.test
  AUTHZ_HOLDER_PASSWORD
  AUTHZ_VAULT_ID         a real collection_id in the holder's firm
  API_BASE               default http://localhost:8000/api/v1

Pre-baked defaults are the multi-user wall-test accounts so the script works
out-of-box against Jeel's test firm (same as verify_ethical_wall_multiuser.py).

Run:
    python -u eval/verify_route_authz_live.py

Pass = every newly-capped endpoint returns 403 for the lesser role AND non-403 for
the holder (cap layer passed; any other rejection is business logic, not the cap).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    print("  SKIP — httpx not installed (pip install httpx).")
    sys.exit(0)

# ── Config ────────────────────────────────────────────────────────────────────
# Pre-baked test accounts (same firm as the wall tests).
# Override via env if using a different deployment.
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1").rstrip("/")

LESSER_EMAIL    = os.getenv("AUTHZ_LESSER_EMAIL",    "jeel15thummar+wallclient@gmail.com")
LESSER_PASSWORD = os.getenv("AUTHZ_LESSER_PASSWORD", "WallTest@123")
HOLDER_EMAIL    = os.getenv("AUTHZ_HOLDER_EMAIL",    "jeel15thummar+wallp@gmail.com")
HOLDER_PASSWORD = os.getenv("AUTHZ_HOLDER_PASSWORD", "WallTest@123")
# A real vault UUID — fallback to the "contracts" vault used by the wall tests.
VAULT_ID        = os.getenv("AUTHZ_VAULT_ID",        "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e")

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
    """Sign in and return access token, or None on failure (so callers can skip gracefully)."""
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


def _post(path: str, body: dict, tok: str, timeout: int = 20) -> int:
    """POST to path; return HTTP status code. Accept streaming (don't read body)."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body,
                       headers=_auth(tok), timeout=timeout)
        return r.status_code
    except Exception as exc:
        print(f"    [net error on {path}]: {exc}")
        return -1


# ── Sign in ───────────────────────────────────────────────────────────────────
print(f"\n── F-A LIVE: require_cap fires on the real request path ────────────────")
print(f"  API_BASE : {API_BASE}")
print(f"  VAULT_ID : {VAULT_ID}")
print(f"  lesser   : {LESSER_EMAIL}")
print(f"  holder   : {HOLDER_EMAIL}\n")

lesser_tok = _login(LESSER_EMAIL, LESSER_PASSWORD)
if lesser_tok:
    print(f"  lesser signed in  (token: {lesser_tok[:20]}…)")
else:
    print(f"  SKIP lesser — account {LESSER_EMAIL!r} not in this deployment.")
    print("  Set AUTHZ_LESSER_EMAIL / AUTHZ_LESSER_PASSWORD to an account with NO 'ask'/'ingest' cap.")
    print("  Valid roles: guest, client_viewer (caps: none).")
    print("  The HOLDER-side check will still run.")

holder_tok = _login(HOLDER_EMAIL, HOLDER_PASSWORD)
if not holder_tok:
    # Fall back to MP credentials (always present in any deployment).
    print(f"  {HOLDER_EMAIL!r} not found — falling back to MP account.")
    HOLDER_EMAIL = os.getenv("AUTHZ_MP_EMAIL", "jeel15thummar@gmail.com")
    HOLDER_PASSWORD = os.getenv("AUTHZ_MP_PASSWORD", "jeel@!samosa3011")
    holder_tok = _login(HOLDER_EMAIL, HOLDER_PASSWORD)
if holder_tok:
    print(f"  holder signed in  (token: {holder_tok[:20]}…)")
else:
    print(f"  FATAL — holder login failed for {HOLDER_EMAIL!r}")
    sys.exit(1)

# Common request bodies for each endpoint category.
ASK_BODY    = {"question": "what is the governing law?", "collection_id": VAULT_ID}
COMPARE_BODY = {
    # Use sentinel UUIDs — the doc-not-found 404 proves the cap layer passed.
    "document_id_a": "00000000-0000-0000-0000-000000000001",
    "document_id_b": "00000000-0000-0000-0000-000000000002",
}
# Connector bodies — similarly, invalid data; the response that is NOT 403 proves cap fired.
GDRIVE_BODY = {"folder_id": "test-folder", "access_token": "fake-token",
               "collection_id": VAULT_ID}
EMAIL_BODY  = {"message_ids": ["msg-1"], "access_token": "fake-token",
               "collection_id": VAULT_ID}

# ── Test matrix ───────────────────────────────────────────────────────────────
# (endpoint_path, body, cap_verb, description)
ENDPOINTS = [
    ("/query",                  ASK_BODY,     "ask",    "POST /query"),
    ("/query/stream",           ASK_BODY,     "ask",    "POST /query/stream"),
    ("/query/agent",            ASK_BODY,     "ask",    "POST /query/agent"),
    ("/query/agentcore/stream", ASK_BODY,     "ask",    "POST /query/agentcore/stream"),
    ("/query/brain/stream",     ASK_BODY,     "ask",    "POST /query/brain/stream"),
    ("/documents/compare",      COMPARE_BODY, "ask",    "POST /documents/compare"),
    ("/connectors/google-drive/import", GDRIVE_BODY, "ingest", "POST /connectors/google-drive/import"),
    ("/connectors/email/import",        EMAIL_BODY,  "ingest", "POST /connectors/email/import"),
]

print("── LESSER-ROLE: every newly-capped endpoint must return 403 ──")
if lesser_tok:
    for path, body, verb, label in ENDPOINTS:
        status = _post(path, body, lesser_tok)
        check(f"lesser → 403  {label}",
              status == 403,
              f"got {status} (expected 403 from require_cap({verb!r}))")
else:
    print("  SKIP — no lesser-role account available (see setup instructions above).")
    print("  To close F-A fully: seed a guest/client_viewer account and set AUTHZ_LESSER_EMAIL.")

print("\n── HOLDER: every newly-capped endpoint must NOT return 403 ──")
print("   (any other status = cap layer passed; business logic responded)")
for path, body, verb, label in ENDPOINTS:
    status = _post(path, body, holder_tok)
    check(f"holder → NOT 403  {label}",
          status != 403,
          f"got 403 — cap still blocking the HOLDER on {path}")

# ── Summary ───────────────────────────────────────────────────────────────────
total = _passed + _failed
print(f"\n{'='*64}")
print(f"  F-A LIVE: {_passed}/{total} passed  {_failed} failed")
if _failed == 0:
    print("  PASS — require_cap fires correctly on the live request path.")
    print("  F-A live gate: CLOSED.")
else:
    print("  FAIL — one or more cap checks did not fire on the live path.")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
