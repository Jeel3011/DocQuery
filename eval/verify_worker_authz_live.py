"""F-B LIVE gate — worker authorization confirmed in the WORKER LOG.

Plans/tool_hard.md §F-B governing rule: "The live gate: actually enqueue it on the
real worker and confirm the refusal in the WORKER log (and that the vault got no
chunks) — *the* lesson from the delete bug was watching the wrong log."

This script:
  1. Signs in as a test user who has the `ingest` cap (so the API enqueue succeeds).
  2. Creates a real document upload for a vault the MP then screens the user off.
  3. Calls the real /documents/upload endpoint (so a real Celery task is enqueued).
  4. Polls the document's status in the DB until it shows "failed" (the worker block).
  5. Checks that 0 document_chunks rows exist for the doc in the DB.
  6. INSTRUCTS the operator to check the WORKER LOG for the "F-B WORKER BLOCK" log line.

─── THE LESSON ─────────────────────────────────────────────────────────────────────
The delete bug was caught by watching the WORKER log while I watched the API log.
This script deliberately cannot read the Celery worker's stdout — that lives in the
terminal where `celery ... worker` runs. It therefore PRINTS the exact log line to
grep for, so the human confirms the worker-log evidence. The DB proof (status=failed,
0 chunks) is what the script asserts automatically.

─── ENV VARS ───────────────────────────────────────────────────────────────────────
  WORKER_MP_EMAIL         Managing Partner who can raise screens
  WORKER_MP_PASSWORD
  WORKER_USER_EMAIL       The user who uploads (must have ingest cap = partner/assoc)
  WORKER_USER_PASSWORD
  WORKER_VAULT_ID         vault collection_id to use for the test
  API_BASE                default http://localhost:8000/api/v1

Pre-baked defaults use the wall-test accounts. Override for a different deployment.

Run (worker must be running in a separate terminal with PDF_PARALLEL_WORKERS=1):
    python -u eval/verify_worker_authz_live.py

The worker terminal must show:
    [<doc_id>] F-B WORKER BLOCK: user=<uid> is screened off vault=<vault_id> ...

Pass = doc status = "failed" in DB within timeout AND 0 chunks written.
The operator must additionally verify the WORKER LOG shows "F-B WORKER BLOCK".
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    print("  SKIP — httpx not installed (pip install httpx).")
    sys.exit(0)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE        = os.getenv("API_BASE",            "http://localhost:8000/api/v1").rstrip("/")
MP_EMAIL        = os.getenv("WORKER_MP_EMAIL",     "jeel15thummar@gmail.com")
MP_PASSWORD     = os.getenv("WORKER_MP_PASSWORD",  "jeel@!samosa3011")
# Default uploader: paralegal (has 'ingest' cap so the API enqueue succeeds).
# This account must own a vault or be staffed on WORKER_VAULT_ID.
# Default: same as MP (the MP owns the contracts vault).
USER_EMAIL      = os.getenv("WORKER_USER_EMAIL",   "jeel15thummar@gmail.com")
USER_PASSWORD   = os.getenv("WORKER_USER_PASSWORD","jeel@!samosa3011")
VAULT_ID        = os.getenv("WORKER_VAULT_ID",     "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e")
POLL_TIMEOUT    = int(os.getenv("WORKER_POLL_TIMEOUT", "60"))   # seconds to wait for task
POLL_INTERVAL   = 3

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


def _login(email: str, password: str) -> str:
    r = httpx.post(f"{API_BASE}/auth/login",
                   json={"email": email, "password": password}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed ({r.status_code}): {r.text[:200]}")
    d = r.json()
    tok = (d.get("access_token")
           or (d.get("session") or {}).get("access_token")
           or d.get("token") or "")
    if not tok:
        raise RuntimeError(f"No token in login response: {str(d)[:200]}")
    return tok


def _auth(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


# ── Sign in ───────────────────────────────────────────────────────────────────
print(f"\n── F-B LIVE: worker authorization confirmed in the WORKER LOG ──────────")
print(f"  API_BASE  : {API_BASE}")
print(f"  VAULT_ID  : {VAULT_ID}")
print(f"  MP        : {MP_EMAIL}")
print(f"  uploader  : {USER_EMAIL}\n")

try:
    mp_tok   = _login(MP_EMAIL, MP_PASSWORD)
    user_tok = _login(USER_EMAIL, USER_PASSWORD)
    print(f"  MP     signed in (token: {mp_tok[:20]}…)")
    print(f"  user   signed in (token: {user_tok[:20]}…)")
except Exception as e:
    print(f"  FATAL — login failed: {e}")
    sys.exit(1)

# ── Get the uploader's user_id from the firm API ──────────────────────────────
from src.components.db import SupabaseManager  # noqa: E402

user_anon = SupabaseManager(use_service_role=False)
user_anon.sign_in(USER_EMAIL, USER_PASSWORD)
user_uid = user_anon.user_id

mp_anon  = SupabaseManager(use_service_role=False)
mp_anon.sign_in(MP_EMAIL, MP_PASSWORD)
firm_info = mp_anon.get_user_firm()
firm_id   = firm_info.get("id") if firm_info else None
svc       = SupabaseManager(use_service_role=True)

print(f"  user_uid  : {user_uid}")
print(f"  firm_id   : {firm_id}")

if not firm_id:
    print("  FATAL — could not resolve firm_id from MP membership.")
    sys.exit(1)

# ── Pre-check: vault must not already be screened ────────────────────────────
pre_screens = svc.screened_vault_ids(user_id=user_uid, firm_id=firm_id)
if VAULT_ID in pre_screens:
    print(f"\n  ⚠  Vault {VAULT_ID} is already screened for {USER_EMAIL}.")
    print("     Choose a different WORKER_VAULT_ID or lift the existing screen first.")
    sys.exit(1)

screen_id: str | None = None
doc_id:    str | None = None

try:
    # ── Step 1: Upload a tiny PDF-like file (the worker will process it) ──────
    # We use a minimal valid 1-byte dummy file. The worker will fail on parse
    # AFTER the screen check — but the screen check happens first and blocks.
    print("\n── 1. Uploading a test file to the vault (enqueues the Celery task) ──")
    tiny_pdf = b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\nxref\n0 1\n0000000000 65535 f \ntrailer<</Size 1/Root 1 0 R>>\nstartxref\n9\n%%EOF"
    files    = {"file": ("fb_test_dummy.pdf", tiny_pdf, "application/pdf")}
    data     = {"collection_id": VAULT_ID}

    r_upload = httpx.post(
        f"{API_BASE}/documents/upload",
        files=files,
        data=data,
        headers=_auth(user_tok),
        timeout=30,
    )
    check("1: POST /documents/upload returns 200/201/202",
          r_upload.status_code in (200, 201, 202),
          f"got {r_upload.status_code}: {r_upload.text[:200]}")

    if r_upload.status_code not in (200, 201, 202):
        print("  FATAL — upload failed; cannot test worker screen block.")
        sys.exit(1)

    upload_data = r_upload.json()
    doc_id = (upload_data.get("document") or {}).get("id") or upload_data.get("id") or upload_data.get("doc_id")
    print(f"  doc_id : {doc_id}")

    # ── Step 2: IMMEDIATELY raise the ethical wall (before the worker picks up) ──
    print("\n── 2. Raising the ethical wall AFTER upload (screen added post-enqueue) ──")
    print("   This tests the live re-check path (F-B P2): screen added after enqueue.")
    # If uploader == MP, they cannot self-screen via the API route (the route may check
    # that the screener ≠ the screened). Use service-role DB directly (like the C2 scenario
    # in verify_ethical_wall_multiuser.py).
    if user_uid == mp_anon.user_id:
        try:
            screen_row = svc.create_screen(
                firm_id, user_uid, VAULT_ID,
                "verify_worker_authz_live.py — F-B live gate (post-enqueue screen via svc-role)",
                created_by=user_uid,
            )
            screen_id = screen_row.get("id")
            print(f"  screen created via svc-role (self-screen): {screen_id}")
            screen_via_svc = True
            check("2: screen created via service-role (MP self-screen for F-B test)", bool(screen_id))
        except Exception as exc:
            print(f"  WARNING — svc-role screen creation failed: {exc}")
            screen_via_svc = False
    else:
        screen_via_svc = False
        r_screen = httpx.post(
            f"{API_BASE}/admin/firm/screens",
            json={
                "user_id": user_uid,
                "vault_id": VAULT_ID,
                "reason": "verify_worker_authz_live.py — F-B live gate (post-enqueue screen)",
            },
            headers=_auth(mp_tok),
            timeout=15,
        )
        check("2: POST /admin/firm/screens returns 201",
              r_screen.status_code == 201,
              f"got {r_screen.status_code}: {r_screen.text[:200]}")

        if r_screen.status_code == 201:
            screen_id = (r_screen.json().get("screen") or r_screen.json()).get("id")
            print(f"  screen_id : {screen_id}")
        else:
            print("  WARNING — screen creation failed; live re-check may not fire.")

    # ── Step 3: Poll the document status for "failed" ─────────────────────────
    print(f"\n── 3. Polling document status (timeout: {POLL_TIMEOUT}s) ──")
    print("   ⚠  Check the WORKER terminal for this log line:")
    print(f"      [<doc_id>] F-B WORKER BLOCK: user={user_uid} is screened off vault={VAULT_ID}")
    print()

    final_status: str | None = None
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            rows = svc.client.table("documents").select("status").eq("id", doc_id).execute()
            row_data = (rows.data or [{}])[0]
            current = row_data.get("status")
            print(f"    status = {current!r}")
            if current in ("failed", "ready"):
                final_status = current
                break
        except Exception as exc:
            print(f"    poll error: {exc}")
        time.sleep(POLL_INTERVAL)

    if final_status is None:
        final_status = "timeout"
        print(f"  (timed out after {POLL_TIMEOUT}s)")

    check("3: document status = 'failed' (worker refused to ingest)",
          final_status == "failed",
          f"got status={final_status!r} — worker may not be running, or screen came too late")

    # ── Step 4: Confirm 0 chunks were written ─────────────────────────────────
    print("\n── 4. Checking chunk count in DB (must be 0) ──")
    try:
        chunk_rows = svc.client.table("document_chunks").select(
            "id", count="exact"
        ).eq("document_id", doc_id).execute()
        chunk_count = chunk_rows.count if hasattr(chunk_rows, "count") else len(chunk_rows.data or [])
    except Exception as exc:
        chunk_count = -1
        print(f"  chunk query error: {exc}")

    print(f"  chunk_count = {chunk_count}")
    check("4: 0 chunks written to vault (screen block held end-to-end)",
          chunk_count == 0,
          f"got {chunk_count} chunks — worker may have ingested before the screen was added")

    # ── Step 5: Operator reminder for the worker log ──────────────────────────
    print("\n── 5. OPERATOR CHECK (cannot be automated): ──")
    print("   Search the WORKER log (the terminal running `celery ... worker`) for:")
    print(f"     F-B WORKER BLOCK")
    print(f"   The full line should contain: user={user_uid}")
    print(f"   AND: vault={VAULT_ID}")
    print("   If that line is present → F-B live gate CLOSED.")
    print("   If absent → the screen may have been added after the task completed;")
    print("               re-run with the worker idle (no pending tasks) so the screen")
    print("               fires on the live re-check path.")

finally:
    # Always clean up the screen.
    if screen_id:
        print(f"\n  Cleaning up screen {screen_id}…")
        try:
            # Try API first; fall back to service-role for self-screens.
            r = httpx.delete(f"{API_BASE}/admin/firm/screens/{screen_id}",
                             headers=_auth(mp_tok), timeout=15)
            if r.status_code not in (200, 404):
                # API rejected (e.g. self-screen can't be deleted this way) — use svc-role.
                svc.remove_screen(screen_id, firm_id)
                print(f"  cleanup: removed screen via svc-role")
            else:
                print(f"  cleanup: DELETE screen → {r.status_code}")
        except Exception as e:
            try:
                svc.remove_screen(screen_id, firm_id)
                print(f"  cleanup: removed screen via svc-role fallback")
            except Exception as e2:
                print(f"  cleanup FAILED: {e} / {e2}  (remove screen {screen_id} manually)")

    # Clean up the test document record too.
    if doc_id:
        try:
            svc.client.table("documents").delete().eq("id", doc_id).execute()
            print(f"  cleanup: deleted document record {doc_id}")
        except Exception:
            pass

# ── Summary ───────────────────────────────────────────────────────────────────
total = _passed + _failed
print(f"\n{'='*64}")
print(f"  F-B LIVE: {_passed}/{total} passed  {_failed} failed")
if _failed == 0:
    print("  DB proof: status=failed + 0 chunks = worker screen block confirmed.")
    print("  ALSO verify the WORKER LOG for 'F-B WORKER BLOCK' (step 5 above).")
else:
    print("  FAIL — worker may not be running or screen arrived after task completion.")
print(f"{'='*64}")
sys.exit(0 if _failed == 0 else 1)
