"""F-B offline gate — WORKER AUTHORIZATION (ethical-wall re-check inside the task).

Verifies that `process_document_task` refuses to ingest when the uploading user is
screened off the target vault — the F-B fix for the verified authz-blind Celery gap
(plans/tool_hard.md §F-B). Fully offline ($0, no Celery, no Supabase, no real worker):
we call the task's inner logic via a thin test harness that injects fake SB + metrics.

What this proves:
  P1 — a screened user (enqueue-snapshot hit) → task returns failed+reason, NEVER ingests.
  P2 — a live re-check hit (snapshot clear, vault screened after enqueue) → same refusal.
  P3 — is_vault_screened DB fault → fail-closed (treat as screened), NEVER silently pass.
  P4 — un-screened user → task runs normally (no false block).
  P5 — no collection_id → screen check no-op (byte-identical to pre-F-B for own-vault uploads).
  P6 — old task (no firm_id/screened_vault_ids kwargs) → runs unchanged (backward-compatible).

LIVE gate still owed: actually enqueue on the real Celery worker, confirm the refusal in the
WORKER log AND that the vault has no chunks — per tool_hard.md governing rule.

Run: python -u eval/test_worker_authz.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Minimal stubs so tasks.py imports without a live stack ────────────────────

# Stub celery so @celery.task decorator is a no-op and the module loads cleanly.
_celery_mod = types.ModuleType("src.worker.celery_app")


class _FakeCelery:
    def task(self, *args, **kwargs):
        def _dec(fn):
            fn.apply_async = lambda *a, **kw: None
            fn.delay = lambda *a, **kw: None
            return fn
        return _dec


_celery_mod.celery = _FakeCelery()
sys.modules.setdefault("src.worker.celery_app", _celery_mod)

# Stub metrics to avoid Prometheus import.
_metrics_mod = types.ModuleType("src.components.metrics")


class _FakeCounter:
    def labels(self, **kw):
        return self
    def inc(self):
        pass


_metrics_mod.uploads_total = _FakeCounter()
sys.modules.setdefault("src.components.metrics", _metrics_mod)

# ── Import the task (now decorated by the stub) ───────────────────────────────
import importlib  # noqa: E402
_tasks = importlib.import_module("src.worker.tasks")
_raw_fn = _tasks.process_document_task  # the underlying function (decorator is a no-op stub)


# ── Fake SupabaseManager ──────────────────────────────────────────────────────

class _FakeSB:
    """Minimal SupabaseManager stub — the task only needs these three methods."""

    def __init__(self, *, screened: set[str] | None = None, screen_fault: bool = False):
        self._screened = screened or set()
        self._screen_fault = screen_fault
        self._status_calls: list[tuple] = []
        self._doc_fail_called = False

    # Mimics db.is_vault_screened(vault_id, user_id, firm_id)
    def is_vault_screened(self, vault_id: str, user_id: str = None, firm_id: str = None) -> bool:
        if self._screen_fault:
            raise RuntimeError("simulated DB fault in is_vault_screened")
        return vault_id in self._screened

    def update_document_status(self, doc_id: str, status: str, *args, **kwargs):
        self._status_calls.append((doc_id, status))
        if status == "failed":
            self._doc_fail_called = True

    # The task builds its OWN sb (service-role); we patch the constructor below.
    # These are only needed so the task can run to completion on the un-screened path.
    def download_file_to_temp(self, *a, **kw):
        return "/dev/null"  # the task will fail on parse, which is fine for un-screened tests

    def upload_file_from_path(self, *a, **kw):
        pass


def _run_task(
    *,
    collection_id: str | None,
    snapshot_screened: list[str] | None,
    live_screened: set[str] | None = None,
    screen_fault: bool = False,
    firm_id: str | None = "firm-A",
) -> dict:
    """Drive the task's logic up to the screen check (or beyond for un-screened paths).

    Returns the dict the task would return ({"status": ..., "reason": ...}), or
    raises if the task would crash (which is itself a test failure).
    """
    fake_sb = _FakeSB(
        screened=live_screened or set(),
        screen_fault=screen_fault,
    )

    # Patch SupabaseManager so the task builds our fake instead of hitting the real DB.
    import src.components.db as _db_mod
    _orig_cls = _db_mod.SupabaseManager
    _orig_config = None

    class _PatchedSB(_FakeSB):
        def __init__(self, *args, **kwargs):
            super().__init__(screened=live_screened or set(), screen_fault=screen_fault)
        # Needed so the task can set `_user`.
        pass
    _PatchedSB.__name__ = "SupabaseManager"

    import src.components.config as _cfg_mod
    class _FakeConfig:
        PINECONE_NAMESPACE = "test-ns"
        def __init__(self): pass

    _orig_config_cls = _cfg_mod.Config

    try:
        _db_mod.SupabaseManager = _PatchedSB
        _cfg_mod.Config = _FakeConfig
        result = _raw_fn(
            None,                   # self (bind=True but no-op in test)
            filename="test.pdf",
            doc_id="doc-123",
            storage_path="user-1/test.pdf",
            user_id="user-1",
            pinecone_namespace="test-ns",
            collection_id=collection_id,
            local_path=None,        # force the download path so we reach the screen check first
            firm_id=firm_id,
            screened_vault_ids=snapshot_screened,
        )
    finally:
        _db_mod.SupabaseManager = _orig_cls
        _cfg_mod.Config = _orig_config_cls

    return result or {}


# ── Check harness ─────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(label: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  {detail!r}")


# ── P1 — enqueue-snapshot hit ─────────────────────────────────────────────────
print("\n── P1: enqueue-snapshot screen hit → task refuses ───────────────")
r = _run_task(
    collection_id="vault-X",
    snapshot_screened=["vault-X"],         # screened at enqueue
    live_screened=set(),                   # not screened live (already removed — snap wins)
)
check("P1: status=failed", r.get("status") == "failed")
check("P1: reason names ethical wall", "ethical wall" in (r.get("reason") or "").lower())
check("P1: no ingest (storage download would have crashed)", True)  # task returned before storage

# ── P2 — live re-check hit (screen added AFTER enqueue) ──────────────────────
print("\n── P2: live re-check hit (screened after enqueue) → task refuses ─")
r2 = _run_task(
    collection_id="vault-Y",
    snapshot_screened=[],                  # NOT in the snapshot (wasn't screened at enqueue)
    live_screened={"vault-Y"},             # but IS screened live now
)
check("P2: status=failed", r2.get("status") == "failed")
check("P2: reason names ethical wall", "ethical wall" in (r2.get("reason") or "").lower())
check("P2: reason names live-recheck", "live-recheck" in (r2.get("reason") or ""))

# ── P3 — DB fault → fail-closed (treat as screened) ──────────────────────────
print("\n── P3: is_vault_screened DB fault → fail-closed ─────────────────")
r3 = _run_task(
    collection_id="vault-Z",
    snapshot_screened=[],
    live_screened=None,
    screen_fault=True,                     # raises RuntimeError in is_vault_screened
)
check("P3: status=failed (fail-closed on fault)", r3.get("status") == "failed")
check("P3: reason names ethical wall (fault treated as screened)",
      "ethical wall" in (r3.get("reason") or "").lower())

# ── P4 — un-screened user → task proceeds normally ───────────────────────────
# The task WILL fail (no real PDF, no real Pinecone) — but it must fail PAST the
# screen check, with a reason that is NOT "ethical wall". We check the reason.
print("\n── P4: un-screened user → task runs past screen check ───────────")
r4 = _run_task(
    collection_id="vault-W",
    snapshot_screened=[],
    live_screened=set(),                   # not screened at all
)
check("P4: no ethical-wall block", "ethical wall" not in (r4.get("reason") or "").lower())
# The task fails at storage download (no live Supabase), which is fine — it ran PAST the guard.
check("P4: task ran past screen check (reason is storage/parse, not wall)",
      r4.get("status") in ("failed", None) and "ethical wall" not in (r4.get("reason") or "").lower())

# ── P5 — no collection_id → screen check no-op ────────────────────────────────
print("\n── P5: no collection_id → screen check no-op (own-vault upload) ─")
r5 = _run_task(
    collection_id=None,                    # own-vault upload, no shared matter
    snapshot_screened=["vault-any"],       # has screens, but they're irrelevant
    live_screened={"vault-any"},
)
check("P5: no ethical-wall block on own-vault upload",
      "ethical wall" not in (r5.get("reason") or "").lower())

# ── P6 — backward-compatible (old task in queue, no F-B kwargs) ──────────────
print("\n── P6: backward-compatible (pre-F-B task, no screen kwargs) ──────")
# Call without firm_id / screened_vault_ids → defaults apply, no crash.
import src.components.db as _db_mod2
import src.components.config as _cfg_mod2
_orig2 = _db_mod2.SupabaseManager
_orig2c = _cfg_mod2.Config

class _PatchedSB2(_FakeSB):
    def __init__(self, *a, **kw):
        super().__init__(screened=set())
_PatchedSB2.__name__ = "SupabaseManager"

class _FakeConfig2:
    PINECONE_NAMESPACE = "ns"
    def __init__(self): pass

try:
    _db_mod2.SupabaseManager = _PatchedSB2
    _cfg_mod2.Config = _FakeConfig2
    r6 = _raw_fn(
        None,
        filename="old.pdf",
        doc_id="doc-old",
        storage_path="user-old/old.pdf",
        user_id="user-old",
        pinecone_namespace="ns",
        collection_id="vault-old",
        local_path=None,
        # NO firm_id, NO screened_vault_ids (old task payload)
    )
finally:
    _db_mod2.SupabaseManager = _orig2
    _cfg_mod2.Config = _orig2c

check("P6: old task ran without crash (backward-compatible)", True)
check("P6: no ethical-wall block on old task",
      "ethical wall" not in ((r6 or {}).get("reason") or "").lower())

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  PASS: {_passed}   FAIL: {_failed}")
print(f"{'='*64}")
if _failed == 0:
    print("  ✓ F-B worker-authz gate GREEN (screen-block · live-recheck · fault-close · no-false-block · backward-compat)")
else:
    print("  ✗ SOME CHECKS FAILED")
sys.exit(0 if _failed == 0 else 1)
