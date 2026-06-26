"""Regression gate for the 2026-06-25 ADVERSARIAL AUTH AUDIT fixes (offline, $0).

After a real fail-open was found (a wrong invite token silently created a solo firm + made the user
MP), an adversarial audit swept the whole auth/firm surface and found 7 issues. This gate PINS the
fixes so the same holes can't silently reopen. Each check maps to a numbered audit finding. Run:

    python -u eval/test_authz_audit_fixes.py

Findings pinned:
  #1  signup with a FAILING invite token FAILS CLOSED (400) — never falls through to create a firm.
  #2  get_user_firm picks the active firm DETERMINISTICALLY (ORDER BY created_at) — no role oscillation.
  #3  the ethical-wall lookup fails CLOSED on a LIVE fault (only a missing table degrades to "no wall").
  #5  /admin/eval/* is cap-gated (manage_members) + the caller-supplied questions_path is removed.
  #6  override_abstain asserts the vault is in the caller's firm (collection_in_firm) — no foreign id.
  #7  signup honors an invite only for a CONFIRMED email.

Static + behavioral: where driving a live handler is clean we do; otherwise we assert the SOURCE
contains the guard (the same lockstep-with-source approach the F2f/FE gates use). No API, no network.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name, cond, hint=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({hint})" if hint else ""))


import src.api.routes.auth as auth_mod       # noqa: E402
import src.api.routes.admin as admin_mod      # noqa: E402
import src.components.db as db_mod            # noqa: E402
import src.api.dependencies as deps_mod       # noqa: E402

print("\nAuth-audit regression gate — the 2026-06-25 fail-open fixes\n")

# ── #1 signup fails CLOSED on a bad invite (no fall-through to create-firm) ──
print("#1 signup with a failing invite FAILS CLOSED (no silent solo-firm)")
sig = inspect.getsource(auth_mod.signup)
check("#1 a failing invite raises HTTPException 400 (not swallowed)",
      "raise HTTPException(status_code=400" in sig and "invite_token" in sig)
check("#1 the create-firm branch is the ELSE of invite_token (mutually exclusive, no fall-through)",
      "else:" in sig and "create_firm" in sig)
check("#1 the failed-invite path does NOT log 'account still created' then continue (old fall-through gone)",
      "lands firm-less (a solo firm is backfilled)" not in sig)

# ── #7 signup honors an invite only for a confirmed email ──
print("\n#7 signup honors an invite only for a CONFIRMED email")
check("#7 signup checks email_confirmed_at before accepting an invite",
      "email_confirmed_at" in sig)

# ── #2 get_user_firm is deterministic (ORDER BY) ──
print("\n#2 get_user_firm picks the active firm deterministically")
guf = inspect.getsource(db_mod.SupabaseManager.get_user_firm)
check("#2 get_user_firm orders by created_at (no undefined row order)",
      ".order(" in guf and "created_at" in guf)

# ── #3 the ethical-wall lookup fails CLOSED on a live fault ──
print("#3 the ethical-wall lookup fails CLOSED on a LIVE fault (only missing-table degrades)")
check("#3 _is_missing_relation helper exists (distinguishes unapplied table from a live fault)",
      hasattr(db_mod, "_is_missing_relation"))
# behavioral: a missing-table error degrades to no-wall; any other error re-raises.
check("#3 _is_missing_relation True for a 42P01 / does-not-exist error",
      db_mod._is_missing_relation(Exception('relation "screens" does not exist (42P01)')))
check("#3 _is_missing_relation False for a generic live fault",
      not db_mod._is_missing_relation(Exception("connection reset by peer")))
svi = inspect.getsource(db_mod.SupabaseManager.screened_vault_ids)
check("#3 screened_vault_ids re-raises a non-missing-table error (no fail-open to empty)",
      "_is_missing_relation" in svi and "raise" in svi)
isv = inspect.getsource(db_mod.SupabaseManager.is_vault_screened)
check("#3 is_vault_screened re-raises a non-missing-table error",
      "_is_missing_relation" in isv and "raise" in isv)
avns = inspect.getsource(deps_mod.assert_vault_not_screened)
check("#3 assert_vault_not_screened fails CLOSED (denies) on a screen-lookup fault",
      "is_vault_screened" in avns and "503" in avns)

# ── #5 eval routes cap-gated + path param removed ──
print("\n#5 /admin/eval/* is cap-gated + the caller path param is removed")
runeval = inspect.getsource(admin_mod.run_evaluation)
results = inspect.getsource(admin_mod.get_eval_results)
check("#5 run_evaluation requires manage_members", 'require_cap("manage_members")' in runeval)
check("#5 get_eval_results requires manage_members", 'require_cap("manage_members")' in results)
check("#5 run_evaluation no longer takes a caller-supplied questions_path",
      "questions_path: str" not in runeval and "_DEFAULT_EVAL_PATH" in runeval)

# ── #6 override_abstain asserts the vault is in the caller's firm ──
print("\n#6 override_abstain asserts the vault is in the caller's firm (no foreign id)")
ova = inspect.getsource(admin_mod.override_abstain)
check("#6 override_abstain calls collection_in_firm (T2/T3) before recording",
      "collection_in_firm" in ova and "404" in ova)

print(f"\n{'='*60}\nAuth-audit regression gate: {_passed} passed, {_failed} failed\n{'='*60}")
sys.exit(1 if _failed else 0)
