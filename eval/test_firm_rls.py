"""F2f firm-scoped RLS backstop — the SQL-logic gate (offline, $0, no live DB).

Companion to the live verification (eval/verify_firm_rls_live.py, run ON DEMAND only — it needs
migration 016 applied). The live DB proves the Postgres PROPERTY (a non-member's SELECT returns 0
rows even with the app `.eq(user_id)` filter dropped). THIS gate proves the row-policy LOGIC,
fully offline, by modelling the policy as a PURE PREDICATE — the exact Python mirror of the SQL
helper `public.f2f_can_access_vault(p_user, p_firm, p_owner, p_vault)` in 016_firm_rls.sql:

    a collections row is visible/writable to a user IFF
      (the row has a firm  AND  the user is a member of that firm  AND  the user is NOT walled
       off that vault)                                                       -- firm + wall floor
      OR (the row has NO firm yet  AND  the user owns it)                    -- legacy/solo fallback

The same predicate backs BOTH the SELECT USING clause and the write WITH CHECK clause (so reads
and writes stay in lockstep — no read-protected-but-write-open asymmetry).

What this gate proves (the §F2f gate row):
  A  a firm member SEES their firm's vaults (the positive case).
  B  a NON-member sees 0 — EVEN with the app filter dropped (T2 — the row layer is the floor).
  C  a SCREENED member sees 0 at the ROW layer for the walled vault, and the screen beats a
     Managing Partner (T5 — deny-overrides at the row, proven to beat the most senior role).
  D  a cross-firm INSERT/UPDATE is blocked by WITH CHECK (T3) — placing a row in another firm,
     or moving one into a walled vault, fails the check.
  E  legacy / solo-MP + firm_id-NULL rows still resolve to EXACTLY the owner (THE non-regression
     that matters most — a firm-less user is never orphaned, a solo user sees only their own).
  F  deny-by-default — a row matching NO clause (other firm, someone else's legacy row, NULL
     user under a NULL firm) is invisible.
  G  lockstep with the SQL — the migration text actually contains the firm-membership clause, the
     active-screen (removed_at IS NULL) clause, the legacy NULL-firm owner fallback, a WITH CHECK,
     and the SECURITY DEFINER helpers — so this Python predicate isn't drifting from the shipped SQL.
  H  the non-regression is structural, not just today's-data: on a population where every firm is
     solo (one member each), the firm-JOIN policy yields the SAME visible set as the old per-user
     policy for every user (so the swap is dormant until a real multi-member firm exists).

Run:
    python -u eval/test_firm_rls.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ════════════════════════════════════════════════════════════════════════════════════════════
# The in-memory firm model + the PURE predicate mirror of f2f_can_access_vault (016_firm_rls.sql).
# This is the row policy expressed as data + a function, so we can assert its behavior exactly the
# way the authz gate asserts authorize()'s decision — no DB, no mocks, deterministic.
# ════════════════════════════════════════════════════════════════════════════════════════════

class FirmModel:
    """A tiny relational model: firm memberships, active screens (ethical walls), and collections.
    Mirrors the rows the SQL policy reads (firm_memberships, screens WHERE removed_at IS NULL,
    collections). Membership/screen sets are keyed exactly as the SECURITY DEFINER helpers query."""

    def __init__(self):
        self.memberships: set[tuple[str, str]] = set()      # (user_id, firm_id)
        self.active_screens: set[tuple[str, str]] = set()   # (user_id, vault_id) with removed_at IS NULL
        self.collections: dict[str, dict] = {}              # vault_id -> {firm_id|None, user_id(owner)}

    # ── the SQL helpers, 1:1 ──
    def f2f_user_in_firm(self, p_user, p_firm) -> bool:
        # SELECT EXISTS(... firm_memberships WHERE user_id=p_user AND firm_id=p_firm)
        return (p_user, p_firm) in self.memberships

    def f2f_vault_screened(self, p_user, p_vault) -> bool:
        # SELECT EXISTS(... screens WHERE user_id=p_user AND vault_id=p_vault AND removed_at IS NULL)
        return (p_user, p_vault) in self.active_screens

    def f2f_can_access_vault(self, p_user, p_firm, p_owner, p_vault) -> bool:
        # CASE WHEN p_firm IS NOT NULL THEN user_in_firm AND NOT screened ELSE owner = user END
        if p_firm is not None:
            return self.f2f_user_in_firm(p_user, p_firm) and not self.f2f_vault_screened(p_user, p_vault)
        return p_owner == p_user

    # ── what the SELECT USING clause would return for a given caller ──
    def visible_vaults(self, user_id) -> set[str]:
        """The set of collection ids `user_id` may SELECT under the F2f policy (row layer ONLY —
        NO app `.eq(user_id)` filter, deliberately, to prove the row floor stands alone)."""
        out = set()
        for vid, row in self.collections.items():
            if self.f2f_can_access_vault(user_id, row["firm_id"], row["user_id"], vid):
                out.add(vid)
        return out

    # ── what the OLD per-user policy (002) would return — for the non-regression comparison ──
    def visible_vaults_legacy_peruser(self, user_id) -> set[str]:
        return {vid for vid, row in self.collections.items() if row["user_id"] == user_id}

    # ── the WITH CHECK gate on a write of a row's target (firm, owner, vault) ──
    def write_check_passes(self, user_id, target_firm, target_owner, target_vault) -> bool:
        return self.f2f_can_access_vault(user_id, target_firm, target_owner, target_vault)


# ── Fixture identities ──────────────────────────────────────────────────────────────────────
MP        = "user-managing-partner"     # Managing Partner of Firm A (most senior)
ASSOC     = "user-associate"            # an Associate, also Firm A
OUTSIDER  = "user-outsider"             # member of Firm B only — a non-member of Firm A
SOLO      = "user-solo-legacy"          # legacy/solo user, owns a NULL-firm vault, no membership row needed

FIRM_A    = "firm-A"
FIRM_B    = "firm-B"

VAULT_A1  = "vault-A1"                   # Firm A vault, owned by MP
VAULT_A2  = "vault-A2"                   # Firm A vault, owned by ASSOC
VAULT_B1  = "vault-B1"                   # Firm B vault, owned by OUTSIDER
VAULT_LEG = "vault-legacy"              # NULL-firm legacy vault owned by SOLO


def base_model() -> FirmModel:
    m = FirmModel()
    # Firm A has TWO members (a real multi-member firm — this is where firm-scoping bites).
    m.memberships.add((MP, FIRM_A))
    m.memberships.add((ASSOC, FIRM_A))
    # Firm B has one member.
    m.memberships.add((OUTSIDER, FIRM_B))
    # SOLO is a legacy user — no firm membership (firm_id NULL on their vault).
    m.collections[VAULT_A1] = {"firm_id": FIRM_A, "user_id": MP}
    m.collections[VAULT_A2] = {"firm_id": FIRM_A, "user_id": ASSOC}
    m.collections[VAULT_B1] = {"firm_id": FIRM_B, "user_id": OUTSIDER}
    m.collections[VAULT_LEG] = {"firm_id": None, "user_id": SOLO}
    return m


# ════════════════════════════════════════════════════════════════════════════════════════════
# A — a firm member SEES their firm's vaults (positive case).
# ════════════════════════════════════════════════════════════════════════════════════════════
print("── A. A firm member sees their firm's vaults ──")
m = base_model()
mp_view = m.visible_vaults(MP)
check("A: MP of Firm A sees BOTH Firm-A vaults (incl. a colleague's vault in the same firm)",
      mp_view == {VAULT_A1, VAULT_A2}, str(mp_view))
assoc_view = m.visible_vaults(ASSOC)
check("A: the Associate of Firm A also sees both Firm-A vaults (firm-scoped, not owner-scoped)",
      assoc_view == {VAULT_A1, VAULT_A2}, str(assoc_view))


# ════════════════════════════════════════════════════════════════════════════════════════════
# B — a NON-member sees 0, even with the app filter dropped (T2 — row layer is the floor).
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── B. A non-member sees 0 at the row layer (T2 — app filter dropped) ──")
# visible_vaults() does NO .eq(user_id) — it is the raw row policy. The OUTSIDER (Firm B) must see
# none of Firm A's vaults, and the Firm-A members must see none of Firm B's.
check("B: OUTSIDER (Firm B) sees ZERO Firm-A vaults even with no app filter (T2)",
      VAULT_A1 not in m.visible_vaults(OUTSIDER) and VAULT_A2 not in m.visible_vaults(OUTSIDER),
      str(m.visible_vaults(OUTSIDER)))
check("B: MP (Firm A) sees ZERO Firm-B vaults (the boundary holds both ways)",
      VAULT_B1 not in mp_view)
check("B: a user with NO membership at all sees ZERO firm vaults (deny-by-default)",
      m.visible_vaults("user-nobody") == set(), str(m.visible_vaults("user-nobody")))


# ════════════════════════════════════════════════════════════════════════════════════════════
# C — a SCREENED member sees 0 at the ROW layer; the screen BEATS a Managing Partner (T5).
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── C. A screened member sees 0 at the row layer; the wall beats an MP (T5) ──")
mw = base_model()
# Raise an ethical wall: MP (the MOST senior role) is screened off VAULT_A1.
mw.active_screens.add((MP, VAULT_A1))
mp_walled = mw.visible_vaults(MP)
check("C: the screened MP no longer sees the walled vault at the ROW layer (deny-overrides-MP, T5)",
      VAULT_A1 not in mp_walled, str(mp_walled))
check("C: the screened MP STILL sees their other (un-walled) Firm-A vault (the wall is per-vault)",
      VAULT_A2 in mp_walled)
check("C: the wall is per-USER — the Associate (un-screened) still sees the walled vault",
      VAULT_A1 in mw.visible_vaults(ASSOC), str(mw.visible_vaults(ASSOC)))
# Lifting the screen (removed_at set ⇒ no longer in active_screens) restores access on the next eval.
mw.active_screens.discard((MP, VAULT_A1))
check("C: lifting the screen restores the MP's row visibility (soft-remove ⇒ active set drops it)",
      VAULT_A1 in mw.visible_vaults(MP))


# ════════════════════════════════════════════════════════════════════════════════════════════
# D — a cross-firm INSERT/UPDATE is blocked by WITH CHECK (T3).
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── D. Cross-firm write blocked by WITH CHECK (T3) ──")
md = base_model()
# OUTSIDER (Firm B) tries to INSERT a collection stamped with FIRM_A (a cross-firm confused-deputy).
check("D: OUTSIDER cannot write a row into Firm A (WITH CHECK fails — not a member of Firm A, T3)",
      md.write_check_passes(OUTSIDER, FIRM_A, OUTSIDER, "vault-new") is False)
# A Firm-A member CAN write a row into their own firm (the legitimate case still works).
check("D: an MP CAN write a row into their OWN firm (the legitimate write is not broken)",
      md.write_check_passes(MP, FIRM_A, MP, "vault-new") is True)
# A member screened off the target vault cannot write to it (the wall covers writes too — lockstep).
md.active_screens.add((ASSOC, VAULT_A2))
check("D: a member screened off a vault cannot WRITE it either (WITH CHECK uses the same predicate)",
      md.write_check_passes(ASSOC, FIRM_A, ASSOC, VAULT_A2) is False)
# Moving a row from your firm into another firm fails the WITH CHECK on the new firm value.
check("D: an MP cannot UPDATE a row to move it into another firm (WITH CHECK on the new firm_id, T3)",
      md.write_check_passes(MP, FIRM_B, MP, VAULT_A1) is False)


# ════════════════════════════════════════════════════════════════════════════════════════════
# E — legacy / solo-MP + firm_id-NULL rows resolve to EXACTLY the owner (THE non-regression).
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── E. Legacy / solo / NULL-firm rows resolve to exactly the owner (non-regression) ──")
me = base_model()
check("E: SOLO sees their own NULL-firm legacy vault (a firm-less user is NOT orphaned)",
      VAULT_LEG in me.visible_vaults(SOLO), str(me.visible_vaults(SOLO)))
check("E: SOLO sees ONLY their own vault — no firm vaults bleed in (they have no membership)",
      me.visible_vaults(SOLO) == {VAULT_LEG}, str(me.visible_vaults(SOLO)))
check("E: a NULL-firm legacy vault is INVISIBLE to a non-owner (no firm ⇒ owner-only, never shared)",
      VAULT_LEG not in me.visible_vaults(MP) and VAULT_LEG not in me.visible_vaults(OUTSIDER))
# The solo-MP shape: a user who is the sole member of their own firm, owning a firm-stamped vault —
# the live backfill (012) state. They must see exactly their own vault, same as pre-F2.
ms = FirmModel()
ms.memberships.add((SOLO, "firm-solo"))
ms.collections["v-solo"] = {"firm_id": "firm-solo", "user_id": SOLO}
check("E: a solo-MP (sole member of own firm, firm-stamped vault) sees exactly their own vault",
      ms.visible_vaults(SOLO) == {"v-solo"}, str(ms.visible_vaults(SOLO)))


# ════════════════════════════════════════════════════════════════════════════════════════════
# F — deny-by-default: a row matching NO clause is invisible.
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── F. Deny-by-default (a row matching no clause is invisible) ──")
mf = base_model()
# A non-owner against a NULL-firm row → CASE-ELSE owner check fails → invisible.
check("F: NULL-firm row + non-owner ⇒ invisible (CASE-ELSE owner branch denies)",
      mf.f2f_can_access_vault(MP, None, SOLO, VAULT_LEG) is False)
# A firmed row whose firm the user isn't in → firm-member EXISTS is false → invisible.
check("F: firmed row + non-member ⇒ invisible (firm EXISTS denies)",
      mf.f2f_can_access_vault(OUTSIDER, FIRM_A, MP, VAULT_A1) is False)
# A NULL user (shouldn't happen post-auth) against any row ⇒ invisible (no membership, owner≠null).
check("F: a null/unknown caller sees nothing (no membership, owner mismatch)",
      mf.visible_vaults(None) == set(), str(mf.visible_vaults(None)))


# ════════════════════════════════════════════════════════════════════════════════════════════
# G — lockstep with the shipped SQL: the migration actually contains the clauses we model.
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── G. The Python predicate is in lockstep with the shipped SQL (no drift) ──")
SQL_PATH = Path(__file__).resolve().parent.parent / "docs" / "migrations" / "016_firm_rls.sql"
sql = SQL_PATH.read_text() if SQL_PATH.exists() else ""
sql_l = sql.lower()
check("G: migration 016_firm_rls.sql exists", bool(sql))
check("G: it defines the membership helper f2f_user_in_firm over firm_memberships",
      "f2f_user_in_firm" in sql_l and "firm_memberships" in sql_l)
check("G: it defines the screen helper checking active (removed_at IS NULL) screens",
      "f2f_vault_screened" in sql_l and "removed_at is null" in sql_l)
check("G: the composite predicate ANDs firm-membership with NOT-screened (deny-overrides at the row)",
      "f2f_can_access_vault" in sql_l and "not public.f2f_vault_screened" in sql_l)
check("G: it keeps the legacy NULL-firm owner fallback (p_owner = p_user)",
      "p_owner = p_user" in sql_l)
check("G: it REPLACES the old per-user collections policy (drops 'Users see own collections')",
      'drop policy if exists "users see own collections"' in sql_l)
check("G: the new collections SELECT policy uses the firm+wall predicate",
      'for select' in sql_l and 'f2f_can_access_vault' in sql_l)
check("G: a WITH CHECK clause is present (the T3 write backstop)", "with check" in sql_l)
check("G: the helpers are SECURITY DEFINER with a pinned search_path (recursion + hardening)",
      sql_l.count("security definer") >= 3 and "set search_path = public" in sql_l)
check("G: it documents the service-role / T9 honesty (WITH CHECK is a backstop, not the write guard)",
      "service-role" in sql_l and "t9" in sql_l)
check("G: it documents the rollback to the per-user policy (behavior-changing on apply)",
      "rollback" in sql_l and "users see own collections" in sql_l)
# The shared firm_id-less tables are KEPT per-user, not firm-JOINed (the decision).
check("G: documents/conversations/messages/audit_log are re-asserted per-user (no firm_id JOIN)",
      all(t in sql_l for t in ("f2f documents owner only", "f2f conversations owner only",
                               "f2f messages owner only", "f2f audit_log owner only")))


# ════════════════════════════════════════════════════════════════════════════════════════════
# H — the non-regression is STRUCTURAL: on a solo-firm population, firm-JOIN == per-user.
# ════════════════════════════════════════════════════════════════════════════════════════════
print("\n── H. Structural non-regression: solo-firm population ⇒ firm-JOIN equals per-user ──")
# Build the live shape: N users, each the SOLE member of their own firm, each owning a firm-stamped
# vault. For EVERY user, the new firm-JOIN policy must yield the SAME visible set as the old
# per-user policy. (This is why flipping the policy changes nothing until a real multi-member firm
# exists — exactly the live state proven on 2026-06-25: 3 solo firms, 0 cross-member leakage.)
mh = FirmModel()
solo_users = [f"u{i}" for i in range(5)]
for i, u in enumerate(solo_users):
    mh.memberships.add((u, f"f{i}"))
    mh.collections[f"cv{i}"] = {"firm_id": f"f{i}", "user_id": u}
mismatch = [u for u in solo_users
            if mh.visible_vaults(u) != mh.visible_vaults_legacy_peruser(u)]
check("H: every solo-firm user sees EXACTLY what the old per-user policy showed (swap is dormant)",
      mismatch == [], f"users whose visibility changed: {mismatch}")
# And the moment a firm gains a second member, the firm-JOIN INTENTIONALLY diverges (the feature):
mh.memberships.add(("u-new", "f0"))                 # add a colleague to firm f0
mh.collections["cv0b"] = {"firm_id": "f0", "user_id": "u0"}
check("H: once a firm is multi-member, a colleague gains firm-scoped visibility (the intended change)",
      "cv0" in mh.visible_vaults("u-new") and "cv0b" in mh.visible_vaults("u-new"),
      str(mh.visible_vaults("u-new")))
# …but a per-user policy would NOT have shown u-new anything (proving the divergence is real).
check("H: the old per-user policy would have shown that colleague nothing (the divergence is real)",
      mh.visible_vaults_legacy_peruser("u-new") == set())


# ── tally ──
print(f"\n{'='*60}")
print(f"  test_firm_rls: {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(0 if _failed == 0 else 1)
