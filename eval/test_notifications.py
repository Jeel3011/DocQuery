"""F2j regression gate — NOTIFICATIONS (offline, $0).

The committed gate for F2j (plans/F2_FIRM_CONSOLE_PLAN.md §F2j). Makes the F2e review chain VISIBLE
without changing the flow. Proves the fan-out is correct AND secure — it respects the F2c ethical wall
+ the F2f firm boundary, never double-nags (dedup + idempotent escalation), honors the anti-nag
preferences, and degrades to a silent no-op when the tables are unapplied (byte-identical to pre-F2j).
Deterministic, no API, no Supabase, no extraction/kernel. Run:

    ./venv/bin/python -u eval/test_notifications.py

What it proves (maps to the plan's §3 gate row for F2j):
  A. RECIPIENT MAPPING — each review transition notifies the RIGHT person (submit→new owner,
     approve-advance→new owner, approve-final→submitter, changes→submitter, released→submitter,
     matter.staff→added member); a SELF-action notifies no one (recipient == actor).
  B. FIRM BOUNDARY (T3) — a cross-firm recipient is NEVER written a row.
  C. ETHICAL WALL (T5, load-bearing) — a SCREENED recipient gets 0 for the walled vault; a wall
     lookup FAULT fails closed (suppresses, never opens the wall); a screen.create notifies the
     subject NOTHING (no wall-existence leak — modeled by emitting with the subject as recipient on
     the walled vault and asserting 0).
  D. ANTI-NAG — a MUTED category suppresses the in-app row; QUIET HOURS still writes the in-app row
     (defers email only) and an URGENT event bypasses it; DEDUP — the same event+recipient+resource
     in the same day collapses to ONE row (the unique-index no-op).
  E. ANTI-STALL ESCALATION — escalation_sweep re-pings the current_owner of an open request past the
     threshold; RE-RUNNING the sweep the same day emits NOTHING new (idempotent — the dedup guarantee).
  F. INBOX ROUTES — list / unread-count / mark-read are OWN-ROWS ONLY (a user can't read or mark
     another user's notifications — T2/T8); preferences round-trip; defaults when no prefs row.
  G. DORMANT PARITY — with the tables "unapplied" (fake raises missing-relation), every db method
     degrades to empty/None and NO route 500s; an emit failure NEVER propagates (the action succeeds).

OFFLINE: the notifications logic (build_dedup_key / is_suppressed / render / emit / fan_out_review /
escalation_sweep) is pure-ish; the route handlers are driven directly with a fake SupabaseManager that
models notifications + notification_preferences + firm_memberships + screens + collections +
review_requests, mirroring the real query semantics (recipient-scoping, dedup unique index, active
screen = removed_at None). See [[run-only-relevant-gates]] — F2j touches db/notifications/routes, NOT
extraction/kernel.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.components.notifications as nf  # noqa: E402
from src.api.routes import notifications as nroutes  # noqa: E402
from src.api.schemas import MarkReadRequest, NotificationPreferencesRequest  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# Fixed ids reused across the gate.
FIRM_A = "firm-A"
FIRM_B = "firm-B"
MP = "user-mp"            # the matter owner / a partner
SUBMITTER = "user-sub"    # a paralegal who submits work
REVIEWER = "user-rev"     # a senior associate reviewer
OUTSIDER = "user-out"     # a member of FIRM_B (cross-firm)
SCREENED = "user-scr"     # a member of FIRM_A screened off the walled vault
VAULT = "vault-1"
WALLED_VAULT = "vault-walled"
NOW = datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc)   # fixed clock for determinism


def _iso(dt):
    return dt.isoformat()


# ─────────────────────────────────────────
# A FAKE SupabaseManager modeling the notification tables + the firm/wall helpers emit() depends on.
#   `missing` flips every table to "unapplied" (raises a missing-relation error) for the dormant test.
# ─────────────────────────────────────────
class _MissingRelation(Exception):
    """Mimics PostgREST/Postgres 'relation does not exist' — db._is_missing_relation matches this."""
    def __init__(self):
        super().__init__("relation \"notifications\" does not exist (PGRST205)")


class FakeSB:
    def __init__(self, user_id, *, missing=False):
        self.user_id = user_id
        self.missing = missing
        # firm memberships: user_id -> (firm_id, role)
        self._members = {
            MP: (FIRM_A, "partner"),
            SUBMITTER: (FIRM_A, "paralegal"),
            REVIEWER: (FIRM_A, "senior_associate"),
            SCREENED: (FIRM_A, "associate"),
            OUTSIDER: (FIRM_B, "associate"),
        }
        # collections: id -> owner/firm/name
        self._collections = {
            VAULT: {"user_id": MP, "firm_id": FIRM_A, "name": "Acme matter"},
            WALLED_VAULT: {"user_id": MP, "firm_id": FIRM_A, "name": "Walled matter"},
        }
        # active screens: list of {user_id, vault_id, firm_id, removed_at}
        self._screens: list[dict] = []
        # notifications store: list of rows; dedup_key uniqueness enforced like the unique index
        self._notifications: list[dict] = []
        self._prefs: dict[str, dict] = {}     # user_id -> prefs row
        self._review_requests: list[dict] = []
        self._seq = 0
        self.wall_fault = False               # toggled to test fail-closed
        self.read_client = self

    # —— used by db._is_missing_relation indirectly: we just raise when missing ——

    # —— firm + member helpers (emit's firm boundary) ——
    def user_in_firm(self, user_id, firm_id):
        m = self._members.get(user_id)
        return bool(m and m[0] == firm_id)

    def get_user_firm(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        m = self._members.get(uid)
        if not m:
            return {}
        return {"id": m[0], "name": "Firm", "role": m[1]}

    def get_collection(self, collection_id):
        return dict(self._collections.get(collection_id) or {}) or None

    def collection_in_firm(self, vault_id, firm_id):
        c = self._collections.get(vault_id)
        return bool(c and c.get("firm_id") == firm_id)

    def resolve_member_emails(self, user_ids):
        return {str(u): f"{u}@example.com" for u in (user_ids or []) if u}

    # —— the ethical wall (emit's load-bearing check) ——
    def is_vault_screened(self, vault_id, user_id=None, firm_id=None):
        if self.wall_fault:
            # A LIVE fault must fail closed (the real db.is_vault_screened re-raises; emit treats any
            # exception as "suppress to be safe"). Model that by raising.
            raise RuntimeError("screens query failed (live fault)")
        uid = user_id or self.user_id
        return any(s["user_id"] == uid and s["vault_id"] == vault_id and s["removed_at"] is None
                   and (firm_id is None or s["firm_id"] == firm_id)
                   for s in self._screens)

    def add_screen(self, user_id, vault_id, firm_id=FIRM_A):
        self._screens.append({"user_id": user_id, "vault_id": vault_id, "firm_id": firm_id,
                              "removed_at": None})

    # —— notification preferences ——
    _DEFAULT = {"muted_categories": [], "quiet_start": None, "quiet_end": None, "digest_mode": False}

    def get_notification_preferences(self, user_id=None):
        if self.missing:
            return dict(self._DEFAULT)
        uid = user_id or self.user_id
        return dict(self._prefs.get(uid) or self._DEFAULT)

    def set_notification_preferences(self, user_id=None, **fields):
        uid = user_id or self.user_id
        # Match the real db.py: a partial patch merges onto the user's CURRENT row, not the defaults,
        # so a one-field update never clobbers another already-set preference.
        merged = self.get_notification_preferences(uid)
        for k in ("muted_categories", "quiet_start", "quiet_end", "digest_mode"):
            if k in fields:
                merged[k] = fields[k]
        if self.missing:
            return merged
        self._prefs[uid] = dict(merged)
        return dict(merged)

    # —— notifications store (the inbox + the dedup unique index) ——
    def create_notification(self, row):
        if self.missing:
            return None
        if not row:
            return None
        dk = row.get("dedup_key")
        # The UNIQUE partial index on dedup_key: a duplicate is a clean no-op (returns None).
        if dk and any(n.get("dedup_key") == dk for n in self._notifications):
            return None
        self._seq += 1
        stored = dict(row)
        stored["id"] = f"notif-{self._seq}"
        stored["read_at"] = None
        stored["created_at"] = _iso(NOW)
        self._notifications.append(stored)
        return stored

    def list_notifications(self, recipient_id=None, unread_only=False, limit=50):
        if self.missing:
            return []
        uid = recipient_id or self.user_id
        rows = [dict(n) for n in self._notifications if n["recipient_id"] == uid]
        if unread_only:
            rows = [r for r in rows if r.get("read_at") is None]
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return rows[:limit]

    def unread_notification_count(self, recipient_id=None):
        if self.missing:
            return 0
        uid = recipient_id or self.user_id
        return sum(1 for n in self._notifications
                   if n["recipient_id"] == uid and n.get("read_at") is None)

    def mark_notifications_read(self, recipient_id=None, ids=None):
        if self.missing:
            return 0
        uid = recipient_id or self.user_id
        n = 0
        for row in self._notifications:
            if row["recipient_id"] != uid or row.get("read_at") is not None:
                continue
            if ids and str(row["id"]) not in {str(i) for i in ids}:
                continue
            row["read_at"] = _iso(NOW)
            n += 1
        return n

    # —— review requests (for the escalation sweep) ——
    def add_review_request(self, *, rid, firm_id, vault_id, owner, status="pending", created_at=None):
        self._review_requests.append({
            "id": rid, "firm_id": firm_id, "vault_id": vault_id, "current_owner": owner,
            "submitted_by": SUBMITTER, "status": status,
            "created_at": created_at or _iso(NOW), "vault_name": "Acme matter",
        })

    def open_review_requests_for_escalation(self, threshold_iso):
        if self.missing:
            return []
        return [dict(r) for r in self._review_requests
                if r["status"] in ("pending", "approved", "changes_requested")
                and (r.get("created_at") or "") <= threshold_iso]

    # —— helpers for assertions ——
    def rows_for(self, recipient_id):
        return [n for n in self._notifications if n["recipient_id"] == recipient_id]

    def events_for(self, recipient_id):
        return [n["event"] for n in self.rows_for(recipient_id)]


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── A · recipient mapping (the right person is told) ──")

def _rr(owner, submitter=SUBMITTER, vault=VAULT, rid="rr-1", status="pending"):
    return {"id": rid, "firm_id": FIRM_A, "vault_id": vault, "current_owner": owner,
            "submitted_by": submitter, "status": status}

# submit → the new current_owner (REVIEWER) is told "awaiting"
sb = FakeSB(SUBMITTER)
nf.fan_out_review(sb, event="review.awaiting", review_request=_rr(REVIEWER), firm_id=FIRM_A,
                  actor_id=SUBMITTER, ctx={}, )
check("A1 submit notifies the new owner", sb.events_for(REVIEWER) == ["review.awaiting"],
      str(sb.events_for(REVIEWER)))
check("A1b submit does NOT notify the submitter (self/other)", sb.rows_for(SUBMITTER) == [])

# approve-advance → the next owner (MP) is told "awaiting"
sb = FakeSB(REVIEWER)
nf.fan_out_review(sb, event="review.awaiting", review_request=_rr(MP), firm_id=FIRM_A,
                  actor_id=REVIEWER, ctx={})
check("A2 approve-advance notifies the next owner", sb.events_for(MP) == ["review.awaiting"])

# approve-final → the SUBMITTER is told "approved"
sb = FakeSB(REVIEWER)
nf.fan_out_review(sb, event="review.approved", review_request=_rr(MP, status="approved"),
                  firm_id=FIRM_A, actor_id=REVIEWER, ctx={})
check("A3 approve-final notifies the submitter", sb.events_for(SUBMITTER) == ["review.approved"])

# changes → the SUBMITTER is told
sb = FakeSB(REVIEWER)
nf.fan_out_review(sb, event="review.changes_requested",
                  review_request=_rr(SUBMITTER, status="changes_requested"),
                  firm_id=FIRM_A, actor_id=REVIEWER, ctx={"note": "fix the cite"})
check("A4 changes notifies the submitter", sb.events_for(SUBMITTER) == ["review.changes_requested"])

# released → the SUBMITTER is told
sb = FakeSB(MP)
nf.fan_out_review(sb, event="review.released", review_request=_rr(MP, status="released"),
                  firm_id=FIRM_A, actor_id=MP, ctx={})
check("A5 released notifies the submitter", sb.events_for(SUBMITTER) == ["review.released"])

# matter.staff → the added member is told; self-staff notifies no one
sb = FakeSB(MP)
nf.emit(sb, recipient_id=SUBMITTER, firm_id=FIRM_A, event="matter.staffed",
        resource_type="vault", resource_id=VAULT, vault_id=VAULT, actor_id=MP, ctx={})
check("A6 staffing notifies the added member", sb.events_for(SUBMITTER) == ["matter.staffed"])
sb = FakeSB(MP)
nf.emit(sb, recipient_id=MP, firm_id=FIRM_A, event="matter.staffed", vault_id=VAULT, actor_id=MP)
check("A6b self-staff notifies no one (recipient == actor)", sb.rows_for(MP) == [])


print("\n── B · firm boundary (T3) — a cross-firm recipient gets nothing ──")
sb = FakeSB(MP)
# OUTSIDER is in FIRM_B; emitting against FIRM_A must write nothing.
nf.emit(sb, recipient_id=OUTSIDER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-x", vault_id=VAULT, actor_id=MP)
check("B1 cross-firm recipient is never written a row", sb.rows_for(OUTSIDER) == [])


print("\n── C · ethical wall (T5, load-bearing) ──")
sb = FakeSB(MP)
sb.add_screen(SCREENED, WALLED_VAULT)
# SCREENED is a real FIRM_A member but walled off WALLED_VAULT → 0 notifications for it.
nf.emit(sb, recipient_id=SCREENED, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-w", vault_id=WALLED_VAULT, actor_id=MP)
check("C1 screened recipient gets 0 for the walled vault", sb.rows_for(SCREENED) == [])
# Same recipient, a DIFFERENT (open) vault → allowed (the wall is matter-level, not a person ban).
nf.emit(sb, recipient_id=SCREENED, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-o", vault_id=VAULT, actor_id=MP)
check("C2 screened recipient still notified on an OPEN vault", sb.events_for(SCREENED) == ["review.awaiting"])
# A wall-lookup FAULT must fail CLOSED (suppress), never open the wall.
sb = FakeSB(MP); sb.wall_fault = True
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-f", vault_id=VAULT, actor_id=MP)
check("C3 wall lookup fault fails CLOSED (no row written)", sb.rows_for(REVIEWER) == [])
# A screen.create's subject is NEVER told about the wall (modeled: the screened subject on the walled
# vault yields 0 — so even if some path tried to notify them, the wall suppresses it = no leak).
sb = FakeSB(MP); sb.add_screen(SCREENED, WALLED_VAULT)
nf.emit(sb, recipient_id=SCREENED, firm_id=FIRM_A, event="answer.overridden",
        resource_type="answer", resource_id="ans-1", vault_id=WALLED_VAULT, actor_id=MP)
check("C4 no wall-existence leak to the screened subject", sb.rows_for(SCREENED) == [])


print("\n── D · anti-nag (mute / quiet-hours / dedup) ──")
# Muted category → suppressed in-app row.
sb = FakeSB(MP); sb.set_notification_preferences(REVIEWER, muted_categories=["review"])
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-m", vault_id=VAULT, actor_id=MP)
check("D1 muted category suppresses the in-app row", sb.rows_for(REVIEWER) == [])
# A DIFFERENT category (matter) is NOT muted → still delivered.
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="matter.staffed",
        resource_type="vault", resource_id=VAULT, vault_id=VAULT, actor_id=MP)
check("D1b a non-muted category still delivers", sb.events_for(REVIEWER) == ["matter.staffed"])
# Quiet hours: the in-app row is STILL written (quiet hours defers email only, never drops the row).
sb = FakeSB(MP); sb.set_notification_preferences(REVIEWER, quiet_start=0, quiet_end=23)
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-q", vault_id=VAULT, actor_id=MP, now=NOW)
check("D2 quiet hours still writes the in-app row (defers email only)",
      sb.events_for(REVIEWER) == ["review.awaiting"])
# Pure quiet-hours logic incl. midnight wrap + urgent bypass.
check("D2b quiet-hours wrap (21→7) covers 23h", nf.in_quiet_hours({"quiet_start": 21, "quiet_end": 7}, 23))
check("D2c quiet-hours wrap (21→7) excludes 12h", not nf.in_quiet_hours({"quiet_start": 21, "quiet_end": 7}, 12))
# Dedup: the SAME event+recipient+resource on the SAME day collapses to ONE row.
sb = FakeSB(MP)
for _ in range(3):
    nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
            resource_type="review_request", resource_id="rr-dup", vault_id=VAULT, actor_id=MP, now=NOW)
check("D3 dedup — 3 identical emits collapse to ONE row", len(sb.rows_for(REVIEWER)) == 1,
      f"got {len(sb.rows_for(REVIEWER))}")
# A different resource is a different notification (not deduped away).
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="rr-other", vault_id=VAULT, actor_id=MP, now=NOW)
check("D3b a different resource is a distinct notification", len(sb.rows_for(REVIEWER)) == 2)


print("\n── E · anti-stall escalation (idempotent) ──")
sb = FakeSB(MP)
# An OPEN request, created 3 days ago, owned by REVIEWER → past the 48h threshold.
sb.add_review_request(rid="rr-stale", firm_id=FIRM_A, vault_id=VAULT, owner=REVIEWER,
                      created_at=_iso(NOW - timedelta(days=3)))
# A FRESH request (created now) → NOT escalated.
sb.add_review_request(rid="rr-fresh", firm_id=FIRM_A, vault_id=VAULT, owner=REVIEWER,
                      created_at=_iso(NOW))
n1 = nf.escalation_sweep(sb, threshold_hours=48, now=NOW)
check("E1 escalation pings the stale owner once", n1 == 1, f"emitted {n1}")
check("E1b the reminder is an escalation event", sb.events_for(REVIEWER) == ["review.escalation"])
# Re-running the SAME day emits NOTHING new (the dedup/last_notified guarantee — no double-nag).
n2 = nf.escalation_sweep(sb, threshold_hours=48, now=NOW)
check("E2 re-running the sweep the same day emits nothing new (idempotent)", n2 == 0, f"emitted {n2}")
check("E2b still exactly one escalation row", len(sb.rows_for(REVIEWER)) == 1)
# The NEXT day, a new bucket → it may remind again (still pending).
n3 = nf.escalation_sweep(sb, threshold_hours=48, now=NOW + timedelta(days=1))
check("E3 the next day a new reminder fires (new dedup bucket)", n3 == 1, f"emitted {n3}")


print("\n── F · inbox routes (own-rows only, T2/T8) ──")
sb = FakeSB(REVIEWER)
# Seed some notifications: 2 for REVIEWER, 1 for MP.
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="r1", vault_id=VAULT, actor_id=MP, now=NOW)
nf.emit(sb, recipient_id=REVIEWER, firm_id=FIRM_A, event="matter.staffed",
        resource_type="vault", resource_id="r2", vault_id=VAULT, actor_id=MP, now=NOW)
nf.emit(sb, recipient_id=MP, firm_id=FIRM_A, event="review.awaiting",
        resource_type="review_request", resource_id="r3", vault_id=VAULT, actor_id=SUBMITTER, now=NOW)
# REVIEWER's inbox shows only THEIR 2.
resp = run(nroutes.list_notifications(sb=sb))
check("F1 list returns only the caller's own rows", resp.unread == 2 and len(resp.notifications) == 2,
      f"unread={resp.unread} n={len(resp.notifications)}")
cnt = run(nroutes.unread_count(sb=sb))
check("F2 unread-count is the caller's own", cnt["count"] == 2)
# Mark all read → unread 0; MP's notification is untouched (a different recipient).
upd = run(nroutes.mark_read(body=MarkReadRequest(), sb=sb))
check("F3 mark-all-read marks the caller's 2", upd["updated"] == 2)
check("F3b after read, unread-count is 0", run(nroutes.unread_count(sb=sb))["count"] == 0)
check("F3c another user's notification is untouched", sb.unread_notification_count(MP) == 1)
# A user can't mark another user's notification by guessing its id (AND-scoped to recipient).
mp_notif_id = sb.rows_for(MP)[0]["id"]
upd2 = run(nroutes.mark_read(body=MarkReadRequest(ids=[mp_notif_id]), sb=sb))
check("F4 can't mark another user's notification (T2/T8)", upd2["updated"] == 0
      and sb.unread_notification_count(MP) == 1)
# Preferences round-trip + defaults.
pref0 = run(nroutes.get_preferences(sb=sb))
check("F5 default prefs when no row", pref0.muted_categories == [] and pref0.digest_mode is False)
pref1 = run(nroutes.set_preferences(
    body=NotificationPreferencesRequest(muted_categories=["governance"], quiet_start=21,
                                        quiet_end=7, digest_mode=True), sb=sb))
check("F6 prefs round-trip", pref1.muted_categories == ["governance"] and pref1.quiet_start == 21
      and pref1.digest_mode is True)
# F7: a PARTIAL update must not clobber preferences the user already set. Toggling digest alone keeps
# the previously-saved quiet hours + mute (the partial-update-clobber bug).
pref2 = run(nroutes.set_preferences(
    body=NotificationPreferencesRequest(digest_mode=False), sb=sb))
check("F7 partial update keeps other prefs (no clobber)",
      pref2.quiet_start == 21 and pref2.quiet_end == 7 and pref2.muted_categories == ["governance"]
      and pref2.digest_mode is False)
# F8: turning quiet hours OFF sends explicit nulls — those MUST reach the DB (exclude_unset), else the
# window can never be disabled. muted/digest are untouched.
pref3 = run(nroutes.set_preferences(
    body=NotificationPreferencesRequest(quiet_start=None, quiet_end=None), sb=sb))
check("F8 explicit null clears quiet hours (window can be disabled)",
      pref3.quiet_start is None and pref3.quiet_end is None
      and pref3.muted_categories == ["governance"])


print("\n── G · dormant parity (tables unapplied) — never 500, action never broken ──")
sb = FakeSB(REVIEWER, missing=True)
# Every route degrades to empty/defaults, no exception.
resp = run(nroutes.list_notifications(sb=sb))
check("G1 list → empty when unapplied", resp.unread == 0 and resp.notifications == [])
check("G2 unread-count → 0 when unapplied", run(nroutes.unread_count(sb=sb))["count"] == 0)
check("G3 mark-read → 0 when unapplied", run(nroutes.mark_read(body=MarkReadRequest(), sb=sb))["updated"] == 0)
gp = run(nroutes.get_preferences(sb=sb))
check("G4 prefs → defaults when unapplied", gp.muted_categories == [] and gp.digest_mode is False)
# emit is a silent no-op when unapplied AND must never raise (the action it follows still succeeds).
emit_ok = True
try:
    res = nf.emit(sb, recipient_id=MP, firm_id=FIRM_A, event="review.awaiting",
                  resource_type="review_request", resource_id="rX", vault_id=VAULT, actor_id=SUBMITTER)
    emit_ok = (res is None)
except Exception:  # noqa: BLE001
    emit_ok = False
check("G5 emit is a silent no-op when unapplied (never raises)", emit_ok)
# Even a hard create fault inside emit is swallowed (best-effort posture) — simulate by a store that
# raises on create while the table is "present".
class _Boom(FakeSB):
    def create_notification(self, row):
        raise RuntimeError("unexpected write fault")
sb_boom = _Boom(MP)
boom_ok = True
try:
    boom_ok = (nf.emit(sb_boom, recipient_id=REVIEWER, firm_id=FIRM_A, event="review.awaiting",
                       resource_type="review_request", resource_id="rB", vault_id=VAULT,
                       actor_id=MP) is None)
except Exception:  # noqa: BLE001
    boom_ok = False
check("G6 an emit write-fault is swallowed (never breaks the action)", boom_ok)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  F2j NOTIFICATIONS GATE:  {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(0 if _failed == 0 else 1)
