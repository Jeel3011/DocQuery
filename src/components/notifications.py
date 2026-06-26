"""F2j — NOTIFICATIONS: the fan-out logic (mostly pure, offline-testable; thin DB writes).

plans/F2_FIRM_CONSOLE_PLAN.md §F2j. Makes the F2e review chain VISIBLE (the engine already can't
stall — every review_requests row names a current_owner — but nobody was TOLD). This module is the
"telling": it turns a governance/review EVENT into an in-app notification row for the RIGHT person,
respecting the F2c ethical wall + the F2f firm boundary, with anti-nag (mute / quiet-hours / dedup)
and anti-stall (escalation) baked in.

DESIGN (research-grounded — the industry-standard, security-safe choices Jeel asked for):
  • Fan-out = the TRANSACTIONAL-OUTBOX pattern. For an in-app inbox the notification IS a row in the
    SAME Postgres as the action, so we write it in the action's own flow (no dual-write inconsistency,
    no notification lost on a broker outage). The `notifications` table is the inbox + the outbox + the
    queue seam; a worker drains the EMAIL channel later (email_status='pending') — no schema change.
  • Idempotency = a dedup_key + a UNIQUE index (the "last_notified_at" rule). The same event to the
    same person about the same resource on the same day collapses to ONE row — a re-fired event or a
    restarted escalation sweep NEVER double-nags.

PURITY: build_dedup_key / is_suppressed / render take primitives only (no DB, no clock, no FastAPI) —
exactly like authz.py / review_chain.py — so the gate eval/test_notifications.py is $0/offline. emit /
fan_out_review / escalation_sweep are the thin glue that reuses the verified db helpers (user_in_firm,
is_vault_screened, get_notification_preferences, create_notification).

SECURITY (the F2 posture, enforced in the DATA PATH not assumed):
  • FIRM BOUNDARY (T3): a recipient must be a member of the event's firm (db.user_in_firm) — a
    cross-firm user is never written a row.
  • ETHICAL WALL (T5, load-bearing): if the event names a vault, a recipient screened off it
    (db.is_vault_screened — fail-closed on a live fault) is never written a row. A screen.create is
    deliberately NEVER notified to its subject (the wall's existence must not leak — F2c hides it).
  • NEVER BREAKS THE ACTION: emit is best-effort (try/except, never raises) — same posture as
    log_audit. A notification failure must never 500 the review/staffing action that triggered it.

DORMANT-ON-EMPTY: every db method degrades to empty/None on a missing table (migration 017 unapplied),
so with no tables emit is a silent no-op and the app is byte-identical to pre-F2j.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Event → category (the preference bucket the anti-nag mute toggles act on) ─────────────────────
# Three buckets: review-chain flow, matter staffing, governance. A muted category suppresses the
# in-app row for its events. Unknown events default to 'review' (the dominant flow).
EVENT_CATEGORY = {
    "review.awaiting": "review",
    "review.approved": "review",
    "review.changes_requested": "review",
    "review.released": "review",
    "review.escalation": "review",
    "matter.staffed": "matter",
    "answer.overridden": "governance",
}

# Events that bypass quiet-hours (urgent — the anti-stall reminder must not be silently deferred).
_URGENT_EVENTS = frozenset({"review.escalation"})

# The full set of categories (for the preferences UI + validation).
CATEGORIES = ("review", "matter", "governance")


def category_for(event: str) -> str:
    """The preference bucket an event belongs to (defaults to 'review')."""
    return EVENT_CATEGORY.get(event, "review")


# ── PURE: the idempotency key ─────────────────────────────────────────────────────────────────────

def build_dedup_key(recipient_id: str, event: str, resource_id: Optional[str],
                     window_bucket: str) -> str:
    """The idempotency key for a notification (the "last_notified_at" anti-duplicate rule). sha256 of
    (recipient, event, resource, a coarse time bucket). The same event to the same person about the
    same resource within the same bucket collapses to ONE row via the UNIQUE index — so a re-fired
    event / a restarted escalation sweep never double-nags. `window_bucket` is typically the date
    (YYYY-MM-DD) so reminders dedup per-day. PURE."""
    raw = f"{recipient_id}|{event}|{resource_id or ''}|{window_bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def day_bucket(now: Optional[datetime] = None) -> str:
    """The default dedup window: the UTC date. A clock value may be injected (the gate passes a fixed
    `now` so the test is deterministic)."""
    n = now or datetime.now(timezone.utc)
    return n.strftime("%Y-%m-%d")


# ── PURE: the anti-nag gate ─────────────────────────────────────────────────────────────────────

def is_category_muted(prefs: dict, category: str) -> bool:
    """Has the recipient turned this category OFF? A muted category suppresses the IN-APP row.
    PURE (takes the prefs dict)."""
    return category in set((prefs or {}).get("muted_categories") or [])


def in_quiet_hours(prefs: dict, now_hour: int) -> bool:
    """Is `now_hour` (0-23) inside the recipient's quiet-hours window? Handles a window that wraps
    midnight (e.g. 21→7). Returns False if no window is set. PURE.

    NOTE: quiet hours defer the EMAIL channel only — the in-app row is ALWAYS written (queued, not
    dropped), so the user still sees it in their inbox when they look. This function is consulted when
    computing the email_status, not whether to write the row."""
    qs, qe = (prefs or {}).get("quiet_start"), (prefs or {}).get("quiet_end")
    if qs is None or qe is None:
        return False
    qs, qe = int(qs), int(qe)
    if qs == qe:
        return False
    if qs < qe:
        return qs <= now_hour < qe
    # Wraps midnight: in the window if after start OR before end.
    return now_hour >= qs or now_hour < qe


def email_status_for(prefs: dict, category: str, now_hour: int, urgent: bool) -> str:
    """The outbox channel state to stamp on the row. v1 ships IN-APP ONLY, so this is always 'skipped'
    today (no transport) — but the LOGIC is here so turning email on is a one-line flip, not a
    redesign: a muted category or quiet-hours (non-urgent) would set 'skipped'/deferred, else 'pending'.
    For v1 we always return 'skipped' (documented seam). PURE."""
    # v1: no email transport — every row is in-app only.
    return "skipped"
    # When email lands, replace the line above with:
    #   if is_category_muted(prefs, category): return "skipped"
    #   if not urgent and in_quiet_hours(prefs, now_hour): return "skipped"  # or a 'deferred' state
    #   return "pending"


# ── PURE: the human text per event ─────────────────────────────────────────────────────────────

def render(event: str, ctx: dict) -> tuple[str, str, str]:
    """Render (category, title, body) for an event. ctx carries the human handles the route resolved
    (actor_email, vault_name, etc.). Verb+object, DESIGN.md voice (no em-dashes, no buzzwords). PURE.

    Falls back to a generic title for an unknown event so a new event type never crashes the fan-out."""
    actor = ctx.get("actor_email") or ctx.get("actor_id") or "Someone"
    vault = ctx.get("vault_name") or "a matter"
    cat = category_for(event)

    if event == "review.awaiting":
        return cat, "Review waiting for you", f"{actor} sent work on {vault} up for your review."
    if event == "review.approved":
        return cat, "Your work was approved", f"Your work on {vault} cleared review and is ready to release."
    if event == "review.changes_requested":
        note = (ctx.get("note") or "").strip()
        tail = f' "{note}"' if note else ""
        return cat, "Changes requested", f"{actor} asked for changes on your work on {vault}.{tail}"
    if event == "review.released":
        return cat, "Your work was released", f"Your work on {vault} was released externally."
    if event == "review.escalation":
        return cat, "Review still waiting", f"Work on {vault} is still waiting for your review. Please act when you can."
    if event == "matter.staffed":
        return "matter", "Added to a matter", f"You were added to {vault}. You have full access on this matter."
    if event == "answer.overridden":
        return "governance", "An abstain was overridden", f"{actor} overrode an abstain on {vault}."
    # Unknown event — generic, never crash.
    return cat, "Update", f"{actor} did something on {vault}."


# ── GLUE: emit one notification (best-effort, never raises) ──────────────────────────────────────

def emit(
    sb,
    *,
    recipient_id: Optional[str],
    firm_id: Optional[str],
    event: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    vault_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    ctx: Optional[dict] = None,
    urgent: bool = False,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """Write ONE in-app notification for `recipient_id`, applying the full data-path security + anti-nag
    pipeline. BEST-EFFORT: any failure is logged and swallowed (returns None) — this MUST NEVER break
    the originating action (same posture as log_audit). Returns the created row, or None if suppressed/
    deduped/failed/dormant.

    Pipeline (in order — each step is in the DATA PATH, not assumed):
      1. SELF-SKIP — you are not notified about your own action (recipient == actor).
      2. FIRM BOUNDARY (T3) — recipient must be a member of `firm_id` (db.user_in_firm).
      3. ETHICAL WALL (T5) — if `vault_id` set and the recipient is screened off it, write nothing.
      4. ANTI-NAG — a muted category suppresses the in-app row (quiet-hours only defers the email).
      5. DEDUP — a dedup_key + the UNIQUE index make a duplicate a clean no-op.
      6. WRITE — the row (email_status per the channel logic; 'skipped' in v1).
    """
    try:
        ctx = ctx or {}
        if not recipient_id or not event:
            return None
        # 1. Self-skip.
        if actor_id and str(recipient_id) == str(actor_id):
            return None
        # 2. Firm boundary (T3) — a cross-firm recipient gets nothing.
        if firm_id and not sb.user_in_firm(str(recipient_id), str(firm_id)):
            return None
        # 3. Ethical wall (T5, load-bearing) — a screened recipient is never told about the walled
        #    matter. is_vault_screened fails CLOSED on a live fault (treats a fault as screened),
        #    degrading to False only on a missing table (legacy parity).
        if vault_id:
            try:
                if sb.is_vault_screened(str(vault_id), str(recipient_id), str(firm_id) if firm_id else None):
                    return None
            except Exception as e:  # noqa: BLE001 — a wall lookup fault must NOT open the wall: skip.
                logger.warning("notify: wall lookup failed, suppressing to be safe: %s", e)
                return None
        # 4. Anti-nag — load prefs (defaults if none) and suppress a muted category's in-app row.
        prefs = sb.get_notification_preferences(str(recipient_id))
        category = category_for(event)
        if is_category_muted(prefs, category):
            return None
        # 5/6. Dedup + write.
        now_hour = (now or datetime.now(timezone.utc)).hour
        cat, title, body = render(event, ctx)
        row = {
            "firm_id": str(firm_id) if firm_id else None,
            "recipient_id": str(recipient_id),
            "actor_id": str(actor_id) if actor_id else None,
            "event": event,
            "category": category,
            "resource_type": resource_type,
            "resource_id": str(resource_id) if resource_id else None,
            "vault_id": str(vault_id) if vault_id else None,
            "title": title,
            "body": body,
            "dedup_key": build_dedup_key(str(recipient_id), event, resource_id, day_bucket(now)),
            "email_status": email_status_for(prefs, category, now_hour, urgent),
        }
        return sb.create_notification(row)
    except Exception as e:  # noqa: BLE001 — the whole point: a notify must never break the action.
        logger.warning("notify emit failed (non-fatal): %s", e)
        return None


# ── GLUE: fan out a REVIEW transition to the right person (the explicit-notify mapping) ──────────

def fan_out_review(
    sb,
    *,
    event: str,
    review_request: dict,
    firm_id: str,
    actor_id: str,
    ctx: Optional[dict] = None,
) -> Optional[dict]:
    """The review routes' convenience wrapper: pick the RIGHT recipient for a review transition, then
    emit. `event` is the route's notification event (NOT the audit action). `review_request` is the
    persisted row (carries current_owner, submitted_by, vault_id, id). Recipient mapping (Jeel's
    explicit-notify decision, D5):

        review.awaiting           → the new current_owner   ("X is awaiting your review" / next reviewer)
        review.approved           → the submitter           (their work cleared internal review)
        review.changes_requested  → the submitter           (work returned to revise)
        review.released           → the submitter           (their work went out)

    Best-effort via emit (never raises). The vault_id is passed so the wall + firm filters fire."""
    rr = review_request or {}
    vault_id = rr.get("vault_id")
    rid = rr.get("id")
    submitter = rr.get("submitted_by")
    current_owner = rr.get("current_owner")

    if event == "review.awaiting":
        recipient = current_owner
    elif event in ("review.approved", "review.changes_requested", "review.released"):
        recipient = submitter
    else:
        recipient = current_owner

    return emit(
        sb,
        recipient_id=str(recipient) if recipient else None,
        firm_id=firm_id,
        event=event,
        resource_type="review_request",
        resource_id=str(rid) if rid else None,
        vault_id=str(vault_id) if vault_id else None,
        actor_id=actor_id,
        ctx=ctx or {},
        urgent=False,
    )


# ── GLUE: the anti-stall escalation sweep (idempotent; Celery-beat is a documented seam) ─────────

def escalation_sweep(sb, *, threshold_hours: int = 48, now: Optional[datetime] = None) -> int:
    """Re-ping the current_owner of any OPEN review request they've been sitting on past
    `threshold_hours` — the anti-stall reminder. IDEMPOTENT: each emit's dedup_key is bucketed by DAY,
    so re-running the sweep the same day emits NOTHING new (the "last_notified_at" guarantee — no
    double-nag on a restart). Returns the number of reminders newly emitted.

    The notification is `urgent` (review.escalation) so it bypasses quiet-hours (a stall reminder must
    not be silently deferred). Runs system-side across all firms; the per-recipient firm + wall checks
    still fire inside emit, so a screened/cross-firm owner is never reminded.

    v1: this function ships + is gate-tested; the CELERY-BEAT registration that runs it on a schedule
    is a documented seam (commented in src/worker/celery_app.py), OFF until an email transport lands —
    the in-app inbox already shows the pending item, so nothing rots while the timer waits.
    """
    n = now or datetime.now(timezone.utc)
    from datetime import timedelta
    cutoff = (n - timedelta(hours=threshold_hours)).isoformat()
    emitted = 0
    try:
        rows = sb.open_review_requests_for_escalation(cutoff)
    except Exception as e:  # noqa: BLE001 — a sweep that can't read degrades to a no-op.
        logger.warning("escalation_sweep: selection failed (non-fatal): %s", e)
        return 0
    for rr in rows or []:
        owner = rr.get("current_owner")
        if not owner:
            continue
        created = emit(
            sb,
            recipient_id=str(owner),
            firm_id=str(rr.get("firm_id")) if rr.get("firm_id") else None,
            event="review.escalation",
            resource_type="review_request",
            resource_id=str(rr.get("id")) if rr.get("id") else None,
            vault_id=str(rr.get("vault_id")) if rr.get("vault_id") else None,
            actor_id=None,  # system reminder, no human actor
            ctx={"vault_name": rr.get("vault_name")},
            urgent=True,
            now=n,
        )
        if created is not None:
            emitted += 1
    return emitted
