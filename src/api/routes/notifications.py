"""
DocQuery — NOTIFICATIONS (F2j): the in-app inbox + anti-nag preferences.

plans/F2_FIRM_CONSOLE_PLAN.md §F2j. These routes expose a user's OWN notification inbox (the bell +
the dropdown) and their preferences. The FAN-OUT (who gets told about a review/staffing event) lives
in src/components/notifications.py, called from the event sources (matters.py / admin.py) — these
routes are purely the recipient's read/manage surface.

SECURITY: every route is server-scoped to `sb.user_id` — a user can ONLY read / mark / configure THEIR
OWN notifications (T2/T8). There is no IDOR surface (no other-user id ever flows in), so no `require_cap`
is needed: a notification inbox is personal, not a firm-governance action. The db methods AND-scope
every query by recipient_id == sb.user_id, so even a guessed notification id touches nothing.

DORMANT-ON-EMPTY: with migration 017 unapplied, every db method degrades to empty/None/defaults — so
these routes return an empty inbox / default prefs and NEVER 500 (byte-identical to pre-F2j).
"""
import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_current_user
from src.api.schemas import (
    NotificationResponse, NotificationListResponse, MarkReadRequest,
    NotificationPreferencesResponse, NotificationPreferencesRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Notifications"])


def _to_response(row: dict) -> NotificationResponse:
    return NotificationResponse(
        id=str(row.get("id")),
        event=row.get("event", ""),
        category=row.get("category", "review"),
        title=row.get("title"),
        body=row.get("body"),
        resource_type=row.get("resource_type"),
        resource_id=str(row.get("resource_id")) if row.get("resource_id") else None,
        vault_id=str(row.get("vault_id")) if row.get("vault_id") else None,
        read=bool(row.get("read_at")),
        created_at=row.get("created_at"),
    )


@router.get("/notifications", response_model=NotificationListResponse)
async def list_notifications(
    unread_only: bool = False,
    limit: int = 50,
    sb=Depends(get_current_user),
):
    """The caller's inbox (own rows only, newest first) + the unread count. `?unread_only=true` filters
    to unread. Dormant → empty + 0."""
    limit = max(1, min(limit, 100))
    rows = sb.list_notifications(recipient_id=sb.user_id, unread_only=unread_only, limit=limit)
    unread = sb.unread_notification_count(recipient_id=sb.user_id)
    return NotificationListResponse(
        notifications=[_to_response(r) for r in rows],
        unread=unread,
    )


@router.get("/notifications/unread-count")
async def unread_count(sb=Depends(get_current_user)):
    """The bell badge — the caller's unread count (cheap; polled by the FE). Dormant → 0."""
    return {"count": sb.unread_notification_count(recipient_id=sb.user_id)}


@router.post("/notifications/read")
async def mark_read(
    body: MarkReadRequest = MarkReadRequest(),
    sb=Depends(get_current_user),
):
    """Mark notifications read. `ids` = specific ones (AND-scoped to the caller's own rows server-side,
    so a guessed id touches nothing — T2/T8); omit ⇒ mark ALL the caller's unread. Returns the count
    updated. Dormant → 0."""
    updated = sb.mark_notifications_read(recipient_id=sb.user_id, ids=body.ids)
    return {"updated": updated}


@router.get("/notifications/preferences", response_model=NotificationPreferencesResponse)
async def get_preferences(sb=Depends(get_current_user)):
    """The caller's anti-nag preferences, or sane defaults when no row / the table is unapplied
    ("notify normally")."""
    prefs = sb.get_notification_preferences(user_id=sb.user_id)
    return NotificationPreferencesResponse(**prefs)


@router.put("/notifications/preferences", response_model=NotificationPreferencesResponse)
async def set_preferences(
    body: NotificationPreferencesRequest,
    sb=Depends(get_current_user),
):
    """Update the caller's preferences (own row, server-scoped). Only the fields the client SENT
    change — `exclude_unset` distinguishes an omitted field from one explicitly set to null. This is
    load-bearing: turning quiet hours OFF sends quiet_start/quiet_end = null, and those nulls MUST
    reach the DB to clear the window (a None-filter would silently drop them — the window could never
    be disabled)."""
    fields = body.model_dump(exclude_unset=True)
    prefs = sb.set_notification_preferences(user_id=sb.user_id, **fields)
    return NotificationPreferencesResponse(**prefs)
