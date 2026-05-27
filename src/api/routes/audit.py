"""
DocQuery — Audit Trail (Phase 6)

Logs user actions for compliance tracking and security auditing.
Provides an API to query the audit log.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ─── Audit Logger ──────────────────────────────────────────────────────────────

def log_audit(
    sb,
    action: str,
    resource_type: str = None,
    resource_id: str = None,
    metadata: dict = None,
    ip_address: str = None,
):
    """Log an audit event. Non-blocking — failures are logged but not raised."""
    try:
        sb.client.table("audit_log").insert({
            "user_id": sb.user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
            "ip_address": ip_address,
        }).execute()
    except Exception as e:
        logger.warning("Audit log insert failed (non-fatal): %s", e)


# ─── Response Models ───────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    id: str
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Optional[dict] = None
    ip_address: Optional[str] = None
    created_at: Optional[str] = None


class AuditLogResponse(BaseModel):
    entries: List[AuditEntry]
    total: int
    page: int
    per_page: int


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/audit/log", response_model=AuditLogResponse)
async def get_audit_log(
    page: int = 1,
    per_page: int = 50,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    days: int = 30,
    sb=Depends(get_current_user),
):
    """Get paginated audit log for the current user.

    Args:
        page: Page number (1-indexed).
        per_page: Items per page (max 100).
        action: Filter by action type (e.g., 'document.upload').
        resource_type: Filter by resource type (e.g., 'document').
        days: How many days back to search (default: 30).
    """
    per_page = min(per_page, 100)
    offset = (page - 1) * per_page
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        query = sb.client.table("audit_log").select("*", count="exact").eq(
            "user_id", sb.user_id
        ).gte("created_at", since)

        if action:
            query = query.eq("action", action)
        if resource_type:
            query = query.eq("resource_type", resource_type)

        res = query.order(
            "created_at", desc=True
        ).range(offset, offset + per_page - 1).execute()

        entries = [
            AuditEntry(
                id=e["id"],
                action=e["action"],
                resource_type=e.get("resource_type"),
                resource_id=e.get("resource_id"),
                metadata=e.get("metadata"),
                ip_address=e.get("ip_address"),
                created_at=e.get("created_at"),
            )
            for e in (res.data or [])
        ]

        return AuditLogResponse(
            entries=entries,
            total=res.count or 0,
            page=page,
            per_page=per_page,
        )

    except Exception as e:
        logger.warning("Failed to fetch audit log: %s", e)
        return AuditLogResponse(entries=[], total=0, page=page, per_page=per_page)


@router.get("/audit/actions")
async def get_audit_actions(
    sb=Depends(get_current_user),
):
    """Get list of distinct action types in the user's audit log."""
    try:
        res = sb.client.rpc("get_distinct_actions", {"p_user_id": sb.user_id}).execute()
        return {"actions": res.data or []}
    except Exception:
        # Fallback: return known action types
        return {
            "actions": [
                "document.upload",
                "document.delete",
                "collection.create",
                "collection.delete",
                "collection.add_document",
                "conversation.create",
                "conversation.delete",
                "query.ask",
                "query.agentic",
                "export.conversation",
                "auth.login",
                "auth.signup",
            ]
        }
