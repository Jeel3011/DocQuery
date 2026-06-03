"""
Run ledger — Phase 4 / §3.2 of the Brain plan.

Every Brain (and later harness) run records, per query, how many documents were
routed / read / produced evidence / failed, plus confidence and wall-clock time,
into the `runs` table (docs/migrations/006_brain_foundation.sql).

"Coverage is itself a sellable feature in regulated work." — CDB plan §3.

Writing the ledger is strictly best-effort: a failure here (e.g. the migration
hasn't been applied yet) must never break the user's answer, so every call is
wrapped in a non-fatal try/except.
"""

from __future__ import annotations

from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


def record_run(
    *,
    user_id: Optional[str],
    collection_id: Optional[str],
    conversation_id: Optional[str],
    query_text: str,
    run_type: str,
    docs_routed: int,
    docs_read: int,
    docs_relevant: int,
    docs_failed: int,
    confidence: float,
    wall_ms: int,
) -> None:
    """Insert a coverage-ledger row for a Brain/harness run.  Non-fatal."""
    if not user_id:
        return
    try:
        from src.components.db import SupabaseManager

        db = SupabaseManager(use_service_role=True)
        db._user = type("User", (), {"id": user_id})()
        db.client.table("runs").insert({
            "user_id": user_id,
            "collection_id": collection_id,
            "conversation_id": conversation_id,
            "query_text": (query_text or "")[:2000],
            "run_type": run_type,
            "docs_routed": docs_routed,
            "docs_read": docs_read,
            "docs_relevant": docs_relevant,
            "docs_failed": docs_failed,
            "confidence": confidence,
            "wall_ms": wall_ms,
        }).execute()
        logger.info(
            "[brain.ledger] run recorded: type=%s routed=%d read=%d relevant=%d failed=%d conf=%.2f %dms",
            run_type, docs_routed, docs_read, docs_relevant, docs_failed, confidence, wall_ms,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[brain.ledger] record_run failed (non-fatal): %s", exc)
