"""
DocQuery — Analytics API (Phase 4)

Provides usage statistics, performance metrics, and activity data
for the analytics dashboard.
"""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

from src.api.dependencies import get_current_user
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ─── Response Models ───────────────────────────────────────────────────────────

class DailyQueryCount(BaseModel):
    date: str
    count: int

class AnalyticsSummary(BaseModel):
    total_queries: int
    avg_latency_ms: Optional[float]
    cache_hit_rate: Optional[float]
    agentic_query_rate: Optional[float]
    web_search_rate: Optional[float]
    queries_today: int
    queries_this_week: int
    top_queries: List[dict]
    daily_queries: List[DailyQueryCount]

class UsageSummary(BaseModel):
    documents_count: int
    total_chunks: int
    collections_count: int
    conversations_count: int
    total_messages: int


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    days: int = 30,
    sb=Depends(get_current_user),
):
    """Get query analytics summary for the current user.

    Args:
        days: Number of days to look back (default: 30).
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Fetch query logs for the period
    try:
        res = sb.client.table("query_logs").select("*").eq(
            "user_id", sb.user_id
        ).gte("created_at", since).order(
            "created_at", desc=True
        ).execute()
        logs = res.data or []
    except Exception as e:
        logger.warning("Failed to fetch query_logs: %s", e)
        logs = []

    if not logs:
        return AnalyticsSummary(
            total_queries=0,
            avg_latency_ms=None,
            cache_hit_rate=None,
            agentic_query_rate=None,
            web_search_rate=None,
            queries_today=0,
            queries_this_week=0,
            top_queries=[],
            daily_queries=[],
        )

    total = len(logs)

    # Average latency
    latencies = [l["latency_ms"] for l in logs if l.get("latency_ms") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # Cache hit rate
    cache_hits = sum(1 for l in logs if l.get("cache_hit"))
    cache_rate = (cache_hits / total * 100) if total else None

    # Agentic rate
    agentic_count = sum(1 for l in logs if l.get("agentic"))
    agentic_rate = (agentic_count / total * 100) if total else None

    # Web search rate
    web_count = sum(1 for l in logs if l.get("web_search_used"))
    web_rate = (web_count / total * 100) if total else None

    # Today's queries
    today = datetime.now(timezone.utc).date()
    today_count = sum(1 for l in logs if l.get("created_at", "").startswith(str(today)))

    # This week's queries
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date()
    week_count = sum(
        1 for l in logs
        if l.get("created_at", "")[:10] >= str(week_ago)
    )

    # Top documents queried (from questions mentioning filenames)
    # We'll count by sources_count as a proxy
    doc_counts: dict[str, int] = {}
    for l in logs:
        q = l.get("question", "")[:50]
        doc_counts[q] = doc_counts.get(q, 0) + 1
    top_docs = sorted(
        [{"query": k, "count": v} for k, v in doc_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    # Daily query counts
    day_counts: dict[str, int] = {}
    for l in logs:
        day = l.get("created_at", "")[:10]
        if day:
            day_counts[day] = day_counts.get(day, 0) + 1

    daily = sorted(
        [DailyQueryCount(date=k, count=v) for k, v in day_counts.items()],
        key=lambda x: x.date,
    )

    return AnalyticsSummary(
        total_queries=total,
        avg_latency_ms=round(avg_latency, 1) if avg_latency else None,
        cache_hit_rate=round(cache_rate, 1) if cache_rate is not None else None,
        agentic_query_rate=round(agentic_rate, 1) if agentic_rate is not None else None,
        web_search_rate=round(web_rate, 1) if web_rate is not None else None,
        queries_today=today_count,
        queries_this_week=week_count,
        top_queries=top_docs,
        daily_queries=daily,
    )


@router.get("/analytics/usage", response_model=UsageSummary)
async def get_usage_summary(
    sb=Depends(get_current_user),
):
    """Get resource usage summary for the current user."""
    # Documents count
    try:
        docs_res = sb.client.table("documents").select("id", count="exact").eq(
            "user_id", sb.user_id
        ).execute()
        docs_count = docs_res.count or 0
    except Exception:
        docs_count = 0

    # Total chunks
    try:
        chunks_res = sb.client.table("document_chunks").select("id", count="exact").eq(
            "user_id", sb.user_id
        ).execute()
        chunks_count = chunks_res.count or 0
    except Exception:
        chunks_count = 0

    # Collections count
    try:
        colls_res = sb.client.table("collections").select("id", count="exact").eq(
            "user_id", sb.user_id
        ).execute()
        colls_count = colls_res.count or 0
    except Exception:
        colls_count = 0

    # Conversations count
    try:
        convs_res = sb.client.table("conversations").select("id", count="exact").eq(
            "user_id", sb.user_id
        ).execute()
        convs_count = convs_res.count or 0
    except Exception:
        convs_count = 0

    # Total messages
    try:
        msgs_res = sb.client.table("messages").select("id", count="exact").eq(
            "user_id", sb.user_id
        ).execute()
        msgs_count = msgs_res.count or 0
    except Exception:
        msgs_count = 0

    return UsageSummary(
        documents_count=docs_count,
        total_chunks=chunks_count,
        collections_count=colls_count,
        conversations_count=convs_count,
        total_messages=msgs_count,
    )
