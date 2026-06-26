"""F2k — DPDP data-principal rights + the erase/preserve distinction (the PURE core).

plans/F2_FIRM_CONSOLE_PLAN.md §F2k + F2_ARCHITECTURE.md §0.2/§6. This module is the pure, dependency-free
core of the DPDP rights slice — the retention floor, the erase-vs-preserve policy made explicit, the
90-day window arithmetic, and the export-payload shaping. It has NO Supabase / FastAPI / LLM dependency
(so the gate runs offline, $0); db.py owns the reads/writes, routes/dpdp.py owns the wiring + authz.

The three rights (DPDP Sections 11–13, Rules 2025):

  1. ACCESS / EXPORT (§11). A data principal may obtain a summary of their data + processing. We export
     their documents, conversations, messages, and audit (processing) rows. Server-scoped to the
     requester, or admin-initiated ON BEHALF (cap-gated manage_members, firm-boundary respected).

  2. CORRECTION / ERASURE within 90 days (§12), incl. consent-withdrawal. THE LOAD-BEARING DISTINCTION:
     erasure targets a person's personal CONTENT (documents / conversations / messages). It does NOT
     erase the immutable RECORDS OF PROCESSING — the audit_log (Rule 6.4/6.5, retained ≥ 1 year) and the
     F2i `signatures` hash-chain. Those are legal evidence + non-repudiation; deleting a signature to
     honor an erasure would break every OTHER firm member's sign-off in that chain and destroy the very
     integrity the chain exists to prove. So erasure is a SOFT-DELETE of content + a tombstone (the
     `data_erasures` ledger) that PROVES the erasure was honored without retaining the erased thing.
     See PRESERVED_RECORDS / ERASABLE_CONTENT below — this is the resolution of the classic
     "right-to-erasure vs immutable-audit-log" conflict, stated in one place so it can't drift.

  3. GRIEVANCE to a NAMED officer (§13), 90-day completion window. A tracked record carrying the officer
     identity captured at filing time + a due date (filed + 90d) so the firm can see what is overdue.

HONESTY (recorded here + in the runbook): the DPDP duties are the FIRM's (the Data Fiduciary); DocQuery
is the Processor that ENABLES them. We provide the find/export/erase/route machinery; the firm exercises
it. DPDP is enforceable ~mid-2027 — this ships the machinery now without blocking anything today.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# ── Rule 6.5 — the mandatory log-retention FLOOR (one year). Asserted, not assumed. ─────────────────
# DPDP Rule 6 requires access/processing logs be retained for AT LEAST ONE YEAR. We express the floor
# as a constant the gate pins; config.AUDIT_LOG_RETENTION_DAYS is the deployment's actual value and must
# be >= this. (A longer retention is fine; a shorter one is a compliance regression.)
MIN_LOG_RETENTION_DAYS: int = 365

# The §12 erasure / §13 grievance completion window (90 days), per the DPDP Rules.
RIGHTS_RESPONSE_DAYS: int = 90

# ── The erase/preserve policy, made explicit (the distinction, in data) ─────────────────────────────
# CONTENT that an erasure SOFT-DELETES (the data principal's personal data):
ERASABLE_CONTENT: tuple[str, ...] = ("documents", "conversations", "messages")
# RECORDS that an erasure DELIBERATELY PRESERVES (immutable records of processing / legal evidence):
#   - audit_log: Rule 6.4/6.5 processing record, retained >= 1 year (a record of WHAT WAS DONE, not the
#     personal content itself).
#   - signatures: the F2i per-firm hash-chain — non-repudiation + integrity; breaking it to erase one
#     person would invalidate every other member's sign-off and destroy the chain's whole purpose.
PRESERVED_RECORDS: tuple[str, ...] = ("audit_log", "signatures")

GRIEVANCE_STATUSES: frozenset[str] = frozenset({"open", "acknowledged", "resolved", "rejected"})


def assert_retention_floor(configured_days: int) -> None:
    """Raise if the deployment's configured audit-log retention is below the Rule 6.5 one-year floor.
    Called at app import (a misconfiguration that silently under-retains logs is a compliance breach we
    must catch loudly, not ship). The gate also asserts this directly over the constant + config."""
    if configured_days < MIN_LOG_RETENTION_DAYS:
        raise ValueError(
            f"AUDIT_LOG_RETENTION_DAYS={configured_days} is below the DPDP Rule 6.5 minimum of "
            f"{MIN_LOG_RETENTION_DAYS} days (one year). Logs of processing must be retained >= 1 year."
        )


def retention_ok(configured_days: int) -> bool:
    """Non-raising form of assert_retention_floor (for the UI / a health check)."""
    return configured_days >= MIN_LOG_RETENTION_DAYS


def window_due_at(created_at: Optional[datetime] = None) -> str:
    """The ISO due date for a rights request: filed + 90 days (§12/§13). `created_at` defaults to now
    (UTC). Returned as an ISO string so it drops straight into a DB row / a response."""
    base = created_at or datetime.now(timezone.utc)
    return (base + timedelta(days=RIGHTS_RESPONSE_DAYS)).isoformat()


def is_preserved(record_kind: str) -> bool:
    """True if `record_kind` is an immutable record of processing that an erasure must NOT touch. The
    single source of truth the erase path checks before deleting anything — a kind in PRESERVED_RECORDS
    is never soft-deleted, no matter what an erase request asks for."""
    return record_kind in PRESERVED_RECORDS


def assemble_export(
    *,
    principal_id: str,
    documents: list,
    conversations: list,
    messages: list,
    audit_rows: list,
    firm: Optional[dict] = None,
    generated_at: Optional[str] = None,
) -> dict:
    """Shape a data principal's access/export payload (§11) into a stable, self-describing structure.
    Pure: the caller (db.export_personal_data) supplies the rows; this only assembles + counts them, so
    the gate can assert the shape without a DB. Includes a `manifest` (counts) and the explicit note that
    processing records (audit) are INCLUDED as a summary of processing but are RETAINED, not erasable."""
    return {
        "data_principal": str(principal_id),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "firm": {"id": firm.get("id"), "name": firm.get("name")} if firm else None,
        "documents": documents or [],
        "conversations": conversations or [],
        "messages": messages or [],
        "processing_records": audit_rows or [],  # §11 "summary of processing" — RETAINED, not erasable
        "manifest": {
            "documents": len(documents or []),
            "conversations": len(conversations or []),
            "messages": len(messages or []),
            "processing_records": len(audit_rows or []),
        },
        "notes": {
            "access_right": "DPDP §11 — summary of personal data and processing.",
            "retention": (
                "Processing records (audit log) are retained for the legal period (>= 1 year, Rule "
                "6.5) and are records of WHAT WAS DONE, not erasable personal content."
            ),
        },
    }
