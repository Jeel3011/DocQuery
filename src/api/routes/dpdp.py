"""
DocQuery — DPDP data-principal rights + the named grievance officer  (F2k).

plans/F2_FIRM_CONSOLE_PLAN.md §F2k + F2_ARCHITECTURE.md §0.2/§6. The first-class data-principal rights
the DPDP Act 2023 + Rules 2025 require a Data Fiduciary (the firm) to honor — and which DocQuery (the
Data Processor) must ENABLE:

  - ACCESS / EXPORT (§11):  GET  /dpdp/export                      — the caller's OWN data
                            POST /dpdp/admin/export                — admin ON BEHALF of a principal
  - ERASURE within 90d (§12): POST /dpdp/erase                     — the caller's OWN content
                              POST /dpdp/admin/erase               — admin ON BEHALF
  - GRIEVANCE to a NAMED officer (§13):
                            POST /dpdp/grievances                  — file (routes to the named officer)
                            GET  /dpdp/grievances                  — mine (or the firm's, if a manager)
                            PATCH /dpdp/grievances/{id}            — action it (manager)
                            PUT  /dpdp/firm/grievance-officer      — name the officer (manager)
                            GET  /dpdp/firm/grievance-officer      — read the named officer
  - The §12 erasure ledger (compliance proof):
                            GET  /dpdp/admin/erasures              — the firm's erasure records (manager)

AUTH MODEL (the on-behalf decision, recommended default — cap-gated on-behalf):
  - A data principal may EXPORT / ERASE their OWN data with no special capability — it is THEIR data
    (the authenticated identity is the scope; no firm needed; a firm-less/legacy user works).
  - An ADMIN may do it ON BEHALF of another principal ONLY with `manage_members`, and ONLY within their
    own firm — the target principal MUST be a member of the caller's firm (T2/T3, resolved server-side;
    a guessed cross-firm principal_id is rejected). The on-behalf routes are SEPARATE so the cap-gate is
    explicit and a self-action can never silently widen to someone else's data.

THE ERASE DISTINCTION (load-bearing, documented in src/components/dpdp.py too): erasure SOFT-DELETES
personal CONTENT (documents/conversations/messages) and PRESERVES the immutable records of processing —
the audit_log (Rule 6.5, retained >= 1 year) AND the F2i `signatures` hash-chain. We never break a
signature to honor an erasure: it would invalidate every other member's sign-off and destroy the
non-repudiation the chain exists for. The chain still verifies end-to-end after an erasure.

SECURITY (the F2 posture): firm resolved SERVER-SIDE (T2/T3 — no body firm_id); every rights action
audit-logged (T10). HONESTY: the firm owns the DPDP duty; we enable it. DPDP enforceable ~mid-2027.

DORMANT-ON-EMPTY: migration 019 unapplied ⇒ export still returns the caller's own content, erase still
soft-deletes it (the ledger write is skipped), grievance reports "officer not configured" — never 500.
"""
import logging

from fastapi import APIRouter, HTTPException, Depends

from src.api.dependencies import get_current_user, require_cap, resolve_membership
from src.api.schemas import (
    DataExportResponse, OnBehalfRequest, ErasureResponse,
    GrievanceOfficerRequest, GrievanceOfficerResponse,
    GrievanceRequest, GrievanceStatusRequest, GrievanceResponse, GrievanceListResponse,
)
from src.api.routes.audit import log_audit
from src.components import dpdp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dpdp", tags=["DPDP"])


def _grievance_response(row: dict) -> GrievanceResponse:
    return GrievanceResponse(
        id=str(row.get("id")),
        firm_id=str(row.get("firm_id")) if row.get("firm_id") else "",
        principal_id=str(row.get("principal_id")) if row.get("principal_id") else "",
        subject=row.get("subject", ""),
        officer_name=row.get("officer_name"),
        officer_email=row.get("officer_email"),
        status=row.get("status", "open"),
        resolution_note=row.get("resolution_note"),
        created_at=row.get("created_at"),
        due_at=row.get("due_at"),
        resolved_at=row.get("resolved_at"),
    )


# ─── ACCESS / EXPORT (§11) ───────────────────────────────────────────────────────────────────────

@router.get("/export", response_model=DataExportResponse)
async def export_my_data(sb=Depends(get_current_user)):
    """Export the CALLER's own data (DPDP §11 — access). It is their data, so no capability is needed;
    the authenticated identity is the scope. Firm (if any) is resolved server-side for the manifest."""
    membership = resolve_membership(sb)
    firm = None
    if membership.firm_id:
        firm = {"id": membership.firm_id, "name": sb.get_user_firm().get("name")}
    payload = sb.export_personal_data(principal_id=sb.user_id, firm=firm)
    log_audit(sb, "dpdp.export", "data_principal", str(sb.user_id), {"on_behalf": False})
    return DataExportResponse(**payload)


@router.post("/admin/export", response_model=DataExportResponse)
async def export_on_behalf(
    body: OnBehalfRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Admin-initiated export ON BEHALF of another data principal (§11). Cap-gated (manage_members);
    the target MUST be a member of the caller's OWN firm (T2/T3 — server-resolved, a cross-firm
    principal_id is rejected)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    if not sb.user_in_firm(body.principal_id, firm_id):
        raise HTTPException(status_code=404, detail="That data principal is not in your firm.")
    firm = {"id": firm_id, "name": sb.get_user_firm().get("name")}
    payload = sb.export_personal_data(principal_id=body.principal_id, firm=firm)
    log_audit(sb, "dpdp.export", "data_principal", str(body.principal_id),
              {"on_behalf": True, "firm_id": str(firm_id)})
    return DataExportResponse(**payload)


# ─── ERASURE within 90 days (§12) ────────────────────────────────────────────────────────────────

@router.post("/erase", response_model=ErasureResponse)
async def erase_my_data(sb=Depends(get_current_user)):
    """Erase the CALLER's own personal CONTENT (§12 — correction/erasure + consent-withdrawal). SOFT:
    documents/conversations/messages are blanked; the immutable audit log + the F2i signature chain are
    PRESERVED (records of processing / legal evidence — see dpdp.PRESERVED_RECORDS). A tombstone proves
    the erasure was honored without retaining the content."""
    membership = resolve_membership(sb)
    result = sb.erase_personal_data(
        principal_id=sb.user_id, firm_id=membership.firm_id, requested_by=sb.user_id,
        reason="data-principal erasure request (DPDP §12)",
    )
    log_audit(sb, "dpdp.erase", "data_principal", str(sb.user_id),
              {"on_behalf": False, **{k: result[k] for k in ("documents", "conversations", "messages")}})
    return ErasureResponse(
        data_principal=str(sb.user_id),
        documents_erased=result["documents"], conversations_erased=result["conversations"],
        messages_erased=result["messages"], preserved=result["preserved"],
        erasure_id=str(result["erasure_id"]) if result.get("erasure_id") else None,
        note="Personal content erased; the audit log and signature chain are retained (records of "
             "processing / legal evidence).",
    )


@router.post("/admin/erase", response_model=ErasureResponse)
async def erase_on_behalf(
    body: OnBehalfRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Admin-initiated erasure ON BEHALF of a data principal (§12). Cap-gated (manage_members); the
    target MUST be in the caller's firm (T2/T3). Same erase/preserve distinction as the self path."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    if not sb.user_in_firm(body.principal_id, firm_id):
        raise HTTPException(status_code=404, detail="That data principal is not in your firm.")
    result = sb.erase_personal_data(
        principal_id=body.principal_id, firm_id=firm_id, requested_by=sb.user_id,
        reason=body.reason or "admin-initiated erasure on behalf of the data principal (DPDP §12)",
    )
    log_audit(sb, "dpdp.erase", "data_principal", str(body.principal_id),
              {"on_behalf": True, "firm_id": str(firm_id),
               **{k: result[k] for k in ("documents", "conversations", "messages")}})
    return ErasureResponse(
        data_principal=str(body.principal_id),
        documents_erased=result["documents"], conversations_erased=result["conversations"],
        messages_erased=result["messages"], preserved=result["preserved"],
        erasure_id=str(result["erasure_id"]) if result.get("erasure_id") else None,
        note="Personal content erased; the audit log and signature chain are retained.",
    )


@router.get("/admin/erasures")
async def list_firm_erasures(
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """The firm's erasure ledger (the §12 compliance proof). Manager-gated; firm server-resolved (T2)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    return {"erasures": sb.list_erasures(firm_id)}


# ─── GRIEVANCE to a NAMED officer (§13) ──────────────────────────────────────────────────────────

@router.put("/firm/grievance-officer", response_model=GrievanceOfficerResponse)
async def set_officer(
    body: GrievanceOfficerRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Name (or change) the firm's grievance officer (§13). Manager-gated; firm server-resolved (T3)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    if body.user_id and not sb.user_in_firm(body.user_id, firm_id):
        raise HTTPException(status_code=404, detail="That officer is not a member of your firm.")
    try:
        officer = sb.set_grievance_officer(firm_id, body.name, body.email, body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    log_audit(sb, "dpdp.grievance_officer.set", "firm", str(firm_id),
              {"officer_name": body.name, "officer_email": body.email})
    return GrievanceOfficerResponse(
        name=officer.get("name") or body.name, email=officer.get("email", body.email),
        user_id=officer.get("user_id", body.user_id), configured=bool(officer.get("name") or body.name),
    )


@router.get("/firm/grievance-officer", response_model=GrievanceOfficerResponse)
async def get_officer(sb=Depends(get_current_user)):
    """Read the firm's named grievance officer (§13 — a data principal must be told who to complain to).
    Any firm member may read it (it is the publicly-named contact, not a secret)."""
    membership = resolve_membership(sb)
    if not membership.firm_id:
        return GrievanceOfficerResponse(configured=False)
    officer = sb.get_grievance_officer(membership.firm_id)
    return GrievanceOfficerResponse(
        name=officer.get("name"), email=officer.get("email"), user_id=officer.get("user_id"),
        configured=bool(officer.get("name")),
    )


@router.post("/grievances", response_model=GrievanceResponse, status_code=201)
async def file_grievance(body: GrievanceRequest, sb=Depends(get_current_user)):
    """File a §13 grievance, ROUTED to the firm's named officer (captured at filing time). The
    complainant is the caller; firm resolved server-side (T3). A 90-day due date is stamped."""
    membership = resolve_membership(sb)
    if not membership.firm_id:
        raise HTTPException(
            status_code=400,
            detail="You are not a member of a firm, so there is no grievance officer to route to.",
        )
    try:
        row = sb.create_grievance(membership.firm_id, body.subject, principal_id=sb.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not row:
        raise HTTPException(
            status_code=503,
            detail="Grievance intake is not available yet (the data model is not applied).",
        )
    log_audit(sb, "dpdp.grievance.file", "firm", str(membership.firm_id),
              {"grievance_id": str(row.get("id")), "officer": row.get("officer_name")})
    return _grievance_response(row)


@router.get("/grievances", response_model=GrievanceListResponse)
async def list_grievances(sb=Depends(get_current_user)):
    """List grievances. A manager (manage_members) sees the FIRM's; everyone else sees only THEIR OWN
    (server-filtered — no cross-principal leak, T2). Also returns the named officer for display."""
    from src.components import authz
    membership = resolve_membership(sb)
    if not membership.firm_id:
        return GrievanceListResponse()
    is_manager = "manage_members" in (membership.caps or set())
    rows = sb.list_grievances(
        membership.firm_id, principal_id=None if is_manager else sb.user_id
    )
    officer = sb.get_grievance_officer(membership.firm_id)
    return GrievanceListResponse(
        grievances=[_grievance_response(r) for r in rows],
        officer=GrievanceOfficerResponse(
            name=officer.get("name"), email=officer.get("email"),
            user_id=officer.get("user_id"), configured=bool(officer.get("name")),
        ),
    )


@router.patch("/grievances/{grievance_id}", response_model=GrievanceResponse)
async def action_grievance(
    grievance_id: str,
    body: GrievanceStatusRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Action a grievance (acknowledge/resolve/reject). Manager-gated; firm-scoped (T2/T3 — a
    cross-firm id touches nothing)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    try:
        row = sb.update_grievance_status(grievance_id, firm_id, body.status, body.resolution_note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not row:
        raise HTTPException(status_code=404, detail="That grievance is not in your firm.")
    log_audit(sb, "dpdp.grievance.action", "firm", str(firm_id),
              {"grievance_id": grievance_id, "status": body.status})
    return _grievance_response(row)
