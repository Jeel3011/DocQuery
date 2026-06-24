"""
DocQuery — Matter staffing + the REVIEW CHAIN of command  (F2e — THE PRODUCTIVITY ENGINE).

plans/F2_FIRM_CONSOLE_PLAN.md §F2e (D0/D3/D5). This is the slice that makes the product *increase*
productivity for the whole firm:

  - Matter staffing (D3): a senior (a `manage_matter_team` holder — partners + senior associates)
    staffs their team onto a matter. EVERYONE staffed gets the FULL working toolkit on that matter
    (D0 — explicitly NOT read-only). Removing a member is an INSTANT revoke (their next request loses
    matter access — T7).
  - The review chain (D5 — the heart): finished work flows UP a chain of command for review and, at
    the end, external release. The chain routes UP by role rank among the matter's members by default
    (zero setup), or follows the matter's custom chain. The anti-stall invariant: every review_request
    ALWAYS names a `current_owner` — nothing is ever ownerless (the #1 documented review failure).

SECURITY (the same posture F2a–d shipped): the VAULT is the path/body param, but the FIRM is resolved
SERVER-SIDE (T2/T3 — the body never carries a firm_id); every vault is asserted to belong to the
caller's firm (db.collection_in_firm) before a write. The ethical wall floor (assert_vault_not_screened)
is applied on the matter paths. Every staffing add/remove + every review transition is cap-checked
(authz.authorize) and audit-logged (T10 — the row F3 later hash-chains).

FLAG-OFF / pre-F2e byte-identical: empty tables ⇒ no staffing, no review requests ⇒ resolve_membership
adds an empty matter set ⇒ the rest of the app is unchanged. The migration is NOT applied live (Jeel
applies it); every db method degrades to an empty result when its table is absent — never a 500.
"""
import logging

from fastapi import APIRouter, HTTPException, Depends

from src.api.dependencies import (
    get_current_user, require_cap, resolve_membership, assert_vault_not_screened,
)
from src.api.schemas import (
    MatterTeamAddRequest, MatterTeamMember, MatterTeamResponse,
    ReviewSubmitRequest, ReviewDecisionRequest, ReviewRequestResponse, ReviewQueueResponse,
)
from src.api.routes.audit import log_audit
from src.components import authz
from src.components import review_chain as rc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Matters"])


# ─────────────────────────────────────────
# Helpers — resolve the matter's team as chain-builder input (server-side, firm-scoped).
# ─────────────────────────────────────────

def _matter_members(sb, firm_id: str, vault_id: str) -> list:
    """The matter's staffed team as review_chain.Member objects (user_id + firm role). Ensures the
    matter has a usable team for chain-building. Firm- + vault-scoped (resolved server-side)."""
    rows = sb.list_matter_members(firm_id, vault_id)
    return [rc.Member(user_id=str(r.get("user_id")), role=r.get("role") or "") for r in rows
            if r.get("user_id")]


def _review_response(row: dict, firm_id: str) -> ReviewRequestResponse:
    return ReviewRequestResponse(
        id=str(row.get("id")),
        firm_id=str(row.get("firm_id", firm_id)),
        vault_id=str(row.get("vault_id")),
        artifact_ref=row.get("artifact_ref", ""),
        submitted_by=str(row.get("submitted_by")),
        status=row.get("status", "pending"),
        current_owner=str(row.get("current_owner")) if row.get("current_owner") else None,
        chain=[str(c) for c in (row.get("chain") or [])],
        created_at=row.get("created_at"),
        decided_at=row.get("decided_at"),
    )


# ─────────────────────────────────────────
# MATTER STAFFING (D3)
#   `manage_matter_team` gates who may staff (partners + senior associates). The vault is the path
#   param; the firm is the CALLER's own (server-resolved — T3); the vault MUST belong to it (T2). A
#   staffed member gets the FULL toolkit on the matter (D0). Remove = instant revoke (T7). Audited.
# ─────────────────────────────────────────

@router.post("/matters/{vault_id}/team", response_model=MatterTeamResponse, status_code=201)
async def add_matter_team_member(
    vault_id: str,
    body: MatterTeamAddRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_matter_team")),
):
    """Staff a member onto a matter (D3). `require_cap("manage_matter_team")` 403s a non-staffer
    first. The firm is the caller's own (T3); the vault MUST belong to it (T2 — a guessed cross-firm
    vault is a 404) and the member MUST be in the firm. The added member immediately gets the FULL
    working toolkit on the matter (D0 — not read-only). Audited (T10)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    # T2/T3 (IDOR / cross-firm): the matter must belong to the caller's own firm, and the added
    # member must be in that firm — both resolved server-side; a body cannot point staffing elsewhere.
    if not sb.collection_in_firm(vault_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")
    if not sb.user_in_firm(body.user_id, firm_id):
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    # The ethical wall floor: a screened staffer cannot reach into a walled matter to staff it.
    assert_vault_not_screened(sb, vault_id)

    try:
        sb.add_matter_member(firm_id=firm_id, vault_id=vault_id,
                             user_id=body.user_id, added_by=sb.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # T10: staffing is a security/governance action — audit it.
    log_audit(sb, "matter.staff", "vault", str(vault_id),
              {"firm_id": str(firm_id), "added_user": body.user_id, "added_by": str(sb.user_id)})
    rows = sb.list_matter_members(firm_id, vault_id)
    return MatterTeamResponse(vault_id=str(vault_id), members=[
        MatterTeamMember(user_id=str(r.get("user_id")), role=r.get("role"),
                         added_by=str(r.get("added_by")) if r.get("added_by") else None,
                         created_at=r.get("created_at"))
        for r in rows
    ])


@router.delete("/matters/{vault_id}/team/{user_id}")
async def remove_matter_team_member(
    vault_id: str,
    user_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_matter_team")),
):
    """Un-staff a member from a matter — INSTANT, total revocation of matter access (their NEXT
    request loses it, T7). `require_cap("manage_matter_team")` gates it; firm- + vault-scoped (T2 —
    a guessed cross-firm vault touches nothing). Idempotent: removing a non-member is a clean 404,
    never a 500. Audited (T10)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    if not sb.collection_in_firm(vault_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")
    removed = sb.remove_matter_member(firm_id=firm_id, vault_id=vault_id, user_id=user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="That member is not staffed on this matter.")
    log_audit(sb, "matter.unstaff", "vault", str(vault_id),
              {"firm_id": str(firm_id), "removed_user": str(user_id)})
    return {"message": "Member removed from matter", "vault_id": vault_id, "user_id": user_id}


@router.get("/matters/{vault_id}/team", response_model=MatterTeamResponse)
async def list_matter_team(
    vault_id: str,
    sb=Depends(get_current_user),
):
    """List the matter's staffed team. Server-filtered to the caller's firm (T2). Any firm member may
    view the roster; staffing mutations are cap-gated above."""
    firm = sb.get_user_firm()
    firm_id = firm.get("id") if firm else None
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    if not sb.collection_in_firm(vault_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")
    rows = sb.list_matter_members(firm_id, vault_id)
    return MatterTeamResponse(vault_id=str(vault_id), members=[
        MatterTeamMember(user_id=str(r.get("user_id")), role=r.get("role"),
                         added_by=str(r.get("added_by")) if r.get("added_by") else None,
                         created_at=r.get("created_at"))
        for r in rows
    ])


# ─────────────────────────────────────────
# THE REVIEW CHAIN (D5) — submit · approve · request changes · release · my queue
#   A review_request is a state machine; current_owner is ALWAYS set in a non-terminal state (the
#   anti-stall invariant). Only the current_owner may advance/return; only a partner (release_external
#   cap) may release at the chain's end. Every transition is cap-checked + audited.
# ─────────────────────────────────────────

@router.post("/review", response_model=ReviewRequestResponse, status_code=201)
async def submit_for_review(
    body: ReviewSubmitRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("send_for_review")),
):
    """Send a piece of work UP the review chain (D5). `require_cap("send_for_review")` gates it
    (everyone on a matter holds it — D0). The firm is the caller's own (T3); the vault MUST belong to
    it (T2). The chain + the FIRST owner are computed SERVER-SIDE — the matter's custom chain if set,
    else up by role rank among the matter's members. The current_owner is ALWAYS set (anti-stall):
    the first reviewer, or the most-senior other member when no one is above the submitter. Audited."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    vault_id = body.collection_id
    if not sb.collection_in_firm(vault_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")
    # The ethical wall floor: a screened member cannot submit work on a walled matter.
    assert_vault_not_screened(sb, vault_id)

    members = _matter_members(sb, firm_id, vault_id)
    custom = sb.get_matter_review_chain(vault_id, firm_id)
    chain = rc.build_chain(members, submitted_by=sb.user_id, custom_chain=custom)
    owner = rc.first_owner(members, submitted_by=sb.user_id, custom_chain=custom)
    # ANTI-STALL: a request must NEVER be ownerless. If the matter has no other member to review
    # (a solo senior working their own matter), the submitter owns it (a one-person matter cannot
    # stall on someone else) — it can be released directly by a partner.
    if not owner:
        owner = sb.user_id

    req = sb.create_review_request(
        firm_id=firm_id, vault_id=vault_id, artifact_ref=body.artifact_ref,
        submitted_by=sb.user_id, current_owner=owner, chain=chain,
    )
    log_audit(sb, "review.submit", "review_request", str(req.get("id")),
              {"firm_id": str(firm_id), "vault_id": str(vault_id),
               "artifact_ref": body.artifact_ref, "current_owner": str(owner),
               "chain": chain})
    return _review_response(req, firm_id)


def _load_owned_request(sb, firm_id: str, request_id: str) -> dict:
    """Load a review request scoped to the firm and assert the CALLER is its current_owner. The
    anti-stall guard's other half: only the person who OWNS the next step may act on it. Raises 404
    for a cross-firm / unknown id (T2), 409 if already terminal, 403 if the caller isn't the owner."""
    req = sb.get_review_request(request_id, firm_id)
    if not req:
        raise HTTPException(status_code=404, detail="No such review request in your firm.")
    if req.get("status") == "released":
        raise HTTPException(status_code=409, detail="This review request is already released (terminal).")
    if str(req.get("current_owner")) != str(sb.user_id):
        raise HTTPException(status_code=403,
                            detail="Only the current owner of this review step may act on it.")
    return req


@router.post("/review/{request_id}/approve", response_model=ReviewRequestResponse)
async def approve_review(
    request_id: str,
    body: ReviewDecisionRequest = ReviewDecisionRequest(),
    sb=Depends(get_current_user),
    membership=Depends(require_cap("send_for_review")),
):
    """Approve the current review step → advance UP one owner in the chain (D5). Only the
    current_owner may approve (server-checked). When there is a NEXT reviewer, ownership moves to
    them (status stays `pending`). When the current owner is the LAST in the chain, the request
    becomes `approved` and ownership moves to the partner who may release_external at the end
    (chain_end_owner) — still a NAMED owner (anti-stall). Audited (T10)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    req = _load_owned_request(sb, firm_id, request_id)
    chain = [str(c) for c in (req.get("chain") or [])]

    nxt = rc.next_owner(chain, sb.user_id)
    if nxt:
        # More reviewers above: advance one step up, stay pending.
        new_status, new_owner = "pending", nxt
    else:
        # End of the internal chain: APPROVED, owned by the partner who may release externally.
        members = _matter_members(sb, firm_id, req.get("vault_id"))
        new_owner = rc.chain_end_owner(members, chain) or sb.user_id
        new_status = "approved"

    updated = sb.update_review_request(
        request_id, firm_id, status=new_status, current_owner=new_owner,
    )
    log_audit(sb, "review.approve", "review_request", str(request_id),
              {"firm_id": str(firm_id), "by": str(sb.user_id),
               "new_status": new_status, "new_owner": str(new_owner), "note": body.note})
    return _review_response(updated or {**req, "status": new_status, "current_owner": new_owner},
                            firm_id)


@router.post("/review/{request_id}/changes", response_model=ReviewRequestResponse)
async def request_changes(
    request_id: str,
    body: ReviewDecisionRequest = ReviewDecisionRequest(),
    sb=Depends(get_current_user),
    membership=Depends(require_cap("send_for_review")),
):
    """Request changes on the current review step → return the work to the SUBMITTER to revise (D5).
    Only the current_owner may do this. The request becomes `changes_requested` and ownership returns
    to `submitted_by` (still a NAMED owner — anti-stall; the ball is in the submitter's court, never
    nobody's). The submitter revises and re-submits. Audited (T10)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    req = _load_owned_request(sb, firm_id, request_id)
    submitter = str(req.get("submitted_by"))
    updated = sb.update_review_request(
        request_id, firm_id, status="changes_requested", current_owner=submitter,
    )
    log_audit(sb, "review.changes_requested", "review_request", str(request_id),
              {"firm_id": str(firm_id), "by": str(sb.user_id),
               "returned_to": submitter, "note": body.note})
    return _review_response(updated or {**req, "status": "changes_requested",
                                        "current_owner": submitter}, firm_id)


@router.post("/review/{request_id}/release", response_model=ReviewRequestResponse)
async def release_review(
    request_id: str,
    body: ReviewDecisionRequest = ReviewDecisionRequest(),
    sb=Depends(get_current_user),
    membership=Depends(require_cap("release_external")),
):
    """Release the reviewed work OUTSIDE the firm (the chain's terminal step — D5/ABA-512). This is
    the one tightly-held verb: `require_cap("release_external")` 403s anyone who can't release (only
    partners + senior associates hold it; the chain END is a PARTNER — can_release_external). The
    request MUST be `approved` (the chain cleared internal review) and the CALLER must be its
    current_owner. The ethical wall floor still applies (a screened partner can't release a walled
    matter). Becomes `released` (terminal); decided_at is set. Audited (T10 — the release record)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    req = sb.get_review_request(request_id, firm_id)
    if not req:
        raise HTTPException(status_code=404, detail="No such review request in your firm.")
    if req.get("status") == "released":
        raise HTTPException(status_code=409, detail="This review request is already released.")
    if req.get("status") != "approved":
        raise HTTPException(status_code=409,
                            detail="Work must be APPROVED through the review chain before external release.")
    if str(req.get("current_owner")) != str(sb.user_id):
        raise HTTPException(status_code=403,
                            detail="Only the current owner (the releasing partner) may release this work.")
    # The chain END is a PARTNER: the releaser must sit in the partner tier (defense in depth — the
    # release_external cap is also held by senior associates for their own/reviewed work, but the
    # chain's external-release endpoint is a partner; D5).
    if not rc.can_release_external(membership.role):
        raise HTTPException(status_code=403,
                            detail="External release at the end of the chain is a partner action.")
    # The ethical wall floor — a screened partner cannot release a walled matter (precedence, T6).
    assert_vault_not_screened(sb, req.get("vault_id"))

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = sb.update_review_request(
        request_id, firm_id, status="released", decided_at=now_iso,
    )
    log_audit(sb, "review.release", "review_request", str(request_id),
              {"firm_id": str(firm_id), "released_by": str(sb.user_id),
               "role": membership.role, "vault_id": str(req.get("vault_id")),
               "artifact_ref": req.get("artifact_ref"), "note": body.note})
    return _review_response(updated or {**req, "status": "released", "decided_at": now_iso}, firm_id)


@router.get("/review/queue", response_model=ReviewQueueResponse)
async def my_review_queue(
    sb=Depends(get_current_user),
):
    """My review queue — the OPEN requests I currently OWN (the anti-stall UX, D5: the system owns
    the next step, no one chases anyone). Server-filtered to the caller's firm (T2) + to requests
    where current_owner == me."""
    firm = sb.get_user_firm()
    firm_id = firm.get("id") if firm else None
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    rows = sb.list_my_review_queue(firm_id, owner_id=sb.user_id)
    return ReviewQueueResponse(requests=[_review_response(r, firm_id) for r in rows])
