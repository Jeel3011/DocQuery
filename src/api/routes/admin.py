"""
DocQuery — Admin API Routes (Phase 5)

Provides internal endpoints for triggering and reading RAGAS evaluation runs.
These endpoints require authentication (same Bearer token as all other routes)
but are intended for admin/owner use only.

Why this is differentiating:
- Very few portfolio RAG projects have a live evaluation API endpoint.
- Demonstrates eval-driven development: you don't ship model changes without
  running evals, and you can trigger evals programmatically via API.
- Harvey runs "BigLaw Bench" before every model deployment — this is the
  student-scale equivalent.
"""

import os
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends

from src.api.dependencies import get_current_user, require_cap, assert_vault_not_screened
from src.api.schemas import (
    InviteRequest,
    InviteResponse,
    InviteListResponse,
    MemberResponse,
    MemberListResponse,
    ScreenRequest,
    ScreenResponse,
    ScreenListResponse,
    ChangeRoleRequest,
    LifecycleResponse,
    DelegationRequest,
    DelegationResponse,
    DelegationListResponse,
    OverrideAbstainRequest,
    OverrideAbstainResponse,
)
from src.api.routes.audit import log_audit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# F2b: the interim `_require_manage_members` placeholder is REPLACED by the central
# `require_cap("manage_members")` (src/api/dependencies.py → authz.authorize). The guard 403s
# BEFORE the handler and yields the SERVER-RESOLVED membership, so the firm is taken from
# `membership.firm_id` (never a body firm_id — T3) and no second lookup is needed.

# Path to the default eval questions file (relative to project root)
_DEFAULT_EVAL_PATH = os.getenv("EVAL_QUESTIONS_PATH", "eval_questions_v2.json")
_DEFAULT_OUTPUT_PATH = os.getenv("EVAL_OUTPUT_PATH", "eval_results.json")


@router.post("/eval/run", status_code=202)
async def run_evaluation(
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """
    Trigger a RAGAS evaluation run asynchronously via Celery.

    The evaluation runs in the background so it does not block the API.
    Use GET /admin/eval/results to poll for the latest results.

    AUDIT FIX #5: this is an owner/admin operation — gated behind `require_cap("manage_members")`
    (was auth-only, so any authenticated user incl. an external client could trigger background
    jobs). The questions path is FIXED to the server default (the caller-supplied `questions_path`
    query param is removed) to close the path-traversal / arbitrary-file-read vector.
    """
    try:
        from src.worker.tasks import run_evaluation_task  # noqa: F401
        task = run_evaluation_task.delay(
            questions_path=_DEFAULT_EVAL_PATH,
            output_path=_DEFAULT_OUTPUT_PATH,
            user_id=sb.user_id,
        )
        return {
            "task_id": task.id,
            "status": "started",
            "message": "Evaluation started. Poll GET /api/v1/admin/eval/results for output.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start evaluation task: {exc}",
        )


@router.get("/eval/results")
async def get_eval_results(
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """
    Return the latest RAGAS evaluation results from eval_results.json.

    Honest NaN handling: metrics that returned NaN (e.g., faithfulness on
    formula-heavy text) are surfaced as null rather than silently zeroed.
    The response includes an 'average_note' field explaining any exclusions.
    """
    output_path = Path(_DEFAULT_OUTPUT_PATH)
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation results found. "
                "Run POST /api/v1/admin/eval/run first."
            ),
        )
    try:
        with open(output_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read evaluation results: {exc}",
        )


# ─────────────────────────────────────────
# FIRM MANAGEMENT — invites + members  (F2a)
#   Invite-only join (D1). The firm + caller role are resolved SERVER-SIDE (T3); the body never
#   carries a firm_id. Interim cap-guard is `_require_manage_members` (F2b swaps in require_cap).
# ─────────────────────────────────────────

@router.post("/firm/invites", response_model=InviteResponse, status_code=201)
async def create_firm_invite(
    body: InviteRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Invite a new member by email + role (D1). The joiner can never self-assign their role.

    Security: `require_cap("manage_members")` 403s before this runs (central authorize, F2b). The
    firm is the CALLER's own firm (membership.firm_id, resolved server-side — T3); a body-supplied
    firm is impossible (there is none). The INVITED role is additionally bounded by the T1
    self-escalation guard so an inviter can't mint a role ≥ their own. T4 storage properties
    (hashed/single-use/expiring/email-bound) are enforced in db.create_invite + db.accept_invite.
    The raw token is returned ONCE here.
    """
    from src.components import authz
    firm_id = membership.firm_id
    # T1: the invited role must clear the self-escalation guard (can't grant ≥ your own role).
    role_decision = authz.authorize(
        membership, "manage_members", authz.Scope(target_role=body.role, firm_id=firm_id)
    )
    if not role_decision.allow:
        raise HTTPException(status_code=403, detail=role_decision.reason)
    try:
        invite = sb.create_invite(
            firm_id=firm_id,
            email=str(body.email),
            role=body.role,
            invited_by=sb.user_id,
        )
    except Exception as exc:
        # Most likely a duplicate active invite (unique partial index) — a clean 409.
        logger.warning("Create invite failed for firm %s: %s", firm_id, exc)
        raise HTTPException(
            status_code=409,
            detail="An active invite for this email already exists, or the invite could not be created.",
        )
    log_audit(sb, "firm.invite", "firm", str(firm_id),
              {"email": str(body.email), "role": body.role})
    return InviteResponse(
        id=str(invite.get("id")),
        firm_id=str(invite.get("firm_id", firm_id)),
        email=invite.get("email", str(body.email)),
        role=invite.get("role", body.role),
        expires_at=invite.get("expires_at"),
        accepted_at=invite.get("accepted_at"),
        created_at=invite.get("created_at"),
        token=invite.get("token"),   # one-time; never stored, never re-fetchable
    )


@router.get("/firm/invites", response_model=InviteListResponse)
async def list_firm_invites(
    include_inactive: bool = False,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """List the firm's invites (pending by default). `require_cap("manage_members")` gates it;
    server-filtered to the caller's own firm (membership.firm_id — T2). Never returns the token
    or its hash."""
    firm_id = membership.firm_id
    rows = sb.list_invites(firm_id, include_inactive=include_inactive)
    return InviteListResponse(invites=[
        InviteResponse(
            id=str(r.get("id")),
            firm_id=str(r.get("firm_id", firm_id)),
            email=r.get("email"),
            role=r.get("role"),
            expires_at=r.get("expires_at"),
            accepted_at=r.get("accepted_at"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ])


@router.post("/firm/invites/{invite_id}/resend", response_model=InviteResponse)
async def resend_firm_invite(
    invite_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """ROTATE a pending invite's token and return the FRESH one-time token (the researched 'resend /
    copy link' recovery — the original is hash-stored and can never be re-shown, so we mint a new one
    and reset the expiry; the prior link dies on rotation). `require_cap("manage_members")` gates it;
    firm-scoped (membership.firm_id — T2/T3, the path id can't point at another firm's invite).
    Audited (T10). 404 if the invite is missing / already accepted / not in the caller's firm."""
    firm_id = membership.firm_id
    rotated = sb.resend_invite(invite_id=invite_id, firm_id=firm_id)
    if not rotated:
        raise HTTPException(status_code=404, detail="No pending invite to resend.")
    log_audit(sb, "firm.invite.resend", "firm", str(firm_id),
              {"invite_id": invite_id, "email": rotated.get("email")})
    return InviteResponse(
        id=str(rotated.get("id", invite_id)),
        firm_id=str(rotated.get("firm_id", firm_id)),
        email=rotated.get("email"),
        role=rotated.get("role"),
        expires_at=rotated.get("expires_at"),
        accepted_at=rotated.get("accepted_at"),
        created_at=rotated.get("created_at"),
        token=rotated.get("token"),   # the rotated one-time token
    )


@router.delete("/firm/invites/{invite_id}")
async def revoke_firm_invite(
    invite_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Revoke a pending invite. `require_cap("manage_members")` gates it; firm-scoped (T2/T3).
    Idempotent — revoking a missing/cross-firm/accepted invite is a clean 404, never a 500."""
    firm_id = membership.firm_id
    removed = sb.revoke_invite(invite_id=invite_id, firm_id=firm_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No pending invite to revoke.")
    log_audit(sb, "firm.invite.revoke", "firm", str(firm_id), {"invite_id": invite_id})
    return {"message": "Invite revoked", "invite_id": invite_id}


@router.get("/firm/members", response_model=MemberListResponse)
async def list_firm_members(sb=Depends(get_current_user)):
    """List the firm's members (user_id + role). Server-filtered to the caller's firm (T2).
    Any member may view the roster; mutations (promote/demote/remove) are F2d + cap-gated."""
    firm = sb.get_user_firm()
    firm_id = firm.get("id") if firm else None
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    rows = sb.list_members(firm_id)
    # Resolve a human email for each member (service-role admin auth API) so the console shows a
    # real name, not a raw user_id. Best-effort: an unresolved id falls back to None (the UI shows
    # a short id) — never a 500.
    emails = sb.resolve_member_emails([r.get("user_id") for r in rows])
    return MemberListResponse(members=[
        MemberResponse(
            user_id=str(r.get("user_id")),
            firm_id=str(r.get("firm_id", firm_id)),
            role=r.get("role"),
            email=emails.get(str(r.get("user_id"))),
            created_at=r.get("created_at"),
        )
        for r in rows
    ])


# ─────────────────────────────────────────
# ETHICAL WALLS — conflict screens  (F2c — THE REGULATORY MOAT)
#   A `manage_members` holder raises/lifts an ethical wall (screen a member off a matter). The
#   firm is resolved SERVER-SIDE (T3 — the body never carries a firm_id); the route validates the
#   screened user + vault both belong to the caller's firm (T2 IDOR) before writing. Every
#   add/remove is audit-logged (T10). The wall itself is enforced at three layers (require_cap +
#   the retrieval-layer floor `assert_vault_not_screened` + the row RLS backstop) — these routes
#   only POPULATE it.
# ─────────────────────────────────────────

@router.post("/firm/screens", response_model=ScreenResponse, status_code=201)
async def create_firm_screen(
    body: ScreenRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Raise an ethical wall: screen `user_id` off `vault_id` (D4 — matter-level). A reason is
    REQUIRED. `require_cap("manage_members")` 403s a non-manager before this runs. T2/T3: the
    firm is the caller's own (membership.firm_id, server-resolved); the screened user and the
    vault MUST both belong to that firm (validated below) — a body cannot point the wall at
    another firm's user/vault. The wall takes effect on the screened user's NEXT request (T7)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")

    # T2 (IDOR): both the screened member and the target vault must belong to the CALLER's firm.
    # Resolved server-side; a guessed cross-firm user_id/vault_id is rejected, not acted on.
    if not sb.user_in_firm(body.user_id, firm_id):
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    if not sb.collection_in_firm(body.vault_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")

    try:
        screen = sb.create_screen(
            firm_id=firm_id,
            user_id=body.user_id,
            vault_id=body.vault_id,
            reason=body.reason,
            created_by=sb.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Most likely a duplicate active screen (unique partial index) — a clean 409.
        logger.warning("Create screen failed for firm %s: %s", firm_id, exc)
        raise HTTPException(
            status_code=409,
            detail="An active screen for this member + matter already exists.",
        )

    # T10: the wall is a security action — audit it (the row F3 later hash-chains).
    log_audit(sb, "screen.create", "vault", str(body.vault_id),
              {"screened_user": body.user_id, "reason": body.reason, "firm_id": str(firm_id)})
    return ScreenResponse(
        id=str(screen.get("id")),
        firm_id=str(screen.get("firm_id", firm_id)),
        user_id=str(screen.get("user_id", body.user_id)),
        vault_id=str(screen.get("vault_id", body.vault_id)),
        reason=screen.get("reason", body.reason),
        created_by=str(screen.get("created_by")) if screen.get("created_by") else None,
        created_at=screen.get("created_at"),
        removed_at=screen.get("removed_at"),
    )


@router.delete("/firm/screens/{screen_id}")
async def remove_firm_screen(
    screen_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Lift an ethical wall (soft-remove — the audit history is preserved). `require_cap` gates it;
    scoped to the caller's own firm (T2 — a member of firm A can never lift firm B's wall, even
    with a guessed screen_id). Restores the screened member's access on their NEXT request (T7).
    Idempotent: a double-remove (or an unknown/foreign id) is a clean 404, never a 500."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    removed = sb.remove_screen(screen_id, firm_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No active screen with that id in your firm.")
    # T10: lifting a wall is also a security action — audit it.
    log_audit(sb, "screen.remove", "screen", str(screen_id),
              {"firm_id": str(firm_id), "vault_id": str(removed.get("vault_id"))})
    return {"message": "Screen removed", "screen_id": screen_id}


@router.get("/firm/screens", response_model=ScreenListResponse)
async def list_firm_screens(
    include_removed: bool = False,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """List the firm's ethical-wall screens (active by default; include_removed for the audit
    view). `require_cap("manage_members")` gates it; server-filtered to the caller's own firm
    (T2). A screened member cannot see their own screen here (they lack manage_members)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    rows = sb.list_screens(firm_id, include_removed=include_removed)
    return ScreenListResponse(screens=[
        ScreenResponse(
            id=str(r.get("id")),
            firm_id=str(r.get("firm_id", firm_id)),
            user_id=str(r.get("user_id")),
            vault_id=str(r.get("vault_id")),
            reason=r.get("reason"),
            created_by=str(r.get("created_by")) if r.get("created_by") else None,
            created_at=r.get("created_at"),
            removed_at=r.get("removed_at"),
        )
        for r in rows
    ])


# ─────────────────────────────────────────
# MEMBER LIFECYCLE — role change + offboard  (F2d — THE MOST BREACH-PRONE EVENTS)
#   The other half of the lifecycle (join is F2a). Each route resolves the TARGET's CURRENT role +
#   the firm's MP count SERVER-SIDE and builds a precise authz.Scope so the EXISTING
#   _manage_members_guard (last-MP + self-escalation, T1) makes the decision — the guard is proven
#   END-TO-END through the route, not just on the pure decision. The change takes effect on the
#   target's NEXT request (T7 — caps resolved per-request; no session cache to bust). Every event
#   is audit-logged (T10). Firm is the caller's own (membership.firm_id, server-resolved — T3).
# ─────────────────────────────────────────

@router.patch("/firm/members/{user_id}", response_model=LifecycleResponse)
async def change_member_role(
    user_id: str,
    body: ChangeRoleRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Promote/demote a member (T1/last-MP guarded END-TO-END). `require_cap("manage_members")` 403s
    a non-manager first. The firm is the caller's own (T3). We resolve the TARGET's CURRENT role and
    the firm's MP count SERVER-SIDE, build the authz.Scope, and re-run authz.authorize so the
    central guard denies: a self-escalation to ≥ the actor's rank (T1), and demoting the firm's LAST
    Managing Partner (last-MP guard). The change lands on the target's NEXT request (T7)."""
    from src.components import authz
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")

    # Resolve the TARGET server-side (T2/T3): their membership must be in the CALLER's own firm.
    target = sb.get_membership(user_id, firm_id)
    if not target:
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    current_role = target.get("role")
    # The last-MP guard needs the firm's current MP count (resolved server-side, no DB in authz).
    mp_count = sb.count_firm_role(firm_id, "managing_partner")

    # The CENTRAL decision (the seam the plan calls out): build a precise Scope so the EXISTING
    # _manage_members_guard does the work — self-escalation (target_role ≥ actor) + last-MP
    # (demoting the sole MP). is_self lets the guard know the actor targets themselves.
    scope = authz.Scope(
        firm_id=firm_id,
        target_role=body.role,
        target_current_role=current_role,
        mp_count=mp_count,
        is_self=(str(user_id) == str(sb.user_id)),
    )
    decision = authz.authorize(membership, "manage_members", scope)
    if not decision.allow:
        # T10: a blocked lifecycle attempt is a security event — audit the deny.
        log_audit(sb, "authz.deny", "membership", str(user_id),
                  {"role": membership.role, "attempted": body.role, "reason": decision.reason})
        raise HTTPException(status_code=403, detail=decision.reason)

    updated = sb.change_member_role(user_id, firm_id, body.role)
    if not updated:
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    # T10: the transition (old role → new role, by whom) is audited.
    log_audit(sb, "member.role_change", "membership", str(user_id),
              {"firm_id": str(firm_id), "old_role": current_role, "new_role": body.role})
    return LifecycleResponse(
        user_id=str(user_id), firm_id=str(firm_id), role=updated.get("role", body.role),
    )


@router.delete("/firm/members/{user_id}", response_model=LifecycleResponse)
async def offboard_member(
    user_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Offboard a member: INSTANT, TOTAL access revocation. `require_cap` 403s a non-manager first.
    We resolve the TARGET's CURRENT role + the firm MP count server-side and run the central
    _manage_members_guard via authz.authorize with is_removal=True — so removing the firm's LAST
    Managing Partner is DENIED end-to-end (the firm must never be left without an owner), as is a
    self-removal that would orphan the firm. On success we NEVER orphan the member's matters: their
    vaults are reassigned to the firm owner (an MP / the caller) BEFORE the membership is removed,
    and every delegation they hold or granted is revoked. The access loss lands on their NEXT
    request (T7 — caps resolved per-request). Audit-logged (T10)."""
    from src.components import authz
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")

    target = sb.get_membership(user_id, firm_id)
    if not target:
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    current_role = target.get("role")
    mp_count = sb.count_firm_role(firm_id, "managing_partner")

    # The CENTRAL last-MP guard, end-to-end: is_removal=True + the target's current role + mp_count.
    scope = authz.Scope(
        firm_id=firm_id,
        target_current_role=current_role,
        is_removal=True,
        mp_count=mp_count,
        is_self=(str(user_id) == str(sb.user_id)),
    )
    decision = authz.authorize(membership, "manage_members", scope)
    if not decision.allow:
        log_audit(sb, "authz.deny", "membership", str(user_id),
                  {"role": membership.role, "attempted": "offboard", "reason": decision.reason})
        raise HTTPException(status_code=403, detail=decision.reason)

    # Never orphan a matter (the resolved default): reassign the offboarded user's vaults to the
    # firm owner FIRST, so the matters always have an owner across the transition. The caller (a
    # manage_members holder, partner-tier) is the safe reassignment target.
    reassigned = sb.reassign_member_matters(user_id, firm_id, new_owner_id=sb.user_id)
    result = sb.remove_member(user_id, firm_id)
    if not result.get("removed"):
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    # T10: the offboard (who, what was reassigned/revoked) is audited.
    log_audit(sb, "member.offboard", "membership", str(user_id),
              {"firm_id": str(firm_id), "old_role": current_role,
               "matters_reassigned": reassigned,
               "delegations_revoked": result.get("delegations_revoked", 0)})
    return LifecycleResponse(
        user_id=str(user_id), firm_id=str(firm_id), removed=True,
        matters_reassigned=reassigned,
        delegations_revoked=result.get("delegations_revoked", 0),
    )


# ─────────────────────────────────────────
# AUTHORITY DELEGATION — time-boxed, revocable, bounded  (F2d / D6 — the PA grant)
#   A senior grants a bounded verb-set to a delegate. The delegator is the authenticated caller
#   (server-side — T3); a delegate can NEVER hold a verb the delegator lacks (re-bounded at READ
#   time in db.active_delegated_verbs — T1); and the grant STILL cannot beat a screen DENY
#   (precedence, asserted in the gate). Every grant/revoke is audit-logged (T10).
# ─────────────────────────────────────────

@router.post("/firm/delegations", response_model=DelegationResponse, status_code=201)
async def create_firm_delegation(
    body: DelegationRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_matter_team")),
):
    """Grant a bounded, time-boxed delegation (D6 — e.g. a partner lets a PA triage their queue).
    `require_cap("manage_matter_team")` gates who may delegate (partners + senior associates — the
    seniors who staff/run matters). The delegator is the AUTHENTICATED caller (server-side — a body
    cannot name another delegator, T3). T1: a delegator can only grant verbs THEY hold — we reject
    any requested verb outside the delegator's own caps up front (and the resolver re-bounds at read
    time too, so a later demotion can't leak it). The delegate must be in the caller's firm (T2)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    # T2: the delegate must be a member of the caller's own firm.
    if not sb.user_in_firm(body.delegate_id, firm_id):
        raise HTTPException(status_code=404, detail="That member is not in your firm.")
    # T1: a delegate can never be granted a verb the delegator does not hold. Reject up front with a
    # clear reason (the resolver also re-bounds at read time — defense in depth).
    over_reach = sorted(set(body.verbs) - set(membership.caps))
    if over_reach:
        raise HTTPException(
            status_code=403,
            detail=f"You cannot delegate verb(s) you do not hold: {over_reach}.",
        )
    try:
        deleg = sb.create_delegation(
            firm_id=firm_id,
            delegator_id=sb.user_id,
            delegate_id=body.delegate_id,
            verbs=body.verbs,
            expires_at=body.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # T10: a delegation grant is a security action — audit it.
    log_audit(sb, "delegation.grant", "firm", str(firm_id),
              {"delegate_id": body.delegate_id, "verbs": body.verbs, "expires_at": body.expires_at})
    return DelegationResponse(
        id=str(deleg.get("id")),
        firm_id=str(deleg.get("firm_id", firm_id)),
        delegator_id=str(deleg.get("delegator_id", sb.user_id)),
        delegate_id=str(deleg.get("delegate_id", body.delegate_id)),
        verbs=deleg.get("verbs", body.verbs),
        expires_at=deleg.get("expires_at", body.expires_at),
        revoked_at=deleg.get("revoked_at"),
        created_at=deleg.get("created_at"),
    )


@router.delete("/firm/delegations/{delegation_id}")
async def revoke_firm_delegation(
    delegation_id: str,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_matter_team")),
):
    """Revoke a delegation early (soft — history preserved). Firm-scoped (T2 — a guessed cross-firm
    delegation_id revokes nothing). Restores the delegate to role-only caps on their NEXT request
    (T7). Idempotent: a double-revoke / unknown id is a clean 404, never a 500."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    revoked = sb.revoke_delegation(delegation_id, firm_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="No active delegation with that id in your firm.")
    log_audit(sb, "delegation.revoke", "firm", str(firm_id),
              {"delegation_id": str(delegation_id), "delegate_id": str(revoked.get("delegate_id"))})
    return {"message": "Delegation revoked", "delegation_id": delegation_id}


@router.get("/firm/delegations", response_model=DelegationListResponse)
async def list_firm_delegations(
    include_inactive: bool = False,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("manage_matter_team")),
):
    """List the firm's delegations (active by default; include_inactive for the audit view).
    Server-filtered to the caller's own firm (T2)."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    rows = sb.list_delegations(firm_id, include_inactive=include_inactive)
    return DelegationListResponse(delegations=[
        DelegationResponse(
            id=str(r.get("id")),
            firm_id=str(r.get("firm_id", firm_id)),
            delegator_id=str(r.get("delegator_id")),
            delegate_id=str(r.get("delegate_id")),
            verbs=r.get("verbs", []),
            expires_at=r.get("expires_at"),
            revoked_at=r.get("revoked_at"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ])


# ─────────────────────────────────────────
# THE override_abstain MOMENT  (F2d / T6 — the high-trust event)
#   An override_abstain holder (role OR delegation) accepts an agent ABSTAIN over the gate's
#   objection. It REQUIRES a reason, is DENIED in a screened vault (precedence — the screen beats
#   the override grant, even for a partner), and writes a COMPLETE audit row (actor, answer ref,
#   reason, gate objection, timestamp) = the F3 hash-chain contract. The override verb is the
#   riskiest in the matrix — tightly gated, screen-beaten, never silent.
# ─────────────────────────────────────────

@router.post("/firm/answers/override", response_model=OverrideAbstainResponse, status_code=201)
async def override_abstain(
    body: OverrideAbstainRequest,
    sb=Depends(get_current_user),
    membership=Depends(require_cap("override_abstain")),
):
    """Override an agent abstain on a specific answer/artifact (the high-trust moment). Precedence,
    layer by layer:

      1. `require_cap("override_abstain")` 403s anyone without the verb (role OR active delegation) —
         a non-holder can never reach this handler (T6).
      2. The ethical-wall floor: `assert_vault_not_screened` 403s if the holder is SCREENED off the
         answer's vault — the screen DENY beats the override grant (precedence, T6), even for a
         partner. This is also re-checked through the central decision below (defense in depth).
      3. A reason is REQUIRED (schema-enforced) — the override must be justifiable to a regulator.
      4. A COMPLETE audit row is written (actor, answer ref, vault, reason, gate objection,
         timestamp) — the exact payload F3 hash-chains (non-repudiation).
    """
    from src.components import authz
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")

    # AUDIT FIX #6 (T2/T3): the vault MUST belong to the caller's OWN firm, resolved server-side —
    # exactly like the matters.py routes. Without this, a foreign collection_id passes the wall check
    # (the caller isn't screened off another firm's vault) and forges an override record against it,
    # polluting the F3 evidence ledger. A cross-firm / unknown vault is a 404.
    if not sb.collection_in_firm(body.collection_id, firm_id):
        raise HTTPException(status_code=404, detail="That matter is not in your firm.")

    # (2) the wall, in the DATA path: a screened holder cannot override in that vault (precedence).
    # assert_vault_not_screened resolves the screen server-side and 403s + audits a screen.block.
    assert_vault_not_screened(sb, body.collection_id)
    # Defense in depth: re-run the central decision with the vault in scope so the precedence
    # (screen step-1 DENY beats the override grant) is exercised through authorize() too.
    decision = authz.authorize(
        membership, "override_abstain", authz.Scope(vault_id=body.collection_id, firm_id=firm_id)
    )
    if not decision.allow:
        log_audit(sb, "authz.deny", "answer", str(body.answer_ref),
                  {"role": membership.role, "reason": decision.reason, "verb": "override_abstain"})
        raise HTTPException(status_code=403, detail=decision.reason)

    # (4) THE F3 HASH-CHAIN CONTRACT: a complete, non-repudiable record of the override. who / what
    # answer / which vault / why / what the gate objected to / when (created_at on the audit row).
    log_audit(sb, "answer.override_abstain", "answer", str(body.answer_ref),
              {"firm_id": str(firm_id),
               "collection_id": str(body.collection_id),
               "overridden_by": str(sb.user_id),
               "role": membership.role,
               "reason": body.reason,
               "gate_objection": body.gate_objection,
               "trust_state": "overridden"})
    return OverrideAbstainResponse(
        answer_ref=str(body.answer_ref),
        collection_id=str(body.collection_id),
        status="overridden",
        overridden_by=str(sb.user_id),
        reason=body.reason,
    )
