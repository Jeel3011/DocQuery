"""
Authentication endpoints — signup, login, logout, current user.
"""

import logging
import re
from fastapi import APIRouter, HTTPException, Depends, Request

from src.api.schemas import (
    SignUpRequest,
    SignInRequest,
    AuthResponse,
    UserResponse,
    UpdatePreferencesRequest,
    AcceptInviteRequest,
    FirmResponse,
    CapabilitiesResponse,
    RenameFirmRequest,
    BootstrapResponse,
    ConversationResponse,
)
from src.api.dependencies import get_current_user, limiter, resolve_membership, require_cap
from src.components.db import SupabaseManager
from src.api.routes.audit import log_audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth")


# Allowed name chars: letters (incl. accents/unicode word chars), spaces,
# apostrophes, hyphens, periods. Everything else (control chars, brackets,
# braces, backticks, newlines used in prompt-injection) is stripped. The name
# later flows into the LLM system prompt, so this is a security boundary.
_NAME_ALLOWED = re.compile(r"[^\w \.\'\-]", re.UNICODE)


def _sanitize_name(raw):
    """Return a safe, trimmed display name (≤40 chars) or None to clear it."""
    if raw is None:
        return None
    cleaned = _NAME_ALLOWED.sub("", str(raw))
    cleaned = " ".join(cleaned.split())  # collapse whitespace/newlines
    cleaned = cleaned[:40].strip()
    return cleaned or None


@router.post("/signup", response_model=AuthResponse)
@limiter.limit("5/minute")
async def signup(request: Request, body: SignUpRequest):
    """Register a new user account.

    F2a (D1): the first user CREATES a firm and becomes its Managing Partner. A solo signup is
    therefore a firm-of-one whose sole member is a MP — byte-identical powers to pre-F2 (the MP
    holds every capability). Additional members join ONLY by invite (never open auto-join — the
    named Auth0 security hole). Firm creation is best-effort: if it fails, the account still
    works (firm-less, backfilled by migration 012's idempotent block) — we never block signup on
    the firm write.
    """
    sb = SupabaseManager()
    try:
        res = sb.sign_up(body.email, body.password)
        session = getattr(res, "session", None)
        if not session:
            raise HTTPException(
                status_code=400,
                detail="Sign-up succeeded but no session returned. Check email for confirmation.",
            )
        sb._user = res.user
        log_audit(sb, "auth.signup", "user", str(res.user.id), {"email": res.user.email})

        # F2g onboarding. Two MUTUALLY-EXCLUSIVE paths:
        #   (a) invite_token present → JOIN the existing firm at the INVITED role (D1 — the joiner
        #       can't self-assign; email-bound, single-use, server-enforced in accept_invite, T4).
        #   (b) no token → CREATE the user's own firm; they become its Managing Partner (D1).
        #
        # AUDIT FIX #1 (fail-closed): a present-but-FAILING invite_token must NOT be swallowed and
        # fall through to creating a solo firm (the exact defect that let a wrong token mint a new
        # firm + MP). A rejected join raises 400; we do NOT create a firm in that case. The account
        # was already created by Supabase — the user re-tries the invite from the accept-invite page,
        # and the migration-012 backfill only ever provisions a solo firm for a user who arrived with
        # NO invite intent (path b), never as a fallback for a failed join.
        if body.invite_token:
            # AUDIT FIX #7: bind only against a CONFIRMED email. If Supabase email-confirmation is on,
            # res.user.email is unverified at signup; honoring the invite then would let someone claim
            # an invited address they don't control. Defer to the confirmed accept-invite flow.
            if not getattr(res.user, "email_confirmed_at", None):
                logger.info("Signup invite deferred (email unconfirmed) for %s", res.user.id)
                # Account exists; the stashed token is applied on the first CONFIRMED authed load.
            else:
                try:
                    result = sb.accept_invite(
                        raw_token=body.invite_token,
                        accepting_user_id=str(res.user.id),
                        accepting_email=res.user.email or "",
                    )
                    log_audit(sb, "firm.accept_invite", "firm", str(result.get("firm_id")),
                              {"role": result.get("role"), "user_id": str(res.user.id)})
                except ValueError as e:
                    # Expected rejection (expired / used / wrong email). FAIL CLOSED — never create a
                    # firm. The account stays firm-less; the user retries with a valid invite.
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception:
                    logger.exception("Invite-join failed at signup for %s", res.user.id)
                    raise HTTPException(status_code=400, detail="Could not accept the invite. Please try again.")
        else:
            try:
                local = (res.user.email or "").split("@")[0] or "My"
                name = (body.firm_name or "").strip() or f"{local}'s Firm"
                firm = sb.create_firm(name, owner_user_id=str(res.user.id))
                log_audit(sb, "firm.create", "firm", str(firm.get("id")),
                          {"name": firm.get("name"), "owner_role": "managing_partner"})
            except Exception:
                # Firm CREATE (path b) is best-effort — the backfill covers it; no security impact
                # (a firm-less user gets solo-MP over their OWN empty firm only, never another's).
                logger.exception("Firm provisioning failed for new user %s (account still created)", res.user.id)

        return AuthResponse(
            access_token=session.access_token,
            user_id=str(res.user.id),
            email=res.user.email,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Sign-up failed for %s", body.email)   # S8: full detail server-side only
        raise HTTPException(status_code=400, detail="Sign-up failed. Please check your details and try again.")


@router.post("/login", response_model=AuthResponse)
@limiter.limit("10/minute")
async def login(request: Request, body: SignInRequest):
    """Login with email and password. Returns an access token."""
    sb = SupabaseManager()
    try:
        res = sb.sign_in(body.email, body.password)
        sb._user = res.user
        log_audit(sb, "auth.login", "user", str(res.user.id), {"email": res.user.email})
        return AuthResponse(
            access_token=res.session.access_token,
            user_id=str(res.user.id),
            email=res.user.email,
        )
    except Exception as e:
        logger.exception("Login failed for %s", body.email)   # S8: full detail server-side only
        raise HTTPException(status_code=401, detail="Login failed. Check your email and password.")


@router.post("/logout")
async def logout(sb: SupabaseManager = Depends(get_current_user)):
    """Logout the current user (invalidates session server-side)."""
    try:
        sb.sign_out()
        return {"message": "Signed out successfully"}
    except Exception as e:
        logger.exception("Logout failed")
        raise HTTPException(status_code=500, detail="Logout failed. Please try again.")


@router.get("/me", response_model=UserResponse)
async def me(sb: SupabaseManager = Depends(get_current_user)):
    """Get the currently authenticated user's info."""
    return UserResponse(
        user_id=sb.user_id,
        email=sb.user_email,
        preferred_name=sb.preferred_name,
    )


@router.patch("/me/preferences", response_model=UserResponse)
@limiter.limit("20/minute")
async def update_preferences(
    request: Request,
    body: UpdatePreferencesRequest,
    sb: SupabaseManager = Depends(get_current_user),
):
    """Update the authenticated user's preferences (currently: preferred_name).

    Security: scoped to the verified token subject only (sb.user_id). The name is
    sanitised before being persisted because it later flows into the LLM prompt.
    """
    safe_name = _sanitize_name(body.preferred_name)
    try:
        sb.update_preferred_name(safe_name)
        log_audit(sb, "auth.update_preferences", "user", str(sb.user_id),
                  {"preferred_name_set": bool(safe_name)})
    except Exception:
        logger.exception("Failed to update preferences for user %s", sb.user_id)
        raise HTTPException(status_code=500, detail="Could not update preferences.")
    return UserResponse(
        user_id=sb.user_id,
        email=sb.user_email,
        preferred_name=safe_name,
    )


@router.get("/firm", response_model=FirmResponse)
async def my_firm(sb: SupabaseManager = Depends(get_current_user)):
    """The firm the current user belongs to, with their role in it (F2a). 404 if firm-less
    (a pre-backfill legacy user) — the frontend treats that as 'solo, unprovisioned'."""
    firm = sb.get_user_firm()
    if not firm or not firm.get("id"):
        raise HTTPException(status_code=404, detail="You are not a member of any firm.")
    return FirmResponse(id=firm["id"], name=firm.get("name"), role=firm.get("role"))


@router.patch("/firm", response_model=FirmResponse)
async def rename_my_firm(
    body: RenameFirmRequest,
    sb: SupabaseManager = Depends(get_current_user),
    membership=Depends(require_cap("manage_members")),
):
    """Rename the caller's firm (F2g onboarding — name the backfilled solo firm). Cap-gated
    `manage_members` (the firm-governance verb); the firm is the caller's own, resolved server-side
    (T3 — the body carries only the new name). Audited."""
    firm_id = membership.firm_id
    if not firm_id:
        raise HTTPException(status_code=403, detail="You are not a member of any firm.")
    updated = sb.rename_firm(firm_id, body.name)
    log_audit(sb, "firm.rename", "firm", str(firm_id), {"name": updated.get("name")})
    return FirmResponse(id=str(firm_id), name=updated.get("name"), role=membership.role)


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def my_capabilities(sb: SupabaseManager = Depends(get_current_user)):
    """The caller's server-resolved EFFECTIVE capability set (F2g surface 10 — caps source of
    truth). Read-only; no state, no migration. It is built from the SAME path require_cap trusts
    (resolve_membership → authz.caps_for_role + active delegations), so the frontend's render
    decisions can never drift from authz.py ROLE_CAPS. The route guard — not this payload — is the
    security: the UI uses this only to decide what to SHOW; every action re-checks server-side.

    Degrades to a solo Managing-Partner cap set for a firm-less/legacy user (byte-identical to
    pre-F2 — resolve_membership owns that fallback), so the console renders correctly for everyone.
    """
    m = resolve_membership(sb)
    # The EFFECTIVE cap set the UI renders on = the role's caps PLUS any active delegation grants
    # (D6 — a PA acting for a senior holds those verbs for the window). delegated_verbs is also
    # returned separately so the UI can mark a delegated capability as time-boxed. authorize() still
    # honors the same union at every action (caps OR delegated), so the payload mirrors the guard.
    effective = m.caps | m.delegated_verbs
    return CapabilitiesResponse(
        caps=sorted(effective),
        role=m.role,
        firm_id=m.firm_id,
        is_external=m.is_external,
        delegated_verbs=sorted(m.delegated_verbs),
    )


@router.get("/bootstrap", response_model=BootstrapResponse)
async def bootstrap(sb: SupabaseManager = Depends(get_current_user)):
    """One round-trip for the whole app shell (latency): user + caps + firm + collections +
    conversations. Replaces 4 separate mount calls, each of which re-verified the JWT and
    re-resolved membership. No new authority — every field comes from the SAME helpers the
    individual endpoints use; `resolve_membership` is memoized per request so caps + firm share
    one resolution. Firm-less/legacy users get firm=None (no error round-trip)."""
    from src.api.routes.collections import _collection_response

    m = resolve_membership(sb)  # memoized per request (caps + firm in one resolve)
    effective = m.caps | m.delegated_verbs
    caps = CapabilitiesResponse(
        caps=sorted(effective), role=m.role, firm_id=m.firm_id,
        is_external=m.is_external, delegated_verbs=sorted(m.delegated_verbs),
    )

    firm_resp = None
    firm = sb.get_user_firm()
    if firm and firm.get("id"):
        firm_resp = FirmResponse(id=firm["id"], name=firm.get("name"), role=firm.get("role"))

    # Collections + doc-counts in one batched query (no per-vault N+1).
    colls = sb.get_collections()
    counts = sb.batch_collection_doc_counts(colls)
    collections = [_collection_response(c, counts.get(str(c["id"]), 0)) for c in colls]

    convs = sb.get_conversations()
    conversations = [
        ConversationResponse(id=c["id"], title=c["title"],
                             created_at=c.get("created_at"), updated_at=c.get("updated_at"))
        for c in convs
    ]

    return BootstrapResponse(
        user=UserResponse(user_id=sb.user_id, email=sb.user_email,
                          preferred_name=sb.preferred_name),
        capabilities=caps,
        firm=firm_resp,
        collections=collections,
        conversations=conversations,
    )


@router.post("/accept-invite", response_model=FirmResponse)
@limiter.limit("10/minute")
async def accept_invite(
    request: Request,
    body: AcceptInviteRequest,
    sb: SupabaseManager = Depends(get_current_user),
):
    """Accept a firm invite (D1). The membership is created with the INVITED role — the joiner
    can never self-assign. Security (T4), all enforced server-side in db.accept_invite:
      - single-use & un-expired (atomic conditional UPDATE — no replay, no race);
      - EMAIL-BOUND: the accepting user's VERIFIED email must equal the invite email.
    Returns the firm + the role granted.
    """
    try:
        result = sb.accept_invite(
            raw_token=body.token,
            accepting_user_id=sb.user_id,
            accepting_email=sb.user_email or "",
        )
    except ValueError as e:
        # Expected rejections (used/expired/email-mismatch/invalid) → 400 with the safe reason.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Accept-invite failed for user %s", sb.user_id)
        raise HTTPException(status_code=500, detail="Could not accept the invite. Please try again.")

    firm_id = result.get("firm_id")
    log_audit(sb, "firm.accept_invite", "firm", str(firm_id),
              {"role": result.get("role"), "user_id": sb.user_id})
    firm = sb.get_user_firm(firm_id=firm_id)
    return FirmResponse(
        id=firm_id,
        name=firm.get("name") if firm else None,
        role=result.get("role"),
    )
