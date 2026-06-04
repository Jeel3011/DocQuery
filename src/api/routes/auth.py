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
)
from src.api.dependencies import get_current_user, limiter
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
    """Register a new user account."""
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
