"""
Authentication endpoints — signup, login, logout, current user.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Request

from src.api.schemas import (
    SignUpRequest,
    SignInRequest,
    AuthResponse,
    UserResponse,
)
from src.api.dependencies import get_current_user, limiter
from src.components.db import SupabaseManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth")


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
    )
