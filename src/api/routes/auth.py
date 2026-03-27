"""
Authentication endpoints — signup, login, logout, current user.
"""

from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import (
    SignUpRequest,
    SignInRequest,
    AuthResponse,
    UserResponse,
)
from src.api.dependencies import get_current_user
from src.components.db import SupabaseManager

router = APIRouter(prefix="/auth")


@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignUpRequest):
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
        raise HTTPException(status_code=400, detail=f"Sign-up failed: {str(e)}")


@router.post("/login", response_model=AuthResponse)
async def login(body: SignInRequest):
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
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")


@router.post("/logout")
async def logout(sb: SupabaseManager = Depends(get_current_user)):
    """Logout the current user (invalidates session server-side)."""
    try:
        sb.sign_out()
        return {"message": "Signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")


@router.get("/me", response_model=UserResponse)
async def me(sb: SupabaseManager = Depends(get_current_user)):
    """Get the currently authenticated user's info."""
    return UserResponse(
        user_id=sb.user_id,
        email=sb.user_email,
    )
