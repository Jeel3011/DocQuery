"""
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime


# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    email: str


class UserResponse(BaseModel):
    user_id: str
    email: str


# ─────────────────────────────────────────
# DOCUMENTS
# ─────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: Optional[str] = None
    status: str
    chunk_count: Optional[int] = 0
    file_size_bytes: Optional[int] = None
    created_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ─────────────────────────────────────────
# QUERY / CHAT
# ─────────────────────────────────────────

class SourceInfo(BaseModel):
    source_id: int
    filename: Optional[str] = None
    page: Optional[Any] = None
    chunk_type: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filename_filter: Optional[str] = None
    page_filter: Optional[int] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    num_sources_used: int


# ─────────────────────────────────────────
# CONVERSATIONS
# ─────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    title: str = "New Chat"


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int


# ─────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────

class SendMessageRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filename_filter: Optional[str] = None
    page_filter: Optional[int] = None


class MessageResponse(BaseModel):
    id: Optional[str] = None
    role: str
    content: str
    sources: Optional[List[dict]] = None
    created_at: Optional[str] = None


class MessageListResponse(BaseModel):
    messages: List[MessageResponse]
    conversation_id: str


# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
