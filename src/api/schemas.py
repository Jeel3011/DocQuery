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
    preferred_name: Optional[str] = None


class UpdatePreferencesRequest(BaseModel):
    # The display name the assistant should use. Length-capped; None clears it.
    # Server-side sanitisation strips control chars / prompt-injection markers.
    preferred_name: Optional[str] = Field(default=None, max_length=40)


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
    processing_progress: Optional[int] = 0   # C6: 0–100 ingest progress


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
    question: str = Field(..., min_length=1, max_length=2000)
    filename_filter: Optional[str] = None
    page_filter: Optional[int] = None
    conversation_id: Optional[str] = None
    collection_id: Optional[str] = None  # Phase 1: scope retrieval to a collection
    # DEPRECATED (2026-06-12): the sequential multi-hop loop was retired with the
    # law-first agent-core pivot (the agent's tool loop subsumes it). Field retained for
    # request-shape back-compat but IGNORED by the server.
    multi_hop: Optional[bool] = None
    # A4 (agent core §3.1): per-request mode for /query/agentcore/stream. None defaults
    # to "standard"; "deep" raises the step/wall/token budget. Ignored by other endpoints.
    mode: Optional[str] = None


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
    question: str = Field(..., min_length=1, max_length=2000)
    filename_filter: Optional[str] = None
    page_filter: Optional[int] = None
    collection_id: Optional[str] = None  # Phase 1: scope retrieval to a collection


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


# ─────────────────────────────────────────
# COLLECTIONS
# ─────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class UpdateCollectionRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_count: Optional[int] = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse]
    total: int


class AddDocumentToCollectionRequest(BaseModel):
    document_id: str
