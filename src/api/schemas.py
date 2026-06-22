"""
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
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
    # G2 Step F: G1d structural class (financial_filing|legal_contract|mixed|generic)
    # and coarse extraction-fidelity grade (good|partial). None = unknown/legacy → the
    # UI shows a neutral chip/dot. Persisted at ingest (migration 007).
    doc_type: Optional[str] = None
    fidelity: Optional[str] = None
    # G3 Step C: structurally-derived fiscal year (int). None = unknown/legacy → the FY
    # filter treats it as "don't exclude". Persisted at ingest (migration 008).
    fiscal_year: Optional[int] = None
    # F1e privilege firewall: true = attorney-client / work-product → excluded from shared /
    # cross-vault surfaces and watermarked in exports. Defaults false (migration 011); legacy
    # rows read false → byte-identical pre-F1e behavior.
    privileged: Optional[bool] = False


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class UpdateDocumentRequest(BaseModel):
    """F1e: mark/unmark a document as privileged (attorney-client / work-product). Only the
    privilege flag is mutable here; rename/other doc edits are out of scope for F1e."""
    privileged: bool


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
    # G3 Step E: active vault filter set (doc_type / fiscal_year). EXPLICIT in the request
    # (mirror G2 §9 #1 — never driven by a stale global store). Threaded into the
    # retriever's metadata_filter CONJUNCTIVELY — it NARROWS the vault scope, never
    # replaces it. Null/absent → no narrowing (the unfiltered vault).
    filters: Optional[dict] = None
    # G6.1 (drafting): when mode="draft", the deliverable type ("engagement memo",
    # "termination summary", …) and free-text instructions. The route composes them into
    # `question` so drafting rides the SAME loop — no new orchestrator. Ignored otherwise.
    doc_type: Optional[str] = Field(default=None, max_length=200)
    instructions: Optional[str] = Field(default=None, max_length=4000)
    # G8.7: knowledge-source chips — which authorities the agent may use this run. Subset of
    # {"vault","statutes","caselaw"}. The route GATES the backend from these: a source not
    # listed is not offered (its tool stripped) or not retrievable (instrument-type allow-
    # list), so a chip off ⇒ the agent CANNOT cite that source. None/absent ⇒ all enabled
    # (byte-identical to pre-G8.7). ("Web" is intentionally not a chip — agent-core has no
    # web tool; we don't ship a dead toggle.)
    sources: Optional[List[str]] = None


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

# F1a (plans/F1_VAULT_PLAN.md §1): a collection is now a typed MATTER/VAULT. These two closed
# sets MIRROR the CHECK constraints in docs/migrations/010_matter_vaults.sql — keep them in
# lockstep so a bad value is rejected the same way at the API boundary and the DB.
MATTER_KINDS = (
    "litigation", "m&a", "lending", "arbitration", "ip",
    "regulatory", "employment", "advisory", "compliance",
)
MATTER_STATUSES = ("active", "on_hold", "closed", "archived", "legal_hold")


class MatterParty(BaseModel):
    """A named party on a matter (for F1c's metadata-only conflict scan)."""
    name: str = Field(..., min_length=1, max_length=200)
    role: Optional[str] = Field(None, max_length=100)  # e.g. "client", "opposing", "counsel"


# ── F1c — practice-aware self-config (templates + metadata-only conflict scan) ────────────────

class TemplateColumnResponse(BaseModel):
    """One seed review-grid column from a practice template (maps 1:1 onto GridColumnRequest)."""
    key: str
    label: str
    prompt: str
    kind: str = "clause"
    risk_rubric: Optional[str] = None


class PracticeTemplateResponse(BaseModel):
    """The starting config a matter_kind suggests (F1c). All fields are DEFAULTS the user
    overrides — grid columns to seed a review grid, the default KB source chips, and which
    flagship surface to foreground. Served by GET /collections/practice-template/{matter_kind}.
    """
    matter_kind: Optional[str] = None
    label: str
    grid_columns: List[TemplateColumnResponse] = Field(default_factory=list)
    kb_scope: List[str] = Field(default_factory=list)   # subset of {vault,statutes,caselaw}
    flagship: str = "review_grid"
    summary: Optional[str] = None


class ConflictFinding(BaseModel):
    """One conflict-scan hit. `severity='adverse'` = a true ethical-wall collision (our side
    here vs the other side on another matter); `same_party` = an informational note (the firm
    already touches this name elsewhere). Carries only matter METADATA — no document content."""
    party: str
    matched_party: str
    collection_id: Optional[str] = None
    matter_name: Optional[str] = None
    matter_kind: Optional[str] = None
    severity: str                       # "adverse" | "same_party"
    new_side: Optional[str] = None      # "our" | "other" | "neutral"
    existing_side: Optional[str] = None


class ConflictScanRequest(BaseModel):
    """Pre-create ethical-wall screen (F1c). Parties only — metadata, never content. The scan
    compares these party names against the firm's OTHER matters' parties for adverse collisions.
    `exclude_collection_id` skips a matter being edited so it doesn't conflict with itself."""
    parties: List[MatterParty] = Field(default_factory=list)
    exclude_collection_id: Optional[str] = None


class ConflictScanResponse(BaseModel):
    """The scan result. Non-blocking: `conflicts` may be non-empty and the create still proceeds;
    the UI renders a banner. `has_adverse` is the headline the banner keys its tone off."""
    conflicts: List[ConflictFinding] = Field(default_factory=list)
    has_adverse: bool = False


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    # F1a matter fields — all optional ⇒ a pre-F1 client that omits them still creates a vault
    # (status defaults to 'active' at the DB), so the legacy create path is byte-identical.
    matter_kind: Optional[str] = None
    parties: Optional[List[MatterParty]] = None

    @field_validator("matter_kind")
    @classmethod
    def _kind_in_set(cls, v):
        if v is not None and v not in MATTER_KINDS:
            raise ValueError(f"matter_kind must be one of {MATTER_KINDS}")
        return v


class UpdateCollectionRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    # Lifecycle / typing updates (F1a). Each optional ⇒ a rename-only PATCH is unchanged.
    matter_kind: Optional[str] = None
    status: Optional[str] = None
    parties: Optional[List[MatterParty]] = None

    @field_validator("matter_kind")
    @classmethod
    def _kind_in_set(cls, v):
        if v is not None and v not in MATTER_KINDS:
            raise ValueError(f"matter_kind must be one of {MATTER_KINDS}")
        return v

    @field_validator("status")
    @classmethod
    def _status_in_set(cls, v):
        if v is not None and v not in MATTER_STATUSES:
            raise ValueError(f"status must be one of {MATTER_STATUSES}")
        return v


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_count: Optional[int] = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # F1a matter fields (legacy rows: matter_kind/firm_id None, status 'active', parties []).
    matter_kind: Optional[str] = None
    status: Optional[str] = "active"
    parties: Optional[List[MatterParty]] = None
    firm_id: Optional[str] = None
    # F1c: the metadata-only conflict-scan result, populated ONLY on create when parties were
    # supplied (None on list/get/patch and on a no-party create ⇒ legacy shape is unchanged).
    # Non-blocking — the create succeeds regardless; the UI shows a banner when present.
    conflicts: Optional[List[ConflictFinding]] = None
    has_adverse: Optional[bool] = None


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse]
    total: int


class AddDocumentToCollectionRequest(BaseModel):
    document_id: str


# ─────────────────────────────────────────
# REVIEW GRID (Phase B2) — N docs × M columns
# ─────────────────────────────────────────
class GridColumnRequest(BaseModel):
    """One column = one fact to extract from every document."""
    key: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=1000)
    kind: str = Field(default="clause")           # "clause" | "numeric"
    risk_rubric: Optional[str] = Field(default=None, max_length=500)


class ReviewGridRequest(BaseModel):
    """Create a review grid over a collection: pick documents + columns.

    `doc_ids` empty → use ALL documents in the collection. Columns are bounded to keep
    the N×M fan-out (and its cost) sane; the route enforces a hard cell ceiling too.
    """
    title: str = Field(default="Review grid", max_length=160)
    collection_id: str
    doc_ids: List[str] = Field(default_factory=list)
    columns: List[GridColumnRequest] = Field(..., min_length=1, max_length=12)
    # G3 Step E: active vault filter set (doc_type / fiscal_year). Same semantics as
    # QueryRequest.filters — CONJUNCTIVE narrowing of the review scope, never a replace.
    filters: Optional[dict] = None


# ─────────────────────────────────────────
# WORKFLOWS (Phase G7) — a template run over a vault
# ─────────────────────────────────────────
class WorkflowRunRequest(BaseModel):
    """Run an authored workflow template over a vault (collection).

    The template (resolved by `template_id` from the server-side registry) supplies the
    overlay / tool subset / columns / artifact shape; the request supplies only the SCOPE
    (collection + optional doc subset + filters) and the form `params` the template folds
    into its run. A report-shape template streams the agent-core events; a grid-shape one
    streams the review-grid events — both on the SAME engine, gated identically.
    """
    collection_id: str
    # The form values for the template's params_schema (text / multiselect / doc-picker…).
    params: dict = Field(default_factory=dict)
    # Optional row restriction for a grid-shape (fan-out) template; empty → all vault docs.
    doc_ids: List[str] = Field(default_factory=list)
    # G3 Step E: active vault filter set — CONJUNCTIVE narrowing of the run scope.
    filters: Optional[dict] = None
    # Optional conversation id so a report-shape run persists like an Ask answer.
    conversation_id: Optional[str] = None
