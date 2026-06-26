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
    # F2g onboarding (D1): a NEW firm's creator may NAME their firm (else a default is used). If an
    # `invite_token` is present instead, the user JOINS an existing firm at the invited role — no new
    # firm is created (the joiner never self-assigns a role; the token is email-bound, T4). Both are
    # optional so legacy signups are byte-identical (firm_name=None → default name; no token → create).
    firm_name: Optional[str] = Field(default=None, max_length=120)
    invite_token: Optional[str] = Field(default=None, min_length=1)


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
# FIRM / ROLES / INVITES  (F2a — plans/F2_FIRM_CONSOLE_PLAN.md §F2a)
# ─────────────────────────────────────────

# The 9 firm roles, top→bottom in hierarchy. In LOCKSTEP with the CHECK constraints in
# docs/migrations/012_firm_roles.sql AND the matrix in src/components/authz.py (F2b). A drift
# here = a role the API accepts but the DB rejects (or that has no capability line). `client`
# and `guest` are EXTERNAL (deny-by-default in F2b). `ROLE_RANK` orders them for the F2b
# self-escalation guard (T1: you can never set a role ≥ your own) and the F2e review chain.
ROLES = (
    "managing_partner", "senior_partner", "partner", "senior_associate",
    "associate", "paralegal", "assistant", "client", "guest",
)
# Lower index = more senior. Used by F2b (T1 guard) and F2e (default review chain by rank).
ROLE_RANK = {role: i for i, role in enumerate(ROLES)}


class InviteRequest(BaseModel):
    """A `manage_members` holder invites a new member by email + role (D1). The joiner can
    NEVER pick their own role; the firm is resolved server-side (T3 — never from the body)."""
    email: EmailStr
    role: str

    @field_validator("role")
    @classmethod
    def _role_in_set(cls, v):
        if v not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        return v


class AcceptInviteRequest(BaseModel):
    """Accept an invite via its single-use opaque token. The accepting user's VERIFIED email
    must equal the invite's email (email-binding enforced server-side, T4) — not trusted here."""
    token: str = Field(..., min_length=1)


class InviteResponse(BaseModel):
    """The created invite. `token` is returned ONCE at creation (the raw value is never stored —
    only its hash is — so it cannot be re-fetched); the caller delivers it to the invitee."""
    id: str
    firm_id: str
    email: str
    role: str
    expires_at: Optional[str] = None
    accepted_at: Optional[str] = None
    created_at: Optional[str] = None
    token: Optional[str] = None   # present only on the create response, never on a list


class InviteListResponse(BaseModel):
    invites: List[InviteResponse] = Field(default_factory=list)


class MemberResponse(BaseModel):
    """One firm member (user_id + role). Email is best-effort (resolved server-side from the
    auth user when available)."""
    user_id: str
    firm_id: str
    role: str
    email: Optional[str] = None
    created_at: Optional[str] = None


class MemberListResponse(BaseModel):
    members: List[MemberResponse] = Field(default_factory=list)


class FirmResponse(BaseModel):
    """The firm a user belongs to, with their role in it (F2a: get_user_firm now carries role)."""
    id: str
    name: str
    role: Optional[str] = None


class RenameFirmRequest(BaseModel):
    """Rename the caller's firm (F2g onboarding — name the backfilled solo firm). The firm is
    resolved SERVER-SIDE (T3 — never the body); cap-gated manage_members."""
    name: str = Field(..., min_length=1, max_length=120)


# ── F2c — ethical walls (conflict screens) ───────────────────────────────────────────────────
class ScreenRequest(BaseModel):
    """Raise an ethical wall: screen `user_id` off `vault_id`. A reason is REQUIRED (a wall must
    be justifiable to a regulator). The firm is resolved SERVER-SIDE (T3 — never from the body);
    `created_by` is the authenticated caller. The invited/screened user_id and vault_id are the
    conflict pair the route validates belong to the caller's firm before writing (T2)."""
    user_id: str = Field(..., min_length=1)
    vault_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class ScreenResponse(BaseModel):
    """One ethical-wall screen. `removed_at` is None for an active wall, a timestamp once lifted
    (soft-remove keeps the audit history)."""
    id: str
    firm_id: str
    user_id: str
    vault_id: str
    reason: str
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    removed_at: Optional[str] = None


class ScreenListResponse(BaseModel):
    screens: List[ScreenResponse] = Field(default_factory=list)


# ─────────────────────────────────────────
# MEMBER LIFECYCLE + DELEGATION + ABSTAIN-OVERRIDE  (F2d — plans/F2_FIRM_CONSOLE_PLAN.md §F2d)
#   The most breach-prone events. The TARGET's current role + the mp_count are resolved
#   SERVER-SIDE by the route (never the body, T2/T3) and fed into authz.Scope so the EXISTING
#   _manage_members_guard (last-MP + self-escalation, T1) does the work. Every event is audited.
# ─────────────────────────────────────────

class ChangeRoleRequest(BaseModel):
    """Promote/demote a member to `role` (D-lifecycle). The TARGET user is the path param; the
    firm + the target's CURRENT role + the firm's MP count are resolved SERVER-SIDE (T2/T3) — the
    body carries ONLY the new role. The T1 self-escalation + last-MP guards run in the route via
    authz.authorize over the resolved Scope; the joiner/target can never be promoted ≥ the actor."""
    role: str

    @field_validator("role")
    @classmethod
    def _role_in_set(cls, v):
        if v not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        return v


class LifecycleResponse(BaseModel):
    """The outcome of a role change / offboard. `matters_reassigned` is the count of vaults
    re-pointed to the firm/MP on offboard (never orphaned, §F2d); `delegations_revoked` is the
    count of grants lifted on offboard (instant total revocation, T7)."""
    user_id: str
    firm_id: str
    role: Optional[str] = None              # the NEW role after a change (None on offboard)
    removed: Optional[bool] = None          # True on a completed offboard
    matters_reassigned: Optional[int] = None
    delegations_revoked: Optional[int] = None


# ── F2d / D6 — authority delegation (time-boxed, revocable, bounded) ───────────────────────────
class DelegationRequest(BaseModel):
    """A senior delegates a bounded verb-set to a delegate (the PA, D6), until `expires_at`. The
    delegator is the authenticated caller (server-side — a body can't name another delegator, T3);
    `verbs` is the requested subset (the resolver re-bounds it to the delegator's own caps at read
    time, so a delegate can never exceed the delegator — T1). `expires_at` is REQUIRED (time-boxed)."""
    delegate_id: str = Field(..., min_length=1)
    verbs: List[str] = Field(..., min_length=1)
    expires_at: str = Field(..., min_length=1)

    @field_validator("verbs")
    @classmethod
    def _verbs_known(cls, v):
        # Importing CAPABILITIES at validate time keeps schemas lockstep with authz (a typo'd verb
        # is rejected at the boundary, not silently granted-and-ignored). Local import avoids a
        # module-load cycle (authz imports schemas for ROLES/ROLE_RANK).
        from src.components.authz import CAPABILITIES
        bad = [x for x in v if x not in CAPABILITIES]
        if bad:
            raise ValueError(f"unknown capability(ies): {bad}")
        return v


class DelegationResponse(BaseModel):
    id: str
    firm_id: str
    delegator_id: str
    delegate_id: str
    verbs: List[str] = Field(default_factory=list)
    expires_at: Optional[str] = None
    revoked_at: Optional[str] = None
    created_at: Optional[str] = None


class DelegationListResponse(BaseModel):
    delegations: List[DelegationResponse] = Field(default_factory=list)


# ── F2d / T6 — the abstain-override moment (the high-trust event) ──────────────────────────────
class OverrideAbstainRequest(BaseModel):
    """Override an agent ABSTAIN on a specific answer/artifact (T6 — the high-trust moment). A
    REASON is required (the override must be justifiable to a regulator — it is the F3 hash-chain
    contract). `collection_id` (the vault the answer belongs to) is resolved into the authz.Scope
    so a screen on that vault DENIES the override (precedence — screen beats the override grant,
    even for a partner). The answer reference identifies WHAT is being overridden (audited)."""
    answer_ref: str = Field(..., min_length=1)        # answer id / artifact ref being overridden
    collection_id: str = Field(..., min_length=1)     # the vault the answer belongs to (for the wall)
    reason: str = Field(..., min_length=1)
    gate_objection: Optional[str] = Field(default=None, max_length=2000)  # what the gate objected to


class OverrideAbstainResponse(BaseModel):
    """The override record (mirrors the audit row that F3 hash-chains). `status` flips to
    'overridden'; the full who/what/why is in the audit log (the non-repudiation trail)."""
    answer_ref: str
    collection_id: str
    status: str = "overridden"
    overridden_by: str
    reason: str
    created_at: Optional[str] = None


# ─────────────────────────────────────────
# MATTER STAFFING + the REVIEW CHAIN  (F2e — plans/F2_FIRM_CONSOLE_PLAN.md §F2e) — D0/D3/D5
#   THE PRODUCTIVITY ENGINE. The vault is the PATH param (not the body — T2/T3); the firm is
#   resolved server-side. Staffing grants the FULL toolkit on the matter (D0, never read-only);
#   the review chain flows work UP the chain of command (current_owner ALWAYS set = anti-stall).
# ─────────────────────────────────────────

# ── F2e / D3 — matter staffing ─────────────────────────────────────────────────────────────────
class MatterTeamAddRequest(BaseModel):
    """Staff a member onto a matter (D3). The VAULT is the path param; the firm is resolved
    server-side (T3 — never the body). The body carries ONLY who to add. The added member gets the
    FULL working toolkit on that matter (D0 — not read-only)."""
    user_id: str = Field(..., min_length=1)


class MatterTeamMember(BaseModel):
    """One staffed member of a matter (user_id + their firm role + who staffed them)."""
    user_id: str
    role: Optional[str] = None
    email: Optional[str] = None       # best-effort human handle (resolved server-side)
    added_by: Optional[str] = None
    created_at: Optional[str] = None


class MatterTeamResponse(BaseModel):
    """The matter's staffed team."""
    vault_id: str
    members: List[MatterTeamMember] = Field(default_factory=list)


# ── F2e / D5 — the review chain of command ─────────────────────────────────────────────────────
class ReviewSubmitRequest(BaseModel):
    """Send a piece of work UP the review chain (D5). The VAULT is resolved server-side from the
    body's `collection_id` (and asserted in the caller's firm — T3); `artifact_ref` identifies WHAT
    is under review. The chain + first owner are computed server-side (rank-default or the matter's
    custom chain) — the body never picks the reviewer (anti-tamper)."""
    collection_id: str = Field(..., min_length=1)     # the matter (vault) the work belongs to
    artifact_ref: str = Field(..., min_length=1)      # answer id / draft ref / artifact under review


class ReviewDecisionRequest(BaseModel):
    """Approve a review step (advance UP one owner) or request changes (return to the submitter). A
    `note` is optional context for the decision (audited). The request id is the path param; the
    actor must be the current_owner (server-checked)."""
    note: Optional[str] = Field(default=None, max_length=2000)


class ReviewRequestResponse(BaseModel):
    """A review request's current state. `current_owner` is ALWAYS set in a non-terminal state (the
    anti-stall invariant, D5); `status` ∈ {pending, approved, changes_requested, released}; `chain`
    is the ordered reviewer list the work routes through."""
    id: str
    firm_id: str
    vault_id: str
    artifact_ref: str
    submitted_by: str
    status: str
    current_owner: Optional[str] = None
    chain: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    decided_at: Optional[str] = None
    # F2i: the sign-off attached to this transition (internal approve / external release), if one was
    # produced. None on submit/changes and when the signatures table is unapplied (dormant).
    signature: Optional["SignatureResponse"] = None


class ReviewQueueResponse(BaseModel):
    """My review queue — the open requests I currently OWN (the anti-stall UX, D5)."""
    requests: List[ReviewRequestResponse] = Field(default_factory=list)


# ── F2j.1 — reviewable artifacts: let a reviewer SEE the submitted work ──────────────────────────
class ReviewArtifactResponse(BaseModel):
    """The card-preview of a review request's submitted work, for a reviewer who passes the
    review-read authority. `available=False` means the work was deleted after submission (graceful)."""
    available: bool = False
    title: Optional[str] = None
    question: Optional[str] = None        # the user's question that produced the answer
    answer_preview: Optional[str] = None  # first ~280 chars of the answer
    answer_full: Optional[str] = None     # the full answer text
    conversation_id: Optional[str] = None
    submitter_id: Optional[str] = None
    created_at: Optional[str] = None


class ReviewThreadMessage(BaseModel):
    """One turn in the read-only review thread."""
    role: str
    content: str
    sources: Optional[list] = None
    created_at: Optional[str] = None


class ReviewThreadResponse(BaseModel):
    """The whole conversation behind a submitted artifact, read-only — the reviewer's 'View full
    work'. `available=False` if the conversation was deleted."""
    available: bool = False
    title: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: List[ReviewThreadMessage] = Field(default_factory=list)


# ── F2g / surface 6 — set a matter's CUSTOM review chain ───────────────────────────────────────
class SetReviewChainRequest(BaseModel):
    """Set (or clear) a matter's custom review chain (D5 — "Customize review chain" for big
    matters). `chain` is an ordered list of reviewer user_ids the work routes through; `null` (or
    omitted) reverts to the rank default. The VAULT is the path param; the firm is resolved
    server-side (T3). The route validates every reviewer belongs to the caller's firm before the
    write (T2) — the body cannot route work to an outsider."""
    chain: Optional[List[str]] = None


# ── F2g / surface 10 — caps source of truth (the server-resolved effective cap set) ─────────────
class CapabilitiesResponse(BaseModel):
    """The caller's server-resolved effective capability set (F2g surface 10). Returned from the
    SAME path require_cap trusts (resolve_membership + authz.caps_for_role), so the UI's render
    decisions can NEVER drift from authz.py ROLE_CAPS. This payload only decides what RENDERS;
    every action still re-checks server-side (the route guard is the security, not this list).
      - caps: the effective verbs the caller may exercise (role caps ∪ delegated verbs).
      - role: the caller's resolved firm role (`managing_partner` for a legacy solo user).
      - firm_id: the active firm (None for a firm-less/legacy user).
      - is_external: a client/guest — blocked from the whole console (T8).
      - delegated_verbs: the subset of caps granted by an active delegation (time-boxed, D6)."""
    caps: List[str] = Field(default_factory=list)
    role: Optional[str] = None
    firm_id: Optional[str] = None
    is_external: bool = False
    delegated_verbs: List[str] = Field(default_factory=list)


# ── F2j — NOTIFICATIONS (in-app inbox + anti-nag preferences) ───────────────────────────────────
class NotificationResponse(BaseModel):
    """One in-app notification (the recipient's own — the route only ever returns sb.user_id's rows)."""
    id: str
    event: str
    category: str
    title: Optional[str] = None
    body: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    vault_id: Optional[str] = None
    read: bool = False
    created_at: Optional[str] = None


class NotificationListResponse(BaseModel):
    """The inbox payload: the recent notifications + the unread count (the bell badge)."""
    notifications: List[NotificationResponse] = Field(default_factory=list)
    unread: int = 0


class MarkReadRequest(BaseModel):
    """Mark notifications read. `ids` = the specific ones (still scoped to the caller's own rows
    server-side, T2/T8); omit / null ⇒ mark ALL the caller's unread."""
    ids: Optional[List[str]] = None


class NotificationPreferencesResponse(BaseModel):
    """The caller's anti-nag preferences (defaults when no row — "notify normally")."""
    muted_categories: List[str] = Field(default_factory=list)
    quiet_start: Optional[int] = None
    quiet_end: Optional[int] = None
    digest_mode: bool = False


class NotificationPreferencesRequest(BaseModel):
    """Update the caller's preferences (own row, server-scoped). Only the provided fields change.
    quiet_start/quiet_end are hours 0-23 (a window that may wrap midnight, e.g. 21→7); digest_mode
    batches low-signal events into a daily email summary (honored at the email layer)."""
    muted_categories: Optional[List[str]] = None
    quiet_start: Optional[int] = None
    quiet_end: Optional[int] = None
    digest_mode: Optional[bool] = None


# ── F2i — E-SIGNATURES (legally-valid sign-off, IT Act 2000) ─────────────────────────────────────

class SignatureResponse(BaseModel):
    """One tamper-evident, hash-chained sign-off (the secure-electronic-signature record, §85B). The
    hashes are returned so a client can independently verify, and surfaced in the release record."""
    id: str
    firm_id: str
    signer_id: str
    signer_name: Optional[str] = None
    artifact_type: str
    artifact_ref: str
    artifact_hash: str
    intent: str
    signature_method: str
    signed_at: Optional[str] = None
    chain_seq: Optional[int] = None
    content_hash: Optional[str] = None
    row_hash: Optional[str] = None
    note: Optional[str] = None   # e.g. "strong tier unavailable; signed at secure tier"


class SignatureListResponse(BaseModel):
    signatures: List[SignatureResponse] = Field(default_factory=list)


class ChainVerificationResponse(BaseModel):
    """The result of walking a firm's signature chain end-to-end (tamper / deletion / reorder check)."""
    ok: bool
    count: int
    first_broken_seq: Optional[int] = None
    reason: Optional[str] = None


# ── F2k — DPDP DATA-PRINCIPAL RIGHTS (access §11 / erasure §12 / grievance §13) ───────────────────
# The FIRM owns the DPDP duty (Data Fiduciary); these shapes ENABLE it. NONE carries a firm_id (T3 —
# the firm is resolved server-side); an admin on-behalf action names the PRINCIPAL (the subject), and
# the cap (manage_members) is checked in the route. See routes/dpdp.py + src/components/dpdp.py.

class DataExportResponse(BaseModel):
    """A data principal's access/export payload (§11): their content + a processing summary + a manifest.
    `processing_records` (audit) are INCLUDED as a §11 summary of processing but are RETAINED, not
    erasable — the export and the erase honor different scopes (see the erase distinction)."""
    data_principal: str
    generated_at: Optional[str] = None
    firm: Optional[dict] = None
    documents: List[dict] = Field(default_factory=list)
    conversations: List[dict] = Field(default_factory=list)
    messages: List[dict] = Field(default_factory=list)
    processing_records: List[dict] = Field(default_factory=list)
    manifest: dict = Field(default_factory=dict)
    notes: dict = Field(default_factory=dict)


class OnBehalfRequest(BaseModel):
    """An admin-initiated rights action ON BEHALF of another data principal (§11/§12). Cap-gated
    (manage_members) + firm-boundary checked in the route. Carries the SUBJECT (principal_id) and an
    optional reason — NEVER a firm_id (T3: the firm is the caller's own, resolved server-side)."""
    principal_id: str
    reason: Optional[str] = None


class ErasureResponse(BaseModel):
    """The result of a §12 erasure: counts of personal CONTENT soft-deleted, the records deliberately
    PRESERVED (the immutable audit log + the F2i signature chain), and the erasure-ledger proof id."""
    data_principal: str
    documents_erased: int = 0
    conversations_erased: int = 0
    messages_erased: int = 0
    preserved: List[str] = Field(default_factory=list)
    erasure_id: Optional[str] = None
    note: Optional[str] = None


class GrievanceOfficerRequest(BaseModel):
    """Name (or change) the firm's grievance officer (§13). Cap-gated (manage_members), firm
    server-resolved (T3 — no firm_id field). `user_id` links the officer to a firm member when they
    are one; name/email allow an external DPO too."""
    name: str
    email: Optional[str] = None
    user_id: Optional[str] = None


class GrievanceOfficerResponse(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    user_id: Optional[str] = None
    configured: bool = False


class GrievanceRequest(BaseModel):
    """File a §13 grievance. The complainant is the caller (server-side); the named officer is captured
    from the firm record at filing time. No firm_id (T3)."""
    subject: str


class GrievanceStatusRequest(BaseModel):
    """Action a grievance (acknowledge/resolve/reject), manager-gated. No firm_id (T3)."""
    status: str
    resolution_note: Optional[str] = None


class GrievanceResponse(BaseModel):
    id: str
    firm_id: str
    principal_id: str
    subject: str
    officer_name: Optional[str] = None
    officer_email: Optional[str] = None
    status: str
    resolution_note: Optional[str] = None
    created_at: Optional[str] = None
    due_at: Optional[str] = None
    resolved_at: Optional[str] = None


class GrievanceListResponse(BaseModel):
    grievances: List[GrievanceResponse] = Field(default_factory=list)
    officer: GrievanceOfficerResponse = Field(default_factory=GrievanceOfficerResponse)


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


# ─────────────────────────────────────────
# APP BOOTSTRAP (latency) — one round-trip for everything the app shell needs on mount.
# ─────────────────────────────────────────

class BootstrapResponse(BaseModel):
    """Everything the authenticated app shell needs on mount, in ONE request.

    Replaces the 4 independent calls the layout fired (getMe + getCapabilities + getFirm +
    listCollections + listConversations), each of which re-verified the JWT and re-resolved the
    per-request membership (4 sequential Supabase hops apiece). Bundling them shares ONE
    get_current_user + ONE memoized resolve_membership, cutting the mount waterfall.

    Purely a read aggregation — no new authority. Every field is built from the SAME server-side
    helpers the individual endpoints use, so the payload can't drift from them. `firm` is None for
    a firm-less/legacy user (the UI treats that as 'solo, unprovisioned'), exactly as GET /auth/firm
    returning 404 did — but without an error round-trip.
    """
    user: UserResponse
    capabilities: CapabilitiesResponse
    firm: Optional[FirmResponse] = None
    collections: List[CollectionResponse] = Field(default_factory=list)
    conversations: List[ConversationResponse] = Field(default_factory=list)
