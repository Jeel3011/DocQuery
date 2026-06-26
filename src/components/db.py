"""
Supabase integration for DocQuery.
Handles: Auth, Storage (file uploads), Conversations, Messages, Documents metadata.

Place this file at: src/components/db.py
"""

import os
import threading
from datetime import datetime, timezone, timedelta
import tempfile
from pathlib import Path
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# B1/B7: the service-role client carries no per-user session state (identity is
# validated separately and scoped via SupabaseManager._user), so a single shared
# instance is safe and lets the underlying httpx connection pool be reused across
# requests instead of standing up a new client + TCP/TLS handshake every time.
_service_client: Optional[Client] = None
_service_client_lock = threading.Lock()


def _is_missing_relation(exc: Exception) -> bool:
    """True if `exc` looks like 'the table/relation does not exist' (Postgres 42P01 / PostgREST
    PGRST205 / 'could not find the table'). Used to distinguish a NOT-YET-APPLIED migration (safe to
    degrade to a legacy default) from a LIVE query fault (which must fail closed for the ethical
    wall — AUDIT FIX #3). Conservative: only the unambiguous 'missing table' signatures degrade."""
    s = str(exc).lower()
    return (
        "42p01" in s
        or "pgrst205" in s
        or "does not exist" in s
        or "could not find the table" in s
        or "relation" in s and "does not exist" in s
    )


def get_supabase_client(use_service_role: bool = False) -> Client:
    """Create a Supabase client.

    For server-side (FastAPI) operations that need to bypass RLS (file uploads,
    embedding saves etc.), pass use_service_role=True. This uses the
    SUPABASE_SERVICE_KEY which grants full access — validation of the user's
    identity is still done separately via auth.get_user(token).

    For client-side (Streamlit) sessions: use default (anon key). The anon client
    is NOT cached because Streamlit's sign_in mutates per-session auth state on it.
    """
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not anon_key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file")
    if use_service_role:
        if not service_key:
            raise ValueError("SUPABASE_SERVICE_KEY must be set in your .env file for server-side operations")
        global _service_client
        if _service_client is None:
            with _service_client_lock:
                if _service_client is None:
                    _service_client = create_client(url, service_key)
        return _service_client
    return create_client(url, anon_key)


def get_rls_read_client(access_token: str) -> Client:
    """F1 RLS hardening — a per-request, RLS-ENFORCED client for READS.

    The live request path runs on the service-role client (`get_supabase_client(True)`),
    which is exempt from Postgres RLS by design — so today the ONLY thing isolating one
    user's rows from another's is the app-layer `.eq("user_id", …)` filter on every query.
    That is a single layer: one forgotten filter is a cross-user leak (see the F1 isolation
    audit). This builds the data-layer backstop the plan calls for ("isolation must be an
    architecture decision at the data layer, not a query-time filter").

    Mechanism: an ANON-key client (so it is NOT service-role / NOT RLS-exempt) carrying the
    user's verified access token on the PostgREST `Authorization` header. PostgREST then runs
    every query as the `authenticated` role with `auth.uid()` = the token's subject, so the
    EXISTING `auth.uid() = user_id` RLS policies (already enabled on every table) enforce
    isolation — even if an app-layer filter is ever dropped. Reads only: writes/Storage stay
    on the service-role client (writes need the RLS-exempt path; Storage RLS is a separate F2
    concern). NOT cached/shared — each request carries a different JWT, so this is built per
    request (an anon client is cheap; the heavy RAG objects are cached elsewhere).
    """
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not anon_key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set for the RLS read client")
    client = create_client(url, anon_key)
    # Set the user JWT as the PostgREST Authorization header → auth.uid() resolves, RLS applies.
    client.postgrest.auth(access_token)
    return client


class SupabaseManager:
    """
    Central class for all Supabase operations in DocQuery.

    use_service_role=True: FastAPI backend — bypasses RLS for storage/DB writes.
    use_service_role=False: Streamlit frontend — uses anon key + cookie auth.
    """

    BUCKET = "docquery-files"

    def __init__(self, use_service_role: bool = False):
        self.client: Client = get_supabase_client(use_service_role=use_service_role)
        self._user = None  # set after login / after get_user() validation
        # F1 RLS hardening: when a request attaches the user's verified JWT (via
        # `attach_access_token`), `read_client` becomes an RLS-ENFORCED client so reads run
        # under `auth.uid()` and the existing policies are a real backstop. Until then it
        # falls back to `self.client` (service-role) — so the worker/ingest path and any
        # offline/test path are byte-identical. Built lazily, once, per manager.
        self._access_token: Optional[str] = None
        self._read_client: Optional[Client] = None

    def attach_access_token(self, token: Optional[str]) -> "SupabaseManager":
        """Attach the request's verified JWT so READS run RLS-enforced (`read_client`).

        Called by the FastAPI auth dependency after the token is validated. No-op for a
        falsy token (keeps the service-role fallback). Returns self for chaining."""
        if token:
            self._access_token = token
            self._read_client = None  # rebuilt on next access with the new token
        return self

    @property
    def read_client(self) -> Client:
        """The client to use for user-facing READS. RLS-enforced when a JWT is attached
        (auth.uid() resolves → existing `auth.uid() = user_id` policies apply, so a missing
        app-layer filter cannot leak); otherwise the service-role client (worker/offline).

        Writes and Storage must continue to use `self.client` (service-role) — they need the
        RLS-exempt path, and Storage RLS is a separate (F2) concern."""
        if not self._access_token:
            return self.client
        if self._read_client is None:
            self._read_client = get_rls_read_client(self._access_token)
        return self._read_client

    # ─────────────────────────────────────────
    # AUTH
    # ─────────────────────────────────────────

    def sign_up(self, email: str, password: str) -> dict:
        """Register a new user. Returns user dict or raises."""
        res = self.client.auth.sign_up({"email": email, "password": password})
        if res.user:
            self._user = res.user
        return res

    def sign_in(self, email: str, password: str) -> dict:
        """Login existing user. Returns session or raises."""
        res = self.client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            self._user = res.user
        return res

    def sign_out(self):
        self.client.auth.sign_out()
        self._user = None

    def get_user(self):
        """Return currently logged-in user, or None."""
        try:
            res = self.client.auth.get_user()
            self._user = res.user if res else None
            return self._user
        except Exception:
            return None

    @property
    def user_id(self) -> Optional[str]:
        return str(self._user.id) if self._user else None

    @property
    def user_email(self) -> Optional[str]:
        return self._user.email if self._user else None

    @property
    def preferred_name(self) -> Optional[str]:
        """The name the assistant should address the user by, read from the
        VERIFIED user object's user_metadata. Never sourced from a request body —
        only from the token Supabase already authenticated. Returns None if unset."""
        if not self._user:
            return None
        meta = getattr(self._user, "user_metadata", None) or {}
        if isinstance(meta, dict):
            name = meta.get("preferred_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    def update_preferred_name(self, name: Optional[str]) -> None:
        """Persist the user's preferred display name into Supabase user_metadata.

        Security:
        - Scoped strictly to self.user_id (the verified token subject) via the
          service-role admin API — a user can never write another user's metadata.
        - Requires a service-role client; refuses otherwise.
        - Caller is responsible for sanitising/validating `name` before this.
        """
        if not self.user_id:
            raise PermissionError("No authenticated user to update.")
        # admin.update_user_by_id is only available on the service-role client.
        self.client.auth.admin.update_user_by_id(
            self.user_id,
            {"user_metadata": {"preferred_name": name}},
        )
        # Reflect locally so the same request sees the new value.
        if self._user is not None:
            meta = dict(getattr(self._user, "user_metadata", None) or {})
            meta["preferred_name"] = name
            try:
                self._user.user_metadata = meta
            except Exception:
                pass

    # ─────────────────────────────────────────
    # STORAGE — File Upload / Download / Delete
    # ─────────────────────────────────────────

    def upload_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Upload file to Supabase Storage under user's folder.
        Returns the storage path (user_id/filename).
        """
        if not self.user_id:
            raise ValueError("User must be logged in to upload files.")
        storage_path = f"{self.user_id}/{filename}"
        self.client.storage.from_(self.BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"upsert": "true"},
        )
        return storage_path

    def upload_file_from_path(self, local_path: str, filename: str) -> str:
        """A2: Upload directly from a local file path so the client streams it,
        instead of reading the whole file into memory first. Returns the storage
        path (user_id/filename).
        """
        if not self.user_id:
            raise ValueError("User must be logged in to upload files.")
        storage_path = f"{self.user_id}/{filename}"
        self.client.storage.from_(self.BUCKET).upload(
            path=storage_path,
            file=local_path,
            file_options={"upsert": "true"},
        )
        return storage_path

    def download_file_to_temp(self, storage_path: str, suffix: str = "") -> str:
        """
        Download file from Supabase Storage to a local temp file.
        Returns the local temp file path so the pipeline can process it.
        Uses httpx directly with a generous timeout so large PDFs don't hit
        the Supabase client's default short read timeout.
        """
        import httpx
        url = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_KEY")
        download_url = f"{url}/storage/v1/object/{self.BUCKET}/{storage_path}"
        with httpx.Client(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            resp = client.get(
                download_url,
                headers={"Authorization": f"Bearer {service_key}"},
            )
            resp.raise_for_status()
            file_bytes = resp.content
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()
        return tmp.name

    # ─────────────────────────────────────────
    # DOCUMENT CHUNKS TABLE
    # ─────────────────────────────────────────

    def save_document_chunks(self, document_id: str, chunks: list) -> None:
        """Persist LangChain Document chunks to Supabase document_chunks table.

        Each chunk is stored as plain text + JSONB metadata — no pickle involved.
        Safe to call multiple times; existing chunks for the document_id are replaced.
        """
        if not self.user_id:
            raise ValueError("User must be logged in.")

        # Delete any previous chunks for this document so we get a clean re-index
        self.client.table("document_chunks").delete().eq(
            "document_id", document_id
        ).eq("user_id", self.user_id).execute()

        rows = [
            {
                "document_id": document_id,
                "user_id": self.user_id,
                "chunk_index": idx,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            for idx, chunk in enumerate(chunks)
        ]

        if rows:
            self.client.table("document_chunks").insert(rows).execute()

    def get_document_chunks(self, document_id: str) -> list:
        """Retrieve all stored chunks for a document (ordered by chunk_index)."""
        res = (
            self.client.table("document_chunks")
            .select("*")
            .eq("document_id", document_id)
            .eq("user_id", self.user_id)
            .order("chunk_index")
            .execute()
        )
        return res.data or []


    def delete_document_chunks(self, document_id: str) -> None:
        """Delete all stored chunks for a document (called on document deletion)."""
        if not self.user_id:
            return
        self.client.table("document_chunks").delete().eq(
            "document_id", document_id
        ).eq("user_id", self.user_id).execute()

    def delete_file(self, storage_path: str):
        """Delete a file from storage."""
        self.client.storage.from_(self.BUCKET).remove([storage_path])

    def list_user_files(self) -> list:
        """List all files uploaded by this user (excludes cache folder)."""
        if not self.user_id:
            return []
        try:
            items = self.client.storage.from_(self.BUCKET).list(self.user_id)
            return [
                item["name"] for item in items
                if item.get("name") and not item["name"].startswith("cache")
                and item.get("metadata")  # real files have metadata; folders don't
            ]
        except Exception:
            return []

    # ─────────────────────────────────────────
    # DOCUMENTS METADATA TABLE
    # ─────────────────────────────────────────

    def create_document_record(self, filename: str, storage_path: str,
                                file_type: str, file_size_bytes: int,
                                owner_user_id: str = None) -> dict:
        """Create a 'processing' document row.

        F2m (D0 — paralegal uploads INTO a shared matter): `owner_user_id` lets a staffed
        member's upload be stamped with the VAULT OWNER's user_id, so the doc + its chunks
        belong to the matter (the owner's namespace) and appear for the whole team — not
        orphaned in the uploader's own space. The caller MUST have authorized the owner via
        accessible_vault_owner first; this method just writes what it's told (service-role).
        Defaults to self.user_id (own upload) ⇒ byte-identical to the legacy path."""
        res = self.client.table("documents").insert({
            "user_id": owner_user_id or self.user_id,
            "filename": filename,
            "storage_path": storage_path,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "status": "processing",
        }).execute()
        return res.data[0] if res.data else {}

    def update_document_status(self, doc_id: str, status: str,
                               chunk_count: int = 0, progress_pct: int = None,
                               doc_type: str = None, fidelity: str = None,
                               fiscal_year: int = None):
        """Update document processing status.

        Args:
            doc_id: Document UUID.
            status: 'processing', 'ready', or 'failed'.
            chunk_count: Number of chunks (set when status='ready').
            progress_pct: Optional 0-100 progress percentage shown during processing.
            doc_type: G2 Step F — G1d structural class (financial_filing|legal_contract|
                mixed|generic), persisted when the doc flips to 'ready'.
            fidelity: G2 Step F — coarse extraction-fidelity grade (good|partial),
                persisted when the doc flips to 'ready'.
            fiscal_year: G3 Step C — structurally-derived fiscal year (int), persisted
                when the doc flips to 'ready'. None = unknown (never guessed).
        """
        update_data = {
            "status": status,
            "chunk_count": chunk_count,
        }
        # C6: persist the progress percentage so the UI can show real progress.
        if progress_pct is not None:
            update_data["processing_progress"] = max(0, min(100, int(progress_pct)))
        # G2 Step F: persist G1d class + fidelity grade so the doc table can show the
        # type chip + trust dot. These columns are added by migration 007; the retry
        # below keeps status updating if that migration isn't applied yet.
        if doc_type is not None:
            update_data["doc_type"] = doc_type
        if fidelity is not None:
            update_data["fidelity"] = fidelity
        # G3 Step C: persist the fiscal_year so the FY filter chip can read it (migration
        # 008). Forward-compat retry below drops it if the column isn't present yet.
        if fiscal_year is not None:
            update_data["fiscal_year"] = fiscal_year
        try:
            self.client.table("documents").update(
                update_data
            ).eq("id", doc_id).eq("user_id", self.user_id).execute()
        except Exception:
            # Forward-compat: if the newer columns aren't present yet (migration 005 /
            # 007 / 008 not applied), retry without them so status still updates.
            forward_compat_keys = ("processing_progress", "doc_type", "fidelity", "fiscal_year")
            if any(k in update_data for k in forward_compat_keys):
                for k in forward_compat_keys:
                    update_data.pop(k, None)
                self.client.table("documents").update(
                    update_data
                ).eq("id", doc_id).eq("user_id", self.user_id).execute()
            else:
                raise

    def set_document_privileged(self, doc_id: str, privileged: bool) -> dict:
        """F1e: mark/unmark a document as privileged (attorney-client / work-product).

        A WRITE → uses the service-role client scoped by `.eq(user_id)` for ownership (same
        pattern as update_document_status). Forward-compatible: if migration 011 (the
        `privileged` column) isn't applied yet, this raises a column-missing error the route
        translates — it never silently no-ops. Returns the updated row (or {} if not found)."""
        if not self.user_id:
            return {}
        res = self.client.table("documents").update(
            {"privileged": bool(privileged)}
        ).eq("id", doc_id).eq("user_id", self.user_id).execute()
        return res.data[0] if res.data else {}

    def get_user_documents(self) -> list:
        """Get all documents for current user."""
        if not self.user_id:
            return []
        res = self.read_client.table("documents").select("*").eq(
            "user_id", self.user_id
        ).order("created_at", desc=True).execute()
        return res.data or []

    def get_document(self, doc_id: str) -> dict:
        """Get a document record by ID for current user."""
        if not self.user_id:
            return {}
        res = self.read_client.table("documents").select("*").eq(
            "user_id", self.user_id
        ).eq("id", doc_id).execute()
        return res.data[0] if res.data else {}

    def delete_document_record(self, doc_id: str):
        """Delete document record by ID for current user."""
        self.client.table("documents").delete().eq(
            "user_id", self.user_id
        ).eq("id", doc_id).execute()

    # ─────────────────────────────────────────
    # CONVERSATIONS (THREADS)
    # ─────────────────────────────────────────

    def create_conversation(self, title: str = "New Chat") -> dict:
        res = self.client.table("conversations").insert({
            "user_id": self.user_id,
            "title": title,
        }).execute()
        return res.data[0] if res.data else {}

    def get_conversations(self) -> list:
        """Get all threads for current user, newest first."""
        if not self.user_id:
            return []
        res = self.read_client.table("conversations").select("*").eq(
            "user_id", self.user_id
        ).order("updated_at", desc=True).execute()
        return res.data or []

    def rename_conversation(self, conversation_id: str, new_title: str):
        if not self.user_id:
            return
        self.client.table("conversations").update({
            "title": new_title
        }).eq("id", conversation_id).eq("user_id", self.user_id).execute()

    def delete_conversation(self, conversation_id: str):
        # Messages are deleted by CASCADE in DB
        if not self.user_id:
            return
        self.client.table("conversations").delete().eq(
            "id", conversation_id
        ).eq("user_id", self.user_id).execute()

    # ─────────────────────────────────────────
    # MESSAGES
    # ─────────────────────────────────────────

    def save_message(self, conversation_id: str, role: str,
                     content: str, sources: list = None) -> dict:
        """Persist a single message to the DB."""
        res = self.client.table("messages").insert({
            "conversation_id": conversation_id,
            "user_id": self.user_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        }).execute()
        # Bump conversation updated_at so it rises to top of list (B3 fix: use real timestamp)
        self.client.table("conversations").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", conversation_id).execute()
        return res.data[0] if res.data else {}

    def save_messages(self, conversation_id: str, items: list) -> list:
        """B4: bulk-insert several messages in one round-trip, bumping the
        conversation's updated_at once (vs. one INSERT + one UPDATE per message).

        Each item is a dict with keys: role, content, and optional sources.
        Explicit incrementing created_at preserves insertion order even though
        all rows share a single INSERT (get_messages orders by created_at).
        """
        base = datetime.now(timezone.utc)
        rows = [
            {
                "conversation_id": conversation_id,
                "user_id": self.user_id,
                "role": m["role"],
                "content": m["content"],
                "sources": m.get("sources") or [],
                "created_at": (base + timedelta(milliseconds=i)).isoformat(),
            }
            for i, m in enumerate(items)
        ]
        res = self.client.table("messages").insert(rows).execute()
        self.client.table("conversations").update(
            {"updated_at": (base + timedelta(milliseconds=len(items))).isoformat()}
        ).eq("id", conversation_id).execute()
        return res.data or []

    def get_messages(self, conversation_id: str) -> list:
        """Load all messages for a conversation, oldest first."""
        if not self.user_id:
            return []
        res = self.read_client.table("messages").select("*").eq(
            "conversation_id", conversation_id
        ).eq("user_id", self.user_id).order("created_at", desc=False).execute()
        return res.data or []

    def auto_title_conversation(self, conversation_id: str, first_question: str):
        """Set conversation title from first user message (truncated)."""
        title = first_question[:50] + ("..." if len(first_question) > 50 else "")
        self.rename_conversation(conversation_id, title)

    # ─────────────────────────────────────────
    # COLLECTIONS
    # ─────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        description: str = None,
        matter_kind: str = None,
        parties: list = None,
        firm_id: str = None,
    ) -> dict:
        """Create a new collection (matter/vault) for the current user.

        F1a: `matter_kind`/`parties`/`firm_id` are optional — omitting them creates a plain
        vault exactly as before (status defaults to 'active' at the DB), so the legacy create
        path is byte-identical. The DB CHECK constraints (migration 010) reject a bad kind.
        """
        if not self.user_id:
            raise ValueError("User must be logged in.")
        row = {
            "user_id": self.user_id,
            "name": name,
            "description": description,
        }
        # Only send matter columns when provided ⇒ no dependency on migration 010 for the
        # legacy path, and DB defaults (status='active', parties='[]') apply otherwise.
        if matter_kind is not None:
            row["matter_kind"] = matter_kind
        if parties is not None:
            row["parties"] = parties
        if firm_id is not None:
            row["firm_id"] = firm_id
        res = self.client.table("collections").insert(row).execute()
        return res.data[0] if res.data else {}

    def get_user_firm(self, user_id: str = None, firm_id: str = None) -> dict:
        """Return the firm this user belongs to (F1a tenancy) with their ROLE in it, or {} if none.

        Server-side lookup (not a JWT claim) so the auth flow is unchanged. F2a: also returns
        `role` (the membership's role) so callers/`resolve_membership` (F2b) get firm + role in
        one read. With `firm_id`, picks that specific membership (the active-firm choice, §0.8);
        otherwise the user's first/primary membership.

        Shape: {"id", "name", "role"}  (role added in F2a; absent if no membership).
        """
        uid = user_id or self.user_id
        if not uid:
            return {}
        q = self.read_client.table("firm_memberships").select(
            "firm_id, role, created_at, firms!inner(id, name)"
        ).eq("user_id", uid)
        if firm_id:
            q = q.eq("firm_id", firm_id)
        # AUDIT FIX #2: pick the active firm DETERMINISTICALLY. Without an ORDER BY, PostgREST row
        # order is undefined, so a user who belongs to >1 firm (e.g. a backfilled solo firm PLUS a
        # firm they joined by invite) could resolve as a DIFFERENT firm/role on each request — an
        # intra-firm cap oscillation. Oldest membership = the stable "primary" firm; an explicit
        # firm_id (the active-firm switcher) still overrides.
        res = q.order("created_at", desc=False).limit(1).execute()
        rows = res.data or []
        if not rows:
            return {}
        firm = dict(rows[0].get("firms", {}) or {})
        firm["role"] = rows[0].get("role")
        return firm

    # ─────────────────────────────────────────
    # FIRM POPULATION + ROLES + INVITES  (F2a — plans/F2_FIRM_CONSOLE_PLAN.md §F2a)
    #   No enforcement here (that is F2b's authorize/require_cap). These are the writes that
    #   POPULATE the firm tenancy 010 created. Service-role writes (the live request client is
    #   service-role; app-layer scoping is the guard until F2f's firm RLS lands under them).
    # ─────────────────────────────────────────

    def create_firm(self, name: str, owner_user_id: str = None, owner_role: str = "managing_partner") -> dict:
        """Create a firm and make `owner_user_id` (default: the current user) its first member
        with `owner_role` (default Managing Partner, D1). Returns {"id","name","role"}.

        Used by signup (D1: first user creates the firm + becomes MP) and the backfill path.
        """
        uid = owner_user_id or self.user_id
        if not uid:
            raise ValueError("An owner user is required to create a firm.")
        res = self.client.table("firms").insert({"name": name}).execute()
        firm = res.data[0] if res.data else {}
        firm_id = firm.get("id")
        if not firm_id:
            raise RuntimeError("Firm create returned no id.")
        self.add_membership(uid, firm_id, owner_role)
        return {"id": firm_id, "name": firm.get("name"), "role": owner_role}

    def rename_firm(self, firm_id: str, name: str) -> dict:
        """Rename a firm (F2g onboarding — name the backfilled solo firm). The CALLER must have
        resolved `firm_id` as their own + authorized (manage_members) server-side; this is the
        write only. Returns the updated {"id","name"}."""
        if not (firm_id and name and name.strip()):
            raise ValueError("firm_id and a non-empty name are required.")
        res = self.client.table("firms").update({"name": name.strip()}).eq("id", firm_id).execute()
        row = res.data[0] if res.data else {"id": firm_id, "name": name.strip()}
        return {"id": str(row.get("id", firm_id)), "name": row.get("name", name.strip())}

    def add_membership(self, user_id: str, firm_id: str, role: str) -> dict:
        """Add (or upsert) a membership row. Idempotent on the (user_id, firm_id) PK — a repeat
        join updates the role rather than erroring. The caller authorizes this (F2b); F2a just
        writes the row."""
        row = {"user_id": user_id, "firm_id": firm_id, "role": role}
        res = self.client.table("firm_memberships").upsert(
            row, on_conflict="user_id,firm_id"
        ).execute()
        return res.data[0] if res.data else row

    def list_members(self, firm_id: str) -> list:
        """All memberships of a firm (user_id + role). Email resolution is best-effort and left
        to the route (needs the admin auth API). Scoped to the passed firm_id — the CALLER must
        have resolved that firm server-side as their own (T2/T3); F2a never trusts a body firm."""
        res = self.client.table("firm_memberships").select(
            "user_id, firm_id, role, created_at"
        ).eq("firm_id", firm_id).execute()
        return res.data or []

    def resolve_member_emails(self, user_ids) -> dict:
        """Map {user_id: email} for a set of user_ids via the service-role admin auth API. Used to
        give the firm console a HUMAN handle for each member instead of a raw user_id (F2g). The
        live request client is service-role (db.py:94), so auth.admin is available. Best-effort:
        any id we cannot resolve is simply omitted (the caller falls back to a short id). Never
        raises — a missing email must never 500 the members list."""
        ids = {str(u) for u in (user_ids or []) if u}
        if not ids:
            return {}
        out: dict[str, str] = {}
        for uid in ids:
            try:
                resp = self.client.auth.admin.get_user_by_id(uid)
                user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)
                email = getattr(user, "email", None) if user else None
                if email:
                    out[uid] = email
            except Exception:
                # admin API unavailable / id not found → leave it out (UI shows a short id).
                continue
        return out

    # ── invites (D1) — single-use, expiring, hash-stored, email-bound (T4) ────────────────────

    @staticmethod
    def _hash_invite_token(raw_token: str) -> str:
        """sha256 hex of the raw token. We store only the hash (T4: a DB read never reveals a
        usable token); accept hashes the presented token and matches by hash."""
        import hashlib
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def create_invite(self, firm_id: str, email: str, role: str, invited_by: str,
                      ttl_hours: int = 168) -> dict:
        """Create a firm invite. Returns the row PLUS the one-time raw `token` (never stored —
        only its hash is). `firm_id`/`invited_by` are resolved server-side by the caller (T3).
        Default TTL = 7 days. The unique active-invite index (migration 012) blocks a duplicate
        pending invite to the same (firm, email)."""
        import secrets
        raw_token = secrets.token_urlsafe(32)
        token_hash = self._hash_invite_token(raw_token)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        row = {
            "firm_id": firm_id,
            "email": email.strip().lower(),
            "role": role,
            "token_hash": token_hash,
            "invited_by": invited_by,
            "expires_at": expires_at,
        }
        res = self.client.table("firm_invites").insert(row).execute()
        created = res.data[0] if res.data else dict(row)
        created.pop("token_hash", None)   # never leak the hash to the client
        created["token"] = raw_token      # the ONLY time the raw token exists post-create
        return created

    def get_invite_by_token(self, raw_token: str) -> dict:
        """Look up a PENDING, UN-EXPIRED invite by its raw token (matched via hash). Returns {}
        if not found / already accepted / expired. Read-only; accept_invite does the atomic claim."""
        token_hash = self._hash_invite_token(raw_token)
        res = self.client.table("firm_invites").select("*").eq(
            "token_hash", token_hash
        ).is_("accepted_at", "null").limit(1).execute()
        rows = res.data or []
        if not rows:
            return {}
        inv = rows[0]
        # expiry check in code (the atomic UPDATE in accept_invite re-checks at the DB)
        exp = inv.get("expires_at")
        try:
            if exp and datetime.fromisoformat(str(exp).replace("Z", "+00:00")) <= datetime.now(timezone.utc):
                return {}
        except Exception:
            return {}
        inv.pop("token_hash", None)
        return inv

    def accept_invite(self, raw_token: str, accepting_user_id: str, accepting_email: str) -> dict:
        """Atomically claim an invite and create the membership with the INVITED role (D1 — the
        joiner can never self-assign). Enforces T4 in full:
          - single-use & un-expired: the accept is a CONDITIONAL UPDATE
            (WHERE accepted_at IS NULL AND expires_at > now()) so a replay/expired token claims
            nothing (no race);
          - email-bound: the accepting user's VERIFIED email must equal the invite email.
        Returns {"firm_id","role"} on success; raises ValueError on any rejection.
        """
        token_hash = self._hash_invite_token(raw_token)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Fetch the candidate (still pending) to validate email-binding before the atomic claim.
        res = self.client.table("firm_invites").select("*").eq(
            "token_hash", token_hash
        ).is_("accepted_at", "null").limit(1).execute()
        rows = res.data or []
        if not rows:
            raise ValueError("Invite is invalid or already used.")
        inv = rows[0]

        # email-binding (T4): the accepting user's verified email must match the invite email.
        if (accepting_email or "").strip().lower() != (inv.get("email") or "").strip().lower():
            raise ValueError("This invite was addressed to a different email.")

        # expiry (T4): re-checked atomically below, but fail fast with a clear message.
        exp = inv.get("expires_at")
        try:
            if exp and datetime.fromisoformat(str(exp).replace("Z", "+00:00")) <= datetime.now(timezone.utc):
                raise ValueError("This invite has expired.")
        except ValueError:
            raise
        except Exception:
            raise ValueError("This invite has expired.")

        # Atomic single-use claim: set accepted_at only if STILL pending and un-expired. If a
        # concurrent accept already won, this matches 0 rows → we treat that as "already used".
        claim = self.client.table("firm_invites").update(
            {"accepted_at": now_iso}
        ).eq("id", inv["id"]).is_("accepted_at", "null").gt(
            "expires_at", now_iso
        ).execute()
        if not (claim.data or []):
            raise ValueError("Invite is invalid or already used.")

        # The claim succeeded → create the membership with the INVITED role.
        self.add_membership(accepting_user_id, inv["firm_id"], inv["role"])
        return {"firm_id": inv["firm_id"], "role": inv["role"]}

    def list_invites(self, firm_id: str, include_inactive: bool = False) -> list:
        """Invites for a firm (pending by default). Never returns `token_hash`. Scoped to the
        firm the CALLER resolved server-side (T2/T3)."""
        q = self.client.table("firm_invites").select(
            "id, firm_id, email, role, invited_by, expires_at, accepted_at, created_at"
        ).eq("firm_id", firm_id)
        if not include_inactive:
            q = q.is_("accepted_at", "null")
        res = q.order("created_at", desc=True).execute()
        return res.data or []

    def resend_invite(self, invite_id: str, firm_id: str, ttl_hours: int = 168) -> dict:
        """ROTATE a pending invite's token (the security-correct 'resend / copy link' recovery —
        researched best practice). Because we store only the hash (T4), the original raw token can
        never be re-shown; instead we mint a FRESH token, replace the stored hash, and reset the
        expiry. Returns the row PLUS the new one-time raw `token`. Firm-scoped (T2/T3 — the caller
        resolved firm_id server-side) and only acts on a still-PENDING invite (a replay of an old
        link is dead the instant a new one is issued — rotation invalidates the prior token).
        Returns {} if the invite is missing / already accepted / not in this firm."""
        import secrets
        raw_token = secrets.token_urlsafe(32)
        token_hash = self._hash_invite_token(raw_token)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        # Conditional update: only a pending invite of THIS firm rotates (accepted/cross-firm → no-op).
        res = self.client.table("firm_invites").update(
            {"token_hash": token_hash, "expires_at": expires_at}
        ).eq("id", invite_id).eq("firm_id", firm_id).is_("accepted_at", "null").execute()
        rows = res.data or []
        if not rows:
            return {}
        row = rows[0]
        row.pop("token_hash", None)
        row["token"] = raw_token          # the ONLY time the rotated raw token exists
        return row

    def revoke_invite(self, invite_id: str, firm_id: str) -> bool:
        """Revoke (delete) a pending invite. Firm-scoped (T2/T3). Idempotent: revoking a missing /
        cross-firm invite returns False, never raises. An accepted invite is left intact (history)."""
        res = self.client.table("firm_invites").delete().eq(
            "id", invite_id
        ).eq("firm_id", firm_id).is_("accepted_at", "null").execute()
        return bool(res.data)

    def count_firm_role(self, firm_id: str, role: str) -> int:
        """How many members of a firm hold a given role. Used by the F2d last-MP guard and the
        F2b self-escalation context; introduced in F2a so the data path exists early."""
        res = self.client.table("firm_memberships").select(
            "user_id", count="exact"
        ).eq("firm_id", firm_id).eq("role", role).execute()
        return res.count or 0

    def get_membership(self, user_id: str, firm_id: str) -> dict:
        """The membership row for (user, firm), or {} if none. Used by the F2d lifecycle routes to
        resolve the TARGET member's CURRENT role SERVER-SIDE (never from the body) before building
        the authz.Scope the guard checks (T2/T3). Firm-scoped: a user_id from another firm returns
        {} (you can only act on a member of your own firm)."""
        if not (user_id and firm_id):
            return {}
        res = self.client.table("firm_memberships").select(
            "user_id, firm_id, role, created_at"
        ).eq("user_id", user_id).eq("firm_id", firm_id).limit(1).execute()
        return (res.data or [{}])[0] if res.data else {}

    # ── member lifecycle (F2d) — role change + offboard, firm-scoped (T3) ──────────────────────
    #   These are the WRITES; the route authorizes (manage_members + the T1/last-MP guards via the
    #   central authz.authorize over a SERVER-RESOLVED Scope) BEFORE calling them. db.py never
    #   authorizes itself and never trusts a body firm_id — every call is scoped to the firm the
    #   caller resolved as their own. The CHANGE takes effect on the target's NEXT request (T7 —
    #   caps are resolved per-request in resolve_membership; there is no session cache to bust).

    def change_member_role(self, user_id: str, firm_id: str, new_role: str) -> dict:
        """Change a member's role within a firm. Firm-scoped (the UPDATE matches only a row whose
        firm_id == the resolved firm — a guessed cross-firm user_id touches nothing, T2/T3). The
        guards (last-MP, self-escalation) ran in the route via authz.authorize; this only writes.
        Returns the updated row, or {} if nothing matched (not in this firm / unknown id)."""
        if not (user_id and firm_id and new_role):
            return {}
        res = self.client.table("firm_memberships").update(
            {"role": new_role}
        ).eq("user_id", user_id).eq("firm_id", firm_id).execute()
        return (res.data or [{}])[0] if res.data else {}

    def reassign_member_matters(self, user_id: str, firm_id: str, new_owner_id: str) -> int:
        """Reassign every vault (collection) AUTHORED by `user_id` in `firm_id` to `new_owner_id`
        (the firm / an MP) so an offboard NEVER orphans a matter (the resolved-default policy,
        §F2d). Firm-scoped so we only touch this firm's collections. Returns the number reassigned.
        Called BEFORE remove_member so the matters always have an owner across the transition."""
        if not (user_id and firm_id and new_owner_id):
            return 0
        # Only the offboarded user's OWN matters in THIS firm; re-point user_id to the new owner.
        sel = self.client.table("collections").select("id").eq(
            "user_id", user_id
        ).eq("firm_id", firm_id).execute()
        ids = [r["id"] for r in (sel.data or []) if r.get("id")]
        if not ids:
            return 0
        self.client.table("collections").update(
            {"user_id": new_owner_id}
        ).in_("id", ids).execute()
        return len(ids)

    def remove_member(self, user_id: str, firm_id: str) -> dict:
        """Offboard a member: remove their firm membership AND revoke every delegation they hold
        or granted within the firm (instant, total revocation — their next request is powerless,
        T7). Firm-scoped (T2/T3 — a cross-firm user_id deletes nothing). The caller has already
        reassigned the member's matters (reassign_member_matters) so nothing is orphaned, and run
        the last-MP guard (the firm always keeps an owner). Returns {"removed": bool, "delegations_revoked": int}."""
        if not (user_id and firm_id):
            return {"removed": False, "delegations_revoked": 0}
        # Revoke delegations first (both directions) so a stale grant can't outlive the membership.
        revoked = self.revoke_member_delegations(user_id, firm_id)
        res = self.client.table("firm_memberships").delete().eq(
            "user_id", user_id
        ).eq("firm_id", firm_id).execute()
        removed = bool(res.data)
        return {"removed": removed, "delegations_revoked": revoked}

    # ── delegations (F2d / D6) — time-boxed, revocable, bounded grants ─────────────────────────
    #   migration 014. A delegation is a POSITIVE grant authorize() honors at step 1.5, but it
    #   STILL cannot beat a screen DENY (precedence holds) and a delegate can NEVER hold a verb the
    #   delegator lacks (bounded HERE, in active_delegated_verbs). Service-role writes; the route
    #   authorizes + resolves firm server-side (T3) before calling these.

    def active_delegated_verbs(self, user_id: str = None, firm_id: str = None) -> set:
        """The set of verbs `user_id` currently holds via an ACTIVE delegation within a firm (the
        DELEGATE side — what a PA may do for the senior). THE HOT PATH: resolve_membership calls
        this per request to populate authz.Membership.delegated_verbs, so authorize() honors a
        delegation on every cap-gated route (Layer 1). Active = revoked_at IS NULL AND expires_at >
        now() (a revoked/expired grant stops on the NEXT request — T7).

        BOUNDED (T1): a delegate can never be granted a verb the DELEGATOR does not themselves hold.
        We intersect each delegation's verbs with the delegator's OWN role caps at resolution time,
        so a later demotion of the delegator (or a grant that over-reached) can never leak a verb
        the delegator lacks. Empty set when no firm / no delegations / the table is unapplied ⇒
        byte-identical to pre-F2d (delegated_verbs = ∅).
        """
        uid = user_id or self.user_id
        if not uid:
            return set()
        now_iso = datetime.now(timezone.utc).isoformat()
        q = self.read_client.table("delegations").select(
            "delegator_id, verbs, expires_at"
        ).eq("delegate_id", uid).is_("revoked_at", "null").gt("expires_at", now_iso)
        if firm_id:
            q = q.eq("firm_id", firm_id)
        try:
            res = q.execute()
        except Exception:
            # Table not applied (migration 014 not live) ⇒ no delegations (legacy parity), never 500.
            return set()
        from src.components import authz
        granted: set = set()
        for row in (res.data or []):
            verbs = set(row.get("verbs") or [])
            if not verbs:
                continue
            # BOUND to the delegator's own caps — resolve the delegator's CURRENT role server-side.
            delegator_firm = self.get_user_firm(user_id=row.get("delegator_id"), firm_id=firm_id)
            delegator_role = (delegator_firm or {}).get("role")
            delegator_caps = authz.caps_for_role(delegator_role) if delegator_role else frozenset()
            granted |= (verbs & set(delegator_caps))
        return granted

    def create_delegation(self, firm_id: str, delegator_id: str, delegate_id: str,
                          verbs: list, expires_at: str) -> dict:
        """Grant a bounded, time-boxed delegation. The caller has authorized (the delegator is the
        accountable senior) and resolved firm_id server-side (T3). `verbs` is the requested subset;
        the resolver re-bounds it to the delegator's own caps at READ time (active_delegated_verbs),
        so even a verb the delegator later loses can never be exercised. `expires_at` is required
        (time-boxed). Returns the inserted row."""
        if not (firm_id and delegator_id and delegate_id):
            raise ValueError("firm_id, delegator_id and delegate_id are all required.")
        if not expires_at:
            raise ValueError("A delegation must be time-boxed (expires_at is required).")
        row = {
            "firm_id": firm_id,
            "delegator_id": delegator_id,
            "delegate_id": delegate_id,
            "verbs": list(verbs or []),
            "expires_at": expires_at,
        }
        res = self.client.table("delegations").insert(row).execute()
        return res.data[0] if res.data else row

    def revoke_delegation(self, delegation_id: str, firm_id: str) -> dict:
        """Revoke a delegation early (soft — the audit history is preserved). Firm-scoped (T2 — a
        guessed cross-firm delegation_id revokes nothing). Only an active (revoked_at IS NULL) row
        is touched so a double-revoke is a clean no-op. Takes effect on the delegate's NEXT request
        (T7). Returns the updated row, or {} if nothing matched."""
        now_iso = datetime.now(timezone.utc).isoformat()
        res = self.client.table("delegations").update(
            {"revoked_at": now_iso}
        ).eq("id", delegation_id).eq("firm_id", firm_id).is_(
            "revoked_at", "null"
        ).execute()
        return (res.data or [{}])[0] if res.data else {}

    def revoke_member_delegations(self, user_id: str, firm_id: str) -> int:
        """Revoke EVERY active delegation a user holds OR granted within a firm (the offboard hook —
        a removed member's grants must not outlive their membership). Soft-revoke (history kept).
        Firm-scoped. Returns the number revoked. Degrades to 0 if the table is unapplied."""
        if not (user_id and firm_id):
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            sel = self.client.table("delegations").select("id").eq(
                "firm_id", firm_id
            ).is_("revoked_at", "null").or_(
                f"delegator_id.eq.{user_id},delegate_id.eq.{user_id}"
            ).execute()
            ids = [r["id"] for r in (sel.data or []) if r.get("id")]
            if not ids:
                return 0
            self.client.table("delegations").update(
                {"revoked_at": now_iso}
            ).in_("id", ids).execute()
            return len(ids)
        except Exception:
            return 0

    def list_delegations(self, firm_id: str, include_inactive: bool = False) -> list:
        """All delegations of a firm (active by default; include_inactive shows revoked/expired for
        the audit view). Scoped to the firm the caller resolved server-side (T2/T3)."""
        q = self.client.table("delegations").select(
            "id, firm_id, delegator_id, delegate_id, verbs, expires_at, revoked_at, created_at"
        ).eq("firm_id", firm_id)
        if not include_inactive:
            q = q.is_("revoked_at", "null")
        res = q.order("created_at", desc=True).execute()
        return res.data or []

    # ─────────────────────────────────────────
    # ETHICAL WALLS — conflict screens  (F2c — plans/F2_FIRM_CONSOLE_PLAN.md §F2c) — THE MOAT
    #   A screen hard-blocks one member from one matter (vault) at EVERY layer. These are the
    #   writes + the read that POPULATE the wall; F2b's authorize() already checks the screen
    #   first in its deny-overrides precedence, and resolve_membership (dependencies.py) +
    #   the retrieval-layer guard (P1–P9) are what ENFORCE it. Service-role writes (same as F2a);
    #   the caller resolves firm_id SERVER-SIDE and authorizes (manage_members) BEFORE calling
    #   these — db.py never trusts a body firm_id (T3), and these never authorize themselves.
    # ─────────────────────────────────────────

    def screened_vault_ids(self, user_id: str = None, firm_id: str = None) -> set:
        """The set of vault_ids this user is ACTIVELY screened off (the ethical wall) within a
        firm. THE HOT PATH: resolve_membership calls this per request to populate
        authz.Membership.screened_vault_ids, so authorize() denies a screened vault on every
        cap-gated route automatically (Layer 1), and the retrieval-layer guard tests against it
        (Layer 2). Active = removed_at IS NULL (a soft-removed screen restores access on the
        NEXT request — T7). Empty set when no firm / no screens ⇒ byte-identical to pre-F2c.

        firm-scoped so a screen in firm A never bleeds into the user's membership of firm B.
        """
        uid = user_id or self.user_id
        if not uid:
            return set()
        q = self.read_client.table("screens").select("vault_id").eq(
            "user_id", uid
        ).is_("removed_at", "null")
        if firm_id:
            q = q.eq("firm_id", firm_id)
        try:
            res = q.execute()
        except Exception as e:
            # AUDIT FIX #3 (fail-closed wall): ONLY a missing `screens` table (migration 013 not yet
            # applied) may degrade to "no screens" — that is legacy parity. ANY OTHER error (a live
            # query fault) must NOT silently drop the ethical wall; we re-raise so the caller fails
            # CLOSED (assert_vault_not_screened treats a lookup failure as screened). The wall is the
            # regulatory moat — a DB blip must never quietly open it.
            if _is_missing_relation(e):
                return set()
            raise
        return {str(r["vault_id"]) for r in (res.data or []) if r.get("vault_id")}

    def is_vault_screened(self, vault_id: str, user_id: str = None, firm_id: str = None) -> bool:
        """Is THIS user actively screened off THIS vault? The single-vault check the
        retrieval/read layer (P1–P7) uses on the resolved collection_id. Degrades to False
        (no wall) if the table is absent (migration unapplied) — same legacy parity rule."""
        uid = user_id or self.user_id
        if not uid or not vault_id:
            return False
        q = self.read_client.table("screens").select("id").eq(
            "user_id", uid
        ).eq("vault_id", vault_id).is_("removed_at", "null")
        if firm_id:
            q = q.eq("firm_id", firm_id)
        try:
            res = q.limit(1).execute()
        except Exception as e:
            # AUDIT FIX #3: missing table → degrade to "no wall" (legacy parity); any live fault
            # re-raises so the read-layer guard fails CLOSED rather than silently opening the wall.
            if _is_missing_relation(e):
                return False
            raise
        return bool(res.data)

    def create_screen(self, firm_id: str, user_id: str, vault_id: str, reason: str,
                      created_by: str = None) -> dict:
        """Raise an ethical wall: screen `user_id` off `vault_id` within `firm_id`. The caller
        has already authorized (manage_members) and resolved `firm_id` server-side (T3). `reason`
        is REQUIRED (a wall must be justifiable to a regulator). `created_by` defaults to the
        current user. Idempotent on an ACTIVE (firm,user,vault) via the partial unique index —
        a duplicate active screen surfaces as a clean conflict to the route, not a dup row.
        Returns the inserted row."""
        if not (firm_id and user_id and vault_id):
            raise ValueError("firm_id, user_id and vault_id are all required to create a screen.")
        if not (reason or "").strip():
            raise ValueError("A reason is required to raise an ethical wall.")
        row = {
            "firm_id": firm_id,
            "user_id": user_id,
            "vault_id": vault_id,
            "reason": reason.strip(),
            "created_by": created_by or self.user_id,
        }
        res = self.client.table("screens").insert(row).execute()
        return res.data[0] if res.data else row

    def remove_screen(self, screen_id: str, firm_id: str) -> dict:
        """SOFT-remove a screen (set removed_at = now) — never a hard delete (the audit history
        is preserved). Scoped to `firm_id` (resolved server-side by the caller — T3) so a member
        of firm A can never lift firm B's wall even with a guessed screen_id (T2 IDOR). Only an
        ACTIVE screen is removed (removed_at IS NULL) so a double-remove is a no-op, not an error.
        Restores access on the NEXT request (per-request screen resolution — T7). Returns the
        updated row, or {} if nothing matched (wrong firm / already removed / unknown id)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        res = self.client.table("screens").update(
            {"removed_at": now_iso}
        ).eq("id", screen_id).eq("firm_id", firm_id).is_(
            "removed_at", "null"
        ).execute()
        return (res.data or [{}])[0] if res.data else {}

    def list_screens(self, firm_id: str, include_removed: bool = False) -> list:
        """All screens of a firm (active by default; include_removed shows the lifted ones for the
        audit view). Scoped to the firm the CALLER resolved server-side as their own (T2/T3)."""
        q = self.client.table("screens").select(
            "id, firm_id, user_id, vault_id, reason, created_by, created_at, removed_at"
        ).eq("firm_id", firm_id)
        if not include_removed:
            q = q.is_("removed_at", "null")
        res = q.order("created_at", desc=True).execute()
        return res.data or []

    def collection_in_firm(self, vault_id: str, firm_id: str) -> bool:
        """Does `vault_id` belong to `firm_id`? The T2 (IDOR) check a screen-create must pass: a
        manager can only wall a vault OWNED BY THEIR OWN FIRM (resolved server-side). A
        firm-scoped lookup (NOT get_collection, which is owner-scoped) — the manager raising the
        wall need not own the vault. Returns False on any miss (foreign firm / unknown id)."""
        if not (vault_id and firm_id):
            return False
        res = self.client.table("collections").select("id").eq(
            "id", vault_id
        ).eq("firm_id", firm_id).limit(1).execute()
        return bool(res.data)

    def user_in_firm(self, user_id: str, firm_id: str) -> bool:
        """Is `user_id` a member of `firm_id`? The T2 check the screen-create must pass for the
        SCREENED user: you can only wall a member of your own firm (no walling a stranger / a
        member of another firm). Returns False on any miss."""
        if not (user_id and firm_id):
            return False
        res = self.client.table("firm_memberships").select("user_id").eq(
            "user_id", user_id
        ).eq("firm_id", firm_id).limit(1).execute()
        return bool(res.data)

    # ─────────────────────────────────────────
    # MATTER STAFFING + the REVIEW CHAIN  (F2e — plans/F2_FIRM_CONSOLE_PLAN.md §F2e) — D0/D3/D5
    #   migration 015. THE PRODUCTIVITY ENGINE. matter_memberships = who is on a matter (being on it
    #   grants the FULL toolkit on that matter — D0, never read-only). review_requests = a piece of
    #   work flowing UP the chain of command (current_owner ALWAYS set = anti-stall). Service-role
    #   writes (same as F2a–d); the route authorizes (manage_matter_team / send_for_review /
    #   release_external) + resolves firm + vault server-side (T3) BEFORE calling these — db.py never
    #   authorizes itself and never trusts a body firm_id. Every method degrades to an empty result
    #   when its table is unapplied (migration not yet live) — byte-identical to pre-F2e, never a 500.
    # ─────────────────────────────────────────

    # ── matter staffing (D3) — who is on a matter (vault) ──────────────────────────────────────

    def add_matter_member(self, firm_id: str, vault_id: str, user_id: str, added_by: str) -> dict:
        """Staff `user_id` onto a matter (vault) — they get the FULL working toolkit on it (D0). The
        caller has authorized (manage_matter_team) + asserted the vault is in their firm (T3). The
        firm + vault are resolved server-side; `added_by` is the staffing senior. Idempotent on the
        ACTIVE (vault, user) via the unique index — a re-staff returns the existing row, not a dup.
        Returns the staffing row."""
        if not (firm_id and vault_id and user_id):
            raise ValueError("firm_id, vault_id and user_id are all required to staff a matter.")
        row = {"firm_id": firm_id, "vault_id": vault_id, "user_id": user_id,
               "added_by": added_by or self.user_id}
        try:
            res = self.client.table("matter_memberships").upsert(
                row, on_conflict="vault_id,user_id"
            ).execute()
        except Exception:
            # The unique-conflict path or table-absent: fall back to a plain insert / return the row.
            res = self.client.table("matter_memberships").insert(row).execute()
        return res.data[0] if res.data else row

    def remove_matter_member(self, firm_id: str, vault_id: str, user_id: str) -> bool:
        """Un-staff a member from a matter — INSTANT revoke (they lose matter access on their NEXT
        request, T7). Firm- AND vault-scoped (T2 — a guessed cross-firm vault touches nothing).
        Returns True if a row was removed, False otherwise (already removed / not staffed)."""
        if not (firm_id and vault_id and user_id):
            return False
        res = self.client.table("matter_memberships").delete().eq(
            "firm_id", firm_id
        ).eq("vault_id", vault_id).eq("user_id", user_id).execute()
        return bool(res.data)

    def list_matter_members(self, firm_id: str, vault_id: str) -> list:
        """Everyone STAFFED on a matter (user_id + their firm ROLE, joined). The team roster + the
        default chain-builder's input. Firm- + vault-scoped (the caller resolved both server-side —
        T2/T3). Returns [{user_id, role, added_by, created_at}]. Empty if the table is unapplied."""
        if not (firm_id and vault_id):
            return []
        try:
            res = self.read_client.table("matter_memberships").select(
                "user_id, added_by, created_at, firm_memberships!inner(role)"
            ).eq("firm_id", firm_id).eq("vault_id", vault_id).execute()
        except Exception:
            # Table not applied (015 not live) OR the embed isn't resolvable — fall back to a plain
            # select + a per-row role lookup, never 500.
            try:
                res = self.read_client.table("matter_memberships").select(
                    "user_id, added_by, created_at"
                ).eq("firm_id", firm_id).eq("vault_id", vault_id).execute()
            except Exception:
                return []
            out = []
            for r in (res.data or []):
                role = (self.get_membership(r.get("user_id"), firm_id) or {}).get("role")
                out.append({"user_id": r.get("user_id"), "role": role,
                            "added_by": r.get("added_by"), "created_at": r.get("created_at")})
            return out
        out = []
        for r in (res.data or []):
            fm = r.get("firm_memberships") or {}
            role = fm.get("role") if isinstance(fm, dict) else None
            out.append({"user_id": r.get("user_id"), "role": role,
                        "added_by": r.get("added_by"), "created_at": r.get("created_at")})
        return out

    def is_matter_member(self, vault_id: str, user_id: str = None, firm_id: str = None) -> bool:
        """Is `user_id` STAFFED on `vault_id`? Used by the team route (idempotency / membership check)
        and available for matter-scoped checks. Degrades to False if the table is unapplied."""
        uid = user_id or self.user_id
        if not (uid and vault_id):
            return False
        q = self.read_client.table("matter_memberships").select("id").eq(
            "user_id", uid
        ).eq("vault_id", vault_id)
        if firm_id:
            q = q.eq("firm_id", firm_id)
        try:
            res = q.limit(1).execute()
        except Exception:
            return False
        return bool(res.data)

    def matter_member_vault_ids(self, user_id: str = None, firm_id: str = None) -> set:
        """The set of vault_ids `user_id` is STAFFED on within a firm. THE HOT PATH:
        resolve_membership calls this per request to populate authz.Membership.matter_vault_ids — the
        productivity seam (D0). Resolved per-request (T7 — a removed staffing drops on the next
        request). Empty set when no firm / no staffing / the table is unapplied ⇒ byte-identical to
        pre-F2e."""
        uid = user_id or self.user_id
        if not uid:
            return set()
        q = self.read_client.table("matter_memberships").select("vault_id").eq("user_id", uid)
        if firm_id:
            q = q.eq("firm_id", firm_id)
        try:
            res = q.execute()
        except Exception:
            return set()
        return {str(r["vault_id"]) for r in (res.data or []) if r.get("vault_id")}

    # ── the review chain (D5) — review_requests + matter_review_config ─────────────────────────

    def get_matter_review_chain(self, vault_id: str, firm_id: str) -> "list | None":
        """The CUSTOM review chain for a matter (matter_review_config.chain — an ordered list of
        reviewer user_ids), or None when the matter has no custom config ⇒ the rank-based default
        applies. Firm- + vault-scoped. Degrades to None if the table is unapplied (default applies)."""
        if not (vault_id and firm_id):
            return None
        try:
            res = self.read_client.table("matter_review_config").select("chain").eq(
                "vault_id", vault_id
            ).eq("firm_id", firm_id).limit(1).execute()
        except Exception:
            return None
        rows = res.data or []
        if not rows:
            return None
        chain = rows[0].get("chain")
        return chain if chain else None

    def set_matter_review_chain(self, firm_id: str, vault_id: str, chain: "list | None",
                                set_by: str = None) -> dict:
        """Set (or clear) a matter's CUSTOM review chain. `chain` = an ordered list of reviewer
        user_ids, or None to clear it (revert to the rank default). The caller authorized
        (manage_matter_team) + resolved firm + vault server-side (T3). Upsert on the vault PK."""
        if not (firm_id and vault_id):
            raise ValueError("firm_id and vault_id are required.")
        row = {"vault_id": vault_id, "firm_id": firm_id, "chain": chain,
               "set_by": set_by or self.user_id,
               "updated_at": datetime.now(timezone.utc).isoformat()}
        res = self.client.table("matter_review_config").upsert(
            row, on_conflict="vault_id"
        ).execute()
        return res.data[0] if res.data else row

    def create_review_request(self, firm_id: str, vault_id: str, artifact_ref: str,
                              submitted_by: str, current_owner: str, chain: list) -> dict:
        """Submit a piece of work UP the review chain. status='pending'; current_owner = the first
        reviewer (ALWAYS set — the anti-stall invariant, D5); chain = the ordered reviewer list the
        builder produced. The caller authorized (send_for_review) + resolved firm + vault + the chain
        server-side. Returns the inserted request."""
        if not (firm_id and vault_id and artifact_ref and submitted_by and current_owner):
            raise ValueError("firm_id, vault_id, artifact_ref, submitted_by and current_owner are required.")
        row = {
            "firm_id": firm_id,
            "vault_id": vault_id,
            "artifact_ref": artifact_ref,
            "submitted_by": submitted_by,
            "status": "pending",
            "current_owner": current_owner,
            "chain": list(chain or []),
        }
        res = self.client.table("review_requests").insert(row).execute()
        return res.data[0] if res.data else row

    def get_review_request(self, request_id: str, firm_id: str) -> dict:
        """One review request, scoped to the caller's firm (T2 — a guessed cross-firm request id
        returns {}). The route resolves the current state server-side before any transition."""
        if not (request_id and firm_id):
            return {}
        try:
            res = self.read_client.table("review_requests").select("*").eq(
                "id", request_id
            ).eq("firm_id", firm_id).limit(1).execute()
        except Exception:
            return {}
        return (res.data or [{}])[0] if res.data else {}

    def update_review_request(self, request_id: str, firm_id: str, **fields) -> dict:
        """Apply a state transition to a review request (status / current_owner / decided_at). The
        route computed the next state via authorize() + the chain-builder; this only writes. Firm-
        scoped (T2). Returns the updated row, or {} if nothing matched."""
        if not (request_id and firm_id) or not fields:
            return {}
        res = self.client.table("review_requests").update(fields).eq(
            "id", request_id
        ).eq("firm_id", firm_id).execute()
        return (res.data or [{}])[0] if res.data else {}

    def list_my_review_queue(self, firm_id: str, owner_id: str = None) -> list:
        """The OPEN review requests `owner_id` currently OWNS (my review queue — the anti-stall UX,
        D5). status in the non-terminal set; current_owner = the user. Firm-scoped. Empty if the
        table is unapplied."""
        uid = owner_id or self.user_id
        if not (firm_id and uid):
            return []
        try:
            res = self.read_client.table("review_requests").select("*").eq(
                "firm_id", firm_id
            ).eq("current_owner", uid).in_(
                "status", ["pending", "approved", "changes_requested"]
            ).order("created_at", desc=True).execute()
        except Exception:
            return []
        return res.data or []

    # Partner-tier sees the WHOLE firm (they run it); everyone else sees need-to-know:
    # the vaults they OWN plus the matters they are STAFFED on. In LOCKSTEP with
    # review_chain._RELEASE_TIER (the same "runs the firm" set).
    _FIRM_WIDE_ROLES = frozenset({"managing_partner", "senior_partner", "partner"})

    def get_collections(self) -> list:
        """The vaults the current user should SEE (F2 — role-scoped visibility).

        Visibility model (app-layer, mirrors the F2f firm+wall row floor but adds the
        matter-scope for juniors that row-RLS deliberately does NOT — matter-membership
        stays app-layer per the F2f decision):
          • Partner-tier (MP/senior_partner/partner) → EVERY vault in their firm. They run
            the firm; need-to-know does not bound them.
          • Any other firm member (associate/paralegal/assistant/…) → the vaults they OWN
            ∪ the matters they are STAFFED on (matter_memberships). A staffed paralegal
            finally SEES the matter they were added to — the gap that made the everyday UI
            look identical for every role.
          • Firm-less / legacy user → owned vaults only (byte-identical to pre-F2).
        The ethical wall is subtracted in ALL firm cases (a screened vault never shows,
        deny-overrides-role — even for a partner).
        """
        if not self.user_id:
            return []

        firm = {}
        try:
            firm = self.get_user_firm() or {}
        except Exception:
            firm = {}
        firm_id = firm.get("id")
        role = firm.get("role") or "managing_partner"

        # Legacy / firm-less: unchanged — owned vaults only.
        if not firm_id:
            res = self.read_client.table("collections").select("*").eq(
                "user_id", self.user_id
            ).order("updated_at", desc=True).execute()
            return res.data or []

        if role in self._FIRM_WIDE_ROLES:
            # Partner-tier: the whole firm.
            res = self.read_client.table("collections").select("*").eq(
                "firm_id", firm_id
            ).order("updated_at", desc=True).execute()
            rows = res.data or []
        else:
            # Need-to-know: owned ∪ staffed. Two reads, merged + de-duped (PostgREST has no
            # clean OR across an IN-list of staffed ids without risking an over-broad filter,
            # and the staffed set is small).
            owned = self.read_client.table("collections").select("*").eq(
                "user_id", self.user_id
            ).order("updated_at", desc=True).execute().data or []
            staffed_ids = self.matter_member_vault_ids(firm_id=firm_id)
            staffed_rows = []
            if staffed_ids:
                staffed_rows = self.read_client.table("collections").select("*").in_(
                    "id", list(staffed_ids)
                ).eq("firm_id", firm_id).execute().data or []
            seen, rows = set(), []
            for r in [*owned, *staffed_rows]:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
            rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)

        # The ethical wall floor: never surface a screened vault (deny-overrides-role).
        try:
            screened = self.screened_vault_ids(firm_id=firm_id)
        except Exception:
            # screened_vault_ids fails CLOSED on a live fault (re-raises); a missing table
            # returns ∅. If it raised, the wall is uncertain — drop nothing here (the
            # per-vault assert_vault_not_screened guard at entry is the load-bearing block),
            # but log it.
            screened = set()
            import logging
            logging.getLogger(__name__).warning(
                "get_collections: screen lookup failed; entry guard remains the wall")
        if screened:
            rows = [r for r in rows if str(r["id"]) not in screened]
        return rows

    def get_collection(self, collection_id: str) -> dict:
        """Get a single collection by ID — for the caller's own vault OR a matter they're staffed
        on (F2m, owner-resolved via accessible_vault_owner so a staffed member can open the shared
        matter's header/metadata). Returns {} when the caller has no access (non-member or screened)."""
        owner = self.accessible_vault_owner(collection_id)
        if not owner:
            return {}
        res = self.client.table("collections").select("*").eq(
            "user_id", owner
        ).eq("id", collection_id).execute()
        return res.data[0] if res.data else {}

    def update_collection(
        self,
        collection_id: str,
        name: str = None,
        description: str = None,
        matter_kind: str = None,
        status: str = None,
        parties: list = None,
    ) -> dict:
        """Rename or update a collection (matter/vault).

        F1a: also updates the matter lifecycle/typing (`matter_kind`/`status`/`parties`). Each
        field is independent — a rename-only call (the legacy shape) sends only `name` and is
        unchanged. Returns {} when nothing changed (the route falls back to the existing row).
        """
        if not self.user_id:
            return {}
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if matter_kind is not None:
            update_data["matter_kind"] = matter_kind
        if status is not None:
            update_data["status"] = status
        if parties is not None:
            update_data["parties"] = parties
        if not update_data:
            return {}
        res = self.client.table("collections").update(
            update_data
        ).eq("id", collection_id).eq("user_id", self.user_id).execute()
        return res.data[0] if res.data else {}

    def delete_collection(self, collection_id: str):
        """Delete a collection (documents stay, just unlinked)."""
        if not self.user_id:
            return
        self.client.table("collections").delete().eq(
            "id", collection_id
        ).eq("user_id", self.user_id).execute()

    def add_document_to_collection(self, collection_id: str, document_id: str) -> dict:
        """Add a document to a collection."""
        res = self.client.table("collection_documents").insert({
            "collection_id": collection_id,
            "document_id": document_id,
        }).execute()
        return res.data[0] if res.data else {}

    def remove_document_from_collection(self, collection_id: str, document_id: str):
        """Remove a document from a collection."""
        self.client.table("collection_documents").delete().eq(
            "collection_id", collection_id
        ).eq("document_id", document_id).execute()

    def accessible_vault_owner(self, collection_id: str) -> "str | None":
        """THE matter-access authority (F2m / D3): the user_id whose namespace + documents back a
        vault the CURRENT user is allowed to read — or None if they have no access.

        Returns:
          • self.user_id          — the caller OWNS the vault (the legacy, byte-identical path).
          • the OWNER's user_id    — the caller is STAFFED on the matter (matter_memberships) within
                                     the SAME firm AND is NOT screened off it. A staffed member reads
                                     the owner's docs/namespace = "full access on this matter" (D0/D3).
          • None                   — no ownership, no staffing, or a screen blocks it (deny-overrides).

        This is the ONE place cross-user vault read is granted. It is HARD-gated by matter membership
        + same-firm + the ethical-wall screen — never a raw ownership bypass. Read paths
        (get_collection_document_ids / get_collection_documents / the retrieval namespace) route
        through it so a staffed paralegal can actually open and query the shared matter, while a
        non-staffed user still resolves to nothing.

        Uses the service-role client for the cross-user lookups (resolving ANOTHER user's vault owner
        is inherently cross-user) but only AFTER the membership+screen gate passes — the gate is the
        boundary, not the client.

        Per-request memo: a single vault load calls this 3× (get_collection / *_document_ids /
        *_documents), each ~3-4 sequential Supabase round-trips. The SupabaseManager is built per
        request, so a tiny per-instance cache keyed on collection_id collapses those to one resolve
        (the screen check stays per-request across DIFFERENT requests — T7 — since the instance is new
        each time). Safe: same boundary, far less latency.
        """
        if not (self.user_id and collection_id):
            return None
        _memo = getattr(self, "_vault_owner_memo", None)
        if _memo is None:
            _memo = {}
            self._vault_owner_memo = _memo
        if collection_id in _memo:
            return _memo[collection_id]

        def _remember(val):
            _memo[collection_id] = val
            return val

        # Owner fast-path (RLS-enforced read of the caller's own row). If they own it, done.
        try:
            own = self.read_client.table("collections").select("id").eq(
                "user_id", self.user_id).eq("id", collection_id).limit(1).execute()
            if own.data:
                return _remember(self.user_id)
        except Exception:
            pass

        # Not the owner — is the caller STAFFED on this matter? Resolve the caller's firm + the
        # vault's owner/firm via service-role (cross-user by nature), then require:
        #   (1) the vault is in the caller's firm, (2) the caller is staffed on it, (3) no screen.
        firm = {}
        try:
            firm = self.get_user_firm() or {}
        except Exception:
            firm = {}
        firm_id = firm.get("id")
        if not firm_id:
            return _remember(None)  # firm-less ⇒ no matter sharing ⇒ owner-only (handled above)

        try:
            coll = self.client.table("collections").select("user_id, firm_id").eq(
                "id", collection_id).limit(1).execute()
        except Exception:
            return _remember(None)
        if not coll.data:
            return _remember(None)
        owner_id = coll.data[0].get("user_id")
        vault_firm = coll.data[0].get("firm_id")
        # T2/T3: the vault MUST belong to the caller's firm. A cross-firm vault is invisible.
        if not owner_id or str(vault_firm) != str(firm_id):
            return _remember(None)

        # (2) staffed on the matter? (the productivity grant — D3)
        try:
            staffed = self.client.table("matter_memberships").select("id").eq(
                "firm_id", firm_id).eq("vault_id", collection_id).eq(
                "user_id", self.user_id).limit(1).execute()
        except Exception:
            return _remember(None)
        if not staffed.data:
            return _remember(None)

        # (3) the ethical wall (deny-overrides): a screened member reads nothing. screened_vault_ids
        # fails CLOSED on a live fault (re-raises) — so a wall lookup blip denies, never opens.
        # NOT memoized through a raise: a screen fault must re-deny on each call, never cache "open".
        try:
            if str(collection_id) in self.screened_vault_ids(firm_id=firm_id):
                return _remember(None)
        except Exception:
            return None  # fail closed — uncertain wall ⇒ deny (don't memo a transient fault)

        return _remember(str(owner_id))

    def get_collection_document_ids(self, collection_id: str) -> list[str]:
        """Get all document IDs in a collection, scoped to the OWNER resolved by the matter-access
        authority (accessible_vault_owner). For the caller's own vault that owner IS the caller
        (byte-identical legacy path); for a matter they're STAFFED on it's the vault owner (D3); for
        no access it's None ⇒ []. So a staffed member sees the shared matter's real documents, and a
        non-member still sees nothing."""
        owner = self.accessible_vault_owner(collection_id)
        if not owner:
            return []
        # Service-role join scoped to the resolved owner (the gate already passed in
        # accessible_vault_owner — this read is bounded to exactly that owner's docs in this vault).
        res = self.client.table("collection_documents").select(
            "document_id, collections!inner(user_id)"
        ).eq("collection_id", collection_id).eq(
            "collections.user_id", owner
        ).execute()
        return [row["document_id"] for row in (res.data or [])]

    def batch_collection_doc_counts(self, collections: list) -> dict:
        """Doc-count per collection in ONE query (kills the per-vault N+1 in list_collections).

        `collections` are rows already resolved + access-checked by get_collections (each carries
        its OWNER as `user_id` + its `id`). We fetch all collection_documents for those ids in a
        single round-trip and tally per collection_id — instead of one get_collection_document_ids
        (which itself resolves accessible_vault_owner) per vault. The access gate already ran in
        get_collections, so this is a pure count over the visible set. Returns {collection_id: int};
        a vault with no docs is simply absent (caller defaults to 0). Never raises — degrades to {}
        so the list still renders (just without counts) on a query fault."""
        ids = [str(c.get("id")) for c in (collections or []) if c.get("id")]
        if not ids:
            return {}
        try:
            res = self.client.table("collection_documents").select(
                "collection_id"
            ).in_("collection_id", ids).execute()
        except Exception:  # noqa: BLE001 — counts are cosmetic; never break the list
            return {}
        counts: dict[str, int] = {}
        for row in (res.data or []):
            cid = str(row.get("collection_id"))
            counts[cid] = counts.get(cid, 0) + 1
        return counts

    def get_collection_documents(self, collection_id: str) -> list:
        """Get full document records for all documents in a collection (owner-resolved, F2m)."""
        owner = self.accessible_vault_owner(collection_id)
        if not owner:
            return []
        doc_ids = self.get_collection_document_ids(collection_id)
        if not doc_ids:
            return []
        res = self.client.table("documents").select("*").in_(
            "id", doc_ids
        ).eq("user_id", owner).execute()
        return res.data or []

    def get_document_collections(self, document_id: str) -> list:
        """Get all collections that contain a specific document."""
        res = self.read_client.table("collection_documents").select(
            "collection_id"
        ).eq("document_id", document_id).execute()
        collection_ids = [row["collection_id"] for row in (res.data or [])]
        if not collection_ids:
            return []
        res = self.read_client.table("collections").select("*").in_(
            "id", collection_ids
        ).eq("user_id", self.user_id).execute()
        return res.data or []

    def scan_conflicts(
        self,
        parties: list,
        firm_id: str = None,
        exclude_collection_id: str = None,
    ) -> list:
        """Metadata-only conflict scan (F1c) — the ethical-wall screen across the FIRM.

        THE WALL: this reads ONLY matter metadata (`collections.parties` JSONB — party names +
        roles). It NEVER reads `documents` or `document_chunks`; no document content is
        touched. It compares the new matter's `parties` against the OTHER matters in the same
        firm and returns adverse-party / same-party findings (see `find_conflicts`).

        Firm scope: a conflict screen must see OTHER users' matters in the firm (that is the
        whole point), which crosses the per-user RLS boundary. So this uses the service-role
        client (`self.client`, RLS-exempt) but is HARD-SCOPED to `firm_id` and selects ONLY the
        metadata columns — the narrow, audited cross-user read. With no `firm_id` (a user with
        no firm yet) there is nothing firm-wide to screen against → returns []. Never raises;
        a failure degrades to [] (non-blocking — a conflict scan never blocks create).
        """
        try:
            if not parties:
                return []
            # Defense-in-depth: NEVER trust a passed firm_id. Resolve the caller's OWN firm
            # server-side and use that as the hard scope. A passed firm_id is honored ONLY if it
            # equals the caller's firm; anything else falls back to the caller's firm (so a
            # future mis-wiring that forwards a client value cannot read another firm's matters).
            own = (self.get_user_firm() or {}).get("id")
            if not own:
                return []  # no firm → no firm-wide matters to screen against
            fid = firm_id if (firm_id and firm_id == own) else own
            # METADATA ONLY: select party + label columns; never a content table.
            q = self.client.table("collections").select(
                "id, name, matter_kind, parties"
            ).eq("firm_id", fid)
            if exclude_collection_id:
                q = q.neq("id", exclude_collection_id)
            res = q.execute()
            existing = [
                {
                    "collection_id": r.get("id"),
                    "name": r.get("name"),
                    "matter_kind": r.get("matter_kind"),
                    "parties": r.get("parties") or [],
                }
                for r in (res.data or [])
            ]
            from src.components.practice_templates import find_conflicts
            return find_conflicts(parties, existing)
        except Exception as exc:  # noqa: BLE001 — non-blocking; scan failure must not break create
            import logging
            logging.getLogger(__name__).warning("Conflict scan failed (non-blocking): %s", exc)
            return []