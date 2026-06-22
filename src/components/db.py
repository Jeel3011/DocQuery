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
                                file_type: str, file_size_bytes: int) -> dict:
        res = self.client.table("documents").insert({
            "user_id": self.user_id,
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

    def get_user_firm(self, user_id: str = None) -> dict:
        """Return the firm this user belongs to (F1a multi-firm tenancy), or {} if none.

        Server-side lookup (not a JWT claim) so the auth flow is unchanged. Returns the first
        membership's firm; F2 will formalize multi-firm membership + the active-firm choice.
        """
        uid = user_id or self.user_id
        if not uid:
            return {}
        res = self.read_client.table("firm_memberships").select(
            "firm_id, firms!inner(id, name)"
        ).eq("user_id", uid).limit(1).execute()
        rows = res.data or []
        return rows[0].get("firms", {}) if rows else {}

    def get_collections(self) -> list:
        """Get all collections for the current user."""
        if not self.user_id:
            return []
        res = self.read_client.table("collections").select("*").eq(
            "user_id", self.user_id
        ).order("updated_at", desc=True).execute()
        return res.data or []

    def get_collection(self, collection_id: str) -> dict:
        """Get a single collection by ID for the current user."""
        if not self.user_id:
            return {}
        res = self.read_client.table("collections").select("*").eq(
            "user_id", self.user_id
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

    def get_collection_document_ids(self, collection_id: str) -> list[str]:
        """Get all document IDs in a collection, verifying ownership via collections table."""
        # Join through collections to ensure this user owns the collection.
        # RLS backstop: collection_documents' policy already restricts to the user's own
        # collections, so the join runs RLS-enforced under read_client too.
        res = self.read_client.table("collection_documents").select(
            "document_id, collections!inner(user_id)"
        ).eq("collection_id", collection_id).eq(
            "collections.user_id", self.user_id
        ).execute()
        return [row["document_id"] for row in (res.data or [])]

    def get_collection_documents(self, collection_id: str) -> list:
        """Get full document records for all documents in a collection."""
        doc_ids = self.get_collection_document_ids(collection_id)
        if not doc_ids:
            return []
        res = self.read_client.table("documents").select("*").in_(
            "id", doc_ids
        ).eq("user_id", self.user_id).execute()
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