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

    def download_file_to_temp(self, storage_path: str, suffix: str = "") -> str:
        """
        Download file from Supabase Storage to a local temp file.
        Returns the local temp file path so the pipeline can process it.
        """
        file_bytes = self.client.storage.from_(self.BUCKET).download(storage_path)
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
                               chunk_count: int = 0, progress_pct: int = None):
        """Update document processing status.

        Args:
            doc_id: Document UUID.
            status: 'processing', 'ready', or 'failed'.
            chunk_count: Number of chunks (set when status='ready').
            progress_pct: Optional 0-100 progress percentage shown during processing.
        """
        update_data = {
            "status": status,
            "chunk_count": chunk_count,
        }
        self.client.table("documents").update(
            update_data
        ).eq("id", doc_id).eq("user_id", self.user_id).execute()

    def get_user_documents(self) -> list:
        """Get all documents for current user."""
        if not self.user_id:
            return []
        res = self.client.table("documents").select("*").eq(
            "user_id", self.user_id
        ).order("created_at", desc=True).execute()
        return res.data or []

    def get_document(self, doc_id: str) -> dict:
        """Get a document record by ID for current user."""
        if not self.user_id:
            return {}
        res = self.client.table("documents").select("*").eq(
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
        res = self.client.table("conversations").select("*").eq(
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
        res = self.client.table("messages").select("*").eq(
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

    def create_collection(self, name: str, description: str = None) -> dict:
        """Create a new collection for the current user."""
        if not self.user_id:
            raise ValueError("User must be logged in.")
        res = self.client.table("collections").insert({
            "user_id": self.user_id,
            "name": name,
            "description": description,
        }).execute()
        return res.data[0] if res.data else {}

    def get_collections(self) -> list:
        """Get all collections for the current user."""
        if not self.user_id:
            return []
        res = self.client.table("collections").select("*").eq(
            "user_id", self.user_id
        ).order("updated_at", desc=True).execute()
        return res.data or []

    def get_collection(self, collection_id: str) -> dict:
        """Get a single collection by ID for the current user."""
        if not self.user_id:
            return {}
        res = self.client.table("collections").select("*").eq(
            "user_id", self.user_id
        ).eq("id", collection_id).execute()
        return res.data[0] if res.data else {}

    def update_collection(self, collection_id: str, name: str = None, description: str = None) -> dict:
        """Rename or update description of a collection."""
        if not self.user_id:
            return {}
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
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
        # Join through collections to ensure this user owns the collection
        res = self.client.table("collection_documents").select(
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
        res = self.client.table("documents").select("*").in_(
            "id", doc_ids
        ).eq("user_id", self.user_id).execute()
        return res.data or []

    def get_document_collections(self, document_id: str) -> list:
        """Get all collections that contain a specific document."""
        res = self.client.table("collection_documents").select(
            "collection_id"
        ).eq("document_id", document_id).execute()
        collection_ids = [row["collection_id"] for row in (res.data or [])]
        if not collection_ids:
            return []
        res = self.client.table("collections").select("*").in_(
            "id", collection_ids
        ).eq("user_id", self.user_id).execute()
        return res.data or []