"""
Supabase integration for DocQuery.
Handles: Auth, Storage (file uploads), Conversations, Messages, Documents metadata.

Place this file at: src/components/db.py
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file")
    return create_client(url, key)


class SupabaseManager:
    """
    Central class for all Supabase operations in DocQuery.
    One instance per Streamlit session is enough.
    """

    BUCKET = "docquery-files"

    def __init__(self):
        self.client: Client = get_supabase_client()
        self._user = None  # set after login

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

    def upload_pkl(self, pkl_bytes: bytes, filename: str) -> str:
        """Upload a pickle cache file to storage."""
        if not self.user_id:
            raise ValueError("User must be logged in.")
        storage_path = f"{self.user_id}/cache/{filename}"
        self.client.storage.from_(self.BUCKET).upload(
            path=storage_path,
            file=pkl_bytes,
            file_options={"upsert": "true"},
        )
        return storage_path

    def download_pkl(self, filename: str) -> Optional[bytes]:
        """Download a pickle cache file. Returns None if not found."""
        if not self.user_id:
            return None
        storage_path = f"{self.user_id}/cache/{filename}"
        try:
            return self.client.storage.from_(self.BUCKET).download(storage_path)
        except Exception:
            return None

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

    def update_document_status(self, doc_id: str, status: str, chunk_count: int = 0):
        self.client.table("documents").update({
            "status": status,
            "chunk_count": chunk_count,
        }).eq("id", doc_id).execute()

    def get_user_documents(self) -> list:
        """Get all documents for current user."""
        if not self.user_id:
            return []
        res = self.client.table("documents").select("*").eq(
            "user_id", self.user_id
        ).order("created_at", desc=True).execute()
        return res.data or []

    def delete_document_record(self, filename: str):
        """Delete document record by filename for current user."""
        self.client.table("documents").delete().eq(
            "user_id", self.user_id
        ).eq("filename", filename).execute()

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
        self.client.table("conversations").update({
            "title": new_title
        }).eq("id", conversation_id).execute()

    def delete_conversation(self, conversation_id: str):
        # Messages are deleted by CASCADE in DB
        self.client.table("conversations").delete().eq(
            "id", conversation_id
        ).execute()

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
        # Bump conversation updated_at so it rises to top of list
        self.client.table("conversations").update(
            {"updated_at": "now()"}
        ).eq("id", conversation_id).execute()
        return res.data[0] if res.data else {}

    def get_messages(self, conversation_id: str) -> list:
        """Load all messages for a conversation, oldest first."""
        res = self.client.table("messages").select("*").eq(
            "conversation_id", conversation_id
        ).order("created_at", desc=False).execute()
        return res.data or []

    def auto_title_conversation(self, conversation_id: str, first_question: str):
        """Set conversation title from first user message (truncated)."""
        title = first_question[:50] + ("..." if len(first_question) > 50 else "")
        self.rename_conversation(conversation_id, title)