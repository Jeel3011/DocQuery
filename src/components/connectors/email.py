"""Email connector (G8 §G8.6) — emails → docs in the vault (the planned SECOND connector).

Connects to an IMAP mailbox, scans recent messages, and surfaces each supported ATTACHMENT
as an importable file. Like the Drive connector it is a SOURCE only: the route fetches the
attachment bytes and hands them to the SAME ingestion task — no new parser path.

v1 uses IMAP with an app-password / token the user supplies per-request (stdlib `imaplib`,
no new dependency, no stored credential). A future v2 can add OAuth + a forwarding address
behind the same `Connector` contract.

The body of an email is intentionally NOT ingested in v1 — only attachments, which are the
actual documents (contracts, statements). Ingesting message bodies is a deliberate later
choice (noise vs signal), flagged not silently skipped.
"""

from __future__ import annotations

import email
import imaplib
import logging
from email.message import Message
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import RemoteFile

logger = logging.getLogger(__name__)

# Extensions the vault pipeline ingests; attachments outside this set are reported skipped.
_DEFAULT_SUPPORTED = {"pdf", "docx", "pptx", "xlsx", "txt"}


class EmailConnector:
    """Read supported attachments from an IMAP mailbox. Never ingests.

    A `RemoteFile.remote_id` here encodes "<message_uid>:<attachment_index>" so the route
    can fetch exactly one attachment back out without re-listing.
    """

    name = "email"

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int = 993,
        mailbox: str = "INBOX",
        supported_exts: Optional[set] = None,
        max_messages: int = 50,
    ) -> None:
        if not (host and username and password):
            raise ValueError("EmailConnector requires host, username and password")
        self._host = host
        self._port = port
        self._user = username
        self._password = password
        self._mailbox = mailbox
        self._supported = supported_exts or set(_DEFAULT_SUPPORTED)
        self._max_messages = max_messages

    # ── connection (isolated so the gate stubs them) ────────────────────────────────
    def _connect(self) -> imaplib.IMAP4_SSL:
        conn = imaplib.IMAP4_SSL(self._host, self._port)
        conn.login(self._user, self._password)
        conn.select(self._mailbox, readonly=True)
        return conn

    @staticmethod
    def _iter_attachments(msg: Message):
        """Yield (filename, payload_bytes) for each attachment part of a message."""
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue
            fname = part.get_filename()
            if not fname:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            yield fname, payload

    # ── the Connector contract ──────────────────────────────────────────────────────
    def list_files(self, folder: str = "") -> List[RemoteFile]:
        """List supported attachments across recent messages. `folder` overrides the
        mailbox when given (e.g. a label). Non-recursive; newest `max_messages` scanned."""
        conn = self._connect()
        try:
            if folder:
                conn.select(folder, readonly=True)
            typ, data = conn.search(None, "ALL")
            uids = (data[0].split() if data and data[0] else [])
            uids = uids[-self._max_messages:]  # newest N
            out: List[RemoteFile] = []
            for uid in reversed(uids):
                typ, msg_data = conn.fetch(uid, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                msg = email.message_from_bytes(msg_data[0][1])
                uid_s = uid.decode() if isinstance(uid, bytes) else str(uid)
                for idx, (fname, payload) in enumerate(self._iter_attachments(msg)):
                    ext = Path(fname).suffix.lower().strip(".")
                    out.append(RemoteFile(
                        remote_id=f"{uid_s}:{idx}",
                        name=fname,
                        mime_type=ext,  # email has no reliable MIME; use the extension
                        size_bytes=len(payload),
                    ))
            return out
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def fetch_file(self, remote: RemoteFile) -> bytes:
        """Re-fetch one attachment's bytes by its "<uid>:<index>" remote_id."""
        uid_s, _, idx_s = remote.remote_id.partition(":")
        target_idx = int(idx_s or 0)
        conn = self._connect()
        try:
            typ, msg_data = conn.fetch(uid_s.encode(), "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                raise ValueError(f"message {uid_s} not retrievable")
            msg = email.message_from_bytes(msg_data[0][1])
            for idx, (_fname, payload) in enumerate(self._iter_attachments(msg)):
                if idx == target_idx:
                    return payload
            raise ValueError(f"attachment {target_idx} not found in message {uid_s}")
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @staticmethod
    def vault_filename(remote: RemoteFile) -> str:
        """The attachment filename, directory components stripped."""
        return Path(remote.name).name
