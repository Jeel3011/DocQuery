"""Google Drive connector (G8 §G8.6) — vault import FIRST (India mid-market reality).

Lists a Drive folder and fetches each file's bytes via the Drive REST API, using an OAuth
**access token** the frontend obtains through Google's browser-side token flow (Google
Identity Services / the Picker). v1 deliberately does NOT store refresh tokens or a server
client-secret: the token is short-lived, passed per-request, and used only to read the
folder the user just picked. This keeps the connector a thin SOURCE (no OAuth secret
vault, no background sync) — exactly the §G8.6 scope. A future v2 can add stored refresh
tokens + scheduled sync behind the same `Connector` contract.

MIME mapping: native PDFs/Office files download as-is; Google-native Docs/Sheets/Slides
are EXPORTED to the portable Office types the ingestion pipeline already accepts (docx /
xlsx / pptx). Anything else (images, folders, unknown types) is reported skipped — never
silently dropped (the ImportResult lists every skip + reason).

Network calls go through the stdlib (`urllib`) so the connector adds no new dependency;
they are isolated in `_get`/`_download` so the gate can stub them ($0, offline).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from .base import RemoteFile

logger = logging.getLogger(__name__)

_DRIVE_API = "https://www.googleapis.com/drive/v3"

# Drive MIME → (vault extension, optional export MIME for Google-native docs).
# A native binary (pdf/docx/…) has export_mime=None (downloaded as-is). A Google-native
# doc is exported to the matching Office type the pipeline ingests.
_MIME_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "application/pdf": ("pdf", None),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("docx", None),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ("xlsx", None),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ("pptx", None),
    "text/plain": ("txt", None),
    # Google-native → export to Office formats the pipeline accepts.
    "application/vnd.google-apps.document": (
        "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "application/vnd.google-apps.spreadsheet": (
        "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "application/vnd.google-apps.presentation": (
        "pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}


def ext_for_mime(mime: str) -> Optional[str]:
    """The vault file extension for a Drive MIME, or None if unsupported."""
    entry = _MIME_MAP.get(mime)
    return entry[0] if entry else None


class GoogleDriveConnector:
    """Read a Drive folder with a per-request OAuth access token. Never ingests."""

    name = "google_drive"

    def __init__(self, access_token: str, *, supported_exts: Optional[set] = None) -> None:
        if not access_token:
            raise ValueError("GoogleDriveConnector requires an OAuth access token")
        self._token = access_token
        # Only import file types the vault pipeline actually ingests (defaults match
        # Config.SUPPORTED_FILE_TYPES; injected so the connector tracks that set).
        self._supported = supported_exts or {"pdf", "docx", "pptx", "xlsx", "txt"}

    # ── network (isolated so the gate stubs them) ───────────────────────────────────
    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
            return json.loads(resp.read().decode("utf-8"))

    def _download(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._token}"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return resp.read()

    # ── the Connector contract ──────────────────────────────────────────────────────
    def list_files(self, folder: str) -> List[RemoteFile]:
        """List importable files directly under `folder` (non-recursive v1). Folders and
        trashed files are excluded by the query; unsupported MIME types are still returned
        so the route can report them as skipped (never a silent drop)."""
        q = urllib.parse.quote(f"'{folder}' in parents and trashed = false")
        fields = urllib.parse.quote("files(id,name,mimeType,size)")
        out: List[RemoteFile] = []
        page_token: Optional[str] = None
        while True:
            url = (f"{_DRIVE_API}/files?q={q}&fields={fields},nextPageToken"
                   f"&pageSize=200&supportsAllDrives=true&includeItemsFromAllDrives=true")
            if page_token:
                url += f"&pageToken={urllib.parse.quote(page_token)}"
            data = self._get(url)
            for f in data.get("files", []):
                mime = f.get("mimeType", "")
                if mime == "application/vnd.google-apps.folder":
                    continue  # non-recursive v1 — skip sub-folders
                out.append(RemoteFile(
                    remote_id=f.get("id", ""),
                    name=f.get("name", "unnamed"),
                    mime_type=mime,
                    size_bytes=int(f.get("size") or 0),
                ))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return out

    def fetch_file(self, remote: RemoteFile) -> bytes:
        """Download (or export) one file's bytes. Raises on an unsupported MIME so the
        route records an explicit error rather than spooling garbage."""
        entry = _MIME_MAP.get(remote.mime_type)
        if entry is None:
            raise ValueError(f"unsupported Drive MIME: {remote.mime_type}")
        _ext, export_mime = entry
        rid = urllib.parse.quote(remote.remote_id)
        if export_mime:
            url = f"{_DRIVE_API}/files/{rid}/export?mimeType={urllib.parse.quote(export_mime)}"
        else:
            url = f"{_DRIVE_API}/files/{rid}?alt=media&supportsAllDrives=true"
        return self._download(url)

    def vault_filename(self, remote: RemoteFile) -> str:
        """The filename the vault should store this under — the remote name, with the
        correct extension appended for an exported Google-native doc (which has no
        extension in Drive). Idempotent if the name already carries the extension."""
        ext = ext_for_mime(remote.mime_type)
        name = remote.name
        if ext and not name.lower().endswith(f".{ext}"):
            # Only append for exported native docs (no extension in Drive); a native pdf
            # already ends in .pdf so this is a no-op.
            if remote.mime_type.startswith("application/vnd.google-apps."):
                name = f"{name}.{ext}"
        return name
