"""Connector endpoints (G8 §G8.6) — import files from an external source into the vault.

A connector is a SOURCE, not a parser: each imported file is spooled to local disk and
handed to the SAME `process_document_task` a manual upload uses, so it gets the identical
G1 ingestion (classify_document + fidelity receipt + doc-type chip + fidelity dot). No new
ingestion path.

v1 surfaces Google Drive (import a folder the user picks) using a per-request OAuth access
token from the browser-side Google token flow — no server-stored secret/refresh token.
Email ingestion is the planned second connector (same dispatch, different source).

All routes are behind `config.USE_CONNECTORS` (off ⇒ 404 ⇒ byte-identical to pre-G8.6).
"""

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.dependencies import get_current_user, get_user_config, limiter
from src.api.routes.audit import log_audit
from src.components.config import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connectors")


def _require_connectors_enabled(config: Config) -> None:
    """404 (not 403) when the feature is off — the route simply doesn't exist, the
    byte-identical guarantee. Mirrors how the other flag-gated routes degrade."""
    if not getattr(config, "USE_CONNECTORS", False):
        raise HTTPException(status_code=404, detail="Connectors are not enabled.")


def _spool_and_dispatch(file_bytes: bytes, filename: str, sb, user_config: Config) -> str:
    """Spool one in-memory file to the worker's spool dir and dispatch the SAME Celery
    ingestion task a manual upload uses. Returns the new vault doc_id.

    This is the shared tail of `documents.py::upload_document` (record → spool → dispatch),
    so an imported file and an uploaded file are processed identically — the §G8.6 rule.
    """
    safe_filename = Path(filename).name
    file_ext = Path(safe_filename).suffix.lower().strip(".")
    if file_ext not in user_config.SUPPORTED_FILE_TYPES:
        raise ValueError(f"unsupported file type: .{file_ext}")

    spool_dir = os.getenv("UPLOAD_SPOOL_DIR", "/tmp/docquery_spool")
    os.makedirs(spool_dir, exist_ok=True)
    spool_path = os.path.join(spool_dir, f"{uuid.uuid4().hex}.{file_ext}")
    with open(spool_path, "wb") as fh:
        fh.write(file_bytes)

    storage_path = f"{sb.user_id}/{safe_filename}"
    doc_record = sb.create_document_record(
        filename=safe_filename,
        storage_path=storage_path,
        file_type=file_ext,
        file_size_bytes=len(file_bytes),
    )
    doc_id = doc_record.get("id")

    from src.worker.tasks import process_document_task
    size = len(file_bytes)
    queue = ("documents.fast" if size < 500_000
             else "documents.normal" if size < 5_000_000
             else "documents.heavy")
    process_document_task.apply_async(
        kwargs=dict(
            filename=safe_filename,
            doc_id=doc_id,
            storage_path=storage_path,
            user_id=sb.user_id,
            pinecone_namespace=user_config.PINECONE_NAMESPACE,
            local_path=spool_path,
        ),
        queue=queue,
    )
    return doc_id


def _ext_of(name: str) -> str:
    return Path(name).suffix.lower().strip(".")


def _import_via_connector(connector, files, folder_label, sb, user_config, ext_for):
    """The shared connector loop: for each listed file, skip unsupported, else fetch →
    spool → dispatch the existing ingestion task. One bad file never aborts the batch;
    every skip/error is recorded (never silent). Returns an ImportResult.

    `ext_for(remote_file) -> Optional[str]` maps a source file to a vault extension (Drive
    uses MIME, email uses the filename suffix). Used by BOTH the Drive and email routes."""
    from src.components.connectors.base import ImportResult, ImportedFile

    result = ImportResult(source=connector.name, folder=folder_label)
    for rf in files:
        ext = ext_for(rf)
        if not ext or ext not in user_config.SUPPORTED_FILE_TYPES:
            result.files.append(ImportedFile(
                name=rf.name, remote_id=rf.remote_id, status="skipped",
                reason=f"unsupported type ({rf.mime_type or ext or '?'})"))
            continue
        try:
            data = connector.fetch_file(rf)
            doc_id = _spool_and_dispatch(data, connector.vault_filename(rf), sb, user_config)
            result.files.append(ImportedFile(
                name=rf.name, remote_id=rf.remote_id, status="queued", doc_id=doc_id))
        except Exception as exc:  # noqa: BLE001 — one bad file never aborts the batch
            logger.warning("[connector:%s] import %s failed: %s", connector.name, rf.name, exc)
            result.files.append(ImportedFile(
                name=rf.name, remote_id=rf.remote_id, status="error", reason=str(exc)))
    return result


def _finish_import(result, sb, user_config):
    """Shared post-import: log, audit, invalidate cache, shape the response body."""
    logger.info(result.summary())
    log_audit(sb, "connector.import", "connector", result.source,
              {"folder": result.folder, "queued": result.queued,
               "skipped": result.skipped, "errored": result.errored})
    try:
        from src.components.semantic_cache import SemanticCache
        SemanticCache(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            namespace=sb.user_id,
        ).invalidate_namespace()
    except Exception:
        pass
    return {
        "source": result.source,
        "folder": result.folder,
        "queued": result.queued,
        "skipped": result.skipped,
        "errored": result.errored,
        "files": [
            {"name": f.name, "status": f.status, "doc_id": f.doc_id, "reason": f.reason}
            for f in result.files
        ],
        "message": f"{result.queued} file(s) queued for processing. Poll GET /documents.",
    }


class DriveImportRequest(BaseModel):
    access_token: str          # browser-side Google OAuth token (drive.readonly scope)
    folder_id: str             # the Drive folder the user picked
    folder_name: str = ""      # display label (for the receipt; optional)


class EmailImportRequest(BaseModel):
    host: str                  # IMAP host (e.g. imap.gmail.com)
    username: str              # mailbox login
    password: str             # app-password / token (used per-request, never stored)
    mailbox: str = "INBOX"     # folder/label to scan
    port: int = 993
    max_messages: int = 50     # newest N messages scanned for attachments


@router.get("/config")
def connectors_config(
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """What the frontend needs to start the Drive token flow (the public client id +
    which connectors are live). 404 when the feature is off."""
    _require_connectors_enabled(user_config)
    return {
        "enabled": True,
        "google_drive": {
            "client_id": getattr(user_config, "GOOGLE_DRIVE_CLIENT_ID", ""),
            "scope": "https://www.googleapis.com/auth/drive.readonly",
        },
        "email": {"available": True, "default_port": 993},  # IMAP attachment import
    }


@router.post("/google-drive/import")
@limiter.limit("6/minute")
def import_google_drive_folder(
    request: Request,                 # required by slowapi
    body: DriveImportRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """Import every supported file directly under a Google Drive folder into the vault.

    Lists the folder, fetches each file's bytes, and dispatches the EXISTING ingestion
    task per file. Unsupported types (images, sub-folders) are reported skipped — never
    silently dropped. Returns the per-file receipt; the client polls GET /documents for
    processing status (same as a manual upload).
    """
    _require_connectors_enabled(user_config)
    if not body.access_token or not body.folder_id:
        raise HTTPException(status_code=400, detail="access_token and folder_id are required.")

    from src.components.connectors.gdrive import GoogleDriveConnector, ext_for_mime

    try:
        connector = GoogleDriveConnector(
            body.access_token, supported_exts=set(user_config.SUPPORTED_FILE_TYPES))
        remote_files = connector.list_files(body.folder_id)
    except Exception as exc:  # noqa: BLE001 — surface a clean 502, never a 500 stack
        logger.warning("[connector:google_drive] list failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Google Drive list failed: {exc}")

    result = _import_via_connector(
        connector, remote_files, body.folder_name or body.folder_id,
        sb, user_config, ext_for=lambda rf: ext_for_mime(rf.mime_type))
    return _finish_import(result, sb, user_config)


@router.post("/email/import")
@limiter.limit("6/minute")
def import_email_attachments(
    request: Request,                 # required by slowapi
    body: EmailImportRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """Import supported ATTACHMENTS from a mailbox into the vault (G8.6 second connector).

    Scans the newest `max_messages` of `mailbox` over IMAP, surfaces each supported
    attachment, and dispatches the EXISTING ingestion task per file. The email body is not
    ingested in v1 (attachments are the documents). Credentials are used per-request and
    never stored. Returns the per-file receipt; poll GET /documents for status.
    """
    _require_connectors_enabled(user_config)
    if not (body.host and body.username and body.password):
        raise HTTPException(status_code=400, detail="host, username and password are required.")

    from src.components.connectors.email import EmailConnector

    try:
        connector = EmailConnector(
            body.host, body.username, body.password,
            port=body.port, mailbox=body.mailbox,
            supported_exts=set(user_config.SUPPORTED_FILE_TYPES),
            max_messages=body.max_messages)
        remote_files = connector.list_files(body.mailbox)
    except Exception as exc:  # noqa: BLE001 — clean 502, never a 500 stack with creds
        logger.warning("[connector:email] list failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Email scan failed: {exc}")

    result = _import_via_connector(
        connector, remote_files, body.mailbox,
        sb, user_config, ext_for=lambda rf: _ext_of(rf.name))
    return _finish_import(result, sb, user_config)
