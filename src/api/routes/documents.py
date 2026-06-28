"""
Document management endpoints — upload (async via Celery), list, delete.

Note on tmp files: We use tempfile.mkstemp() instead of a fixed tmp_uploads/
directory because Railway has an ephemeral filesystem — the directory is lost
on redeploy. mkstemp() writes to /tmp which always exists in the container.
"""

import asyncio
import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse

from src.api.schemas import DocumentResponse, DocumentListResponse, UpdateDocumentRequest
from src.api.dependencies import (
    get_current_user, get_user_config, limiter, require_cap, assert_vault_not_screened,
)
from src.components.config import Config
from src.components.metrics import uploads_total
from src.api.routes.audit import log_audit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")

# MIME magic bytes for allowed file types (S6)
_MAGIC_BYTES: dict[str, bytes] = {
    "pdf":  b"%PDF",
    "docx": b"PK\x03\x04",   # ZIP container (Office Open XML)
    "pptx": b"PK\x03\x04",
    "xlsx": b"PK\x03\x04",
    "txt":  None,              # plain text — no magic bytes
}

def _validate_mime(file_ext: str, header: bytes) -> bool:
    """Return True if file header matches expected magic bytes."""
    magic = _MAGIC_BYTES.get(file_ext)
    if magic is None:
        return True   # txt: accept any
    return header.startswith(magic)

# Env-overridable (MAX_UPLOAD_MB): real-world annual reports run 10–25MB+
# (Indian test corpus: 8–25MB). 50MB also matches Supabase Storage's default
# per-object cap, so anything accepted here can actually land in the bucket.
MAX_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))


@router.post("/upload", status_code=202)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,                            # P1: required by slowapi
    file: UploadFile = File(...),
    collection_id: str = Form(None),
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    _cap=Depends(require_cap("ingest")),
):
    """
    Upload a document and immediately return 202 Accepted.
    Processing (ingest -> chunk -> embed) happens in the background.
    Poll GET /documents to check when status changes from 'processing' -> 'ready'.

    F2m (D0 — the PRODUCTIVITY grant): when `collection_id` is a SHARED matter (a vault the
    caller is staffed on but doesn't own), the doc is stamped with the VAULT OWNER's user_id
    + processed into the OWNER's Pinecone namespace, and auto-linked to the matter — so a
    paralegal's upload lands IN the matter for the whole team, never orphaned in their own
    space. This is the point of sharing: lower-tier members do real work on the matter. The
    `ingest` cap (held by paralegals/assistants per D0) + accessible_vault_owner (authorizes
    the share, fails closed on an ethical wall) gate it. Own-vault / no collection_id ⇒
    byte-identical to the legacy path (owner == caller).
    """
    # F2m: resolve WHOSE vault this upload belongs to. For an own vault or no collection_id,
    # owner == caller (legacy). For a shared matter, owner is the matter owner; None means the
    # caller has no access (non-staffed / screened) → refuse before writing anything.
    owner_id = sb.user_id
    if collection_id:
        # Ethical wall floor (P7-equivalent for the write path): a screened member cannot
        # ingest into a walled matter. Fails CLOSED on a screen-lookup fault.
        assert_vault_not_screened(sb, collection_id)
        owner_id = sb.accessible_vault_owner(collection_id)
        if not owner_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this matter, so you can't upload into it.",
            )
    owner_namespace = owner_id  # PINECONE_NAMESPACE is the owner's user_id (the matter's vectors)

    # Validate file type
    safe_filename = Path(file.filename).name  # Strip directory components (S5)
    file_ext = Path(safe_filename).suffix.lower().strip(".")
    logger.info("Upload request received: %s (ext=%s, user=%s)", safe_filename, file_ext, getattr(sb, 'user_id', 'unknown'))
    if file_ext not in user_config.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_ext}. Supported: {list(user_config.SUPPORTED_FILE_TYPES)}",
        )

    # P2: Stream file to disk in chunks to avoid large memory spikes.
    # Use tempfile.mkstemp() — Railway has an ephemeral filesystem so
    # a fixed tmp_uploads/ directory disappears between deploys.
    suffix = Path(safe_filename).suffix  # preserve extension for unstructured
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    file_size = 0
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    first_chunk = True

    os.close(tmp_fd)  # close the OS-level fd; aiofiles will reopen
    logger.info("Streaming file to disk: %s", tmp_path)
    async with aiofiles.open(tmp_path, "wb") as out:
        while True:
            chunk = await file.read(8192)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > max_bytes:
                await out.close()
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB",
                )
            if first_chunk:
                # S6: validate MIME magic bytes on first chunk
                if not _validate_mime(file_ext, chunk):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=400,
                        detail="File content does not match its extension.",
                    )
                first_chunk = False
            await out.write(chunk)

    # 1. DO NOT upload to Supabase Storage on the request path. The Storage upload
    # is slow and unreliable (observed: an 18MB file took >141s and still timed out
    # → 504 → the user waited minutes and the doc never processed). The user must
    # NEVER wait on Storage. Instead: the file is already on local disk (the chunk
    # write above, ~0.3s), so we hand the WORKER the local path; the worker uploads
    # to Storage in the background (for durable re-download / sharing) AND processes.
    # API and worker share this host's filesystem, so the path is directly readable.
    #
    # Move the temp file into a stable spool dir the worker owns (out of the
    # request-scoped NamedTemporaryFile lifetime), keyed by a fresh id.
    spool_dir = os.getenv("UPLOAD_SPOOL_DIR", "/tmp/docquery_spool")
    os.makedirs(spool_dir, exist_ok=True)
    # file_ext is dot-stripped (line ~68: "pdf"); the parser dispatches on the
    # extension, so the spool file MUST carry a real ".pdf" suffix.
    spool_path = os.path.join(spool_dir, f"{uuid.uuid4().hex}.{file_ext}")
    try:
        os.replace(tmp_path, spool_path)  # atomic on same fs; no re-read of bytes
    except OSError:
        # cross-device fallback: copy then remove
        import shutil
        shutil.copy2(tmp_path, spool_path)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    logger.info("File spooled locally (%d bytes) → %s; deferring storage upload to worker",
                file_size, spool_path)

    # The canonical storage_path the worker will upload TO (and later download FROM).
    # F2m: key it by the OWNER (the matter's namespace) on a shared upload, so storage +
    # vectors + db row all live under the same id.
    storage_path = f"{owner_id}/{safe_filename}"

    # 2. Create document record with status=processing, owned by the matter owner (F2m).
    doc_record = sb.create_document_record(
        filename=safe_filename,
        storage_path=storage_path,
        file_type=file_ext,
        file_size_bytes=file_size,
        owner_user_id=owner_id,
    )
    doc_id = doc_record.get("id")
    logger.info("DB record created for %s: doc_id=%s (owner=%s%s)", safe_filename, doc_id,
                owner_id, " [shared matter]" if owner_id != sb.user_id else "")

    # F2m: auto-link the doc to the matter server-side (atomic + owner-correct). The frontend's
    # separate addDocToCollection still works for own vaults, but on a SHARED matter the caller's
    # add would mis-scope ownership — so we link it here, where the owner is already resolved.
    if collection_id:
        try:
            sb.add_document_to_collection(collection_id, doc_id)
        except Exception as exc:  # noqa: BLE001 — link failure must not lose the upload
            logger.warning("Could not link doc %s to collection %s: %s", doc_id, collection_id, exc)

    # 3. Dispatch to Celery worker — route to priority queue based on file size
    from src.worker.tasks import process_document_task
    from src.api.dependencies import resolve_membership

    if file_size < 500_000:           # < 500 KB → fast queue (txt, tiny PDFs)
        celery_queue = "documents.fast"
    elif file_size < 5_000_000:       # < 5 MB  → normal queue
        celery_queue = "documents.normal"
    else:                             # ≥ 5 MB  → heavy queue (large PDFs)
        celery_queue = "documents.heavy"

    # F-B: snapshot the uploader's ethical-wall state at enqueue time so the worker can
    # re-check it inside the task (the worker is otherwise authz-blind — service-role,
    # user_id only). resolve_membership is memoized per-request so this is a cached read.
    try:
        _membership = resolve_membership(sb)
        _enqueue_firm_id = _membership.firm_id or None
        _enqueue_screened = list(_membership.screened_vault_ids)
    except Exception:  # noqa: BLE001 — a lookup failure must not block the upload
        _enqueue_firm_id = None
        _enqueue_screened = []

    process_document_task.apply_async(
        kwargs=dict(
            filename=safe_filename,
            doc_id=doc_id,
            storage_path=storage_path,
            # F2m: process under the OWNER so chunks land in the matter's namespace (not the
            # uploader's). For an own upload owner_id == sb.user_id ⇒ unchanged.
            user_id=owner_id,
            pinecone_namespace=owner_namespace,
            local_path=spool_path,  # worker reads bytes from here; uploads to storage itself
            collection_id=collection_id or None,
            # F-B: ethical-wall snapshot (firm + screened vaults at enqueue time).
            firm_id=_enqueue_firm_id,
            screened_vault_ids=_enqueue_screened,
        ),
        queue=celery_queue,
    )

    # NOTE: do NOT remove the spooled file here — the worker owns it now (it reads
    # the bytes for processing and uploads them to Storage, then deletes it).

    # Phase 2: Invalidate semantic cache — uploaded docs change what answers are valid
    try:
        import os as _os
        from src.components.semantic_cache import SemanticCache
        _cache = SemanticCache(
            redis_url=_os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            namespace=owner_namespace,  # F2m: bust the MATTER's cache (owner ns), not the uploader's
        )
        _cache.invalidate_namespace()
    except Exception:
        pass  # cache invalidation failure is non-fatal

    log_audit(sb, "document.upload", "document", doc_id,
              {"filename": safe_filename, "file_size_bytes": file_size,
               "collection_id": collection_id, "owner_id": owner_id,
               "shared_matter": owner_id != sb.user_id})

    # 4. Return 202 Accepted immediately — client should poll GET /documents
    return JSONResponse(
        status_code=202,
        content={
            "id": doc_id,
            "filename": safe_filename,
            "file_type": file_ext,
            "status": "processing",
            "message": "Document accepted for processing. Poll GET /documents to check status.",
            "file_size_bytes": file_size,
        },
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(sb=Depends(get_current_user)):
    """List all documents for the current user."""
    docs = sb.get_user_documents()
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d["id"],
                filename=d["filename"],
                file_type=d.get("file_type"),
                status=d["status"],
                chunk_count=d.get("chunk_count", 0),
                file_size_bytes=d.get("file_size_bytes"),
                created_at=d.get("created_at"),
                processing_progress=d.get("processing_progress", 0),
                doc_type=d.get("doc_type"),
                fidelity=d.get("fidelity"),
                privileged=bool(d.get("privileged", False)),
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.patch("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    body: UpdateDocumentRequest,
    sb=Depends(get_current_user),
):
    """F1e: mark/unmark a document as privileged (attorney-client / work-product).

    A privileged doc is excluded from shared / cross-vault surfaces and watermarked in exports.
    Ownership is enforced by the db method's `.eq(user_id)`; a doc the user doesn't own → 404.
    F2 will partner-gate WHO may set this; F1e ships the control + the data-layer flag.
    """
    existing = sb.get_document(doc_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        updated = sb.set_document_privileged(doc_id, body.privileged) or existing
    except Exception:
        logger.exception("Failed to set privilege on document %s", doc_id)
        raise HTTPException(status_code=500, detail="Failed to update document.")
    log_audit(sb, "document.set_privileged", "document", doc_id, {"privileged": body.privileged})
    return DocumentResponse(
        id=updated["id"],
        filename=updated["filename"],
        file_type=updated.get("file_type"),
        status=updated["status"],
        chunk_count=updated.get("chunk_count", 0),
        file_size_bytes=updated.get("file_size_bytes"),
        created_at=updated.get("created_at"),
        processing_progress=updated.get("processing_progress", 0),
        doc_type=updated.get("doc_type"),
        fidelity=updated.get("fidelity"),
        privileged=bool(updated.get("privileged", False)),
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    _cap=Depends(require_cap("delete")),
):
    """Delete a document: removes vectors from Pinecone, file from Storage, and DB record.

    F2b: `require_cap("delete")` 403s a member without the delete capability before any cleanup
    runs (T9 — the app-layer guard is load-bearing on the service-role write client until F2f's
    row RLS lands under it). Solo-MP / legacy users are allowed (byte-identical to pre-F2)."""
    from src.components.retrieval import RetrievalManager

    try:
        # Get document record to find filename
        doc = sb.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        filename = doc["filename"]
        storage_path = doc["storage_path"]

        # Each backend cleanup is independent and best-effort: a transient flake on
        # one store (e.g. a Supabase Storage HTTP/2 ConnectionTerminated when many
        # deletes fire at once) must NOT abort the others or leave the doc half-gone.
        # We track failures and only fail the request if the AUTHORITATIVE record
        # deletion (the row the UI lists from) did not succeed. The storage blob and
        # vector/chunk rows are reconcilable orphans, not user-visible state.
        errors: List[str] = []

        retrieval_mgr = RetrievalManager(user_config)
        try:
            retrieval_mgr.delete_document_by_filename(filename)
        except Exception:
            logger.exception("delete: vector cleanup failed for %s", doc_id)
            errors.append("vectors")

        for step, fn in (
            ("storage", lambda: sb.delete_file(storage_path)),
            ("chunks", lambda: sb.delete_document_chunks(doc_id)),
            ("record", lambda: sb.delete_document_record(doc_id)),
        ):
            try:
                fn()
            except Exception:
                logger.exception("delete: %s cleanup failed for %s", step, doc_id)
                errors.append(step)

        if "record" in errors:
            # the document is still listed → this is a genuine failure the user must retry
            raise HTTPException(status_code=500, detail="Failed to delete document.")
        if errors:
            logger.warning("delete: doc %s removed with orphaned %s", doc_id, errors)

        # Phase 2: Invalidate semantic cache — deleted doc makes old answers stale
        try:
            from src.components.semantic_cache import SemanticCache
            _cache = SemanticCache(
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                namespace=sb.user_id,
            )
            _cache.invalidate_namespace()
        except Exception:
            pass  # cache invalidation failure is non-fatal

        log_audit(sb, "document.delete", "document", doc_id, {"filename": filename})

        return {"message": f"Document '{filename}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete document %s", doc_id)  # S8: full detail stays server-side
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",  # S8: generic message to client
        )
