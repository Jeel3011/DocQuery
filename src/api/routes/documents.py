"""
Document management endpoints — upload (async via Celery), list, delete.

Note on tmp files: We use tempfile.mkstemp() instead of a fixed tmp_uploads/
directory because Railway has an ephemeral filesystem — the directory is lost
on redeploy. mkstemp() writes to /tmp which always exists in the container.
"""

import os
import logging
import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import JSONResponse

from src.api.schemas import DocumentResponse, DocumentListResponse
from src.api.dependencies import get_current_user, get_user_config, limiter
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

MAX_FILE_SIZE_MB = 10


@router.post("/upload", status_code=202)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,                            # P1: required by slowapi
    file: UploadFile = File(...),
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """
    Upload a document and immediately return 202 Accepted.
    Processing (ingest -> chunk -> embed) happens in the background.
    Poll GET /documents to check when status changes from 'processing' -> 'ready'.
    """
    # Validate file type
    safe_filename = Path(file.filename).name  # Strip directory components (S5)
    file_ext = Path(safe_filename).suffix.lower().strip(".")
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

    # L7 FIX: Read file from disk for Supabase upload instead of keeping
    # a second copy in memory via chunks_buf.
    async with aiofiles.open(tmp_path, "rb") as f:
        file_bytes = await f.read()

    # 1. Upload raw file to Supabase Storage immediately
    try:
        storage_path = sb.upload_file(file_bytes, safe_filename)
    except Exception as e:
        logger.exception("Storage upload failed for %s", safe_filename)  # S8: log detail server-side
        raise HTTPException(status_code=500, detail="Failed to upload file to storage.")  # S8: generic message to client

    # 2. Create document record with status=processing
    doc_record = sb.create_document_record(
        filename=safe_filename,
        storage_path=storage_path,
        file_type=file_ext,
        file_size_bytes=file_size,
    )
    doc_id = doc_record.get("id")

    # 3. Dispatch to Celery worker — route to priority queue based on file size
    from src.worker.tasks import process_document_task

    if file_size < 500_000:           # < 500 KB → fast queue (txt, tiny PDFs)
        celery_queue = "documents.fast"
    elif file_size < 5_000_000:       # < 5 MB  → normal queue
        celery_queue = "documents.normal"
    else:                             # ≥ 5 MB  → heavy queue (large PDFs)
        celery_queue = "documents.heavy"

    process_document_task.apply_async(
        kwargs=dict(
            filename=safe_filename,
            doc_id=doc_id,
            storage_path=storage_path,
            user_id=sb.user_id,
            pinecone_namespace=user_config.PINECONE_NAMESPACE,
        ),
        queue=celery_queue,
    )

    # Clean up local temp file — the worker will download from Supabase Storage
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    # Phase 2: Invalidate semantic cache — uploaded docs change what answers are valid
    try:
        import os as _os
        from src.components.semantic_cache import SemanticCache
        _cache = SemanticCache(
            redis_url=_os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            namespace=sb.user_id,
        )
        _cache.invalidate_namespace()
    except Exception:
        pass  # cache invalidation failure is non-fatal

    log_audit(sb, "document.upload", "document", doc_id, {"filename": safe_filename, "file_size_bytes": file_size})

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
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """Delete a document: removes vectors from Pinecone, file from Storage, and DB record."""
    from src.components.retrieval import RetrievalManager

    try:
        # Get document record to find filename
        doc = sb.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        filename = doc["filename"]
        storage_path = doc["storage_path"]

        retrieval_mgr = RetrievalManager(user_config)
        retrieval_mgr.delete_document_by_filename(filename)

        sb.delete_file(storage_path)
        sb.delete_document_chunks(doc_id)
        sb.delete_document_record(doc_id)

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
