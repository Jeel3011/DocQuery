"""
Document management endpoints — upload (async with BackgroundTasks), list, delete.
"""

import os
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from src.api.schemas import DocumentResponse, DocumentListResponse
from src.api.dependencies import get_current_user, get_user_config
from src.components.config import Config

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


def _process_document_background(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
    tmp_path: str,
    user_config: Config,
    user_id: str,          # B2: pass only the ID, not the request-scoped sb
):
    """
    Background task: ingest → chunk → embed → save chunks to Supabase.
    Updates document status to 'ready' or 'failed' when done.
    Runs AFTER the API has already returned 202 Accepted to the client.
    Creates its own SupabaseManager so it is not tied to the request lifecycle.
    """
    from src.components.data_ingestion import DocumentProcessor
    from src.components.embeddings import EmbeddingManager
    from src.components.db import SupabaseManager

    # B2: fresh service-role client scoped to the validated user_id
    sb = SupabaseManager(use_service_role=True)
    sb._user = type("User", (), {"id": user_id})()

    try:
        # Write to temp file for Unstructured
        os.makedirs(user_config.UPLOAD_DIR, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        # Ingest + chunk
        processor = DocumentProcessor(config=user_config)
        elements = processor.process_documents(file_paths=tmp_path)

        if not elements:
            sb.update_document_status(doc_id, "failed")
            return

        chunks = processor.build_langchain_documents(elements=elements)

        # Embed into Pinecone (vector store)
        embed_mgr = EmbeddingManager(config=user_config)
        embed_mgr.create_vector_store(chunks)

        # Persist text chunks to Supabase (replaces insecure .pkl caching)
        sb.save_document_chunks(doc_id, chunks)

        # Mark document as ready
        sb.update_document_status(doc_id, "ready", len(chunks))

    except Exception as e:
        logger.exception("Background processing failed for doc %s", doc_id)
        sb.update_document_status(doc_id, "failed")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@router.post("/upload", status_code=202)
async def upload_document(
    request: Request,                            # P1: required by slowapi
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """
    Upload a document and immediately return 202 Accepted.
    Processing (ingest → chunk → embed) happens in the background.
    Poll GET /documents to check when status changes from 'processing' → 'ready'.
    """
    # Validate file type
    safe_filename = Path(file.filename).name  # Strip directory components (S5)
    file_ext = Path(safe_filename).suffix.lower().strip(".")
    if file_ext not in user_config.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_ext}. Supported: {list(user_config.SUPPORTED_FILE_TYPES)}",
        )

    # P2: Stream file to disk in chunks to avoid large memory spikes
    os.makedirs(user_config.UPLOAD_DIR, exist_ok=True)
    tmp_path = os.path.join(user_config.UPLOAD_DIR, safe_filename)
    file_size = 0
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    first_chunk = True
    chunks_buf: list[bytes] = []

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
            chunks_buf.append(chunk)
            await out.write(chunk)

    file_bytes = b"".join(chunks_buf)

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

    # 3. Schedule background processing — B2: pass user_id string, not the request-scoped sb
    background_tasks.add_task(
        _process_document_background,
        file_bytes=file_bytes,
        filename=safe_filename,
        doc_id=doc_id,
        tmp_path=tmp_path,
        user_config=user_config,
        user_id=sb.user_id,
    )

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
        sb.delete_document_chunks(doc_id)      # BX: remove orphaned text chunks from Supabase
        sb.delete_document_record(doc_id)

        return {"message": f"Document '{filename}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete document %s", doc_id)  # S8: full detail stays server-side
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",  # S8: generic message to client
        )
