"""
Document management endpoints — upload (async with BackgroundTasks), list, delete.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.schemas import DocumentResponse, DocumentListResponse
from src.api.dependencies import get_current_user, get_user_config
from src.components.config import Config

router = APIRouter(prefix="/documents")

MAX_FILE_SIZE_MB = 10


def _process_document_background(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
    tmp_path: str,
    user_config: Config,
    sb,
):
    """
    Background task: ingest → chunk → embed → save chunks to Supabase.
    Updates document status to 'ready' or 'failed' when done.
    Runs AFTER the API has already returned 202 Accepted to the client.
    """
    from src.components.data_ingestion import DocumentProcessor
    from src.components.embeddings import EmbeddingManager

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

    except Exception:
        sb.update_document_status(doc_id, "failed")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@router.post("/upload", status_code=202)
async def upload_document(
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
    file_ext = Path(file.filename).suffix.lower().strip(".")
    if file_ext not in user_config.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_ext}. Supported: {user_config.SUPPORTED_FILE_TYPES}",
        )

    # Read + validate file size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB",
        )

    # 1. Upload raw file to Supabase Storage immediately
    try:
        storage_path = sb.upload_file(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

    # 2. Create document record with status=processing
    doc_record = sb.create_document_record(
        filename=file.filename,
        storage_path=storage_path,
        file_type=file_ext,
        file_size_bytes=len(file_bytes),
    )
    doc_id = doc_record.get("id")

    # 3. Schedule background processing — returns immediately after this
    tmp_path = os.path.join(user_config.UPLOAD_DIR, file.filename)
    background_tasks.add_task(
        _process_document_background,
        file_bytes=file_bytes,
        filename=file.filename,
        doc_id=doc_id,
        tmp_path=tmp_path,
        user_config=user_config,
        sb=sb,
    )

    # 4. Return 202 Accepted immediately — client should poll GET /documents
    return JSONResponse(
        status_code=202,
        content={
            "id": doc_id,
            "filename": file.filename,
            "file_type": file_ext,
            "status": "processing",
            "message": "Document accepted for processing. Poll GET /documents to check status.",
            "file_size_bytes": len(file_bytes),
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


@router.delete("/{filename}")
async def delete_document(
    filename: str,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """Delete a document: removes vectors from Pinecone, file from Storage, and DB record."""
    from src.components.retrieval import RetrievalManager

    try:
        retrieval_mgr = RetrievalManager(user_config)
        retrieval_mgr.delete_document_by_filename(filename)

        storage_path = f"{sb.user_id}/{filename}"
        sb.delete_file(storage_path)
        sb.delete_document_record(filename)

        return {"message": f"Document '{filename}' deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}",
        )
