"""
Document management endpoints — upload, list, delete.
Uses lazy imports for heavy RAG components.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File

from src.api.schemas import DocumentResponse, DocumentListResponse
from src.api.dependencies import get_current_user, get_user_config
from src.components.config import Config

router = APIRouter(prefix="/documents")

MAX_FILE_SIZE_MB = 10


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """
    Upload a document, process it through the RAG pipeline
    (ingest → chunk → embed), and store vectors in ChromaDB.
    """
    from src.components.data_ingestion import DocumentProcessor
    from src.components.embeddings import EmbeddingManager
    from src.components.retrieval import RetrievalManager

    # Validate file type
    file_ext = Path(file.filename).suffix.lower().strip(".")
    if file_ext not in user_config.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_ext}. Supported: {user_config.SUPPORTED_FILE_TYPES}",
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB",
        )

    # 1. Upload raw file to Supabase Storage
    try:
        storage_path = sb.upload_file(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

    # 2. Create document record (status=processing)
    doc_record = sb.create_document_record(
        filename=file.filename,
        storage_path=storage_path,
        file_type=file_ext,
        file_size_bytes=len(file_bytes),
    )
    doc_id = doc_record.get("id")

    # 3. Write to local temp file for Unstructured processing
    try:
        os.makedirs(user_config.UPLOAD_DIR, exist_ok=True)
        tmp_path = os.path.join(user_config.UPLOAD_DIR, file.filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        # 4. Ingest + chunk
        processor = DocumentProcessor(config=user_config)
        elements = processor.process_documents(file_paths=tmp_path)

        if not elements:
            if doc_id:
                sb.update_document_status(doc_id, "failed")
            raise HTTPException(
                status_code=422,
                detail="Could not extract content from this file.",
            )

        chunks = processor.build_langchain_documents(elements=elements)

        # 5. Embed into ChromaDB
        embed_mgr = EmbeddingManager(config=user_config)
        embed_mgr.create_vector_store(chunks, user_config.VECTOR_DB_PATH)

        # 6. Update DB record to ready
        if doc_id:
            sb.update_document_status(doc_id, "ready", len(chunks))

        # 7. Upload pkl caches to Supabase so they survive restarts
        for pkl_suffix in [".pkl", ".docs.pkl"]:
            pkl_path = Path(tmp_path).with_suffix(pkl_suffix)
            if pkl_path.exists():
                sb.upload_pkl(pkl_path.read_bytes(), file.filename + pkl_suffix)

        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            file_type=file_ext,
            status="ready",
            chunk_count=len(chunks),
            file_size_bytes=len(file_bytes),
            created_at=doc_record.get("created_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        if doc_id:
            sb.update_document_status(doc_id, "failed")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
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
    """Delete a document: removes vectors from ChromaDB, file from Supabase Storage, and DB record."""
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
