"""
Collections endpoints — CRUD for document collections (Phase 1: Multi-File Q&A).
"""

from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import (
    CreateCollectionRequest,
    UpdateCollectionRequest,
    CollectionResponse,
    CollectionListResponse,
    AddDocumentToCollectionRequest,
    DocumentResponse,
    DocumentListResponse,
)
from src.api.dependencies import get_current_user
from src.api.routes.audit import log_audit
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/collections")


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CreateCollectionRequest,
    sb=Depends(get_current_user),
):
    """Create a new collection."""
    try:
        coll = sb.create_collection(body.name, body.description)
        if not coll:
            raise HTTPException(status_code=500, detail="Failed to create collection")
        log_audit(sb, "collection.create", "collection", coll["id"], {"name": coll["name"]})
        return CollectionResponse(
            id=coll["id"],
            name=coll["name"],
            description=coll.get("description"),
            document_count=0,
            created_at=coll.get("created_at"),
            updated_at=coll.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create collection")
        raise HTTPException(status_code=500, detail="Failed to create collection.")


@router.get("", response_model=CollectionListResponse)
async def list_collections(sb=Depends(get_current_user)):
    """List all collections for the current user."""
    colls = sb.get_collections()
    items = []
    for c in colls:
        doc_count = len(sb.get_collection_document_ids(c["id"]))
        items.append(CollectionResponse(
            id=c["id"],
            name=c["name"],
            description=c.get("description"),
            document_count=doc_count,
            created_at=c.get("created_at"),
            updated_at=c.get("updated_at"),
        ))
    return CollectionListResponse(collections=items, total=len(items))


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    sb=Depends(get_current_user),
):
    """Get a single collection with document count."""
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    doc_count = len(sb.get_collection_document_ids(collection_id))
    return CollectionResponse(
        id=coll["id"],
        name=coll["name"],
        description=coll.get("description"),
        document_count=doc_count,
        created_at=coll.get("created_at"),
        updated_at=coll.get("updated_at"),
    )


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    body: UpdateCollectionRequest,
    sb=Depends(get_current_user),
):
    """Rename or update description of a collection."""
    existing = sb.get_collection(collection_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Collection not found.")
    updated = sb.update_collection(collection_id, body.name, body.description)
    if not updated:
        # No fields changed — return existing
        updated = existing
    doc_count = len(sb.get_collection_document_ids(collection_id))
    return CollectionResponse(
        id=updated["id"],
        name=updated["name"],
        description=updated.get("description"),
        document_count=doc_count,
        created_at=updated.get("created_at"),
        updated_at=updated.get("updated_at"),
    )


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: str,
    sb=Depends(get_current_user),
):
    """Delete a collection (documents remain, just unlinked)."""
    existing = sb.get_collection(collection_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Collection not found.")
    try:
        sb.delete_collection(collection_id)
        log_audit(sb, "collection.delete", "collection", collection_id, {"name": existing.get("name")})
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.exception("Failed to delete collection %s", collection_id)
        raise HTTPException(status_code=500, detail="Failed to delete collection.")


@router.post("/{collection_id}/documents", status_code=201)
async def add_document_to_collection(
    collection_id: str,
    body: AddDocumentToCollectionRequest,
    sb=Depends(get_current_user),
):
    """Add a document to a collection."""
    # Verify collection exists and belongs to user
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    # Verify document exists and belongs to user
    doc = sb.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        result = sb.add_document_to_collection(collection_id, body.document_id)
        log_audit(sb, "collection.add_document", "collection", collection_id, {"document_id": body.document_id})
        return {"message": "Document added to collection", "collection_id": collection_id, "document_id": body.document_id}
    except Exception as e:
        logger.exception("Failed to add document to collection")
        raise HTTPException(status_code=500, detail="Failed to add document to collection.")


@router.delete("/{collection_id}/documents/{document_id}")
async def remove_document_from_collection(
    collection_id: str,
    document_id: str,
    sb=Depends(get_current_user),
):
    """Remove a document from a collection."""
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    try:
        sb.remove_document_from_collection(collection_id, document_id)
        return {"message": "Document removed from collection"}
    except Exception as e:
        logger.exception("Failed to remove document from collection")
        raise HTTPException(status_code=500, detail="Failed to remove document from collection.")


@router.get("/{collection_id}/documents", response_model=DocumentListResponse)
async def list_collection_documents(
    collection_id: str,
    sb=Depends(get_current_user),
):
    """List all documents in a collection."""
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    docs = sb.get_collection_documents(collection_id)
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
            )
            for d in docs
        ],
        total=len(docs),
    )
