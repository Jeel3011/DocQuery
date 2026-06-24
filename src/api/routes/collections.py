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
    PracticeTemplateResponse,
    ConflictScanRequest,
    ConflictScanResponse,
    ConflictFinding,
)
from src.api.dependencies import get_current_user, require_cap, assert_vault_not_screened
from src.api.routes.audit import log_audit
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/collections")


def _collection_response(coll: dict, doc_count: int, conflicts: list = None) -> CollectionResponse:
    """Serialize a collection row → CollectionResponse, including F1a matter fields.

    Legacy rows (pre-migration-010 or untyped) read as matter_kind=None, status='active',
    parties=[] — so the response shape is stable whether or not 010 has been applied.

    F1c: `conflicts` (the metadata-only scan findings) is threaded ONLY on create-with-parties;
    None on every other path ⇒ the legacy response shape is unchanged.
    """
    has_adverse = None
    findings = None
    if conflicts is not None:
        findings = [ConflictFinding(**c) for c in conflicts]
        has_adverse = any(c.get("severity") == "adverse" for c in conflicts)
    return CollectionResponse(
        id=coll["id"],
        name=coll["name"],
        description=coll.get("description"),
        document_count=doc_count,
        created_at=coll.get("created_at"),
        updated_at=coll.get("updated_at"),
        matter_kind=coll.get("matter_kind"),
        status=coll.get("status") or "active",
        parties=coll.get("parties") or [],
        firm_id=coll.get("firm_id"),
        conflicts=findings,
        has_adverse=has_adverse,
    )


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CreateCollectionRequest,
    sb=Depends(get_current_user),
    _cap=Depends(require_cap("create_vault")),
):
    """Create a new collection (matter/vault). F2b: cap-gated on `create_vault`."""
    try:
        # F1a: a vault is owned by the user's firm (server-side lookup, not a JWT claim → the
        # auth flow is unchanged). None when the user has no firm yet — the vault still works.
        firm = sb.get_user_firm()
        firm_id = firm.get("id") if firm else None
        parties = [p.model_dump() for p in body.parties] if body.parties else None
        coll = sb.create_collection(
            body.name,
            body.description,
            matter_kind=body.matter_kind,
            parties=parties,
            firm_id=firm_id,
        )
        if not coll:
            raise HTTPException(status_code=500, detail="Failed to create collection")
        log_audit(sb, "collection.create", "collection", coll["id"], {
            "name": coll["name"], "matter_kind": coll.get("matter_kind"),
        })
        # F1c: metadata-only ethical-wall conflict scan on intake. Runs ONLY when parties were
        # named; excludes the just-created matter so it can't conflict with itself. Non-blocking
        # — a hit becomes a banner, never a refusal. Touches ZERO document content (party
        # metadata only). Skipped (conflicts=None) for a no-party create ⇒ legacy shape.
        conflicts = None
        if parties:
            conflicts = sb.scan_conflicts(parties, firm_id=firm_id, exclude_collection_id=coll["id"])
            if conflicts:
                log_audit(sb, "collection.conflict_scan", "collection", coll["id"], {
                    "hits": len(conflicts),
                    "adverse": sum(1 for c in conflicts if c.get("severity") == "adverse"),
                })
        return _collection_response(coll, doc_count=0, conflicts=conflicts)
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
        items.append(_collection_response(c, doc_count))
    return CollectionListResponse(collections=items, total=len(items))


# NOTE: these two F1c routes use STATIC path prefixes ("practice-template", "scan-conflicts")
# and MUST be declared before GET/POST "/{collection_id}" below — otherwise FastAPI would bind
# "practice-template" as a collection_id and 404. Order is load-bearing here.

@router.get("/practice-template/{matter_kind}", response_model=PracticeTemplateResponse)
async def get_practice_template(
    matter_kind: str,
    sb=Depends(get_current_user),
):
    """F1c: the practice template a matter_kind suggests (grid columns + KB scope + flagship).

    Static, $0, no LLM. An unknown / 'generic' kind returns the neutral generic template (the
    safe fall-back), so the create dialog always has a sensible default to render. Auth-gated
    (consistent with the rest of the namespace) but reads no user data.
    """
    from src.components.practice_templates import template_for
    tpl = template_for(None if matter_kind in ("generic", "none", "") else matter_kind)
    return PracticeTemplateResponse(**tpl.to_dict())


@router.post("/scan-conflicts", response_model=ConflictScanResponse)
async def scan_conflicts(
    body: ConflictScanRequest,
    sb=Depends(get_current_user),
):
    """F1c: pre-create ethical-wall screen — metadata only, never document content.

    The create dialog calls this as the user adds parties, so the conflict banner can appear
    BEFORE the matter is committed. Compares the supplied party names against the firm's other
    matters' parties (the `collections.parties` metadata) for adverse collisions. Non-blocking:
    returns findings; the UI decides what to show. Empty parties ⇒ empty result (no scan).
    """
    parties = [p.model_dump() for p in body.parties] if body.parties else []
    firm = sb.get_user_firm()
    firm_id = firm.get("id") if firm else None
    hits = sb.scan_conflicts(parties, firm_id=firm_id,
                             exclude_collection_id=body.exclude_collection_id)
    findings = [ConflictFinding(**c) for c in hits]
    return ConflictScanResponse(
        conflicts=findings,
        has_adverse=any(c.severity == "adverse" for c in findings),
    )


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
    return _collection_response(coll, doc_count)


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    body: UpdateCollectionRequest,
    sb=Depends(get_current_user),
):
    """Rename a collection or update its matter typing/lifecycle (F1a)."""
    existing = sb.get_collection(collection_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Collection not found.")
    parties = [p.model_dump() for p in body.parties] if body.parties is not None else None
    updated = sb.update_collection(
        collection_id, body.name, body.description,
        matter_kind=body.matter_kind, status=body.status, parties=parties,
    )
    if not updated:
        # No fields changed — return existing
        updated = existing
    doc_count = len(sb.get_collection_document_ids(collection_id))
    return _collection_response(updated, doc_count)


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: str,
    sb=Depends(get_current_user),
    _cap=Depends(require_cap("delete")),
):
    """Delete a collection (documents remain, just unlinked). F2b: cap-gated on `delete`."""
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
    _cap=Depends(require_cap("ingest")),
):
    """Add a document to a collection. F2b: cap-gated on `ingest`."""
    # Verify collection exists and belongs to user
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    # F2c P7 (the ethical wall, worker re-ingest path): a screened actor cannot ingest INTO a
    # walled vault. This is the enqueue-side enforcement (§service-wiring) — the worker is
    # authz-blind, so the screen must block at the route that targets this vault, before the
    # doc joins the vault (and would later flow into its retrieval scope).
    assert_vault_not_screened(sb, collection_id)
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
    _cap=Depends(require_cap("ingest")),
):
    """Remove a document from a collection. F2b: cap-gated on `ingest` (matter content edit)."""
    coll = sb.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found.")
    # F2c P7: a screened actor cannot mutate a walled vault's contents (no remove either).
    assert_vault_not_screened(sb, collection_id)
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
                doc_type=d.get("doc_type"),
                fidelity=d.get("fidelity"),
                privileged=bool(d.get("privileged", False)),
            )
            for d in docs
        ],
        total=len(docs),
    )
