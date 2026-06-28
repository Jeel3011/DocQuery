"""`list_documents` tool — the harness `ls` (DOCUMENT_HARNESS §6.1).

Lists the documents in the current matter with their type + a short outline so the
model can ORIENT before reading or searching (the first move Claude Code makes).
Read-only and vault-scoped: it returns ONLY the docs in this run's scope.

Two dependency shapes, both honored so it's testable offline and live:
  - LIVE: pass `db_client`; docs come from `db.get_collection_documents(collection_id)`
    (already owner-resolved for shared matters, F2m).
  - OFFLINE / pre-scoped: pass `filename_by_doc` (the run's {uuid: filename}); the
    listing is built from that map alone (the DB is never touched).

No new intelligence: it lists what the run scope already knows. Gate:
`eval/test_doc_harness.py` (envelope + scope only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "list_documents",
    "description": (
        "List the documents in the current matter with their type and a short outline "
        "(filename, doc_type, page count). Use this FIRST to orient before reading or "
        "searching — like `ls` in a folder. Returns only this matter's documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "description": (
                    "Optional narrowing, e.g. {\"doc_type\": \"legal_contract\"} or "
                    "{\"fiscal_year\": 2023}. Conjunctive — narrows the matter, never "
                    "widens it."
                ),
            },
        },
    },
}


def _matches(rec: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """Conjunctive equality match of a doc record against the filter dict."""
    for k, v in flt.items():
        if str(rec.get(k)) != str(v):
            return False
    return True


@safe_tool
def list_documents(
    *,
    db_client: Any = None,
    collection_id: Optional[str] = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002 — matches the schema arg name
) -> Dict[str, Any]:
    """List this matter's documents (+outline) as the §3.3 envelope.

    LIVE: full records via db.get_collection_documents (owner-resolved). OFFLINE:
    the {uuid: filename} map. Vault scope is the listing's bound — it can only return
    documents already in this run's scope; there is no cross-vault fan-out.
    """
    flt = filter or {}
    docs: List[Dict[str, Any]] = []

    if db_client is not None and collection_id:
        # LIVE — get_collection_documents resolves the owner (F2m) and returns ONLY
        # this collection's docs, so vault isolation is the query's bound (§8).
        records = db_client.get_collection_documents(collection_id) or []
        for r in records:
            if flt and not _matches(r, flt):
                continue
            docs.append({
                "doc_id": r.get("id"),
                "filename": r.get("filename"),
                "doc_type": r.get("doc_type"),
                "fiscal_year": r.get("fiscal_year"),
                "n_pages": r.get("page_count") or r.get("n_pages"),
            })
    elif filename_by_doc:
        # OFFLINE / pre-scoped — list from the run's map (no DB, no outline metadata).
        for did, fn in filename_by_doc.items():
            docs.append({"doc_id": did, "filename": fn, "doc_type": None,
                         "fiscal_year": None, "n_pages": None})
    else:
        return error_result(
            "list_documents has no scope to list (no db_client+collection_id and no "
            "filename_by_doc). This is a routing bug, not a model error — report it."
        )

    summary = f"list_documents: {len(docs)} document(s) in matter"
    if flt:
        summary += f" matching {flt}"
    # No provenance: a listing is not evidence (DOCUMENT_HARNESS §6.1).
    return ok_result(summary=summary, data={"documents": docs}, provenance=[])
