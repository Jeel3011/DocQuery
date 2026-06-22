"""`search_vault` tool — thin adapter over retrieval (AGENT_CORE_PLAN §3.3).

Wraps `RetrievalManager.retrieve` (text) / `retrieve_table_chunks` (tables), with the
`DocumentRouter` invoked for large (>20-doc) scopes per §3b. Returns the top chunks as
spans with source addressing + score. No new intelligence: the retriever ranks, we
shape. Requires a live `RetrievalManager` (vector index), so its primary gate is a
schema/shape unit with a stub manager — the live recall fixture set is a Phase-B item.

Gate: `eval/test_tools.py` (envelope + never-raise with a stub manager); a small
retrieval-recall fixture set is added later (§3.3 names it as the tool's eval gate).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool, span_to_dict

SCHEMA: Dict[str, Any] = {
    "name": "search_vault",
    "description": (
        "Semantically search the user's documents for passages relevant to a query. "
        "Returns the top matching chunks with their document, page, and a snippet. Use "
        "kind='table' to find structured financial tables, 'text' for prose, 'both' to "
        "mix. This finds WHERE to look; read_document/table_lookup then read the cells."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "scope": {
                "type": "object",
                "description": "Where to search.",
                "properties": {
                    "collection_id": {"type": "string"},
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                    "filenames": {"type": "array", "items": {"type": "string"}},
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional metadata narrowing (e.g. {\"doc_type\": "
                            "\"legal_contract\", \"fiscal_year\": {\"$in\": [2023]}}). "
                            "Conjunctive — narrows the vault scope, never replaces it."
                        ),
                    },
                },
            },
            "k": {"type": "integer", "description": "How many chunks to return (default 8)."},
            "kind": {
                "type": "string",
                "enum": ["text", "table", "both"],
                "description": "Search prose, table chunks, or both (default 'both').",
            },
        },
        "required": ["query"],
    },
}

_ROUTER_DOC_THRESHOLD = 20  # >this many docs in scope → route first (§3b / CDB §7.3)


@safe_tool
def search_vault(
    query: str,
    retrieval_manager: Any,
    *,
    scope: Optional[Dict[str, Any]] = None,
    k: int = 8,
    kind: str = "both",
) -> Dict[str, Any]:
    """Search the vault for `query`; return the §3.3 envelope with chunk spans.

    `retrieval_manager` is a live `RetrievalManager`. `scope` narrows the search
    (collection_id / doc_ids / filenames). `kind` selects text/table/both.
    """
    if not query:
        return error_result("search_vault requires a non-empty 'query'")
    if retrieval_manager is None:
        return error_result("search_vault requires a retrieval_manager")

    scope = scope or {}
    collection_id = scope.get("collection_id")
    # G3: doc_id is the stable, ingest-stamped scope axis (vault isolation as a DATA
    # property). The retriever filters `doc_id $in` when given; the per-USER namespace
    # already isolates users, this isolates VAULTS. `filenames` stays as the legacy
    # fallback for any un-stamped vector — the retriever's scope cascade prefers doc_ids.
    doc_ids: Optional[List[str]] = scope.get("doc_ids")
    filenames: Optional[List[str]] = scope.get("filenames")
    # G3 Step B: doc_type / fiscal_year / … narrowing — CONJUNCTIVE on top of scope,
    # never a replacement (a bug there would be a cross-vault leak).
    metadata_filter: Optional[Dict[str, Any]] = scope.get("filters") or None

    # ── F1b: the VAULT-SCOPE FLOOR (the cross-vault leak guard) ──────────────────────────
    # Within a user, multiple vaults share the per-user Pinecone namespace, so an UNSCOPED
    # query (no doc_id / collection_id / filename filter) would fan out across ALL the user's
    # vaults — a cross-vault leak. The agent-core route ALWAYS runs inside a vault (it requires
    # collection_id; routes/agent_core.py), so the run carries `vault_active=True`. F1b makes
    # the vault scope a HARD FLOOR: when vault_active is set, the query MUST resolve to at
    # least one vault-scoping key (doc_ids / collection_id / filenames) — else we ERROR rather
    # than run unscoped. `collection_id` ALONE is a valid floor (the retriever filters on the
    # stamped field — vectors carry doc_id + collection_id from ingest, so NO re-ingest), which
    # binds a vault whose doc_ids weren't preloaded. The flag is separate from the scope keys
    # precisely so that "active but unscoped" is detectable (a key, not a falsy collection_id).
    vault_active = bool(scope.get("vault_active") or collection_id)
    has_vault_scope = bool(doc_ids or collection_id or filenames)
    if vault_active and not has_vault_scope:
        # A vault is active but nothing scopes the query → refuse (never fan out).
        return error_result(
            "search_vault refused an unscoped query inside an active vault "
            "(cross-vault leak guard). Scope to the vault's documents."
        )

    docs: List[Any] = []

    if kind in ("text", "both"):
        # `retrieve` handles the doc_ids axis end-to-end: len>1 balances per-doc, len==1
        # becomes a scalar doc_id filter. Filenames are the legacy fallback only when no
        # doc_ids are present (un-stamped vectors).
        text_docs = retrieval_manager.retrieve(
            query,
            doc_ids=doc_ids or None,
            filename_filters=(filenames if filenames and len(filenames) > 1 else None) if not doc_ids else None,
            filename_filter=(filenames[0] if filenames and len(filenames) == 1 else None) if not doc_ids else None,
            metadata_filter=metadata_filter,
            # F1b: bind the text query to the active vault when doc_ids/filenames aren't the
            # scope axis — the floor's collection_id-only case (the retriever filters on the
            # stamped `collection_id` field via _build_filter's scope cascade).
            collection_id=(collection_id if not doc_ids and not filenames else None),
            top_k=k,
            apply_threshold=False,
            use_reranker=True,
        )
        docs.extend(text_docs or [])

    if kind in ("table", "both"):
        # G3: scope table chunks by doc_id (stable, ingest-stamped) when present, else
        # fall back to FILENAME. F1b: ingest NOW stamps collection_id on every chunk
        # (text AND table — worker/tasks.py:157-161, after build_langchain_documents), so
        # the collection_id-only floor binds table chunks too. (The BUG-F 2026-06-11 note
        # predates that stamping; collection_id is a valid table scope now.)
        table_docs = retrieval_manager.retrieve_table_chunks(
            query,
            doc_ids=doc_ids if doc_ids else None,
            collection_id=(collection_id if not doc_ids and not filenames else None),
            # Normalize an empty list to None here (consistent with the text path's
            # `doc_ids or None`): the floor above already refused an unscoped active vault,
            # so reaching here with no scope is a non-vault run, not an empty vault.
            filename_filters=(filenames or None),
            metadata_filter=metadata_filter,
            k=k,
        )
        docs.extend(table_docs or [])

    spans = [span_to_dict(d) for d in docs][: max(k, 1) if kind != "both" else max(k * 2, 1)]
    return ok_result(
        summary=f"search_vault {query!r} ({kind}): {len(spans)} chunk(s)",
        data={"chunks": spans},
        provenance=spans,
    )
