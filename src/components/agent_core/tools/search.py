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
    filenames: Optional[List[str]] = scope.get("filenames") or scope.get("doc_ids")

    docs: List[Any] = []

    if kind in ("text", "both"):
        text_docs = retrieval_manager.retrieve(
            query,
            filename_filters=filenames if filenames and len(filenames) > 1 else None,
            filename_filter=filenames[0] if filenames and len(filenames) == 1 else None,
            top_k=k,
            apply_threshold=False,
            use_reranker=True,
        )
        docs.extend(text_docs or [])

    if kind in ("table", "both"):
        table_docs = retrieval_manager.retrieve_table_chunks(
            query,
            collection_id=collection_id,
            filename_filters=filenames,
            k=k,
        )
        docs.extend(table_docs or [])

    spans = [span_to_dict(d) for d in docs][: max(k, 1) if kind != "both" else max(k * 2, 1)]
    return ok_result(
        summary=f"search_vault {query!r} ({kind}): {len(spans)} chunk(s)",
        data={"chunks": spans},
        provenance=spans,
    )
