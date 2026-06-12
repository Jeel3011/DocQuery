"""Tool registry (AGENT_CORE_PLAN §3.2/§3.3).

Holds the tool JSON schemas and dispatches a model `ToolCall` to the right A1 adapter,
threading the run-scoped dependencies (grids, retrieval manager, db client) from the
RunScope so the model's args (pure JSON) never carry live objects. `schemas(scope)`
filters the exposed tools by mode/scope — this is how the UI's "knowledge-source
chips" work (§3.3): a tool the mode doesn't allow is simply not offered to the model.

`execute` NEVER raises — the adapters are `@safe_tool`, and the registry guards
unknown tools / bad args with an error envelope too (the §3.2 contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .model import ToolCall
from .tools import (
    SCHEMAS,
    compute as compute_tool,
    list_metrics as metrics_tool,
    read_document as read_tool,
    search_vault as search_tool,
    table_lookup as table_tool,
)
from .tools._envelope import error_result


@dataclass
class RunScope:
    """The live dependencies + data scope for one run. Tools read what they need."""
    collection_id: Optional[str] = None
    doc_ids: List[str] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)
    grids: List[Any] = field(default_factory=list)        # preloaded analyst.Grid for this scope
    retrieval_manager: Any = None                          # live RetrievalManager (search_vault)
    db_client: Any = None                                  # live SupabaseClient (read_document)
    filename_by_doc: Dict[str, str] = field(default_factory=dict)
    question: Optional[str] = None                         # the run's question (grid relevance ranking)

    def scope_dict(self) -> Dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "doc_ids": self.doc_ids,
            "filenames": self.filenames,
        }


# Which tools each mode may use (§3.3 registry mechanics). survey_collection/draft/
# knowledge tools arrive in later phases; A1 ships the four core tools.
_MODE_TOOLS = {
    "standard": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"],
    "deep": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"],
    "fast": [],  # fast mode does not loop / call tools
    # Review-grid cell (B2): the run is already scoped to ONE known document, so there
    # is no `search_vault` (no world-search to do, and it prevents cross-doc leakage /
    # wasted cost). The cell agent reads that doc and computes/looks-up within it.
    "grid": ["read_document", "list_metrics", "table_lookup", "compute"],
}


class ToolRegistry:
    """Schemas + dispatch. One instance per process is fine (stateless besides config)."""

    def schemas(self, mode: str) -> List[Dict[str, Any]]:
        """The tool schemas exposed for `mode`, in the model's native tool shape."""
        names = _MODE_TOOLS.get(mode, _MODE_TOOLS["standard"])
        return [SCHEMAS[n] for n in names if n in SCHEMAS]

    def names(self, mode: str) -> List[str]:
        return list(_MODE_TOOLS.get(mode, _MODE_TOOLS["standard"]))

    def execute(self, call: ToolCall, scope: RunScope) -> Dict[str, Any]:
        """Dispatch one ToolCall to its adapter with scope-injected deps. Never raises."""
        name = call.name
        args = call.args or {}
        try:
            if name == "compute":
                spec = {k: v for k, v in args.items()}
                return compute_tool(spec, scope.grids)

            if name == "list_metrics":
                return metrics_tool(
                    args.get("doc_id", ""),
                    scope.grids,
                    contains=args.get("contains"),
                    period=args.get("period"),
                )

            if name == "table_lookup":
                return table_tool(
                    args.get("metric", ""),
                    args.get("period", ""),
                    scope.grids,
                    entity_or_section=args.get("entity_or_section", "") or "",
                    aggregation=args.get("aggregation", "any") or "any",
                )

            if name == "read_document":
                return read_tool(
                    args.get("doc_id", ""),
                    db_client=scope.db_client,
                    grids=scope.grids or None,
                    question=scope.question,
                    filename_by_doc=scope.filename_by_doc,
                    page_range=args.get("page_range"),
                    table_grids=args.get("table_grids", True),
                    # The run's LIVE grid scope (mutable): grids read_document loads
                    # fresh from the DB join it, so a later `compute` can use them.
                    # Live (2026-06-11) the model read the right doc but compute
                    # couldn't see it — "document not in scope".
                    scope_grids=scope.grids,
                )

            if name == "search_vault":
                # Merge the model's scope arg over the run scope (model may narrow it).
                model_scope = args.get("scope") or {}
                merged = {**scope.scope_dict(), **model_scope}
                # The retriever filters by FILENAME. The model may scope by doc_ids —
                # usually the filenames it saw in results, but possibly real UUIDs;
                # translate via filename_by_doc so a UUID never reaches the retriever
                # as a filename filter.
                if merged.get("doc_ids") and not merged.get("filenames"):
                    fmap = scope.filename_by_doc or {}
                    merged["doc_ids"] = [fmap.get(d, d) for d in merged["doc_ids"]]
                return search_tool(
                    args.get("query", ""),
                    scope.retrieval_manager,
                    scope=merged,
                    k=args.get("k", 8),
                    kind=args.get("kind", "both"),
                )

            return error_result(f"unknown tool {name!r}", summary=f"no such tool: {name}")
        except Exception as exc:  # noqa: BLE001 — belt-and-braces; adapters already guard
            return error_result(f"{type(exc).__name__}: {exc}", summary=f"registry error in {name}")


REGISTRY = ToolRegistry()
