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
    survey_collection as survey_tool,
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
    # G3 Step D/E: the active vault metadata filter (doc_type / fiscal_year). Threaded to
    # search_vault as `scope.filters` → the retriever's CONJUNCTIVE metadata_filter. It
    # NARROWS the vault scope, never replaces it (a bug there = cross-vault leak).
    filters: Optional[Dict[str, Any]] = None
    # G5: the user Config the demoted Brain MAP step runs on (survey_collection, deep mode
    # only). None ⇒ survey_collection returns an error envelope (never raises); the other
    # tools never read it.
    config: Any = None

    def scope_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "collection_id": self.collection_id,
            "doc_ids": self.doc_ids,
            "filenames": self.filenames,
        }
        if self.filters:
            d["filters"] = self.filters
        return d


# Which tools each mode may use (§3.3 registry mechanics). draft/knowledge tools arrive
# in later phases; survey_collection (G5) is the deep-mode breadth tool only.
_MODE_TOOLS = {
    "standard": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"],
    # G5: deep mode adds `survey_collection` — the broad whole-vault pass it opens with
    # before drilling. It is deliberately ABSENT from standard mode (breadth is the long
    # run's tool; a standard answer searches narrowly).
    "deep": ["survey_collection", "search_vault", "read_document", "list_metrics", "table_lookup", "compute"],
    "fast": [],  # fast mode does not loop / call tools
    # Review-grid cell (B2): the run is locked to ONE document via RunScope, so
    # `search_vault` cannot leak across docs — and a PROSE document (a contract) needs
    # it: clause text lives in TEXT chunks the agent finds via search_vault(kind="text"),
    # then reads. (Without it, a contract grid starved on table-only preload and abstained
    # every cell — the finance-origin bug, fixed 2026-06-12.) Keeps read/compute for the
    # numeric columns that share the grid.
    "grid": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"],
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

            if name == "survey_collection":
                # G5 deep-mode breadth tool. Runs over the WHOLE vault scope (the run's
                # routed filenames); the model only chooses the topic + how wide to go.
                # Scope is authoritative here — survey isn't a place to narrow to one doc.
                return survey_tool(
                    args.get("query", "") or scope.question or "",
                    scope.retrieval_manager,
                    scope.config,
                    filenames=scope.filenames,
                    filename_by_doc=scope.filename_by_doc,
                    k_docs=args.get("k_docs"),
                    per_doc_k=args.get("per_doc_k", 8),
                )

            if name == "search_vault":
                # Merge the model's scope arg over the run scope (model may narrow it).
                model_scope = args.get("scope") or {}
                merged = {**scope.scope_dict(), **model_scope}
                # G3: doc_id is the stable, ingest-stamped scope axis. The run scope's
                # doc_ids are real UUIDs (resolved from the vault in Postgres), so they
                # flow straight to the retriever's `doc_id $in` filter — vault isolation
                # as a DATA property, not a correctly-assembled filename list.
                #
                # The MODEL, however, often "narrows" by passing the FILENAMES it saw in
                # earlier results back as doc_ids. Those aren't UUIDs — route any non-UUID
                # entry into `filenames` (the legacy fallback) so it still scopes, and
                # keep only true doc_ids (those present in the run's doc_id set) on the
                # doc_ids axis. This keeps the model's narrowing AND the doc_id moat.
                if model_scope.get("doc_ids"):
                    run_ids = set(scope.doc_ids or [])
                    fmap = scope.filename_by_doc or {}
                    real_ids, as_names = [], []
                    for d in model_scope["doc_ids"]:
                        if d in run_ids:
                            real_ids.append(d)
                        else:
                            # filename echoed back as a doc_id, or a UUID we can map
                            as_names.append(fmap.get(d, d))
                    merged["doc_ids"] = real_ids or None
                    if as_names and not merged.get("filenames"):
                        merged["filenames"] = as_names
                # G3 Step E: the run's active vault filters are AUTHORITATIVE — fold them
                # in conjunctively so the model can ADD a narrowing but never DROP the
                # UI's filter (e.g. "FY2023 only" must hold even if the model omits it).
                if scope.filters:
                    merged["filters"] = {**(model_scope.get("filters") or {}), **scope.filters}
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
