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
    list_documents as list_documents_tool,
    list_metrics as metrics_tool,
    read_document as read_tool,
    read_section as read_section_tool,
    search_knowledge as knowledge_tool,
    search_text as search_text_tool,
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
    # G8: a RetrievalManager scoped to the shared KB namespace (`kb_in`). Present ONLY
    # when USE_KNOWLEDGE is on AND the route threads it — its presence is what makes
    # `search_knowledge` offered (loop.py gates the schema on it) and executable. None
    # ⇒ the tool is never offered ⇒ byte-identical to pre-G8.
    kb_retrieval_manager: Any = None                       # live KB RetrievalManager (search_knowledge)
    db_client: Any = None                                  # live SupabaseClient (read_document)
    # F2m (shared-matter read): the user_id whose namespace + chunks back this vault. For the
    # caller's OWN vault it equals db_client.user_id (byte-identical); for a matter the caller is
    # STAFFED on it's the vault OWNER, so the read tools scope chunks to the owner (the access was
    # authorized upstream by db.accessible_vault_owner). None ⇒ fall back to db_client.user_id.
    vault_owner: Optional[str] = None
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
    # G8.7: the knowledge-source chips' SERVER-SIDE gate on the KB. When a chip narrows the
    # legal sources (e.g. "Case law" off), the route sets the ALLOWED instrument types here;
    # `search_knowledge` dispatch INTERSECTS the model's requested instrument_type with this
    # allow-list, so the agent physically cannot retrieve a source the user turned off — the
    # chip gates the backend, not just the UI. None ⇒ no restriction (all in-scope types).
    kb_instrument_types: Optional[List[str]] = None
    # DOCUMENT_HARNESS Phase 1: when True, the run offers the document-filesystem tools
    # (list_documents/search_text/read_section + whole-doc read_document) instead of
    # search_vault. Set from config.USE_DOC_HARNESS at the route. False ⇒ byte-identical.
    harness: bool = False

    def scope_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "collection_id": self.collection_id,
            "doc_ids": self.doc_ids,
            "filenames": self.filenames,
            # F1b: the run is inside a vault whenever a collection_id is set (the agent-core
            # route requires one). `search_vault` reads this as the cross-vault-leak FLOOR —
            # an active vault with no resolvable scope key is refused, never fanned out.
            "vault_active": bool(self.collection_id),
        }
        if self.filters:
            d["filters"] = self.filters
        return d


# G8: `search_knowledge` (the legal Knowledge Base) is allowed in these modes — but only
# OFFERED when the run threads a KB retrieval manager (loop.py passes
# include_knowledge=scope.kb_retrieval_manager is not None). It is listed centrally so the
# allow-set is in one place; `names(...)` strips it when knowledge is off.
_KNOWLEDGE_MODES = frozenset({"standard", "deep", "grid", "draft"})

# Which tools each mode may use (§3.3 registry mechanics). survey_collection (G5) is the
# deep-mode breadth tool only; search_knowledge (G8) is in _KNOWLEDGE_MODES, gated on.
_MODE_TOOLS = {
    "standard": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute",
                 "search_knowledge"],
    # G5: deep mode adds `survey_collection` — the broad whole-vault pass it opens with
    # before drilling. It is deliberately ABSENT from standard mode (breadth is the long
    # run's tool; a standard answer searches narrowly).
    "deep": ["survey_collection", "search_vault", "read_document", "list_metrics",
             "table_lookup", "compute", "search_knowledge"],
    "fast": [],  # fast mode does not loop / call tools
    # Review-grid cell (B2): the run is locked to ONE document via RunScope, so
    # `search_vault` cannot leak across docs — and a PROSE document (a contract) needs
    # it: clause text lives in TEXT chunks the agent finds via search_vault(kind="text"),
    # then reads. (Without it, a contract grid starved on table-only preload and abstained
    # every cell — the finance-origin bug, fixed 2026-06-12.) Keeps read/compute for the
    # numeric columns that share the grid.
    "grid": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute",
             "search_knowledge"],
    # G6.1: drafting is the SAME engine + the SAME gate — it only changes the shape of the
    # deliverable (a cited memo/summary/section). It needs to GATHER evidence, not a new
    # capability, so it gets the standard toolset; the draft prompt overlay (prompt.py)
    # supplies the deliverable shape. Recommended shape (a): prompt-overlay, no new tool.
    "draft": ["search_vault", "read_document", "list_metrics", "table_lookup", "compute",
              "search_knowledge"],
}

# DOCUMENT_HARNESS Phase 1: when USE_DOC_HARNESS is on, the offered tools in each mode
# become the "document filesystem" — ls/grep/read replace the embedding top-k (§5). The
# numeric moat (table_lookup/compute/list_metrics) is UNCHANGED. `search_vault`, vector
# survey, and the decomposer are NOT offered in harness mode (kept flag-off, retired at
# Phase 4). The map is consulted ONLY when the harness flag is set, so flag-OFF is
# byte-identical to `_MODE_TOOLS` above.
_HARNESS_TOOLS = {
    "standard": ["list_documents", "search_text", "read_document", "read_section",
                 "list_metrics", "table_lookup", "compute", "search_knowledge"],
    "deep": ["list_documents", "search_text", "read_document", "read_section",
             "list_metrics", "table_lookup", "compute", "search_knowledge"],
    "fast": [],
    "grid": ["list_documents", "search_text", "read_document", "read_section",
             "list_metrics", "table_lookup", "compute", "search_knowledge"],
    "draft": ["list_documents", "search_text", "read_document", "read_section",
              "list_metrics", "table_lookup", "compute", "search_knowledge"],
}


class ToolRegistry:
    """Schemas + dispatch. One instance per process is fine (stateless besides config)."""

    def schemas(self, mode: str, *, tools: Optional[List[str]] = None,
                include_knowledge: bool = False, harness: bool = False) -> List[Dict[str, Any]]:
        """The tool schemas exposed for a run, in the model's native tool shape.

        G7: a workflow template names its OWN `tool_subset` — when `tools` is given it is
        used DIRECTLY (validated against SCHEMAS — an unknown name is dropped, never
        offered), so a template restricts the model to exactly its subset without a new
        mode in `_MODE_TOOLS`. When `tools` is None (every existing caller), it falls back
        to the mode map and is byte-identical to before. Purely additive.

        G8: `include_knowledge` (default False — the byte-identical default) controls
        whether `search_knowledge` is OFFERED. The loop passes
        `include_knowledge=scope.kb_retrieval_manager is not None`, so the legal KB tool
        appears only when USE_KNOWLEDGE is on AND a KB manager was threaded."""
        names = self.names(mode, tools=tools, include_knowledge=include_knowledge,
                           harness=harness)
        return [SCHEMAS[n] for n in names if n in SCHEMAS]

    def names(self, mode: str, *, tools: Optional[List[str]] = None,
              include_knowledge: bool = False, harness: bool = False) -> List[str]:
        """The tool NAMES for a run. `tools` (a workflow's subset) takes precedence, kept
        only where the name is a real registered tool (validated against SCHEMAS); else the
        mode map. Additive — `tools=None` is the old behavior.

        G8: unless `include_knowledge` is True, `search_knowledge` is stripped from the
        result — so the default (every existing caller) is byte-identical to pre-G8.

        DOCUMENT_HARNESS Phase 1: `harness` (default False — the byte-identical default)
        switches the mode map from `_MODE_TOOLS` to `_HARNESS_TOOLS` (the document-
        filesystem tools). A workflow `tools` subset still takes precedence over both.
        Flag off ⇒ `names(...)` is byte-identical to pre-harness."""
        if tools is not None:
            out = [n for n in tools if n in SCHEMAS]
        else:
            mode_map = _HARNESS_TOOLS if harness else _MODE_TOOLS
            default = mode_map.get("standard", _MODE_TOOLS["standard"])
            out = list(mode_map.get(mode, default))
        if not include_knowledge:
            out = [n for n in out if n != "search_knowledge"]
        return out

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
                    # DOCUMENT_HARNESS §6.3: whole-clean-text read (harness `cat`).
                    full_text=bool(args.get("full_text", False)),
                    owner_id=scope.vault_owner,  # F2m: shared matter ⇒ read the owner's chunks

                    # The run's LIVE grid scope (mutable): grids read_document loads
                    # fresh from the DB join it, so a later `compute` can use them.
                    # Live (2026-06-11) the model read the right doc but compute
                    # couldn't see it — "document not in scope".
                    scope_grids=scope.grids,
                )

            # ── DOCUMENT_HARNESS Phase 1: the document-filesystem tools ──────────────
            # Security L2 (G-c): identity keys (user_id/collection_id/owner_id) are
            # NEVER read from the model's `args` — they come from `scope`, which is built
            # server-side from the authenticated session. The model can only propose a
            # `doc_id`, validated against scope inside each tool (resolve_doc_id → None
            # ⇒ refuse). A forged identity arg is simply ignored here.
            if name == "list_documents":
                return list_documents_tool(
                    db_client=scope.db_client,
                    collection_id=scope.collection_id,
                    filename_by_doc=scope.filename_by_doc,
                    filter=args.get("filter") or None,
                )

            if name == "search_text":
                return search_text_tool(
                    args.get("query", ""),
                    any_of=args.get("any_of"),
                    doc_ids=args.get("doc_ids"),
                    is_regex=bool(args.get("is_regex", False)),
                    k=args.get("k", 20),
                    db_client=scope.db_client,
                    filename_by_doc=scope.filename_by_doc,
                    scope_doc_ids=scope.doc_ids,
                    owner_id=scope.vault_owner,
                )

            if name == "read_section":
                return read_section_tool(
                    args.get("doc_id", ""),
                    heading=args.get("heading"),
                    page_range=args.get("page_range"),
                    db_client=scope.db_client,
                    filename_by_doc=scope.filename_by_doc,
                    owner_id=scope.vault_owner,
                )

            if name == "search_knowledge":
                # G8: the legal Knowledge Base hand. Threads the KB-scoped retrieval
                # manager (a different namespace from the user vault). Absent ⇒ the
                # adapter returns an error envelope (it never raises), but in practice
                # the tool isn't even offered unless the manager is present (loop gating).
                return knowledge_tool(
                    args.get("query", "") or scope.question or "",
                    scope.kb_retrieval_manager,
                    jurisdiction=args.get("jurisdiction", "IN") or "IN",
                    source=args.get("source"),
                    instrument_type=args.get("instrument_type"),
                    as_of=args.get("as_of"),
                    k=args.get("k", 8),
                    # G8.7: the chips' server-side allow-list. The tool drops any retrieved
                    # span whose instrument_type isn't allowed — so "Case law off" means
                    # judgments are unreachable even if the model asks for them.
                    allowed_instrument_types=scope.kb_instrument_types,
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
