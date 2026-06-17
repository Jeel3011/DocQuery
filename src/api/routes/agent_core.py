"""Agent-core route (AGENT_CORE_PLAN §3.1, §3.6 · roadmap A4).

`POST /query/agentcore/stream` — the frontier-model agent loop behind the
`USE_AGENT_CORE` flag. The model IS the orchestrator: it calls our verified tools
(search_vault / read_document / table_lookup / compute) and its draft is bound to the
evidence ledger by the non-bypassable output gates (A3). NO question-type routing.

PRIME DIRECTIVE (the plan's): flag OFF ⇒ this endpoint 404s and nothing else in the
system changes. It is purely additive until `USE_AGENT_CORE=true`.

Shape mirrors `/query/brain/stream`: scope assembly (embed → resolve filenames →
preload grids) happens async/off-loop FIRST; then the loop — a *sync* generator wrapping
a *blocking* `model.invoke` — is iterated by Starlette in a worker thread via
StreamingResponse, exactly like the other streaming endpoints here. The loop's §3.6
event dicts are translated to `data: {...}\n\n` SSE lines; a `token` event is emitted as
`{type:token, content:...}` so the existing `_saving_stream_wrapper` persists the answer.

Degradation (§3.2): a model-error degrade signal from the loop, OR any exception during
scope assembly / iteration, falls back to a clean abstain line — never a 500 mid-stream.
The first PAID model call in the whole system lives here (only when the flag is on).
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_current_user,
    get_user_config,
    get_retrieval_mgr,
    get_kb_retrieval_mgr,
    limiter,
)
from src.api.schemas import QueryRequest
from src.components.config import Config

# Reuse the brain route's proven collection→filenames resolver and the query embed +
# the save-to-conversation wrapper — same scope semantics, no duplication.
from src.api.routes.chat import (
    _resolve_collection_filters,
    _saving_stream_wrapper,
    _embed_query,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    """One SSE line."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


# Map loop §3.6 event dicts → SSE payloads the frontend (and saving wrapper) consume.
# `token` is renamed `content` so `_saving_stream_wrapper` captures the final answer; the
# agent_* / tool_* / gate / artifact events pass through for the Trust-UI timeline.
def _translate(ev: dict) -> dict:
    t = ev.get("type")
    if t == "token":
        return {"type": "token", "content": ev.get("text", "")}
    if t == "sources":
        # The ledger emits each source with a `doc` field (the filename), but the
        # frontend SourceInfo expects `filename` + `source_id` — without the remap every
        # source rendered as "Unknown" and citations couldn't resolve a name (the live
        # bug 2026-06-11). Map doc→filename and assign 1-based ids in order so the
        # [doc p.N] citation chips match a numbered source.
        mapped = []
        for i, s in enumerate(ev.get("sources", []) or [], start=1):
            if not isinstance(s, dict):
                continue
            mapped.append({
                **s,
                "source_id": s.get("source_id", i),
                "filename": s.get("filename") or s.get("doc"),
                "content": s.get("content") or s.get("snippet"),
            })
        return {"type": "sources", "sources": mapped}
    return ev


@router.post("/query/agentcore/stream")
@limiter.limit("5/minute")
async def agentcore_query_stream(
    request: Request,
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    kb_retrieval_mgr=Depends(get_kb_retrieval_mgr),  # G8: None unless USE_KNOWLEDGE on
):
    """The agent core (§3.2). Flag-gated, SSE per §3.6, scope-scoped to a collection.

    Events: agent_step → agent_thought → tool_call/tool_result* → gate* → sources →
    token → meta → [DONE]. A model degrade or any failure ends with a clean abstain.
    """
    # ── Prime directive: flag off ⇒ 404 (endpoint does not exist) ──────────────────
    if not getattr(user_config, "USE_AGENT_CORE", False):
        raise HTTPException(status_code=404, detail="Not Found")

    collection_id = getattr(body, "collection_id", None)
    if not collection_id:
        raise HTTPException(
            status_code=400,
            detail="collection_id is required for the agent-core endpoint.",
        )

    # Mode dispatch (§3.1): the user may force fast|standard|deep|draft; default standard.
    # (Fast does not loop — the agent core is the smart path; fast stays on /query/stream.)
    mode = (getattr(body, "mode", None) or "standard").lower()
    if mode not in ("standard", "deep", "draft"):
        mode = "standard"

    # G6.1 drafting: the deliverable request (doc_type + instructions) is composed into the
    # question so drafting rides the SAME loop — no second orchestrator. The draft prompt
    # overlay (prompt.py) supplies the deliverable shape; the gate binds it like any answer.
    if mode == "draft":
        _doc_type = (getattr(body, "doc_type", None) or "").strip()
        _instr = (getattr(body, "instructions", None) or "").strip()
        if _doc_type or _instr:
            composed = "Produce a client-ready " + (_doc_type or "deliverable")
            if _instr:
                composed += f". Instructions: {_instr}"
            # Overwrite the question the loop drafts from (keep the request immutable
            # elsewhere by setting only what the loop reads — body.question).
            body.question = composed

    # ── Scope assembly (async, off the loop) — mirror the brain route ──────────────
    query_embedding: list = []
    try:
        query_embedding = await _embed_query(
            body.question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
    except Exception:
        pass

    # G3 Step D/E: the active vault filter set (doc_type / fiscal_year) is EXPLICIT in the
    # request (never a stale global store). It pre-narrows the router's candidate set for
    # a big vault AND becomes the retriever's conjunctive metadata_filter via scope.filters.
    metadata_filter = getattr(body, "filters", None) or None

    filename_filters = _resolve_collection_filters(
        sb, collection_id,
        query=body.question, query_embedding=query_embedding, user_config=user_config,
        metadata_filter=metadata_filter,
    )
    if not filename_filters:
        def _empty():
            yield _sse({"type": "token", "content": "No documents found in this collection."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # doc_id ↔ filename map for the scope (the retriever filters by filename; tools speak
    # doc ids). Build it from the collection's docs (same query the resolver used).
    doc_ids = sb.get_collection_document_ids(collection_id) or []
    docs = sb.client.table("documents").select("id,filename").in_(
        "id", doc_ids
    ).eq("user_id", sb.user_id).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    # Keep only docs that survived routing (filename_filters is the routed subset).
    routed = set(filename_filters)
    scoped_doc_ids = [did for did, fn in filename_by_doc.items() if fn in routed]
    if not scoped_doc_ids:  # routing returned filenames we can't map back — use all
        scoped_doc_ids = doc_ids

    # Preload the table grids for this scope ONCE (the spine already shares grids; this is
    # the §I2 grid-preload lever). Off the event loop — load_grids does blocking DB reads.
    def _load_grids():
        try:
            from src.components.brain.table_intent import load_grids_for_docs
            # per_doc_top (not the global top-8): every scoped doc keeps its most
            # relevant grids, so no document silently vanishes from the compute scope
            # (the 2026-06-11 "document not in scope" starvation).
            return load_grids_for_docs(
                sb, scoped_doc_ids,
                question=body.question, filename_by_doc=filename_by_doc,
                # 8 was far too few — a 10-K has ~100 table grids, and the primary
                # financial statements (where R&D/revenue/net-income live) lost the
                # lexical ranking to MD&A prose pages, so the kernel never saw them and
                # every numeric question quietly starved (live 2026-06-11, R&D-ratio).
                # 20/doc + the statement-priority boost in load_grids_for_docs keeps the
                # core statements AND the top lexical matches. Grids only enter the
                # model's context via read_document; compute/list_metrics query them
                # without dumping JSON — so a higher cap costs memory, not tokens.
                per_doc_top=20,
            )
        except Exception as exc:  # noqa: BLE001 — grids are best-effort; tools degrade gracefully
            logger.warning("[agentcore] grid preload failed: %s", exc)
            return []

    grids = await asyncio.to_thread(_load_grids)

    # ── Conversation memory (G5 Ask polish): thread PRIOR turns into the run so a
    # follow-up ("what about 2022?", "and the termination clause?") resolves against the
    # conversation, not from scratch. The loop already accepts `history` — we only load it.
    # Prior turns are READ-ONLY context; the current question is appended by the loop and
    # the current turn is persisted by `_saving_stream_wrapper` AFTER the stream, so it is
    # NOT yet in `get_messages` (no duplication). Bounded to the recent turns to keep the
    # context (and cost) in check; off the event loop (blocking DB read).
    history: list[dict] = []
    _conv_id_for_history = getattr(body, "conversation_id", None)
    if _conv_id_for_history:
        def _load_history():
            try:
                rows = sb.get_messages(_conv_id_for_history) or []
                msgs = [
                    {"role": r.get("role"), "content": r.get("content") or ""}
                    for r in rows
                    if r.get("role") in ("user", "assistant") and (r.get("content") or "").strip()
                ]
                return msgs[-12:]  # ~6 recent turns — enough for follow-ups, bounded cost
            except Exception as exc:  # noqa: BLE001 — memory is best-effort; a fresh run still works
                logger.warning("[agentcore] history load failed (%s) — running without memory", exc)
                return []
        history = await asyncio.to_thread(_load_history)

    # ── G8.7 knowledge-source chips → SERVER-SIDE gate ─────────────────────────────
    # `body.sources` (a subset of {"vault","statutes","caselaw"}) decides which authorities
    # the agent may use THIS run. A source off is gated where it can't be bypassed:
    #   • vault off    → drop the vault retrieval manager + strip vault/grid tools from the
    #                    offered set, so the agent cannot search or cite the user's docs.
    #   • statutes/caselaw → an instrument-type ALLOW-LIST on the KB (search_knowledge drops
    #                    spans outside it); both off → the KB manager isn't passed at all.
    # None/absent ⇒ everything enabled (byte-identical). Unknown chips are ignored.
    _STATUTE_TYPES = ["act", "regulation", "circular", "article", "schedule", "rule"]
    _CASELAW_TYPES = ["judgment"]
    chips = body.sources
    if chips is None:
        vault_on = statutes_on = caselaw_on = True   # default: all sources enabled
    else:
        sel = {str(s).lower() for s in chips}
        vault_on = "vault" in sel
        statutes_on = "statutes" in sel
        caselaw_on = "caselaw" in sel

    # Vault gate: drop the manager AND strip the vault-dependent tools so they aren't offered.
    effective_retrieval_mgr = retrieval_mgr if vault_on else None
    run_tools = None  # None ⇒ the mode's default toolset
    if not vault_on:
        from src.components.agent_core.registry import _MODE_TOOLS
        _VAULT_TOOLS = {"search_vault", "read_document", "list_metrics", "table_lookup",
                        "compute", "survey_collection"}
        run_tools = [t for t in _MODE_TOOLS.get(mode, _MODE_TOOLS["standard"])
                     if t not in _VAULT_TOOLS]

    # KB gate: pass the manager only if at least one legal source is on; build the
    # instrument-type allow-list when exactly one of statutes/caselaw is selected.
    effective_kb_mgr = kb_retrieval_mgr if (statutes_on or caselaw_on) else None
    kb_instrument_types = None
    if effective_kb_mgr is not None and not (statutes_on and caselaw_on):
        kb_instrument_types = (_STATUTE_TYPES if statutes_on else []) + \
                              (_CASELAW_TYPES if caselaw_on else [])

    # ── Build the run: scope + budget + model + system prompt ──────────────────────
    from src.components.agent_core.registry import RunScope, REGISTRY
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.prompt import system_prompt
    from src.components.agent_core.loop import run_agent

    scope = RunScope(
        collection_id=collection_id,
        doc_ids=scoped_doc_ids,
        filenames=list(filename_filters),
        grids=grids,
        retrieval_manager=effective_retrieval_mgr,
        # G8: present only when USE_KNOWLEDGE is on AND a legal source chip is enabled
        # (else None ⇒ search_knowledge never offered). Shared + read-only — not the vault.
        kb_retrieval_manager=effective_kb_mgr,
        db_client=sb,
        filename_by_doc=filename_by_doc,
        question=body.question,
        # G3 Step E: conjunctive narrowing on top of the doc_id vault scope.
        filters=metadata_filter,
        # G5: the demoted Brain MAP step (survey_collection, deep mode) runs on this.
        config=user_config,
        # G8.7: the source-chip instrument allow-list enforced in search_knowledge.
        kb_instrument_types=kb_instrument_types,
    )
    budget = budget_for(mode, user_config)
    # G5: deep mode gets the Deep Analysis overlay (sectioned report + breadth-first
    # workflow) on top of the shared base contract; standard/fast get the base prompt.
    sys_prompt = system_prompt("v1", mode=mode)

    try:
        model = build_model(mode, budget, user_config, system=sys_prompt)
    except Exception as exc:  # vendor key missing etc. → degrade to a clean message, no 500
        logger.warning("[agentcore] model build failed (%s) — degrading", exc)

        def _degraded():
            yield _sse({"type": "meta", "mode": mode, "degrade": True, "error": str(exc)})
            yield _sse({"type": "token",
                        "content": "The agent core is unavailable right now. Please try the "
                                   "standard chat for this question."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_degraded(), media_type="text/event-stream")

    # ── The stream: iterate the sync loop generator (Starlette runs it in a worker
    # thread, so the blocking model.invoke stays off the event loop). Translate events. ─
    # Every run is TRACED to a durable JSONL journal + auto-flagged for loose-end signals
    # (a dead tool, a stuck model, budget exhaustion) so the next BUG-F shows up in the
    # health summary instead of being found by luck (§I1).
    from src.components.agent_core.tracer import RunTracer
    import uuid as _uuid
    tracer = RunTracer(run_id=_uuid.uuid4().hex[:12], question=body.question, mode=mode)

    # G5: deep mode binds the report PER SECTION (an unsupported section is withheld
    # visibly, the grounded ones ship intact). Standard mode keeps the whole-draft gate.
    # Both are non-bypassable; deep just applies the same logic section-by-section.
    # G6.1: a draft is a multi-section deliverable too (`##` headings), so it gets the
    # SAME per-section gate — one unsupported section is withheld visibly, never sinks the
    # whole document. Same non-bypassable rules; never a softer gate for prose.
    from src.components.agent_core.gates import run_output_gates, gate_sectioned
    gate_fn = gate_sectioned if mode in ("deep", "draft") else run_output_gates

    def _agent_stream():
        try:
            for ev in run_agent(
                body.question,
                model=model,
                scope=scope,
                budget=budget,
                system_prompt=sys_prompt,
                history=history,            # G5: prior conversation turns (read-only memory)
                registry=REGISTRY,
                gate_fn=gate_fn,
                tools=run_tools,            # G8.7: vault-off strips vault tools; None = default
            ):
                tracer.record(ev)  # durable journal + health (never raises)
                # Also log compactly to the API stdout for live tailing.
                try:
                    logger.info("[agentcore.ev] %s", json.dumps(ev, default=str)[:2000])
                except Exception:  # noqa: BLE001 — logging must never break the stream
                    logger.info("[agentcore.ev] <unserialisable %s>", ev.get("type"))
                yield _sse(_translate(ev))
        except Exception as exc:  # noqa: BLE001 — the loop shouldn't raise, but never 500 the stream
            logger.error("[agentcore] loop raised — degrading: %s", exc)
            tracer.record({"type": "meta", "mode": mode, "degrade": True, "error": str(exc),
                           "abstained": True})
            yield _sse({"type": "meta", "mode": mode, "degrade": True, "error": str(exc)})
            yield _sse({"type": "token",
                        "content": "I hit an internal error before I could verify an answer, "
                                   "so I'm not stating one. Please try again."})
        finally:
            # Persist the journal + surface health flags (warns in the log if flagged).
            health = tracer.finish()
            if health.get("flags"):
                # Emit a non-fatal trace_health event so the route/UI can see loose ends.
                yield _sse({"type": "trace_health", "run_id": health["run_id"],
                            "flags": health["flags"], "journal": health.get("journal")})
        yield "data: [DONE]\n\n"

    conversation_id = getattr(body, "conversation_id", None)
    return StreamingResponse(
        _saving_stream_wrapper(
            _agent_stream(),
            sb,
            conversation_id,
            body.question,
            is_agentic=True,
            retrieval_docs_count=len(scoped_doc_ids),
        ),
        media_type="text/event-stream",
    )
