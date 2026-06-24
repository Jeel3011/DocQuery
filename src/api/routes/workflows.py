"""Workflows route (GRAND_PLAN §G7 · G7_WORKFLOWS_PLAN §4 G7.1).

`GET  /workflows`           — the gallery: the authored template cards (params_schema → form).
`POST /workflows/{id}/run`  — run a template over a vault, streaming the SAME events the
                              underlying engine already emits (agent-core for a REPORT
                              shape, review-grid for a GRID shape).

THE WHOLE POINT (§0): a workflow is **DATA the ONE engine interprets**, not a new code
path. This route reads the template, calls `resolve_run`, and then drives the EXISTING
engine — `run_agent` (report) or the review-grid `build_cell` fan-out (grid). It adds NO
orchestrator and NO per-template branch; it branches only on `shape`, a property of the
data. Scope assembly is the proven brain-route resolver, reused verbatim.

PRIME DIRECTIVE (the plan's): flag OFF ⇒ these routes 404 and nothing changes. Purely
additive until `USE_AGENT_CORE=true` — the same flag as Ask / Deep / draft / review-grid.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_current_user, get_user_config, get_retrieval_mgr, limiter, require_cap,
    assert_vault_not_screened,
)
from src.api.schemas import WorkflowRunRequest
from src.components.config import Config

# Reuse the proven scope helpers (same semantics as agent-core / review-grid).
from src.api.routes.chat import (
    _resolve_collection_filters,
    _saving_stream_wrapper,
    _embed_query,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Grid-shape fan-out ceiling — the SAME discipline as the review-grid route (a workflow
# never silently runs hundreds of paid calls; a template may also declare a lower ceiling).
_MAX_WORKFLOW_CELLS = 120


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


# ── GET /workflows — the gallery ────────────────────────────────────────────────
@router.get("/workflows")
async def list_workflows(
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """The authored templates as gallery cards. Flag off ⇒ 404 (the gallery doesn't exist)."""
    if not getattr(user_config, "USE_AGENT_CORE", False):
        raise HTTPException(status_code=404, detail="Not Found")
    from src.components.agent_core.workflows import list_templates, template_card
    return {"templates": [template_card(t) for t in list_templates()]}


# ── POST /workflows/{template_id}/run — run a template over a vault ───────────────
@router.post("/workflows/{template_id}/run")
@limiter.limit("3/minute")
async def run_workflow_stream(
    request: Request,
    template_id: str,
    body: WorkflowRunRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    _cap=Depends(require_cap("run_workflow")),
):
    """Resolve the template + params → a RunConfig, then drive the EXISTING engine and
    stream its events. Report → agent-core stream; grid → review-grid stream. Same gates.

    F2b: cap-gated on `run_workflow` — a working-toolkit verb (everyone on a matter, D0).
    """
    if not getattr(user_config, "USE_AGENT_CORE", False):
        raise HTTPException(status_code=404, detail="Not Found")

    from src.components.agent_core.workflows import get_template, resolve_run

    template = get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"No workflow template {template_id!r}.")

    collection_id = body.collection_id
    if not collection_id:
        raise HTTPException(status_code=400, detail="collection_id is required to run a workflow.")

    # F2c P5 (the ethical wall, workflow path): refuse a screened user before either shape
    # (grid / report) runs — both descend into search_vault / read_document on this vault.
    assert_vault_not_screened(sb, collection_id)

    run = resolve_run(template, body.params or {})
    metadata_filter = body.filters or None

    if run.shape == "grid":
        return await _run_grid_workflow(
            template, run, body, sb, user_config, retrieval_mgr, metadata_filter
        )
    # report (Draft) and output (Output) both ride a single run_agent — they differ only in
    # the gate (per-section vs whole-answer), resolved from the shape inside the report path.
    return await _run_report_workflow(
        template, run, body, sb, user_config, retrieval_mgr, metadata_filter
    )


# ── REPORT shape — one run_agent, mirroring the agent-core route ──────────────────
async def _run_report_workflow(template, run, body, sb, user_config, retrieval_mgr, metadata_filter):
    """A single bounded agent run shaped by the template overlay; the per-section gate binds
    the artifact. This is the agent-core route with (a) the composed question + overlay from
    the template and (b) the template's tool_subset threaded to the loop. No new engine."""
    from src.api.routes.agent_core import _translate  # reuse the SSE event translation

    question = run.question
    # Scope assembly — identical to the agent-core route.
    query_embedding: list = []
    try:
        query_embedding = await _embed_query(
            question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
    except Exception:
        pass

    filename_filters = _resolve_collection_filters(
        sb, body.collection_id, query=question, query_embedding=query_embedding,
        user_config=user_config, metadata_filter=metadata_filter,
    )
    if not filename_filters:
        def _empty():
            yield _sse({"type": "token", "content": "No documents found in this vault."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    doc_ids = sb.get_collection_document_ids(body.collection_id) or []
    docs = sb.read_client.table("documents").select("id,filename").in_(
        "id", doc_ids).eq("user_id", sb.user_id).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    routed = set(filename_filters)
    scoped_doc_ids = [did for did, fn in filename_by_doc.items() if fn in routed] or doc_ids

    def _load_grids():
        try:
            from src.components.brain.table_intent import load_grids_for_docs
            return load_grids_for_docs(
                sb, scoped_doc_ids, question=question,
                filename_by_doc=filename_by_doc, per_doc_top=20,
            )
        except Exception as exc:  # noqa: BLE001 — grids best-effort
            logger.warning("[workflow] grid preload failed: %s", exc)
            return []

    grids = await asyncio.to_thread(_load_grids)

    from src.components.agent_core.registry import RunScope, REGISTRY
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.loop import run_agent
    from src.components.agent_core.gates import gate_sectioned, run_output_gates

    # report (Draft) = multi-section memo → per-section gate; output (Output) = a freeform
    # single deliverable → the whole-answer gate. Both non-bypassable; never a softer gate.
    gate_fn = gate_sectioned if run.shape == "report" else run_output_gates

    scope = RunScope(
        collection_id=body.collection_id, doc_ids=scoped_doc_ids,
        filenames=list(filename_filters), grids=grids,
        retrieval_manager=retrieval_mgr, db_client=sb,
        filename_by_doc=filename_by_doc, question=question,
        filters=metadata_filter, config=user_config,
    )
    # Budget/model from the template's base_mode (reuse — report = standard/deep tier).
    budget = budget_for(run.base_mode if run.base_mode in ("standard", "deep") else "standard",
                        user_config)
    sys_prompt = run.system_prompt

    try:
        model = build_model(budget.mode, budget, user_config, system=sys_prompt)
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("[workflow] model build failed (%s) — degrading", exc)

        def _degraded():
            yield _sse({"type": "meta", "workflow": template.id, "degrade": True, "error": str(exc)})
            yield _sse({"type": "token", "content": "This workflow is unavailable right now."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_degraded(), media_type="text/event-stream")

    from src.components.agent_core.tracer import RunTracer
    import uuid as _uuid
    tracer = RunTracer(run_id=_uuid.uuid4().hex[:12], question=question, mode=f"workflow:{template.id}")

    def _agent_stream():
        try:
            yield _sse({"type": "meta", "workflow": template.id, "shape": run.shape,
                        "output_type": run.output_type})
            for ev in run_agent(
                question, model=model, scope=scope, budget=budget,
                system_prompt=sys_prompt, registry=REGISTRY,
                gate_fn=gate_fn,                 # per-section (report) or whole-answer (output)
                tools=run.tool_subset,           # G7: the template's restricted tool subset
            ):
                tracer.record(ev)
                # Mirror the agent-core route: log each event compactly so a workflow run
                # is tailable in the API stdout exactly like an Ask run (`[workflow.ev]`).
                try:
                    logger.info("[workflow.ev] %s %s", template.id, json.dumps(ev, default=str)[:1500])
                except Exception:  # noqa: BLE001 — logging must never break the stream
                    logger.info("[workflow.ev] %s <unserialisable %s>", template.id, ev.get("type"))
                yield _sse(_translate(ev))
        except Exception as exc:  # noqa: BLE001 — never 500 the stream
            logger.error("[workflow] report loop raised — degrading: %s", exc)
            yield _sse({"type": "meta", "workflow": template.id, "degrade": True, "error": str(exc)})
            yield _sse({"type": "token",
                        "content": "I hit an internal error before I could verify the workflow output."})
        finally:
            health = tracer.finish()
            if health.get("flags"):
                yield _sse({"type": "trace_health", "run_id": health["run_id"],
                            "flags": health["flags"], "journal": health.get("journal")})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _saving_stream_wrapper(
            _agent_stream(), sb, body.conversation_id, question,
            is_agentic=True, retrieval_docs_count=len(scoped_doc_ids),
        ),
        media_type="text/event-stream",
    )


# ── GRID shape — fan out the template's fixed columns, mirroring the review-grid route ─
async def _run_grid_workflow(template, run, body, sb, user_config, retrieval_mgr, metadata_filter):
    """The template fixes the columns (clause topics); the PROVEN review-grid engine runs
    them. Reuses the per-item loop + ceiling + per-cell grid gate verbatim — only the column
    source differs (the template, not the request). Streams grid_start → cell* → grid_done."""
    from src.components.agent_core.review_grid import GridSpec, GridResult
    from src.components.agent_core.workflows import resolve_grid_columns

    all_doc_ids = sb.get_collection_document_ids(body.collection_id) or []
    if not all_doc_ids:
        raise HTTPException(status_code=400, detail="The vault has no documents.")

    docs = sb.read_client.table("documents").select("id,filename,doc_type,fiscal_year").in_(
        "id", all_doc_ids).eq("user_id", sb.user_id).execute()
    rows_by_doc = {d["id"]: d for d in (docs.data or [])}
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}

    requested = [d for d in body.doc_ids if d in filename_by_doc] if body.doc_ids else all_doc_ids

    if metadata_filter:
        from src.components.document_router import _doc_matches_filter
        narrowed = [d for d in requested if _doc_matches_filter(rows_by_doc.get(d, {}), metadata_filter)]
        if narrowed:
            requested = narrowed

    if not requested:
        raise HTTPException(status_code=400, detail="No documents in scope for this workflow.")

    # Columns are dynamic where a template builds them from a param (e.g. the diligence
    # request list = one column per item); else the template's static columns.
    columns = resolve_grid_columns(template, body.params or {})
    if not columns:
        raise HTTPException(status_code=400,
                            detail="This workflow needs at least one item/column to review.")
    ceiling = min(run.max_cells, _MAX_WORKFLOW_CELLS)
    cell_count = len(requested) * len(columns)
    if cell_count > ceiling:
        raise HTTPException(
            status_code=400,
            detail=(f"Workflow is {cell_count} cells ({len(requested)} docs × {len(columns)} "
                    f"columns); the limit is {ceiling}. Narrow the documents."),
        )

    spec = GridSpec(title=template.title, collection_id=body.collection_id,
                    doc_ids=requested, columns=columns)

    def _load_grids_by_doc() -> Dict[str, List[Any]]:
        try:
            from src.components.brain.table_intent import load_grids_for_docs
            by_doc: Dict[str, List[Any]] = {}
            for did in requested:
                by_doc[did] = load_grids_for_docs(
                    sb, [did], question=None, filename_by_doc=filename_by_doc, per_doc_top=20,
                ) or []
            return by_doc
        except Exception as exc:  # noqa: BLE001 — grids best-effort
            logger.warning("[workflow-grid] grid preload failed: %s", exc)
            return {}

    grids_by_doc = await asyncio.to_thread(_load_grids_by_doc)

    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.grid_engine import build_cell

    grid_budget = budget_for("standard", user_config)
    try:
        model = build_model("standard", grid_budget, user_config, system="")
        model_id = grid_budget.model
    except Exception as exc:  # noqa: BLE001 — degrade cleanly
        logger.warning("[workflow-grid] model build failed (%s) — degrading", exc)

        def _degraded():
            yield _sse({"type": "grid_start", "title": spec.title, "rows": len(requested),
                        "columns": len(columns), "cells": cell_count})
            yield _sse({"type": "grid_done", "error": "agent core unavailable", "coverage": {}})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_degraded(), media_type="text/event-stream")

    def _run_all_cells(emit) -> Dict[str, int]:
        cells = []
        for did in spec.doc_ids:
            for column in spec.columns:
                cell = build_cell(
                    did, column, collection_id=body.collection_id, model=model,
                    filename_by_doc=filename_by_doc, grids_by_doc=grids_by_doc,
                    retrieval_manager=retrieval_mgr, db_client=sb, model_id=model_id,
                )
                cells.append(cell)
                emit({
                    "type": "cell", "doc_id": cell.doc_id, "doc_name": cell.doc_name,
                    "column_key": cell.column_key, "status": cell.status.value,
                    "value": cell.value, "quote": cell.quote, "risk": cell.risk.value,
                    "note": cell.note, "abstain_reason": cell.abstain_reason,
                    "provenance": cell.provenance, "verified": cell.is_verified,
                })
        return GridResult(spec=spec, cells=cells).coverage()

    async def _stream():
        yield _sse({"type": "grid_start", "title": spec.title, "rows": len(requested),
                    "columns": len(columns), "cells": cell_count,
                    "workflow": template.id,
                    "doc_names": [filename_by_doc.get(d, d) for d in requested],
                    "column_labels": [c.label for c in columns]})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        _DONE = object()

        def emit(ev: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        async def _worker():
            try:
                coverage = await asyncio.to_thread(_run_all_cells, emit)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "grid_done", "coverage": coverage})
            except Exception as exc:  # noqa: BLE001 — never 500 mid-stream
                logger.error("[workflow-grid] run failed: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "grid_done", "error": str(exc), "coverage": {}})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        task = asyncio.create_task(_worker())
        try:
            while True:
                ev = await queue.get()
                if ev is _DONE:
                    break
                yield _sse(ev)
        finally:
            task.cancel()
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
