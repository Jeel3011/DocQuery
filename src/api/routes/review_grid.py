"""Review-grid route (AGENT_CORE_PLAN Phase B2).

`POST /review-grid/stream` — create + run a review grid (N documents × M columns) and
stream per-cell results over SSE. Each cell is one bounded agent run (grid_engine), so
the grid emits `grid_start → cell* → grid_done`. Behind `USE_AGENT_CORE` (flag off ⇒
404), exactly like the agent-core route, and it reuses that route's proven scope
assembly (collection → filenames → doc map → preloaded grids).

COST SAFETY: a grid is N×M PAID agent runs. The route enforces a hard cell ceiling
(`_MAX_GRID_CELLS`) and runs cells with a tight per-cell budget (grid_engine). Cells run
sequentially here (simple, resumable-friendly, predictable cost); a worker/pool fan-out
is a later Track-I optimization.

This is the first multi-run (and therefore potentially multi-dollar) endpoint. It is
purely additive until the flag is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_current_user, get_user_config, get_retrieval_mgr, limiter
from src.api.schemas import ReviewGridRequest
from src.components.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)

# Hard ceiling on N×M cells per grid — the cost/runtime guard. A 10-doc × 8-col grid =
# 80 paid runs; beyond that, the caller should narrow scope. Tunable via config later.
_MAX_GRID_CELLS = 120


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@router.post("/review-grid/stream")
@limiter.limit("3/minute")
async def review_grid_stream(
    request: Request,
    body: ReviewGridRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
):
    """Run a review grid, streaming `grid_start`, one `cell` per (doc×column), `grid_done`.

    Degradation: a per-cell failure becomes a cell with status=error (never a 500); a
    fatal setup error ends with a clean `grid_done` carrying an error.
    """
    # ── Prime directive: flag off ⇒ 404 ────────────────────────────────────────────
    if not getattr(user_config, "USE_AGENT_CORE", False):
        raise HTTPException(status_code=404, detail="Not Found")

    collection_id = body.collection_id
    if not collection_id:
        raise HTTPException(status_code=400, detail="collection_id is required for a review grid.")

    # ── Resolve the document set (rows) ────────────────────────────────────────────
    # All docs in the collection, mapped id ↔ filename (tools speak ids; retriever filenames).
    all_doc_ids = sb.get_collection_document_ids(collection_id) or []
    if not all_doc_ids:
        raise HTTPException(status_code=400, detail="The collection has no documents.")
    docs = sb.client.table("documents").select("id,filename").in_(
        "id", all_doc_ids
    ).eq("user_id", sb.user_id).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}

    # Caller may restrict rows; default = all docs in the collection.
    requested = [d for d in body.doc_ids if d in filename_by_doc] if body.doc_ids else all_doc_ids
    if not requested:
        raise HTTPException(status_code=400, detail="None of the requested doc_ids are in this collection.")

    # ── Cost ceiling ───────────────────────────────────────────────────────────────
    cell_count = len(requested) * len(body.columns)
    if cell_count > _MAX_GRID_CELLS:
        raise HTTPException(
            status_code=400,
            detail=(f"Grid is {cell_count} cells ({len(requested)} docs × {len(body.columns)} "
                    f"columns); the limit is {_MAX_GRID_CELLS}. Narrow the documents or columns."),
        )

    # ── Build the typed spec ───────────────────────────────────────────────────────
    from src.components.agent_core.review_grid import GridSpec, GridColumn, ColumnKind

    def _col(c) -> GridColumn:
        try:
            kind = ColumnKind(c.kind)
        except ValueError:
            kind = ColumnKind.CLAUSE
        return GridColumn(key=c.key, label=c.label, prompt=c.prompt, kind=kind,
                          risk_rubric=c.risk_rubric)

    spec = GridSpec(
        title=body.title,
        collection_id=collection_id,
        doc_ids=requested,
        columns=[_col(c) for c in body.columns],
    )

    # ── Preload grids for the whole doc set ONCE (shared across every cell) ─────────
    def _load_grids_by_doc() -> Dict[str, List[Any]]:
        try:
            from src.components.brain.table_intent import load_grids_for_docs
            by_doc: Dict[str, List[Any]] = {}
            for did in requested:
                grids = load_grids_for_docs(
                    sb, [did], question=None, filename_by_doc=filename_by_doc, per_doc_top=20,
                )
                by_doc[did] = grids or []
            return by_doc
        except Exception as exc:  # noqa: BLE001 — grids best-effort; cells degrade to read
            logger.warning("[review-grid] grid preload failed: %s", exc)
            return {}

    grids_by_doc = await asyncio.to_thread(_load_grids_by_doc)

    # ── Build a model factory (fresh model per cell, no shared state) ───────────────
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.grid_engine import build_cell

    # Cells use the "grid" tool set but the STANDARD model tier (one focused extraction
    # each). budget_for("standard") gives the configured model id; build_model wires the
    # vendor client. We build one model and reuse it across cells (it is stateless per
    # invoke); a failure to build → degrade the whole grid cleanly.
    grid_budget = budget_for("standard", user_config)
    try:
        model = build_model("standard", grid_budget, user_config, system="")
        model_id = grid_budget.model
    except Exception as exc:  # noqa: BLE001
        logger.warning("[review-grid] model build failed (%s) — degrading", exc)

        def _degraded():
            yield _sse({"type": "grid_start", "title": spec.title, "rows": len(requested),
                        "columns": len(spec.columns), "cells": cell_count})
            yield _sse({"type": "grid_done", "error": "agent core unavailable (model build failed)",
                        "coverage": {}})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_degraded(), media_type="text/event-stream")

    # ── Stream: grid_start → cell* → grid_done ─────────────────────────────────────
    # The cell loop is BLOCKING (each build_cell runs a sync agent loop). Run it in a
    # worker thread and hand results back through a queue the async generator drains, so
    # the event loop stays free (mirrors how the other streaming routes offload work).
    def _run_all_cells(emit) -> Dict[str, int]:
        from src.components.agent_core.review_grid import GridResult, CellStatus
        cells = []
        for did in spec.doc_ids:
            for column in spec.columns:
                cell = build_cell(
                    did, column,
                    collection_id=collection_id,
                    model=model,
                    filename_by_doc=filename_by_doc,
                    grids_by_doc=grids_by_doc,
                    retrieval_manager=retrieval_mgr,
                    db_client=sb,
                    model_id=model_id,
                )
                cells.append(cell)
                emit({
                    "type": "cell",
                    "doc_id": cell.doc_id,
                    "doc_name": cell.doc_name,
                    "column_key": cell.column_key,
                    "status": cell.status.value,
                    "value": cell.value,
                    "quote": cell.quote,
                    "risk": cell.risk.value,
                    "note": cell.note,
                    "provenance": cell.provenance,
                    "verified": cell.is_verified,
                })
        return GridResult(spec=spec, cells=cells).coverage()

    async def _stream():
        yield _sse({"type": "grid_start", "title": spec.title, "rows": len(requested),
                    "columns": len(spec.columns), "cells": cell_count,
                    "doc_names": [filename_by_doc.get(d, d) for d in requested],
                    "column_labels": [c.label for c in spec.columns]})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        _DONE = object()

        def emit(ev: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        async def _worker():
            try:
                coverage = await asyncio.to_thread(_run_all_cells, emit)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "grid_done", "coverage": coverage})
            except Exception as exc:  # noqa: BLE001 — never a 500 mid-stream
                logger.error("[review-grid] run failed: %s", exc)
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
