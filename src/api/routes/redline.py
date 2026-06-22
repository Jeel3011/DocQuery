"""G6.3 — Redline route.

`POST /redline/stream` — run the redline engine for one target document against a
playbook, streaming findings as SSE `finding` events (one per clause topic) then a
`redline_done` event. Behind `USE_AGENT_CORE` (flag off ⇒ 404), same posture as
the review-grid route.

Cost: N paid agent runs (one per clause topic). The route enforces a hard ceiling
on clause topics per run. Small contracts only; Jeel's explicit go per live run.

Shape of `finding` events:
  {"type": "finding", "clause_topic": str, "status": str,
   "target_quote": str|null, "deviation": str|null,
   "suggested_edit": str|null, "rationale": str|null, "grounded": bool}

Shape of `redline_done`:
  {"type": "redline_done", "deviations": int, "conforming": int,
   "missing": int, "abstained": int}
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_current_user,
    get_user_config,
    get_retrieval_mgr,
    limiter,
)
from src.components.config import Config
from src.api.routes.agent_core import _sse

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_REDLINE_TOPICS = 30   # hard ceiling per run (cost guard × N topics)


class RedlineRequest(BaseModel):
    collection_id: str
    doc_id: str                                   # the single target document to redline
    # Either pass explicit playbook_rows, OR a catalog doc_type to derive them from
    # (§2.1 REDLINE-from-a-catalog-doc-type link). At least one must be present.
    playbook_rows: List[Dict[str, Any]] = Field(  # [{clause_topic, standard_position, ...}]
        default_factory=list, max_length=_MAX_REDLINE_TOPICS
    )
    doc_type: Optional[str] = None                # a catalog DocType id (e.g. "nda", "spa")
    title: Optional[str] = None


@router.post("/redline/stream")
async def redline_stream(
    request: Request,
    body: RedlineRequest,
    sb=Depends(get_current_user),
    user_config=Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
):
    cfg: Config = user_config

    if not cfg.USE_AGENT_CORE:
        raise HTTPException(status_code=404, detail="Redline is unavailable (USE_AGENT_CORE is off).")

    # §2.1 — redline FROM a catalog doc type. If no explicit playbook_rows were passed but a
    # catalog doc_type was, derive the clause-topic rows from the doc type's structure,
    # preferring the firm's stored playbook positions where they exist (§3.4 split).
    if not body.playbook_rows and body.doc_type:
        from src.components.agent_core.doc_catalog import get_doc_type, playbook_rows_for_doc_type

        dt = get_doc_type(body.doc_type)
        if dt is None:
            raise HTTPException(status_code=404, detail=f"Unknown doc type '{body.doc_type}'.")

        # Pull the firm's playbook (best-effort; an empty/failed fetch → structure-only check).
        user_rows: List[Dict[str, Any]] = []
        try:
            q = sb.read_client.table("playbooks").select(
                "clause_topic, standard_position, fallback_position"
            ).eq("user_id", sb.user_id)
            res = q.execute()
            user_rows = res.data or []
        except Exception as exc:  # noqa: BLE001 — a playbook fetch failure must not 500 the run
            logger.warning("[redline] playbook fetch failed (structure-only fallback): %s", exc)

        body.playbook_rows = playbook_rows_for_doc_type(
            dt, user_rows, max_rows=_MAX_REDLINE_TOPICS
        )

    if not body.playbook_rows:
        raise HTTPException(
            status_code=400,
            detail="Provide either playbook_rows or a known catalog doc_type to redline against.",
        )

    n_topics = len(body.playbook_rows)
    if n_topics > _MAX_REDLINE_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many clause topics ({n_topics}); limit is {_MAX_REDLINE_TOPICS}.",
        )

    # ── Scope assembly + vault-membership check (H6) ────────────────────────────────────
    # Build the vault's full doc map (id ↔ filename) the SAME way the agent-core route does
    # (routes/agent_core.py:192-196): join via collections.user_id so a foreign collection_id
    # resolves to nothing (no cross-user read), and the map IS the vault's membership set that
    # read_document's H2 guard checks against. This runs OUTSIDE the SSE generator so a bad
    # request fails fast with a real status code (a 400/404), not a mid-stream `error` event.
    all_doc_ids = sb.get_collection_document_ids(body.collection_id) or []
    if not all_doc_ids:
        raise HTTPException(status_code=400, detail="The collection has no documents.")
    docs = sb.read_client.table("documents").select("id,filename").in_(
        "id", all_doc_ids
    ).eq("user_id", sb.user_id).execute()
    filename_by_doc: Dict[str, str] = {d["id"]: d["filename"] for d in (docs.data or [])}

    # H6: REJECT a target doc_id that is not a member of this vault. The redline is scoped to
    # ONE document; if it isn't in the collection we must NOT silently fall back to all vault
    # docs (the pre-fix behaviour) — that would either review the wrong document or, worse,
    # widen the scope. A foreign/unknown id is a hard 404 (the agent-core floor would refuse
    # it anyway, but failing here is honest and immediate). This is the cross-vault membership
    # check the data layer already enforces (read_document H2) surfaced up front.
    if body.doc_id not in filename_by_doc:
        raise HTTPException(
            status_code=404,
            detail="doc_id is not a document in this collection.",
        )
    doc_name = filename_by_doc[body.doc_id]

    async def _stream():
        try:
            from src.components.agent_core.model import build_model
            from src.components.agent_core.budgets import budget_for
            from src.components.agent_core.redline import build_redline_cell
            from src.components.agent_core.registry import RunScope

            # Preload the target doc's grids ONCE (shared across every clause-topic cell).
            # Off the event loop — load_grids_for_docs does blocking DB reads. Best-effort:
            # a contract is prose, so a failed/empty grid load just means cells lean on
            # search_vault(kind="text") — never a 500. The membership check + filename map
            # were resolved above (outside the stream), so this only fetches grids.
            def _load_grids():
                try:
                    from src.components.brain.table_intent import load_grids_for_docs
                    return load_grids_for_docs(
                        sb, [body.doc_id],
                        question=None, filename_by_doc=filename_by_doc, per_doc_top=20,
                    ) or []
                except Exception as exc:  # noqa: BLE001 — grids best-effort; cells degrade to read
                    logger.warning("[redline] grid preload failed: %s", exc)
                    return []

            grids = await asyncio.to_thread(_load_grids)

            # Scope to ONE document (the redline target). doc_ids is the stable, ingest-stamped
            # axis the agent-core vault-scope FLOOR (search.py) and the H2 read-membership guard
            # (read.py, via filename_by_doc) both enforce — so every clause cell is provably
            # locked to this one doc in this one vault. filename_by_doc is the FULL vault map so
            # read_document's membership check has the real vault boundary to test against.
            scope = RunScope(
                collection_id=body.collection_id,
                doc_ids=[body.doc_id],
                filenames=[doc_name],
                grids=grids,
                retrieval_manager=retrieval_mgr,
                db_client=sb,
                filename_by_doc=filename_by_doc,
                question=None,
                config=cfg,
            )

            budget = budget_for("grid", cfg)
            sys_prompt_placeholder = ""  # per-cell prompt is built inside build_redline_cell

            def model_factory(system: str = ""):
                return build_model("grid", budget, cfg, system=system)

            counts: Dict[str, int] = {"deviation": 0, "conforming": 0, "missing": 0, "abstain": 0}

            for row in body.playbook_rows:
                finding = await asyncio.to_thread(
                    build_redline_cell,
                    row["clause_topic"],
                    row["standard_position"],
                    row.get("fallback_position"),
                    model_factory,
                    scope,
                    doc_name,
                )
                counts[finding.status] = counts.get(finding.status, 0) + 1
                yield _sse({
                    "type": "finding",
                    "clause_topic": finding.clause_topic,
                    "status": finding.status,
                    "target_quote": finding.target_quote,
                    "deviation": finding.deviation,
                    "suggested_edit": finding.suggested_edit,
                    "rationale": finding.rationale,
                    "playbook_standard": finding.playbook_standard,
                    "grounded": finding.grounded,
                })

            yield _sse({
                "type": "redline_done",
                "deviations": counts.get("deviation", 0),
                "conforming": counts.get("conforming", 0),
                "missing": counts.get("missing", 0),
                "abstained": counts.get("abstain", 0),
            })
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("[redline] stream error: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
