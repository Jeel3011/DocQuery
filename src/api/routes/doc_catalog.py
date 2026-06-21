"""Doc-type catalog route (LEGAL_TASK_CATALOG_PLAN §2.3 — the card picker over the catalog).

`GET /doc-types` — the DRAFT cards: the catalog rendered as a picker. The card grid is
**generated from the catalog table, not authored one-by-one** (§2.3) — add a `DocType`
row and a card appears here. Each card carries its practice area + the form fields
(`required_inputs`) the §2.1 draft sources from the vault or brackets.

There is NO separate "run the catalog" endpoint: a card LAUNCHES the EXISTING draft path
(`POST /query/agentcore/stream` with `mode="draft"`). That route already composes
`doc_type` + `instructions` into the question; for a catalog doc-type it now builds the
India-correct, cite-or-bracket `instructions` from the catalog via `render_draft_request`
(see `agent_core.py`). One engine, one gate — this route only exposes the data.

PRIME DIRECTIVE (the plan's): flag OFF ⇒ this route 404s and nothing changes. Purely
additive until `USE_AGENT_CORE=true` — the same flag as Ask / Deep / draft / workflows.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_current_user, get_user_config
from src.components.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/doc-types")
async def list_doc_types_route(
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
):
    """The DRAFT catalog as picker cards, grouped by practice-area order. Flag off ⇒ 404
    (the catalog doesn't exist on a live path — byte-identical to pre-catalog)."""
    if not getattr(user_config, "USE_AGENT_CORE", False):
        raise HTTPException(status_code=404, detail="Not Found")
    from src.components.agent_core.doc_catalog import list_doc_types, doc_type_card
    return {"doc_types": [doc_type_card(d) for d in list_doc_types()]}
