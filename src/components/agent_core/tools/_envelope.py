"""The uniform tool-result envelope + provenance serializers (AGENT_CORE_PLAN §3.3).

Every agent-core tool returns the SAME JSON-able dict shape so the loop runner and
the output gates (§3.4) can treat tool results uniformly and accumulate provenance
into the evidence ledger without knowing which tool produced it:

    {
      "ok": bool,                # did the tool succeed?
      "summary": str,            # ONE line for the Trust-UI timeline
      "data": <json>,            # the tool's payload (tool-specific shape)
      "provenance": [ {...}, ],  # CellRef / chunk-span dicts — the ledger entries
      "abstain_reason": str?,    # present (and ok=False) when the tool declined to answer
      "error": str?,             # present (and ok=False) when the tool failed
    }

There is NO intelligence here — only shaping and serialization. The functions are
pure and never raise; `safe_tool` is the decorator that enforces the §3.2 contract
("a tool NEVER raises — an exception becomes {ok:false, error} AS A RESULT").
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def ok_result(
    summary: str,
    data: Any = None,
    provenance: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """A successful tool result."""
    return {
        "ok": True,
        "summary": summary,
        "data": data,
        "provenance": provenance or [],
    }


def abstain_result(
    reason: str,
    *,
    summary: Optional[str] = None,
    data: Any = None,
    provenance: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """A principled non-answer: the tool COULD have guessed but declined.

    `ok` is False (the model must adapt — disambiguate, read more, or surface the
    abstention to the user), but `abstain_reason` + any `candidates` in `data` give
    the model what it needs to act. This is the correct-or-abstain contract carried
    up to the agent layer; it is DISTINCT from `error_result` (a failure/exception).
    """
    return {
        "ok": False,
        "summary": summary or f"abstained: {reason}",
        "data": data,
        "provenance": provenance or [],
        "abstain_reason": reason,
    }


def error_result(error: str, *, summary: Optional[str] = None) -> Dict[str, Any]:
    """A tool failure (bad args, downstream exception). Never a guess, never a raise."""
    return {
        "ok": False,
        "summary": summary or f"error: {error}",
        "data": None,
        "provenance": [],
        "error": error,
    }


def safe_tool(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Decorator enforcing the §3.2 contract: a tool NEVER raises.

    Any exception escaping the wrapped function becomes an `error_result` so the
    loop runner can hand it back to the model as data. The exception is logged (the
    run ledger / observability floor, §3.7) but never propagated.
    """

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs) -> Dict[str, Any]:
        try:
            out = fn(*args, **kwargs)
            # A tool that returns the wrong shape is a programming error we still
            # don't want to crash the loop over — coerce defensively.
            if not isinstance(out, dict) or "ok" not in out:
                return error_result(
                    f"tool {fn.__name__} returned a non-envelope value",
                    summary="internal: bad tool result shape",
                )
            return out
        except Exception as exc:  # noqa: BLE001 — the whole point is to never raise
            logger.warning("[agent_core.tools] %s raised: %s", fn.__name__, exc)
            return error_result(f"{type(exc).__name__}: {exc}",
                                summary=f"error in {fn.__name__}")

    return _wrapped


# ── Provenance serializers (CellRef / chunk-span → plain dicts) ─────────────────

def cellref_to_dict(cell) -> Dict[str, Any]:
    """Serialize an analyst.CellRef into a ledger-ready provenance dict.

    Mirrors CellRef's fields exactly (doc/page/table_id/section/label/period/raw/
    value) plus a one-line `trace` so the UI and gates can render/check it without
    importing the dataclass. Accepts any object exposing those attributes.
    """
    if cell is None:
        return {}
    out = {
        "kind": "cell",
        "doc": getattr(cell, "doc", None),
        "page": getattr(cell, "page", None),
        "table_id": getattr(cell, "table_id", None),
        "section": getattr(cell, "section", ""),
        "label": getattr(cell, "label", ""),
        "period": getattr(cell, "period", ""),
        "raw": getattr(cell, "raw", ""),
        "value": getattr(cell, "value", None),
    }
    trace = getattr(cell, "trace", None)
    if callable(trace):
        try:
            out["trace"] = trace()
        except Exception:  # noqa: BLE001
            pass
    return out


def span_to_dict(doc: Any) -> Dict[str, Any]:
    """Serialize a retrieved chunk (LangChain Document-like) into a span provenance dict.

    Pulls the chunk text + the source addressing the Brain already stores in
    `metadata` (filename, page_number, chunk_id, plus the similarity score when the
    retriever attached one). Never raises; missing fields degrade to None.
    """
    md = getattr(doc, "metadata", None) or {}
    return {
        "kind": "span",
        "doc": md.get("filename") or md.get("source"),
        "page": md.get("page_number"),
        "chunk_id": md.get("chunk_id") or md.get("id"),
        "score": md.get("score") or md.get("relevance_score"),
        "snippet": (getattr(doc, "page_content", "") or "")[:500],
        # F1b: carry the vault id on every span so the cross-vault-leak invariant is
        # provable downstream (the F3 ledger / trust UI can show which vault each cited
        # chunk came from — and a leak is a span whose collection_id != the active vault).
        "collection_id": md.get("collection_id"),
    }
