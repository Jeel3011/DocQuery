"""`read_document` tool — thin adapter over grid loading + chunk fetch (AGENT_CORE_PLAN §3.3).

Wraps `table_intent.load_grids_for_docs` (the structured-grid loader the spine already
uses) plus a page-scoped text-chunk fetch. Returns the page text and the grid JSONs
(headers/rows/periods) for the requested document so the model can read structure
before it claims anything. No new intelligence: it loads what ingest stored.

Two dependency shapes, both honored so the tool is testable offline and live:
  - LIVE: pass `db_client` (a `db.SupabaseClient`-like). Grids come from
    `load_grids_for_docs`; text chunks from a doc/page-scoped select.
  - OFFLINE / pre-loaded: pass `grids=[...]` (already-built `analyst.Grid`s, e.g. from
    `extract_tables_from_pdf` in a gate). The DB is never touched.

Gate: extraction benchmark (the grids themselves, exists, 100%); `eval/test_tools.py`
(this adapter's envelope + never-raise on real grids).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._envelope import error_result, ok_result, safe_tool

SCHEMA: Dict[str, Any] = {
    "name": "read_document",
    "description": (
        "Read a document's structured table grids (and optionally its page text) so "
        "you can see the actual rows, sections, and periods before making any claim. "
        "Returns grid JSONs (headers, rows, periods) with source addressing. Use this "
        "before table_lookup/compute when you are unsure what rows or periods exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "The document id to read."},
            "page_range": {
                "type": "string",
                "description": "Optional 'start-end' page range to scope text, e.g. '40-45'.",
            },
            "table_grids": {
                "type": "boolean",
                "description": "Include structured table grids (default true).",
            },
        },
        "required": ["doc_id"],
    },
}


def _grid_to_json(g: Any) -> Dict[str, Any]:
    """Serialize an analyst.Grid into a compact, model-readable JSON view."""
    return {
        "doc": getattr(g, "doc", None),
        "page": getattr(g, "page", None),
        "table_id": getattr(g, "table_id", None),
        "summary": getattr(g, "summary", "") or "",
        "headers": getattr(g, "headers", []),
        "periods": getattr(g, "periods", []),
        "units": getattr(g, "units", None),
        "rows": getattr(g, "rows", []),
    }


def _parse_page_range(page_range: Optional[str]):
    if not page_range:
        return None
    try:
        if "-" in page_range:
            a, b = page_range.split("-", 1)
            return int(a), int(b)
        p = int(page_range)
        return p, p
    except Exception:  # noqa: BLE001
        return None


@safe_tool
def read_document(
    doc_id: str,
    *,
    db_client: Any = None,
    grids: Optional[List[Any]] = None,
    question: Optional[str] = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    page_range: Optional[str] = None,
    table_grids: bool = True,
    scope_grids: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Read grids (+ optional page text) for `doc_id`; return the §3.3 envelope.

    Provenance lists one span per grid (doc/page) so the ledger records what was read.
    """
    if not doc_id and not grids:
        return error_result("read_document requires a 'doc_id' (or pre-loaded grids)")

    def _fresh_load(target: str) -> List[Any]:
        """Load `target`'s grids from the DB and JOIN them to the run's grid scope.

        Live (2026-06-11): the model read the right document, but the grids never
        reached `compute`'s scope — "document not in scope" on an answerable
        question. Freshly loaded grids are appended to `scope_grids` (the SAME list
        the registry hands to compute/table_lookup), de-duplicated, so read → compute
        works even when the preload missed the doc."""
        from src.components.brain.table_intent import load_grids_for_docs

        # load_grids_for_docs keys on the DB doc_id (UUID). The model speaks filenames;
        # resolve exact, then unique-substring.
        _fb = filename_by_doc or {}
        _uuid_by_fn = {fn: did for did, fn in _fb.items()}
        gid = _uuid_by_fn.get(target)
        if gid is None:
            subs = [did for fn, did in _uuid_by_fn.items()
                    if target and (target in fn or fn in target)]
            gid = subs[0] if len(subs) == 1 else target
        # F1b/H2 (cross-vault membership guard): `gid` must be a doc IN THIS VAULT. The run's
        # filename_by_doc IS the vault's document set (resolved from collection_id at the
        # route). A `target` that resolves to a raw, non-vault id (the old passthrough) would
        # let the agent read another vault's grids — refuse it. (Empty _fb ⇒ offline/test path
        # with no vault scope: allow, since there's no vault boundary to cross.)
        if _fb and gid not in _fb:
            return []  # not in this vault → caller surfaces "not in scope" (no cross-vault read)
        fresh = load_grids_for_docs(
            db_client, [gid], question=question, filename_by_doc=filename_by_doc,
        )
        if fresh and scope_grids is not None:
            seen = {(getattr(g, "doc", None), getattr(g, "page", None),
                     getattr(g, "table_id", None)) for g in scope_grids}
            for g in fresh:
                k = (getattr(g, "doc", None), getattr(g, "page", None),
                     getattr(g, "table_id", None))
                if k not in seen:
                    scope_grids.append(g)
                    seen.add(k)
        return fresh

    def _doc_like(grid_doc: Any, want: str) -> bool:
        """Kernel-consistent doc matching (a distinctive substring is enough)."""
        try:
            from src.components.brain.analyst import _doc_match
            return _doc_match(grid_doc, want)
        except Exception:  # noqa: BLE001 — fall back to exact/substring
            gd = str(grid_doc or "")
            return bool(want) and (gd == want or want in gd or gd in want)

    loaded: List[Any] = []
    if grids:
        # Pre-loaded path: the run scope already holds grids. Honor the model's doc_id
        # against the grids' doc labels (the model knows docs by the filename it saw in
        # search results).
        loaded = list(grids)
        if doc_id:
            scoped = [g for g in loaded if _doc_like(getattr(g, "doc", None), doc_id)]
            if scoped:
                loaded = scoped
            elif table_grids and db_client is not None:
                # The doc is real but its grids weren't preloaded — load them fresh and
                # join the compute scope. (The old behaviour fell back to ALL preloaded
                # grids: the model believed it read doc X while seeing other documents'
                # tables — actively misleading, observed live.)
                loaded = _fresh_load(doc_id)
                if not loaded:
                    have = sorted({str(getattr(g, "doc", "")) for g in grids})
                    return error_result(
                        f"document {doc_id!r} is not in the loaded scope and no grids "
                        f"were found for it in the database (loaded docs: {have})"
                    )
            else:
                have = sorted({str(getattr(g, "doc", "")) for g in grids})
                return error_result(
                    f"document {doc_id!r} is not in the loaded scope "
                    f"(loaded docs: {have}) — use one of those document names"
                )
    elif table_grids and db_client is not None:
        loaded = _fresh_load(doc_id)

    # Honor the model's page_range against the grids (both paths). Without this the
    # adapter returned EVERY grid on the doc (e.g. 60) regardless of the requested
    # pages — each grid's full JSON floods the message history and blows the token
    # budget before the model can answer (observed live: read_document → 60 grids →
    # 75k tokens at step 5 → forced abstain on an ALREADY-CORRECT figure). Scope to
    # the range when the page is known; if nothing falls in range, keep the doc-scoped
    # set (showing structure beats returning nothing).
    _pr = _parse_page_range(page_range)
    if _pr is not None and loaded:
        lo, hi = _pr

        def _page_num(g):
            # grid.page may be int OR float ('39.0' from JSON) OR a numeric string. The
            # old `isinstance(..., int)` check rejected floats → page_range="39-39" kept
            # ALL grids (live 2026-06-11: read_document returned 20 grids/call, ~4k
            # tokens each, exhausting the 120k budget before compute ran). Coerce.
            p = getattr(g, "page", None)
            try:
                return int(float(p)) if p is not None and str(p).strip() != "" else None
            except (ValueError, TypeError):
                return None

        in_range = [g for g in loaded
                    if (_pn := _page_num(g)) is not None and lo <= _pn <= hi]
        if in_range:
            loaded = in_range

    grid_jsons = [_grid_to_json(g) for g in loaded] if table_grids else []

    # Page text: only available with a live db_client; scoped to the page range.
    # The model addresses docs by the FILENAME it saw in search results, but the DB's
    # document_id is a UUID — passing a filename as document_id is a 400 Bad Request
    # (observed live on an MSFT question: the model retried read_document, burned steps,
    # and produced no answer). Resolve filename→UUID via the reverse of filename_by_doc;
    # if we can't resolve to a UUID, skip the text query (the grids already answer the
    # numeric question — text is supplementary).
    page_text: List[Dict[str, Any]] = []
    pr = _parse_page_range(page_range)
    fb = filename_by_doc or {}
    uuid_by_filename = {fn: did for did, fn in fb.items()}
    resolved_id = uuid_by_filename.get(doc_id, doc_id if doc_id in fb else None)
    if db_client is not None and pr is not None and resolved_id is not None:
        try:
            # BUG-2 fix: push the chunk_type filter to the DB (JSONB ->> operator, same
            # as load_grids_for_docs) and cap the rows. The old query pulled EVERY chunk
            # for the doc then filtered in Python — ~15s/call and a token sink (observed
            # live). Text chunks for a 1-2 page range are few; 64 is a generous ceiling.
            # F1b/H2 (cross-user leak fix): the service-role client bypasses RLS, so this
            # user_id filter is the app-layer guard isolating chunks per user — without it an
            # injected/foreign document_id loads its text. _uid None ⇒ offline/test (no live
            # data). See the same guard in brain/table_intent.load_grids_for_docs.
            # F1 RLS hardening (defense-in-depth): READ through `read_client` when present — on
            # the request path it carries the user's JWT so Postgres RLS ALSO enforces
            # auth.uid()=user_id (a data-layer backstop). Falls back to `.client` offline.
            _uid = getattr(db_client, "user_id", None)
            _reader = getattr(db_client, "read_client", None) or db_client.client
            q = (
                _reader.table("document_chunks")
                .select("content,metadata")
                .eq("document_id", resolved_id)
            )
            if _uid:
                q = q.eq("user_id", _uid)
            try:
                rows = q.eq("metadata->>chunk_type", "text").limit(64).execute().data or []
            except Exception:  # noqa: BLE001 — JSONB filter unsupported → fall back, still capped
                rows = q.limit(256).execute().data or []
            lo, hi = pr
            for r in rows:
                md = r.get("metadata") or {}
                if md.get("chunk_type") == "table":
                    continue
                pg = md.get("page_number")
                if isinstance(pg, int) and lo <= pg <= hi:
                    page_text.append({"page": pg, "text": r.get("content", "")})
        except Exception:  # noqa: BLE001 — degrade to grids-only, never raise
            page_text = []

    provenance = [
        {"kind": "span", "doc": gj["doc"], "page": gj["page"], "table_id": gj["table_id"]}
        for gj in grid_jsons
    ]
    summary = f"read_document {doc_id}: {len(grid_jsons)} grid(s), {len(page_text)} text page(s)"
    return ok_result(
        summary=summary,
        data={"doc_id": doc_id, "grids": grid_jsons, "page_text": page_text},
        provenance=provenance,
    )
