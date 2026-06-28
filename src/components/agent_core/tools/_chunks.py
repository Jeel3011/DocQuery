"""Shared chunk-store access for the document harness (DOCUMENT_HARNESS §8 / §15.2).

The harness read tools (`search_text`, `read_document` whole-text, `read_section`)
all touch `document_chunks`, so the vault-isolation guard MUST be identical and in
ONE place — a leak here is a cross-vault read. This module owns:

  - `resolve_doc_id`  — the model speaks FILENAMES (what it saw in list_documents /
    results); the DB keys on the UUID `document_id`. Resolve exact, then unique
    substring, against the run's `filename_by_doc` — which IS the vault's document
    set. A `target` that doesn't resolve to a vault doc returns None ⇒ the caller
    refuses (no cross-vault read). Empty map ⇒ offline/test, pass through.

  - `text_chunks_for_doc` — the AUTHORITATIVE isolation pattern, copied verbatim from
    read.py:236-258 (the page-text query): resolve the reader client + user_id through
    the F2m `_shared`/`owner_id` logic, filter `user_id` (the service-role client
    bypasses RLS, so this app-layer filter IS the wall), and `metadata->>chunk_type =
    'text'`. Own vault reads via `read_client` (RLS defense-in-depth); a shared matter
    reads the OWNER's chunks via the service-role client (authorized upstream by
    db.accessible_vault_owner).

There is NO intelligence here — only scoped fetch + ordering. Never raises (callers
are @safe_tool; this returns [] on any failure so the tool degrades, never crashes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_doc_id(
    target: str,
    filename_by_doc: Optional[Dict[str, str]],
) -> Optional[str]:
    """Resolve a model-supplied doc reference (filename or UUID) to a vault UUID.

    `filename_by_doc` maps {uuid: filename} for THIS vault (resolved from collection_id
    at the route). Returns the UUID when `target` is one of this vault's docs (by id,
    exact filename, or unique substring), else None (caller refuses → no cross-vault
    read). Empty map ⇒ offline/test with no vault boundary: return `target` as-is.
    """
    fb = filename_by_doc or {}
    if not fb:
        return target or None  # offline/test: no vault boundary to enforce
    if target in fb:
        return target  # already a vault UUID
    uuid_by_fn = {fn: did for did, fn in fb.items()}
    if target in uuid_by_fn:
        return uuid_by_fn[target]  # exact filename
    # unique substring match (the kernel-consistent fallback read.py uses)
    subs = [did for fn, did in uuid_by_fn.items()
            if target and (target in fn or fn in target)]
    return subs[0] if len(subs) == 1 else None


def _reader_and_uid(db_client: Any, owner_id: Optional[str]):
    """The §15.2 isolation dance: resolve (reader_client, user_id) for the query.

    Copied from read.py:236-239. Own vault ⇒ read_client (caller's JWT, RLS as
    defense-in-depth) + caller's user_id. Shared matter (owner_id != caller) ⇒
    service-role client + the OWNER's user_id (RLS would block the owner's rows under
    the caller's JWT; access was authorized upstream by db.accessible_vault_owner).
    """
    caller_uid = getattr(db_client, "user_id", None)
    shared = bool(owner_id) and owner_id != caller_uid
    uid = owner_id if shared else caller_uid
    reader = db_client.client if shared else (getattr(db_client, "read_client", None) or db_client.client)
    return reader, uid


def text_chunks_for_doc(
    db_client: Any,
    resolved_id: str,
    *,
    owner_id: Optional[str] = None,
    limit: int = 4096,
) -> List[Dict[str, Any]]:
    """Fetch this doc's TEXT chunks (content + page + index), vault-isolated, ordered.

    Returns [{content, page, chunk_index}] sorted by (page_number, chunk_index) — the
    clean prose order (DOCUMENT_HARNESS §6.3). Vault isolation is baked into the query
    via `_reader_and_uid` + the `user_id` filter (§8). `limit` caps a runaway fetch
    (whole-doc reads of a huge filing — the read tool's token guard handles the rest).
    Never raises: returns [] on any error so the tool degrades to grids-only.
    """
    if not resolved_id:
        return []
    try:
        reader, uid = _reader_and_uid(db_client, owner_id)
        q = (
            reader.table("document_chunks")
            .select("content,metadata")
            .eq("document_id", resolved_id)
        )
        if uid:
            q = q.eq("user_id", uid)
        try:
            rows = q.eq("metadata->>chunk_type", "text").limit(limit).execute().data or []
        except Exception:  # noqa: BLE001 — JSONB filter unsupported → fetch + post-filter
            rows = q.limit(limit).execute().data or []
    except Exception:  # noqa: BLE001 — degrade, never raise
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        md = r.get("metadata") or {}
        if md.get("chunk_type") not in (None, "text"):
            continue
        out.append({
            "content": r.get("content", "") or "",
            "page": md.get("page_number"),
            "chunk_index": md.get("chunk_index"),
        })

    def _key(c: Dict[str, Any]):
        # page/chunk_index may be int, float, str, or None — coerce for a stable sort,
        # missing values sort last (huge sentinel).
        def _num(v):
            try:
                return int(float(v)) if v is not None and str(v).strip() != "" else 10**9
            except (ValueError, TypeError):
                return 10**9
        return (_num(c.get("page")), _num(c.get("chunk_index")))

    out.sort(key=_key)
    return out
