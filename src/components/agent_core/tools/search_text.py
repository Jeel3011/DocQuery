"""`search_text` tool — the harness `grep` (DOCUMENT_HARNESS §6.2 / §15.2).

EXACT, lexical, deterministic search over the ACTUAL document text — the substrate
that replaces `search_vault`'s embedding top-k (which MISSES when the query words
don't match the clause's words → false abstain). The model supplies meaning by
searching several ways (`any_of` synonym sets); grep supplies precision.

Phase 1 implementation (no new infra): fetch this matter's TEXT chunks (vault-isolated
via `_chunks`), then in Python extract the matching LINES per chunk and emit one result
per hit with its doc/page/snippet. The matter is bounded (a few hundred–few thousand
chunks), so a scan is fast (§6.2). Phase 4 swaps in a Postgres FTS GIN index behind the
same tool contract — never embeddings.

Security (§8): every fetch is scoped to the run's `doc_ids`/`filename_by_doc` (the
vault's document set) and isolated by `user_id`/`owner_id`. A foreign or out-of-scope
`doc_ids` entry resolves to nothing → it never fans out (the leak test asserts this).

G-e (ReDoS): `is_regex` defaults False; a regex is length-capped and compiled once.
Catastrophic-backtracking input can't reach the DB — Phase 1 matches in Python, and a
bad regex just fails to compile → error envelope, never a CPU spike or a raise.

Gate: `eval/test_doc_harness.py` (envelope + line extraction + the leak test).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._chunks import resolve_doc_id, text_chunks_for_doc
from ._envelope import error_result, ok_result, safe_tool

_MAX_REGEX_LEN = 200          # G-e: cap regex source length (ReDoS guard)
_DEFAULT_K = 20               # max matching passages returned
_SNIPPET_PAD = 120            # chars of context around a hit
_MATTER_CHUNK_CAP = 6000      # ceiling on chunks scanned per doc (matter is bounded)

SCHEMA: Dict[str, Any] = {
    "name": "search_text",
    "description": (
        "Search the ACTUAL TEXT of the matter's documents for exact words/phrases (or a "
        "regex). Returns matching lines with their document and page, so you can read "
        "around them — like `grep`. This is EXACT matching, not semantic: if unsure of "
        "the wording, pass several synonyms via `any_of` (e.g. ['indemnify','hold "
        "harmless']) and they are OR-ed. For a single small document, prefer reading it "
        "whole with read_document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words/phrase to find (or a regex if is_regex=true)."},
            "any_of": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional synonym set — match ANY of these terms (OR-ed with query).",
            },
            "doc_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict to these documents (filenames or ids). Default: the whole matter.",
            },
            "is_regex": {"type": "boolean", "description": "Treat query as a regex (default false)."},
            "k": {"type": "integer", "description": "Max matching passages (default 20)."},
        },
        "required": ["query"],
    },
}


def _compile_terms(query: str, any_of: Optional[List[str]], is_regex: bool):
    """Build a list of compiled, case-insensitive matchers from query + any_of.

    Plain terms are escaped (literal match); a regex query is compiled raw after a
    length check (G-e). Returns (matchers, error) — error is a string when a regex is
    invalid/too long, so the caller emits an error envelope instead of raising.
    """
    terms = [t for t in ([query] + list(any_of or [])) if t and t.strip()]
    if not terms:
        return None, "search_text requires a non-empty 'query' (or 'any_of' terms)."
    matchers = []
    for t in terms:
        if is_regex:
            if len(t) > _MAX_REGEX_LEN:
                return None, f"regex too long ({len(t)} > {_MAX_REGEX_LEN} chars) — refused (ReDoS guard)."
            try:
                matchers.append(re.compile(t, re.IGNORECASE))
            except re.error as exc:
                return None, f"invalid regex {t!r}: {exc}"
        else:
            matchers.append(re.compile(re.escape(t), re.IGNORECASE))
    return matchers, None


def _split_lines(text: str) -> List[str]:
    """Split a chunk into matchable lines (newline first; fall back to sentences)."""
    parts = [ln for ln in re.split(r"\n+", text) if ln.strip()]
    if len(parts) > 1:
        return parts
    # Single block (clause-chunked prose) → sentence-ish split so a hit is localized.
    return [s for s in re.split(r"(?<=[.;:])\s+", text) if s.strip()]


@safe_tool
def search_text(
    query: str,
    *,
    any_of: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    is_regex: bool = False,
    k: int = _DEFAULT_K,
    db_client: Any = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    scope_doc_ids: Optional[List[str]] = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Grep the matter's text for `query`/`any_of`; return the §3.3 envelope with one
    span per hit (doc/page/snippet) — the same span shape a `search_vault` hit yields,
    so the ledger and the output gate are unchanged.

    Scope (§8): the docs searched are the INTERSECTION of the run's vault docs
    (`scope_doc_ids` / `filename_by_doc`) with the model's optional `doc_ids` — each
    resolved through `resolve_doc_id`, so an out-of-scope reference resolves to nothing.
    """
    matchers, err = _compile_terms(query, any_of, is_regex)
    if err:
        return error_result(err)

    if db_client is None:
        return error_result(
            "search_text requires a live db_client (internal: the run context did not "
            "inject one; this is a routing bug, not a model error — report it)."
        )

    # ── Resolve the target document set, vault-scoped (§8) ───────────────────────────
    # Start from the vault's docs; if the model named doc_ids, intersect with them.
    vault_ids = list(scope_doc_ids or list((filename_by_doc or {}).keys()))
    if doc_ids:
        resolved = [resolve_doc_id(d, filename_by_doc) for d in doc_ids]
        resolved = [r for r in resolved if r]  # drop out-of-scope (None) → no leak
        # keep only those that are actually in the vault when a vault set is known
        targets = [r for r in resolved if (not vault_ids or r in vault_ids)]
    else:
        targets = vault_ids

    if not targets:
        # Active vault with nothing resolvable to search → refuse (never fan out).
        return error_result(
            "search_text found no in-scope documents to search (the requested doc_ids "
            "are not in this matter, or the matter is empty)."
        )

    cap = max(int(k or _DEFAULT_K), 1)
    fname = filename_by_doc or {}
    hits: List[Dict[str, Any]] = []

    for did in targets:
        if len(hits) >= cap:
            break
        chunks = text_chunks_for_doc(db_client, did, owner_id=owner_id, limit=_MATTER_CHUNK_CAP)
        doc_name = fname.get(did, did)
        for c in chunks:
            content = c.get("content", "")
            if not content:
                continue
            # Cheap pre-filter: does ANY matcher hit the chunk at all?
            if not any(m.search(content) for m in matchers):
                continue
            for line in _split_lines(content):
                if len(hits) >= cap:
                    break
                hit_spans = [m.search(line) for m in matchers]
                first = next((s for s in hit_spans if s), None)
                if first is None:
                    continue
                start = max(first.start() - _SNIPPET_PAD, 0)
                end = min(first.end() + _SNIPPET_PAD, len(line))
                snippet = line[start:end].strip()
                hits.append({
                    "kind": "span",
                    "doc": doc_name,
                    "doc_id": did,
                    "page": c.get("page"),
                    "chunk_id": None,
                    "chunk_index": c.get("chunk_index"),
                    "snippet": snippet,
                })

    n_docs = len({h["doc_id"] for h in hits})
    summary = f"search_text {query!r}: {len(hits)} hit(s) across {n_docs} doc(s)"
    if not hits:
        summary = (
            f"search_text {query!r}: 0 hits — try synonyms via any_of, or read the "
            f"document whole if it's small"
        )
    # Provenance == the hit spans (same shape as a search_vault span → ledger-identical).
    return ok_result(summary=summary, data={"matches": hits}, provenance=hits)
