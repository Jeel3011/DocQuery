"""`read_section` tool — the harness `sed -n` (DOCUMENT_HARNESS §6.4).

Read ONE section (by heading) or a page range of a document, in full — the move you
make after list_documents/search_text point you to the right part of a LARGE document
(so you read the right part, not a truncated whole-dump).

Phase 1 (no re-ingest): headings are detected at RUNTIME from the clean chunk text
(`_outline`) — `heading` matches fuzzily, `page_range` is exact. Phase 1.5 will persist
`section_path` and make `heading` exact too; until then `page_range` is the precise lever.

Security (§8): vault isolation is in `_chunks.text_chunks_for_doc` — the doc must be in
this matter (`resolve_doc_id` → None ⇒ refuse). Provenance: one span per page in range.

Gate: `eval/test_doc_harness.py` (envelope + heading slice + page_range + leak).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._chunks import resolve_doc_id, text_chunks_for_doc
from ._envelope import error_result, ok_result, safe_tool
from ._outline import build_outline, slice_by_heading

SCHEMA: Dict[str, Any] = {
    "name": "read_section",
    "description": (
        "Read ONE section (by heading) or a page range of a document, in full. Use after "
        "list_documents/search_text point you to the right part of a large document. Give "
        "either `heading` (e.g. 'Limitation of Liability', 'Item 8') or `page_range` "
        "(e.g. '40-46'). page_range is exact; heading matches the document's detected "
        "section headings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "The document to read from."},
            "heading": {"type": "string", "description": "A section/heading to read."},
            "page_range": {"type": "string", "description": "Alternatively 'start-end', e.g. '40-46'."},
        },
        "required": ["doc_id"],
    },
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


def _page_num(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v is not None and str(v).strip() != "" else None
    except (ValueError, TypeError):
        return None


@safe_tool
def read_section(
    doc_id: str,
    *,
    heading: Optional[str] = None,
    page_range: Optional[str] = None,
    db_client: Any = None,
    filename_by_doc: Optional[Dict[str, str]] = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a section (heading or page range) of `doc_id`; return the §3.3 envelope.

    Exactly one of `heading` / `page_range` should be given; if neither, return the
    document's outline so the model can pick. Vault-isolated via `_chunks` (§8).
    """
    if not doc_id:
        return error_result("read_section requires a 'doc_id'.")
    if db_client is None:
        return error_result(
            "read_section requires a live db_client (internal routing bug, not a model error)."
        )
    resolved = resolve_doc_id(doc_id, filename_by_doc)
    if resolved is None:
        return error_result(
            f"document {doc_id!r} is not in this matter — cannot read it (no cross-vault read)."
        )

    chunks = text_chunks_for_doc(db_client, resolved, owner_id=owner_id)
    if not chunks:
        return error_result(
            f"document {doc_id!r} has no readable text (scanned image or empty ingest)."
        )

    fname = (filename_by_doc or {}).get(resolved, doc_id)

    # No selector → return the outline (so the model can choose a heading/range).
    if not heading and not page_range:
        return ok_result(
            summary=f"read_section {doc_id}: no heading/page_range given — returning outline",
            data={"doc_id": doc_id, "outline": build_outline(chunks)},
            provenance=[],
        )

    selected: List[Dict[str, Any]]
    how: str
    if page_range:
        pr = _parse_page_range(page_range)
        if pr is None:
            return error_result(f"read_section could not parse page_range {page_range!r} (use 'start-end').")
        lo, hi = pr
        selected = [c for c in chunks if (pn := _page_num(c.get("page"))) is not None and lo <= pn <= hi]
        how = f"pages {lo}-{hi}"
        if not selected:
            return abstain_or_outline(doc_id, chunks, f"no text on pages {lo}-{hi}")
    else:
        selected = slice_by_heading(chunks, heading or "")
        how = f"heading {heading!r}"
        if not selected:
            return abstain_or_outline(doc_id, chunks,
                                      f"heading {heading!r} not found among detected sections")

    # Build the section text + per-page provenance.
    blocks: List[str] = []
    provenance: List[Dict[str, Any]] = []
    last_page = object()
    for c in selected:
        pg = c.get("page")
        if pg != last_page:
            blocks.append(f"\n\n[p.{pg}]\n")
            provenance.append({"kind": "span", "doc": fname, "doc_id": resolved, "page": pg})
            last_page = pg
        blocks.append(c.get("content", ""))
    body = "".join(blocks).strip()

    summary = f"read_section {doc_id} ({how}): {len(selected)} chunk(s)"
    return ok_result(
        summary=summary,
        data={"doc_id": doc_id, "section_text": body, "matched": how},
        provenance=provenance,
    )


def abstain_or_outline(doc_id: str, chunks: List[Dict[str, Any]], why: str) -> Dict[str, Any]:
    """When the requested section isn't found, give the model the outline + the reason
    so it can pick a real heading/range (one-step self-heal), rather than a bare error."""
    return ok_result(
        summary=f"read_section {doc_id}: {why} — returning outline so you can pick a section",
        data={"doc_id": doc_id, "not_found": why, "outline": build_outline(chunks)},
        provenance=[],
    )
