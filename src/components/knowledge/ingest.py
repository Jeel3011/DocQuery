"""The Knowledge Base ingest adapter (G8 §G8.0).

Turns one legal source (an Act, the Constitution, a regulation) into:
  1. provision-granular ROWS in the relational store (the §G8.4/§G8.5 ground truth),
  2. provision-granular VECTORS in the `kb_in` namespace (retrieval), and
  3. an `IngestReceipt` (the operator-facing fidelity number).

It REUSES the G1 ingestion machinery rather than inventing a parser:
  - `_strip_boilerplate` drops page-chrome (URLs / page-no / timestamps) before chunking
    — the same noise that buried clause signal in the legal-extraction work (§2).
  - `chunk_legal_prose` is the clause-aware chunker (one chunk per numbered heading,
    heading kept with body) — a statute is exactly its target shape.

Two entry points:
  - `ingest_provisions(...)`  — the statute path: the source gives us structured
    provisions (Article id + text + ToC), so we ingest them directly at provision
    granularity. This is the cleanest, most measurable path and what §G8.1a uses.
  - `ingest_text_elements(...)` — the prose-document path: unstructured elements →
    `_strip_boilerplate` → `chunk_legal_prose` → provisions. Used when a source arrives
    as a PDF without a clean machine-readable structure.

The vector WRITE is injected as a `vector_writer` callable so the offline gate runs
with a stub (no Pinecone, $0). `make_kb_vector_writer(config)` builds the live one.

Nothing here raises out: a per-provision failure is logged and skipped; the receipt
reports what actually landed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.utils import _stable_id

from .provision import IngestReceipt, Provision, SourceMeta
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# A vector writer takes the provisions (each carrying text + vector_metadata) and
# persists them to the KB namespace. Returns the count actually embedded. The offline
# gate passes a stub that just counts; the live one embeds + upserts.
VectorWriter = Callable[[List[Provision]], int]


def _stamp_chunk_ids(source_key: str, provisions: List[Provision]) -> None:
    """Give every provision a stable chunk_id shared by its row and its vector, so the
    completeness gate can join them. Deterministic over (source_key, id, text)."""
    for i, p in enumerate(provisions):
        if not p.chunk_id:
            p.chunk_id = _stable_id(source_key, "kb_provision", i, p.text or p.citation)


def _build_receipt(
    source: SourceMeta,
    provisions: List[Provision],
    embedded: int,
    problems: List[str],
) -> IngestReceipt:
    """Compute the ingest receipt: provisions parsed/embedded, ToC coverage, gaps.

    Coverage is the relational diff §G8.4 makes a committed gate — here it is the
    log-only self-report (the same shape, at ingest). `toc_ids` is the source's OWN
    table of contents captured into `SourceMeta`; the diff is against the ingested
    provisions' `section_or_article_id`."""
    toc = list(source.toc_ids or [])
    ingested_ids = {p.section_or_article_id for p in provisions}
    missing = [i for i in toc if i not in ingested_ids]
    return IngestReceipt(
        source_key=source.source_key,
        title=source.title,
        provisions_parsed=len(provisions),
        provisions_embedded=embedded,
        toc_total=len(toc),
        toc_covered=len(toc) - len(missing),
        as_of_date=source.as_of_date,
        missing_ids=missing,
        problems=problems,
    )


def ingest_provisions(
    source: SourceMeta,
    provisions: List[Provision],
    store: KnowledgeStore,
    vector_writer: VectorWriter,
) -> IngestReceipt:
    """Ingest a structured source (the statute / Constitution path, §G8.1a).

    Inherits `as_of_date` / `jurisdiction` from the source onto any provision that
    omits them (a single snapshot horizon per source), stamps chunk ids, validates,
    writes rows + vectors, and returns the receipt. Read-only-shared by construction:
    only the KB namespace is touched via `vector_writer`.
    """
    problems: List[str] = []

    # Inherit source-level defaults onto provisions (one snapshot horizon per source).
    for p in provisions:
        if p.as_of_date is None:
            p.as_of_date = source.as_of_date
        if p.enacted_date is None and source.enacted_date is not None:
            p.enacted_date = source.enacted_date
        if not p.jurisdiction:
            p.jurisdiction = source.jurisdiction

    _stamp_chunk_ids(source.source_key, provisions)

    # Validate; drop the structurally-broken ones but RECORD them (never silent).
    valid: List[Provision] = []
    for p in provisions:
        issues = p.is_valid()
        if issues:
            problems.append(f"{p.citation or p.section_or_article_id}: {'; '.join(issues)}")
            logger.warning("[KB ingest] dropping provision %s: %s", p.citation, issues)
            continue
        valid.append(p)

    # Relational write first (the ground truth), then vectors.
    store.upsert_source(source)
    store.upsert_provisions(valid)

    embedded = 0
    try:
        embedded = vector_writer(valid)
    except Exception as exc:  # noqa: BLE001 — ingest never crashes the caller
        problems.append(f"vector write failed: {type(exc).__name__}: {exc}")
        logger.warning("[KB ingest] vector write failed for %s: %s", source.source_key, exc)

    receipt = _build_receipt(source, valid, embedded, problems)
    # Reflect partial coverage back onto the source row (mirror the finance self-report).
    if not receipt.complete and source.toc_ids:
        source.partial = True
        store.upsert_source(source)
    logger.info(receipt.render())
    return receipt


def ingest_text_elements(
    source: SourceMeta,
    text_elements: List[Any],
    store: KnowledgeStore,
    vector_writer: VectorWriter,
    *,
    max_chars: int = 3000,
) -> IngestReceipt:
    """Ingest a prose-document source (the PDF path) by reusing G1 clause-aware chunking.

    `_strip_boilerplate` removes page chrome, `chunk_legal_prose` splits at numbered
    clause/section boundaries — each resulting chunk becomes one `Provision`. The
    chunk's heading drives the citation/section id (best-effort; sources with clean
    structure should use `ingest_provisions`).
    """
    # Import here so this module needs no unstructured/ingestion deps unless used.
    from src.components.data_ingestion import _strip_boilerplate, chunk_legal_prose

    problems: List[str] = []
    try:
        kept, _stats = _strip_boilerplate(text_elements)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"boilerplate strip failed: {exc}")
        kept = text_elements
    try:
        raw_chunks = chunk_legal_prose(kept, max_chars)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"clause chunking failed: {exc}")
        raw_chunks = []

    provisions: List[Provision] = []
    for ch in raw_chunks:
        heading = (ch.get("heading") or "").strip()
        sec_id, citation = _heading_to_ids(heading, source)
        provisions.append(Provision(
            source_key=source.source_key,
            instrument_type=source.instrument_type,
            title=heading[:120] or sec_id,
            citation=citation,
            section_or_article_id=sec_id,
            text=ch.get("text") or "",
            page=ch.get("page"),
        ))

    return ingest_provisions(source, provisions, store, vector_writer)


def _heading_to_ids(heading: str, source: SourceMeta) -> tuple:
    """Best-effort (section_id, citation) from a clause heading like '14. Equality...'.
    Falls back to the raw heading when no leading number is present."""
    import re
    m = re.match(r"^\s*(\d+(?:\.\d+){0,3})", heading or "")
    sec_id = m.group(1) if m else (heading[:40] or "?")
    prefix = source.citation_prefix or source.title
    joiner = "Art." if source.instrument_type in ("article", "schedule") else "s."
    citation = f"{prefix} {joiner}{sec_id}" if m else f"{prefix}: {sec_id}"
    return sec_id, citation


# ── live vector writer (Pinecone, the `kb_in` namespace) ─────────────────────────

def make_kb_vector_writer(config: Any) -> VectorWriter:
    """Build the LIVE vector writer that embeds provisions into the KB namespace.

    Constructs a KB-scoped Config (a copy with `PINECONE_NAMESPACE = KNOWLEDGE_NAMESPACE`)
    so the write targets the shared read-only namespace and NEVER a user's. Reuses the
    proven `EmbeddingManager`/vector-store upsert path — no new write code. Imported
    lazily so the offline gate never pulls Pinecone in.
    """
    from copy import copy

    from langchain_core.documents import Document

    from src.components.embeddings import EmbeddingManager

    kb_config = copy(config)
    kb_config.PINECONE_NAMESPACE = getattr(config, "KNOWLEDGE_NAMESPACE", "kb_in")
    embedder = EmbeddingManager(kb_config)

    # Pinecone caps per-vector METADATA at 40 KB, and the vector store mirrors
    # `page_content` into the `text` metadata key for retrieval — so a huge provision (a
    # Schedule's Union/State lists run ~64 KB) blows the limit. The vector's job is
    # RETRIEVAL, not storage: the full quotable text lives in the relational row
    # (`knowledge_provisions.text`), which the agent reads by `chunk_id` after a hit. So we
    # embed the title + a leading slice — enough to be findable — capped well under 40 KB
    # to leave room for the addressing metadata. (A future refinement: split a long
    # Schedule into multiple sub-vectors; v1 keeps one row + one capped vector per
    # provision. Logged when a provision is truncated — never silent.)
    EMBED_CHAR_CAP = 20_000  # ~5–6k tokens; comfortably inside the 8k embed window + 40KB meta

    def _write(provisions: List[Provision]) -> int:
        if not provisions:
            return 0
        docs = []
        truncated = 0
        for p in provisions:
            text = (p.text or "").strip()
            if not text:
                continue
            if len(text) > EMBED_CHAR_CAP:
                # Lead with the title so the retrievable slice stays on-topic.
                head = f"{p.title}. " if p.title else ""
                text = (head + text)[:EMBED_CHAR_CAP]
                truncated += 1
            docs.append(Document(page_content=text, metadata={**p.vector_metadata()}))
        if not docs:
            return 0
        if truncated:
            logger.info("[KB ingest] embedded a truncated retrieval slice for %d long "
                        "provision(s); full text remains in the relational row.", truncated)
        embedder.create_vector_store(docs)
        return len(docs)

    return _write
