"""The Knowledge Base relational store (G8 §G8.0).

The source-of-truth for provisions + sources, kept BESIDE the vectors so the §G8.4
completeness gate and the §G8.5 as-of gate are computable from a ground truth.

Two implementations behind one tiny interface:
  - `InMemoryKnowledgeStore` — pure-Python, zero deps. The offline gate
    (`eval/test_knowledge_infra.py`) and the §G8.1a Constitution proof both run on it
    with $0 / no network. It is also the natural test double for the tool gate.
  - `SupabaseKnowledgeStore` — the production store, writing the `knowledge_provisions`
    / `knowledge_sources` tables. Thin; mirrors the document-table pattern. It is
    constructed lazily so importing this module needs no Supabase client.

The store NEVER writes a user namespace and never touches the vector index — it is the
relational half only. Writing vectors is the ingest adapter's job (`ingest.py`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from .provision import Provision, SourceMeta


class KnowledgeStore(Protocol):
    """The relational store contract the ingest adapter + gates depend on."""

    def upsert_source(self, source: SourceMeta) -> None: ...
    def upsert_provisions(self, provisions: List[Provision]) -> None: ...
    def get_source(self, source_key: str) -> Optional[SourceMeta]: ...
    def list_sources(self) -> List[SourceMeta]: ...
    def list_provisions(self, source_key: Optional[str] = None) -> List[Provision]: ...


class InMemoryKnowledgeStore:
    """A dict-backed `KnowledgeStore`. Deterministic, dependency-free, $0.

    Upserts are keyed: a source by `source_key`, a provision by `chunk_id` (falling
    back to `citation` when a fixture omits the chunk id). Re-ingesting a source
    replaces its provisions cleanly.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, SourceMeta] = {}
        # source_key -> {provision_key -> Provision}
        self._provisions: Dict[str, Dict[str, Provision]] = {}

    @staticmethod
    def _pkey(p: Provision) -> str:
        return p.chunk_id or p.citation or f"{p.source_key}:{p.section_or_article_id}"

    def upsert_source(self, source: SourceMeta) -> None:
        self._sources[source.source_key] = source
        self._provisions.setdefault(source.source_key, {})

    def upsert_provisions(self, provisions: List[Provision]) -> None:
        for p in provisions:
            bucket = self._provisions.setdefault(p.source_key, {})
            bucket[self._pkey(p)] = p

    def get_source(self, source_key: str) -> Optional[SourceMeta]:
        return self._sources.get(source_key)

    def list_sources(self) -> List[SourceMeta]:
        return list(self._sources.values())

    def list_provisions(self, source_key: Optional[str] = None) -> List[Provision]:
        if source_key is not None:
            return list(self._provisions.get(source_key, {}).values())
        out: List[Provision] = []
        for bucket in self._provisions.values():
            out.extend(bucket.values())
        return out


class SupabaseKnowledgeStore:
    """Production `KnowledgeStore` over the `knowledge_provisions` / `knowledge_sources`
    tables. Mirrors the `documents` table access pattern (a SupabaseClient is injected).

    Constructed only when the live KB is read/written — the offline machine never needs
    it. Kept deliberately thin: the gates run on the relational ground truth, so the
    only intelligence is the read/write mapping.
    """

    PROVISIONS_TABLE = "knowledge_provisions"
    SOURCES_TABLE = "knowledge_sources"

    def __init__(self, supabase_client) -> None:
        # `supabase_client` is the project's SupabaseClient (same one routes touch).
        self._sb = supabase_client

    # The row<->dataclass mapping is intentionally explicit (no reflection) so a schema
    # drift is a loud KeyError at the boundary, not a silent wrong column.
    @staticmethod
    def _source_to_row(s: SourceMeta) -> dict:
        return {
            "source_key": s.source_key,
            "title": s.title,
            "instrument_type": s.instrument_type,
            "jurisdiction": s.jurisdiction,
            "citation_prefix": s.citation_prefix,
            "enacted_date": s.enacted_date,
            "as_of_date": s.as_of_date,
            "toc_ids": s.toc_ids,
            "source_url": s.source_url,
            "partial": s.partial,
            "metadata": s.metadata,
        }

    @staticmethod
    def _provision_to_row(p: Provision) -> dict:
        return {
            "source_key": p.source_key,
            "instrument_type": p.instrument_type,
            "title": p.title,
            "citation": p.citation,
            "section_or_article_id": p.section_or_article_id,
            "text": p.text,
            "jurisdiction": p.jurisdiction,
            "enacted_date": p.enacted_date,
            "as_of_date": p.as_of_date,
            "repealed": p.repealed,
            "superseded_by": p.superseded_by,
            "toc_path": p.toc_path,
            "chunk_id": p.chunk_id,
            "page": p.page,
            "metadata": p.metadata,
        }

    @staticmethod
    def _row_to_source(r: dict) -> SourceMeta:
        return SourceMeta(
            source_key=r["source_key"], title=r["title"],
            instrument_type=r["instrument_type"], jurisdiction=r.get("jurisdiction", "IN"),
            citation_prefix=r.get("citation_prefix", ""), enacted_date=r.get("enacted_date"),
            as_of_date=r.get("as_of_date"), toc_ids=r.get("toc_ids") or [],
            source_url=r.get("source_url"), partial=r.get("partial", False),
            metadata=r.get("metadata") or {},
        )

    @staticmethod
    def _row_to_provision(r: dict) -> Provision:
        return Provision(
            source_key=r["source_key"], instrument_type=r["instrument_type"],
            title=r.get("title", ""), citation=r["citation"],
            section_or_article_id=r["section_or_article_id"], text=r.get("text", ""),
            jurisdiction=r.get("jurisdiction", "IN"), enacted_date=r.get("enacted_date"),
            as_of_date=r.get("as_of_date"), repealed=r.get("repealed", False),
            superseded_by=r.get("superseded_by"), toc_path=r.get("toc_path"),
            chunk_id=r.get("chunk_id"), page=r.get("page"), metadata=r.get("metadata") or {},
        )

    def upsert_source(self, source: SourceMeta) -> None:
        self._sb.table(self.SOURCES_TABLE).upsert(
            self._source_to_row(source), on_conflict="source_key"
        ).execute()

    def upsert_provisions(self, provisions: List[Provision]) -> None:
        if not provisions:
            return
        rows = [self._provision_to_row(p) for p in provisions]
        self._sb.table(self.PROVISIONS_TABLE).upsert(rows, on_conflict="chunk_id").execute()

    def get_source(self, source_key: str) -> Optional[SourceMeta]:
        res = self._sb.table(self.SOURCES_TABLE).select("*").eq(
            "source_key", source_key
        ).limit(1).execute()
        rows = res.data or []
        return self._row_to_source(rows[0]) if rows else None

    def list_sources(self) -> List[SourceMeta]:
        res = self._sb.table(self.SOURCES_TABLE).select("*").execute()
        return [self._row_to_source(r) for r in (res.data or [])]

    def list_provisions(self, source_key: Optional[str] = None) -> List[Provision]:
        q = self._sb.table(self.PROVISIONS_TABLE).select("*")
        if source_key is not None:
            q = q.eq("source_key", source_key)
        res = q.execute()
        return [self._row_to_provision(r) for r in (res.data or [])]
