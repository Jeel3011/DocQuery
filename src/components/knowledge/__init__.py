"""The Indian Legal Knowledge Base (G8_KNOWLEDGE_BASE_PLAN §G8).

A SHARED, read-only corpus of legal authority the agent queries through the
`search_knowledge` tool when the user's own documents don't contain the law it
needs, then cites through the SAME cite-or-abstain gate as everything else.

This package is the G8 machine:
  - `provision.py`   — the relational source-of-truth schema (Provision / SourceMeta)
                       + the IngestReceipt (the KB twin of `extraction_fidelity`).
  - `store.py`       — the relational store interface + an in-memory implementation
                       (the offline gate uses it; a Supabase-backed one lands beside it).
  - `ingest.py`      — the ingest adapter: reuse G1 clause-aware chunking to turn an Act
                       into provision-granular vectors + relational rows + a receipt.

The KB is DATA + one tool. It touches nothing proven (loop, gates, kernel, finance
extraction). `USE_KNOWLEDGE` off ⇒ byte-identical to pre-G8.
"""

from .provision import (
    INSTRUMENT_TYPES,
    IngestReceipt,
    Provision,
    SourceMeta,
)

__all__ = [
    "Provision",
    "SourceMeta",
    "IngestReceipt",
    "INSTRUMENT_TYPES",
]
