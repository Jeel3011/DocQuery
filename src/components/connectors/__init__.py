"""Vault connectors (G8 §G8.6).

A connector is a SOURCE, not a parser: it pulls files from an external store (Google
Drive first, email second) and routes each one into the EXISTING ingestion pipeline, so
every imported file gets the same G1 `classify_document` + fidelity receipt + doc-type
chip + fidelity dot a manual upload gets. No new ingestion path — the connector hands the
worker a local file exactly as `documents.py::upload_document` does.

Architecturally DISTINCT from the Knowledge Base: connectors feed the USER's own vault
(per-user, isolated namespace); the KB (`src/components/knowledge/`) is the shared,
read-only authority corpus. The two never get conflated.

Behind `config.USE_CONNECTORS` (off ⇒ the routes 404 ⇒ byte-identical to pre-G8.6).
"""

from .base import Connector, ImportedFile, ImportResult, RemoteFile
from .gdrive import GoogleDriveConnector
from .email import EmailConnector

__all__ = [
    "Connector",
    "ImportedFile",
    "ImportResult",
    "RemoteFile",
    "GoogleDriveConnector",
    "EmailConnector",
]
