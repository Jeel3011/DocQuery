"""The connector contract (G8 §G8.6).

A `Connector` knows how to LIST a remote folder and FETCH a file's bytes. It does NOT
ingest — that is the existing pipeline's job. The route layer (`api/routes/connectors.py`)
walks `list_files` → `fetch_file` → spool → dispatch `process_document_task`, which is the
same handoff `documents.py::upload_document` performs for a manual upload.

Keeping the contract this thin is the §G8.6 discipline: "a connector is a source, not a
parser." It also makes the connector trivially testable with a stub (the gate uses a fake
connector — no Google network, $0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class RemoteFile:
    """One file discovered in the remote source, before fetch."""
    remote_id: str          # the source's file id (Drive fileId, message id, …)
    name: str               # the filename (used as the vault filename)
    mime_type: str          # the source's MIME type (drives ext mapping + skip rules)
    size_bytes: int = 0     # best-effort; 0 when the source doesn't report it


@dataclass
class ImportedFile:
    """The outcome of importing ONE remote file into the vault."""
    name: str
    remote_id: str
    status: str             # "queued" | "skipped" | "error"
    doc_id: str = ""        # the vault document id, when queued
    reason: str = ""        # why skipped / error (never silent)


@dataclass
class ImportResult:
    """The outcome of a folder import — the operator-/user-facing receipt."""
    source: str             # connector name ("google_drive")
    folder: str             # the folder id/label imported
    files: List[ImportedFile] = field(default_factory=list)

    @property
    def queued(self) -> int:
        return sum(1 for f in self.files if f.status == "queued")

    @property
    def skipped(self) -> int:
        return sum(1 for f in self.files if f.status == "skipped")

    @property
    def errored(self) -> int:
        return sum(1 for f in self.files if f.status == "error")

    def summary(self) -> str:
        return (f"[connector:{self.source}] folder {self.folder}: "
                f"{self.queued} queued · {self.skipped} skipped · {self.errored} errored")


class Connector(Protocol):
    """List + fetch a remote source. Implementations never ingest."""

    name: str

    def list_files(self, folder: str) -> List[RemoteFile]:
        """Return the importable files directly under `folder` (non-recursive v1)."""
        ...

    def fetch_file(self, remote: RemoteFile) -> bytes:
        """Download one file's raw bytes (export Google-native docs to a portable type)."""
        ...
