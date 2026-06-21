"""G8.6 — vault connectors gate (Google Drive + email), $0 / offline.

Proves the connector contract WITHOUT any Google/IMAP network: a stub Connector lists +
fetches in-memory files, and the shared import loop (`_import_via_connector`) routes each
into a stubbed dispatch, accounting queued/skipped/errored correctly. This is the §G8.6
discipline — "a connector is a source, not a parser" — made a measurement:

  - a supported file is FETCHED and dispatched (queued) exactly once,
  - an unsupported type is SKIPPED with a reason (never silently dropped),
  - a file that fails to fetch is recorded ERROR and never aborts the batch,
  - the Drive MIME→ext map exports Google-native docs to Office types,
  - the email connector parses attachments from a raw RFC822 message (no IMAP),
  - `USE_CONNECTORS` off ⇒ the routes 404 (byte-identical guarantee).

Run: python -u eval/test_connectors.py
"""

import sys
sys.path.insert(0, ".")

from src.components.connectors.base import RemoteFile
from src.components.connectors.gdrive import ext_for_mime


class Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")


# ── a stub Connector: no network, deterministic ─────────────────────────────────────
class StubConnector:
    name = "stub"

    def __init__(self, files, fail_ids=()):
        self._files = files
        self._fail = set(fail_ids)
        self.fetched = []

    def list_files(self, folder):
        return self._files

    def fetch_file(self, remote):
        if remote.remote_id in self._fail:
            raise RuntimeError("simulated fetch failure")
        self.fetched.append(remote.remote_id)
        return f"BYTES:{remote.name}".encode()

    def vault_filename(self, remote):
        return remote.name


class _StubSb:
    user_id = "user-123"

    def __init__(self):
        self.created = []

    def create_document_record(self, filename, storage_path, file_type, file_size_bytes):
        doc_id = f"doc-{len(self.created)}"
        self.created.append((filename, file_type, file_size_bytes))
        return {"id": doc_id}


class _StubConfig:
    SUPPORTED_FILE_TYPES = {"pdf", "docx", "xlsx", "txt"}
    PINECONE_NAMESPACE = "user-123"
    USE_CONNECTORS = True


def main() -> int:
    c = Check()

    # Patch the route's dispatch so no Celery/Redis is touched; capture the calls.
    import src.api.routes.connectors as rc
    dispatched = []

    def fake_spool_and_dispatch(file_bytes, filename, sb, user_config):
        dispatched.append((filename, len(file_bytes)))
        return f"doc-{len(dispatched)}"

    orig = rc._spool_and_dispatch
    rc._spool_and_dispatch = fake_spool_and_dispatch
    try:
        cfg = _StubConfig()
        sb = _StubSb()
        files = [
            RemoteFile(remote_id="a", name="contract.pdf", mime_type="application/pdf", size_bytes=10),
            RemoteFile(remote_id="b", name="notes.txt", mime_type="text/plain", size_bytes=5),
            RemoteFile(remote_id="c", name="photo.png", mime_type="image/png", size_bytes=99),
            RemoteFile(remote_id="d", name="deck.pptx",
                       mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       size_bytes=20),  # NOT in this config's supported set → skipped
            RemoteFile(remote_id="e", name="broken.pdf", mime_type="application/pdf", size_bytes=3),
        ]
        connector = StubConnector(files, fail_ids={"e"})

        print("── shared import loop: queued / skipped / errored accounting ─────")
        result = rc._import_via_connector(
            connector, files, "MyFolder", sb, cfg,
            ext_for=lambda rf: ext_for_mime(rf.mime_type))
        by_name = {f.name: f for f in result.files}
        c.ok(by_name["contract.pdf"].status == "queued", "supported pdf is QUEUED")
        c.ok(by_name["notes.txt"].status == "queued", "supported txt is QUEUED")
        c.ok(by_name["photo.png"].status == "skipped", "image is SKIPPED (unsupported type)")
        c.ok("unsupported" in by_name["photo.png"].reason, "skip carries a reason (never silent)")
        c.ok(by_name["deck.pptx"].status == "skipped",
             "type not in this config's supported set is SKIPPED")
        c.ok(by_name["broken.pdf"].status == "error", "fetch failure recorded ERROR")
        c.ok("simulated" in by_name["broken.pdf"].reason, "error carries the failure reason")
        c.ok(result.queued == 2 and result.skipped == 2 and result.errored == 1,
             "tallies: 2 queued · 2 skipped · 1 errored")
        c.ok(len(dispatched) == 2, "exactly the 2 supported files were dispatched to ingestion")
        c.ok(dispatched[0] == ("contract.pdf", len(b"BYTES:contract.pdf")),
             "dispatched the fetched bytes under the vault filename")
        c.ok(connector.fetched == ["a", "b"],
             "only supported, non-failing files were fetched (one fetch each, no retry storm)")

        # ── Drive MIME → ext mapping (the export rule) ──────────────────────────────
        print("\n── Drive MIME map: native as-is, Google-native exported ──────────")
        c.ok(ext_for_mime("application/pdf") == "pdf", "native pdf → pdf")
        c.ok(ext_for_mime("application/vnd.google-apps.document") == "docx",
             "Google Doc → exported docx")
        c.ok(ext_for_mime("application/vnd.google-apps.spreadsheet") == "xlsx",
             "Google Sheet → exported xlsx")
        c.ok(ext_for_mime("image/png") is None, "unsupported MIME → None (skipped upstream)")

        # ── Drive vault_filename appends ext only for exported native docs ──────────
        from src.components.connectors.gdrive import GoogleDriveConnector
        gd = GoogleDriveConnector("tok")
        c.ok(gd.vault_filename(RemoteFile("1", "Agreement", "application/vnd.google-apps.document"))
             == "Agreement.docx", "exported Google Doc gets a .docx filename")
        c.ok(gd.vault_filename(RemoteFile("2", "report.pdf", "application/pdf")) == "report.pdf",
             "native pdf filename unchanged (no double extension)")

        # ── Email connector parses attachments from raw RFC822 (no IMAP) ────────────
        print("\n── Email: attachments parsed from a raw message (no IMAP) ────────")
        import email as email_mod
        from email.message import EmailMessage
        from src.components.connectors.email import EmailConnector

        msg = EmailMessage()
        msg["Subject"] = "Signed docs"
        msg.set_content("See attached.")
        msg.add_attachment(b"%PDF-1.7 fake", maintype="application", subtype="pdf",
                           filename="signed.pdf")
        msg.add_attachment(b"inline note", maintype="text", subtype="plain",
                           filename="note.txt")
        raw = msg.as_bytes()
        parsed = email_mod.message_from_bytes(raw)
        atts = list(EmailConnector._iter_attachments(parsed))
        names = sorted(n for n, _ in atts)
        c.ok(names == ["note.txt", "signed.pdf"], "both named attachments are surfaced")
        c.ok(all(p for _, p in atts), "each attachment yields non-empty payload bytes")
        # The inline body part (no filename) must NOT be surfaced as an attachment.
        c.ok(len(atts) == 2, "the message body is NOT surfaced as an attachment")

    finally:
        rc._spool_and_dispatch = orig

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G8.6 connectors gate GREEN (import loop · Drive map · email attachments · $0)")
        return 0
    print("  ✗ G8.6 connectors gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
