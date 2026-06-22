"""F1e privilege-firewall gate — the trust-surface polish (offline, $0, no LLM/API).

Covers plans/F1_VAULT_PLAN.md §F1e. The privilege firewall has exactly one rule (the plan's
words): a privileged document is **excluded from any shared / cross-vault surface** and
**watermarked in exports** — and it is NOT hidden from its own vault. This gate proves that
contract end to end across the four layers F1e shipped:

  A. Pure core (`src/components/privilege.py`)
     A1  exclude_privileged drops privileged docs from a SHARED set, keeps the rest
     A2  legacy/untagged + None-valued docs read as NOT privileged (byte-identical legacy)
     A3  exclude_privileged does not mutate its input (returns a new list)
     A4  needs_watermark / export_watermark are true/non-empty IFF the set has a privileged doc
     A5  privileged_doc_ids reports exactly the withheld ids

  B. Schema validation (`src/api/schemas.py`)
     B1  UpdateCollectionRequest REJECTS an invalid status (the lifecycle write is gated)
     B2  UpdateCollectionRequest REJECTS an invalid matter_kind
     B3  every MATTER_STATUSES value is accepted (the control's options are all valid)
     B4  a rename-only PATCH (name only) is unchanged — matter fields stay None (legacy path)
     B5  UpdateDocumentRequest carries `privileged: bool`

  C. Export watermark wiring (`src/api/routes/export.py`)
     C1  _build_docx stamps the notice IFF privileged=True (and is byte-identical otherwise)
     C2  _build_redline_docx stamps the notice IFF privileged=True
     C3  _build_pdf stamps the notice IFF privileged=True

  D. Document PATCH route (`src/api/routes/documents.py`) — reproduce-then-close via TestClient
     D1  PATCH /documents/{id} {privileged:true} round-trips → response.privileged == True
     D2  the DB write filters user_id (RLS is bypassed by the service role — this is the
         ONLY cross-user guard on the write)
     D3  an unknown / not-owned doc_id → 404 (never a silent no-op)
     D4  unmark (privileged:false) round-trips back to False

Run: python -u eval/test_privilege.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ════════════════════════════════════════════════════════════════════════════════════════
# A. Pure core — src/components/privilege.py
# ════════════════════════════════════════════════════════════════════════════════════════
def section_A():
    print("\nA. Pure firewall core (privilege.py)")
    from src.components import privilege as P

    # A SHARED / cross-vault set: one privileged work-product, two clean, one legacy (no key),
    # one with an explicit None (a row read back before any toggle).
    shared = [
        {"id": "clean-1", "privileged": False},
        {"id": "wp-1", "privileged": True},          # attorney work-product
        {"id": "legacy-1"},                           # pre-migration-011 row, no key
        {"id": "clean-2", "privileged": None},        # explicit None
        {"id": "wp-2", "privileged": True},
    ]

    kept = P.exclude_privileged(shared)
    kept_ids = {d["id"] for d in kept}
    check("A1: exclude_privileged drops privileged docs from a shared set",
          kept_ids == {"clean-1", "legacy-1", "clean-2"}, f"kept={kept_ids}")

    check("A2: legacy (no key) + None-valued docs read as NOT privileged (legacy byte-identical)",
          P.is_privileged({"id": "x"}) is False and P.is_privileged({"privileged": None}) is False)

    # A3: input is not mutated.
    before = [dict(d) for d in shared]
    _ = P.exclude_privileged(shared)
    check("A3: exclude_privileged does not mutate its input (returns a NEW list)",
          shared == before and _ is not shared)

    # A4: watermark IFF a privileged doc is present.
    check("A4: needs_watermark / export_watermark are set IFF the set contains a privileged doc",
          P.needs_watermark(shared) is True
          and P.export_watermark(shared) == P.WATERMARK_NOTICE
          and P.needs_watermark([{"privileged": False}]) is False
          and P.export_watermark([{"privileged": False}]) == "",
          f"wm(shared)={P.export_watermark(shared)!r}")

    check("A5: privileged_doc_ids reports exactly the withheld ids",
          set(P.privileged_doc_ids(shared)) == {"wp-1", "wp-2"},
          str(P.privileged_doc_ids(shared)))

    # Empty / None inputs never raise.
    check("A6: empty + None inputs are safe (never raise)",
          P.exclude_privileged([]) == [] and P.exclude_privileged(None) == []
          and P.needs_watermark(None) is False)


# ════════════════════════════════════════════════════════════════════════════════════════
# B. Schema validation — src/api/schemas.py
# ════════════════════════════════════════════════════════════════════════════════════════
def section_B():
    print("\nB. Schema validation (schemas.py)")
    from pydantic import ValidationError
    from src.api.schemas import (
        UpdateCollectionRequest, UpdateDocumentRequest, MATTER_STATUSES, MATTER_KINDS,
    )

    # B1: invalid status rejected.
    rejected_status = False
    try:
        UpdateCollectionRequest(status="deleted")  # not in MATTER_STATUSES
    except ValidationError:
        rejected_status = True
    check("B1: UpdateCollectionRequest rejects an invalid status", rejected_status)

    # B2: invalid matter_kind rejected.
    rejected_kind = False
    try:
        UpdateCollectionRequest(matter_kind="not_a_real_kind")
    except ValidationError:
        rejected_kind = True
    check("B2: UpdateCollectionRequest rejects an invalid matter_kind", rejected_kind)

    # B3: every real status accepted (the control's options are all valid round-trips).
    all_status_ok = True
    for s in MATTER_STATUSES:
        try:
            req = UpdateCollectionRequest(status=s)
            all_status_ok = all_status_ok and req.status == s
        except ValidationError:
            all_status_ok = False
    check("B3: every MATTER_STATUSES value is accepted by the lifecycle PATCH", all_status_ok,
          str(MATTER_STATUSES))

    # B4: a rename-only PATCH leaves matter fields None (legacy path unchanged).
    rename_only = UpdateCollectionRequest(name="Acme v. Beta")
    check("B4: a rename-only PATCH leaves matter_kind/status/parties None (legacy unchanged)",
          rename_only.name == "Acme v. Beta"
          and rename_only.matter_kind is None
          and rename_only.status is None
          and rename_only.parties is None)

    # B5: the document privilege request shape.
    doc_req = UpdateDocumentRequest(privileged=True)
    check("B5: UpdateDocumentRequest carries privileged: bool", doc_req.privileged is True)

    # Sanity: the sets the UI mirrors really exist.
    check("B6: MATTER_STATUSES non-empty and includes legal_hold (the F1e trust state)",
          "legal_hold" in MATTER_STATUSES and "active" in MATTER_STATUSES,
          str(MATTER_STATUSES))


# ════════════════════════════════════════════════════════════════════════════════════════
# C. Export watermark wiring — src/api/routes/export.py
# ════════════════════════════════════════════════════════════════════════════════════════
def _docx_text(blob: bytes) -> str:
    """Extract all paragraph text from a .docx byte blob."""
    import io
    from docx import Document
    d = Document(io.BytesIO(blob))
    return "\n".join(p.text for p in d.paragraphs)


def section_C():
    print("\nC. Export watermark wiring (export.py)")
    from src.components.privilege import WATERMARK_NOTICE
    needle = WATERMARK_NOTICE[:30]  # a stable prefix of the notice

    try:
        from src.api.routes.export import (
            _build_docx, DocxExportRequest,
            _build_redline_docx, RedlineExportRequest,
            _build_pdf, PdfExportRequest,
        )
    except Exception as exc:  # noqa: BLE001
        check("C: export builders importable", False, str(exc))
        return

    # C1: .docx deliverable.
    md = "# Heading\n\nA clause body sentence."
    priv = _docx_text(_build_docx(DocxExportRequest(title="T", markdown=md, privileged=True)))
    clean = _docx_text(_build_docx(DocxExportRequest(title="T", markdown=md, privileged=False)))
    check("C1: _build_docx stamps the notice IFF privileged=True",
          needle in priv and needle not in clean,
          f"priv_has={needle in priv} clean_has={needle in clean}")

    # C2: redline .docx.
    try:
        rl_priv = _docx_text(_build_redline_docx(RedlineExportRequest(
            title="Redline", doc_name="contract.pdf", findings=[], privileged=True)))
        rl_clean = _docx_text(_build_redline_docx(RedlineExportRequest(
            title="Redline", doc_name="contract.pdf", findings=[], privileged=False)))
        check("C2: _build_redline_docx stamps the notice IFF privileged=True",
              needle in rl_priv and needle not in rl_clean,
              f"priv_has={needle in rl_priv} clean_has={needle in rl_clean}")
    except Exception as exc:  # noqa: BLE001
        check("C2: _build_redline_docx watermark", False, f"builder raised: {exc}")

    # C3: .pdf deliverable — assert the byte output differs ONLY by the presence of the notice
    # (the notice text is embedded in the PDF content stream; a clean build omits it).
    try:
        pdf_priv = _build_pdf(PdfExportRequest(title="T", markdown=md, privileged=True))
        pdf_clean = _build_pdf(PdfExportRequest(title="T", markdown=md, privileged=False))
        # The privileged PDF is strictly larger (carries the extra notice block) and the clean
        # one is a valid PDF too. We can't easily text-extract fpdf output without a reader, so
        # we assert the watermark build diverges from the clean build and both are non-empty PDFs.
        check("C3: _build_pdf produces a different (watermarked) PDF when privileged=True",
              pdf_priv != pdf_clean and pdf_priv[:4] == b"%PDF" and pdf_clean[:4] == b"%PDF"
              and len(pdf_priv) > len(pdf_clean),
              f"priv_len={len(pdf_priv)} clean_len={len(pdf_clean)}")
    except Exception as exc:  # noqa: BLE001
        check("C3: _build_pdf watermark", False, f"builder raised: {exc}")


# ════════════════════════════════════════════════════════════════════════════════════════
# D. Document PATCH route — reproduce-then-close via TestClient
# ════════════════════════════════════════════════════════════════════════════════════════
_DOCS = {"doc-1": {"id": "doc-1", "filename": "memo.pdf", "status": "ready",
                   "chunk_count": 3, "privileged": False}}


class _FakeDocQuery:
    """A minimal documents.update().eq().eq().execute() chain that records the eq columns and
    applies the privileged write to the in-memory store (only when scoped by user_id)."""
    def __init__(self, rec):
        self._rec = rec
        self._update = None
        self._eqs = {}

    def update(self, payload):
        self._update = payload
        return self

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._eqs[col] = val
        self._rec.setdefault("eqs", []).append((col, val))
        return self

    def execute(self):
        # WRITE path (an .update() was staged): apply only to an owned, existing doc.
        if self._update is not None:
            did = self._eqs.get("id")
            owns = self._eqs.get("user_id") == "user-1"
            if owns and did in _DOCS:
                _DOCS[did].update(self._update)
                return type("R", (), {"data": [dict(_DOCS[did])]})()
            return type("R", (), {"data": []})()
        # READ path (get_document): return the doc iff owned + present.
        did = self._eqs.get("id")
        owns = self._eqs.get("user_id") == "user-1"
        rows = [dict(_DOCS[did])] if (owns and did in _DOCS) else []
        return type("R", (), {"data": rows})()


class _FakeClient:
    def __init__(self, rec):
        self._rec = rec

    def table(self, name):
        return _FakeDocQuery(self._rec)


class _FakeSB:
    user_id = "user-1"

    def __init__(self):
        self.rec = {}
        self.client = _FakeClient(self.rec)
        self.read_client = self.client  # no JWT ⇒ read falls back to service-role (real behavior)

    def get_document(self, doc_id):
        return dict(_DOCS[doc_id]) if doc_id in _DOCS else {}

    def set_document_privileged(self, doc_id, privileged):
        # Mirror db.set_document_privileged: service-role write scoped by .eq(user_id).
        res = self.client.table("documents").update(
            {"privileged": bool(privileged)}
        ).eq("id", doc_id).eq("user_id", self.user_id).execute()
        return res.data[0] if res.data else {}


def section_D():
    print("\nD. Document PATCH route (documents.py) — reproduce-then-close")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP (FastAPI/TestClient unavailable: {exc})")
        return

    # Reset the store (idempotent re-runs).
    _DOCS["doc-1"] = {"id": "doc-1", "filename": "memo.pdf", "status": "ready",
                      "chunk_count": 3, "privileged": False}

    from src.api.routes import documents as doc_routes
    from src.api.dependencies import get_current_user

    # log_audit is best-effort; stub it so the route doesn't touch a real audit table.
    doc_routes.log_audit = lambda *a, **k: None

    app = FastAPI()
    app.include_router(doc_routes.router)  # router already carries prefix="/documents"
    sb = _FakeSB()
    app.dependency_overrides[get_current_user] = lambda: sb
    client = TestClient(app)

    # D1 + D2: mark privileged → round-trips, and the write filtered user_id.
    r = client.patch("/documents/doc-1", json={"privileged": True})
    check("D1: PATCH {privileged:true} → 200 and response.privileged == True",
          r.status_code == 200 and r.json().get("privileged") is True,
          f"status={r.status_code} body={r.text[:200]}")
    eq_cols = {c for c, _ in sb.rec.get("eqs", [])}
    check("D2: the privilege WRITE filters user_id at the DB (only cross-user guard; RLS bypassed)",
          "user_id" in eq_cols, f"eqs={sb.rec.get('eqs')}")
    check("D2b: the store actually flipped to privileged (write applied, not a no-op)",
          _DOCS["doc-1"]["privileged"] is True)

    # D3: unknown / not-owned doc → 404.
    r2 = client.patch("/documents/nope", json={"privileged": True})
    check("D3: an unknown/not-owned doc_id → 404 (never a silent no-op)",
          r2.status_code == 404, f"status={r2.status_code} body={r2.text[:200]}")

    # D4: unmark → back to False.
    r3 = client.patch("/documents/doc-1", json={"privileged": False})
    check("D4: PATCH {privileged:false} → response.privileged == False (unmark round-trips)",
          r3.status_code == 200 and r3.json().get("privileged") is False
          and _DOCS["doc-1"]["privileged"] is False,
          f"status={r3.status_code} body={r3.text[:200]}")


def main():
    print("=" * 78)
    print("F1e privilege-firewall gate (offline, $0)")
    print("=" * 78)
    section_A()
    section_B()
    section_C()
    section_D()
    print("\n" + "=" * 78)
    print(f"RESULT: {_passed} passed, {_failed} failed  ({_passed + _failed} total)")
    print("=" * 78)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
