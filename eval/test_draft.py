"""G6.1 — the `draft` deliverable gate (offline, $0, no model burn).

A draft is the riskiest surface in the product: a long, confident memo is exactly
where a hallucinated number hides best, and the user is about to SEND it. The whole
moat is that a draft is JUST ANOTHER agent output — every factual sentence cites a
real span/cell or it is WITHHELD, caught by the SAME output gate as any answer
(`gates.run_output_gates`). And export is dumb plumbing: markdown → .docx PRESERVING
the citation contract, never re-gating, never adding a number.

This proves three things with a scripted draft + a real (hand-built) ledger — no LLM:
  1. A cited, traced draft PASSES the gate, markdown intact.
  2. A draft with an UNCITED figure gets REDACTED (withhold is visible), rest ships.
  3. Export round-trip: markdown → /export/docx builder → reopen the .docx →
     - with citations: headings + body + endnote "Sources" present;
     - without citations: markers absent but the BODY NUMBERS are identical.

Run: python eval/test_draft.py
"""
import io
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.agent_core.ledger import EvidenceLedger
from src.components.agent_core.gates import run_output_gates
from src.api.routes.export import _build_docx, DocxExportRequest

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


def _ledger_with(value, *, label, doc, page, period=""):
    """A ledger holding ONE traced cell — the figure a draft may legitimately cite."""
    led = EvidenceLedger()
    led.record("table", 1, [{
        "kind": "cell", "value": value, "label": label,
        "doc": doc, "page": page, "period": period,
        "display": f"{value:,.0f}", "raw": f"{value:,.0f}",
    }])
    return led


# ── Test 1: a cited, traced draft passes the gate untouched ─────────────────────

def test_cited_draft_passes():
    print("\nTest 1 — cited + traced draft PASSES the gate")
    led = _ledger_with(198270, label="Total revenue", doc="msft-2022", page=7, period="FY2022")
    draft = (
        "Microsoft reported total revenue of 198,270 for fiscal 2022 [msft-2022 p.7]. "
        "This figure anchors the engagement summary below."
    )
    outcome = run_output_gates(draft, led)
    check("gate passes a traced, cited figure", outcome.passed, str(outcome.failures))
    check("not abstained", not outcome.abstained)


# ── Test 2: an UNCITED figure is redacted; the rest of the draft ships ───────────

def test_uncited_figure_redacted():
    print("\nTest 2 — UNCITED figure is REDACTED (withhold visible), rest ships")
    led = _ledger_with(198270, label="Total revenue", doc="msft-2022", page=7, period="FY2022")
    draft = (
        "Microsoft reported total revenue of 198,270 for fiscal 2022 [msft-2022 p.7]. "
        "Net income reached 72,738 in fiscal 2022."  # NO citation — must be withheld
    )
    outcome = run_output_gates(draft, led)
    check("gate fails (does not ship as-is)", not outcome.passed)
    check("gate abstains/redacts", outcome.abstained)
    red = outcome.redacted_draft or ""
    # The uncited net-income claim must not survive verbatim; the cited revenue must.
    check("uncited figure 72,738 is removed", "72,738" not in red, red)
    check("cited figure 198,270 survives", "198,270" in red, red)
    check("a withhold/abstain note is visible", red.strip() != "" and red != draft)


# ── Test 3: export round-trip preserves the citation contract ────────────────────

def _docx_text(blob: bytes):
    """Return (full_text, has_sources_heading) by reopening the .docx."""
    from docx import Document
    doc = Document(io.BytesIO(blob))
    paras = [p.text for p in doc.paragraphs]
    full = "\n".join(paras)
    has_sources = any(p.strip().lower() == "sources" for p in paras)
    return full, has_sources


def test_export_roundtrip():
    print("\nTest 3 — markdown → .docx round-trip preserves the contract")
    md = (
        "# Engagement Summary\n\n"
        "Microsoft reported total revenue of 198,270 for fiscal 2022 [msft-2022 p.7].\n\n"
        "## Key Points\n\n"
        "- Revenue grew year over year [msft-2022 p.7].\n"
        "- The figure traces to the income statement.\n"
    )

    # With citations: body present + endnote "Sources" section + the marker text retained there.
    with_blob = _build_docx(DocxExportRequest(title="Memo", markdown=md, include_citations=True))
    wtext, wsources = _docx_text(with_blob)
    check("with: heading present", "Engagement Summary" in wtext)
    check("with: body number 198,270 present", "198,270" in wtext)
    check("with: Sources endnote section present", wsources)
    check("with: source marker text retained in endnotes", "msft-2022 p.7" in wtext)
    check("with: inline marker bracket NOT left in body", "[msft-2022 p.7]" not in wtext)

    # Without citations: same body numbers, no markers, no Sources section.
    wo_blob = _build_docx(DocxExportRequest(title="Memo", markdown=md, include_citations=False))
    wotext, wosources = _docx_text(wo_blob)
    check("without: body number 198,270 IDENTICAL", "198,270" in wotext)
    check("without: no citation marker text", "msft-2022 p.7" not in wotext)
    check("without: no Sources section", not wosources)

    # The export is valid .docx (non-trivial zip payload).
    check("with: valid non-empty .docx blob", len(with_blob) > 1000)
    check("without: valid non-empty .docx blob", len(wo_blob) > 1000)


if __name__ == "__main__":
    print("=" * 64)
    print("G6.1 DRAFT GATE — citation contract + export round-trip ($0)")
    print("=" * 64)
    test_cited_draft_passes()
    test_uncited_figure_redacted()
    test_export_roundtrip()
    print("\n" + "=" * 64)
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all draft-gate + export checks green.")
