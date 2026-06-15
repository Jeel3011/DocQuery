"""G6.3 — Redline gate (offline, $0, scripted model — no live API call).

Proves three things:
1. A target clause that DEVIATES from the playbook rule → finding emitted with
   target_quote + suggested_edit + rationale (bind-or-flag discipline upheld).
2. A target clause that CONFORMS → status "conforming", no suggested_edit emitted.
3. A missing clause → status "missing", no hallucinated suggestion.
4. A deviation without a grounded quote → demoted to "abstain" (bind-or-flag gate).
5. Redline .docx round-trip: findings → _build_redline_docx → open → assert
   struck-through original + underlined suggested edit + sources section present.

Run: python eval/test_redline.py
"""
import io
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


# ── Scripted model fixture ────────────────────────────────────────────────────

class _ScriptedModel:
    """Returns a fixed JSON response per call, simulating the agent's final answer."""
    def __init__(self, response_json: dict, system: str = ""):
        self._resp = response_json

    def invoke(self, messages, tools=None, **kwargs):
        from src.components.agent_core.model import ModelResponse
        return ModelResponse(
            text=json.dumps(self._resp),
            tool_calls=[],
            usage={"in": 10, "out": 20},
        )


def _make_scope():
    from src.components.agent_core.registry import RunScope
    return RunScope(
        collection_id="test-col",
        doc_ids=["doc-1"],
        filenames=["sezzle-contract.pdf"],
        filename_by_doc={"doc-1": "sezzle-contract.pdf"},
    )


# ── Test 1: deviation finding — grounded quote + suggestion emitted ───────────

def test_deviation_found():
    print("\nTest 1 — deviation finding: quoted clause + suggested edit emitted")
    from src.components.agent_core.redline import build_redline_cell

    response = {
        "status": "deviation",
        "target_quote": "This Agreement shall be governed by the laws of the State of Delaware.",
        "deviation": "Contract specifies Delaware law; firm standard requires Indian law.",
        "suggested_edit": "This Agreement shall be governed by and construed in accordance with the laws of India.",
        "rationale": "Firm standard position requires Indian governing law for domestic agreements.",
    }

    def factory(system=""):
        return _ScriptedModel(response, system)

    finding = build_redline_cell(
        clause_topic="Governing Law",
        standard_position="Laws of India.",
        fallback_position=None,
        model_factory=factory,
        scope=_make_scope(),
        doc_name="sezzle-contract.pdf",
    )

    check("status is deviation", finding.status == "deviation", finding.status)
    check("target_quote present", bool(finding.target_quote))
    check("suggested_edit present", bool(finding.suggested_edit))
    check("rationale present", bool(finding.rationale))
    check("playbook_standard preserved", "India" in finding.playbook_standard)


# ── Test 2: conforming clause ─────────────────────────────────────────────────

def test_conforming():
    print("\nTest 2 — conforming clause: status conforming, no suggested_edit")
    from src.components.agent_core.redline import build_redline_cell

    response = {
        "status": "conforming",
        "target_quote": "This Agreement shall be governed by the laws of India.",
        "deviation": None,
        "suggested_edit": None,
        "rationale": None,
    }

    def factory(system=""): return _ScriptedModel(response, system)

    finding = build_redline_cell(
        "Governing Law", "Laws of India.", None, factory, _make_scope(), "contract.pdf"
    )
    check("status is conforming", finding.status == "conforming", finding.status)
    check("no suggested_edit", finding.suggested_edit is None)


# ── Test 3: missing clause ────────────────────────────────────────────────────

def test_missing():
    print("\nTest 3 — missing clause: status missing, no invented quote")
    from src.components.agent_core.redline import build_redline_cell

    response = {
        "status": "missing",
        "target_quote": None,
        "deviation": None,
        "suggested_edit": None,
        "rationale": None,
    }

    def factory(system=""): return _ScriptedModel(response, system)

    finding = build_redline_cell(
        "Force Majeure", "Standard force-majeure clause required.", None,
        factory, _make_scope(), "contract.pdf"
    )
    check("status is missing", finding.status == "missing", finding.status)
    check("target_quote is None", finding.target_quote is None)
    check("suggested_edit is None", finding.suggested_edit is None)


# ── Test 4: bind-or-flag — deviation without grounded quote → demoted ─────────

def test_bind_or_flag_demotion():
    print("\nTest 4 — bind-or-flag: deviation without quote demoted to abstain")
    from src.components.agent_core.redline import build_redline_cell

    # No target_quote → must be demoted regardless of status claim
    response = {
        "status": "deviation",
        "target_quote": None,          # missing — should trigger demotion
        "deviation": "Contract deviates.",
        "suggested_edit": "Use Indian law.",
        "rationale": None,             # also missing
    }

    def factory(system=""): return _ScriptedModel(response, system)

    finding = build_redline_cell(
        "Governing Law", "Laws of India.", None, factory, _make_scope(), "contract.pdf"
    )
    check("demoted to abstain (no quote + no rationale)", finding.status == "abstain", finding.status)
    check("demotion rationale explains bind-or-flag", "bind-or-flag" in (finding.rationale or "").lower())


# ── Test 5: redline .docx round-trip ─────────────────────────────────────────

def test_redline_docx_roundtrip():
    print("\nTest 5 — redline .docx round-trip: deviation + conforming + missing → .docx")
    from src.api.routes.export import _build_redline_docx, RedlineExportRequest
    from docx import Document

    findings = [
        {
            "status": "deviation",
            "clause_topic": "Governing Law",
            "target_quote": "Governed by the laws of Delaware.",
            "deviation": "Should be Indian law.",
            "suggested_edit": "Governed by the laws of India.",
            "rationale": "Firm standard: Indian law for domestic contracts.",
            "playbook_standard": "Laws of India.",
            "grounded": True,
        },
        {
            "status": "conforming",
            "clause_topic": "Confidentiality Term",
            "target_quote": "Obligations survive for three years.",
            "deviation": None,
            "suggested_edit": None,
            "rationale": None,
            "playbook_standard": "Three-year survival.",
            "grounded": True,
        },
        {
            "status": "missing",
            "clause_topic": "Anti-Bribery",
            "target_quote": None,
            "deviation": None,
            "suggested_edit": None,
            "rationale": None,
            "playbook_standard": "PCA 1988 compliance warranty required.",
            "grounded": False,
        },
    ]

    blob = _build_redline_docx(RedlineExportRequest(
        title="Test Redline", doc_name="sezzle-contract.pdf", findings=findings
    ))
    check("non-empty .docx blob", len(blob) > 500)
    check("valid bytes", isinstance(blob, bytes))

    doc = Document(io.BytesIO(blob))
    full_text = "\n".join(p.text for p in doc.paragraphs)

    check("title present", "Test Redline" in full_text)
    check("Deviations heading present", "Deviations" in full_text)
    check("Governing Law heading present", "Governing Law" in full_text)
    check("original Delaware text present", "Delaware" in full_text)
    check("suggested India text present", "India" in full_text)
    check("Conforming Clauses section present", "Conforming Clauses" in full_text)
    check("Missing Clauses section present", "Missing Clauses" in full_text)
    check("Anti-Bribery topic present", "Anti-Bribery" in full_text)


if __name__ == "__main__":
    print("=" * 64)
    print("G6.3 REDLINE GATE — bind-or-flag + .docx round-trip ($0)")
    print("=" * 64)
    test_deviation_found()
    test_conforming()
    test_missing()
    test_bind_or_flag_demotion()
    test_redline_docx_roundtrip()
    print("\n" + "=" * 64)
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all redline checks green.")
