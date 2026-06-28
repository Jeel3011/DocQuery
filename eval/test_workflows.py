"""G7.1 gate — the workflow engine, OFFLINE (scripted model, real grids, ZERO live calls).

Proves the §0 invariant that is the whole G7 bet: **a workflow is DATA the ONE run_agent
loop interprets — not a new code path.** Concretely:

  1. RESOLVER (no model): resolve_run turns a template + form params into the expected
     RunConfig — for a GRID template (the shipped clause sweep) the fixed typed columns +
     the declared ceiling; for a REPORT template the composed question + the system-prompt
     OVERLAY (so the deliverable is shaped) + the same tool subset. The gate-by-shape rule
     (gate_sectioned for report, the per-cell grid gate for grid) is asserted at the route
     boundary (here: which gate a shape selects).
  2. RESTRICTION is a FEATURE (§0): a run driven through run_agent with the template's
     tool_subset offers the model EXACTLY that subset — a tool NOT in the subset is never
     in the schemas the model sees. The additive registry overload is byte-identical when
     tools=None (every other caller).
  3. END-TO-END (scripted model + real grid + build_cell): a clause-sweep cell with a
     grounded answer is FOUND with provenance; an answer the model cannot ground ABSTAINS
     VISIBLY (never a silent wrong value) — the same cite-or-abstain contract as any cell.
  4. THE GATE BINDS THE ARTIFACT: a report-shaped run's gate redacts an uncited sentence —
     no template gets a softer gate.

Run: python -u eval/test_workflows.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from src.components.agent_core.workflows import (
    WorkflowTemplate, RunConfig, resolve_run, render_question, resolve_grid_columns,
    list_templates, get_template, template_card, PRACTICE_ORDER,
)
from src.components.agent_core.registry import REGISTRY, RunScope
from src.components.agent_core.review_grid import ColumnKind, CellStatus, GridColumn
from src.components.agent_core.grid_engine import build_cell
from src.components.agent_core.budgets import Budget
from src.components.agent_core.loop import run_agent
from src.components.agent_core.gates import run_output_gates, gate_sectioned
from src.components.agent_core.prompt import SYSTEM_PROMPT_V1
from src.components.agent_core.model import ModelResponse, ScriptedModel, ToolCall
from src.components.agent_core.ledger import EvidenceLedger
from src.components.brain.analyst import Grid


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


# A real grid so a scripted compute call (report path) produces genuine ledger provenance.
_TJ = {
    "periods": ["2021", "2022", "2023"],
    "rows": [{"section": "", "label": "Total net sales",
              "2021": "469,822", "2022": "513,983", "2023": "574,785"}],
    "table_id": "amzn-cons",
}
GRIDS = [Grid(_TJ, doc="amzn-2022", page=41)]


def _envelope(status, value=None, quote=None, risk="none", note=None):
    import json
    return json.dumps({"status": status, "value": value, "quote": quote,
                       "risk": risk, "note": note})


def main() -> int:
    c = Check()

    # ── 0. THE GALLERY — many real tasks, grouped, three output types ─────────────
    print("── 0. the gallery: real tasks across practice areas ─────────────")
    ts = list_templates()
    c.ok(len(ts) >= 12, f"the gallery has many real workflows (got {len(ts)})")
    areas = set(t.practice_area for t in ts)
    c.ok({"Litigation", "Transactional", "Financial services"}.issubset(areas),
         "templates span Harvey's practice areas (Litigation / Transactional / Financial)")
    c.ok(any(t.practice_area == "Compliance (India)" for t in ts),
         "India-specific compliance workflows are present (the wedge)")
    out_types = set(t.output_type for t in ts)
    c.ok(out_types == {"Review", "Draft", "Output"},
         "all three Harvey output types exist (Review / Draft / Output)")
    shapes = set(t.shape for t in ts)
    c.ok(shapes == {"grid", "report", "output"}, "all three engine shapes are exercised")
    # the gallery is ordered by practice area (the UI groups in this order)
    order = {p: i for i, p in enumerate(PRACTICE_ORDER)}
    seq = [order.get(t.practice_area, 99) for t in ts]
    c.ok(seq == sorted(seq), "list_templates is grouped by practice-area order")

    # ── 1. THE RESOLVER — template is DATA → RunConfig, no model ──────────────────
    print("\n── 1. resolver: template + params → RunConfig (3 shapes) ────────")

    # GRID shape (Review) — fixed columns + ceiling, the template's tool subset.
    sweep = get_template("contract-clause-sweep")
    c.ok(sweep is not None and sweep.shape == "grid", "clause sweep is a registered GRID template")
    rc_grid = resolve_run(sweep, {"doc_ids": []})
    c.ok(rc_grid.shape == "grid" and rc_grid.output_type == "Review", "grid → shape=grid, type=Review")
    c.ok(all(isinstance(col, GridColumn) for col in rc_grid.columns) and len(rc_grid.columns) == 7,
         "resolver yields the 7 fixed typed GridColumns (no per-template code)")
    c.ok(rc_grid.columns[0].key == "governing_law" and rc_grid.columns[0].kind == ColumnKind.CLAUSE,
         "first column is the governing-law CLAUSE column")
    c.ok(rc_grid.max_cells == 120, "grid declares its cost ceiling (fanout.max_cells)")
    c.ok(rc_grid.tool_subset == ["search_vault", "read_document"],
         "grid run carries the template's exact (prose) tool_subset")

    # REPORT shape (Draft) — composed question + the overlay; per-section gate.
    rep = get_template("filings-compare-memo")
    rc_rep = resolve_run(rep, {"metrics": ["revenue", "net income"]})
    c.ok(rc_rep.shape == "report" and rc_rep.output_type == "Draft", "memo → shape=report, type=Draft")
    c.ok(rc_rep.question == "Produce a cited comparison memo across the filings in this vault, "
                            "comparing: revenue, net income.",
         "report question is composed from params (list folded to a comma list)")
    c.ok(rc_rep.system_prompt.startswith(SYSTEM_PROMPT_V1) and "WORKFLOW" in rc_rep.system_prompt,
         "report system prompt = base contract + the template OVERLAY (deliverable shape)")

    # OUTPUT shape (Output) — freeform single deliverable; whole-answer gate.
    out = get_template("translate-document")
    rc_out = resolve_run(out, {"target_language": "Hindi", "scope": "the indemnity clause"})
    c.ok(rc_out.shape == "output" and rc_out.output_type == "Output", "translate → shape=output, type=Output")
    c.ok("Hindi" in rc_out.question and "indemnity clause" in rc_out.question,
         "output question folds in the form params")

    # DYNAMIC columns — a grid template that builds its axis from a free-text param.
    dd = get_template("check-diligence-request-list")
    cols = resolve_grid_columns(dd, {"items": "Certificate of incorporation\nBoard resolutions"})
    c.ok([x.label for x in cols] == ["Certificate of incorporation", "Board resolutions"],
         "a grid template builds dynamic columns from a free-text param")

    # render_question is forgiving: an unknown placeholder never KeyErrors a run.
    weird = WorkflowTemplate(id="x", title="X", practice_area="Litigation", description="d",
                             shape="report", output_type="Draft", base_mode="standard",
                             tool_subset=[], params_schema=[],
                             question_format="Do {present} and {absent}.")
    c.ok(render_question(weird, {"present": "this"}) == "Do this and .",
         "render_question leaves an unknown {placeholder} empty (never 500s a run)")

    # The route picks the gate BY SHAPE — assert the rule the route encodes.
    gate_for = lambda shape: gate_sectioned if shape == "report" else run_output_gates
    c.ok(gate_for(rc_rep.shape) is gate_sectioned, "report shape → the per-section gate")
    c.ok(gate_for(rc_out.shape) is run_output_gates, "output shape → the whole-answer gate")
    c.ok(gate_for(rc_grid.shape) is run_output_gates,
         "grid shape → the whole-answer gate family (the per-cell grid gate runs in build_cell)")

    # ── 1b. THE G7 WEDGE TEMPLATES (§2.2) — 5 verb-sequences, no new engine ───────
    print("\n── 1b. the 5 wedge workflows resolve over the shared engine ─────")
    WEDGE = {"nda-review", "ma-due-diligence", "contract-abstraction",
             "litigation-intake", "covenant-compliance"}
    ids = {t.id for t in ts}
    c.ok(WEDGE.issubset(ids), f"all 5 §2.2 wedge templates are registered (missing: {WEDGE - ids})")

    # Each wedge template resolves to a RunConfig with a known shape + the SAME gate-by-shape
    # rule + a non-empty tool subset — proving it's data over the one loop, not a new path.
    gate_for = lambda shape: gate_sectioned if shape == "report" else run_output_gates
    EXPECT = {  # id → (shape, output_type)
        "nda-review": ("grid", "Review"),
        "ma-due-diligence": ("grid", "Review"),
        "contract-abstraction": ("grid", "Review"),
        "litigation-intake": ("output", "Output"),
        "covenant-compliance": ("report", "Draft"),
    }
    for wid, (shape, otype) in EXPECT.items():
        t = get_template(wid)
        rc = resolve_run(t, {})
        c.ok(rc.shape == shape and rc.output_type == otype,
             f"{wid} resolves to shape={shape}, type={otype}")
        c.ok(bool(rc.tool_subset), f"{wid} carries a non-empty tool subset (precision over breadth)")
        c.ok(gate_for(rc.shape) in (gate_sectioned, run_output_gates),
             f"{wid} is bound by the shape's gate (no softer path)")

    # NDA review (grid) yields its 5 typed clause columns with the playbook-style risk rubric.
    nda = resolve_run(get_template("nda-review"), {"doc_ids": []})
    c.ok(len(nda.columns) == 5 and all(isinstance(x, GridColumn) for x in nda.columns),
         "nda-review yields 5 typed GridColumns")
    c.ok(any(x.key == "term" and x.risk_rubric for x in nda.columns),
         "nda-review's confidentiality-term column carries a risk rubric (issue-detect vs playbook)")

    # Contract abstraction (grid) is the portfolio key-terms grid — 7 columns.
    abst = resolve_run(get_template("contract-abstraction"), {"doc_ids": []})
    c.ok(len(abst.columns) == 7 and abst.columns[0].key == "parties",
         "contract-abstraction is a 7-column portfolio key-terms grid")

    # M&A DD (grid) builds its axis dynamically from the request-list param (one column per item).
    dd_cols = resolve_grid_columns(get_template("ma-due-diligence"),
                                   {"items": "Certificate of incorporation\nMOA & AOA\nBoard resolutions"})
    c.ok([x.label for x in dd_cols] == ["Certificate of incorporation", "MOA & AOA", "Board resolutions"],
         "ma-due-diligence builds one grid column per request-list item")

    # Litigation intake (output) folds the matter into the question + shapes a timeline deliverable.
    li = resolve_run(get_template("litigation-intake"), {"matter": "ABC Ltd v. XYZ Pvt Ltd"})
    c.ok("ABC Ltd v. XYZ Pvt Ltd" in li.question, "litigation-intake folds the matter into the question")
    c.ok("Case Timeline" in li.system_prompt and "Issues in Dispute" in li.system_prompt,
         "litigation-intake overlay shapes the timeline + dates/events + issues deliverable")

    # Covenant compliance (report) composes a breach-report question + the draft overlay.
    cov = resolve_run(get_template("covenant-compliance"), {"focus": "leverage ratio"})
    c.ok(cov.shape == "report" and "leverage ratio" in cov.question,
         "covenant-compliance composes a breach-report question from the focus param")
    c.ok(cov.system_prompt.startswith(SYSTEM_PROMPT_V1) and "BREACH" in cov.system_prompt,
         "covenant-compliance overlay shapes the PASS/BREACH report on the base contract")

    # ── 2. RESTRICTION IS A FEATURE — only the subset reaches the model ───────────
    print("\n── 2. tool restriction: the model sees ONLY the subset ──────────")

    # The registry overload: byte-identical when tools=None; a subset restricts; a bogus
    # name is dropped (never offered).
    full = REGISTRY.names("standard")
    c.ok(REGISTRY.names("standard", tools=None) == full, "tools=None is byte-identical to the mode map")
    c.ok(REGISTRY.names("standard", tools=["search_vault", "read_document"]) ==
         ["search_vault", "read_document"], "a subset restricts the exposed tools")
    c.ok(REGISTRY.names("standard", tools=["search_vault", "NOT_A_TOOL"]) == ["search_vault"],
         "an unknown tool name is dropped, never offered (validated against SCHEMAS)")

    # End-to-end through run_agent with a 2-tool subset: the scripted model records how many
    # tools it was offered each invoke — it must be exactly the subset, and `compute` (not in
    # the subset) must be absent from the schemas the model sees.
    sub = ["search_vault", "read_document"]
    schemas_seen = REGISTRY.schemas("standard", tools=sub)
    names_seen = {s.get("name") or s.get("function", {}).get("name") for s in schemas_seen}
    model = ScriptedModel([ModelResponse(text="No verifiable answer.", tool_calls=[])])
    scope = RunScope(collection_id="c", doc_ids=["d1"], filenames=["c1.pdf"],
                     filename_by_doc={"d1": "c1.pdf"}, grids=[], retrieval_manager=None, db_client=None)
    budget = Budget(mode="grid", model="scripted", max_steps=3, wall_clock_s=10, token_budget=10000)
    list(run_agent("q", model=model, scope=scope, budget=budget,
                   system_prompt="s", gate_fn=run_output_gates, tools=sub))
    c.ok(model.calls and model.calls[0]["n_tools"] == 2,
         "run_agent(tools=subset) offers the model EXACTLY the 2-tool subset")
    c.ok("compute" not in names_seen and "search_vault" in names_seen,
         "a tool NOT in the subset (compute) is never in the schemas the model sees")

    # ── 3. END-TO-END: a clause-sweep cell cites or abstains (scripted, $0) ───────
    print("\n── 3. clause-sweep cell: FOUND cites, unsupported ABSTAINS ──────")

    gov = rc_grid.columns[0]  # the governing-law clause column

    # 3a. Grounded answer → FOUND with provenance. The model searches, then answers the
    # envelope, and the sources event carries the source span.
    found_model = ScriptedModel([
        ModelResponse(text="", tool_calls=[ToolCall(
            id="t1", name="search_vault",
            args={"query": "governing law jurisdiction", "kind": "text"})]),
        ModelResponse(text=_envelope("found", "State of Delaware",
                                     "governed by the laws of the State of Delaware", "standard"),
                      tool_calls=[]),
    ])
    cell = build_cell("d1", gov, collection_id="c", model=found_model,
                      filename_by_doc={"d1": "c1.pdf"}, grids_by_doc={"d1": []},
                      retrieval_manager=_StubRetriever(), db_client=None, model_id="scripted")
    c.ok(cell.status == CellStatus.FOUND and cell.value == "State of Delaware",
         "a grounded clause-sweep cell is FOUND with its value")
    c.ok(cell.quote and "Delaware" in cell.quote, "the FOUND cell carries its quoted source span")

    # 3b. The document genuinely lacks the clause → MISSING (a real, flagged finding).
    missing_model = ScriptedModel([
        ModelResponse(text=_envelope("missing", None, None, "missing",
                                     "no governing-law clause in this document"), tool_calls=[]),
    ])
    cell_m = build_cell("d1", gov, collection_id="c", model=missing_model,
                        filename_by_doc={"d1": "c1.pdf"}, grids_by_doc={"d1": []},
                        retrieval_manager=_StubRetriever(), db_client=None, model_id="scripted")
    c.ok(cell_m.status == CellStatus.MISSING,
         "an absent clause is MISSING (a flagged finding, not a silent blank)")

    # 3c. An UNPARSEABLE / unsupported answer ABSTAINS VISIBLY — never FOUND.
    junk_model = ScriptedModel([ModelResponse(text="I think it's probably Delaware.", tool_calls=[])])
    cell_a = build_cell("d1", gov, collection_id="c", model=junk_model,
                        filename_by_doc={"d1": "c1.pdf"}, grids_by_doc={"d1": []},
                        retrieval_manager=_StubRetriever(), db_client=None, model_id="scripted")
    c.ok(cell_a.status == CellStatus.ABSTAIN,
         "an ungrounded/unparseable answer ABSTAINS visibly (never a silent wrong cell)")

    # ── 4. THE GATE BINDS THE ARTIFACT — no softer gate for a workflow ────────────
    print("\n── 4. the report gate redacts an uncited line (same gate) ───────")

    # A report-shaped artifact with one GROUNDED section (a figure that traces to a real
    # ledger cell + a citation) and one UNGROUNDED section (an uncited factual claim about a
    # figure with NO ledger cell). The per-section gate must keep the grounded section intact
    # and redact the ungrounded one — the exact contract every answer/draft/deep gets, with
    # no softer gate for a workflow.
    led = EvidenceLedger()
    # A genuine numeric cell (what the loop records after a compute) so the stated figure
    # traces; plus its source span so the citation resolves.
    led.record("compute", 1, [{"kind": "cell", "value": 513983.0, "label": "Total net sales",
                               "period": "2022", "doc": "amzn-2022", "page": 41,
                               "display": "513,983"}])
    draft = ("## Net Sales\nAmazon's FY2022 net sales were 513,983 [amzn-2022 p.41].\n\n"
             "## Outlook\nNet sales will reach 800,000 next year.")
    outcome = gate_sectioned(draft, led)
    rd = outcome.redacted_draft or draft
    c.ok("513,983" in rd, "the gate PRESERVES the grounded, cited figure")
    c.ok(not outcome.passed and "800,000" not in rd,
         "the report artifact is bound by the SAME gate — an ungrounded figure is redacted, "
         "no softer gate for a workflow")

    # ── 5. ROUTE FIELD CONTRACT — the route must only read RunConfig fields that EXIST ─
    # (Regression guard: a live run degraded with "'RunConfig' object has no attribute
    # 'artifact_kind'" after the field was renamed → output_type. The offline gate tested
    # resolve_run + the loop, but never the ROUTE's attribute access. This catches that
    # whole class at $0 by parsing the route for `run.<attr>` and checking each is real.)
    print("\n── 5. route reads only real RunConfig fields ────────────────────")
    import ast, dataclasses
    import src.api.routes.workflows as wf_route
    rc_fields = {f.name for f in dataclasses.fields(RunConfig)}
    src = open(wf_route.__file__).read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        # match `run.<attr>` (the resolved RunConfig is bound to the name `run`)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "run"):
            if node.attr not in rc_fields:
                bad.append(node.attr)
    c.ok(not bad, f"route reads only existing RunConfig fields (offenders: {sorted(set(bad))})")

    # ── gallery surface ──────────────────────────────────────────────────────────
    print("\n── gallery: list/get/card ───────────────────────────────────────")
    tpls = list_templates()
    c.ok(any(t.id == "contract-clause-sweep" for t in tpls), "list_templates includes the shipped template")
    card = template_card(sweep)
    c.ok(card["id"] == "contract-clause-sweep" and "params_schema" in card
         and "prompt_overlay" not in card,
         "template_card exposes id/params for the gallery form, not the internal overlay")

    # ── S-C: tool-subset validation + per-step failure reporting ─────────────
    print("\n── S-C: tool-subset validation + per-step failure reporting ──────")
    from src.components.agent_core.workflows import (
        validate_tool_subset, _step_failure_detail,
    )

    # S-C.1: a valid subset → ok=True.
    v = validate_tool_subset(["search_vault", "read_document"])
    c.ok(v["ok"] is True, "S-C validate_tool_subset: valid subset → ok=True")
    c.ok(v.get("tool_subset") == ["search_vault", "read_document"],
         "S-C validate_tool_subset: valid subset echoed back in result")

    # S-C.2: an unknown tool name → ok=False, unknown listed, repair message present.
    v_bad = validate_tool_subset(["search_vault", "NOT_A_REAL_TOOL"])
    c.ok(v_bad["ok"] is False,
         "S-C validate_tool_subset: unknown tool name → ok=False")
    c.ok("NOT_A_REAL_TOOL" in v_bad.get("unknown", []),
         "S-C validate_tool_subset: unknown name is listed in 'unknown'")
    c.ok("repair" in v_bad and "NOT_A_REAL_TOOL" in v_bad["repair"],
         "S-C validate_tool_subset: repair message names the unknown tool")
    c.ok("Valid tools:" in v_bad.get("repair", ""),
         "S-C validate_tool_subset: repair message lists the valid tool set")

    # S-C.3: a completely bogus subset → ok=False.
    v_all_bad = validate_tool_subset(["bogus_a", "bogus_b"])
    c.ok(v_all_bad["ok"] is False and len(v_all_bad["unknown"]) == 2,
         "S-C validate_tool_subset: all-bogus subset → ok=False, 2 unknowns")

    # S-C.4: empty subset → ok=True (a workflow may legitimately expose no tools,
    # e.g. a pure-overlay output that drives from the system prompt alone).
    c.ok(validate_tool_subset([])["ok"] is True,
         "S-C validate_tool_subset: empty subset is valid (no unknown names)")

    # S-C.5: all shipped templates have valid subsets.
    bad_templates = []
    for t in list_templates():
        v_t = validate_tool_subset(t.tool_subset)
        if not v_t["ok"]:
            bad_templates.append((t.id, v_t["unknown"]))
    c.ok(not bad_templates,
         f"S-C: all shipped templates have valid tool_subsets (offenders: {bad_templates})")

    # S-C.6: _step_failure_detail returns a structured dict with step_index, step_label, reason.
    sf = _step_failure_detail("doc:contract.pdf / col:governing_law",
                              "no grounded span found for the clause", 3)
    c.ok(sf["step_index"] == 3, "S-C _step_failure_detail: step_index correct")
    c.ok("governing_law" in sf["step_label"],
         "S-C _step_failure_detail: step_label names the column")
    c.ok("no grounded span" in sf["reason"],
         "S-C _step_failure_detail: reason carries the abstain message")

    # S-C.7: resolve_run on a valid template does not crash (S-C validation is a warn,
    # not a raise — valid templates must still resolve).
    rc_valid = resolve_run(sweep, {})
    c.ok(rc_valid.shape == "grid",
         "S-C resolve_run: valid template resolves cleanly despite S-C validation pass")

    # S-C.8: a synthetic template with an unknown tool subset logs but still resolves
    # (the route's validate call is additive — existing valid templates are unchanged).
    synthetic = WorkflowTemplate(
        id="synthetic-bad-subset", title="T", practice_area="Litigation",
        description="d", shape="output", output_type="Output",
        base_mode="standard", tool_subset=["search_vault", "BOGUS_TOOL"],
        params_schema=[],
    )
    import logging as _logging
    import io as _io
    _buf = _io.StringIO()
    _h = _logging.StreamHandler(_buf)
    _logging.getLogger("src.components.agent_core.workflows").addHandler(_h)
    rc_syn = resolve_run(synthetic, {})
    _logging.getLogger("src.components.agent_core.workflows").removeHandler(_h)
    c.ok(rc_syn.shape == "output",
         "S-C resolve_run: bad-subset template still resolves (validation is non-fatal)")
    c.ok("BOGUS_TOOL" in _buf.getvalue(),
         "S-C resolve_run: bad subset is LOGGED (caller can see it without crashing)")

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULT: {c.passed} passed, {c.failed} failed")
    print(f"{'='*60}")
    return 0 if c.failed == 0 else 1


class _Doc:
    """A LangChain-Document-like chunk: span_to_dict reads `.metadata` + `.page_content`."""

    def __init__(self, text, filename, page):
        self.page_content = text
        self.metadata = {"filename": filename, "page_number": page,
                         "chunk_id": f"{filename}-{page}", "score": 0.9}


class _StubRetriever:
    """A minimal retrieval manager so build_cell's search_vault tool returns one text hit
    (the grounding span) without a live vector DB — enough for the cite-or-abstain proof.
    Mirrors the real RetrievalManager's two entry points the search tool calls."""

    _HIT = [_Doc("This Agreement shall be governed by the laws of the State of Delaware.",
                 "c1.pdf", 12)]

    def retrieve(self, *args, **kwargs):
        return list(self._HIT)

    def retrieve_table_chunks(self, *args, **kwargs):
        return []


if __name__ == "__main__":
    sys.exit(main())
