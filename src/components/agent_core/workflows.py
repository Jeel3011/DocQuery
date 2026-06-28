"""Workflow templates + resolver (GRAND_PLAN §G7 · G7_WORKFLOWS_PLAN §3).

A workflow is **declarative DATA** (a `WorkflowTemplate` row), NOT code. The ONE
`run_agent` loop interprets it. The only things a template changes are:
  1. which tools are exposed (`tool_subset` → registry),
  2. the prompt overlay (the deliverable shape + steps),
  3. the form params, folded into the question,
  4. the output SHAPE the answer is rendered/gated as.
Everything else — retrieval, the kernel, the gates, tracing — is shared, untouched, and
already proven. **A workflow that needs a new loop branch is a DESIGN BUG, not a feature.**

THREE output shapes (matching Harvey's Review / Draft / Output tags):
  - **report** (Draft)  — a cited multi-section memo/document. One run_agent →
                          gate_sectioned → the ArtifactPanel (exportable via G6).
  - **output** (Output) — a freeform single deliverable (a summary, a transcript, a
                          translation, a timeline). One run_agent → the whole-answer gate.
  - **grid**   (Review) — a per-document/per-clause extraction TABLE. Fan-out → the
                          per-cell grid gate → the grid surface (review-grid engine).

A new Harvey "task" = a new `WorkflowTemplate` row + an overlay string + a fixture (hours),
not a new system. That is the §1b "250 tasks = overlays" bet made concrete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .prompt import SYSTEM_PROMPT_V1
from .review_grid import ColumnKind, GridColumn


# ──────────────────────────────────────────────────────────────────────────────
# The template — declarative data.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkflowTemplate:
    id: str                          # "extract-terms-supply-agreements" — the route key
    title: str                       # gallery card header
    practice_area: str               # "Litigation" | "Transactional" | "Financial services" | "Compliance (India)"
    description: str                 # one line for the gallery card
    shape: str                       # "report" | "output" | "grid"
    output_type: str                 # the Harvey tag: "Review" | "Draft" | "Output"
    base_mode: str                   # which existing mode's budget/model tier to reuse
    tool_subset: List[str]           # the EXACT tools this workflow may use (→ registry)
    params_schema: List[Dict[str, Any]]  # [{name,label,type,required,help}] → renders the form
    step_count: int = 1              # shown on the card (Harvey shows "· N steps")

    # report / output shape: the overlay + how params → question.
    prompt_overlay: str = ""
    question_format: str = ""        # str.format_map(params) → the composed question

    # grid shape: the fixed columns the fan-out runs, and the declared ceiling.
    columns: List[Dict[str, Any]] = field(default_factory=list)
    fanout: Optional[Dict[str, Any]] = None   # {"max_cells": int}


@dataclass
class RunConfig:
    """What a `/workflows/{id}/run` request resolves to. The route reads these and drives
    the SAME engine — no per-template code path."""
    shape: str                       # "report" | "output" | "grid"
    output_type: str
    base_mode: str
    tool_subset: List[str]

    # report / output shape
    question: str = ""
    system_prompt: str = ""

    # grid shape
    columns: List[GridColumn] = field(default_factory=list)
    max_cells: int = 120


def render_question(template: WorkflowTemplate, params: Dict[str, Any]) -> str:
    """Compose the run's question from `question_format` + the form params (the same trick
    agent_core uses for draft mode). Forgiving: an unknown {placeholder} renders empty, never
    KeyErrors a run; a list param folds to a comma list."""
    fmt = template.question_format or template.description
    if not fmt:
        return template.title

    class _Forgiving(dict):
        def __missing__(self, key: str) -> str:  # noqa: D401 — leave {unknown} intact-as-empty
            return ""

    flat: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        flat[k] = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else (v if v is not None else "")
    try:
        return fmt.format_map(_Forgiving(flat)).strip()
    except Exception:  # noqa: BLE001 — a bad format string must never break a run
        return fmt


def _to_columns(specs: List[Dict[str, Any]]) -> List[GridColumn]:
    out: List[GridColumn] = []
    for c in specs or []:
        try:
            kind = ColumnKind(c.get("kind", "clause"))
        except ValueError:
            kind = ColumnKind.CLAUSE
        out.append(GridColumn(key=c["key"], label=c["label"], prompt=c["prompt"],
                              kind=kind, risk_rubric=c.get("risk_rubric")))
    return out


def resolve_run(template: WorkflowTemplate, params: Dict[str, Any]) -> RunConfig:
    """Turn a template + form params into a RunConfig. NO per-template branch: it branches
    on `shape` (a property of the DATA). Adding a template never touches this function.

    S-C: validates the tool_subset against the known tool registry at resolve time so an
    unknown name is flagged immediately (not silently dropped at run time).  The result
    carries a `subset_validation` key; callers that need a hard stop check it.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    # S-C: validate the declared tool subset before building the RunConfig.
    validation = validate_tool_subset(list(template.tool_subset))
    if not validation["ok"]:
        _log.warning(
            "[workflows] template %r tool_subset validation failed: %s",
            template.id, validation.get("repair"),
        )

    if template.shape == "grid":
        return RunConfig(
            shape="grid", output_type=template.output_type, base_mode=template.base_mode,
            tool_subset=list(template.tool_subset),
            columns=_to_columns(template.columns),
            max_cells=int((template.fanout or {}).get("max_cells", 120)),
        )

    # report / output — a single run, deliverable shaped by the overlay.
    return RunConfig(
        shape=template.shape, output_type=template.output_type, base_mode=template.base_mode,
        tool_subset=list(template.tool_subset),
        question=render_question(template, params),
        system_prompt=SYSTEM_PROMPT_V1 + (template.prompt_overlay or ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared tool subsets. A restricted subset is a FEATURE (§0): the workflow spends its
# budget on precision, not breadth.
# ──────────────────────────────────────────────────────────────────────────────
_ANALYST = ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"]
_PROSE = ["search_vault", "read_document"]          # clause/transcript prose work
_NUMERIC = ["search_vault", "read_document", "list_metrics", "table_lookup", "compute"]


# ──────────────────────────────────────────────────────────────────────────────
# Reusable overlay fragments — every overlay INHERITS the base cite-or-abstain contract;
# it only adds the deliverable SHAPE + the steps. Never relaxes the gate.
# ──────────────────────────────────────────────────────────────────────────────
def _draft_overlay(body: str) -> str:
    return (
        "\n\n────────────────────────────────────────────────────────────────────────────\n"
        "WORKFLOW — produce a CLIENT-READY DELIVERABLE (not a chat answer). Everything above "
        "still holds: every factual sentence carries a [doc p.N] citation or is OMITTED; never "
        "invent a figure, date, party, or clause. Use `##` section headings (the report is "
        "gated and split section-by-section). If a section has no verifiable support, write a "
        "single line `_Insufficient evidence in the vault to report on this._`.\n\n" + body
    )


def _output_overlay(body: str) -> str:
    return (
        "\n\n────────────────────────────────────────────────────────────────────────────\n"
        "WORKFLOW — produce the requested OUTPUT as your single final message (clean markdown, "
        "no preamble). Everything above still holds: every fact cites [doc p.N] or is omitted; "
        "no invented content. " + body
    )


# ══════════════════════════════════════════════════════════════════════════════
# THE GALLERY — real, distinct legal/finance tasks (Harvey's surface), as DATA.
# Grouped by practice area. Each = an overlay + a tool subset + a form + an output shape.
# ══════════════════════════════════════════════════════════════════════════════

_T: List[WorkflowTemplate] = [

    # ── Litigation ──────────────────────────────────────────────────────────────
    WorkflowTemplate(
        id="summarize-discovery-responses",
        title="Summarize discovery responses and objections",
        practice_area="Litigation", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=5,
        description="Upload discovery responses, and DocQuery generates a summary table of each "
                    "request, the response, and any objections raised.",
        params_schema=[{"name": "doc_ids", "label": "Discovery documents", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "request", "label": "Request", "kind": "clause",
             "prompt": "Identify the discovery request this response addresses. Quote its text."},
            {"key": "response", "label": "Response", "kind": "clause",
             "prompt": "Summarize the substantive response. Quote the key sentence."},
            {"key": "objections", "label": "Objections", "kind": "clause",
             "prompt": "List any objections raised (privilege, relevance, overbreadth, burden). Quote them, or MISSING."},
        ],
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="analyze-transcript-key-topics",
        title="Analyze a transcript for key topics",
        practice_area="Litigation", output_type="Output", shape="output", base_mode="deep",
        tool_subset=_PROSE, step_count=5,
        description="Upload a court or deposition transcript, and DocQuery produces a summary of "
                    "the key topics discussed, with cited line references.",
        params_schema=[{"name": "focus", "label": "Topics to focus on (optional)", "type": "text",
                        "required": False, "help": "e.g. liability admissions, damages, timeline of events"}],
        question_format="Analyze the transcript(s) in this vault and summarize the key topics "
                        "discussed. {focus}",
        prompt_overlay=_output_overlay(
            "Read the transcript through `search_vault`/`read_document`. Produce, as markdown: a "
            "short overview, then a `## Key Topics` list where each topic has 2-4 cited bullet "
            "points quoting the relevant testimony with its [doc p.N] reference. Note any "
            "admissions or contradictions explicitly. Do NOT summarize a topic you cannot cite."),
    ),
    WorkflowTemplate(
        id="extract-email-metadata",
        title="Extract metadata and contents of emails",
        practice_area="Litigation", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=3,
        description="Generate a table of key email details — date, sender, recipient, subject, and "
                    "a one-line summary — across an email set.",
        params_schema=[{"name": "doc_ids", "label": "Email documents", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "date", "label": "Date", "kind": "clause", "prompt": "Extract the email date. Quote it."},
            {"key": "sender", "label": "From", "kind": "clause", "prompt": "Extract the sender. Quote the From line."},
            {"key": "recipient", "label": "To", "kind": "clause", "prompt": "Extract the recipient(s). Quote the To line."},
            {"key": "subject", "label": "Subject", "kind": "clause", "prompt": "Extract the subject line. Quote it."},
            {"key": "summary", "label": "Summary", "kind": "clause", "prompt": "One-sentence summary of the email body, grounded in its text."},
        ],
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="draft-memo-from-research",
        title="Draft memo from legal research",
        practice_area="Litigation", output_type="Draft", shape="report", base_mode="deep",
        tool_subset=_PROSE, step_count=2,
        description="Provide a legal research question and the relevant sources in the vault, and "
                    "DocQuery drafts a detailed, cited memorandum.",
        params_schema=[{"name": "question", "label": "Research question", "type": "textarea",
                        "required": True, "help": "The legal question the memo must answer."}],
        question_format="Draft a legal research memorandum answering: {question}",
        prompt_overlay=_draft_overlay(
            "Structure: `## Question Presented`, `## Short Answer`, `## Analysis` (with cited "
            "sub-points grounded in the vault sources), `## Conclusion`. Every proposition cites "
            "its source span [doc p.N]. Do not assert a holding you cannot ground in the vault."),
    ),

    # G7 wedge §2.2 — Litigation intake → case timeline + dates/events + issues (→ F6 Layer A).
    WorkflowTemplate(
        id="litigation-intake",
        title="Litigation intake — timeline, dates & events, issues",
        practice_area="Litigation", output_type="Output", shape="output", base_mode="deep",
        tool_subset=_PROSE, step_count=4,
        description="From the pleadings and filings in this vault, build a chronological case "
                    "timeline, a list of dates & events, and the issues in dispute — each tied to "
                    "the document that establishes it.",
        params_schema=[{"name": "matter", "label": "Matter / parties (optional)", "type": "text",
                        "required": False, "help": "e.g. ABC Ltd v. XYZ Pvt Ltd — helps frame the parties."}],
        question_format="Conduct litigation intake over the pleadings and filings in this vault "
                        "for the matter {matter}: build the case timeline, the list of dates & "
                        "events, and the issues in dispute.",
        prompt_overlay=_output_overlay(
            "Produce three markdown sections. `## Case Timeline` — a chronological ordered list, "
            "each entry: the date, the event/filing, and the [doc p.N] citation that establishes "
            "it (sort by date; an undated event goes last, marked _undated_). `## List of Dates & "
            "Events` — the same facts as a compact table-style bullet list (Date — Event — Source). "
            "`## Issues in Dispute` — the legal issues joined between the parties, each a cited "
            "bullet quoting the plaint/written-statement text that raises it. Never assert a date, "
            "event, or issue you cannot cite to a filing; omit anything ungrounded."),
    ),

    # ── Transactional ────────────────────────────────────────────────────────────
    WorkflowTemplate(
        id="extract-terms-supply-agreements",
        title="Extract terms from supply agreements",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=2,
        description="Upload supply agreements, and DocQuery generates a table of key terms from "
                    "each — parties, term, pricing, termination, liability, and governing law.",
        params_schema=[{"name": "doc_ids", "label": "Supply agreements", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "parties", "label": "Parties", "kind": "clause", "prompt": "Identify the buyer and supplier. Quote the parties clause."},
            {"key": "term", "label": "Term", "kind": "clause", "prompt": "Find the term/duration clause. Quote the period."},
            {"key": "pricing", "label": "Pricing", "kind": "clause", "prompt": "Find the price / payment terms clause. Quote it."},
            {"key": "termination", "label": "Termination", "kind": "clause", "prompt": "Find the termination clause and notice period. Quote it."},
            {"key": "liability", "label": "Liability Cap", "kind": "clause",
             "prompt": "Find the limitation-of-liability clause. Quote the cap.",
             "risk_rubric": "standard if mutual & capped; non_standard if uncapped or one-sided; missing if absent"},
            {"key": "governing_law", "label": "Governing Law", "kind": "clause", "prompt": "Find the governing-law clause. Quote it."},
        ],
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="contract-clause-sweep",
        title="Contract clause sweep",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=7,
        description="Sweep the standard clauses (governing law, term, termination, indemnity, "
                    "liability cap, confidentiality, assignment) across every contract — each cell "
                    "quoted to its source or flagged.",
        params_schema=[{"name": "doc_ids", "label": "Contracts", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to sweep every document in the vault."}],
        columns=[
            {"key": "governing_law", "label": "Governing Law", "kind": "clause",
             "prompt": "Find the governing-law (and any seat/jurisdiction) clause. Quote it exactly.",
             "risk_rubric": "standard if a major Indian state, England, NY or Delaware; else non_standard"},
            {"key": "term", "label": "Term", "kind": "clause", "prompt": "Find the term/duration clause. Quote the period exactly."},
            {"key": "termination", "label": "Termination", "kind": "clause", "prompt": "Find the termination clause + notice periods. Quote it."},
            {"key": "indemnity", "label": "Indemnity", "kind": "clause", "prompt": "Find the indemnification clause. Quote scope and any cap."},
            {"key": "liability_cap", "label": "Liability Cap", "kind": "clause",
             "prompt": "Find the limitation-of-liability/cap clause. Quote the cap.",
             "risk_rubric": "standard if capped & mutual; non_standard if uncapped/one-sided; missing if absent"},
            {"key": "confidentiality", "label": "Confidentiality", "kind": "clause", "prompt": "Find the confidentiality clause + survival period. Quote it."},
            {"key": "assignment", "label": "Assignment / CoC", "kind": "clause", "prompt": "Find the assignment / change-of-control clause. Quote it."},
        ],
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="check-diligence-request-list",
        title="Check a diligence request list",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=4,
        description="Provide diligence items, and DocQuery determines whether each is satisfied by "
                    "the data-room documents — present, partial, or missing, with a citation.",
        params_schema=[{"name": "doc_ids", "label": "Data-room documents", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to search the whole vault."},
                       {"name": "items", "label": "Diligence items (one per line)", "type": "textarea",
                        "required": True, "help": "e.g. Certificate of incorporation\\nBoard resolutions\\nMaterial contracts"}],
        columns=[],  # NOTE: dynamic — built from `items` at resolve time (see _dynamic_columns)
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="post-closing-timeline",
        title="Generate a post-closing timeline",
        practice_area="Transactional", output_type="Output", shape="output", base_mode="deep",
        tool_subset=_PROSE, step_count=3,
        description="Upload an agreement, and DocQuery generates a chronological timeline of "
                    "post-closing obligations, each tied to its clause.",
        params_schema=[],
        question_format="Build a chronological timeline of all post-closing obligations in the "
                        "agreement(s) in this vault.",
        prompt_overlay=_output_overlay(
            "Produce a markdown `## Post-Closing Timeline` as an ordered list. Each entry: the "
            "deadline/trigger, the obligation, the responsible party, and the [doc p.N] citation "
            "to the clause that creates it. Sort by time where dates are given. Omit any obligation "
            "you cannot ground in a clause."),
    ),

    # G7 wedge §2.2 — NDA review → intake → extract → issue-detect vs playbook → memo.
    WorkflowTemplate(
        id="nda-review",
        title="NDA review against the playbook",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=5,
        description="Review an NDA clause-by-clause against the firm's standard positions — for "
                    "each topic, the clause as drafted, whether it conforms or deviates, and the "
                    "issue flagged — each cited to the document or marked missing.",
        params_schema=[{"name": "doc_ids", "label": "NDA(s)", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to review every document in the vault."}],
        columns=[
            {"key": "term", "label": "Confidentiality Term", "kind": "clause",
             "prompt": "Find the confidentiality survival/term clause. Quote it.",
             "risk_rubric": "standard if a defined finite term (e.g. 2-5 years); non_standard if perpetual or undefined; missing if absent"},
            {"key": "definition", "label": "Definition of Confidential Info", "kind": "clause",
             "prompt": "Find how 'Confidential Information' is defined. Quote the definition.",
             "risk_rubric": "standard if scoped with carve-outs (public/independently-developed); non_standard if unbounded; missing if absent"},
            {"key": "permitted", "label": "Permitted Disclosures", "kind": "clause",
             "prompt": "Find the permitted-disclosure / need-to-know carve-out. Quote it, or MISSING."},
            {"key": "mutual", "label": "Mutual vs One-Way", "kind": "clause",
             "prompt": "Is the NDA mutual or one-way? Quote the operative obligation that shows which.",
             "risk_rubric": "standard if mutual; non_standard if one-way against us; missing if unclear"},
            {"key": "governing_law", "label": "Governing Law / Seat", "kind": "clause",
             "prompt": "Find the governing-law and dispute-resolution/seat clause. Quote it.",
             "risk_rubric": "standard if Indian law with a named seat; else non_standard; missing if absent"},
        ],
        fanout={"max_cells": 120},
    ),
    # G7 wedge §2.2 — Contract abstraction → key terms across a portfolio into a grid.
    WorkflowTemplate(
        id="contract-abstraction",
        title="Contract abstraction across a portfolio",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=6,
        description="Abstract the key commercial terms from every contract in the vault into one "
                    "grid — parties, effective date, term, value, renewal, termination, and "
                    "governing law — each cell quoted to its source or flagged.",
        params_schema=[{"name": "doc_ids", "label": "Contracts", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to abstract every document in the vault."}],
        columns=[
            {"key": "parties", "label": "Parties", "kind": "clause", "prompt": "Identify the contracting parties. Quote the parties clause."},
            {"key": "effective_date", "label": "Effective Date", "kind": "clause", "prompt": "Find the effective/commencement date. Quote it, or MISSING."},
            {"key": "term", "label": "Term", "kind": "clause", "prompt": "Find the term/duration clause. Quote the period."},
            {"key": "value", "label": "Contract Value", "kind": "clause", "prompt": "Find the consideration / contract value / fees. Quote it, or MISSING."},
            {"key": "renewal", "label": "Renewal", "kind": "clause", "prompt": "Find the renewal / auto-renewal clause. Quote it, or MISSING."},
            {"key": "termination", "label": "Termination", "kind": "clause", "prompt": "Find the termination clause + notice period. Quote it."},
            {"key": "governing_law", "label": "Governing Law", "kind": "clause", "prompt": "Find the governing-law clause. Quote it."},
        ],
        fanout={"max_cells": 120},
    ),
    # G7 wedge §2.2 — M&A due-diligence → request list → ingest data-room → present/partial/missing.
    WorkflowTemplate(
        id="ma-due-diligence",
        title="M&A due-diligence — data-room request review",
        practice_area="Transactional", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=6,
        description="Provide the due-diligence request list, and DocQuery sweeps the data-room "
                    "documents to determine whether each item is present, partial, or missing — "
                    "each answer cited to the supporting document — the red-flag report.",
        params_schema=[{"name": "doc_ids", "label": "Data-room documents", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to search the whole data room (vault)."},
                       {"name": "items", "label": "Due-diligence request items (one per line)", "type": "textarea",
                        "required": True, "help": "e.g. Certificate of incorporation\\nMOA & AOA\\nBoard resolutions\\nMaterial contracts\\nShareholding pattern\\nLitigation summary"}],
        columns=[],  # NOTE: dynamic — one column per request item at resolve time (see _dynamic_columns)
        fanout={"max_cells": 120},
    ),

    # ── Financial services ───────────────────────────────────────────────────────
    WorkflowTemplate(
        id="filings-compare-memo",
        title="Compare filings — comparison memo",
        practice_area="Financial services", output_type="Draft", shape="report", base_mode="deep",
        tool_subset=_NUMERIC, step_count=3,
        description="Pick filings and metrics, and DocQuery drafts a cited comparison memo across "
                    "them — every figure computed from the source tables.",
        params_schema=[{"name": "metrics", "label": "Metrics to compare", "type": "text",
                        "required": True, "help": "e.g. revenue, net income, operating margin"}],
        question_format="Produce a cited comparison memo across the filings in this vault, "
                        "comparing: {metrics}.",
        prompt_overlay=_draft_overlay(
            "Structure: `## Overview`, then one `## <Metric>` section per metric with the computed "
            "figure for each filing (via `compute`/`table_lookup`, cited to the cell) and a "
            "one-line comparison. End with `## Takeaways`. Every number traces to a cell."),
    ),
    WorkflowTemplate(
        id="covenant-check",
        title="Covenant compliance check",
        practice_area="Financial services", output_type="Draft", shape="report", base_mode="standard",
        tool_subset=_NUMERIC, step_count=3,
        description="Provide covenants and thresholds, and DocQuery checks each against the "
                    "computed figures — pass or breach, with the cited cell.",
        params_schema=[{"name": "covenants", "label": "Covenants & thresholds", "type": "textarea",
                        "required": True, "help": "e.g. Net debt / EBITDA must be below 3.0x\\nInterest cover above 4.0x"}],
        question_format="For each covenant below, state the threshold, the actual computed value, "
                        "and PASS or BREACH, citing the source cell:\n{covenants}",
        prompt_overlay=_draft_overlay(
            "One `## <Covenant>` section each: threshold, the computed actual (via `compute`, "
            "cited), and a bold PASS or BREACH. If a figure isn't in the documents, say so and "
            "abstain on that covenant — never guess a value."),
    ),
    # G7 wedge §2.2 — Covenant compliance → extract covenants → compute current state → breach report (→ F3).
    WorkflowTemplate(
        id="covenant-compliance",
        title="Covenant compliance — extract, compute, breach report",
        practice_area="Financial services", output_type="Draft", shape="report", base_mode="deep",
        tool_subset=_NUMERIC, step_count=5,
        description="Extract the financial covenants from the credit agreement, compute each ratio "
                    "from the borrower's filings, and produce a breach report — PASS or BREACH per "
                    "covenant, every figure traced to its source cell.",
        params_schema=[{"name": "focus", "label": "Covenants to focus on (optional)", "type": "text",
                        "required": False, "help": "e.g. leverage ratio, interest cover, DSCR — leave empty to cover all found."}],
        question_format="Extract the financial covenants from the credit agreement(s) in this "
                        "vault, compute each from the borrower's financials, and report PASS or "
                        "BREACH for each. {focus}",
        prompt_overlay=_draft_overlay(
            "Structure: `## Covenants` — for EACH covenant found in the agreement, a sub-heading "
            "with: the covenant as drafted (cited to the clause [doc p.N]), the threshold, the "
            "computed actual value (via `compute`/`table_lookup`, cited to the source cell), and a "
            "bold **PASS** or **BREACH**. Then `## Summary` — the count of passes and breaches and "
            "any covenant you could not test. NEVER guess a ratio: if a figure needed to test a "
            "covenant is not in the documents, mark that covenant **UNTESTED** and say which input "
            "was missing — never a fabricated value."),
    ),
    WorkflowTemplate(
        id="extract-terms-credit-agreements",
        title="Extract terms from credit agreements",
        practice_area="Financial services", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=4,
        description="Upload credit agreements, and DocQuery generates a table of key terms — "
                    "facility, rate, maturity, financial covenants, and events of default.",
        params_schema=[{"name": "doc_ids", "label": "Credit agreements", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "facility", "label": "Facility / Amount", "kind": "clause", "prompt": "Find the facility type and committed amount. Quote it."},
            {"key": "rate", "label": "Interest Rate", "kind": "clause", "prompt": "Find the interest rate / margin clause. Quote it."},
            {"key": "maturity", "label": "Maturity", "kind": "clause", "prompt": "Find the maturity/repayment date. Quote it."},
            {"key": "covenants", "label": "Financial Covenants", "kind": "clause", "prompt": "Find the financial covenants. Quote the key ratios, or MISSING."},
            {"key": "default", "label": "Events of Default", "kind": "clause", "prompt": "Find the events-of-default clause. Quote the principal triggers."},
        ],
        fanout={"max_cells": 120},
    ),
    WorkflowTemplate(
        id="extract-terms-stock-purchase",
        title="Extract terms from stock purchase agreements",
        practice_area="Financial services", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=2,
        description="Upload stock purchase agreements, and DocQuery generates a table of key terms "
                    "from each — consideration, conditions, reps, and indemnification.",
        params_schema=[{"name": "doc_ids", "label": "SPAs", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "consideration", "label": "Consideration", "kind": "clause", "prompt": "Find the purchase price / consideration clause. Quote it."},
            {"key": "conditions", "label": "Closing Conditions", "kind": "clause", "prompt": "Find the conditions to closing. Quote the principal ones."},
            {"key": "reps", "label": "Reps & Warranties", "kind": "clause", "prompt": "Summarize the key representations. Quote the survival period."},
            {"key": "indemnity", "label": "Indemnification", "kind": "clause", "prompt": "Find the indemnification clause + any cap/basket. Quote it."},
        ],
        fanout={"max_cells": 120},
    ),

    # ── Compliance (India) — the wedge ─────────────────────────────────────────────
    WorkflowTemplate(
        id="sebi-lodr-disclosure-check",
        title="SEBI LODR disclosure check",
        practice_area="Compliance (India)", output_type="Draft", shape="report", base_mode="deep",
        tool_subset=_PROSE, step_count=3,
        description="Check a filing/announcement against the principal SEBI (LODR) disclosure "
                    "requirements — present, partial, or absent, each cited.",
        params_schema=[],
        question_format="Review the document(s) in this vault for compliance with the principal "
                        "SEBI (LODR) disclosure requirements.",
        prompt_overlay=_draft_overlay(
            "Use `## <Requirement>` sections (board composition, related-party transactions, "
            "material events, financial results). State whether each appears satisfied, citing the "
            "disclosed text [doc p.N], or mark it absent. Flag gaps. Do not assert compliance you "
            "cannot ground in the document."),
    ),
    WorkflowTemplate(
        id="companies-act-resolution-check",
        title="Companies Act 2013 resolution check",
        practice_area="Compliance (India)", output_type="Review", shape="grid", base_mode="grid",
        tool_subset=_PROSE, step_count=4,
        description="Review board/shareholder resolutions for the formalities the Companies Act "
                    "2013 requires — quorum, notice, authority, and filing references.",
        params_schema=[{"name": "doc_ids", "label": "Resolutions", "type": "doc_multiselect",
                        "required": False, "help": "Leave empty to cover every document in the vault."}],
        columns=[
            {"key": "type", "label": "Resolution Type", "kind": "clause", "prompt": "Identify the resolution type (ordinary/special; board/shareholder). Quote the heading."},
            {"key": "quorum", "label": "Quorum", "kind": "clause", "prompt": "Find the quorum recital. Quote it, or MISSING."},
            {"key": "notice", "label": "Notice", "kind": "clause", "prompt": "Find the notice-of-meeting recital. Quote it, or MISSING."},
            {"key": "authority", "label": "Authority", "kind": "clause", "prompt": "Find the statutory section / authority cited. Quote it, or MISSING."},
        ],
        fanout={"max_cells": 120},
    ),

    # ── Cross-cutting — engine-level tasks every practice uses ──────────────────────
    WorkflowTemplate(
        id="translate-document",
        title="Translate a document",
        practice_area="Cross-cutting", output_type="Output", shape="output", base_mode="standard",
        tool_subset=_PROSE, step_count=2,
        description="Translate a document (or a passage) into another language, preserving legal "
                    "meaning — useful for India's multilingual documents.",
        params_schema=[{"name": "target_language", "label": "Target language", "type": "text",
                        "required": True, "help": "e.g. English, Hindi, Gujarati"},
                       {"name": "scope", "label": "What to translate (optional)", "type": "text",
                        "required": False, "help": "e.g. the indemnity clause; leave empty for the whole document"}],
        question_format="Translate {scope} the document(s) in this vault into {target_language}, "
                        "preserving legal meaning.",
        prompt_overlay=_output_overlay(
            "Read the source via `search_vault`/`read_document`, then produce the translation as "
            "markdown, preserving structure (headings, numbered clauses). Keep a [doc p.N] citation "
            "at the start of each translated section so the source is traceable. Do not add or "
            "drop substantive content; note any term that is untranslatable in-line."),
    ),
    WorkflowTemplate(
        id="summarize-document",
        title="Summarize a document",
        practice_area="Cross-cutting", output_type="Output", shape="output", base_mode="standard",
        tool_subset=_PROSE, step_count=2,
        description="Produce a concise, cited summary of a document or the whole vault — the key "
                    "points, parties, obligations, and risks.",
        params_schema=[{"name": "focus", "label": "Focus (optional)", "type": "text",
                        "required": False, "help": "e.g. obligations on our client; risk flags"}],
        question_format="Summarize the document(s) in this vault. {focus}",
        prompt_overlay=_output_overlay(
            "Produce a short cited summary: a one-paragraph overview, then `## Key Points` as cited "
            "bullets (parties, term, obligations, notable risks). Every bullet cites [doc p.N]."),
    ),
]


def _dynamic_columns(template: WorkflowTemplate, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A few grid templates build their columns from a free-text param (e.g. the diligence
    request list = one column per item). Keeps the template DATA while letting the user define
    the axis. Falls back to the template's static columns when no dynamic param is present."""
    if template.id in ("check-diligence-request-list", "ma-due-diligence"):
        raw = str(params.get("items", "") or "")
        items = [ln.strip() for ln in raw.replace("\\n", "\n").splitlines() if ln.strip()][:10]
        cols = []
        for i, it in enumerate(items):
            cols.append({
                "key": f"item_{i}", "label": it[:40], "kind": "clause",
                "prompt": f"Is the diligence item '{it}' satisfied by this document? "
                          f"Quote the supporting text, or mark MISSING.",
                "risk_rubric": "standard if present; missing if absent",
            })
        return cols or template.columns
    return template.columns


def resolve_grid_columns(template: WorkflowTemplate, params: Dict[str, Any]) -> List[GridColumn]:
    """Columns for a grid run — dynamic where a template defines them from params, else static."""
    return _to_columns(_dynamic_columns(template, params))


# The registry. The route reads this map.
TEMPLATES: Dict[str, WorkflowTemplate] = {t.id: t for t in _T}

# Practice-area display order (gallery groups in this order; matches Harvey's grouping).
PRACTICE_ORDER = ["Litigation", "Transactional", "Financial services",
                  "Compliance (India)", "Cross-cutting"]


def list_templates() -> List[WorkflowTemplate]:
    order = {p: i for i, p in enumerate(PRACTICE_ORDER)}
    return sorted(TEMPLATES.values(), key=lambda t: (order.get(t.practice_area, 99), t.title))


def get_template(template_id: str) -> Optional[WorkflowTemplate]:
    return TEMPLATES.get(template_id)


# ── S-C: tool-subset validation ───────────────────────────────────────────────

def validate_tool_subset(tool_subset: List[str], mode: str = "standard") -> Dict[str, Any]:
    """Validate that every tool in `tool_subset` is a real, registered tool for `mode`.

    Returns {"ok": True} when every tool name is known, or
    {"ok": False, "unknown": [...], "repair": "<message>"} for unrecognised names.

    This is the S-C runtime guard: a template whose tool_subset contains a bogus
    name would silently drop it via REGISTRY.names() — but the caller would never
    know a step was inadvertently denied a tool it expected.  Making the bad name
    LOUD at resolve/validate time is the fix.
    """
    from .registry import REGISTRY, SCHEMAS
    known = set(SCHEMAS.keys())  # the full universe of registered tool names
    bad = [t for t in (tool_subset or []) if t not in known]
    if not bad:
        return {"ok": True, "tool_subset": list(tool_subset)}
    repair = (
        f"Unknown tool name(s) in workflow tool_subset: {bad}. "
        f"Valid tools: {sorted(known)}. "
        "Remove or correct the unknown names — a workflow step cannot use a tool "
        "outside its declared subset."
    )
    return {"ok": False, "unknown": bad, "repair": repair}


def _step_failure_detail(step_label: str, reason: str, step_index: int) -> Dict[str, Any]:
    """Build a structured per-step failure payload for the S-C reporting contract.

    The route streams this as a `step_failure` event so the UI (and the caller)
    sees WHICH step failed and WHY — not a blanket 'workflow failed'.

    Args:
        step_label: human-readable step identifier (e.g. 'doc:doc_id / col:governing_law').
        reason: the abstain_reason or error string from that step's result envelope.
        step_index: 0-based position in the fan-out so the UI can locate it.
    """
    return {
        "step_index": step_index,
        "step_label": step_label,
        "reason": reason,
    }


def template_card(t: WorkflowTemplate) -> Dict[str, Any]:
    """The gallery-card view (no internal overlay/columns leaked — the form renders from
    params_schema, the result from the streamed events)."""
    return {
        "id": t.id,
        "title": t.title,
        "practice_area": t.practice_area,
        "description": t.description,
        "shape": t.shape,
        "output_type": t.output_type,     # Harvey's Review / Draft / Output tag
        "step_count": t.step_count,
        "params_schema": t.params_schema,
    }
