"""QueryIR + comprehend() — BRAIN_REASONING_PLAN §5.2 (Wernicke → typed IR).

Turn a natural-language question into a typed intermediate representation that the
executive layer (planner/workspace/executor) consumes. The LLM proposes the STRUCTURE
of the ask (type/metrics/entities/periods/constraints) as DATA — it never retrieves and
never emits a number. On any parse/validation failure we degrade to a minimal IR
(`question_type="lookup"`, raw passthrough) so the fast path is never worse than today.

Why this organ kills the `aws_margin` confident-wrong class (§5.5 lesson): the failure
was that the LLM free-picked `margin_pct(AWS op income ÷ Consolidated op income)` — a
spec string-identical to a correct margin. Here, "AWS operating margin" comprehends to
metrics=["operating margin"], entities=["AWS"], constraints={aggregation:...}; the
planner then DERIVES the denominator (AWS net sales) from the entity, so the consolidated
denominator can't be expressed. The fix lives upstream, as structure — not a downstream
guard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# The QueryIR question-type basis (§5.2). A small set that RECOMBINES into thousands of
# question shapes (§1) — NOT a handler per question. Each maps to how the planner
# decomposes:
#   lookup          — one stated value, no pivot ("Amazon FY2023 net sales")
#   extremum_pivot  — argmax/argmin/first_exceeds binds a pivot, then read a value
#   lookup_pivot    — a value/condition binds a pivot, then read another value
#   compare         — two+ entities/periods compared
#   exists          — a yes/no existence/threshold ("did Google Cloud reach profitability")
#   qualitative     — narrative/explanatory; routed to claims+verifier, never the kernel
#   compound        — multiple sub-questions to decompose
QUESTION_TYPES = (
    "lookup", "extremum_pivot", "lookup_pivot", "compare", "exists", "qualitative", "compound",
)


@dataclass
class QueryIR:
    raw: str
    question_type: str = "lookup"
    metrics: List[str] = field(default_factory=list)     # ["operating income", "net sales"]
    entities: List[str] = field(default_factory=list)    # ["AWS"], ["Microsoft","Amazon"]
    periods: List[str] = field(default_factory=list)     # ["2022"], [] = to-be-resolved
    constraints: Dict[str, Any] = field(default_factory=dict)  # {threshold, predicate, aggregation}
    degraded: bool = False    # True when comprehension failed → minimal passthrough IR

    @property
    def is_pivot(self) -> bool:
        return self.question_type in ("extremum_pivot", "lookup_pivot")


def _minimal(question: str) -> QueryIR:
    """The safe degrade target: a plain lookup with raw passthrough (§5.2 invariant)."""
    return QueryIR(raw=question, question_type="lookup", degraded=True)


_COMPREHEND_SYSTEM = (
    "You parse a finance/legal question into a TYPED structure. You DO NOT answer it, "
    "retrieve anything, or produce any number — you only describe the SHAPE of the ask.\n\n"
    "Output ONE JSON object (no prose, no code fences) with these fields:\n"
    '  "question_type": one of '
    '["lookup","extremum_pivot","lookup_pivot","compare","exists","qualitative","compound"]\n'
    '  "metrics":  array of the financial/legal line-items the question involves, lowercase, '
    'short noun phrases. ORDER IS A CONTRACT: when one metric\'s condition FINDS the year/'
    'entity (the pivot clause) and another metric is then READ for it, metrics[0] MUST be the '
    'pivot-clause metric and metrics[1] the read-clause metric. NEVER omit the metric inside '
    'the condition clause — in "the year X first exceeded $N, what was Y", X is metrics[0] '
    'and Y is metrics[1], even though the question ultimately asks for Y. If the read clause '
    'asks for the SAME metric\'s figure ("…and what was that figure"), repeat it.\n'
    '  "entities": array of the companies/segments/parties (e.g. ["AWS","Amazon"], '
    '["Microsoft","Amazon"], ["Google Cloud","Alphabet"]). [] if none named. When a SEGMENT '
    'and its PARENT company are both named, list the segment FIRST.\n'
    '  "periods":  array of explicit fiscal years/periods AS WRITTEN (e.g. ["2022"], ["2021","2023"]). '
    'If the period is to be FOUND by the question (a pivot), leave it [] — do NOT guess it.\n'
    '  "constraints": object; include only what applies: '
    '{"predicate":"first_exceeds"|"last_below"|"argmax"|"argmin", "threshold":<number in the cells\' '
    'units, e.g. $20B over $-millions = 20000>, "aggregation":"total"|"component"}.\n\n'
    "Type guide (pick the SINGLE best fit; prefer these specific types over the vaguer "
    "ones — use compound/qualitative only as a last resort):\n"
    "- lookup: one stated value, no pivot ('Amazon total net sales in FY2023').\n"
    "- extremum_pivot: an EXTREMUM picks a year OR an entity — 'which YEAR did X first exceed/"
    "cross/peak/grow most' (predicate=first_exceeds/last_below) OR 'which COMPANY/SEGMENT among "
    "a set had the highest/lowest/most/least Y' (predicate=argmax/argmin). 'which of A, B, C has "
    "the highest …' is ALWAYS extremum_pivot (argmax over entities), NOT compare and NOT "
    "lookup_pivot — even when a follow-on value is then read for the winner.\n"
    "- lookup_pivot: a stated NON-extremum condition picks the pivot, then read another value "
    "('the year MSFT revenue was CLOSEST TO $200B, what did Alphabet report'; 'the year Amazon "
    "posted a net loss, what were its net sales'). Use this only when the pivot is a match/"
    "equality/closest condition, not a max/min.\n"
    "- compare: a direct side-by-side of NAMED entities/periods with no extremum and no pivot "
    "('was Amazon's 2023 revenue higher than 2022's'). If the question asks 'WHICH has the most/"
    "least', that is extremum_pivot, not compare.\n"
    "- exists: a YES/NO about whether something holds — 'did X reach profitability', 'profit OR a "
    "loss', 'did Y increase every year', 'was Z profitable'. Choose exists for any profit/loss or "
    "happened-or-not question, EVEN IF it also asks for the related figure.\n"
    "- qualitative: asks to EXPLAIN/DESCRIBE in narrative ('why did margins fall', 'summarize the "
    "risk factors') — not a yes/no and not a single figure.\n"
    "- compound: several INDEPENDENT sub-questions joined ('what were 2021, 2022, and 2023 sales "
    "AND who was CEO'). A single pivot question that then reads one value is NOT compound.\n\n"
    "Rules: aggregation='total' when the wording says total/consolidated/overall; 'component' for a "
    "named sub-line. Put a $ threshold in the SAME units the figures use. Never invent a period that "
    "the question is asking you to find. SIGN RULE for losses: filings record losses as NEGATIVE "
    "values, so 'largest/larger LOSS' = argmin (most negative), 'smallest loss' = argmax — the "
    "predicate direction follows the signed value, not the word 'larger'. If a pivot is ALREADY "
    "resolved by the question itself ('X happened in fiscal 2023, what was Y that year'), type it "
    "lookup with that period — the pivot needs no finding.\n\n"
    "Worked bridge example (the shape that matters most): \"In the year Contoso's cloud segment "
    "operating income first exceeded $5 billion, what was Contoso's total revenue?\" →\n"
    '{"question_type":"extremum_pivot","metrics":["operating income","total revenue"],'
    '"entities":["cloud","Contoso"],"periods":[],'
    '"constraints":{"predicate":"first_exceeds","threshold":5000,"aggregation":"total"}}\n'
    "(threshold 5000 because filing figures are in $ millions; operating income is metrics[0] "
    "because its condition finds the year, total revenue is metrics[1] because it is read for "
    "that year.)\n\n"
    "Output ONLY the JSON object."
)


# ── Deterministic IR validation (the comprehension-side monitor) ─────────────────
#
# The §5.5 lesson applied upstream: the one LLM in the spine can silently DROP the
# pivot-clause metric (observed live — "…AWS operating income first exceeded $20B, what
# was total net sales" came back with metrics=["total net sales"] only), and every
# downstream organ is deterministic, so nothing recovers. These checks read STRUCTURE
# (the question's own text), never content: a pivot question with a separate read clause
# must carry ≥2 metrics, and every extracted metric must actually appear in the question
# (no invented line-items). A failed check triggers ONE corrective retry; degraded only
# if a metric stays hallucinated. No per-question tuning — the checks are class-level.

_READ_CLAUSE = re.compile(
    r",\s*(what|how much|how many)\b|\bwhat (was|were|is|are)\b", re.I
)


def _metric_in_question(metric: str, question_lower: str) -> bool:
    """Does the metric (fuzzily) appear in the question? Token-level: any informative
    token (≥4 chars) of the metric occurring in the question counts — extraction is
    expected to lightly normalize ('R&D expenses' → 'research and development expenses'
    still shares 'research'/'development'/'expenses')."""
    toks = [t for t in re.split(r"\W+", (metric or "").lower()) if len(t) >= 4]
    if not toks:
        return True   # too short to judge either way
    return any(t in question_lower for t in toks)


def ir_structural_issues(question: str, ir: "QueryIR") -> List[str]:
    """Class-level structural problems in an IR, as human-readable strings (empty = OK).
    Used by comprehend() for the corrective retry and by the comprehension gate as the
    bridge-completeness metric."""
    issues: List[str] = []
    if ir.degraded:
        return issues
    ql = (question or "").lower()
    if ir.question_type in ("extremum_pivot", "lookup_pivot"):
        if _READ_CLAUSE.search(question or "") and len(ir.metrics) < 2:
            issues.append(
                "the question has BOTH a pivot/condition clause and a separate read clause, "
                f"but \"metrics\" has only {len(ir.metrics)} item(s) — metrics[0] must be the "
                "pivot-clause metric and metrics[1] the read-clause metric"
            )
    for m in ir.metrics:
        if not _metric_in_question(m, ql):
            issues.append(
                f'metric "{m}" does not appear in the question — extract metrics from the '
                "question's own wording, never invent one"
            )
    return issues


def _ask(question: str, llm, extra_messages=None) -> Optional[QueryIR]:
    """One comprehend round: invoke → parse JSON-as-data → schema-validate. Returns None
    on any parse/invoke failure (the caller decides whether to retry or degrade)."""
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        messages = [
            SystemMessage(content=_COMPREHEND_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nJSON object only:"),
        ]
        if extra_messages:
            messages.extend(extra_messages)
        resp = llm.invoke(messages)
        text = (resp.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):] if "{" in text else text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        obj = json.loads(text[start:end + 1])
        if not isinstance(obj, dict):
            return None
        return _validate(question, obj)
    except Exception:
        return None


def comprehend(question: str, llm) -> QueryIR:
    """Question → QueryIR. JSON-as-data, schema-validated, degrade-on-failure (§5.2).

    No retrieval, no numbers. Any non-JSON / off-schema output → a minimal lookup IR so
    the downstream fast path still works (never worse than baseline).

    Structural validation + ONE corrective retry: the deterministic `ir_structural_issues`
    checks (pivot question with a read clause must carry both metrics; every metric must
    appear in the question) run on the first parse; if they fail, the model gets ONE
    retry with the precise problems quoted back. The better of the two IRs wins (fewer
    issues). A persistently HALLUCINATED metric degrades to the minimal IR (a wrong
    metric is worse than falling through); a persistently missing second metric keeps
    the IR — a pivot-only plan still executes or abstains safely downstream.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    if not (question or "").strip():
        return _minimal(question or "")

    ir = _ask(question, llm)
    if ir is None:
        return _minimal(question)

    issues = ir_structural_issues(question, ir)
    if not issues:
        return ir

    # one corrective retry with the exact structural problems quoted back (class-level
    # feedback, not per-question tuning — the same message fixes every bridge shape)
    feedback = (
        "Your JSON had structural problems:\n- " + "\n- ".join(issues) +
        "\nRe-output the corrected JSON object only."
    )
    retry = _ask(question, llm, extra_messages=[
        AIMessage(content="(previous attempt)"),
        HumanMessage(content=feedback),
    ])
    if retry is not None:
        retry_issues = ir_structural_issues(question, retry)
        if len(retry_issues) < len(issues):
            ir, issues = retry, retry_issues

    # a metric that STILL doesn't appear in the question is an invention — degrade
    # (fall through to the existing path) rather than plan over a hallucinated row.
    if any("does not appear in the question" in i for i in issues):
        return _minimal(question)
    return ir


def _as_str_list(v) -> List[str]:
    """Coerce a field to a clean list[str] (the model sometimes returns a bare string)."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


_YEARISH = re.compile(r"^(?:19|20)\d{2}$|^FY\s?\d{2,4}$|^Q[1-4]\b", re.I)


def _validate(question: str, obj: Dict[str, Any]) -> QueryIR:
    """Validate/normalize the LLM object into a QueryIR. Off-schema fields are dropped,
    NOT errors — the IR degrades field-by-field rather than failing whole.
    """
    qt = str(obj.get("question_type", "")).strip().lower()
    if qt not in QUESTION_TYPES:
        qt = "lookup"   # unknown type → safest default; downstream still runs

    metrics = [m.lower() for m in _as_str_list(obj.get("metrics"))]
    entities = _as_str_list(obj.get("entities"))
    # periods: keep only year/period-looking tokens (the model must not smuggle a guess
    # for a to-be-resolved pivot; a non-year string is dropped)
    periods = [p for p in _as_str_list(obj.get("periods")) if _YEARISH.match(p)]

    constraints: Dict[str, Any] = {}
    raw_c = obj.get("constraints")
    if isinstance(raw_c, dict):
        pred = str(raw_c.get("predicate", "")).strip().lower()
        if pred in ("first_exceeds", "last_below", "argmax", "argmin"):
            constraints["predicate"] = pred
        thr = raw_c.get("threshold")
        if isinstance(thr, (int, float)):
            constraints["threshold"] = float(thr)
        agg = str(raw_c.get("aggregation", "")).strip().lower()
        if agg in ("total", "component"):
            constraints["aggregation"] = agg

    # Deterministic type correction (structure over the LLM's label): an extremum
    # PREDICATE is definitive — a question carrying first_exceeds/last_below/argmax/
    # argmin IS an extremum_pivot, even when it has a "then read Y" tail that tempts the
    # model to call it lookup_pivot. The model reliably extracts the predicate; we let
    # that predicate (not the looser type word) decide the type. This is the
    # constraint-determines-type invariant, not a per-question patch.
    if constraints.get("predicate") in ("first_exceeds", "last_below", "argmax", "argmin"):
        qt = "extremum_pivot"

    return QueryIR(
        raw=question, question_type=qt, metrics=metrics,
        entities=entities, periods=periods, constraints=constraints, degraded=False,
    )
