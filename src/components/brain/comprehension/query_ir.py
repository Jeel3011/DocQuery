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
from typing import List, Dict, Any


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
    '  "metrics":  array of the financial/legal line-items asked about, lowercase, as short '
    'noun phrases (e.g. ["operating income","net sales"], ["total revenue"], ["operating margin"]).\n'
    '  "entities": array of the companies/segments/parties (e.g. ["AWS"], ["Microsoft","Amazon"], '
    '["Google Cloud"]). [] if none named.\n'
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
    "the question is asking you to find. Output ONLY the JSON object."
)


def comprehend(question: str, llm) -> QueryIR:
    """Question → QueryIR. JSON-as-data, schema-validated, degrade-on-failure (§5.2).

    No retrieval, no numbers. Any non-JSON / off-schema output → a minimal lookup IR so
    the downstream fast path still works (never worse than baseline).
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    if not (question or "").strip():
        return _minimal(question or "")
    try:
        resp = llm.invoke([
            SystemMessage(content=_COMPREHEND_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nJSON object only:"),
        ])
        text = (resp.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):] if "{" in text else text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return _minimal(question)
        obj = json.loads(text[start:end + 1])
        if not isinstance(obj, dict):
            return _minimal(question)
        return _validate(question, obj)
    except Exception:
        return _minimal(question)


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
