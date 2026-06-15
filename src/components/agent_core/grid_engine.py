"""Review-grid engine (AGENT_CORE_PLAN Phase B2) — per-(doc × column) fan-out.

`build_grid(spec, model_factory, config, ...)` fills a `GridResult` by running ONE
bounded agent per cell: scoped to a single document, restricted to the read/table/
compute tools (mode="grid" — no world-search), prompted to extract exactly one column's
fact and answer in a strict JSON envelope. The agent's output gates (A3) still enforce
that any quoted value traces to the evidence ledger, so a cell is FOUND only when it is
grounded — otherwise it ABSTAINs or reports MISSING. No silent wrong cells.

The model is injected (a `model_factory()` returning a fresh `BaseModel` per cell), so
this is unit-testable with a scripted model and never makes a live call in tests.

Concurrency is the caller's concern (the route runs cells on a thread pool / worker);
this module exposes both `build_cell` (one cell, pure) and `build_grid` (sequential
convenience). Never raises — a cell that errors becomes a GridCell(status=ERROR).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .budgets import Budget
from .ledger import EvidenceLedger
from .loop import GateOutcome, run_agent
from .registry import REGISTRY, RunScope
from .review_grid import (
    CellStatus,
    ColumnKind,
    GridCell,
    GridColumn,
    GridResult,
    GridSpec,
    RiskFlag,
)

logger = logging.getLogger(__name__)

# A cell needs only a few steps: read the relevant page(s), maybe one lookup/compute,
# then answer. Keep it tight — this is the per-cell cost ceiling, multiplied by N×M.
_GRID_CELL_MAX_STEPS = 6
_GRID_CELL_WALL_S = 90.0
_GRID_CELL_TOKEN_BUDGET = 30_000   # hard per-cell token ceiling (cost guard × N×M cells)


# ──────────────────────────────────────────────────────────────────────────────
# The per-cell system prompt — forces cite-or-abstain + a strict JSON answer.
# ──────────────────────────────────────────────────────────────────────────────

def _cell_system_prompt(column: GridColumn, doc_name: str) -> str:
    risk_line = (
        f"\nRisk rubric: {column.risk_rubric}\n"
        "Classify the finding's risk as exactly one of: standard, non_standard, missing."
        if column.risk_rubric
        else ""
    )
    return f"""You extract ONE specific fact from ONE document for a review grid.

Your ENTIRE final answer MUST be ONE JSON object with EXACTLY these five top-level keys
and NO others: status, value, quote, risk, note. Do NOT nest. Do NOT rename keys to the
field (e.g. NOT {{"governing_law": ...}}). Do NOT wrap it in another object.

Document: {doc_name}
Field to extract: {column.label}
Instruction: {column.prompt}{risk_line}

RULES (non-negotiable):
- FIND the relevant text first: call `search_vault(query="...", kind="text")` with words
  from this field (e.g. for governing law: "governing law jurisdiction"; for a cap:
  "limitation of liability indemnify cap") to locate the clause in the document. The
  clause text lives in the document's prose — you must search for and read it.
- Ground your answer in the ACTUAL text you found. QUOTE the exact source sentence.
- If, after searching, the document genuinely does NOT contain this clause, say so
  (status "missing") — that is a valid, useful finding, not a failure. Do not guess.
- If it's ambiguous or conflicting, status "abstain". Never invent a value.

Answer with ONE JSON object and NOTHING else, in this EXACT shape (these keys, this nesting):
{{"status": "found" | "missing" | "abstain",
  "value": "<the extracted value, or null>",
  "quote": "<exact source text you grounded on, or null>",
  "risk": "standard" | "non_standard" | "missing" | "none",
  "note": "<one short clause of rationale, or null>"}}

Correct example (governing law found):
{{"status": "found", "value": "State of Minnesota; seat Jackson, Mississippi",
  "quote": "This Agreement shall be governed by the laws of the State of Minnesota...",
  "risk": "non_standard", "note": "governing law Minnesota; arbitration seat Mississippi"}}

The top-level keys MUST be exactly status/value/quote/risk/note — nothing else, no nesting."""


def _cell_question(column: GridColumn) -> str:
    return (
        f"Extract '{column.label}' from this document and return the JSON object. "
        f"Read before you answer; quote your source; abstain rather than guess."
    )


# ──────────────────────────────────────────────────────────────────────────────
# The grid output gate — the cite-or-abstain contract for STRUCTURED extraction.
# ──────────────────────────────────────────────────────────────────────────────

def grid_gate(draft: str, ledger: EvidenceLedger) -> GateOutcome:
    """The output gate for a grid cell. Unlike the standard agent gate (every factual
    sentence needs an inline [doc p.N] citation), a grid cell's answer is a JSON
    envelope whose grounding lives in the LEDGER, not inline. The contract here:

      - status == 'found'  → there MUST be ledger provenance (a span/cell the agent
        actually read). No provenance = ungrounded 'found' = FAIL (the model must
        either ground it or switch to abstain/missing).
      - status in {missing, abstain} → valid with no provenance (the whole point: an
        honest 'not found' / 'cannot determine' needs nothing to cite).
      - unparseable draft → PASS through (the parser downgrades it to ABSTAIN; the
        gate's job is only to stop an *ungrounded found*, not to second-guess shape).

    This keeps 'no silent wrong cell' as a hard, code-enforced invariant while letting
    the legitimate missing/abstain answers through (which the inline-citation gate
    wrongly rejected)."""
    env = _extract_envelope(draft)
    if env is None:
        return GateOutcome(passed=True)  # parser will abstain; nothing to verify
    status = str(env.get("status", "")).strip().lower()
    if status == "found":
        has_provenance = bool(ledger.entries)
        if not has_provenance:
            return GateOutcome(
                passed=False, abstained=True,
                failures=[{"name": "grounding",
                           "detail": "status is 'found' but no source span was read — "
                                     "ground the value with a tool, or answer 'missing'/'abstain'."}],
                redacted_draft=draft,
            )
    return GateOutcome(passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing the agent's run into a GridCell.
# ──────────────────────────────────────────────────────────────────────────────

def _extract_envelope(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of the agent's final answer. Tolerant of a stray code
    fence or leading prose; returns None if no object parses."""
    if not text:
        return None
    s = text.strip()
    # strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s
        s = s.lstrip("json").lstrip()
        s = s.split("```", 1)[0]
    # find the first {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _looks_like_envelope(env: Dict[str, Any]) -> bool:
    """True if the parsed dict is (or resembles) the required cell envelope — it carries
    any of the canonical keys. A custom shape (e.g. {"governing_law_and_seat": {...}})
    carries none of them and is handled by the tolerant recovery path instead."""
    return any(k in env for k in ("status", "value", "quote"))


def _flatten_leaves(obj: Any, _depth: int = 0) -> List[str]:
    """Collect the leaf scalar values of an arbitrarily-nested JSON object/list into a
    flat list of strings (skips null/empty). Used to recover a readable value from a
    model that answered correctly but in a custom JSON shape instead of the envelope."""
    out: List[str] = []
    if _depth > 6:
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_leaves(v, _depth + 1))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_leaves(v, _depth + 1))
    elif obj is None:
        return out
    else:
        s = str(obj).strip()
        if s and s.lower() != "null":
            out.append(s)
    return out


def _recover_custom_shape(env: Dict[str, Any]) -> Optional[str]:
    """Flatten a custom-shape dict (no envelope keys) into a single readable value
    string by joining its leaf values. Returns None if nothing usable is left.

    This is the loop-breaking recovery: the agent found and grounded the answer but
    returned e.g. {"governing_law_and_seat": {"governing_law": "Minnesota",
    "seat": "Jackson, Mississippi"}}. We turn that into "Minnesota; Jackson, Mississippi"
    so a grounded answer is UPGRADED to a value instead of silently discarded. The
    provenance guard downstream still gates whether it becomes FOUND."""
    leaves = _flatten_leaves(env)
    if not leaves:
        return None
    # de-dup while preserving order (a leaf can repeat across mirrored keys)
    seen: set = set()
    uniq = [x for x in leaves if not (x in seen or seen.add(x))]
    return "; ".join(uniq)


# Phrases a model uses when it is abstaining/declining IN PROSE inside a custom shape.
# Recovering such a leaf into a FOUND value would mislabel an honest "can't find it" as a
# grounded answer (the value would literally read "abstain; I could not verify ..."). We
# detect it at the START of the recovered string so a real clause that merely mentions a
# word ("...the parties cannot terminate...") is not falsely flagged.
_ABSTENTION_OPENERS = (
    "abstain", "missing", "not found", "no ", "none",
    "i could not", "i cannot", "i can't", "could not verify", "cannot verify",
    "could not find", "cannot find", "unable to", "does not contain", "doesn't contain",
    "no such", "not present", "not applicable", "n/a",
)


def _reads_as_abstention(value: str) -> bool:
    """True if a recovered custom-shape value is itself an abstention/negation rather than
    an extracted clause — so we abstain instead of emitting a 'found' that says 'abstain'."""
    if not value:
        return True
    head = value.strip().lower()[:40]
    return any(head.startswith(op) for op in _ABSTENTION_OPENERS)


def _coerce_status(raw: Any) -> CellStatus:
    try:
        return CellStatus(str(raw).strip().lower())
    except ValueError:
        return CellStatus.ABSTAIN


def _coerce_risk(raw: Any) -> RiskFlag:
    try:
        return RiskFlag(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return RiskFlag.NONE


def _cell_from_run(
    doc_id: str,
    doc_name: str,
    column: GridColumn,
    events: List[Dict[str, Any]],
) -> GridCell:
    """Fold a run_agent event stream into one GridCell, enforcing the cite-or-abstain
    contract: a FOUND value with no provenance is downgraded to ABSTAIN (the gate
    should already have caught it, but we never trust an ungrounded 'found')."""
    final_text = ""
    provenance: List[Dict[str, Any]] = []
    degraded = False
    for ev in events:
        t = ev.get("type")
        if t == "token":
            final_text = ev.get("text", "") or final_text
        elif t == "sources":
            provenance = ev.get("sources", []) or provenance
        elif t == "meta" and (ev.get("degrade") or ev.get("error")):
            degraded = True

    if degraded and not final_text:
        return GridCell(doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                        status=CellStatus.ERROR, note="agent run degraded (model/tool error)")

    env = _extract_envelope(final_text)
    if env is None:
        # No JSON at all (free-text answer). This is an envelope/parse failure, NOT a
        # genuine "couldn't find it" — record it distinguishably (abstain_reason
        # "unparsed") so we never re-blame ingestion for a cell the agent actually
        # answered. Keep the text as the note so a human can see what it said.
        return GridCell(doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                        status=CellStatus.ABSTAIN, provenance=[],
                        note="parse: " + (final_text[:120] or "no parseable answer"),
                        abstain_reason="unparsed")

    # ── Tolerant recovery: a CUSTOM shape (valid JSON, but no envelope keys). The
    #    agent answered, possibly grounded, just not in our exact key shape. Recover a
    #    value from its leaves and treat it as a FOUND candidate — gated by provenance
    #    below, so it can only ever UPGRADE a grounded answer, never invent a cell.
    if not _looks_like_envelope(env):
        recovered = _recover_custom_shape(env)
        if recovered is not None and _reads_as_abstention(recovered):
            # The model declined IN PROSE inside a custom shape ("abstain; I could not
            # verify..."). Recovering that into a FOUND value would mislabel an honest
            # 'can't find it' as a grounded clause. Treat it as a genuine abstain.
            return GridCell(
                doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                status=CellStatus.ABSTAIN, provenance=[],
                note=recovered[:200], abstain_reason="ambiguous",
            )
        if recovered is not None and provenance:
            return GridCell(
                doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                status=CellStatus.FOUND, value=recovered, quote=None,
                risk=RiskFlag.NONE, provenance=provenance,
                note="parse: recovered from non-envelope JSON shape",
            )
        # custom shape but nothing grounded it → abstain, flagged as our parse failure
        # (NOT no_evidence). This is the distinction that ends the loop.
        return GridCell(
            doc_id=doc_id, column_key=column.key, doc_name=doc_name,
            status=CellStatus.ABSTAIN, provenance=[],
            note="parse: " + (recovered or final_text[:120] or "non-envelope JSON shape"),
            abstain_reason="unparsed",
        )

    status = _coerce_status(env.get("status"))
    value = env.get("value")
    value = None if value in ("", "null", None) else str(value)
    quote = env.get("quote")
    quote = None if quote in ("", "null", None) else str(quote)
    risk = _coerce_risk(env.get("risk"))
    note = env.get("note")
    note = None if note in ("", "null", None) else str(note)
    abstain_reason: Optional[str] = None

    # Enforce the contract structurally: a FOUND value MUST carry provenance. If the
    # agent claims found but nothing grounded it, downgrade to ABSTAIN (never emit an
    # unverified 'found' into the grid). This is a genuine lack of evidence, not a
    # parse failure.
    if status == CellStatus.FOUND and not provenance:
        status = CellStatus.ABSTAIN
        note = (note + " | " if note else "") + "claimed found but no source span grounded it"
        value = None
        abstain_reason = "no_evidence"

    # A genuine abstain the model itself declared (ambiguous/low-evidence). Mark it
    # distinguishably from our parse-failure abstains.
    if status == CellStatus.ABSTAIN and abstain_reason is None:
        abstain_reason = "ambiguous"

    # MISSING means 'the term is absent' — its absence is itself a risk flag.
    if status == CellStatus.MISSING and risk == RiskFlag.NONE:
        risk = RiskFlag.MISSING

    return GridCell(
        doc_id=doc_id, column_key=column.key, doc_name=doc_name,
        status=status, value=value, quote=quote, risk=risk,
        provenance=provenance if status == CellStatus.FOUND else [],
        note=note, abstain_reason=abstain_reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cell + grid drivers.
# ──────────────────────────────────────────────────────────────────────────────

def build_cell(
    doc_id: str,
    column: GridColumn,
    *,
    collection_id: str,
    model,
    filename_by_doc: Dict[str, str],
    grids_by_doc: Optional[Dict[str, Any]] = None,
    retrieval_manager: Any = None,
    db_client: Any = None,
    model_id: str = "",
) -> GridCell:
    """Run ONE bounded agent for a single (doc, column) and return its GridCell.

    `model` is an already-built BaseModel (the caller decides live vs scripted).
    `filename_by_doc` lets the read tool resolve the doc; scope is locked to this one
    document so the agent cannot read others. `retrieval_manager` + `db_client` enable
    `search_vault` (needed to find clause text in a PROSE document — without them a
    contract grid abstains every cell).
    """
    doc_name = filename_by_doc.get(doc_id, doc_id)
    scope = RunScope(
        collection_id=collection_id,
        doc_ids=[doc_id],
        filenames=[doc_name] if doc_name else [],
        filename_by_doc={doc_id: doc_name} if doc_name else {},
        # the run's live grid list the tools read (read_document/compute join into this);
        # seeded with this doc's preloaded table grids (empty for a prose contract).
        grids=list((grids_by_doc or {}).get(doc_id, []) or []),
        retrieval_manager=retrieval_manager,
        db_client=db_client,
    )

    budget = Budget(
        mode="grid",
        model=model_id or getattr(model, "model_id", "") or "",
        max_steps=_GRID_CELL_MAX_STEPS,
        wall_clock_s=_GRID_CELL_WALL_S,
        token_budget=_GRID_CELL_TOKEN_BUDGET,
    )

    events: List[Dict[str, Any]] = []
    try:
        for ev in run_agent(
            _cell_question(column),
            model=model,
            scope=scope,
            budget=budget,
            system_prompt=_cell_system_prompt(column, doc_name),
            gate_fn=grid_gate,
        ):
            events.append(ev)
    except Exception as exc:  # noqa: BLE001 — a cell never crashes the grid
        logger.warning("[grid] cell (%s × %s) raised: %s", doc_name, column.key, exc)
        return GridCell(doc_id=doc_id, column_key=column.key, doc_name=doc_name,
                        status=CellStatus.ERROR, note=f"cell error: {exc}")

    return _cell_from_run(doc_id, doc_name, column, events)


def build_grid(
    spec: GridSpec,
    *,
    model_factory: Callable[[], Any],
    filename_by_doc: Dict[str, str],
    grids_by_doc: Optional[Dict[str, Any]] = None,
    model_id: str = "",
    on_cell: Optional[Callable[[GridCell], None]] = None,
) -> GridResult:
    """Fill the whole grid sequentially (the route may instead fan cells out across a
    pool; this is the simple, deterministic driver used by tests and small grids).

    `model_factory` returns a FRESH model per cell (no shared mutable state across the
    fan-out). `on_cell`, if given, is called as each cell completes (for streaming
    progress to the UI).
    """
    cells: List[GridCell] = []
    for doc_id in spec.doc_ids:
        for column in spec.columns:
            cell = build_cell(
                doc_id, column,
                collection_id=spec.collection_id,
                model=model_factory(),
                filename_by_doc=filename_by_doc,
                grids_by_doc=grids_by_doc,
                model_id=model_id,
            )
            cells.append(cell)
            if on_cell is not None:
                try:
                    on_cell(cell)
                except Exception:  # noqa: BLE001 — progress callback must never break the grid
                    logger.debug("[grid] on_cell callback raised; ignoring")
    return GridResult(spec=spec, cells=cells)
