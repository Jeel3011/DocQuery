"""Non-bypassable output gates (AGENT_CORE_PLAN §3.4).

The model can talk its way past a prompt; it CANNOT talk its way past code. Every draft
the loop produces is bound to the evidence ledger here before it ships:

  1. verify_numbers   — every substantive figure must trace to a ledger cell (or a
                        scaled/round restatement of one). Wraps the promoted spine
                        invariant `figure_traces_to_cells` (already scaling-aware).
  2. verify_citations — every factual sentence must carry a citation marker resolvable
                        to a ledger span/cell. Optional: sample-verify claim↔span
                        entailment with `verifier.verify_claims` (cheap LLM) when one is
                        injected — off in the offline gate so it stays $0.
  3. repair → redact  — ONE feedback turn; a second failure REDACTS the ungated claims
                        (replaces them with an explicit withhold line) and ships the rest.
                        This is C4's binary-withhold, generalized to every answer.

The loop (loop.py) already implements the repair-once-then-redact CONTROL FLOW around
`run_output_gates`; A3 provides the real `run_output_gates`. WRONG-rate stays the
headline metric — a gate that lets an untraced figure through is the product failing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ledger import EvidenceLedger
from .loop import GateOutcome  # reuse the dataclass the loop already imports

# A citation marker like [amzn-2022 p.41], [doc p.3], [MSFT FY22 p.7]. Deliberately
# permissive on the inside — the model's format guidance (§3.5) supplies the shape; the
# gate only requires SOME bracketed source-ish marker on a factual sentence.
_CITE_RE = re.compile(r"\[[^\]]*\bp\.?\s*\d+[^\]]*\]|\[[^\]]+\]")

# A sentence is "factual" (must be cited) if it makes an assertion — we approximate by
# length + the presence of a digit or a finance/legal content word. Short connective
# sentences ("Here is the summary:", "Let me explain.") are exempt.
_CONTENT_WORD = re.compile(
    r"\b(revenue|sales|income|profit|loss|margin|growth|increase|decrease|ratio|"
    r"total|assets|liabilities|equity|cash|expense|cost|clause|section|party|"
    r"shall|agreement|liability|obligation|year|fiscal|quarter)\b",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _strip_citations(text: str) -> str:
    """Remove citation markers before number-tracing. A marker like [amzn-20231231.pdf
    p.62] is PROVENANCE, not a claimed figure — but its embedded digits (e.g. the
    -20231231 in the filename) read as a free-floating number to figure_traces_to_cells,
    which then fails a fully-correct, fully-traced answer and redacts it. Observed live:
    the AWS bridge resolved 2022/80,096 correctly, cited the source, and was redacted
    solely because '-20231231' inside the citation traced to no cell. Strip them first."""
    return _CITE_RE.sub(" ", text or "")


@dataclass
class _CellLike:
    """Minimal CellRef-shaped object for figure_traces_to_cells (it only reads .value)."""
    value: Optional[float]


def _ledger_cells(ledger: EvidenceLedger) -> List[_CellLike]:
    """Values a stated figure may trace to: every read CELL plus every PARAM (e.g. a
    selection-op threshold that came from a verified compute spec). Both are evidence the
    figure is grounded in the executed computation, not invented by the model."""
    out: List[_CellLike] = []
    for e in ledger.entries:
        if e.kind in ("cell", "param") and e.value is not None:
            out.append(_CellLike(value=e.value))
    return out


# ── Gate 1: every figure traces to a cell ───────────────────────────────────────

def verify_numbers(draft: str, ledger: EvidenceLedger) -> Dict[str, Any]:
    """Fail if any substantive figure in the draft traces to no ledger cell."""
    from src.components.brain.monitoring.invariants import figure_traces_to_cells

    cells = _ledger_cells(ledger)
    traceable = _strip_citations(draft)  # citation markers are provenance, not figures
    check = figure_traces_to_cells(traceable, cells)
    # decided=False (no cells to trace against) → can't vouch for any stated figure.
    # If the draft states figures but the ledger is empty, that's a FAIL (free-floating
    # numbers); if the draft states no figures, it passes vacuously.
    if not check.decided:
        has_fig = bool(_has_substantive_figure(traceable))
        if has_fig:
            return {"name": "verify_numbers", "pass": False,
                    "detail": "draft states figures but no source cells were gathered"}
        return {"name": "verify_numbers", "pass": True, "detail": "no figures to verify"}
    return {"name": "verify_numbers", "pass": bool(check.ok), "detail": check.detail}


_NUM_RE = re.compile(r"\$?\s*\d[\d,]*\.?\d*")


def _has_substantive_figure(text: str) -> bool:
    for tok in _NUM_RE.findall(text or ""):
        t = tok.strip()
        if "," in t or "." in t or "$" in t:
            return True
        try:
            if abs(float(t.replace(",", ""))) >= 10000:
                return True
        except ValueError:
            continue
    return False


# ── Gate 2: every factual sentence carries a resolvable citation ────────────────

def _is_factual(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 25:
        return False
    return bool(re.search(r"\d", s)) or bool(_CONTENT_WORD.search(s))


def verify_citations(draft: str, ledger: EvidenceLedger, llm=None) -> Dict[str, Any]:
    """Fail if a factual sentence carries no citation marker.

    Deterministic core (always): each factual sentence must contain a `[... p.N]`-style
    marker. Optional entailment sample (only when `llm` given): the cited sentences are
    checked against ledger spans with `verify_claims` — off in the offline gate.
    """
    sentences = [s for s in _SENT_SPLIT.split(draft or "") if s.strip()]
    uncited = [s.strip() for s in sentences if _is_factual(s) and not _CITE_RE.search(s)]
    if uncited:
        return {"name": "verify_citations", "pass": False,
                "detail": f"uncited factual sentence(s): {uncited[:2]}",
                "uncited": uncited}

    # Optional LLM entailment sample over cited sentences (Phase A: cheap model).
    if llm is not None and not ledger.is_empty():
        try:
            _sample_entailment(sentences, ledger, llm)
        except Exception:  # noqa: BLE001 — the deterministic check already passed; never crash
            pass
    return {"name": "verify_citations", "pass": True, "detail": "all factual sentences cited"}


def _sample_entailment(sentences: List[str], ledger: EvidenceLedger, llm) -> None:
    """Best-effort: build Claims from cited sentences + ledger spans and run verify_claims.
    A failure here does not (in Phase A) hard-fail the gate — it's a sampling signal; the
    deterministic marker check is the binding rule. Kept thin and exception-safe."""
    from src.components.brain.claims import Claim, EvidenceSpan
    from src.components.brain.verifier import verify_claims

    spans = [e for e in ledger.entries if e.kind == "span"]
    if not spans:
        return
    ev = [EvidenceSpan(doc_id=str(s.payload.get("doc") or ""),
                       chunk_id=str(s.payload.get("chunk_id") or ""),
                       verbatim_span=str(s.payload.get("snippet") or s.trace()))
          for s in spans[:5]]
    claims = [Claim(text=snt.strip(), evidence=ev)
              for snt in sentences if _is_factual(snt)][:3]
    if claims:
        verify_claims(claims, llm)


# ── Redaction: withhold ungated claims, ship the rest ───────────────────────────

_WITHHELD_LINE = (
    "\n\n_Note: I removed one or more statements I could not verify against the source "
    "documents. I'm not stating anything I couldn't confirm._"
)


def _redact(draft: str, ledger: EvidenceLedger, failures: List[Dict[str, Any]]) -> str:
    """Strip the offending content and append an explicit withhold line (§3.4).

    Drops: (a) any sentence flagged uncited by verify_citations; (b) any sentence that
    contains an untraced figure (if verify_numbers failed). Conservative — when in doubt
    it removes the sentence rather than ship an unverifiable claim."""
    # `failures` only ever contains gates that FAILED, so a verify_numbers entry being
    # present IS the signal that numbers failed (no `pass` key to consult).
    uncited = set()
    numbers_failed = False
    for f in failures:
        if f.get("name") == "verify_citations":
            uncited |= {u.strip() for u in f.get("uncited", [])}
        if f.get("name") == "verify_numbers":
            numbers_failed = True

    cells = _ledger_cells(ledger)
    from src.components.brain.monitoring.invariants import figure_traces_to_cells

    kept: List[str] = []
    for s in _SENT_SPLIT.split(draft or ""):
        st = s.strip()
        if not st:
            continue
        if st in uncited:
            continue
        st_traceable = _strip_citations(st)  # don't count citation-marker digits as figures
        if numbers_failed and _has_substantive_figure(st_traceable):
            # Keep only if THIS sentence's figures PROVABLY trace. Undecided (e.g. an
            # EMPTY ledger — the model gathered no evidence at all) must redact, not
            # pass: `decided=False` means "nothing to trace against", and shipping an
            # unverifiable figure on absence-of-evidence is the exact fail-open §3.4
            # forbids. Fail closed.
            chk = figure_traces_to_cells(st_traceable, cells)
            if not (chk.decided and chk.ok):
                continue
        kept.append(st)

    body = " ".join(kept).strip()
    return (body + _WITHHELD_LINE) if body else (
        "I could not verify the requested figures against the source documents, so I'm "
        "not stating them. Please check the cited pages directly." )


# ── The entry point the loop calls ──────────────────────────────────────────────

def run_output_gates(draft: str, ledger: EvidenceLedger, *, llm=None) -> GateOutcome:
    """Run all gates over `draft`. Returns a GateOutcome the loop acts on (§3.4).

    pass → ship as-is. fail (first time) → the loop feeds `failures` back for ONE repair.
    fail (after repair) → the loop ships `redacted_draft`. This function is pure/
    deterministic (modulo the optional llm sample) and never raises.
    """
    failures: List[Dict[str, Any]] = []
    for result in (verify_numbers(draft, ledger), verify_citations(draft, ledger, llm=llm)):
        if not result.get("pass", False):
            failures.append({"name": result["name"], "detail": result["detail"],
                             **({"uncited": result["uncited"]} if "uncited" in result else {})})

    if not failures:
        return GateOutcome(passed=True, failures=[])

    return GateOutcome(
        passed=False,
        failures=failures,
        redacted_draft=_redact(draft, ledger, failures),
        abstained=True,
    )
