"""Non-bypassable output gates (AGENT_CORE_PLAN §3.4).

The model can talk its way past a prompt; it CANNOT talk its way past code. Every draft
the loop produces is bound to the evidence ledger here before it ships:

  1. verify_numbers    — every substantive figure must trace to a ledger cell (or a
                         scaled/round restatement of one). Wraps the promoted spine
                         invariant `figure_traces_to_cells` (already scaling-aware).
  2. verify_citations  — every factual sentence must carry a citation marker resolvable
                         to a ledger span/cell. Optional: sample-verify claim↔span
                         entailment with `verifier.verify_claims` (cheap LLM) when one is
                         injected — off in the offline gate so it stays $0.
  3. verify_completeness — multi-entity questions must be addressed in full: each asked
                         entity/sub-question must appear in the answer as a stated result
                         OR an explicit abstain. A silently-dropped entity → ONE repair
                         turn ("you did not address Z"), then the answer ships visibly
                         incomplete. Single-entity questions are a no-op.
  4. repair → redact   — ONE feedback turn; a second failure REDACTS the ungated claims
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

def _untraced_figures(text: str, cells: List[_CellLike]) -> List[str]:
    """The substantive figure-tokens in `text` that figure_traces_to_cells does NOT vouch
    for. Used to ask the de-correlated question: are these grounded in a READ span instead?"""
    from src.components.brain.monitoring.invariants import figure_traces_to_cells
    out: List[str] = []
    for tok in _NUM_RE.findall(text or ""):
        t = tok.strip()
        if not _has_substantive_figure(t):
            continue
        # A figure is "traced" only if it POSITIVELY matches a cell (decided AND ok).
        # With no cells the check is undecided (ok=True by default) — that is NOT a trace,
        # so the figure is untraced and must find a verbatim span home or fail.
        chk = figure_traces_to_cells(t, cells)
        if not (chk.decided and chk.ok):
            out.append(t)
    return out


def _figure_in_spans(figure: str, corpus: str) -> bool:
    """True if a stated figure appears VERBATIM in the read-span corpus. The document
    itself stated this number in prose (e.g. '...approximately 53%...', '$2.3 billion')
    and the agent READ that passage — so the span IS the figure's provenance, the same
    evidence principle the qualitative gate uses (T8). This does NOT loosen the numeric
    moat: a COMPUTED or invented figure has no verbatim home in any read passage; only a
    figure the source explicitly asserts and the agent actually read can match."""
    digits = figure.strip().lstrip("$").strip()
    if not digits:
        return False
    return digits.lower() in corpus


def _figures_grounded_in_spans(figures: List[str], ledger: EvidenceLedger) -> bool:
    """Every untraced figure must appear verbatim in a read span — else fail closed."""
    if not figures:
        return True
    corpus = _span_text(ledger)
    if not corpus:
        return False
    return all(_figure_in_spans(f, corpus) for f in figures)


def verify_numbers(draft: str, ledger: EvidenceLedger) -> Dict[str, Any]:
    """Fail if any substantive figure in the draft traces to neither a ledger cell NOR a
    verbatim read span (a figure the document STATES IN PROSE, read + quoted by the agent)."""
    from src.components.brain.monitoring.invariants import figure_traces_to_cells

    cells = _ledger_cells(ledger)
    traceable = _strip_citations(draft)  # citation markers are provenance, not figures
    check = figure_traces_to_cells(traceable, cells)
    # decided=False (no cells to trace against) → can't vouch via cells. But a figure the
    # SOURCE stated in prose and the agent READ is grounded by that span — check there
    # before failing (kills the false-abstain on a document-stated percentage like 53%).
    if not check.decided:
        untraced = _untraced_figures(traceable, cells)  # cells empty ⇒ all substantive figs
        if untraced and not _figures_grounded_in_spans(untraced, ledger):
            return {"name": "verify_numbers", "pass": False,
                    "detail": "draft states figures grounded in neither a source cell nor a read span"}
        return {"name": "verify_numbers", "pass": True,
                "detail": "no figures to verify" if not untraced else "figures grounded in read spans"}
    if check.ok:
        return {"name": "verify_numbers", "pass": True, "detail": check.detail}
    # Cell-trace failed for SOME figure — give the document-stated ones a span home before redacting.
    untraced = _untraced_figures(traceable, cells)
    if _figures_grounded_in_spans(untraced, ledger):
        return {"name": "verify_numbers", "pass": True,
                "detail": "untraced figures grounded verbatim in read spans"}
    return {"name": "verify_numbers", "pass": False, "detail": check.detail}


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

# Draft scaffolding that is NOT a factual assertion and must never require a citation:
#   - markdown headings  (`# Non-Disclosure Agreement`, `## 8. Governing law`)
#   - explicit input placeholders  (`[INPUT NEEDED: disclosing_party]`) — the OPPOSITE of an
#     invented fact: the model is honestly flagging a gap, per the drafting contract
#   - the abstain line  (`_Insufficient evidence in the vault to draft this section._`)
# Flagging these as "uncited claims" makes a correct, honest draft over-abstain (observed
# live 2026-06-21: an NDA skeleton redacted because the gate flagged the title + brackets).
_SCAFFOLD_RE = re.compile(
    r"^\s*#{1,6}\s"                       # a markdown heading line
    r"|^\s*_?Insufficient evidence",     # the honest-withhold line
    re.IGNORECASE,
)


def _is_factual(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 25:
        return False
    if _SCAFFOLD_RE.search(s):
        return False                      # headings / abstain lines are structure, not claims
    # A sentence carrying an `[INPUT NEEDED: …]` placeholder is the model HONESTLY flagging a
    # gap (the drafting contract), not asserting a fact. Once the placeholder is removed, the
    # remainder is a stub ("The Disclosing Party is .") — verifiable only if it still states a
    # concrete FIGURE. So: a bracketed sentence with no digit in its non-bracket text is not a
    # factual claim. (A bracketed sentence that ALSO states a number stays checkable.)
    has_placeholder = "[INPUT NEEDED" in s.upper()
    stripped = re.sub(r"\[INPUT NEEDED:[^\]]*\]", "", s, flags=re.IGNORECASE).strip()
    if has_placeholder and not re.search(r"\d", stripped):
        return False
    if len(stripped) < 25:
        return False
    return bool(re.search(r"\d", stripped)) or bool(_CONTENT_WORD.search(stripped))


# ── T8: evidence-grounded support (re-root citation on the LEDGER, not on marker shape) ──
#
# The binding question (plans/tool_hard.md §0.2 / T8) is "does the ledger SUPPORT this sentence?",
# NOT "does this sentence carry a `[doc p.N]` substring?". A grounded factual sentence whose marker
# formatting drifted (or which summarizes the retrieved SET rather than one page) was being redacted
# to "I could not verify…" — five reactive special-cases (scaffold, INPUT-NEEDED, qualitative-
# exemption, citation-digit-strip, empty-ledger) each patched one shape of that bug. This is the
# general escape: a NON-NUMERIC factual sentence is SUPPORTED if its salient content actually appears
# in a ledger SPAN's text. Deterministic, $0, no LLM — an EXTERNAL evidence check (T7: never the model
# grading itself; the optional `_sample_entailment` is the stronger lens when an llm is injected).
#
# The NUMERIC MOAT is deliberately untouched: a sentence stating a substantive figure ALWAYS needs a
# marker AND must trace to a cell via verify_numbers — lexical span overlap can NEVER vouch for a
# number (that is the unfixable-by-overlap class). So this only ever RELAXES the marker rule for
# qualitative prose that is provably echoed by gathered evidence; it never loosens a figure.

_STOPWORDS = frozenset(
    "the a an of to in on at by for and or but with as is are was were be been being this that "
    "these those it its their our your his her they them we i you he she which who whom whose "
    "from into over under between each any all some no not nor so than then there here will shall "
    "may might can could would should must have has had do does did a's per via".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&./'-]+")


def _norm_token(t: str) -> str:
    """Lower-case a token and strip leading/trailing word-joining punctuation so a sentence-final
    `Delaware.` matches `delaware` in the span corpus (the `.`/`-`/`'` are kept INSIDE — U.S., R&D —
    but not as trailing noise that would defeat an exact-substring match)."""
    return t.lower().strip(".'-/&")


def _content_tokens(text: str) -> set[str]:
    """The lower-cased salient tokens of a sentence — content words + named entities, stopwords
    and very short tokens dropped. These are what a span must echo for the sentence to be grounded."""
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        t = _norm_token(raw)
        if len(t) >= 3 and t not in _STOPWORDS:
            out.add(t)
    return out


def _entity_tokens(text: str) -> set[str]:
    """The CAPITALIZED, mid-sentence content tokens — a cheap named-entity proxy (Acme, Delaware,
    Reliance). These are the load-bearing SUBJECTS of a claim; generic content words ('agreement',
    'parties') are not. A claim is only grounded if its entities are echoed by the evidence — overlap
    on generic words alone must NOT vouch for an invented entity. A leading-word capital (sentence
    start) is excluded so 'The' / 'This' don't read as entities."""
    ents: set[str] = set()
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        if m.start() == 0:
            continue                          # sentence-initial capital is not an entity signal
        norm = _norm_token(tok)
        if tok[0].isupper() and norm not in _STOPWORDS and len(norm) >= 3:
            ents.add(norm)
    return ents


def _span_text(ledger: EvidenceLedger) -> str:
    """All ledger SPAN text concatenated + lower-cased — the corpus a qualitative claim must echo.
    Spans are the verbatim snippets search_vault gathered; cells/params are numeric and excluded
    here (numbers go through verify_numbers, never this lexical path)."""
    chunks: List[str] = []
    for e in ledger.entries:
        if e.kind == "span":
            chunks.append(str(e.payload.get("snippet") or e.trace() or ""))
    return " ".join(chunks).lower()


def _claim_grounded_in_spans(sentence: str, ledger: EvidenceLedger,
                             min_overlap: float = 0.6) -> bool:
    """True if a NON-NUMERIC factual sentence is supported by the ledger's gathered spans.

    Support = a strong majority of the sentence's salient content tokens appear verbatim in the
    span corpus (default ≥60% — high enough that an unrelated invented claim fails, low enough that
    paraphrase/word-order drift on a genuinely-grounded sentence still passes). Requires a minimum
    of real content tokens so a near-empty sentence can't pass vacuously. This is the EXTERNAL signal
    that lets a marker-less but evidence-backed sentence ship (T8) — fail-closed when no spans exist."""
    corpus = _span_text(ledger)
    if not corpus:
        return False                          # no gathered evidence → cannot vouch (fail closed)
    clean = _strip_citations(sentence)
    toks = _content_tokens(clean)
    if len(toks) < 3:
        return False                          # too little signal to ground on — defer to the marker rule
    # ENTITY GATE (the load-bearing check): every named entity the sentence asserts must appear in
    # the evidence. This is what stops "Initech and Umbrella as the governed parties" from riding in
    # on generic-word overlap ('agreement'/'parties'/'governed') while its actual SUBJECTS are
    # invented — lexical overlap alone can't tell a real claim from one that swaps the entities.
    entities = _entity_tokens(clean)
    if entities and not all(e in corpus for e in entities):
        return False
    present = sum(1 for t in toks if t in corpus)
    return (present / len(toks)) >= min_overlap


def verify_citations(draft: str, ledger: EvidenceLedger, llm=None,
                     allow_qualitative: bool = False) -> Dict[str, Any]:
    """Fail if a factual sentence carries no citation marker.

    Deterministic core (always): each factual sentence must contain a `[... p.N]`-style
    marker. Optional entailment sample (only when `llm` given): the cited sentences are
    checked against ledger spans with `verify_claims` — off in the offline gate.

    QUALITATIVE EXEMPTION (`allow_qualitative`, the BUG-2 fix, 2026-06-26): on the SIMPLE
    whole-draft answer path (run_output_gates — a direct agent answer, NOT a deep report
    or a drafted document), a factual sentence that states NO substantive figure — e.g.
    "This vault contains agreements involving Reliance, Tata, and Infosys." in answer to
    "What companies are in this vault?" — is a non-numeric, list/qualitative claim grounded
    in the document SET the agent retrieved, not in a specific cell on a specific page.
    Forcing a `[doc p.N]` marker onto it made a correct, evidence-backed answer redact to
    "I could not verify…" (observed live on a shared-matter read: 38 spans across 7 docs,
    every sentence redacted). With `allow_qualitative=True`, a sentence with NO substantive
    figure is exempt from the inline-marker rule WHEN the ledger holds supporting evidence.
    The NUMERIC moat is untouched: a stated figure ALWAYS needs a marker (+ verify_numbers
    traces it to a cell). An EMPTY ledger still fails (ungrounded → fail closed).

    DEFAULT OFF: the deep-report / drafting paths (gate_sectioned) keep the strict rule —
    a report section MUST cite its claims; that is the moat for the riskiest surface. The
    exemption is enabled ONLY for the simple direct-answer gate.
    """
    sentences = [s for s in _SENT_SPLIT.split(draft or "") if s.strip()]
    # An `[INPUT NEEDED: …]` placeholder is a GAP marker, not a citation — it must not let an
    # uncited figure ride along ("…but the deposit was 50,000." beside a bracket). Strip the
    # placeholders before asking "does this sentence carry a real citation marker?".
    def _has_citation(s: str) -> bool:
        return bool(_CITE_RE.search(re.sub(r"\[INPUT NEEDED:[^\]]*\]", "", s, flags=re.IGNORECASE)))

    def _needs_marker(s: str) -> bool:
        """Does this factual sentence require an inline citation marker?

        Numeric sentences ALWAYS do (the moat — a stated figure must be cited + traced to a cell).
        A NON-NUMERIC sentence is exempt from the marker rule when it is EVIDENCE-GROUNDED — i.e.
        its salient content is actually echoed by a ledger span (T8: re-root on the ledger, not on
        marker shape). This is the general form of the old qualitative exemption: instead of
        blanket-exempting every non-numeric sentence on the simple path, we require the EXTERNAL
        span-grounding signal — so it (a) applies on BOTH the simple AND the deep/draft path
        (a grounded report sentence no longer redacts on marker drift — S-A), and (b) is STRICTER
        on the simple path (a non-numeric sentence the ledger does NOT support still needs a marker,
        closing the 'any non-numeric prose rides free' gap the blanket flag left open).

        `allow_qualitative` is RETAINED as a wider fallback on the simple direct-answer path only:
        a non-numeric sentence there passes if the ledger merely holds evidence even when this
        sentence's specific tokens didn't cross the overlap bar (e.g. a one-line list answer whose
        salient tokens ARE the entities — already covered — or a terse summary). It NEVER applies to
        a figure and NEVER on an empty ledger."""
        traceable = _strip_citations(re.sub(r"\[INPUT NEEDED:[^\]]*\]", "", s, flags=re.IGNORECASE))
        if _has_substantive_figure(traceable):
            return True                       # a stated figure must always be cited + traced
        # Evidence-grounded (span overlap) → no marker needed, on EITHER path (the T8 principle).
        if _claim_grounded_in_spans(s, ledger):
            return False
        # Simple-answer fallback (BUG-2), NARROWED by T8: a non-numeric sentence on the simple path
        # may pass below the overlap bar (a terse list/summary), BUT it still must not assert a named
        # entity absent from the evidence — an invented-entity claim must never ride the fallback in.
        if allow_qualitative and not ledger.is_empty():
            unknown_entity = any(e not in _span_text(ledger) for e in _entity_tokens(_strip_citations(s)))
            return unknown_entity             # exempt only if it introduces no unknown entity
        return True

    uncited = [s.strip() for s in sentences
               if _is_factual(s) and _needs_marker(s) and not _has_citation(s)]
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


# ── Gate 3: multi-entity completeness (T2) ──────────────────────────────────────
#
# A multi-part question ("compare AMZN, GOOG, MSFT" / "for each of the following…")
# must be addressed in FULL: every asked entity must appear in the answer as a stated
# result OR an explicit abstain. A silently-dropped entity → ONE repair turn, then the
# answer ships visibly incomplete. Single-entity questions → no-op (never false-repair).
#
# Decomposition is DETERMINISTIC (no LLM, $0): pattern-matched for the common legal/
# finance query forms. An unrecognized form → single-entity → no-op. This is the safe
# direction: the two-sided risk is (a) silent under-answer (bad) vs (b) false-positive
# repair on a well-formed answer (also bad). We accept the occasional missed multi-part
# over a false repair.

# Patterns that introduce a multi-entity set in the question.
_COMPARE_RE = re.compile(
    r"\b(?:compare|versus|vs\.?|difference between|contrast)\b",
    re.IGNORECASE,
)
_FOR_EACH_RE = re.compile(
    r"\bfor each (?:of (?:the )?)?(?:the (?:following\b)?)?",
    re.IGNORECASE,
)
# A list item after "compare X, Y, Z" or "for each of X, Y, Z" — comma/and-separated
# proper tokens (capitalized or ALL-CAPS abbreviations like AMZN).
_LIST_SPLIT_RE = re.compile(r"\s*(?:,\s*(?:and\s+)?|(?:\s+and\s+))\s*")
# What the answer says when it explicitly can't address something.
_EXPLICIT_ABSTAIN_RE = re.compile(
    r"\b(?:could not|couldn't|unable to|not available|not found|no data|"
    r"insufficient evidence|cannot verify|not in the vault|not ingested)\b",
    re.IGNORECASE,
)


def _extract_entities(question: str) -> List[str]:
    """Return the multi-entity list from `question`, or [] for a single-entity question.

    Handles the two commonest forms:
      • "compare AMZN, GOOG, and MSFT revenue in FY2023"
      • "for each of the following clauses: governing law, payment terms, liability"
    Returns an empty list when the question doesn't match a recognized multi-entity
    form — the gate then no-ops (safe-default: no false repair on an unrecognized form).
    """
    q = question.strip()

    # Form 1 — "compare X, Y, Z …"
    if _COMPARE_RE.search(q):
        # grab the part after the compare keyword up to the first verb/stop word
        m = _COMPARE_RE.search(q)
        tail = q[m.end():].strip()
        # clip at first sentence-ending keyword that signals the entities are done
        tail = re.split(r"\b(?:in|for|from|during|between|with|and their|revenue|margin"
                        r"|profit|sales|income|rate|growth|clause)\b", tail,
                        maxsplit=1, flags=re.IGNORECASE)[0]
        ents = [e.strip() for e in _LIST_SPLIT_RE.split(tail) if e.strip()]
        if len(ents) >= 2:
            return ents

    # Form 2 — "for each of [the following] X, Y, Z"
    m2 = _FOR_EACH_RE.search(q)
    if m2:
        tail2 = q[m2.end():].strip()
        # strip a label prefix like "clauses:" / "following clauses:" / "items:" that
        # precedes the actual list — one or two non-comma words followed by a colon.
        tail2 = re.sub(r"^(?:\w+\s+)?\w+\s*:\s*", "", tail2)
        # clip at sentence end or a stop phrase
        tail2 = re.split(r"\s*[.?!]\s*|\b(?:provide|give|state|list|show|tell)\b",
                         tail2, maxsplit=1, flags=re.IGNORECASE)[0]
        ents2 = [e.strip() for e in _LIST_SPLIT_RE.split(tail2) if e.strip()]
        if len(ents2) >= 2:
            return ents2

    return []


def _entity_addressed(entity: str, answer: str) -> bool:
    """True if `entity` appears in the answer as a stated result or an explicit abstain.

    We check two things:
      (a) the entity name (case-insensitive substring) appears in the answer, AND
      (b) EITHER it is near a substantive result-word OR near an explicit abstain phrase.
    A match on just the name without either signal means the entity was mentioned in
    passing (e.g. the question was repeated back) but not answered — that still fails.
    """
    if not entity:
        return True
    low_ans = answer.lower()
    low_ent = entity.lower()

    # Find every occurrence of the entity name in the answer.
    for m in re.finditer(re.escape(low_ent), low_ans):
        # Look at a ±200-char window around the match.
        start = max(0, m.start() - 200)
        end = min(len(low_ans), m.end() + 200)
        window = low_ans[start:end]
        # Does the window contain a numeric result or a result-signal word?
        has_result = bool(re.search(r"\d[\d,.]*", window)) or bool(re.search(
            r"\b(?:is|was|were|are|has|have|had|totaled|reached|grew|declined|"
            r"reported|recorded|found|agreed|states|provides|requires|governed|"
            r"includes?|covers?|specifies?)\b", window, re.IGNORECASE))
        if has_result:
            return True
        # OR does the window contain an explicit abstain phrase?
        if _EXPLICIT_ABSTAIN_RE.search(window):
            return True
    return False


def verify_completeness(question: str, draft: str) -> Dict[str, Any]:
    """Fail if a multi-entity question has at least one entity silently omitted.

    Returns a gate result dict with the same shape as verify_numbers/verify_citations.
    `dropped` key lists the entity names that were omitted (used by the repair message).
    Single-entity (or unrecognized-form) questions always pass (no-op path).
    """
    entities = _extract_entities(question)
    if len(entities) < 2:
        return {"name": "verify_completeness", "pass": True,
                "detail": "single-entity question — completeness no-op"}

    dropped = [e for e in entities if not _entity_addressed(e, draft or "")]
    if dropped:
        return {"name": "verify_completeness", "pass": False,
                "detail": (f"answer silently omits: {dropped}; "
                           f"state a result or explicitly abstain on each"),
                "dropped": dropped}
    return {"name": "verify_completeness", "pass": True,
            "detail": f"all {len(entities)} entities addressed"}


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


# ── Deep Analysis: per-section gating (G5) ───────────────────────────────────────

_SECTION_SPLIT = re.compile(r"(?m)^(?=##\s)")  # split BEFORE each `## ` header, keep it
# A section the model intentionally abstained on (the deep prompt's withhold line). It
# carries no figures and no claim — passing it through is correct (honest abstention),
# NOT a gate bypass. Matched loosely so minor wording/emphasis variations still count.
_WITHHELD_SECTION_RE = re.compile(r"insufficient evidence", re.IGNORECASE)


def _split_sections(draft: str) -> List[str]:
    """Split a deep report into [preamble?, ## section, ## section, …].

    The leading chunk before the first `##` (the executive summary) is its own section so
    it is gated too. Empty fragments are dropped. A report with no `##` headers is ONE
    section — gate_sectioned then behaves exactly like run_output_gates."""
    parts = [p for p in _SECTION_SPLIT.split(draft or "") if p.strip()]
    return parts or ([draft] if (draft or "").strip() else [])


def gate_sectioned(draft: str, ledger: EvidenceLedger, *, llm=None,
                   question: str = "") -> GateOutcome:
    """Deep-mode output gate: bind EACH section to the ledger independently (G5 §2.B2).

    Same verification intelligence as `run_output_gates` (verify_numbers +
    verify_citations + verify_completeness + the per-claim `_redact`), applied
    section-by-section so one unsupported section is redacted to a VISIBLE withhold line
    while the grounded sections ship intact with their citations. This is the moat for
    the riskiest surface: a long report can abstain a section, never silently assert it.

    Returns a GateOutcome whose `redacted_draft` is the REASSEMBLED report (every section,
    failing ones redacted). `passed` is True only if every section passed as-is.
    """
    sections = _split_sections(draft)
    if len(sections) <= 1:
        # No real sectioning → identical to the whole-draft gate (don't special-case).
        return run_output_gates(draft, ledger, llm=llm, question=question)

    rebuilt: List[str] = []
    all_failures: List[Dict[str, Any]] = []
    any_failed = False

    for sec in sections:
        # An intentionally-withheld section is already honest (no figures, no claim) —
        # ship it verbatim; redacting it would just rewrite one withhold line as another.
        if _WITHHELD_SECTION_RE.search(sec):
            rebuilt.append(sec.rstrip())
            continue

        # Verify the BODY, not the whole section: a `## Title` line is never a factual
        # sentence or a figure, and including it would PREFIX the header onto the first
        # sentence (split on `.!?`, not `\n`) — so `verify_citations`'s `uncited` strings
        # wouldn't match what `_redact` (which iterates body sentences) removes, and the
        # offending claim would survive. Strip the header, gate the body, re-attach the
        # header. A preamble (no `## `) has empty header and is gated whole.
        if sec.lstrip().startswith("##"):
            header, _, body = sec.partition("\n")
            header = header.strip()
        else:
            header, body = "", sec

        failures: List[Dict[str, Any]] = []
        for result in (verify_numbers(body, ledger), verify_citations(body, ledger, llm=llm)):
            if not result.get("pass", False):
                failures.append({"name": result["name"], "detail": result["detail"],
                                 **({"uncited": result["uncited"]} if "uncited" in result else {}),
                                 **({"dropped": result["dropped"]} if "dropped" in result else {})})

        if not failures:
            rebuilt.append(sec.rstrip())
            continue

        # This section failed → redact its claims (from the body), preserving the
        # `## header` so the report keeps its shape and the withheld section is visible
        # in place with an explicit withhold line.
        any_failed = True
        all_failures.extend(failures)
        redacted_body = _redact(body, ledger, failures).rstrip()
        rebuilt.append(f"{header}\n\n{redacted_body}".rstrip() if header else redacted_body)

    report = "\n\n".join(rebuilt).strip()

    # T2: completeness check on the WHOLE assembled report (not per-section — the question
    # asks about multiple entities across the WHOLE answer; each section may cover one entity).
    completeness = verify_completeness(question, report)
    if not completeness.get("pass", True):
        any_failed = True
        all_failures.append({"name": completeness["name"], "detail": completeness["detail"],
                              "dropped": completeness.get("dropped", [])})

    if not any_failed:
        return GateOutcome(passed=True, failures=[])
    return GateOutcome(
        passed=False,
        failures=all_failures,
        redacted_draft=report,
        abstained=True,
    )


# ── The entry point the loop calls ──────────────────────────────────────────────

def run_output_gates(draft: str, ledger: EvidenceLedger, *, llm=None,
                     question: str = "") -> GateOutcome:
    """Run all gates over `draft`. Returns a GateOutcome the loop acts on (§3.4).

    pass → ship as-is. fail (first time) → the loop feeds `failures` back for ONE repair.
    fail (after repair) → the loop ships `redacted_draft`. This function is pure/
    deterministic (modulo the optional llm sample) and never raises.
    """
    # S-A (the latent moat hole tracked in tool_hard.md §T8): a single-section draft of the
    # form `## Heading\n\n<claim>` would otherwise BYPASS gating entirely — `_SENT_SPLIT` splits
    # on `.!?`, not `\n`, so the heading glues onto the first body sentence; the glued blob starts
    # with `##`, `_SCAFFOLD_RE` matches, and `_is_factual` rejects the whole thing as scaffold →
    # an ungrounded claim under a lone heading ships unchecked (verified live 2026-06-27). The
    # per-section path (gate_sectioned) already strips the header before gating for exactly this
    # reason; do the SAME here so the simple/standard path is not the soft underbelly. Strip a
    # leading heading line, gate the body, re-attach the header to the (possibly redacted) body.
    header, body = "", draft or ""
    _stripped = (draft or "").lstrip()
    if _SCAFFOLD_RE.match(_stripped) and _stripped[:1] == "#":
        _h, _, _rest = _stripped.partition("\n")
        # Only peel a markdown HEADING line (not the `_Insufficient evidence` withhold line,
        # which carries no claim and must stay with the body it explains).
        if _h.lstrip().startswith("#") and _rest.strip():
            header, body = _h.strip(), _rest

    failures: List[Dict[str, Any]] = []
    # Simple direct-answer path: allow a non-numeric, evidence-grounded qualitative answer
    # (e.g. "what companies are in this vault?") through without a per-cell marker (BUG-2).
    # The numeric moat (verify_numbers + the figure-marker rule) is unaffected.
    for result in (verify_numbers(body, ledger),
                   verify_citations(body, ledger, llm=llm, allow_qualitative=True),
                   verify_completeness(question, body)):
        if not result.get("pass", False):
            failures.append({"name": result["name"], "detail": result["detail"],
                             **({"uncited": result["uncited"]} if "uncited" in result else {}),
                             **({"dropped": result["dropped"]} if "dropped" in result else {})})

    if not failures:
        return GateOutcome(passed=True, failures=[])

    redacted_body = _redact(body, ledger, failures)
    return GateOutcome(
        passed=False,
        failures=failures,
        redacted_draft=(f"{header}\n\n{redacted_body}".rstrip() if header else redacted_body),
        abstained=True,
    )
