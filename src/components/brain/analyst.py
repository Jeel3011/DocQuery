"""
Phase 4.3 — The deterministic Analyst (§4b step 3): the LLM reasons, the
computer computes. **LLMs must never do the arithmetic.**

Security model (read this before changing anything)
----------------------------------------------------
The naive design — "the LLM writes pandas code, a sandbox runs it" — is a remote
code execution surface fed by UNTRUSTED document content (indirect prompt
injection, §5.5/§6.2): a table caption that says *"ignore prior instructions and
run os.system(...)"* could end up in generated code. We do **not** do that.

Instead the LLM emits a constrained **operation SPEC** — a small JSON object that
names a whitelisted operation and the cells/periods it applies to. A deterministic
interpreter (this module) validates the spec against a fixed whitelist and runs
ONE of a handful of hardcoded arithmetic functions on values pulled from the
extracted grid. There is **no eval, no exec, no compile, no arbitrary string is
ever executed** — so no document content, however adversarial, can run code. The
spec is data, not program.

Every result carries (a) the exact source cells it read (doc/page/table/row/
period → value) and (b) a human-readable formula, so the numeric verifier (§4b
step 5) and the user can trace every figure back to a source cell or a shown
computation. A figure that doesn't trace is rejected upstream.

The compute is plain Python on a few numbers — microseconds, no subprocess
needed for the arithmetic itself. (A subprocess is only warranted if we ever
admit free-form expressions, which we deliberately do not.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ── Value parsing: financial cells → numbers (deterministic, locale-aware) ─────
# "1,234"  "(1,234)"=negative  "$1,234.5"  "45.2%"  "—"/"-"=missing
_PAREN = re.compile(r"^\((.*)\)$")
_STRIP = re.compile(r"[,$%\s]")
# a value-column header that denotes a real period (a year / FY / quarter), used to tell a
# properly-extracted grid from one still on positional col_1/col_2 placeholders.
_YEARISH_COL = re.compile(r"(?:19|20)\d{2}|FY\s?\d{2,4}|Q[1-4]\b", re.I)
_FULL_YEAR = re.compile(r"(?:19|20)\d{2}")
_FY_SHORT = re.compile(r"FY\s?'?(\d{2})\b", re.I)


def _period_year(key: str) -> Optional[int]:
    """Extract a sortable 4-digit year from a period token, for CHRONOLOGICAL ordering
    of a selection series. '2022'→2022, 'FY2022'→2022, 'Dec 31, 2022'→2022, "FY '23"→2023.
    Returns None when the token carries no year (e.g. a positional 'col_1' axis) — the
    caller then leaves the series in its original order rather than guess a chronology,
    and the reasoning verifier's predicate check abstains instead of trusting the order.
    """
    s = str(key)
    m = _FULL_YEAR.search(s)
    if m:
        return int(m.group())
    m = _FY_SHORT.search(s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 70 else 1900 + yy
    return None


class CellError(ValueError):
    """A cell could not be parsed as a number, or a spec referenced a missing cell."""


class AmbiguousCell(CellError):
    """A reference resolved to DIFFERENT values across tables — abstain, don't guess.

    The correct-or-abstain contract (§4a/§4b): when the same (section,label,period)
    yields conflicting numbers across candidate tables, we flag for a human rather
    than confidently pick one. A wrong number that's flagged is acceptable; a wrong
    number stated confidently is the failure this prevents.
    """


def parse_cell(raw: Any) -> float:
    """Parse a financial table cell into a float. Raises CellError if not numeric.

    Parentheses → negative (accounting convention). Strips $, commas, %, spaces.
    A bare dash / em-dash / empty is 'no value' and raises (callers decide policy).
    """
    if raw is None:
        raise CellError("empty cell")
    s = str(raw).strip()
    if s in ("", "-", "—", "–", "n/a", "N/A", "NM"):
        raise CellError(f"non-numeric cell: {s!r}")
    if s.endswith("%"):
        s = s[:-1].strip()  # '%' is a presentation MARK ('(1.1)%' → '(1.1)'), not the number
    neg = False
    m = _PAREN.match(s)
    if m:
        neg = True
        s = m.group(1)
    s = _STRIP.sub("", s)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        raise CellError(f"non-numeric cell: {raw!r}")
    val = float(s)
    return -val if neg else val


# ── Source-cell provenance (every number traces here) ─────────────────────────

@dataclass
class CellRef:
    """A single source cell the Analyst read, fully addressed for verification."""
    doc: Optional[str]       # filename
    page: Optional[int]
    table_id: Optional[str]
    section: str
    label: str
    period: str
    raw: str                 # the verbatim cell text, e.g. "90,757"
    value: float             # parsed numeric value

    def trace(self) -> str:
        loc = f"{self.doc or '?'} p{self.page or '?'}"
        sec = f"{self.section}/" if self.section else ""
        return f"{sec}{self.label} [{self.period}] = {self.raw} ({loc})"


@dataclass
class ComputeResult:
    """The deterministic output of one spec: value + formula + the cells it used.

    For the arithmetic ops `value` is the computed number. For the SELECTION ops
    (§3.3 — argmax/argmin/first_exceeds/last_below/rank/filter) the question is
    "WHICH period/entity?", not "what number?": those set `binding` to the resolved
    answer (a period like "2022" or an entity like "AWS") while `value` carries the
    winning cell's number (so numeric grounding still works downstream). `binding`
    is the pivot the executive layer (Layer 3) will bind a variable to; `candidates`
    is every cell the op actually considered — the completeness trail the reasoning
    verifier (Layer 4) checks ("did argmax see all 3 companies?").
    """
    op: str
    value: Optional[float]
    formula: str                          # human-readable, e.g. "(90757 − 80096) / 80096 × 100"
    unit: str = ""                        # "%", "$M", "x", ""
    cells: List[CellRef] = field(default_factory=list)
    error: Optional[str] = None
    binding: Optional[str] = None         # selection result: the resolved period/entity
    candidates: List[CellRef] = field(default_factory=list)  # everything the op scanned

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def display(self) -> str:
        if not self.ok:
            return f"[compute error: {self.error}]"
        # selection ops answer "which?" — show the resolved pivot, with its number.
        if self.binding is not None:
            num = f"{self.value:,.2f}".rstrip("0").rstrip(".") if self.value % 1 else f"{self.value:,.0f}"
            return f"{self.binding} ({num})"
        v = self.value
        if self.unit == "%":
            return f"{v:,.1f}%"
        if self.unit == "x":
            return f"{v:,.2f}x"
        return f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"


# ── Grid access ────────────────────────────────────────────────────────────────

class Grid:
    """A typed view over one extracted table's normalized JSON (the Analyst's input).

    Wraps the dict produced by table_extraction.ExtractedTable.to_metadata() plus
    provenance (doc/page) so resolved cells carry full source addressing.
    """

    def __init__(self, table_json: Dict[str, Any], *, doc: Optional[str] = None,
                 page: Optional[int] = None):
        self.headers: List[str] = table_json.get("headers", [])
        self.rows: List[Dict[str, Any]] = table_json.get("rows", [])
        self.periods: List[str] = table_json.get("periods", [])
        self.units: Optional[str] = table_json.get("units")
        self.table_id: Optional[str] = table_json.get("table_id")
        # LLM discriminative summary (statement type + value kind) — the strongest
        # signal for the spec-writer to pick the right grid among near-twins. Empty
        # for tables ingested before the summary fix; grid_catalog falls back then.
        self.summary: str = table_json.get("summary", "") or ""
        self.doc = doc
        self.page = page

    def value_columns(self) -> List[str]:
        cols = [h for h in self.headers if h not in ("section", "label")]
        # When a period axis is known, restrict to it so derived columns (e.g.
        # "pct_change" printed alongside the year columns) are never scanned as a
        # period. Backward-compatible: a well-formed grid has periods == its value
        # columns (no change); a fully-unlabelled grid has periods == [] → all cols.
        if self.periods:
            kept = [h for h in cols if h in self.periods]
            if kept:
                return kept
        return cols

    def find_row(self, label: str, section: str = "") -> Optional[Dict[str, Any]]:
        """Resolve a row by label (and optional section), case/space-insensitive.

        Section authority (Layer 0 §2): when a section IS requested it is BINDING —
        we match the label only among rows that genuinely belong to that section, and
        do NOT fall through to other sections of this grid. So asking for "AWS / Net
        sales" never returns a "Consolidated / Net sales" (574,785 vs 90,757). This is
        safe only because §1 now scopes rows correctly by indentation; before that the
        loose fallthrough was masking a wrong-section match as a "harmless" abstain.
        When NO section is requested, behaviour is the original label-only resolution
        (exact, then contains). A value is never invented — we only relax WHICH row.
        """
        def norm(s: str) -> str:
            # lower, collapse whitespace, drop bracketing/quotes the LLM may add
            # (e.g. "[Net Sales]" → "net sales") so spec labels match grid labels
            # without the model having to reproduce punctuation exactly.
            s = re.sub(r"[\[\]\"'`()]", " ", (s or "").lower())
            return re.sub(r"\s+", " ", s).strip()

        nlabel, nsec = norm(label), norm(section)

        def label_match(r) -> bool:
            rl = norm(r.get("label", ""))
            return rl == nlabel or nlabel in rl or rl in nlabel

        def in_section(r) -> bool:
            # The row belongs to the requested section. A row with an EMPTY section
            # never matches a requested (non-empty) section — otherwise every
            # section-less row would satisfy any request via "" being a substring of
            # everything. The row's LABEL is also accepted because the LLM sometimes
            # puts the segment name in the label (e.g. label "AWS net sales").
            rsec = norm(r.get("section", ""))
            sec_hit = bool(rsec) and (nsec in rsec or rsec in nsec)
            label_hit = nsec in norm(r.get("label", ""))
            return sec_hit or label_hit

        # Section requested → section is AUTHORITATIVE: match label ONLY among rows in
        # that section (exact label wins over contains). No cross-section fallthrough —
        # if this grid has no such row, it simply has no answer (resolve() tries the
        # next grid or abstains), never a wrong-section value.
        if nsec:
            in_sec = [r for r in self.rows if in_section(r)]
            for r in in_sec:
                if norm(r.get("label", "")) == nlabel:
                    return r
            for r in in_sec:
                if label_match(r):
                    return r
            return None

        # No section requested → original label-only resolution (exact, then contains).
        for r in self.rows:
            if norm(r.get("label", "")) == nlabel:
                return r
        for r in self.rows:
            if label_match(r):
                return r
        return None

    def cell(self, label: str, period: str, section: str = "") -> CellRef:
        """Resolve (section,label,period) to a fully-addressed, parsed CellRef.

        Raises CellError if the row/period is missing or the value isn't numeric —
        the caller turns that into a graceful compute error, never a guess.
        """
        row = self.find_row(label, section)
        if row is None:
            raise CellError(f"row not found: section={section!r} label={label!r}")
        # resolve the period column: exact, else positional if period is col_N
        col = period
        if period not in self.value_columns():
            raise CellError(f"period not found: {period!r} (have {self.value_columns()})")
        raw = row.get(col, "")
        value = parse_cell(raw)
        return CellRef(
            doc=self.doc, page=self.page, table_id=self.table_id,
            section=row.get("section", ""), label=row.get("label", ""),
            period=col, raw=str(raw), value=value,
        )


# ── The operation whitelist (the ONLY arithmetic that can run) ─────────────────
#
# Each spec selects exactly one of these. The interpreter resolves the spec's cell
# references to CellRefs, then calls the matching hardcoded function. Adding an op
# means adding a reviewed function here — never widening to free-form expressions.

WHITELISTED_OPS = {
    "value",          # echo a single cell (the simplest "compute": just trace it)
    "delta",          # absolute change: to − from
    "growth_pct",     # percent change: (to − from) / |from| × 100
    "sum",            # sum of N cells
    "difference",     # a − b (two named cells)
    "ratio",          # a / b  (e.g. margin = income / revenue)
    "margin_pct",     # a / b × 100 (a ratio expressed as a percentage)
    "average",        # mean of N cells
    "cagr_pct",       # compound annual growth rate across `periods` years
    # ── Selection ops (Layer 1, §3.3): the executive layer's deterministic
    # comparisons over a COMPLETE series. They answer "which period/entity?"
    # (the pivot), not "what number?" — the wrong-year / wrong-company error
    # class becomes COMPUTED, never guessed. Same spec-not-code security model.
    "argmax",         # entity/period with the highest value over a series
    "argmin",         # entity/period with the lowest value over a series
    "first_exceeds",  # EARLIEST ordered period whose value crosses a threshold (>)
    "last_below",     # LATEST ordered period whose value stays under a threshold (<)
    "rank",           # ordered list (descending) of a series — top-N selection
    "filter",         # the subset of a series satisfying a comparison predicate
}


def _fmt(n: float) -> str:
    """Format a number inside a formula compactly (no thousands separators)."""
    return f"{n:g}"


def _doc_match(grid_doc: Optional[str], want: str) -> bool:
    """Does a grid's document label satisfy a requested `doc` scope?

    Normalized substring either way, so "msft-10k_20220630" matches
    "msft-10k_20220630.htm.pdf" without the caller reproducing the extension.
    A grid with NO doc label never matches an explicit scope (fail closed).
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()
    g, w = norm(grid_doc or ""), norm(want)
    if not g or not w:
        return False
    return w in g or g in w


def _norm_section_match(row: Dict[str, Any], section: str) -> bool:
    """True if `row` genuinely belongs to the requested section (strict pass).

    Compares the requested section against the row's section field (the row's
    label is NOT accepted here — that looseness is only for the fallback pass),
    so asking for "AWS" never matches a "Consolidated" row.
    """
    def norm(s: str) -> str:
        s = re.sub(r"[\[\]\"'`()]", " ", (s or "").lower())
        return re.sub(r"\s+", " ", s).strip()
    nsec = norm(section)
    rsec = norm(row.get("section", ""))
    if not nsec:
        return True
    return nsec in rsec or rsec in nsec


# ── Totals-consistency (§5.4 remaining kernel work — used by grounding §5.1 + ──
#    invariants §5.5). Deterministic, VALUE-based, NO label list.
#
# The §2.1 failure ("total revenue" resolves to a component line like "Gross
# margin") is caught structurally: a row USED AS a total must approximately equal
# the sum of its sibling component rows in the same section/period. We never decide
# "is this a total?" from a hardcoded label — we ask the values: does this row equal
# Σ of the other numeric rows that plausibly compose it? If a candidate row is a
# strict SUBSET/child (its value is materially smaller than the section's largest
# numeric row at that period), it fails the total check. This reads structure, not
# content, so it generalizes to any indented statement (and to legal sub-clauses).

_AGGREGATE_RE = re.compile(
    r"\b(total|consolidated|grand total|net (sales|revenue|income))\b", re.I
)


def _is_aggregate_request(label: str) -> bool:
    """Does this requested label SIGNAL that the caller wants an aggregate/total row?

    Advisory only: it merely decides whether to APPLY the value-based aggregation
    guard (`looks_like_total`) during resolution — it never picks a number. A label
    like "total revenue" or "consolidated net sales" → True. This reads the request's
    intent from its words; the GRID's values then decide what actually is a total.
    """
    return bool(_AGGREGATE_RE.search(label or ""))


def is_lineitem_label(label: str, max_words: int = 6) -> bool:
    """Is `label` a plausible financial LINE-ITEM, vs a prose/narrative fragment?

    A grounding safety check (§5.1): when geometry over a NARRATIVE page produces a
    pseudo-table, a row's "label" is a full sentence and its "value" is a number that
    leaked out of the prose (e.g. "Microsoft Cloud revenue Revenue from Azure and other
    cloud services, Office" with value 365 — the '365' of "Microsoft 365"). Binding a
    figure from such a row is a confident-wrong. Real line-items are short noun
    phrases ("Total revenue", "Net income"); narrative fragments are long sentences.

    Structural (reads form, not content; same discipline as Layer 0 `_is_section_header`
    rejecting >6-word headers): reject labels that exceed `max_words` OR read like a
    sentence (a comma mid-label, or a trailing period). No keyword list.
    """
    s = (label or "").strip()
    if not s:
        return False
    if len(s.split()) > max_words:
        return False
    if "," in s:                       # a comma signals an enumerated prose clause
        return False
    if s.endswith(".") and not s.endswith(".."):  # a sentence, not a label
        return False
    return True


_PCT_SECTION = re.compile(r"\b(percent|percentage|growth|rate)\b", re.I)


def row_value_kind(grid: "Grid", row: Dict[str, Any]) -> str:
    """'percent' if this row's cells are a percentage PRESENTATION, else 'currency'.

    Structural + value-based (§precision): (a) any raw cell carries a '%' mark;
    (b) the row's SECTION reads as a percent presentation ("Percent of Net Sales",
    "Year-over-year Percentage Growth", a rate table) AND every numeric value is
    %-scale (|v| <= 100). Used as a FILTER among same-specificity candidates — a
    dollar request never binds the 14.9 '% of net sales' twin of an 85,622 row —
    never as a picker: a reference matching ONLY percent rows still binds them."""
    pct_marks = 0
    vals: List[float] = []
    for p in grid.value_columns():
        raw = str(row.get(p, "") or "").strip()
        if not raw:
            continue
        if "%" in raw:
            pct_marks += 1
        try:
            vals.append(parse_cell(raw))
        except CellError:
            continue
    if pct_marks:
        return "percent"
    sec = str(row.get("section", "") or "")
    if vals and _PCT_SECTION.search(sec) and all(abs(v) <= 100.0 for v in vals):
        return "percent"
    return "currency"


def _norm_label_eq(a: str, b: str) -> bool:
    """Case/space/punct-insensitive label equality (find_row's norm semantics)."""
    def n(s: str) -> str:
        s = re.sub(r"[\[\]\"'`()]", " ", (s or "").lower())
        return re.sub(r"\s+", " ", s).strip()
    return n(a) == n(b)


def _label_suggestions(label: str, grids: List["Grid"], k: int = 3) -> str:
    """Closest real line-item labels to `label` across `grids` (did-you-mean for
    self-healing errors; suggestion only — the model must re-issue the call)."""
    import difflib
    pool: Dict[str, str] = {}
    for g in grids:
        for r in g.rows:
            lab = (r.get("label") or "").strip()
            if lab and is_lineitem_label(lab) and lab not in pool:
                pool[lab] = (r.get("section") or "").strip()
    best = difflib.get_close_matches(label or "", list(pool), n=k, cutoff=0.5)
    return "; ".join(f"'{b}' [{pool[b]}]" if pool[b] else f"'{b}'" for b in best)


def _section_rows(grid: "Grid", section: str) -> List[Dict[str, Any]]:
    """Rows of `grid` that genuinely belong to `section` (strict, value-bearing)."""
    return [r for r in grid.rows if _norm_section_match(r, section or "")]


def looks_like_total(grid: "Grid", row: Dict[str, Any], period: str,
                     rel_tol: float = 0.02) -> Optional[bool]:
    """Is `row` plausibly a TOTAL at `period` (vs a component child)? Structural.

    Returns True  — the row equals (within rel_tol) the sum of the OTHER numeric
                    rows in its section at this period (a genuine total/subtotal),
                    OR it is the single largest numeric row by a clear margin
                    (a lone total with components elsewhere).
            False — the row is materially SMALLER than the section's largest numeric
                    row at this period (it is a component/child, not the total).
            None  — undecidable (period missing, too few numeric siblings) → the
                    caller must not treat absence of evidence as a level failure.

    No label list, no hardcoding — the values decide. `rel_tol` absorbs rounding
    and the "≈" in "total ≈ Σ children".
    """
    try:
        target = parse_cell(row.get(period, ""))
    except CellError:
        return None
    section = row.get("section", "")
    siblings = _section_rows(grid, section)
    # numeric values of the OTHER rows at this period (the candidate components)
    others: List[float] = []
    for r in siblings:
        if r is row:
            continue
        try:
            others.append(parse_cell(r.get(period, "")))
        except CellError:
            continue
    if len(others) < 1:
        return None  # nothing to compare against — undecidable

    # (a) does target ≈ Σ of a plausible component set? The simplest, strongest
    #     signal: target equals the sum of ALL other numeric siblings. (Statements
    #     often interleave non-additive lines, so we also accept being within tol of
    #     that sum — a near-match still indicts a *component* answer being wrong.)
    sib_sum = sum(others)
    if sib_sum != 0 and abs(target - sib_sum) <= rel_tol * abs(sib_sum):
        return True

    # (b) magnitude test: a total is the LARGEST (or tied-largest) numeric line in
    #     its section at this period. If a strictly larger sibling exists, `row` is
    #     a component, not the total → False. If `target` dominates, → True.
    biggest = max(others + [target], key=abs)
    if abs(target) >= abs(biggest) - 1e-9:
        return True
    # there is a materially larger sibling ⇒ this row is a child line
    if abs(target) < abs(biggest) * (1 - rel_tol):
        return False
    return None


def compute(spec: Dict[str, Any], grids: List[Grid]) -> ComputeResult:
    """Interpret ONE validated spec against the available grids. Deterministic.

    The spec shape (all string/number/list — never code):
      {"op": "growth_pct",
       "table": <int grid index, optional — else search all grids>,
       "row": {"section": "AWS", "label": "Net sales"},
       "from_period": "2022", "to_period": "2023"}
      {"op": "sum", "cells": [{"row": {...}, "period": "2023"}, ...]}
      {"op": "ratio", "numerator": {"row": {...}, "period": "2023"},
                      "denominator": {"row": {...}, "period": "2023"}}

    Returns a ComputeResult with value + formula + the exact CellRefs used, or an
    error string (never raises into the caller — a bad spec degrades gracefully).
    """
    op = spec.get("op")
    if op not in WHITELISTED_OPS:
        return ComputeResult(op=str(op), value=None, formula="",
                             error=f"operation not allowed: {op!r}")

    def candidate_grids() -> List[Grid]:
        """Grids to search, named-table FIRST then all others as fallback.

        If the LLM named a `table`, we try it first — but we ALWAYS fall back to
        the remaining grids if the row/period isn't there, because the model often
        mis-indexes among near-identical tables (a $ table vs its growth-% twin).
        The deterministic resolver, not the model, has the last word on placement.
        """
        idx = spec.get("table")
        if isinstance(idx, int) and 0 <= idx < len(grids):
            return [grids[idx]] + [g for i, g in enumerate(grids) if i != idx]
        return grids

    def _ref_doc(ref: Dict[str, Any], row: Dict[str, Any]) -> str:
        """The document scope for one cell reference: ref-level wins, then the row,
        then a spec-level default (so a whole growth/ratio spec scopes once)."""
        return str(ref.get("doc") or row.get("doc") or spec.get("doc") or "")

    def scoped_grids(doc: str) -> List[Grid]:
        """candidate_grids() narrowed to `doc` when one is requested (BUG-1 fix).

        In a multi-document scope every cross-issuer-common label ("total revenue",
        "net income") otherwise resolves in several issuers' statements at once and
        ALWAYS abstains as ambiguous. An explicit doc must scope to that document
        only; an unknown doc is a hard error (we never silently widen back out —
        that would re-open the wrong-issuer bind).
        """
        glist = candidate_grids()
        if not doc:
            return glist
        scoped = [g for g in glist if _doc_match(g.doc, doc)]
        if not scoped:
            have = sorted({g.doc for g in glist if g.doc})
            raise CellError(f"document not in scope: {doc!r} (have {have})")
        return scoped

    def ambiguous(label: str, section: str, period: str,
                  cells: List[CellRef]) -> AmbiguousCell:
        """A self-healing ambiguity: name the right repair for THIS clash.

        Two distinct clashes need two distinct repairs (live 2026-06-11):
          - CROSS-DOC: the same label in several issuers' filings → name each
            value's DOCUMENT(s); repair = add `doc`.
          - INTRA-DOC: a generic label like "Total" appears in many SECTIONS of one
            document (Revenue/Operating Income/Geographic…) → "add doc" is useless
            (the doc is already one); name each value's SECTION; repair = add
            `section`. Without this the model burned its whole step budget guessing
            sections on the MSFT "Total" revenue row (it never tried section:"Revenue").
        """
        docs = {c.doc or "?" for c in cells}
        if len(docs) > 1:
            by_val: Dict[float, set] = {}
            for c in cells:
                by_val.setdefault(round(c.value, 4), set()).add(c.doc or "?")
            detail = "; ".join(f"{v:g} ({', '.join(sorted(d))})"
                               for v, d in sorted(by_val.items()))
            sec = f"[{section}]" if section else ""
            return AmbiguousCell(
                f"ambiguous: {label!r}{sec}[{period}] resolves to {detail} — "
                f'add "doc" to the reference to scope one document; flagged for review'
            )

        # Intra-doc: disambiguate by SECTION. Name each value's section (the empty
        # section is shown as "(none)") so the model re-issues with the right one.
        by_val_sec: Dict[float, set] = {}
        for c in cells:
            by_val_sec.setdefault(round(c.value, 4), set()).add(
                (c.section or "").strip() or "(none)")
        detail = "; ".join(f"{v:g} [{', '.join(sorted(s))}]"
                           for v, s in sorted(by_val_sec.items()))
        avail = sorted({(c.section or "").strip() for c in cells if (c.section or "").strip()})
        avail_hint = (f' available sections: {", ".join(avail)}.' if avail else "")
        sec = f"[{section}]" if section else ""
        return AmbiguousCell(
            f"ambiguous: {label!r}{sec}[{period}] resolves to {detail} — add a "
            f'"section" to the reference to pick one row (e.g. row:{{"section":"…",'
            f'"label":{label!r}}}).{avail_hint} Flagged for review'
        )

    def resolve(ref: Dict[str, Any], period: str) -> CellRef:
        """Resolve a {row:{section,label}, period} reference across candidate grids.

        Correct-or-ABSTAIN (§4a/§4b): we never *guess* among ambiguous tables.
          pass 1 — require the section to match (strict): never returns a
                   Consolidated "Net sales" when the spec asked for AWS.
          pass 2 — section-agnostic, but if DIFFERENT grids yield DIFFERENT values
                   for the same reference, that's genuine ambiguity → raise
                   AmbiguousCell so the figure is flagged for a human rather than
                   silently picking one. Identical values across grids are fine.
        """
        # Accept the row ref under "row", "numerator" (models reuse the ratio shape for
        # delta/growth), or flat {section,label}. Live (2026-06-11) a growth spec carried
        # its ref under "numerator"; resolve() fell through to the WHOLE spec, read an
        # EMPTY label, and bound an arbitrary row — a kernel-level confident-wrong.
        row = ref.get("row") or ref.get("numerator") or ref
        if not isinstance(row, dict):
            row = ref
        section = row.get("section", "")
        label = row.get("label", "")
        if not str(label).strip():
            # An empty label must never resolve: find_row('') matches arbitrary rows.
            # Fail closed with the repair named (self-healing, same style as ambiguous()).
            raise CellError(
                "cell reference needs a non-empty 'label' — pass "
                'row:{"section"?, "label"} (e.g. {"op":"growth_pct",'
                '"row":{"section":"Revenue","label":"Total"},"from_period":...,"to_period":...})'
            )
        grids_list = scoped_grids(_ref_doc(ref, row))

        # pass 1: honor the section if one was given — a unique strict match wins.
        if section:
            strict = []
            for g in grids_list:
                r = g.find_row(label, section)
                if r is not None and _norm_section_match(r, section):
                    try:
                        strict.append(g.cell(label, period, section))
                    except CellError:
                        continue
            if strict:
                vals = {round(c.value, 4) for c in strict}
                if len(vals) == 1:
                    return strict[0]
                raise ambiguous(label, section, period, strict)

        # pass 2: section-agnostic. Collect ALL matches; abstain if they disagree.
        # Grounding delegation (§5.1): when the requested label signals an AGGREGATE
        # ("total revenue", "consolidated net sales"), apply the value-based
        # aggregation-level guard — drop any candidate that is structurally a
        # COMPONENT (looks_like_total → False) before the disagreement check. This is
        # the structural fix for the §2.1 "total revenue → Gross margin" confident-
        # wrong: a total request can never bind a child line. Pure tightening — it can
        # only turn a wrong/ambiguous bind into an abstain, never change a good bind
        # (kernel gates that use precise component labels are unaffected). No label
        # list decides the total; the values do.
        want_total = _is_aggregate_request(label)
        matches, last_err, dropped_prose = [], None, False
        for g in grids_list:
            r = g.find_row(label, "")
            if r is None:
                continue
            # Prose-fragment guard (§5.1): the RESOLVED row's own label must read like
            # a financial line-item, not a narrative sentence — otherwise the figure
            # leaked out of prose (geometry over a text page) and binding it is a
            # confident-wrong. Drop it; let resolution fall to a real table or abstain.
            if not is_lineitem_label(r.get("label", "")):
                dropped_prose = True
                continue
            try:
                cell = g.cell(label, period, "")
            except CellError as e:
                last_err = e
                continue
            if want_total and looks_like_total(g, r, period) is False:
                continue  # asked for a total, this row is a component → disqualify
            matches.append((cell, _norm_label_eq(r.get("label", ""), label),
                            row_value_kind(g, r)))
        # Specificity + value-kind filters, applied WITHIN each document only —
        # they disambiguate metrics inside a doc, never across docs (an exact
        # 'Total revenue' must not silently pick MSFT over GOOG's 'Total
        # revenues'; cross-doc ambiguity stays an explicit doc-scoping abstain).
        #   exact > contains: 'Research and development' never collides with
        #   'Capitalized research and development'; the exact 'R&D credit' row is
        #   never shadowed by the broader expense rows.
        #   currency > percent at equal specificity: a $ request never binds the
        #   14.9 '% of net sales' twin of an 85,622 row. Percent-only matches
        #   survive untouched (rate/credit metrics stay bindable).
        # Filters, not pickers: survivors that still disagree abstain below.
        by_doc: Dict[Any, List[Any]] = {}
        for m in matches:
            by_doc.setdefault(m[0].doc, []).append(m)
        matches = []
        for grp in by_doc.values():
            if any(ex for _, ex, _ in grp):
                grp = [m for m in grp if m[1]]
            if (any(k == "currency" for _, _, k in grp)
                    and any(k == "percent" for _, _, k in grp)):
                grp = [m for m in grp if m[2] != "percent"]
            matches.extend(grp)
        cells = [c for c, _, _ in matches]
        if not cells:
            # nothing resolvable — distinguish the structural drops from "no row at all"
            if want_total and last_err is None:
                raise CellError(
                    f"no aggregate/total row for {label!r}[{period}] "
                    f"(matches were component lines — abstaining, not a child)"
                )
            if dropped_prose:
                raise CellError(
                    f"no line-item row for {label!r}[{period}] "
                    f"(only prose-fragment matches — abstaining, not a narrative number)"
                )
            sugg = _label_suggestions(label, grids_list)
            raise CellError(
                (str(last_err) if last_err else f"cell not resolvable: {label!r} [{period}]")
                + (f" — closest line-items: {sugg}" if sugg else ""))
        vals = {round(c.value, 4) for c in cells}
        if len(vals) == 1:
            return cells[0]
        raise ambiguous(label, "", period, cells)

    def build_series(spec: Dict[str, Any]) -> Tuple[List[Tuple[str, CellRef]], str]:
        """Resolve a selection spec into an ordered series of (key, CellRef) pairs.

        Two axes, exactly mirroring the two real grid shapes (§3.3):

          • over periods — ONE row scanned across its period columns. The key is the
            period (the pivot a `first_exceeds` answer binds, e.g. "2022"). Shape:
              {"over":"period", "row":{"section","label"}, "periods":[...optional]}
            If `periods` is omitted we use every value column of the resolving grid,
            in the grid's column order (chronological) — the COMPLETE series, which
            is the whole point: "first exceeded" needs every year, not a top-k.

          • over rows/entities — a LIST of rows at a FIXED period. The key is the
            entity (section or label — the "which company/segment" pivot). Shape:
              {"over":"entity", "period":"2022", "rows":[{"section","label"}, ...]}

        Returns (series, axis). Raises CellError if nothing resolves — the caller
        turns that into a graceful abstain, never a guess. Cells that don't parse
        (a dash, "NM") are skipped, not invented; the rest of the series stands.
        """
        over = spec.get("over")
        # infer the axis when the model omits it: a `rows` list ⇒ entity, else period
        if over not in ("period", "entity"):
            over = "entity" if isinstance(spec.get("rows"), list) else "period"

        series: List[Tuple[str, CellRef]] = []
        if over == "entity":
            period = spec.get("period")
            if period is None:
                raise CellError("selection over entities needs a `period`")
            for ref in spec.get("rows", []):
                row = ref.get("row", ref)
                key = row.get("section") or row.get("label") or "?"
                try:
                    series.append((str(key), resolve(ref, period)))
                except CellError:
                    continue  # a missing/non-numeric entity drops out; others remain
            if not series:
                raise CellError("selection over entities resolved no numeric cells")
            return series, "entity"

        # over periods: resolve the single row, then read each period column it has.
        ref = spec.get("row", spec)
        row = ref.get("row", ref)
        section, label = row.get("section", ""), row.get("label", "")
        doc_scope = _ref_doc(ref, row)
        periods = spec.get("periods")
        # find the grid that actually holds this row, to know its period columns.
        # Skip a row whose label is PROSE, not a line-item (geometry over a text page
        # leaks numbers — e.g. a sentence "...AWS operating income in 2022..." whose
        # "2022" column parses to the literal 2022). Same prose guard grounding applies
        # for lookups (§5.1); without it a threshold scan can land on a narrative row and
        # return a year-as-value. Falls back to ANY match only if no line-item row exists.
        #
        # Two passes on the section (mirrors resolve()): pass 1 honors the requested
        # section strictly; pass 2 (if nothing matched) is section-AGNOSTIC — so an
        # over-scoped section from comprehension (e.g. the parent company "amazon" when the
        # row lives under the segment "AWS") still resolves by label instead of hard-
        # failing. The pivot is still value-checked by the monitor (§5.5), so a wrong
        # section-agnostic match can't pass silently.
        # A period scan needs a grid whose axis is REAL years — a grid still labelled
        # col_1/col_2 (extraction failed to recover its header) cannot answer "which year",
        # and scanning it binds a garbage pivot like "col_1". So a host must have a real
        # period axis; we only fall back to a col_N grid if NO real-period grid matches.
        def _real_periods(g: Grid) -> bool:
            return any(_YEARISH_COL.search(str(c)) for c in g.value_columns())

        def _find_hosts(require_section: bool, require_periods: bool) -> List[Grid]:
            line_hosts: List[Grid] = []
            text_host: Optional[Grid] = None
            for g in scoped_grids(doc_scope):
                if require_periods and not _real_periods(g):
                    continue
                r = g.find_row(label, section if require_section else "")
                if r is None:
                    continue
                if require_section and section and not _norm_section_match(r, section):
                    continue
                if is_lineitem_label(r.get("label", "")):
                    line_hosts.append(g)
                elif text_host is None:
                    text_host = g
            if line_hosts:
                # same specificity + value-kind filters as resolve(), WITHIN each
                # doc only (cross-doc ambiguity stays the doc-scoping abstain):
                # exact-label hosts beat contains-hosts; %-presentation hosts lose
                # to currency hosts (a first_exceeds over a $ metric must never
                # scan its '% of net sales' twin series).
                def _host_row(g: Grid) -> Dict[str, Any]:
                    return g.find_row(label, section if require_section else "") or {}
                by_doc: Dict[Any, List[Grid]] = {}
                for g in line_hosts:
                    by_doc.setdefault(g.doc, []).append(g)
                line_hosts = []
                for grp in by_doc.values():
                    exact = [g for g in grp
                             if _norm_label_eq(_host_row(g).get("label", ""), label)]
                    if exact:
                        grp = exact
                    cur_hosts = [g for g in grp
                                 if row_value_kind(g, _host_row(g)) == "currency"]
                    if cur_hosts and len(cur_hosts) < len(grp):
                        grp = cur_hosts
                    line_hosts.extend(grp)
                return line_hosts
            return [text_host] if text_host is not None else []

        # preference order (real-period grids always win): (periods+section) →
        # (periods, any section) → (any grid + section) → (any grid). `eff_section` is the
        # section to READ cells with: the requested section when it matched, else "" (the
        # row sat under a different/over-scoped section label, e.g. parent "amazon").
        hosts = _find_hosts(require_section=True, require_periods=True)
        eff_section = section
        if not hosts and section:
            hosts = _find_hosts(require_section=False, require_periods=True)
            eff_section = ""
        if not hosts:                                 # no real-period grid — degrade
            hosts = _find_hosts(require_section=True, require_periods=False)
            eff_section = section
        if not hosts and section:
            hosts = _find_hosts(require_section=False, require_periods=False)
            eff_section = ""
        if not hosts:
            raise CellError(f"selection: row not found: section={section!r} label={label!r}")

        host_docs = {g.doc for g in hosts if g.doc}
        if len(host_docs) <= 1:
            # single document (or unlabelled grids): the original single-host scan.
            host = hosts[0]
            cols = list(periods) if periods else host.value_columns()
            for p in cols:
                try:
                    series.append((str(p), host.cell(label, p, eff_section)))
                except CellError:
                    continue
        else:
            # The row exists in SEVERAL documents and no `doc` scope was given.
            # "First host wins" would silently scan whichever issuer/filing came
            # first — and a threshold/extremum pivot is WINDOW-sensitive, so the
            # wrong document binds a wrong year (a confident-wrong). Structural
            # resolution, value-decided (no doc guessed):
            #   • merge the per-document series into ONE union series — valid ONLY
            #     when every document AGREES at every shared year (two filings of
            #     the same issuer overlap and agree; the union is the most COMPLETE
            #     window, which is exactly what threshold ops need);
            #   • any shared-year disagreement = different underlying series (e.g.
            #     two issuers' "Total revenue") → abstain, naming the documents and
            #     the `doc` repair;
            #   • no shared year at all = no consistency evidence → abstain too
            #     (silently chaining unrelated windows would fabricate a series).
            by_year: Dict[int, CellRef] = {}
            docs_at_year: Dict[int, set] = {}
            for g in hosts:
                for p in (list(periods) if periods else g.value_columns()):
                    y = _period_year(str(p))
                    if y is None:
                        continue  # only real years are alignable across documents
                    try:
                        c = g.cell(label, p, eff_section)
                    except CellError:
                        continue
                    prev = by_year.get(y)
                    if prev is not None and round(prev.value, 4) != round(c.value, 4):
                        raise AmbiguousCell(
                            f"ambiguous: {label!r} differs across documents at {y} "
                            f"({prev.value:g} in {prev.doc or '?'} vs {c.value:g} in "
                            f"{c.doc or '?'}) — add \"doc\" to scope one document; "
                            f"flagged for review"
                        )
                    if prev is None:
                        by_year[y] = c
                    docs_at_year.setdefault(y, set()).add(g.doc)
            if not any(len(d) > 1 for d in docs_at_year.values()):
                raise AmbiguousCell(
                    f"ambiguous: a {label!r} series exists in several documents "
                    f"({', '.join(sorted(host_docs))}) with no overlapping years to "
                    f"prove they are the same series — add \"doc\" to scope one; "
                    f"flagged for review"
                )
            series = [(str(y), by_year[y]) for y in sorted(by_year)]
        if not series:
            raise CellError(f"selection over periods resolved no numeric cells for {label!r}")
        # Chronological order is REQUIRED for threshold ops: grid COLUMN order is not
        # guaranteed ascending (Microsoft statements list the LATEST year first), so a
        # raw `first_exceeds` scan over column order binds the wrong year (and the monitor,
        # seeing the winner at scan index 0, has no prior to test → certifies it). Sort the
        # period series ascending by extracted year so the scan walks time forward. Only
        # when EVERY key yields a year; a positional/mixed axis (col_1…) is left as-is and
        # the predicate check abstains rather than trust an unknown order. Order-independent
        # ops (argmax/argmin/rank/filter) are unaffected by the sort.
        if all(_period_year(k) is not None for k, _c in series):
            series.sort(key=lambda kv: _period_year(kv[0]))
        return series, "period"

    def _num(spec: Dict[str, Any], *keys: str) -> float:
        for k in keys:
            if k in spec and spec[k] is not None:
                return float(parse_cell(spec[k]))
        raise CellError(f"selection: missing threshold (one of {keys})")

    try:
        if op == "value":
            c = resolve(spec, spec["period"])
            return ComputeResult(op, c.value, formula=c.trace(), unit="", cells=[c])

        if op in ("delta", "growth_pct"):
            frm = resolve(spec, spec["from_period"])
            to = resolve(spec, spec["to_period"])
            if op == "delta":
                val = to.value - frm.value
                f = f"{_fmt(to.value)} − {_fmt(frm.value)} = {_fmt(val)}"
                return ComputeResult(op, val, f, unit="", cells=[frm, to])
            if frm.value == 0:
                return ComputeResult(op, None, "", error="growth_pct: prior value is 0")
            val = (to.value - frm.value) / abs(frm.value) * 100
            f = f"({_fmt(to.value)} − {_fmt(frm.value)}) / |{_fmt(frm.value)}| × 100 = {val:.1f}%"
            return ComputeResult(op, val, f, unit="%", cells=[frm, to])

        if op in ("sum", "average"):
            cells = [resolve(c, c.get("period", spec.get("period"))) for c in spec["cells"]]
            vals = [c.value for c in cells]
            total = sum(vals)
            if op == "sum":
                f = " + ".join(_fmt(v) for v in vals) + f" = {_fmt(total)}"
                return ComputeResult(op, total, f, unit="", cells=cells)
            avg = total / len(vals)
            f = f"({' + '.join(_fmt(v) for v in vals)}) / {len(vals)} = {_fmt(avg)}"
            return ComputeResult(op, avg, f, unit="", cells=cells)

        if op == "difference":
            a = resolve(spec["a"], spec["a"].get("period", spec.get("period")))
            b = resolve(spec["b"], spec["b"].get("period", spec.get("period")))
            val = a.value - b.value
            f = f"{_fmt(a.value)} − {_fmt(b.value)} = {_fmt(val)}"
            return ComputeResult(op, val, f, unit="", cells=[a, b])

        if op in ("ratio", "margin_pct"):
            num = resolve(spec["numerator"], spec["numerator"].get("period", spec.get("period")))
            den = resolve(spec["denominator"], spec["denominator"].get("period", spec.get("period")))
            if den.value == 0:
                return ComputeResult(op, None, "", error=f"{op}: denominator is 0")
            # Ratio operand-consistency guards (correct-or-abstain). A ratio/margin is
            # only meaningful when both operands describe the SAME entity. Two ways it
            # silently goes wrong after section-aware resolution — both abstain, never
            # state a confident figure:
            #   (a) same cell: numerator and denominator collapsed onto one row → a
            #       meaningless 100% (e.g. an AWS margin whose net-sales denominator
            #       mis-resolved to the AWS operating-income row).
            #   (b) cross-segment: operands resolved to DIFFERENT non-empty sections
            #       (e.g. AWS operating income ÷ Consolidated operating income = 66.8%
            #       passed off as "AWS margin"). You cannot mix segments in one ratio.
            nsec_, dsec_ = (num.section or "").strip().lower(), (den.section or "").strip().lower()
            same_cell = (num.doc, num.page, num.table_id, num.section, num.label, num.period) == \
                        (den.doc, den.page, den.table_id, den.section, den.label, den.period)
            cross_segment = bool(nsec_) and bool(dsec_) and nsec_ != dsec_
            if same_cell or cross_segment:
                why = "same cell" if same_cell else f"different segments ({num.section!r} vs {den.section!r})"
                return ComputeResult(op, None, "", error=(
                    f"{op}: numerator and denominator are inconsistent — {why}; abstaining"))
            r = num.value / den.value
            if op == "ratio":
                f = f"{_fmt(num.value)} / {_fmt(den.value)} = {r:.4f}"
                return ComputeResult(op, r, f, unit="x", cells=[num, den])
            pct = r * 100
            f = f"{_fmt(num.value)} / {_fmt(den.value)} × 100 = {pct:.1f}%"
            return ComputeResult(op, pct, f, unit="%", cells=[num, den])

        if op == "cagr_pct":
            cells = [resolve(c, c["period"]) for c in spec["cells"]]
            if len(cells) < 2:
                return ComputeResult(op, None, "", error="cagr_pct needs ≥2 periods")
            begin, end = cells[0].value, cells[-1].value
            n = len(cells) - 1
            if begin <= 0:
                return ComputeResult(op, None, "", error="cagr_pct: begin value must be > 0")
            val = ((end / begin) ** (1 / n) - 1) * 100
            f = f"(({_fmt(end)} / {_fmt(begin)})^(1/{n}) − 1) × 100 = {val:.1f}%"
            return ComputeResult(op, val, f, unit="%", cells=cells)

        # ── Selection ops (Layer 1, §3.3) ─────────────────────────────────────
        # Each builds the COMPLETE series, then picks deterministically. The result
        # sets `binding` (the resolved period/entity = the pivot) and `candidates`
        # (everything scanned = the completeness trail). `value` is the winning
        # cell's number so numeric grounding still applies downstream.
        if op in ("argmax", "argmin"):
            series, axis = build_series(spec)
            pick = (max if op == "argmax" else min)(series, key=lambda kv: kv[1].value)
            key, cell = pick
            scanned = [c for _k, c in series]
            arrow = "max" if op == "argmax" else "min"
            f = (f"{arrow} over {{" +
                 ", ".join(f"{k}={_fmt(c.value)}" for k, c in series) +
                 f"}} = {key} ({_fmt(cell.value)})")
            return ComputeResult(op, cell.value, f, unit="", cells=[cell],
                                 binding=key, candidates=scanned)

        if op in ("first_exceeds", "last_below"):
            series, axis = build_series(spec)
            if axis != "period":
                # threshold-crossing is only meaningful over an ORDERED period series
                return ComputeResult(op, None, "",
                                     error=f"{op}: requires an ordered period series (over='period')")
            thr = _num(spec, "threshold", "value")
            scanned = [c for _k, c in series]
            if op == "first_exceeds":
                hit = next(((k, c) for k, c in series if c.value > thr), None)
                rel = ">"
            else:  # last_below — latest period still under the threshold
                hit = next(((k, c) for k, c in reversed(series) if c.value < thr), None)
                rel = "<"
            if hit is None:
                return ComputeResult(op, None, "",
                                     error=f"{op}: no period {rel} {_fmt(thr)} in "
                                           f"{[(k, c.value) for k, c in series]}",
                                     candidates=scanned)
            key, cell = hit
            f = (f"first period {rel} {_fmt(thr)} over [" if op == "first_exceeds"
                 else f"last period {rel} {_fmt(thr)} over [") + \
                ", ".join(f"{k}={_fmt(c.value)}" for k, c in series) + f"] = {key}"
            return ComputeResult(op, cell.value, f, unit="", cells=[cell],
                                 binding=key, candidates=scanned)

        if op == "rank":
            series, axis = build_series(spec)
            desc = spec.get("descending", True)
            ordered = sorted(series, key=lambda kv: kv[1].value, reverse=bool(desc))
            top = spec.get("top")
            if isinstance(top, int) and top > 0:
                ordered = ordered[:top]
            scanned = [c for _k, c in series]
            best_key, best_cell = ordered[0]
            f = ("rank " + ("↓" if desc else "↑") + ": " +
                 " > ".join(f"{k}({_fmt(c.value)})" for k, c in ordered))
            return ComputeResult(op, best_cell.value, f, unit="", cells=[c for _k, c in ordered],
                                 binding=best_key, candidates=scanned)

        if op == "filter":
            series, axis = build_series(spec)
            cmp = (spec.get("cmp") or spec.get("predicate") or ">").strip()
            thr = _num(spec, "threshold", "value")
            ops_map = {
                ">": lambda v: v > thr, ">=": lambda v: v >= thr,
                "<": lambda v: v < thr, "<=": lambda v: v <= thr,
                "==": lambda v: v == thr, "!=": lambda v: v != thr,
            }
            if cmp not in ops_map:
                return ComputeResult(op, None, "", error=f"filter: bad comparator {cmp!r}")
            kept = [(k, c) for k, c in series if ops_map[cmp](c.value)]
            scanned = [c for _k, c in series]
            if not kept:
                return ComputeResult(op, None, "",
                                     error=f"filter: no member {cmp} {_fmt(thr)}",
                                     candidates=scanned)
            keys = ", ".join(k for k, _c in kept)
            f = f"{{" + ", ".join(f"{k}={_fmt(c.value)}" for k, c in series) + f"}} where x {cmp} {_fmt(thr)} → [{keys}]"
            return ComputeResult(op, kept[0][1].value, f, unit="", cells=[c for _k, c in kept],
                                 binding=keys, candidates=scanned)

    except CellError as e:
        return ComputeResult(op, None, "", error=str(e))
    except KeyError as e:
        return ComputeResult(op, None, "", error=f"spec missing field: {e}")
    except Exception as e:  # never let a bad spec crash the caller
        return ComputeResult(op, None, "", error=f"{type(e).__name__}: {e}")

    return ComputeResult(op, None, "", error="unhandled op")


# ── LLM spec-writer (the ONLY LLM touchpoint; emits DATA, never code) ──────────

def grid_catalog(grids: List[Grid], max_rows: int = 40) -> str:
    """Compact, model-readable catalog of the available grids and their cells.

    The LLM sees this and writes a spec that references rows/periods BY NAME. It
    never sees or writes code. Cells are shown so the model picks real labels,
    but the model's job is only to *select* — the deterministic interpreter does
    the math on the authoritative grid values, not on anything the model echoes.
    """
    lines = []
    for i, g in enumerate(grids):
        cols = g.value_columns()
        unit = f" units={g.units}" if g.units else ""
        # table-level summary: distinct sections + value kind, so the model can
        # pick the right STATEMENT (a $ segment table vs its growth-% twin vs a
        # balance sheet) before picking a row — generic, layout-independent.
        secs, seen = [], set()
        for r in g.rows:
            s = (r.get("section") or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                secs.append(s)
        pct = sum(1 for r in g.rows for c in cols if str(r.get(c, "")).strip().endswith("%"))
        total_vals = sum(1 for r in g.rows for c in cols if str(r.get(c, "")).strip())
        kind = "percentages" if total_vals and pct / total_vals > 0.5 else "dollar/absolute values"
        sec_line = f" sections=[{', '.join(secs[:8])}]" if secs else ""
        lines.append(
            f"TABLE {i} (doc={g.doc} page={g.page} periods={cols}{unit} values={kind}{sec_line}):"
        )
        # NOTE (2026-06-05): the LLM table summary was tried here as a "↳ {summary}"
        # line but an eval showed it REGRESSED selection (5/7 → 3/7) — the extra prose
        # confused the spec-writer rather than disambiguating. Reverted to baseline.
        # The summary's value is in SEMANTIC retrieval (ranking which grids reach this
        # catalog), not in the catalog text itself. Left unused pending that approach.
        for r in g.rows[:max_rows]:
            sec = f"[{r.get('section')}] " if r.get("section") else ""
            vals = " | ".join(f"{c}={r.get(c, '')}" for c in cols)
            lines.append(f"  {sec}{r.get('label')}: {vals}")
        if len(g.rows) > max_rows:
            lines.append(f"  …(+{len(g.rows) - max_rows} more rows)")
    return "\n".join(lines)


_SPEC_SYSTEM = (
    "You are a financial analyst's calculator-router. You DO NOT do arithmetic. "
    "Given a question and a catalog of extracted tables, you output a JSON array of "
    "compute SPECS that a deterministic engine will execute. You only SELECT which "
    "cells and which operation; the engine computes the numbers.\n\n"
    "Allowed operations (use ONLY these): "
    "value, delta, growth_pct, sum, difference, ratio, margin_pct, average, cagr_pct, "
    "argmax, argmin, first_exceeds, last_below, rank, filter.\n\n"
    "Spec shapes:\n"
    '  {"op":"value","table":<i>,"row":{"section":"AWS","label":"Net sales"},"period":"2023"}\n'
    '  {"op":"growth_pct","table":<i>,"row":{...},"from_period":"2022","to_period":"2023"}\n'
    '  {"op":"delta", same fields as growth_pct}\n'
    '  {"op":"sum","table":<i>,"cells":[{"row":{...},"period":"2023"}, ...]}\n'
    '  {"op":"difference","table":<i>,"a":{"row":{...}},"b":{"row":{...}},"period":"2023"}\n'
    '  {"op":"ratio"|"margin_pct","table":<i>,"numerator":{"row":{...}},"denominator":{"row":{...}},"period":"2023"}\n'
    '  {"op":"cagr_pct","table":<i>,"cells":[{"row":{...},"period":"2021"},{"row":{...},"period":"2023"}]}\n'
    "SELECTION ops (answer WHICH year/entity, not a number — never guess the pivot):\n"
    '  over PERIODS (scan one row across years): use op=first_exceeds/last_below to find the\n'
    '  earliest/latest YEAR crossing a threshold; omit "periods" to scan ALL years (preferred):\n'
    '    {"op":"first_exceeds","over":"period","row":{"section":"AWS","label":"Operating income"},"threshold":20000}\n'
    '    {"op":"last_below","over":"period","row":{...},"threshold":40000}\n'
    '  over ENTITIES (compare rows at ONE year): use op=argmax/argmin/rank/filter; list EVERY\n'
    '  candidate row so the comparison is complete:\n'
    '    {"op":"argmax","over":"entity","period":"2023","rows":[{"row":{"section":"AWS","label":"Net sales"}},{"row":{...}}]}\n'
    '    {"op":"argmin","over":"entity","period":"2022","rows":[...]}\n'
    '    {"op":"rank","over":"entity","period":"2023","rows":[...],"top":3}\n'
    '    {"op":"filter","over":"entity","period":"2022","rows":[...],"cmp":">","threshold":0}\n\n'
    "Rules:\n"
    "- For 'the year X first exceeded $N' use first_exceeds with the threshold in the SAME units as\n"
    "  the cells (e.g. $20B over a $-millions grid → threshold 20000). Do NOT pick a year yourself.\n"
    "- For 'which company/segment has the highest/lowest …' use argmax/argmin and include EVERY\n"
    "  candidate in `rows` (e.g. all three companies, or all segments) — completeness matters.\n"
    "- Use row labels/sections/periods EXACTLY as they appear in the catalog.\n"
    "- `table` is OPTIONAL: include it only if you are sure which table; if you "
    "omit it, the engine finds the row across all tables. When several tables share "
    "a label (e.g. a dollar table and a growth-% table), prefer OMITTING `table` "
    "and let the engine resolve the row that has the actual values, OR pick the "
    "table whose values are dollar amounts (not percentages) for $ questions.\n"
    "- If the question needs no calculation (just a stated figure), use op=value.\n"
    "- For an AGGREGATE question ('total revenue', 'consolidated net sales'), pick "
    "the explicit total/consolidated row, not a single component line.\n"
    "- If the question cannot be answered from these tables, output [].\n"
    "- Output ONLY a JSON array of specs. No prose, no code, no markdown fences."
)


def write_specs(question: str, grids: List[Grid], llm) -> List[Dict[str, Any]]:
    """Ask the LLM to emit compute specs (DATA). Parsed with json — never executed.

    Returns a list of spec dicts (possibly empty). Any non-JSON / disallowed
    content yields [] rather than raising — the model cannot make us run code.
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    catalog = grid_catalog(grids)
    user = f"Question: {question}\n\nAvailable tables:\n{catalog}\n\nSpecs (JSON array only):"
    try:
        resp = llm.invoke([SystemMessage(content=_SPEC_SYSTEM), HumanMessage(content=user)])
        text = (resp.content or "").strip()
        # tolerate a stray ```json fence even though we asked for none
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("["):] if "[" in text else text
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        specs = json.loads(text[start:end + 1])
        if not isinstance(specs, list):
            return []
        # keep only well-formed dict specs naming a whitelisted op
        return [s for s in specs if isinstance(s, dict) and s.get("op") in WHITELISTED_OPS]
    except Exception:
        return []


def analyze(question: str, grids: List[Grid], llm) -> List[ComputeResult]:
    """Full Analyst pass: LLM selects specs (data) → deterministic engine computes.

    The LLM never touches a number's value and never emits executable code; the
    returned ComputeResults each carry their formula + source CellRefs for the
    numeric verifier and the user.
    """
    if not grids:
        return []
    specs = write_specs(question, grids, llm)
    return [compute(s, grids) for s in specs]


# ── Numeric verifier (§4b step 5 — the backstop) ──────────────────────────────
#
# The numeric analogue of §4a claim-grounding: every figure in a generated answer
# must trace to either a SOURCE CELL or a COMPUTED RESULT. Deterministic (no LLM):
# we extract the numbers from the answer and check each against the allowed set.
# This catches a wrong-row pick or an LLM that "helpfully" restated a number it
# made up — exactly the silent-error class §4b exists to eliminate.

_ANSWER_NUM = re.compile(r"-?\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")
# years (2019-2099) and tiny ordinals are not "figures to verify" — they're labels
_LABELISH = re.compile(r"^(19|20)\d{2}$")


@dataclass
class NumericVerdict:
    grounded: List[str]
    ungrounded: List[str]            # figures with no source cell / computed result
    groundedness: float              # |grounded| / |checked|

    @property
    def ok(self) -> bool:
        return not self.ungrounded


def _canon_num(s: str) -> Optional[float]:
    try:
        return abs(parse_cell(s))   # compare magnitudes; sign/parens handled separately
    except CellError:
        return None


def verify_numbers(
    answer: str,
    cells: List[CellRef],
    computed: List["ComputeResult"],
    *,
    tolerance: float = 0.01,
) -> NumericVerdict:
    """Check every figure in `answer` against source cells + computed results.

    A figure is grounded if it matches (within `tolerance`, relative) any source
    cell value or any computed result value. Years and small integers (≤ a couple
    digits, likely counts/labels) are skipped — they're not the financial figures
    the §4b contract is about. Deterministic; never raises.
    """
    allowed: List[float] = []
    for c in cells:
        allowed.append(abs(c.value))
    for r in computed:
        if r.ok and r.value is not None:
            allowed.append(abs(r.value))

    def matches(x: float) -> bool:
        for a in allowed:
            if a == 0:
                if abs(x) < tolerance:
                    return True
            elif abs(x - a) / max(abs(a), 1e-9) <= tolerance:
                return True
        return False

    grounded, ungrounded = [], []
    seen = set()
    for tok in _ANSWER_NUM.findall(answer or ""):
        t = tok.strip()
        if t in seen:
            continue
        seen.add(t)
        bare = t.strip("()$%,")
        if _LABELISH.match(bare):
            continue  # a year label, not a figure
        val = _canon_num(t)
        if val is None:
            continue
        # skip very small integers (1-2 digits, no decimal/comma) — counts/ordinals
        if "." not in t and "," not in t and val < 100:
            continue
        (grounded if matches(val) else ungrounded).append(t)

    checked = len(grounded) + len(ungrounded)
    groundedness = (len(grounded) / checked) if checked else 1.0
    return NumericVerdict(grounded=grounded, ungrounded=ungrounded, groundedness=groundedness)


def cells_from_results(results: List["ComputeResult"]) -> List[CellRef]:
    """Flatten the source cells used across compute results (for the verifier)."""
    out: List[CellRef] = []
    for r in results:
        out.extend(r.cells)
    return out


def corroborate_with_prose(results: List["ComputeResult"], prose: str) -> List["ComputeResult"]:
    """Cross-check each result's SOURCE CELLS against the retrieved prose (§4b backstop).

    The deterministic Analyst can mis-SELECT a row (e.g. a sub-line read as a
    "total"). The numeric verifier can't catch that — the number is real, just the
    wrong one. So as a final, generic guard (no prompt-tuning, works on any doc) we
    require each ``value``/``op=value`` result's raw source-cell figure to ALSO
    appear in the retrieved prose for this question. A figure the prose never
    mentions is demoted to an abstention (error set), not surfaced as confident.

    Computed ratios/growth aren't expected verbatim in prose, so only the direct
    "value" reads are corroborated; their inputs are what a wrong-row pick corrupts.
    """
    if not prose:
        return results

    def in_prose(raw: str) -> bool:
        # match the grouped-digits form, e.g. "125,538", ignoring $/() decoration
        digits = re.sub(r"[^\d,]", "", raw)
        return bool(digits) and digits in prose

    out = []
    for r in results:
        if r.ok and r.op == "value" and r.cells:
            if not any(in_prose(c.raw) for c in r.cells):
                r = ComputeResult(
                    op=r.op, value=None, formula=r.formula, unit=r.unit, cells=r.cells,
                    error=f"uncorroborated: {r.cells[0].raw} not found in retrieved text — flagged for review",
                )
        out.append(r)
    return out


# ── Output rendering (§4b step 4 — render as tables, cite cells) ───────────────

def render_markdown(results: List["ComputeResult"]) -> str:
    """Render Analyst results as a GFM table: result | value | formula | sources.

    Every row shows the computed value, the EXACT formula used, and the source
    cells it traces to — so the user (and the §4b numeric contract) can audit each
    number back to a source cell or a shown computation. The app already renders
    GFM tables, so this drops straight into the answer.
    """
    ok = [r for r in results if r.ok]
    if not ok:
        return ""
    lines = [
        "| Metric | Value | Formula | Source cells |",
        "| --- | --- | --- | --- |",
    ]
    for r in ok:
        # for selection ops the audit trail is the FULL set scanned (completeness),
        # not just the winning cell — show candidates so a reviewer can see the op
        # considered every entity/year before resolving the pivot.
        trace_cells = r.candidates or r.cells
        srcs = "; ".join(c.trace() for c in trace_cells) or "—"
        # escape pipes so cell text can't break the table
        formula = (r.formula or "").replace("|", "\\|")
        srcs = srcs.replace("|", "\\|")
        label = r.op.replace("_", " ")
        lines.append(f"| {label} | {r.display()} | {formula} | {srcs} |")
    return "\n".join(lines)


def results_to_rows(results: List["ComputeResult"]) -> List[Dict[str, Any]]:
    """Flatten results into plain dict rows for XLSX export (computed cols kept).

    Each row: operation, value, unit, formula, and the source-cell trace — so the
    exported workbook preserves the computed columns AND their provenance/formula.
    """
    rows = []
    for r in results:
        rows.append({
            "operation": r.op,
            "value": r.value if r.ok else None,
            "unit": r.unit,
            "formula": r.formula,
            "sources": "; ".join(c.trace() for c in r.cells),
            "computed": True,           # flag: this is a computed column, not a raw cell
            "error": r.error or "",
        })
    return rows
