"""
Phase 4.3 — Structured table extraction (the §4b foundation).

Why this module exists
----------------------
The `unstructured` `fast` parser (the working ingest path on this machine —
hi_res is broken AND too slow, see Build Log Entry 4 + 9) flattens financial
tables into *column-detached number salad*: row labels, period headers, and
values land in three separate runs with no 2-D mapping. You cannot trace a
number to its (row, period) cell, so deterministic compute (§4b) is impossible
on top of it.

This module recovers the real grid from the PDF **text layer** with
`pdfplumber` — no OCR, no YOLOX layout model — so it is ~1000× cheaper than
hi_res (measured: 5–15 s/doc vs 15–20 min) while producing a genuine
`(row, column) -> value` table.

Hard design rules (locked with the user)
-----------------------------------------
1. **Format-agnostic.** No special-casing HTML-derived PDFs. Users upload
   arbitrary PDFs; this reads the text layer the same way regardless of origin.
2. **Confidence-gated.** pdfplumber only *proposes* a grid. Every candidate is
   scored; only a grid that passes the gate becomes a `chunk_type="table"`.
   Everything else is rejected and left to the normal prose path — we NEVER emit
   a deterministic-but-wrong grid. A wrong cell is worse than no cell, because
   the entire §4b promise is "every number traces to a *correct* source cell."
3. **Latency-bounded.** Skip pages with no table-like ruling; the per-page work
   is capped. The numeric verifier downstream is the final backstop.

Output: a list of ``ExtractedTable`` — normalized JSON (headers / rows / units /
periods) + a retrieval caption + provenance (doc page, table_id). The *caption*
is what gets embedded for retrieval; the *grid* is carried in metadata for the
deterministic Analyst (§4b step 3). "Embed the summary, carry the grid."
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning knobs (conservative; the gate, not these, is the real safety) ──────
MIN_ROWS = 2                 # a real table needs at least a header-ish row + 1 data row
MIN_DATA_CELLS = 3           # fewer than this and it's almost certainly a layout fragment
MIN_NUMERIC_FRACTION = 0.25  # of the non-empty data cells, this fraction must be numeric
MAX_RAGGED_FRACTION = 0.34   # fraction of rows allowed to deviate from the modal column count
CONFIDENCE_THRESHOLD = 0.60  # below this → reject, fall back to prose

# A "number" in a financial statement: 1,234  (1,234)  $1,234.5  45.2%  (3)  —
_NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")
# Period/year headers: 2021, FY2023, Q3 2024, "Year Ended December 31, 2023"
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Standalone currency/spacer tokens that pdfplumber drops into their own column
_NOISE_CELL = re.compile(r"^[\s$%—–-]*$")


@dataclass
class ExtractedTable:
    """A confidence-gated, normalized table ready to become a chunk_type=table doc."""
    page_number: int
    table_id: str                       # stable within a doc: "p{page}_t{idx}"
    headers: List[str]                  # column headers (period labels live here)
    rows: List[Dict[str, Any]]          # [{"label": "AWS Net sales", "2021": "62,202", ...}]
    units: Optional[str]                # "$ in millions" if detectable, else None
    periods: List[str]                  # detected period/year columns, e.g. ["2021","2022","2023"]
    caption: str                        # deterministic one-line summary (fallback for embedding)
    confidence: float                   # gate score [0,1]
    raw_grid: List[List[str]] = field(default_factory=list)  # cleaned 2-D cells, for the Analyst
    markdown: str = ""                  # GFM rendering for context/output
    summary: str = ""                   # LLM discriminative summary — THIS is what gets embedded
                                        # when present (separates near-identical twin tables); falls
                                        # back to `caption` when empty. Added by the ingest caller,
                                        # NOT by extract_tables_from_pdf (extraction stays LLM-free).

    def to_metadata(self) -> Dict[str, Any]:
        """The normalized JSON carried in chunk metadata for the Analyst (never embedded)."""
        return {
            "table_id": self.table_id,
            "headers": self.headers,
            "rows": self.rows,
            "units": self.units,
            "periods": self.periods,
            "confidence": round(self.confidence, 3),
            "grid": self.raw_grid,
            "summary": self.summary,   # flows to Grid.summary for the LLM catalog (§4b selection)
        }


# ── Cell-level helpers ────────────────────────────────────────────────────────

def _is_numeric(cell: str) -> bool:
    c = (cell or "").strip()
    return bool(c) and bool(_NUM_RE.match(c))


def _is_noise(cell: str) -> bool:
    """A spacer/currency-only cell pdfplumber emitted as its own column."""
    return _NOISE_CELL.match(cell or "") is not None


def _clean(cell: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def _drop_noise_columns(grid: List[List[str]]) -> List[List[str]]:
    """Remove columns that are entirely noise ($ symbols, blank spacers).

    pdfplumber on right-aligned financial tables puts the leading ``$`` in its own
    column and inserts blank spacer columns between periods. Those carry no data
    and wreck header↔value alignment, so we strip any column that is noise in
    *every* row. A column with even one real value is kept.
    """
    if not grid:
        return grid
    ncols = max(len(r) for r in grid)
    keep = []
    for c in range(ncols):
        col_vals = [(_clean(r[c]) if c < len(r) else "") for r in grid]
        if any(not _is_noise(v) for v in col_vals):
            keep.append(c)
    return [[_clean(r[c]) if c < len(r) else "" for c in keep] for r in grid]


def _merge_currency_into_values(grid: List[List[str]]) -> List[List[str]]:
    """After noise-column drop, re-attach a stray leading ``$`` if it survived.

    Defensive: if a row looks like ['Net sales', '$', '90,757'] slipped through,
    fold the bare ``$`` into the following numeric cell so the value column stays
    clean. Pure normalization — never invents or moves a number.
    """
    out = []
    for row in grid:
        merged = []
        skip = False
        for i, cell in enumerate(row):
            if skip:
                skip = False
                continue
            if cell.strip() in ("$", "$ ") and i + 1 < len(row) and _is_numeric(row[i + 1]):
                merged.append(row[i + 1])
                skip = True
            else:
                merged.append(cell)
        out.append(merged)
    return out


# ── Confidence gate ───────────────────────────────────────────────────────────

def _score_grid(grid: List[List[str]]) -> Tuple[float, Dict[str, float]]:
    """Score a cleaned grid in [0,1]. Returns (confidence, component_breakdown).

    A real financial table is: rectangular (consistent column count), has enough
    data cells, and a meaningful fraction of those cells are numeric. A layout
    fragment (running header, a paragraph mis-detected as a 1-col "table") fails
    at least one of these. We deliberately favor *rejection* on ambiguity.
    """
    if len(grid) < MIN_ROWS:
        return 0.0, {"reason_rows": 0.0}

    # Rectangularity is measured on VALUE counts, not raw column counts: the real
    # alignment hazard is rows carrying a different number of *values* (then
    # right-alignment can't reconcile them). Section-header rows (zero values) are
    # excluded — they're legitimately empty, not ragged.
    value_counts = [len(_row_values(r)) for r in grid]
    data_value_counts = [n for n in value_counts if n > 0]
    if not data_value_counts:
        return 0.0, {"reason_no_values": 0.0}
    width = max(set(data_value_counts), key=data_value_counts.count)
    if width < 1:
        return 0.0, {"reason_singlecol": 0.0}

    # Rows whose value-count exceeds the modal width are the dangerous ones
    # (right-alignment would truncate them). Penalize those specifically.
    irreconcilable = sum(1 for n in data_value_counts if n > width) / len(data_value_counts)
    rect_score = max(0.0, 1.0 - (irreconcilable / MAX_RAGGED_FRACTION))

    # Data-cell census over the non-header region (skip row 0 as a header guess).
    data_cells = [
        _clean(c)
        for row in grid[1:]
        for c in row
    ]
    non_empty = [c for c in data_cells if c]
    if len(non_empty) < MIN_DATA_CELLS:
        return 0.0, {"reason_too_few_cells": 0.0}

    numeric = [c for c in non_empty if _is_numeric(c)]
    numeric_frac = len(numeric) / len(non_empty)
    numeric_score = min(1.0, numeric_frac / MIN_NUMERIC_FRACTION) if numeric_frac >= MIN_NUMERIC_FRACTION else (numeric_frac / MIN_NUMERIC_FRACTION) * 0.5

    # A label column: row 0 / first column should have some text (row headers).
    first_col = [_clean(r[0]) for r in grid if r]
    has_labels = any(c and not _is_numeric(c) for c in first_col)
    label_score = 1.0 if has_labels else 0.4

    confidence = 0.40 * rect_score + 0.40 * numeric_score + 0.20 * label_score
    return confidence, {
        "rectangularity": round(rect_score, 3),
        "numeric": round(numeric_score, 3),
        "labels": round(label_score, 3),
        "numeric_frac": round(numeric_frac, 3),
    }


# ── Geometry-based line reading (Layer 0 §1 — section fidelity from indentation) ──
#
# `page.extract_tables()` gives string grids with NO geometry, and on HTML-derived
# PDFs it silently DROPS total/subtotal rows (e.g. MSFT "Total revenue 198,270"
# lives only in the word layer). It also can't express row HIERARCHY, so a sticky
# section label mislabels statement-level lines under the wrong sub-section.
#
# Indentation (a row's left edge, x0) is the UNIVERSAL hierarchy signal in financial
# statements (and legal sub-clauses): section headers sit left, their line items are
# indented under them, totals/subtotals deeper still. Reading rows directly from the
# word layer recovers BOTH the dropped totals and the true indent of every row — the
# raw material §1 needs. This is deterministic, LLM-free, and format-agnostic: it
# reads structure (x-position), never content (no hardcoded labels). Every grid it
# yields still goes through the SAME confidence gate, so a bad read is rejected, not
# shipped (the module's "a wrong cell is worse than no cell" rule holds).

# A bare footnote/reference marker pdfplumber sometimes splits off ("(1)", "(2)").
_FOOTNOTE_TOK = re.compile(r"^\(\d{1,2}\)$")
# A value token at the cell level: number, optionally $-prefixed, parens=negative, %.
_VALUE_TOK = re.compile(r"^\(?\$?-?[\d,]+(?:\.\d+)?\)?%?$")


def _read_geometry_lines(page, y_tol: float = 3.0) -> List[Dict[str, Any]]:
    """Reconstruct logical rows from a page's words, carrying each row's indent.

    Groups words into lines by their vertical position (a y-band of ``y_tol`` pts),
    orders each line left→right, then splits it into a LABEL (the leading text run)
    and its VALUE tokens (the trailing numbers). Returns one dict per line:
        {"label": str, "x0": float, "values": [str], "ntext": int, "nval": int}
    where ``x0`` is the label's left edge (the indent that encodes hierarchy).

    Pure and deterministic — no thresholds beyond grouping tolerance, no hardcoded
    labels. Noise filtering and table-boundary decisions happen in the caller; this
    only faithfully reconstructs lines + geometry. Never raises (a page whose words
    can't be read returns [])."""
    try:
        words = page.extract_words()
    except Exception:
        return []
    from collections import defaultdict
    bands: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for w in words:
        try:
            bands[round(w["top"] / y_tol)].append(w)
        except Exception:
            continue

    lines: List[Dict[str, Any]] = []
    for top in sorted(bands):
        ws = sorted(bands[top], key=lambda w: w.get("x0", 0.0))
        if not ws:
            continue
        label_toks: List[str] = []
        values: List[str] = []
        for w in ws:
            tok = (w.get("text") or "").strip()
            if not tok or tok == "$":
                continue  # standalone currency symbol is not a value or a label word
            stripped = tok.replace("$", "").strip()
            is_value = bool(_VALUE_TOK.match(tok)) and not _FOOTNOTE_TOK.match(tok)
            if is_value and stripped:
                values.append(tok.replace("$", "").strip())
            elif not values:
                # text BEFORE the first number is part of the label; text AFTER the
                # numbers (a trailing footnote marker, stray glyph) is ignored so it
                # can't corrupt the label or be miscounted as a value.
                if not _FOOTNOTE_TOK.match(tok):
                    label_toks.append(tok)
        label = _clean(" ".join(label_toks))
        if not label:
            continue
        lines.append({
            "label": label,
            "x0": float(ws[0].get("x0", 0.0)),
            "values": values,
            "ntext": len(label_toks),
            "nval": len(values),
        })
    return lines


# ── Period / unit / caption derivation ────────────────────────────────────────

_YEAR_TOKEN = re.compile(r"\b(?:19|20)\d{2}\b")


def _header_year_order(page) -> List[str]:
    """GEOMETRY: years in the page's header band, ordered left→right by x-position.

    Column order is NOT universal: most statements print years ascending
    (2021, 2022, 2023) but some segment/geographic tables print them DESCENDING
    (2022, 2021, 2020). The only authoritative source is what's physically printed,
    so we read year tokens and their x-centers from the page and return the
    longest header *row* (a y-band containing ≥2 distinct years) in x-order. The
    builder maps value columns to these years positionally, so descending tables
    get descending labels — no ascending assumption.

    Returns [] when no clear multi-year header row exists; the caller then falls
    back to unlabeled col_N rather than guessing an order.
    """
    try:
        words = page.extract_words()
    except Exception:
        return []
    from collections import defaultdict
    band = defaultdict(list)  # y-band (rounded top) -> [(x_center, year)]
    for w in words:
        tok = w.get("text", "")
        if _YEAR_TOKEN.fullmatch(tok):
            xc = (w["x0"] + w["x1"]) / 2
            band[round(w["top"] / 3)].append((xc, tok))  # /3 tolerance to group a line
    best: List[str] = []
    for _top, items in band.items():
        distinct_years = {y for _x, y in items}
        if len(distinct_years) >= 2:
            ordered = [y for _x, y in sorted(items)]
            # de-dup preserving x-order (a year can repeat across stacked sub-headers)
            seen, dedup = set(), []
            for y in ordered:
                if y not in seen:
                    seen.add(y)
                    dedup.append(y)
            if len(dedup) > len(best):
                best = dedup
    return best


def _detect_periods(grid: List[List[str]], header_years: List[str], width: int) -> List[str]:
    """Resolve the period label for each value column, using geometry first.

    Priority:
    (1) GEOMETRY — if the page's x-ordered header years (``header_years``) number
        exactly ``width``, use them directly: column i ↔ header_years[i]. This is
        authoritative and order-correct (handles ascending AND descending).
    (2) GRID — years found inside this grid's own header rows, if they number
        exactly ``width`` (kept in their printed order).
    Otherwise return [] → the builder uses col_1..col_N (unlabeled but correctly
    right-aligned). We never emit a year label we can't place with confidence — a
    wrong period label is the confidently-wrong failure §4b forbids.
    """
    if width >= 2 and len(header_years) == width:
        return list(header_years)

    in_grid: List[str] = []
    for row in grid[:3]:
        for cell in row:
            for tok in _YEAR_TOKEN.findall(_clean(cell)):
                if tok not in in_grid:
                    in_grid.append(tok)
    if width and len(in_grid) == width:
        return in_grid
    return []


def _detect_units(grid: List[List[str]], page_text: str) -> Optional[str]:
    """Detect '$ in millions / thousands' from the table or surrounding page text."""
    hay = " ".join(_clean(c) for row in grid[:4] for c in row) + " " + (page_text or "")
    hay = hay.lower()
    for unit in ("in millions", "in thousands", "in billions",
                 "$ in millions", "$ in thousands"):
        if unit in hay:
            return unit
    return None


def _row_values(row: List[str]) -> List[str]:
    """The data values of a row: column 0 is the label, the rest minus noise cells.

    pdfplumber drops blank spacer / bare-``$`` columns *inconsistently* across
    rows of the same table, so a positional read silently misaligns one row's
    2023 with another's 2022. We instead collect only the real values per row and
    rely on right-alignment (below) to put them in consistent period columns —
    financial tables are right-aligned by convention, so the rightmost value is
    always the latest period.
    """
    return [_clean(c) for c in row[1:] if _clean(c) and not _is_noise(_clean(c))]


def _value_width(grid: List[List[str]]) -> int:
    """The number of value columns = the modal count of real values across rows.

    Using the mode (not the max) ignores a stray over-split row. Rows with more
    values than this are flagged by the gate as irreconcilable rather than guessed.
    """
    counts = [len(_row_values(r)) for r in grid if _row_values(r)]
    if not counts:
        return 0
    return max(set(counts), key=counts.count)


def _stitch_fragments(grids: List[List[List[str]]]) -> List[List[List[str]]]:
    """Merge consecutive same-page fragments into coherent tables.

    HTML-derived PDFs make pdfplumber split one statement into many 1-row (or
    few-row) "tables". Adjacent fragments belong to the same table when they share
    a compatible value-width (the count of numeric values per data row). We append
    fragments into a running table as long as the width is consistent; a fragment
    with a different non-zero width starts a new table. Label-only fragments
    (section headers, zero values) attach to the current run. Native-render docs
    arrive as single multi-row grids and pass through as-is.

    This is structural (width compatibility), not filing-specific — it works for
    any document whose tables get row-fragmented, not just one vendor's filings.
    """
    if not grids:
        return []
    merged: List[List[List[str]]] = []
    current: List[List[str]] = []
    current_width: Optional[int] = None

    def flush():
        nonlocal current, current_width
        if current:
            merged.append(current)
        current, current_width = [], None

    for grid in grids:
        w = _value_width(grid)
        if w == 0:
            # label-only / header fragment: attach to the current run if one is open
            if current:
                current.extend(grid)
            else:
                merged.append(grid)  # standalone; gate will likely reject it
            continue
        if current_width is None or w == current_width:
            current.extend(grid)
            current_width = w
        else:
            flush()
            current, current_width = list(grid), w
    flush()
    return merged


def _build_rows(grid: List[List[str]], periods: List[str], width: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Turn the cleaned grid into headers + labeled row dicts via RIGHT-ALIGNMENT.

    Each row's real values are right-justified into ``width`` slots, so the last
    slot is always the latest period for every row — consistent alignment, no
    positional drift. Column names come from detected periods when their count
    matches ``width`` (the trustworthy case); otherwise positional col_N names
    (still consistently aligned, just unlabeled periods). This is a *faithful*
    projection — no value is moved between columns within a row, computed, or
    invented; we only right-justify, which financial-table convention guarantees.
    """
    if not grid or width <= 0:
        return [], []

    # period labels only when the count lines up; else positional (right-aligned).
    if len(periods) == width:
        col_names = list(periods)
    else:
        col_names = [f"col_{i + 1}" for i in range(width)]

    headers = ["section", "label"] + col_names
    rows: List[Dict[str, Any]] = []
    current_section = ""  # generic: a label-only row scopes the rows beneath it
    for r in grid:
        label = _clean(r[0]) if r else ""
        if not label:
            continue
        vals = _row_values(r)
        if not vals:
            # A label with no values is a SECTION HEADER (e.g. "AWS", "Net Sales:",
            # "Costs and expenses:"). This is universal to financial statements,
            # not filing-specific — it scopes the line items that follow so the
            # Analyst can address "AWS / Net sales" unambiguously.
            current_section = label.rstrip(":")
            continue
        # right-justify into `width` slots; over-long rows are truncated-left
        # (keep the most-recent `width` values) — but the gate rejects tables where
        # this happens often, so we never silently ship a mangled grid.
        slots = [""] * width
        for i, v in enumerate(reversed(vals[:width])):
            slots[width - 1 - i] = v
        rec: Dict[str, Any] = {"section": current_section, "label": label}
        for name, v in zip(col_names, slots):
            rec[name] = v
        rows.append(rec)
    return headers, rows


def _caption(rows: List[Dict[str, Any]], periods: List[str], units: Optional[str],
             page_number: int) -> str:
    """One-line natural-language summary — THIS is embedded for retrieval.

    Leads with the distinct SECTIONS (e.g. "AWS", "Google Cloud", "North America"),
    then the row labels and periods, so a semantic query ("AWS operating margin")
    matches the right segment table and not a same-keyword statement. Sections are
    the discriminating signal across financial tables, so naming them generically
    sharpens retrieval for any document — not just one filing's layout.
    """
    # distinct sections, in first-seen order (the segments/groupings this table covers)
    sections, seen = [], set()
    for r in rows:
        s = (r.get("section") or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            sections.append(s)
    section_part = f"Segments/groups: {', '.join(sections[:8])}. " if sections else ""

    labels, lseen = [], set()
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab and lab.lower() not in lseen:
            lseen.add(lab.lower())
            labels.append(lab)
    label_part = "; ".join(labels[:10])
    period_part = f" across {', '.join(periods)}" if periods else ""
    unit_part = f" ({units})" if units else ""
    base = (f"Financial table (page {page_number}){unit_part}{period_part}. "
            f"{section_part}Line items: {label_part}")
    return base[:500]


def _to_markdown(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    if not headers or not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_tables_from_pdf(
    file_path: str,
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[ExtractedTable]:
    """Extract confidence-gated, normalized tables from a PDF's text layer.

    Returns only tables that pass the gate. Anything ambiguous is dropped (its
    content still reaches retrieval via the normal prose/text chunk path). Never
    raises into the caller — extraction failure degrades to "no tables", which is
    exactly today's behavior, so it can never regress ingest.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[table_extraction] pdfplumber not installed — skipping table extraction")
        return []

    results: List[ExtractedTable] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    candidates = page.extract_tables() or []
                    header_years = _header_year_order(page)  # x-ordered, geometry
                except Exception as exc:  # a single bad page must not kill the doc
                    logger.debug("[table_extraction] page %d extract failed: %s", page_idx + 1, exc)
                    continue

                # Clean each candidate, then STITCH fragments. HTML-derived PDFs
                # (e.g. MSFT 10-Ks) make pdfplumber emit every table *row* as its
                # own 1-row "table"; stitching reassembles them so the gate sees a
                # real multi-row table. Native-render PDFs (Amazon/Google) already
                # come as whole tables and pass through unchanged.
                cleaned = []
                for raw in candidates:
                    if not raw:
                        continue
                    grid = [[_clean(c) for c in row] for row in raw]
                    grid = _drop_noise_columns(grid)
                    grid = _merge_currency_into_values(grid)
                    grid = [r for r in grid if any(c for c in r)]
                    if grid:
                        cleaned.append(grid)
                stitched = _stitch_fragments(cleaned)

                for t_idx, grid in enumerate(stitched):
                    confidence, breakdown = _score_grid(grid)
                    if confidence < confidence_threshold:
                        logger.debug(
                            "[table_extraction] rejected p%d t%d conf=%.2f %s",
                            page_idx + 1, t_idx, confidence, breakdown,
                        )
                        continue

                    width = _value_width(grid)
                    periods = _detect_periods(grid, header_years, width)
                    units = _detect_units(grid, page_text)
                    headers, rows = _build_rows(grid, periods, width)
                    if not rows:
                        continue

                    table = ExtractedTable(
                        page_number=page_idx + 1,
                        table_id=f"p{page_idx + 1}_t{t_idx}",
                        headers=headers,
                        rows=rows,
                        units=units,
                        periods=periods,
                        caption=_caption(rows, periods, units, page_idx + 1),
                        confidence=confidence,
                        raw_grid=grid,
                        markdown=_to_markdown(headers, rows),
                    )
                    results.append(table)
    except Exception as exc:
        logger.warning("[table_extraction] failed to open %s: %s — no tables extracted", file_path, exc)
        return []

    logger.info("[table_extraction] %s → %d gated tables", file_path, len(results))
    return results


# ── LLM discriminative summary (caller-invoked at ingest; extraction stays pure) ──
#
# The deterministic _caption is built from sections + labels + periods, which are
# IDENTICAL across a table and its near-twins (a $ segment table, its growth-%
# twin, a mix-% table all say "AWS / Net sales / 2021-2023"). That non-discriminative
# caption is what makes the Analyst pick the wrong grid (~60-70% selection). A one-line
# LLM summary that names the STATEMENT TYPE + VALUE KIND ("net sales in DOLLARS" vs
# "year-over-year GROWTH PERCENTAGES") is what separates the twins — so retrieval ranks
# the right statement #1 and the Analyst selects correctly. The number itself is never
# touched here (no arithmetic, no value echo) — this only describes the table.

_SUMMARY_SYSTEM = (
    "You write a ONE-SENTENCE discriminative description of a financial table so it can be "
    "told apart from near-identical sibling tables in the same filing. The description MUST "
    "name explicitly, in this priority order:\n"
    "1. the SEGMENTATION — if the table is broken out by business segments, geographies, or "
    "entities (these are given to you as 'Sections/groups', e.g. AWS, North America, "
    "International), SAY SO and name them. A segment table and the consolidated table look "
    "identical by their line items — the segmentation is what separates them. If there are no "
    "sections, say it is consolidated/total.\n"
    "2. the VALUE KIND — say which one: dollar amounts, GROWTH PERCENTAGES (year-over-year %), "
    "margins/ratios (%), share counts, or balances. A dollar table and its growth-% twin are "
    "identical except for this — never omit it.\n"
    "3. the STATEMENT TYPE (e.g. segment results, income statement, balance sheet, cash flow, "
    "geographic revenue, share/RSU schedule, lease maturity).\n"
    "4. the period span (the years/quarters), if any.\n"
    "Trust the 'Sections/groups' line for segmentation even if the row labels resemble a "
    "consolidated statement. Be factual and specific. Do NOT compute, restate, or invent any "
    "number. Output ONLY the one-sentence description — no prose, no labels, no markdown."
)


def summarize_table(table: "ExtractedTable", llm) -> str:
    """Ask the LLM for a one-sentence discriminative summary of a table. Never raises.

    Caller-invoked at ingest (keeps ``extract_tables_from_pdf`` LLM-free and testable).
    Uses a plain ``invoke([System, Human])`` — NO ChatPromptTemplate (its ``{}`` brace
    handling mangles the table markdown; see Build Log Entry 4). Returns ``""`` on any
    failure so the caller falls back to the deterministic ``caption`` — a summary failure
    can never drop a table or break ingest.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        sections, seen = [], set()
        for r in table.rows:
            s = (r.get("section") or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                sections.append(s)
        user = (
            f"Periods: {table.periods}\n"
            f"Units: {table.units}\n"
            f"Sections/groups: {', '.join(sections[:12]) or '(none)'}\n"
            f"Table (markdown):\n{(table.markdown or '')[:2000]}"
        )
        resp = llm.invoke([SystemMessage(content=_SUMMARY_SYSTEM), HumanMessage(content=user)])
        text = (resp.content or "").strip()
        # one sentence only; strip any stray fencing/quoting the model may add
        text = text.strip("`").strip().strip('"').strip()
        return text[:500]
    except Exception as exc:  # network/timeout/parse — degrade to deterministic caption
        logger.debug("[table_extraction] summary failed for %s: %s", getattr(table, "table_id", "?"), exc)
        return ""
