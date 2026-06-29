"""Runtime heading/outline detection over clean chunk text (DOCUMENT_HARNESS §6.4).

The section heading IS computed during legal-prose chunking but thrown away before
storage (data_ingestion.py — the §0.2 gap). Phase 1 rebuilds the outline ON THE FLY
from the concatenated text by detecting headings; Phase 1.5 will persist `section_path`
and make this exact. Until then `page_range` is exact and `heading` is heuristic but
good enough to scope a big doc to its section.

Detected heading shapes (the common contract/filing forms):
  - numbered clauses:  `8`, `8.2`, `8.2.1` + a Title-Case/UPPER lead
  - ALL-CAPS lines (short)
  - `Item 7`, `Item 7A` (10-K)
  - `Schedule N` / `Exhibit N` / `Annex N` / `Appendix N` markers

No intelligence beyond pattern matching. Pure, never raises.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# A numbered clause heading: "8", "8.2", "8.2.1", "19 Governing Law" then a Title-Case
# word. Bare integers ARE allowed (real legal headings use "19 Governing Law", no dot) —
# the finance noise this used to catch ("2021 Acquisition Activity", "12 Months or
# Greater") is kept out instead by (a) the _YEAR_LEAD guard below and (b) scoping the
# backfill to legal_contract docs, NOT by forbidding the bare-int form (that threw out
# real clause headings — verified against the live corpus). The clause number is small
# (<=3 digits); a longer leading number is a value/id, not a clause.
_NUMBERED = re.compile(r"^\s*(\d{1,3}(?:\.\d+)*)\.?\s+([A-Z][A-Za-z].{0,80})$")
# A year-led line ("2021 Acquisition Activity", "2014 Notes issuance …") is a data/period
# row, never a clause heading — reject before _NUMBERED can match it.
_YEAR_LEAD = re.compile(r"^\s*(19|20)\d{2}\b")
# 10-K style item heading.
_ITEM = re.compile(r"^\s*(Item\s+\d+[A-Z]?)\b(.{0,80})$", re.IGNORECASE)
# Schedule/Exhibit/Annex/Appendix markers.
_MARKER = re.compile(r"^\s*((?:Schedule|Exhibit|Annex|Appendix)\s+[A-Z0-9]+)\b(.{0,80})$",
                     re.IGNORECASE)
# A short ALL-CAPS line is a section header (e.g. "LIMITATION OF LIABILITY").
_ALLCAPS = re.compile(r"^\s*([A-Z][A-Z0-9 ,&/'()\-]{3,70})\s*$")


def _line_heading(line: str) -> Optional[str]:
    """Return a normalized heading string if `line` looks like a heading, else None."""
    s = line.strip()
    if not s or len(s) > 90:
        return None
    if _YEAR_LEAD.match(s):
        return None  # "2021 Acquisition Activity" / "2014 Notes issuance" — a value row
    for rx in (_NUMBERED, _ITEM, _MARKER):
        m = rx.match(s)
        if m:
            return " ".join(p.strip() for p in m.groups() if p and p.strip())
    m = _ALLCAPS.match(s)
    if m:
        # Reject lines that are mostly digits/punctuation (page chrome, timestamps).
        letters = sum(ch.isalpha() for ch in s)
        if letters < 4:
            return None
        words = s.split()
        # A LONE short token is an acronym in the body (GDPR, APAC, BRSR, CAGR), not a
        # section header — verified against the live corpus. Real ALL-CAPS headings are
        # multi-word ("HUMAN CAPITAL RESOURCES", "PART I") or a longer single word.
        if len(words) == 1 and len(s) <= 6:
            return None
        # Repeated-token lines ("MSFT MSFT MSFT") are watermarks/artifacts, not headings.
        if len(set(words)) == 1 and len(words) > 1:
            return None
        return s.strip()
    return None


def build_outline(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect headings across ordered text chunks → an outline list.

    `chunks` is [{content, page, chunk_index, section_path?}] already sorted (from
    _chunks). Returns [{heading, page, chunk_index}] in document order. Phase 1.5: when a
    chunk carries a persisted `section_path`, use it EXACTLY (one outline entry for the
    chunk); else fall back to detecting headings in the chunk text (the Phase-1 heuristic).
    """
    outline: List[Dict[str, Any]] = []
    for c in chunks:
        page = c.get("page")
        ci = c.get("chunk_index")
        sp = c.get("section_path")
        if sp:
            outline.append({"heading": sp, "page": page, "chunk_index": ci})
            continue
        for line in re.split(r"\n+", c.get("content", "") or ""):
            h = _line_heading(line)
            if h:
                outline.append({"heading": h, "page": page, "chunk_index": ci})
    return outline


def _chunk_heading(c: Dict[str, Any]) -> Optional[str]:
    """The chunk's heading: the persisted `section_path` (Phase 1.5, exact) if present,
    else the first detected-heading line in its text (Phase-1 heuristic fallback)."""
    sp = c.get("section_path")
    if sp:
        return sp
    for line in re.split(r"\n+", c.get("content", "") or ""):
        h = _line_heading(line)
        if h:
            return h
    return None


def slice_by_heading(
    chunks: List[Dict[str, Any]],
    heading: str,
) -> List[Dict[str, Any]]:
    """Return the chunks from the chunk headed by `heading` up to the NEXT heading (the
    section body). Fuzzy: case-insensitive substring match. Phase 1.5: matches against a
    persisted `section_path` when present, else a detected-heading line. Empty if not found.
    """
    want = heading.strip().lower()
    if not want:
        return []
    start_idx: Optional[int] = None
    for i, c in enumerate(chunks):
        h = _chunk_heading(c)
        if h and (want in h.lower() or h.lower() in want):
            start_idx = i
            break
    if start_idx is None:
        return []
    # Walk forward to the next chunk that STARTS a new section.
    out = [chunks[start_idx]]
    for c in chunks[start_idx + 1:]:
        if _chunk_heading(c):
            break
        out.append(c)
    return out
