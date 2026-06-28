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

# A numbered clause heading: "8", "8.2", "8.2.1" then a capitalized word.
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z].{0,80})$")
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
    for rx in (_NUMBERED, _ITEM, _MARKER):
        m = rx.match(s)
        if m:
            return " ".join(p.strip() for p in m.groups() if p and p.strip())
    m = _ALLCAPS.match(s)
    if m:
        # Reject lines that are mostly digits/punctuation (page chrome, timestamps).
        letters = sum(ch.isalpha() for ch in s)
        if letters >= 4:
            return s.strip()
    return None


def build_outline(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect headings across ordered text chunks → an outline list.

    `chunks` is [{content, page, chunk_index}] already sorted (from _chunks). Returns
    [{heading, page, chunk_index}] in document order. One entry per detected heading.
    """
    outline: List[Dict[str, Any]] = []
    for c in chunks:
        page = c.get("page")
        ci = c.get("chunk_index")
        for line in re.split(r"\n+", c.get("content", "") or ""):
            h = _line_heading(line)
            if h:
                outline.append({"heading": h, "page": page, "chunk_index": ci})
    return outline


def slice_by_heading(
    chunks: List[Dict[str, Any]],
    heading: str,
) -> List[Dict[str, Any]]:
    """Return the chunks from the chunk whose text contains `heading` up to the NEXT
    detected heading (the section body). Fuzzy: case-insensitive substring on a
    detected-heading line. Empty if the heading isn't found.
    """
    want = heading.strip().lower()
    if not want:
        return []
    start_idx: Optional[int] = None
    for i, c in enumerate(chunks):
        for line in re.split(r"\n+", c.get("content", "") or ""):
            h = _line_heading(line)
            if h and (want in h.lower() or h.lower() in want):
                start_idx = i
                break
        if start_idx is not None:
            break
    if start_idx is None:
        return []
    # Walk forward to the next chunk that STARTS a new detected heading.
    out = [chunks[start_idx]]
    for c in chunks[start_idx + 1:]:
        has_heading = any(
            _line_heading(line) for line in re.split(r"\n+", c.get("content", "") or "")
        )
        if has_heading:
            break
        out.append(c)
    return out
