"""Source an Indian statute from its official India Code PDF (G8 §G8.1b).

India Code publishes each Act as a PDF that begins with an **ARRANGEMENT OF SECTIONS**
(the authoritative table of contents) and then the enacted body. This helper turns that
PDF into the two inputs the statute parser needs:

  1. `toc_ids`  — the authoritative arrangement-of-sections id list (the §G8.4 completeness
                  ground truth AND the parser's decisive footnote discriminator).
  2. `body_text` — the enacted text (everything from the SECOND title line onward), which
                  the parser walks for provisions.

Splitting them is what makes the ingest honest: completeness is measured against the Act's
OWN published ToC, not against "whatever happened to parse" (the G1/finance lesson, applied
to law). Pure text work — pdfplumber for the text layer, regex for the split. $0, no API.

Used by `scripts/ingest_statute.py` when `--pdf` is given; a pre-extracted `--text` file
can be passed instead (the caller then supplies toc_ids, or falls back to the em-dash
discriminator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .parse_constitution import repair_encoding


@dataclass
class StatuteSource:
    """The split inputs for the parser, plus the honesty trail."""
    title_line: str
    toc_ids: List[str]
    body_text: str
    warnings: List[str]


def extract_pdf_text(pdf_path: str) -> str:
    """Extract the text layer of an India Code statute PDF via pdfplumber (the same library
    the finance extractor uses). Page texts joined by newlines. Never raises on a blank
    page (returns '')."""
    import pdfplumber  # lazy — only when sourcing a PDF
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


# An arrangement-of-sections ToC entry: "10. Title." / "19A. Title." at line start. In the
# ToC the title is followed by a period (and often a sub-heading line beneath, which we
# ignore — only the numbered lines are Section ids).
_TOC_ENTRY_RE = re.compile(r"^\s*(\d+[A-Z]{0,3})\.\s+\S")


def split_arrangement_and_body(text: str, title_line: str) -> Tuple[List[str], str, List[str]]:
    """Split an India Code statute text into (toc_ids, body_text, warnings).

    The body starts at the SECOND occurrence of the Act's title line ("THE INDIAN CONTRACT
    ACT, 1872") — the first heads the ARRANGEMENT OF SECTIONS, the second heads the enacted
    text (preceded by "ACT NO. ... OF ..." / "[date]" / "Preamble"). Everything before the
    second title is the ToC; we harvest its numbered entries as the authoritative id list.
    If the title appears only once (some Acts), we fall back to the "ACT NO." marker; if
    that too is absent, the whole text is treated as body and toc_ids is empty (the parser
    then uses the em-dash discriminator) — flagged in warnings, never silent.
    """
    warnings: List[str] = []
    first = text.find(title_line)
    second = text.find(title_line, first + len(title_line)) if first >= 0 else -1
    if second < 0:
        # Fall back to the enacting marker.
        m = re.search(r"ACT\s+NO\.\s+\d+\s+OF\s+\d{4}", text)
        second = m.start() if m else -1
    if second < 0:
        warnings.append("no ToC/body split found (single-title Act) — using em-dash discriminator")
        return [], text, warnings

    toc_text, body_text = text[:second], text[second:]

    ids: List[str] = []
    seen = set()
    for ln in toc_text.split("\n"):
        m = _TOC_ENTRY_RE.match(ln)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    if not ids:
        warnings.append("ToC found but no numbered entries parsed — using em-dash discriminator")
    return ids, body_text, warnings


def source_statute_pdf(pdf_path: str, title_line: str) -> StatuteSource:
    """End-to-end: PDF → repaired text → (toc_ids, body). `title_line` is the Act's title as
    it appears at the top of the PDF (e.g. 'THE INDIAN CONTRACT ACT, 1872'). $0, offline."""
    with open(pdf_path, "rb") as fh:
        _ = fh  # presence check; pdfplumber re-opens by path
    raw = extract_pdf_text(pdf_path)
    text = repair_encoding(raw.encode("utf-8", errors="replace"))
    toc_ids, body_text, warnings = split_arrangement_and_body(text, title_line)
    return StatuteSource(title_line=title_line, toc_ids=toc_ids, body_text=body_text, warnings=warnings)


def source_statute_text(text_path: str, title_line: str) -> StatuteSource:
    """Same split over a pre-extracted plain-text file (the `--text` path)."""
    raw = open(text_path, "rb").read()
    text = repair_encoding(raw)
    toc_ids, body_text, warnings = split_arrangement_and_body(text, title_line)
    return StatuteSource(title_line=title_line, toc_ids=toc_ids, body_text=body_text, warnings=warnings)
