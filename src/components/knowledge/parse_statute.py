"""Parse an Indian statute's plain-text into provision-granular `Provision`s
(G8 §G8.1b — the structured-source path, statute flavour).

This is the GENERAL statute parser — the Constitution's twin (`parse_constitution.py`)
generalised to the Indian-Act convention the bulk of the Tier-0 corpus follows. It is
proven first on the **Indian Contract Act 1872** (the named bedrock for the contract-
review grid G4/G6 cites), then templated across the rest (Companies Act 2013, SEBI/RBI,
GST, Income-Tax, IT Act, Arbitration) — same prove-on-one-then-scale discipline as G8.1a.

WHY a statute-specific parser, not the Constitution one: the addressing differs.

    Constitution            Statute (this file)
    ───────────             ────────────────────
    Article 14              Section 14 (cited "<Act> s.14")
    Part III                Chapter III   (+ optional Part)
    Schedule (Roman/ord)    Schedule (numbered / "THE SCHEDULE")
    "14. Title.—Body"       "14. Title.—Body"   ← same opener shape, reused

The opener shape is shared (a leading number, an em-dash title→body split), so the
hard-won Constitution machinery — the footnote discriminator, the em-dash splitter, the
encoding repair, the reflow normalizer — is REUSED verbatim (imported, not copied). The
deltas this parser adds are purely structural:

  - Sections cite "<Act> s.N" (the `s.` joiner), not "Art.N".
  - CHAPTER / PART headers set the ToC path (statutes nest Chapters; some add Parts).
  - A leading enacting formula / long-title preamble before s.1 is front matter, not a
    provision (dropped to warnings, like the Constitution's pre-Article text).
  - Section ids carry the Indian letter-suffix convention (e.g. "10A", "73") natively
    via the shared `\d+[A-Z]{0,3}` id pattern.

$0 and offline — pure text → dataclasses. It does NOT embed and does NOT touch Supabase
or Pinecone; the driver (`scripts/ingest_statute.py`) decides dry-run vs live. Permissive
and NEVER raises: an unclassifiable line folds into the current Section's body; anything
genuinely unplaceable is reported in `ParseResult.warnings`, never silently dropped.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Reuse the Constitution parser's hard-won, statute-agnostic primitives verbatim — the
# opener shape, the footnote discriminator, the em-dash title splitter, the encoding
# repair, and the reflow normalizer are identical for any Indian numbered instrument.
from .parse_constitution import (
    ParseResult,
    _is_real_article_opener,
    _split_title_body,
    repair_encoding,  # re-exported for the driver
)
from .provision import Provision, SourceMeta

__all__ = ["parse_statute_text", "normalize_statute_text", "repair_encoding"]

# A Section opener: "14." / "10A." / "73." at line start, then a Title, then the body.
# Identical id grammar to the Constitution's Article opener (the Indian letter-suffix
# convention — 10A, 73, 87 — is shared). The `\s*` after the period (NOT `\s+`) tolerates
# India Code PDFs that drop the space after the number ("73.Compensation..."); the
# whitespace at the front is normalized by `_clean_line_start` before matching.
_SECTION_RE = re.compile(r"^\s*(\d+[A-Z]{0,3})\.\s*(\S.*)$", re.DOTALL)

# India Code's amendment convention prefixes a substituted/inserted Section with a
# superscript FOOTNOTE NUMBER and an opening square bracket: "1[16.“Undue influence”..."
# means "footnote 1: this text was substituted, and it is Section 16." pdfplumber renders
# the superscript inline, so a Section opener arrives buried as "1[16." / "23[16A." — the
# bare `_SECTION_RE` never sees it. `_clean_line_start` strips that leading
# `<digits>[`-marker (an amendment-span open before a section number) so the opener
# surfaces. It is deliberately narrow: it ONLY fires on `<digits>[<digit>` so it can never
# eat a real "[" that opens bracketed body text or a repealed stub.
_FN_SECTION_PREFIX_RE = re.compile(r"^(\d+)\[(?=\d)")


def _clean_line_start(line: str) -> str:
    """Strip a leading amendment-footnote marker ("1[" before a section number) so an
    India-Code-buried Section opener surfaces. Idempotent; leaves every other line
    untouched (a line that doesn't start `<digits>[<digit>` returns unchanged)."""
    return _FN_SECTION_PREFIX_RE.sub("", line, count=1)

# A Chapter header: "CHAPTER III" / "CHAPTER IVA" / "CHAPTER 2", optionally with the
# Chapter TITLE on the same line ("CHAPTER I PRELIMINARY" — which is how the title lands
# once the reflow normalizer collapses a wrapped blob). Group 1 = the chapter number,
# group 2 = the (optional) inline title, which must be ALL-CAPS so a body sentence can
# never look like a header. Chapters are the dominant statute division (the Constitution
# used Parts; most Acts use Chapters, some add Parts too).
_CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+[A-Z]?|\d+[A-Z]?)\b\s*([A-Z][A-Z ,.&'\-]{2,})?\s*$"
)

# A Part header (some Acts — e.g. the GST/Income-Tax families — nest Parts inside Chapters
# or use Parts at top level). Same grammar as the Constitution's Part header.
_PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+[A-Z]?|\d+[A-Z]?)\s*$", re.IGNORECASE)

# A Schedule HEADER: "THE SCHEDULE" (the common single-schedule case), "FIRST SCHEDULE",
# "SCHEDULE I", "SCHEDULE 1" — a true header, NOT a body sentence that merely references a
# schedule ("...the Second Schedule, ibid." in a repeal footnote, which the loose form
# wrongly matched and then poisoned EVERY following Section by latching `in_schedule`).
# Discriminator (learned on the real Act, 2026-06-21): after the SCHEDULE keyword the line
# must END, or continue ONLY with a header terminator — a bracketed annotation `[...]`, an
# em-dash heading `.—`/`—`, a colon, or the word ending the title. A comma + lowercase word
# (",ibid", ", as") is a reference, never a header, so it cannot match.
_SCHEDULE_RE = re.compile(
    r"^\s*(?:THE\s+)?"
    r"(?:(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+SCHEDULE"
    r"|SCHEDULE(?:\s+([IVXLCDM]+|\d+))?)"
    r"(?P<tail>\s*$|\s*[\[:]|\s*\.?[—–].*|\s+[A-Z].*)",
    re.IGNORECASE,
)

# A repeal/omission marker for a Section: "[s. 31 ... omitted/repealed ...]" or
# "31. [Repealed.]". Mirrors the Constitution's omission handling.
_OMITTED_RE = re.compile(
    r"^\s*\[?\s*(?:s(?:ec(?:tion)?)?\.?\s*)?(\d+[A-Z]{0,3})\b.*\b(?:omitted|repealed|rep\.)\b",
    re.IGNORECASE,
)

# A sub-clause marker — "(1)", "(a)", "(iv)" — that must NOT be folded into the previous
# line by the reflow normalizer (it starts a new logical line in the source).
_SUBCLAUSE_RE = re.compile(r"^\s*\((?:\d+[A-Z]?|[a-z]{1,3}|[ivxlcdm]+)\)")

_ORDINAL_TO_NUM = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
    "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
}
_ROMAN_TO_NUM = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}


def _digits(s: str) -> Optional[int]:
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else None


def normalize_statute_text(text: str) -> str:
    """Reflow hard-wrapped statute text so each Section/Chapter/Part/Schedule/sub-clause is
    on one logical line — the shape `parse_statute_text` expects. A line is a CONTINUATION
    of the previous unless it opens a new structural element (Section number / Chapter /
    Part / Schedule header) or a sub-clause marker. Collapses wrap whitespace; keeps blank
    lines as paragraph breaks. Idempotent on already-normalized text. (The Constitution's
    twin tuned for Article/Part; this one also recognises Chapter/Section headers.)"""
    out: List[str] = []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = _clean_line_start(raw.strip())
        if not line:
            out.append("")
            continue
        starts_structure = bool(
            _SECTION_RE.match(line) or _CHAPTER_RE.match(line) or _PART_RE.match(line)
            or _SCHEDULE_RE.match(line) or _SUBCLAUSE_RE.match(line)
        )
        if starts_structure or not out or out[-1] == "":
            out.append(line)
        else:
            out[-1] = re.sub(r"\s+", " ", out[-1] + " " + line)
    return "\n".join(out)


def parse_statute_text(
    text: str,
    *,
    source_key: str,
    title: str,
    citation_prefix: str,
    enacted_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
    source_url: Optional[str] = None,
    toc_ids: Optional[List[str]] = None,
) -> ParseResult:
    """Parse normalized statute plain-text into a `ParseResult` at Section granularity.

    `citation_prefix` is the short Act name provisions cite under (e.g. "Indian Contract
    Act 1872" → "Indian Contract Act 1872 s.10"). `as_of_date` is the snapshot horizon you
    vouch for — REQUIRED for the live ingest (`Provision.is_valid()` rejects a provision
    without one); the parser leaves it None if not given so a dry-run still runs (ingest
    then rejects loudly, never silently).

    `toc_ids` is the AUTHORITATIVE arrangement-of-sections id list (captured separately from
    the Act's own ToC page — see `source_statute.py`). When given, it is BOTH the §G8.4
    completeness ground truth AND the decisive footnote discriminator: a numbered opener
    whose id is NOT in the ToC is a page-bottom footnote ("1. Subs. by Act ..."), not a
    Section, so it folds into the current provision's body. Indian statute PDFs interleave
    footnotes heavily (the Constitution parse hit the same class); an em-dash discriminator
    alone misses footnotes that happen to contain a dash. The ToC allow-list is exact. When
    omitted, the parser falls back to the em-dash discriminator (the dry-run / fixture path).

    Structure handled: CHAPTER / PART headers set the ToC path; numbered Sections become
    Provisions (citation "<Act> s.N"); Schedules become one opaque `schedule` Provision
    each (cited as a unit, never sub-parsed — the Constitution's hard-won rule). A leading
    enacting formula / long title before s.1 is front matter → warnings, not a provision.
    """
    warnings: List[str] = []
    provisions: List[Provision] = []
    toc_allow = set(toc_ids) if toc_ids else None

    cur_chapter: Optional[str] = None
    cur_chapter_title: Optional[str] = None
    cur_part: Optional[str] = None
    cur: Optional[Provision] = None
    repealed_ids: set = set()
    in_schedule = False
    sched_seq = 0  # for "THE SCHEDULE" with no number → Schedule.1, .2, ...

    def _flush(p: Optional[Provision]) -> None:
        if p is not None:
            p.text = (p.text or "").strip()
            provisions.append(p)

    def _toc_path_for(sec_id: str) -> str:
        parts: List[str] = []
        if cur_chapter:
            parts.append("Chapter " + cur_chapter + (f" ({cur_chapter_title})" if cur_chapter_title else ""))
        if cur_part:
            parts.append("Part " + cur_part)
        parts.append(f"s.{sec_id}")
        return " > ".join(parts)

    lines = (text or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line:
            if cur is not None:
                cur.text += "\n"
            continue

        # ── repeal/omission marker (not itself a Section opener) ─────────────────
        mo = _OMITTED_RE.match(line)
        if mo and _SECTION_RE.match(line) is None:
            repealed_ids.add(mo.group(1))
            for p in provisions:
                if p.section_or_article_id == mo.group(1):
                    p.repealed = True
            continue

        # ── Chapter header ──────────────────────────────────────────────────────
        mc = _CHAPTER_RE.match(line)
        if mc:
            _flush(cur); cur = None
            in_schedule = False
            cur_chapter = mc.group(1).upper()
            cur_part = None
            # The title is either inline on the header line (group 2 — how it lands once the
            # reflow collapses a wrapped blob) or on the following all-caps/title-case line.
            inline_title = (mc.group(2) or "").strip()
            cur_chapter_title = inline_title.title() if inline_title else None
            if cur_chapter_title is None and i < len(lines) and lines[i].strip() and not _SECTION_RE.match(lines[i].strip()):
                nxt = lines[i].strip()
                if nxt.isupper() or nxt.istitle():
                    cur_chapter_title = nxt.title()
                    i += 1
            continue

        # ── Part header ─────────────────────────────────────────────────────────
        mp = _PART_RE.match(line)
        if mp:
            _flush(cur); cur = None
            in_schedule = False
            cur_part = mp.group(1).upper()
            continue

        # ── Schedule header ───────────────────────────────────────────────────
        ms = _SCHEDULE_RE.match(line)
        if ms:
            _flush(cur)
            in_schedule = True
            sched_seq += 1
            ordn = (ms.group(1) or ms.group(2) or "").upper()
            num = (
                _ORDINAL_TO_NUM.get(ordn)
                or _ROMAN_TO_NUM.get(ordn)
                or _digits(ordn)
                or sched_seq
            )
            sid = f"Schedule.{num}"
            # Any text trailing the header on the same line (the normalizer collapses a
            # wrapped blob, so "THE SCHEDULE Enactments repealed.—..." arrives as one line)
            # is the START of the Schedule body, not a new structure.
            tail = (ms.groupdict().get("tail") or "").strip()
            cur = Provision(
                source_key=source_key, instrument_type="schedule",
                title=(f"{ordn.title()} Schedule" if ordn else "Schedule"),
                citation=f"{citation_prefix} {sid}",
                section_or_article_id=sid, text=tail,
                toc_path=f"Schedules > {sid}",
            )
            continue

        # ── Section opener ────────────────────────────────────────────────────────
        # Inside a Schedule, a numbered line is a Schedule list item (body), not a Section.
        msec = None if in_schedule else _SECTION_RE.match(line)
        recovered_id: Optional[str] = None
        # Reject footnote/continuation lines masquerading as openers. Two discriminators,
        # the ToC allow-list (exact, decisive) taking precedence when present:
        #   1. If we hold the authoritative arrangement-of-sections (`toc_allow`), an opener
        #      whose id is NOT a real Section id is a page-bottom footnote → fold to body.
        #      BUT first try to RECOVER a buried opener: a bracket-less amendment footnote
        #      superscript merges into the number ("1151. Care..." = footnote 1 + Section
        #      151), so if stripping 1–2 leading digits yields a real ToC id, that's the
        #      Section (disambiguated by the ToC — "1151" itself is never a real id).
        #   2. Otherwise fall back to the shared em-dash heuristic (`Title.—Body` or a
        #      repealed stub) — the fixture / no-ToC path.
        if msec and toc_allow is not None and msec.group(1) not in toc_allow:
            raw_id = msec.group(1)
            for strip_n in (1, 2):
                cand = raw_id[strip_n:]
                if cand and cand in toc_allow:
                    recovered_id = cand
                    break
            if recovered_id is None:
                msec = None
        elif msec and toc_allow is None and not _is_real_article_opener(msec.group(2)):
            msec = None
        if msec:
            _flush(cur)
            sec_id = recovered_id or msec.group(1)
            rest = msec.group(2)
            is_repealed_stub = bool(re.match(r"^\s*\[[^\]]*\]\s*(?:Rep\.|Omitted|Repealed)", rest, re.IGNORECASE))
            sec_title, body = _split_title_body(rest)
            if is_repealed_stub:
                mb = re.match(r"^\s*\[([^\]]*)\]\s*(.*)$", rest)
                if mb:
                    sec_title, body = mb.group(1).strip().rstrip("."), mb.group(2).strip()
            cur = Provision(
                source_key=source_key, instrument_type="act",
                title=sec_title or f"Section {sec_id}",
                citation=f"{citation_prefix} s.{sec_id}",
                section_or_article_id=sec_id, text=body,
                repealed=(is_repealed_stub or sec_id in repealed_ids),
                toc_path=_toc_path_for(sec_id),
            )
            continue

        # ── continuation line ───────────────────────────────────────────────────
        if cur is not None:
            cur.text += (" " if cur.text and not cur.text.endswith("\n") else "") + line
        else:
            # Text before the first Section (enacting formula / long title) — front matter.
            warnings.append(f"unplaced line before first provision: {line[:60]!r}")

    _flush(cur)

    # Apply repeal markers that appeared before their Section was seen.
    for p in provisions:
        if p.section_or_article_id in repealed_ids:
            p.repealed = True

    # De-duplicate by id, KEEPING THE FIRST occurrence (the genuine Section precedes any
    # page-bottom footnote that reused its number). Dropped dups are reported, never silent
    # — the same honesty rule the Constitution parser enforces.
    seen: set = set()
    deduped: List[Provision] = []
    dropped: List[str] = []
    for p in provisions:
        key = p.section_or_article_id
        if key in seen:
            dropped.append(key)
            continue
        seen.add(key)
        deduped.append(p)
    if dropped:
        warnings.append(f"dropped {len(dropped)} duplicate-id provision(s) (kept first): {sorted(set(dropped))}")
    provisions = deduped

    # The source's authoritative ToC: the supplied arrangement-of-sections when given (so
    # §G8.4 completeness diffs against the Act's OWN ToC, catching a Section we failed to
    # parse), else the parsed-id list (the fixture path). Schedules parsed from the body are
    # appended so they're counted as covered, never reported missing.
    parsed_ids = [p.section_or_article_id for p in provisions]
    if toc_ids:
        sched_ids = [i for i in parsed_ids if i.startswith("Schedule.")]
        final_toc = list(toc_ids) + [s for s in sched_ids if s not in set(toc_ids)]
    else:
        final_toc = parsed_ids

    source = SourceMeta(
        source_key=source_key, title=title, instrument_type="act",
        citation_prefix=citation_prefix, enacted_date=enacted_date,
        as_of_date=as_of_date, toc_ids=final_toc, source_url=source_url,
    )
    return ParseResult(source=source, provisions=provisions, warnings=warnings)
