"""Parse the Constitution of India plain-text into provision-granular `Provision`s
(G8 §G8.1a — the structured-source path).

WHY a dedicated parser, not the G1 prose chunker: the Constitution has a *clean,
machine-readable* structure (numbered Articles, Parts, Schedules), so the plan's rule is
to ingest it at PROVISION granularity directly via `ingest_provisions` — the cleanest,
most measurable path (one chunk == one Article == one row == one ToC id). The prose
chunker (`ingest_text_elements`) is the fallback for sources that arrive as messy PDFs.

This parser is $0 and offline — pure text → dataclasses. It does NOT embed, does NOT
touch Supabase or Pinecone; the driver (`scripts/ingest_constitution.py`) decides whether
to dry-run (in-memory) or do the live embed.

Input format (the India Code / legislative.gov.in convention, normalized to plain text):

    PART III
    FUNDAMENTAL RIGHTS
    14. Equality before law.—The State shall not deny to any person ...
    15. Prohibition of discrimination ...—(1) The State shall not ...
    [Art. 31 omitted by the Constitution (Forty-fourth Amendment) Act, 1978]

  - A line beginning `<number><optional letter>. <Title>.—<body>` starts an Article.
  - `PART <roman/number>` / a following all-caps line set the current ToC path.
  - A bracketed `[Art. N omitted/repealed ...]` line marks N repealed.
  - SCHEDULES are parsed as `instrument_type="schedule"` with id like "Schedule.1".

The parser is permissive (real source text is irregular) and NEVER raises — a line it
can't classify is appended to the current Article's body. Anything it genuinely can't
place is reported in `ParseResult.warnings`, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .provision import Provision, SourceMeta

# An Article opener: "14." or "21A." or "243ZG." at line start, then a Title up to the
# first sentence break / em-dash, then the body. The title runs to the em-dash (—) or
# the first period that ends the heading; we keep it loose and trim later.
_ARTICLE_RE = re.compile(r"^\s*(\d+[A-Z]{0,3})\.\s+(.*)$", re.DOTALL)

# A Part header: "PART III" / "PART IVA". The line after it is usually the Part title.
_PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+[A-Z]?|\d+[A-Z]?)\s*$", re.IGNORECASE)

# A Schedule HEADER line: "FIRST SCHEDULE" / "THE SEVENTH SCHEDULE [Articles 1 and 4]".
# Anchored at line start AND length-bounded via the bracket-only tail so a body SENTENCE
# that merely mentions "...the Tenth Schedule" is NOT mistaken for a header (the greedy
# `.*` form did exactly that — 6 spurious schedules; dry-run caught it 2026-06-17). The
# tail may only be a bracketed annotation, nothing else.
_SCHEDULE_RE = re.compile(
    r"^\s*(?:THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|\d+(?:ST|ND|RD|TH)?)\s+SCHEDULE(?:\s*\[[^\]]*\])?\s*$",
    re.IGNORECASE,
)

# A repeal/omission marker for an Article: "[Art. 31 ... omitted/repealed ...]".
_OMITTED_RE = re.compile(r"^\s*\[?\s*(?:Art(?:icle)?\.?\s*)?(\d+[A-Z]{0,3})\b.*\b(?:omitted|repealed|rep\.)\b", re.IGNORECASE)

# A REAL Article opener (vs a page-bottom footnote that also starts "N. ..."). The public
# text interleaves amendment footnotes ("1. Subs. by the Constitution ...Act, 1976") and
# case-note footnotes ("2. In Kesavananda Bharati vs. ...") that look like Article openers
# but are NOT provisions. Discriminator (validated on the real text, 2026-06-17: 438/480
# openers are real): a real Article's heading is `Title.—Body` (em-dash within the first
# ~160 chars) OR a repealed stub `[Title.] Rep./Omitted by ...`. Anything else after `N.`
# is footnote/continuation noise → folded into the current provision's body, never a new
# Article. (Cheaper + more precise than a footnote denylist, which would be open-ended.)
_REAL_ARTICLE_HEAD = re.compile(r"^.{0,160}?[—–]")            # has the title→body em-dash
_REPEALED_STUB = re.compile(r"^\s*\[[^\]]*\]\s*(?:Rep\.|Omitted|Repealed|Rep\b)", re.IGNORECASE)


def _is_real_article_opener(rest: str) -> bool:
    """True iff the text after 'N. ' begins a genuine Article (not a footnote)."""
    return bool(_REAL_ARTICLE_HEAD.match(rest) or _REPEALED_STUB.match(rest))

_ORDINAL_TO_NUM = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
    "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11, "TWELFTH": 12,
}


# A sub-clause marker — "(1)", "(a)", "(iv)" — which starts a NEW line in the source and
# must NOT be folded into the previous line by the reflow normalizer.
_SUBCLAUSE_RE = re.compile(r"^\s*\((?:\d+[A-Z]?|[a-z]{1,3}|[ivxlcdm]+)\)")


def repair_encoding(raw: bytes) -> str:
    """Decode source bytes that mix two real-world defects in the public Constitution text:

      1. A hard line-wrap that lands INSIDE a multi-byte UTF-8 char (e.g. the em-dash
         e2 80 94 split as `e2 80 \\n 94`) — we drop a newline sitting between a UTF-8
         lead/continuation byte and a continuation byte.
      2. Stray cp1252 dash/quote bytes (0x94/0x96/0x97) used where an em-dash belongs
         (`interpretation.\\x94In`) — mapped to a real em-dash before decoding.

    Pure, deterministic, never raises (final decode uses errors='replace' as a backstop)."""
    # 1) Repair newlines that split a multi-byte UTF-8 char (drop the stray newline).
    out = bytearray()
    i, n = 0, len(raw)
    while i < n:
        b = raw[i]
        if b == 0x0A and out and out[-1] >= 0x80 and i + 1 < n and 0x80 <= raw[i + 1] <= 0xBF:
            i += 1
            continue
        out.append(b)
        i += 1
    data = bytes(out)

    # 2) Decode UTF-8 where valid, falling back to cp1252 for any byte that is NOT part of
    #    a valid UTF-8 sequence. The public text mixes proper UTF-8 (curly quotes, em-dash)
    #    with STRAY single cp1252 bytes (0x93/0x94/0x99/0x9d used for quotes/dashes/™) —
    #    a flat errors='replace' turned ~20 of those into � inside real provisions. This
    #    byte-walk keeps every character: valid UTF-8 decoded as such, lone high bytes via
    #    cp1252 (which has no undefined slots for these), so nothing is lost or mojibake'd.
    chars: List[str] = []
    j, m = 0, len(data)
    while j < m:
        b = data[j]
        if b < 0x80:
            chars.append(chr(b)); j += 1; continue
        # Expected UTF-8 sequence length from the lead byte.
        seq = 4 if b >= 0xF0 else 3 if b >= 0xE0 else 2 if b >= 0xC0 else 1
        decoded = None
        if seq > 1:
            try:
                decoded = data[j:j + seq].decode("utf-8")
            except UnicodeDecodeError:
                decoded = None
        if decoded is not None:
            chars.append(decoded); j += seq
        else:
            # Lone high byte = a stray cp1252 byte. The source uses 0x93/0x94 (smart quotes)
            # AND 0x96/0x97 (dashes) — but it AMBIGUOUSLY uses 0x94 for the title→body
            # EM-DASH too ("Election of President.<0x94>The President..."). Disambiguate:
            # a 0x94 right after a period (the heading terminator) is the em-dash; elsewhere
            # it is a closing quote. 0x96/0x97 are always dashes. Everything else via cp1252.
            prev = chars[-1] if chars else ""
            if b in (0x96, 0x97) or (b == 0x94 and prev.endswith(".")):
                chars.append("—")            # em-dash
            elif b == 0x93:
                chars.append("“")            # left double quote
            elif b == 0x94:
                chars.append("”")            # right double quote
            else:
                chars.append(data[j:j + 1].decode("cp1252", errors="replace"))
            j += 1
    return "".join(chars)


def normalize_constitution_text(text: str) -> str:
    """Reflow the hard-wrapped public text so each Article/Part/Schedule/sub-clause is on
    one logical line — the shape `parse_constitution_text` expects. A line is a CONTINUATION
    of the previous one unless it opens a new structural element (Article number / Part /
    Schedule header) or a sub-clause marker. Collapses the wrap whitespace; keeps blank
    lines as paragraph breaks. Idempotent on already-normalized text."""
    out: List[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            out.append("")
            continue
        starts_structure = bool(
            _ARTICLE_RE.match(line) or _PART_RE.match(line)
            or _SCHEDULE_RE.match(line) or _SUBCLAUSE_RE.match(line)
        )
        if starts_structure or not out or out[-1] == "":
            out.append(line)
        else:
            out[-1] = re.sub(r"\s+", " ", out[-1] + " " + line)
    return "\n".join(out)


@dataclass
class ParseResult:
    """Everything the driver needs to ingest, plus the honesty trail."""
    source: SourceMeta
    provisions: List[Provision]
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        arts = sum(1 for p in self.provisions if p.instrument_type == "article")
        scheds = sum(1 for p in self.provisions if p.instrument_type == "schedule")
        rep = sum(1 for p in self.provisions if p.repealed)
        return (
            f"[parse] {self.source.title}: {arts} articles · {scheds} schedules · "
            f"{rep} repealed · ToC ids {len(self.source.toc_ids)} · "
            f"{len(self.warnings)} warning(s)"
        )


def _split_title_body(rest: str) -> Tuple[str, str]:
    """From the text after 'N. ', split the Article TITLE from its BODY.

    The official text marks the boundary with an em-dash (—) or a period+space after a
    short heading. Prefer the em-dash (most reliable); else take up to the first period
    if the heading is short (< 120 chars); else treat the whole thing as body with a
    truncated title (permissive — a missing title is a warning, not a failure)."""
    rest = rest.strip()
    # Em-dash variants (— – or .—). This is the dominant, reliable boundary.
    for dash in ("—", "–", ".—", ".–"):
        if dash in rest:
            title, body = rest.split(dash, 1)
            return title.strip().rstrip("."), body.strip()
    # Fallback: first period if the lead is heading-length.
    m = re.match(r"^(.{1,120}?)\.\s+(.*)$", rest, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return rest[:120].strip(), rest


def parse_constitution_text(
    text: str,
    *,
    source_key: str = "constitution_of_india",
    title: str = "The Constitution of India",
    citation_prefix: str = "Constitution",
    enacted_date: str = "1950-01-26",
    as_of_date: Optional[str] = None,
    source_url: Optional[str] = "https://www.indiacode.nic.in/handle/123456789/2263",
) -> ParseResult:
    """Parse normalized plain text into a `ParseResult`.

    `as_of_date` is the snapshot horizon you vouch for — REQUIRED for the live ingest
    (every provision inherits it; `Provision.is_valid()` rejects a provision without one).
    The driver passes it explicitly so the operator consciously states "I vouch this text
    is current as of D"; the parser leaves it None if not given (a dry-run can still run,
    but ingest_provisions will reject — surfaced as a receipt problem, never silent)."""
    warnings: List[str] = []
    provisions: List[Provision] = []

    cur_part: Optional[str] = None
    cur_part_title: Optional[str] = None
    cur: Optional[Provision] = None
    repealed_ids: set = set()
    # Once a Schedule starts, its body contains numbered lists ("1. Defence ...") that are
    # NOT Articles — they belong to the Schedule. Track state so we don't mis-open them as
    # Articles (a fake "Art.1" colliding with the real one). Reset by a Part header.
    in_schedule = False

    def _flush(p: Optional[Provision]) -> None:
        if p is not None:
            p.text = (p.text or "").strip()
            provisions.append(p)

    lines = (text or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line:
            # Blank line: paragraph break inside an Article body (keep the break).
            if cur is not None:
                cur.text += "\n"
            continue

        # ── repeal/omission marker ──────────────────────────────────────────────
        mo = _OMITTED_RE.match(line)
        if mo and _ARTICLE_RE.match(line) is None:
            repealed_ids.add(mo.group(1))
            # If the omitted Article was already parsed, flag it; else record for later.
            for p in provisions:
                if p.section_or_article_id == mo.group(1):
                    p.repealed = True
            continue

        # ── Part header ─────────────────────────────────────────────────────────
        mp = _PART_RE.match(line)
        if mp:
            _flush(cur); cur = None
            in_schedule = False
            cur_part = mp.group(1).upper()
            cur_part_title = None
            # The next non-blank line is usually the Part's title (all-caps).
            if i < len(lines) and lines[i].strip() and lines[i].strip().isupper():
                cur_part_title = lines[i].strip().title()
                i += 1
            continue

        # ── Schedule header ───────────────────────────────────────────────────
        ms = _SCHEDULE_RE.match(line)
        if ms:
            _flush(cur)
            in_schedule = True
            ordn = ms.group(1).upper()
            num = _ORDINAL_TO_NUM.get(ordn) or _digits(ordn)
            sid = f"Schedule.{num}" if num else f"Schedule.{ordn}"
            cur = Provision(
                source_key=source_key, instrument_type="schedule",
                title=f"{ordn.title()} Schedule", citation=f"{citation_prefix} {sid}",
                section_or_article_id=sid, text="",
                toc_path=f"Schedules > {sid}",
            )
            continue

        # ── Article opener ──────────────────────────────────────────────────────
        # Inside a Schedule, a numbered line is a Schedule list item (body), not an
        # Article — keep it as continuation text of the current Schedule provision.
        ma = None if in_schedule else _ARTICLE_RE.match(line)
        # Reject footnote/continuation lines masquerading as openers (no em-dash heading,
        # not a repealed stub) — fold them into the current provision body instead.
        if ma and not _is_real_article_opener(ma.group(2)):
            ma = None
        if ma:
            _flush(cur)
            art_id = ma.group(1)
            rest = ma.group(2)
            is_repealed_stub = bool(_REPEALED_STUB.match(rest))
            art_title, body = _split_title_body(rest)
            if is_repealed_stub:
                # "[Compulsory acquisition of property.] Rep. by ..." → title from brackets,
                # the provision is repealed (kept as a citable, withheld stub).
                mb = re.match(r"^\s*\[([^\]]*)\]\s*(.*)$", rest)
                if mb:
                    art_title, body = mb.group(1).strip().rstrip("."), mb.group(2).strip()
            toc_path = "Articles"
            if cur_part:
                toc_path = f"Part {cur_part}" + (f" ({cur_part_title})" if cur_part_title else "")
                toc_path += f" > Art.{art_id}"
            else:
                toc_path = f"Art.{art_id}"
            cur = Provision(
                source_key=source_key, instrument_type="article",
                title=art_title or f"Article {art_id}",
                citation=f"{citation_prefix} Art.{art_id}",
                section_or_article_id=art_id, text=body,
                repealed=(is_repealed_stub or art_id in repealed_ids),
                toc_path=toc_path,
            )
            continue

        # ── continuation line ───────────────────────────────────────────────────
        if cur is not None:
            cur.text += (" " if cur.text and not cur.text.endswith("\n") else "") + line
        else:
            # Text before the first Article (preamble / front matter) — not a provision.
            warnings.append(f"unplaced line before first provision: {line[:60]!r}")

    _flush(cur)

    # Apply any repeal markers that appeared before their Article was seen.
    for p in provisions:
        if p.section_or_article_id in repealed_ids:
            p.repealed = True

    # De-duplicate by id, KEEPING THE FIRST occurrence. Document order puts the real
    # Article before any page-bottom footnote that slipped past the opener discriminator
    # and happened to reuse its number (e.g. a "1. Now 'thirty', vide Act ..." amendment
    # note). Keeping first preserves the genuine provision; the later dup is dropped and
    # reported (never silent — the operator sees what was discarded).
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

    # The authoritative ToC = every provision id we parsed, in order. (For the live
    # source, the operator should cross-check this against the official ToC count —
    # the completeness gate then diffs ingested vs this list.)
    toc_ids = [p.section_or_article_id for p in provisions]

    source = SourceMeta(
        source_key=source_key, title=title, instrument_type="article",
        citation_prefix=citation_prefix, enacted_date=enacted_date,
        as_of_date=as_of_date, toc_ids=toc_ids, source_url=source_url,
    )
    return ParseResult(source=source, provisions=provisions, warnings=warnings)


def _digits(s: str) -> Optional[int]:
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else None


def make_schedule_provision(
    number: int,
    text: str,
    *,
    source_key: str = "constitution_of_india",
    citation_prefix: str = "Constitution",
    title: Optional[str] = None,
) -> Provision:
    """Build ONE provision for an entire Schedule (G8.1a).

    WHY whole-file, not sub-parsed: a Schedule is a sprawling list/table (the First lists
    States, the Seventh the Union/State legislative lists) cited as a unit ("Seventh
    Schedule"), NOT per-entry. Sub-parsing its numbered items as "Articles" was a confident-
    wrong factory (fake Art.1 = a State name) — the dry-run caught 44 fake schedules + dup
    Art.1/2/3 from exactly this. One Schedule = one citation = one provision = one ToC id."""
    sid = f"Schedule.{number}"
    return Provision(
        source_key=source_key, instrument_type="schedule",
        title=title or f"Schedule {number}",
        citation=f"{citation_prefix} {sid}",
        section_or_article_id=sid, text=(text or "").strip(),
        toc_path=f"Schedules > {sid}",
    )
