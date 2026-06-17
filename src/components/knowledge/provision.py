"""The relational source-of-truth schema for the Knowledge Base (G8 §G8.0).

WHY relational, not vectors-only: the completeness gate (§G8.4) and the as-of gate
(§G8.5) must be *computable from a ground truth*, not inferred from what happened to
embed. So every ingested provision is BOTH a vector (for retrieval) AND a row here
(for the gates). The `knowledge_provisions` Supabase table mirrors these fields; the
`knowledge_sources` table mirrors `SourceMeta`.

There is no intelligence here — only the data shapes + a couple of pure derivations
(`is_in_force_on`, the citation/chunk id helpers) the gates and the tool's as-of
filter read. Nothing raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The instrument taxonomy (§G8.0). A provision belongs to exactly one; the chips
# (§G8.7) and the tool's `instrument_type` filter narrow by it.
INSTRUMENT_TYPES = (
    "act",
    "regulation",
    "circular",
    "judgment",
    "article",      # a Constitution Article
    "schedule",     # a Schedule to an Act / the Constitution
    "rule",
)

# A citation id should look like an addressable legal locator, never prose. We don't
# over-validate (sources vary), only assert it is non-empty and not a sentence — the
# real quality check is the §G8.3 citation gate against retrieved passages.
_CITATION_OK = re.compile(r"\S")


def _norm_date(d: Optional[str]) -> Optional[str]:
    """ISO-8601 date string passthrough (YYYY-MM-DD). We store dates as strings so the
    schema needs no datetime import and the gate compares lexically (ISO sorts
    correctly). Returns None for empty; never raises on a malformed value — a bad date
    is caught by the as-of gate, not silently coerced."""
    s = (d or "").strip()
    return s or None


@dataclass
class Provision:
    """One addressable unit of legal authority — an Article, a section, a judgment.

    Mirrors the `knowledge_provisions` table. `chunk_id` is the stable id shared by the
    vector (in the `kb_in` namespace) and this row, so the completeness gate can join
    "ingested vectors" to "expected provisions" without a vector round-trip.
    """

    source_key: str                       # FK → SourceMeta.source_key (which Act/instrument)
    instrument_type: str                  # one of INSTRUMENT_TYPES
    title: str                            # human title ("Right to equality")
    citation: str                         # the addressable locator ("Constitution Art.14")
    section_or_article_id: str            # the bare id within the source ("14", "149", "II")
    text: str                             # the verbatim provision text (what gets quoted)
    jurisdiction: str = "IN"
    enacted_date: Optional[str] = None    # ISO date the provision came into force (if known)
    as_of_date: Optional[str] = None      # date through which this snapshot is known in force
    repealed: bool = False
    superseded_by: Optional[str] = None   # citation of the superseding provision, if any
    toc_path: Optional[str] = None        # the path in the source's table of contents ("Part III > Art.14")
    chunk_id: Optional[str] = None        # stable id shared with the vector
    page: Optional[int] = None            # source page (provenance)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.enacted_date = _norm_date(self.enacted_date)
        self.as_of_date = _norm_date(self.as_of_date)

    # ── validity (the as-of gate's primitive, §G8.5) ────────────────────────────
    def is_current(self) -> bool:
        """Not repealed and not superseded — the snapshot still treats it as live law."""
        return not self.repealed and not self.superseded_by

    def is_in_force_on(self, when: Optional[str]) -> Optional[bool]:
        """Was this provision in force on ISO date `when`?

        Returns True/False when answerable from the snapshot, or **None when we cannot
        vouch** — which the tool/gate treats as ABSTAIN, never a guess (§G8.5: a version
        we can't vouch for is withheld). Specifically None when `when` is after our
        `as_of_date` snapshot horizon (we don't know what changed since) or before
        `enacted_date` is known but the provision is the only snapshot we hold.
        """
        if when is None:
            # No date asked → "current snapshot" semantics: live unless repealed/superseded.
            return self.is_current()
        w = _norm_date(when)
        if w is None:
            return None
        # Before it was enacted → not in force (answerable, if we know the enacted date).
        if self.enacted_date and w < self.enacted_date:
            return False
        # After our snapshot horizon → we cannot vouch (could have been amended/repealed).
        if self.as_of_date and w > self.as_of_date:
            return None
        # Within the vouchable window: in force iff still current.
        return self.is_current()

    def vector_metadata(self) -> Dict[str, Any]:
        """The metadata stamped on the Pinecone vector so `search_knowledge`'s filters
        (`source`, `instrument_type`, `as_of`) and the citation gate can read them off a
        retrieved span without a relational round-trip. Mirrors the row's keys."""
        md = {
            "kind": "knowledge",
            "source_key": self.source_key,
            "instrument_type": self.instrument_type,
            "citation": self.citation,
            "section_or_article_id": self.section_or_article_id,
            "jurisdiction": self.jurisdiction,
            "enacted_date": self.enacted_date,
            "as_of_date": self.as_of_date,
            "repealed": self.repealed,
            "superseded_by": self.superseded_by,
            "toc_path": self.toc_path,
            "chunk_id": self.chunk_id,
            "page_number": self.page,
        }
        return {k: v for k, v in md.items() if v is not None}

    def is_valid(self) -> List[str]:
        """Structural problems that should block ingest (the infra gate asserts none).
        Returns a list of human-readable problems; empty = ok. Never raises."""
        problems: List[str] = []
        if not (self.source_key or "").strip():
            problems.append("missing source_key")
        if self.instrument_type not in INSTRUMENT_TYPES:
            problems.append(f"instrument_type {self.instrument_type!r} not in {INSTRUMENT_TYPES}")
        if not _CITATION_OK.search(self.citation or ""):
            problems.append("missing/blank citation id")
        if not (self.section_or_article_id or "").strip():
            problems.append("missing section_or_article_id")
        if not (self.text or "").strip():
            problems.append("empty provision text")
        if self.as_of_date is None:
            # The whole version-in-force guarantee rests on as_of — a provision without
            # one cannot be vouched for any date, so we require it at ingest.
            problems.append("missing as_of_date (version-in-force guarantee needs it)")
        return problems


@dataclass
class SourceMeta:
    """Per-instrument metadata + the AUTHORITATIVE table of contents captured at ingest.

    Mirrors `knowledge_sources`. `toc_ids` is the ground truth the completeness gate
    (§G8.4) diffs the ingested provisions against — "every section/Article present, or
    the gap is listed." Without this captured at ingest, completeness is a hope, not a
    measurement.
    """

    source_key: str                       # stable key ("constitution_of_india")
    title: str                            # "The Constitution of India"
    instrument_type: str
    jurisdiction: str = "IN"
    citation_prefix: str = ""             # "Constitution" → provisions cite "Constitution Art.N"
    enacted_date: Optional[str] = None
    as_of_date: Optional[str] = None      # the snapshot horizon for the whole source
    toc_ids: List[str] = field(default_factory=list)   # every expected section/article id
    source_url: Optional[str] = None      # provenance (legislative.gov.in / India Code)
    partial: bool = False                 # set by the completeness gate when the ToC isn't fully covered
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.enacted_date = _norm_date(self.enacted_date)
        self.as_of_date = _norm_date(self.as_of_date)


@dataclass
class IngestReceipt:
    """The KB ingest receipt — the legal twin of `extraction_fidelity`'s per-doc receipt.

    Emitted after ingesting one source so the operator SEES, as a number, what the
    machine actually captured: provisions parsed, vectors embedded, ToC coverage %, the
    source key + snapshot horizon. A missing-from-ToC list makes a silent gap impossible
    (§G8.4's discipline applied at ingest, log-only — the committed gate is the hard
    line).
    """

    source_key: str
    title: str
    provisions_parsed: int
    provisions_embedded: int
    toc_total: int
    toc_covered: int
    as_of_date: Optional[str]
    missing_ids: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def toc_coverage(self) -> float:
        """Fraction of the source's own table of contents present in the ingest [0,1]."""
        if self.toc_total <= 0:
            return 1.0 if self.provisions_parsed else 0.0
        return self.toc_covered / self.toc_total

    @property
    def complete(self) -> bool:
        return self.toc_total > 0 and self.toc_covered >= self.toc_total

    def render(self) -> str:
        """One-line operator-facing summary (the receipt, like the fidelity log line)."""
        pct = round(self.toc_coverage * 100, 1)
        flag = "" if self.complete else f" ⚠ partial (missing {len(self.missing_ids)})"
        return (
            f"[KB ingest] {self.source_key}: {self.provisions_parsed} provisions · "
            f"{self.provisions_embedded} embedded · ToC {self.toc_covered}/{self.toc_total} "
            f"({pct}%) · as_of {self.as_of_date or '—'}{flag}"
        )
