import os
import io
import re
import json
import sys
import math
import time
import atexit
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.text import partition_text

from langchain_core.documents import Document

from src.utils import _log_elements_analysis, _get_element_type, _get_page_number, _element_has_image_payload, _table_html, _stable_id,_create_image_description,_create_table_description

try:
    from .config import Config
except ImportError:
    from src.components.config import Config

try:
    from src.logger import get_logger
except ImportError:
    import logging
    get_logger = logging.getLogger

_logger = get_logger(__name__)


# ── Persistent ProcessPoolExecutor (Layer 2) ─────────────────────────────────
# Created once per Celery worker process. Workers stay alive between PDFs,
# so unstructured models (YOLOX, table model) are only loaded once per worker
# lifetime rather than once per PDF.
#
# NOTE: Models are warmed INSIDE pool workers (via warm_pdf_pool), NOT in the
# parent process before fork. PyTorch models should not cross a fork boundary
# — doing so causes file descriptor leaks and potential deadlocks.
_PDF_POOL: Optional[ProcessPoolExecutor] = None
_PDF_POOL_SIZE: int = 0
_PDF_POOL_LOCK = None  # Lazy import threading.Lock to avoid import cost


def _get_pdf_pool(n_workers: int) -> ProcessPoolExecutor:
    """Return the module-level persistent pool, creating or resizing if needed."""
    global _PDF_POOL, _PDF_POOL_SIZE, _PDF_POOL_LOCK
    import threading
    if _PDF_POOL_LOCK is None:
        _PDF_POOL_LOCK = threading.Lock()
    with _PDF_POOL_LOCK:
        if _PDF_POOL is None or _PDF_POOL_SIZE != n_workers:
            if _PDF_POOL is not None:
                _PDF_POOL.shutdown(wait=False)
            _PDF_POOL = ProcessPoolExecutor(max_workers=n_workers)
            _PDF_POOL_SIZE = n_workers
            _logger.info("PDF pool (re)created: %d workers", n_workers)
        return _PDF_POOL


@atexit.register
def _shutdown_pdf_pool():
    """Gracefully shut down the pool on Celery worker exit."""
    global _PDF_POOL
    if _PDF_POOL is not None:
        _PDF_POOL.shutdown(wait=False)
        _PDF_POOL = None


def _warm_worker():
    """Run inside a pool worker to trigger lazy unstructured model loading.

    Submitting this to each pool worker at startup means YOLOX + table models
    are loaded once per worker process, amortized over all PDFs processed by
    that worker rather than paid fresh for each PDF.

    Deliberately runs inside the child process (via pool.submit) so PyTorch
    models are never initialized in the parent before fork.
    """
    import io as _io
    import tempfile as _tempfile
    import os as _os
    try:
        from pypdf import PdfWriter
        from unstructured.partition.pdf import partition_pdf

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = _io.BytesIO()
        writer.write(buf)
        buf.seek(0)

        with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            partition_pdf(filename=tmp_path, strategy="auto", infer_table_structure=True)
        finally:
            _os.unlink(tmp_path)
    except Exception as exc:
        # Non-fatal — model will still load lazily on first real PDF
        import logging
        logging.getLogger(__name__).warning("Worker warm-up failed (non-fatal): %s", exc)


def warm_pdf_pool(n_workers: int = 4, timeout: int = 90):
    """Pre-warm unstructured models inside each pool worker.

    Called from the Celery worker_init signal so models are loaded once at
    container startup, not on the first PDF upload.
    Each warm-up task runs inside a child process (safe for PyTorch).

    Args:
        n_workers: Number of workers to warm (should match PDF_PARALLEL_WORKERS).
        timeout: Max seconds to wait per worker warm-up task.
    """
    pool = _get_pdf_pool(n_workers)
    futures = [pool.submit(_warm_worker) for _ in range(n_workers)]
    warmed = 0
    for f in futures:
        try:
            f.result(timeout=timeout)
            warmed += 1
        except Exception as exc:
            _logger.warning("Pool warm-up task failed (non-fatal): %s", exc)
    _logger.info("PDF pool pre-warmed: %d/%d workers ready", warmed, n_workers)



def _get_pdf_page_count(file_path: str) -> int:
    """Get the number of pages in a PDF without fully parsing it."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception:
        return 0


def _pdf_text_density(file_path: str, sample_pages: int = 5) -> float:
    """A5: average extractable characters per page over an evenly-spaced sample.

    A born-digital PDF has a real text layer (hundreds–thousands of chars/page);
    a scanned PDF returns ~0 and genuinely needs OCR. Returns -1.0 on error so
    callers can keep their default (OCR) behaviour rather than guess wrong.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        n = len(reader.pages)
        if n == 0:
            return -1.0
        if n == 1 or sample_pages <= 1:
            idxs = [0]
        else:
            idxs = sorted({int(i * (n - 1) / (sample_pages - 1)) for i in range(sample_pages)})
        total = 0
        for i in idxs:
            try:
                total += len((reader.pages[i].extract_text() or "").strip())
            except Exception:
                pass
        return total / len(idxs)
    except Exception:
        return -1.0


def _process_pdf_page_range_from_bytes(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    strategy: str,
    extract_images: bool,
) -> list:
    """Process a page range from in-memory PDF bytes (Layer 5A).

    Receives the full PDF as bytes (read once in the parent, passed via pickle
    to each pool worker) instead of each worker independently opening the file
    from disk. This eliminates N concurrent disk reads of the same large file.

    The page slice is written to a temp file before calling partition_pdf because
    unstructured's hi_res strategy shells out to poppler/tesseract which require
    a real file path, not a file-like object.
    """
    try:
        import io as _io
        from pypdf import PdfReader, PdfWriter

        # Slice pages from the in-memory bytes
        reader = PdfReader(_io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page_num in range(start_page, min(end_page, len(reader.pages))):
            writer.add_page(reader.pages[page_num])

        # Write slice to a temp file — required for hi_res strategy (poppler/tesseract)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            writer.write(tmp)
            tmp_path = tmp.name

        try:
            pdf_kwargs = {
                "filename": tmp_path,
                "strategy": strategy,
                "infer_table_structure": True,
            }
            if extract_images:
                pdf_kwargs["extract_image_block_type"] = ["Image"]
                pdf_kwargs["extract_image_block_to_payload"] = True

            elements = partition_pdf(**pdf_kwargs)

            # Fix page numbers: partition_pdf numbers from 1 within the slice
            for el in elements:
                if hasattr(el, "metadata") and el.metadata:
                    local_page = getattr(el.metadata, "page_number", None)
                    if local_page is not None:
                        el.metadata.page_number = local_page + start_page

            return elements
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except Exception as e:
        print(f"  Error processing pages {start_page}-{end_page}: {e}")
        return []


def _process_pdf_page_range(
    file_path: str,
    start_page: int,
    end_page: int,
    strategy: str,
    extract_images: bool,
) -> list:
    """Legacy worker — kept for single-PDF fallback path.

    Prefer _process_pdf_page_range_from_bytes when calling from the persistent
    pool, which passes pre-read bytes to avoid N concurrent disk opens.
    """
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(file_path)
        writer = PdfWriter()
        for page_num in range(start_page, min(end_page, len(reader.pages))):
            writer.add_page(reader.pages[page_num])

        # Write slice to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            writer.write(tmp)
            tmp_path = tmp.name

        try:
            pdf_kwargs = {
                "filename": tmp_path,
                "strategy": strategy,
                "infer_table_structure": True,
            }
            if extract_images:
                pdf_kwargs["extract_image_block_type"] = ["Image"]
                pdf_kwargs["extract_image_block_to_payload"] = True

            elements = partition_pdf(**pdf_kwargs)

            for el in elements:
                if hasattr(el, "metadata") and el.metadata:
                    local_page = getattr(el.metadata, "page_number", None)
                    if local_page is not None:
                        el.metadata.page_number = local_page + start_page

            return elements
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except Exception as e:
        print(f"  Error processing pages {start_page}-{end_page}: {e}")
        return []



# ── GRAND_PLAN G1a — boilerplate strip ───────────────────────────────────────
# Repeated page chrome (running headers/footers, "Page 3 of 7", the source URL,
# a print timestamp) recurs on every page of an EDGAR .htm→PDF and DOMINATES the
# embedding of whatever clause shares its chunk — so a search for "governing law"
# ranks the header/footer chunks above the one that actually holds the clause
# (measured: 4/4 contract chunks polluted, 7× each of page-no/URL/timestamp).
# Dropping it before chunking unburies the clause signal. Helps financial filings
# too (cleaner prose chunks); the finance TABLE pass is geometry-based and never
# sees these elements, so the moat is untouched. Structural, conservative,
# never-raises — a parse error degrades to "no strip", identical to pre-G1a.

# Standalone-chrome line patterns (matched on SHORT elements only, so a real
# clause that merely mentions a URL is never dropped).
_BP_PAGE_N_OF_M = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
_BP_BARE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BP_PRINT_TS = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(:\d{2})?\s*[ap]\.?m\.?\b",
    re.IGNORECASE,
)
# unstructured element categories that ARE page chrome — drop outright.
_BP_CHROME_CATEGORIES = {"header", "footer", "pagenumber", "page_number"}
# A short element whose text, after removing the chrome patterns, is (near-)empty
# is itself chrome. Length ceiling guards real content from regex over-reach.
_BP_SHORT_CEILING = 140
_BP_RESIDUAL_FLOOR = 8  # chars of real text that must remain to KEEP a short element


def _is_boilerplate_text(text: str) -> bool:
    """True if a SHORT standalone element is page chrome (page-no / URL / timestamp).

    Removes the three chrome signatures; if what remains is shorter than
    ``_BP_RESIDUAL_FLOOR`` (i.e. the element was basically just chrome), it's
    boilerplate. Long elements are never judged here — they're real content even
    if they contain a URL.
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) > _BP_SHORT_CEILING:
        return False
    residual = _BP_PRINT_TS.sub(" ", stripped)
    residual = _BP_BARE_URL.sub(" ", residual)
    residual = _BP_PAGE_N_OF_M.sub(" ", residual)
    # collapse leftover separators/whitespace that framed the chrome
    residual = re.sub(r"[\s|·•\-–—,:;]+", "", residual)
    return len(residual) < _BP_RESIDUAL_FLOOR


def _strip_boilerplate(text_elements: List) -> tuple:
    """Drop page-chrome elements before chunking. Returns (kept, stats).

    Two tiers: (1) unstructured-classified Header/Footer/PageNumber categories;
    (2) short standalone elements that are nothing but page-no / URL / timestamp
    chrome. Conservative by construction — only removes elements that are wholly
    chrome, never trims inside a real element. Never raises.
    """
    stats = {"by_category": 0, "by_pattern": 0, "kept": 0, "total": len(text_elements)}
    if not text_elements:
        return text_elements, stats
    kept: List = []
    for el in text_elements:
        try:
            cat = _get_element_type(el).lower()
            if cat in _BP_CHROME_CATEGORIES:
                stats["by_category"] += 1
                continue
            text = getattr(el, "text", None) or ""
            if _is_boilerplate_text(text):
                stats["by_pattern"] += 1
                continue
        except Exception:  # noqa: BLE001 — never let a stray element break ingestion
            pass
        kept.append(el)
    stats["kept"] = len(kept)
    return kept, stats


# ── GRAND_PLAN G1d — document classification at ingest ───────────────────────
# Decide a coarse doc class so the right extraction path runs: a prose CONTRACT
# wants clause-aware chunking (G1b) and NO financial-table pass (G1c), while a
# FILING wants the geometry table moat untouched. Structural/deterministic v1
# (no LLM — $0, reproducible): the signals separate cleanly in measurement
# (contract: numbered-clause-heading ratio ~0.7, ~0 '$'; 10-K: heading ratio
# <0.01, dense '$'). A small LLM tiebreak is a documented future hook (the
# design note's 2d), NOT needed for the clear cases. Stored on the doc + each
# chunk's metadata as `doc_type` so the grid/agent can use it too.

# A numbered clause/section heading: "1.", "2.1", "9.1 Governing Law", "8. Indemnification".
_CLAUSE_HEADING = re.compile(r"^\s*\d+(\.\d+){0,3}\.?\s+[A-Z(\"']")
# A bare numbered heading line (the heading on its own short element).
_CLAUSE_HEADING_BARE = re.compile(r"^\s*\d+(\.\d+){0,3}\.?\s+\S")


def _leading_heading(chunk_text: str) -> Optional[str]:
    """Phase 1.5: the section heading of a chunk = its first detected-heading line.

    Reuses the harness's canonical heading detector (`_outline._line_heading`) so a
    PERSISTED `section_path` is byte-identical to what the runtime heuristic would
    detect — no drift between the two read_section code paths. Returns None when no
    line in the chunk looks like a heading (then the metadata key is omitted). Lazy
    import + never-raise: a detector hiccup must never break ingestion.
    """
    if not chunk_text:
        return None
    try:
        from src.components.agent_core.tools._outline import _line_heading
        for line in chunk_text.split("\n"):
            h = _line_heading(line)
            if h:
                return h
    except Exception:  # noqa: BLE001 — heading detection is best-effort
        return None
    return None

DOC_TYPE_LEGAL = "legal_contract"
DOC_TYPE_FINANCIAL = "financial_filing"
DOC_TYPE_MIXED = "mixed"
DOC_TYPE_GENERIC = "generic"

# Filename hints (cheap, high-precision signal). EDGAR contract exhibits are EX-10.x.
_LEGAL_NAME_HINTS = ("ex-10", "ex10", "agreement", "contract", "nda", "lease",
                     "deed", "amendment", "addendum", "msa", "sow")
_FINANCIAL_NAME_HINTS = ("10-k", "10k", "10-q", "10q", "20-f", "8-k", "annual",
                         "financial", "balance", "income")


def _heading_count(text_elements: List) -> int:
    """Count elements that BEGIN with a numbered clause heading."""
    n = 0
    for el in text_elements:
        try:
            t = (getattr(el, "text", None) or "").strip()
            if t and _CLAUSE_HEADING.match(t):
                n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


def classify_document(text_elements: List, filename: Optional[str] = None) -> str:
    """Return one of DOC_TYPE_* from cheap structural signals. Never raises.

    Decision (measured-threshold, structural):
      - numbered-clause-heading ratio (headings / text-elements) — high for contracts
      - '$' density (dollar signs / text-element) — high for filings
      - filename hints as a precise tiebreak / booster
    A document with a strong clause-heading ratio AND low '$' density is a legal
    contract; strong '$' density with negligible headings is a financial filing;
    a doc showing both signals is 'mixed' (treated as financial — keep the table
    moat, don't risk a 10-K's tables); neither → 'generic' (default path).
    """
    try:
        name = (filename or "").lower()
        name_legal = any(h in name for h in _LEGAL_NAME_HINTS)
        name_fin = any(h in name for h in _FINANCIAL_NAME_HINTS)

        n_text = max(1, len(text_elements))
        n_head = _heading_count(text_elements)
        dollars = 0
        for el in text_elements:
            try:
                dollars += (getattr(el, "text", None) or "").count("$")
            except Exception:  # noqa: BLE001
                pass

        head_ratio = n_head / n_text
        dollar_density = dollars / n_text

        # Strong structural signals (thresholds from the measured gap: contract
        # head_ratio ~0.72 / $density ~0.03 ; 10-K head_ratio ~0.007 / $density ~0.19).
        legal_struct = head_ratio >= 0.15 and dollar_density < 0.08
        fin_struct = dollar_density >= 0.10 and head_ratio < 0.05

        # Combine structure + name. Name hints break near-threshold ties.
        legal = legal_struct or (name_legal and head_ratio >= 0.05 and dollar_density < 0.10)
        fin = fin_struct or (name_fin and dollar_density >= 0.05)

        if legal and not fin:
            result = DOC_TYPE_LEGAL
        elif fin and not legal:
            result = DOC_TYPE_FINANCIAL
        elif legal and fin:
            result = DOC_TYPE_MIXED
        else:
            result = DOC_TYPE_GENERIC

        _logger.info(
            "[ingest] doc class: %s (file=%s, head_ratio=%.3f, $density=%.3f, "
            "headings=%d/%d, name_legal=%s, name_fin=%s)",
            result, filename, head_ratio, dollar_density, n_head, len(text_elements),
            name_legal, name_fin)
        return result
    except Exception as exc:  # noqa: BLE001 — classification never breaks ingestion
        _logger.warning("[ingest] doc classification failed (%s) — generic", exc)
        return DOC_TYPE_GENERIC


# ── G3 Step C — structural fiscal_year derivation ($0, NULL when unsure) ─────────
# A doc's fiscal year drives the FY filter chip. We derive it STRUCTURALLY only — no
# LLM, no guess — from two signals, in precedence order:
#   1. FILENAME (highest precision): EDGAR/issuer names carry the period-end date,
#      e.g. "amzn-20221231" → 2022, "msft-10k_20220630" → 2022 (FY end June 30),
#      "goog-20231231" → 2023; also "FY2023" / "fy23".
#   2. PERIOD HEADERS the extractor already read off the statements (grid.periods like
#      ["2021","2022","2023"]) → the filing's primary FY is its LATEST reported year.
# If neither yields a year in a sane window, return None — the filter treats NULL as
# "unknown, don't exclude" (a mis-derived FY would silently hide the right doc; §5 risk #3).

_FY_MIN, _FY_MAX = 1990, 2099  # sanity window — reject stray 4-digit tokens outside it

# YYYYMMDD anywhere in the filename (the EDGAR period-end), e.g. amzn-20221231.
_RE_YYYYMMDD = re.compile(r"(?<!\d)(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
# Explicit fiscal-year tag: FY2023 / FY 2023 / fy23.
_RE_FY_TAG = re.compile(r"\bfy[\s_-]?((?:19|20)\d{2}|\d{2})\b", re.IGNORECASE)
# Bare 4-digit year token (lowest-precision filename signal).
_RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def _fy_from_filename(filename: Optional[str]) -> Optional[int]:
    name = (filename or "").lower()
    if not name:
        return None
    # 1. period-end date YYYYMMDD → the YEAR is the fiscal year.
    m = _RE_YYYYMMDD.search(name)
    if m:
        yr = int(m.group(0)[:4])
        if _FY_MIN <= yr <= _FY_MAX:
            return yr
    # 2. explicit FY tag (handles 2-digit fy23 → 2023).
    m = _RE_FY_TAG.search(name)
    if m:
        raw = m.group(1)
        yr = int(raw) if len(raw) == 4 else 2000 + int(raw)
        if _FY_MIN <= yr <= _FY_MAX:
            return yr
    # 3. bare year token (e.g. "annual-report-2022").
    m = _RE_YEAR.search(name)
    if m:
        yr = int(m.group(0))
        if _FY_MIN <= yr <= _FY_MAX:
            return yr
    return None


def derive_fiscal_year(filename: Optional[str], periods: Optional[List[str]] = None) -> Optional[int]:
    """Return the doc's fiscal year as an int, or None when not structurally certain.

    Filename wins (it carries the precise period-end); period headers are the fallback
    (the latest reported year on the statements). Never raises, never guesses.
    """
    try:
        fy = _fy_from_filename(filename)
        if fy is not None:
            return fy
        # Period headers: take the MAX year (the filing's primary/most-recent FY).
        years = []
        for p in (periods or []):
            mm = _RE_YEAR.search(str(p))
            if mm:
                yr = int(mm.group(0))
                if _FY_MIN <= yr <= _FY_MAX:
                    years.append(yr)
        return max(years) if years else None
    except Exception as exc:  # noqa: BLE001 — derivation never breaks ingestion
        _logger.warning("[ingest] fiscal_year derivation failed (%s) — None", exc)
        return None


# ── GRAND_PLAN G1b — clause-aware chunking for prose/legal docs ──────────────
# Contracts are numbered ("1. Engagement", "9.1 Governing Law"). chunk_by_title
# splits on SIZE, cutting a clause mid-sentence so neither half retrieves well
# (the live failure). chunk_legal_prose starts a new chunk at each numbered
# heading and keeps the heading WITH its body (so "Governing Law" embeds with the
# clause it heads), only splitting a single clause if it exceeds a hard ceiling.
# Returns a list of (text, page_number, heading) tuples; the caller builds Documents.

def chunk_legal_prose(text_elements: List, max_chars: int) -> List:
    """Chunk prose at numbered clause/section boundaries. Returns list of
    {"text", "page", "heading"} dicts. Never raises (caller falls back to
    chunk_by_title on empty)."""
    chunks: List[Dict[str, Any]] = []
    if not text_elements:
        return chunks
    try:
        cur_parts: List[str] = []
        cur_page = None
        cur_heading = None

        def _flush():
            nonlocal cur_parts, cur_heading, cur_page
            if cur_parts:
                text = "\n".join(p for p in cur_parts if p).strip()
                if len(text) >= 10:
                    chunks.append({"text": text, "page": cur_page, "heading": cur_heading})
            cur_parts = []
            cur_heading = None

        for el in text_elements:
            t = (getattr(el, "text", None) or "").strip()
            if not t:
                continue
            page = _get_page_number(el)
            is_heading = bool(_CLAUSE_HEADING_BARE.match(t))

            # A new top-or-sub clause heading starts a new chunk (unless the current
            # chunk is still tiny — keep a heading-only line attached to what follows).
            if is_heading and cur_parts:
                cur_len = sum(len(p) for p in cur_parts)
                if cur_len >= 40:  # don't split off a lone heading with no body yet
                    _flush()

            if cur_page is None:
                cur_page = page
            if is_heading and cur_heading is None:
                cur_heading = t[:120]
            cur_parts.append(t)

            # Size ceiling: a single over-long clause is split (with the heading kept
            # as the first chunk's lead; continuation chunks carry the same heading).
            if sum(len(p) for p in cur_parts) >= max_chars:
                _flush()

        _flush()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[ingest] clause chunking failed (%s) — caller falls back", exc)
        return []
    return chunks


class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        # G1d / G2 Step F: the last-processed doc's structural class and coarse
        # extraction-fidelity grade, exposed for the worker to persist on the doc row.
        # doc_type is set in build_langchain_documents; fidelity in _build_table_chunks
        # (legal docs skip the table pass, so they stay None — honest "unknown").
        self._last_doc_type = None
        self._last_fidelity = None
        self._last_fiscal_year = None  # G3 Step C — structural FY (None = unknown)

    def _detect_strategy(self, file_path: str, file_ext: str, page_count: Optional[int] = None) -> str:
        """
        Phase 3: Choose PDF processing strategy based on file type and page count.

        Strategy selection:
          .txt / .md           → always "fast" (no OCR needed)
          .docx / .pptx / .xlsx → "auto" (structure-aware, no OCR)
          .pdf ≤ FAST pages    → "fast"   (no layout model, ~0.5s/doc)
          .pdf ≤ MEDIUM pages  → "auto"   (unstructured decides, ~2-5s/doc)
          .pdf > MEDIUM pages  → "hi_res" (full OCR, ~15-30s/doc)
          .pdf unknown pages   → config fallback

        Args:
            page_count: Pre-computed page count to avoid a second PdfReader read.
                        If None, reads the PDF here. Pass this from process_documents
                        where the page count is already known.
        """
        if file_ext in [".txt", ".md"]:
            return "fast"

        if file_ext in [".docx", ".pptx", ".xlsx"]:
            return "auto"

        if file_ext == ".pdf":
            # Layer 4: use pre-computed count if available — avoids a second PdfReader open
            if page_count is None:
                page_count = _get_pdf_page_count(file_path)
            if page_count == 0:
                return self.config.PDF_STRATEGY
            if page_count <= self.config.PDF_FAST_THRESHOLD_PAGES:
                return "fast"
            # NOTE on strategy choice (2026-06-02):
            # "auto"/"hi_res" run unstructured's native layout+table model to get
            # STRUCTURED tables (HTML) — the §4b ideal. But that pipeline is currently
            # BROKEN in this environment: it raises UnicodeDecodeError (0x89 = PNG
            # byte) decoding rendered page images — a pdfminer.six(20251230)/
            # unstructured(0.18.27) version conflict — yielding 0 elements after ~200s.
            # "fast" (pdfminer text) is reliable and extracts the full text INCLUDING
            # every table cell value (verified: 211,915 / 198,270 / … from the 10-K
            # tables), which is exactly what the Brain reads today. Structured HTML
            # tables for §4b's deterministic compute require fixing the hi_res
            # dependency conflict first (tracked). Until then, text-layer PDFs use
            # "fast"; only genuinely scanned PDFs attempt OCR ("hi_res").
            density = _pdf_text_density(file_path)
            if density >= self.config.PDF_TEXT_LAYER_MIN_CHARS_PER_PAGE:
                _logger.info(
                    "%d-page PDF has text layer (%.0f chars/page) — using 'fast' "
                    "(hi_res table model broken in this env; numbers still extracted)",
                    page_count, density,
                )
                return "fast"
            return "hi_res"

        return "auto"

    def _process_pdf_parallel(
        self, file_path: str, strategy: str = None, page_count: Optional[int] = None,
        progress_cb=None,
    ) -> List:
        """Split a PDF into page-range chunks and process in parallel.

        Uses the module-level persistent ProcessPoolExecutor (Layer 2) so workers
        stay alive between PDFs and keep unstructured models in memory.

        Reads the full PDF bytes ONCE in the parent (Layer 5A) then passes bytes
        to each worker, eliminating N concurrent disk reads of the same file.

        Args:
            page_count: Pre-computed page count. If None, reads the PDF header.
        """
        strategy = strategy or self.config.PDF_STRATEGY
        # Layer 4: use pre-computed page count if provided
        total_pages = page_count if page_count is not None else _get_pdf_page_count(file_path)
        workers = self.config.PDF_PARALLEL_WORKERS

        # Not worth parallelizing small PDFs (overhead > benefit)
        if total_pages < 6 or workers <= 1:
            return self._process_pdf_single(file_path, strategy=strategy)

        # Layer 5A: read the full PDF bytes ONCE in the parent process.
        # Each worker receives this via pickle instead of independently opening
        # the file from disk, which would cause N concurrent reads of a large file.
        try:
            with open(file_path, "rb") as fh:
                pdf_bytes = fh.read()
        except OSError as exc:
            _logger.error("Failed to read PDF for parallel processing: %s", exc)
            return self._process_pdf_single(file_path, strategy=strategy)

        # Split into roughly equal page ranges
        pages_per_worker = math.ceil(total_pages / workers)
        ranges = [
            (start, min(start + pages_per_worker, total_pages))
            for start in range(0, total_pages, pages_per_worker)
        ]

        _logger.info(
            "Parallel PDF processing: %d pages → %d workers (%d pages/worker, strategy=%s)",
            total_pages, len(ranges), pages_per_worker, strategy,
        )
        print(f"  Parallel PDF: {total_pages} pages → {len(ranges)} workers ({strategy})")

        t_start = time.perf_counter()
        results_by_start: dict = {}

        # Layer 2: use persistent pool — workers survive between PDFs
        pool = _get_pdf_pool(workers)
        futures = {
            pool.submit(
                _process_pdf_page_range_from_bytes,  # Layer 5A: bytes, not file path
                pdf_bytes,
                start,
                end,
                strategy,
                self.config.EXTRACT_IMAGES,
            ): (start, end)
            for start, end in ranges
        }

        pages_done = 0
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                elements = future.result()
                results_by_start[start] = elements
                print(f"    Pages {start+1}-{end}: {len(elements)} elements")
            except Exception as exc:
                _logger.error("Pages %d-%d failed: %s", start + 1, end, exc)
                print(f"    Pages {start+1}-{end}: FAILED ({exc})")
                results_by_start[start] = []
            # C6: report page-level parse progress as each range finishes.
            pages_done += (end - start)
            if progress_cb:
                try:
                    progress_cb(pages_done, total_pages)
                except Exception:
                    pass  # progress reporting must never break ingestion

        all_elements: list = []
        for start in sorted(results_by_start):
            all_elements.extend(results_by_start[start])

        elapsed = time.perf_counter() - t_start
        print(f"  Parallel complete: {len(all_elements)} elements in {elapsed:.1f}s")
        return all_elements

    def _process_pdf_single(self, file_path: str, strategy: str = None) -> List:
        """Standard single-pass PDF processing."""
        strategy = strategy or self.config.PDF_STRATEGY
        pdf_kwargs = {
            "filename": file_path,
            "strategy": strategy,
            "infer_table_structure": True,
        }
        if self.config.EXTRACT_IMAGES:
            pdf_kwargs["extract_image_block_type"] = ["Image"]
            pdf_kwargs["extract_image_block_to_payload"] = True
        return partition_pdf(**pdf_kwargs)

    def process_documents(self, file_paths: str, force_strategy: str = None, progress_cb=None) -> List:
        """Process documents and return a list of processed elements.

        Args:
            file_paths:      Path to the file to process.
            force_strategy:  Optional override for PDF strategy (e.g. 'hi_res' for
                             user-requested high-quality scan). If None, auto-detects
                             via _detect_strategy() based on file type + page count.
            progress_cb:     Optional callable(pages_done, total_pages) invoked as
                             PDF page ranges finish parsing (C6: page-level progress).
        """
        file_extension = Path(file_paths).suffix.lower()
        file_name = Path(file_paths).name

        # Layer 4: compute page count ONCE and pass to both _detect_strategy
        # and _process_pdf_parallel, eliminating the double PdfReader open.
        page_count: Optional[int] = None
        if file_extension == ".pdf":
            page_count = _get_pdf_page_count(file_paths)

        # Phase 3: auto-detect strategy unless caller forces one
        strategy = force_strategy or self._detect_strategy(file_paths, file_extension, page_count=page_count)
        _logger.info(
            "Processing %s with strategy=%s (page_count=%s)",
            file_name, strategy, page_count,
        )

        try:
            if file_extension == ".pdf":
                if self.config.PARALLEL_PDF_PAGES:
                    elements = self._process_pdf_parallel(file_paths, strategy=strategy, page_count=page_count, progress_cb=progress_cb)
                else:
                    elements = self._process_pdf_single(file_paths, strategy=strategy)

            elif file_extension == ".docx":
                elements = partition_docx(
                    filename = file_paths,
                    infer_table_structure=True,
                )   

            elif file_extension == ".pptx":
                elements = partition_pptx(
                    filename=file_paths
                )     
            elif file_extension == ".xlsx":
                elements = partition_xlsx(
                    filename= file_paths
                )
            elif file_extension in [".txt",".md"]:
                elements = partition_text(
                    filename= file_paths
                )
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            print(f"Processed {file_paths} successfully. Extracted {len(elements)} elements. \n")

            
            # Analyzed elements 
            _log_elements_analysis(elements)

            # source metadata
            for element in elements:
                element.metadata.filename = file_name
                element.metadata.filetype = file_extension.strip(".")
                element.metadata.filepath = file_paths
                
            return elements
        
        except Exception as e:
            print(f"Error processing {file_paths}: {e}")
            return []


    def build_langchain_documents(self, elements: List, pdf_path: Optional[str] = None) -> List[Document]:
        """Build LangChain Document chunks from parsed elements.

        Phase 4.3: when ``pdf_path`` points to a PDF, a structured table pass
        (pdfplumber, confidence-gated — see table_extraction.py) runs IN ADDITION
        to the normal text/table/image element chunking and appends
        ``chunk_type="table"`` chunks carrying a normalized grid. This is purely
        additive — the existing text (prose) path is untouched, so the Brain's
        proven fast path cannot regress. If ``pdf_path`` is None or extraction
        finds nothing, behavior is identical to before.
        """
        if not elements:
            return []
        
        text_elements = []
        table_elements = []
        image_elements = []

        for el in elements:
            el_type = _get_element_type(el).lower()

            if "table" in el_type:
                table_elements.append(el)
                continue

            if "image" in el_type or _element_has_image_payload(el):
                image_elements.append(el)
                continue

            if getattr(el, "text", None):
                text_elements.append(el)

        docs: List[Document] = []

        # ── G1a: strip page-chrome before chunking (config-gated; pure-text path) ──
        if getattr(self.config, "STRIP_BOILERPLATE", False) and text_elements:
            text_elements, bp_stats = _strip_boilerplate(text_elements)
            dropped = bp_stats["by_category"] + bp_stats["by_pattern"]
            if dropped:
                _logger.info(
                    "[ingest] boilerplate strip: dropped %d/%d text elements "
                    "(%d chrome-category, %d pattern); %d kept",
                    dropped, bp_stats["total"], bp_stats["by_category"],
                    bp_stats["by_pattern"], bp_stats["kept"])

        # ── G1d: classify the document so the right extraction path runs ──
        # Resolve the user-facing filename from the elements (process_documents
        # stamps it). Used by the classifier (name hints) and by G1c (skip-tables).
        _fname = None
        _ftype = None
        for el in text_elements + table_elements:
            md = getattr(el, "metadata", None)
            if md is not None:
                _fname = _fname or getattr(md, "filename", None)
                _ftype = _ftype or getattr(md, "filetype", None)
        doc_type = DOC_TYPE_GENERIC
        if getattr(self.config, "CLASSIFY_DOCS", False) and text_elements:
            doc_type = classify_document(text_elements, _fname)
        self._last_doc_type = doc_type  # exposed for callers/tests
        is_legal_prose = doc_type == DOC_TYPE_LEGAL

        # G3 Step C: structural fiscal_year from the filename (refined from grid period
        # headers after the table pass below). Stamped on every chunk at the end so the
        # FY filter chip can narrow retrieval. None = unknown → never excludes the doc.
        self._last_fiscal_year = derive_fiscal_year(_fname)

        if text_elements:
            # ── G1b: clause-aware chunking for prose contracts; chunk_by_title else ──
            prose_chunks = []
            if is_legal_prose:
                prose_chunks = chunk_legal_prose(text_elements, self.config.CHUNK_SIZE)

            normalized = []  # list of (text, page, filename, filetype, section_path)
            if prose_chunks:
                print(f"Chunking TEXT by CLAUSE (legal prose) → {len(prose_chunks)} chunks...")
                for pc in prose_chunks:
                    # Phase 1.5 (DOCUMENT_HARNESS §6.4): persist the clause heading that
                    # chunk_legal_prose already computed, instead of dropping it here — so
                    # read_section is EXACT, not a runtime heuristic.
                    normalized.append((pc["text"], pc.get("page"), _fname, _ftype, pc.get("heading")))
            else:
                print("Chunking TEXT elements by title...")
                text_chunks = chunk_by_title(
                    elements=text_elements,
                    max_characters=self.config.CHUNK_SIZE,
                    new_after_n_chars=self.config.NEW_AFTER_N_CHARS,
                    combine_text_under_n_chars=self.config.COMBINE_TEXT_UNDER_N_CHARS,
                )
                for chunk in text_chunks:
                    chunk_text = chunk.text.strip() if chunk.text else ""
                    if hasattr(chunk, "metadata") and chunk.metadata:
                        page_number = getattr(chunk.metadata, "page_number", None)
                        filepath = getattr(chunk.metadata, "filepath", None)
                        filename = getattr(chunk.metadata, "filename", None)
                        filetype = getattr(chunk.metadata, "filetype", None)
                    else:
                        page_number = filepath = filename = filetype = None
                    # Phase 1.5: chunk_by_title splits ON the title, so the chunk's leading
                    # line IS its section heading — persist it. None when not derivable;
                    # read_section's runtime heuristic (task 1.5.3) is the fallback.
                    section_path = _leading_heading(chunk_text)
                    normalized.append((chunk_text, page_number, filename or filepath, filetype, section_path))

            for i, (chunk_text, page_number, filename, filetype, section_path) in enumerate(normalized, start=1):
                chunk_text = (chunk_text or "").strip()
                if not chunk_text or len(chunk_text) < 10:
                    continue

                metadata = {
                    "chunk_type": "text",
                    "source": filename,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    "doc_type": doc_type,  # G1d — surfaced for the grid/agent + filtering
                }
                # Phase 1.5 (DOCUMENT_HARNESS §6.4): persist the section heading so
                # read_section is exact. Omitted when None (no key churn for table chunks
                # or undeterminable headings → byte-identical to pre-1.5 for those).
                if section_path:
                    metadata["section_path"] = section_path

                metadata["chunk_id"] = _stable_id(
                    file_path=str(filename) if filename else "unknown",
                    chunk_type="text",
                    index=i,
                    text=chunk_text,
                )

                docs.append(Document(page_content=chunk_text, metadata=metadata))

            print(f"Created {len(normalized)} TEXT chunks (doc_type={doc_type}).")

        if table_elements:
            print("Creating TABLE chunks...")

            for i, el in enumerate(table_elements, start=1):
                page_number = _get_page_number(el)

                html = _table_html(el)
                table_text = html if html else (el.text or "")
                table_text = table_text.strip()

                if not table_text:
                    continue

                table_description =_create_table_description(el)

                # Safe metadata extraction
                if hasattr(el, 'metadata') and el.metadata:
                    source = getattr(el.metadata, "filepath", None)
                    filename = getattr(el.metadata, "filename", None)
                    filetype = getattr(el.metadata, "filetype", None)
                else:
                    source = None
                    filename = None
                    filetype = None

                metadata = {
                    "chunk_type": "table",
                    "source": source,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    "table_format": "html" if html else "text",
                    "description": table_description,  
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(source) if source else "unknown",
                    chunk_type="table",
                    index=i,
                    text=table_text,
                )

                docs.append(Document(page_content=table_text, metadata=metadata))

            print(f"Created {len(table_elements)} TABLE chunks.")


        if image_elements:
            print("Creating IMAGE chunks...")

            for i, el in enumerate(image_elements, start=1):
                page_number = _get_page_number(el)

                image_text =_create_image_description(el, page_number)

                # Safe metadata extraction
                if hasattr(el, 'metadata') and el.metadata:
                    source = getattr(el.metadata, "filepath", None)
                    filename = getattr(el.metadata, "filename", None)
                    filetype = getattr(el.metadata, "filetype", None)
                    image_base64 = getattr(el.metadata, "image_base64", None)
                    image_path = getattr(el.metadata, "image_path", None)
                else:
                    source = None
                    filename = None
                    filetype = None
                    image_base64 = None
                    image_path = None

                metadata = {
                    "chunk_type": "image",
                    "source": source,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    "has_image_payload": bool(image_base64),
                    "image_path": image_path,
                    "description": image_text,  
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(source) if source else "unknown",
                    chunk_type="image",
                    index=i,
                    text=image_text,
                )

                docs.append(Document(page_content=image_text, metadata=metadata))

            print(f"Created {len(image_elements)} IMAGE chunks.")

        # ── Phase 4.3: structured table chunks (additive; never regresses text) ──
        # Resolve the PDF path: explicit arg, else the filepath stamped on elements
        # by process_documents. Only PDFs go through the pdfplumber table pass.
        resolved_pdf = pdf_path
        if resolved_pdf is None:
            for el in elements:
                fp = getattr(getattr(el, "metadata", None), "filepath", None)
                if fp and str(fp).lower().endswith(".pdf"):
                    resolved_pdf = str(fp)
                    break
        # ── G1c: skip the financial-table pass on a prose contract ──
        # On a contract the geometry pass only hallucinates a table from a footer
        # (e.g. "[p5_g0] Financial table … sec.gov … Page 5 of") — pure boilerplate
        # masquerading as structured data, which then pollutes table retrieval. The
        # moat (geometry reader) stays fully on for financial/mixed/generic docs.
        skip_tables = getattr(self.config, "CLASSIFY_DOCS", False) and is_legal_prose
        if skip_tables:
            _logger.info("[ingest] G1c: skipping financial-table pass (doc_type=%s)", doc_type)
        elif resolved_pdf and str(resolved_pdf).lower().endswith(".pdf"):
            docs.extend(self._build_table_chunks(resolved_pdf, elements))

        # ── G3 Step C: refine fiscal_year from the extracted statements' period headers
        # (only when the filename gave nothing), then stamp the final FY on EVERY chunk
        # so the FY filter can scope retrieval. doc_type is already on each chunk (G1d);
        # this keeps fiscal_year on the same footing. None = unknown → never excludes.
        if self._last_fiscal_year is None:
            periods: List[str] = []
            for d in docs:
                tj = d.metadata.get("table_json")
                if not tj:
                    continue
                try:
                    grid = json.loads(tj) if isinstance(tj, str) else tj
                    periods.extend(grid.get("periods") or [])
                except Exception:  # noqa: BLE001 — best-effort; FY stays None on parse error
                    pass
            if periods:
                self._last_fiscal_year = derive_fiscal_year(None, periods)
        if self._last_fiscal_year is not None:
            for d in docs:
                d.metadata["fiscal_year"] = self._last_fiscal_year
            _logger.info("[ingest] fiscal_year=%s stamped on %d chunks",
                         self._last_fiscal_year, len(docs))

        print(f"Total LangChain Documents created: {len(docs)}")
        return docs

    def _build_table_chunks(self, pdf_path: str, elements: List) -> List[Document]:
        """Extract confidence-gated tables and turn each into a chunk_type=table Document.

        "Embed the summary, carry the grid": ``page_content`` is the table CAPTION
        (so retrieval matches a natural-language numeric query, not a wall of
        digits), while the normalized JSON grid + provenance live in ``metadata``
        for the deterministic Analyst (§4b). Never raises — extraction failure
        yields zero table chunks, identical to today's behavior.
        """
        try:
            from src.components.table_extraction import extract_tables_from_pdf
        except Exception as exc:
            _logger.warning("[ingest] table extraction unavailable: %s", exc)
            return []

        # Carry the user-facing filename/filetype if the elements know them.
        filename = None
        filetype = None
        for el in elements:
            md = getattr(el, "metadata", None)
            if md is not None:
                filename = filename or getattr(md, "filename", None)
                filetype = filetype or getattr(md, "filetype", None)

        try:
            tables = extract_tables_from_pdf(pdf_path)
        except Exception as exc:
            _logger.warning("[ingest] table extraction failed for %s: %s", pdf_path, exc)
            return []

        # ── Extraction fidelity self-report (ground-truth-free; ANY doc) ──
        # Cross-checks the grids against the PDF's own text layer so a silent row
        # drop on a NEW document is flagged at ingest, not discovered via a wrong/
        # missing answer later. Log-only; never blocks or alters ingestion.
        try:
            from src.components.extraction_fidelity import fidelity_report
            fid = fidelity_report(pdf_path, tables)
            if fid.get("uncovered"):
                # G2 Step F: any uncovered text-layer data line → 'partial' (the trust
                # dot flags it for a reviewer). Full coverage → 'good'. The report ran
                # (pages_checked may be 0 if there were no table pages — then leave the
                # grade unset rather than claim 'good' on no evidence).
                self._last_fidelity = "partial"
                _logger.warning(
                    "[ingest] FIDELITY: %s — %d/%d text-layer data line(s) NOT covered "
                    "by extracted grids (silent row drops?): %s",
                    pdf_path, len(fid["uncovered"]), fid.get("data_lines", 0),
                    [(u["page"], u["line"][:60]) for u in fid["uncovered"][:5]])
            else:
                if fid.get("pages_checked"):
                    self._last_fidelity = "good"
                _logger.info("[ingest] fidelity ok: %s — %d data lines across %d pages covered",
                             pdf_path, fid.get("data_lines", 0), fid.get("pages_checked", 0))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[ingest] fidelity check skipped for %s: %s", pdf_path, exc)

        # ── No per-table LLM summary at ingest (removed 2026-06-12) ──
        # Per-table gpt-4o-mini summarization was the dominant ingest-latency block
        # (INFY: ~89s for 311 tables) AND the per-doc cost center (one call per table).
        # It is OFF the critical path now. Every table embeds with its DETERMINISTIC
        # caption (`embed_caption = t.summary or t.caption`, and `t.summary` is ""),
        # which carries the statement type + period axis already. The $-vs-%-twin
        # disambiguation the summary used to add is recovered downstream by the
        # kernel's currency>percent / exact>contains filters (analyst.py), not at
        # ingest — so removing this never produces a wrong answer, only (rarely) an
        # extra abstain on a twin-table tie, which is the safe direction.
        import json
        out: List[Document] = []
        for t in tables:
            # The grid is stored as a JSON STRING (not a dict): it round-trips
            # losslessly through the embedding path's clean_metadata() and into
            # Supabase JSONB, and stays well under Pinecone's 40KB metadata cap
            # (measured max ~2.8KB). The Analyst json.loads() it back.
            grid_json = json.dumps(t.to_metadata(), ensure_ascii=False)

            # "Embed the summary, carry the grid": prefer the discriminative LLM summary
            # for embedding (it separates near-identical twin tables); fall back to the
            # deterministic caption when the summary is empty (LLM failure) — never worse
            # than today. page_content drives content_hash → chunk_id, and we PREFIX the
            # unique table_id so distinct tables never collide on a shared description.
            embed_caption = t.summary or t.caption
            embed_text = f"[{t.table_id}] {embed_caption}"

            metadata = {
                "chunk_type": "table",
                "source": pdf_path,
                "filename": filename,
                "filetype": filetype,
                "page_number": t.page_number,
                "chunk_index": t.page_number,  # page-ordered; table_id disambiguates
                "table_id": t.table_id,
                "table_format": "structured_json",
                "table_confidence": t.to_metadata()["confidence"],
                "description": embed_caption,
                "table_summary": t.summary,  # the LLM summary, for the lexical ranker (table_intent)
                "table_json": grid_json,     # normalized grid (carries summary too) for the Analyst
                "table_markdown": t.markdown,
            }
            metadata["chunk_id"] = _stable_id(
                file_path=pdf_path,
                chunk_type="table",
                index=t.page_number,
                text=t.table_id + embed_caption,
            )
            out.append(Document(page_content=embed_text, metadata=metadata))

        if out:
            print(f"Created {len(out)} structured TABLE chunks (Phase 4.3).")
        return out

         
    def process_batch(self, directory: str) -> List:
        """Process docs in dir. Skips files that already have cached documents."""
        all_chunks = []

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            return all_chunks

        supported_formats = self.config.SUPPORTED_FILE_TYPES
        files = [
            f for f in os.listdir(directory)
            if not f.startswith('.')
            and Path(os.path.join(directory, f)).suffix.lower().strip(".") in supported_formats
        ]
        print(f"Found {len(files)} supported files in {directory}")

        files_processed = 0
        files_skipped = 0

        for file_name in files:
            file_path = os.path.join(directory, file_name)
            if not os.path.isfile(file_path):
                continue

            # Check JSON cache — safe alternative to pickle
            docs_cache_path = Path(file_path).with_suffix(".docs.json")
            if docs_cache_path.exists():
                try:
                    with open(docs_cache_path, "r") as f:
                        cached_data = json.load(f)
                    cached_docs = [
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in cached_data
                    ]
                    print(f"  Loaded {len(cached_docs)} cached chunks for {file_name} (already processed)")
                    all_chunks.extend(cached_docs)
                    files_skipped += 1
                    continue
                except Exception:
                    print(f"  Cache corrupted for {file_name}, re-processing...")

            # No cache — process from scratch
            try:
                print(f"Processing file: {file_name}")
                elements = self.process_documents(file_paths=file_path)

                if elements:
                    chunks = self.build_langchain_documents(elements=elements, pdf_path=file_path)
                    # Save JSON cache so next run skips this file
                    cache_data = [
                        {"page_content": doc.page_content, "metadata": doc.metadata}
                        for doc in chunks
                    ]
                    with open(docs_cache_path, "w") as f:
                        json.dump(cache_data, f)
                    all_chunks.extend(chunks)
                    files_processed += 1
                    print(f"  Added {len(chunks)} chunks from {file_name}")
                else:
                    print(f"  No elements extracted from {file_name}, skipping chunking.")

            except Exception as e:
                print(f"  Error processing file {file_name}: {e}")

        print(f"\nBatch complete: {files_processed} processed, {files_skipped} skipped (already cached).")
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks

