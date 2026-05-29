import os
import io
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



class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config

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
            if page_count <= self.config.PDF_MEDIUM_THRESHOLD_PAGES:
                return "auto"
            # A5: a long PDF only needs expensive OCR (hi_res) if it's genuinely
            # scanned. If it carries a real text layer, "auto" is far cheaper and
            # just as accurate. Probe a sample; fall back to hi_res on uncertainty.
            density = _pdf_text_density(file_path)
            if density >= self.config.PDF_TEXT_LAYER_MIN_CHARS_PER_PAGE:
                _logger.info(
                    "A5: %d-page PDF has text layer (%.0f chars/page) — using 'auto', skipping OCR",
                    page_count, density,
                )
                return "auto"
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


    def build_langchain_documents(self, elements: List) -> List[Document]:

        
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

        if text_elements:
            print("Chunking TEXT elements by title...")

            text_chunks = chunk_by_title(
                elements=text_elements,
                max_characters=self.config.CHUNK_SIZE,
                new_after_n_chars=self.config.NEW_AFTER_N_CHARS,
                combine_text_under_n_chars=self.config.COMBINE_TEXT_UNDER_N_CHARS, 
            )

            for i, chunk in enumerate(text_chunks, start=1):
                chunk_text = chunk.text.strip() if chunk.text else ""
                if not chunk_text or len(chunk_text) < 10:  
                    continue

                # Safe metadata extraction - handle both object and dict
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    page_number = getattr(chunk.metadata, "page_number", None)
                    filepath = getattr(chunk.metadata, "filepath", None)
                    filename = getattr(chunk.metadata, "filename", None)
                    filetype = getattr(chunk.metadata, "filetype", None)
                else:
                    page_number = None
                    filepath = None
                    filename = None
                    filetype = None

                metadata = {
                    "chunk_type": "text",
                    "source": filepath if filepath else filename,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    # B7: removed bogus "has_overlap" field (was checking for literal "..." in text)
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(filepath) if filepath else "unknown",
                    chunk_type="text",
                    index=i,
                    text=chunk_text,
                )

                docs.append(Document(page_content=chunk_text, metadata=metadata))

            print(f"Created {len(text_chunks)} TEXT chunks.")

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

        print(f"Total LangChain Documents created: {len(docs)}")
        return docs

         
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
                    chunks = self.build_langchain_documents(elements=elements)
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

