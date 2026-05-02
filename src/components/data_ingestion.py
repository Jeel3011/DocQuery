import os
import json
import sys
import math
import time
import tempfile
from typing import List,Dict,Any
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


def _get_pdf_page_count(file_path: str) -> int:
    """Get the number of pages in a PDF without fully parsing it."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception:
        return 0


def _process_pdf_page_range(
    file_path: str,
    start_page: int,
    end_page: int,
    strategy: str,
    extract_images: bool,
) -> list:
    """Process a page range of a PDF. Used as a worker function for parallel processing.

    Extracts pages [start_page, end_page) into a temp file, then runs
    partition_pdf on that slice. This avoids the overhead of the layout model
    scanning pages that another worker is handling.
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

            # Fix page numbers: partition_pdf will number pages starting at 1
            # within the slice, but we need the global page number.
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

    def _process_pdf_parallel(self, file_path: str) -> List:
        """Split a PDF into page-range chunks and process in parallel.

        Gives ~3-5x speedup on multi-core machines for large PDFs.
        Falls back to single-pass for small PDFs or if splitting fails.
        """
        total_pages = _get_pdf_page_count(file_path)
        workers = self.config.PDF_PARALLEL_WORKERS

        # Not worth parallelizing small PDFs (overhead > benefit)
        if total_pages < 6 or workers <= 1:
            return self._process_pdf_single(file_path)

        # Split into roughly equal page ranges
        pages_per_worker = math.ceil(total_pages / workers)
        ranges = []
        for start in range(0, total_pages, pages_per_worker):
            end = min(start + pages_per_worker, total_pages)
            ranges.append((start, end))

        print(f"  Parallel PDF processing: {total_pages} pages -> {len(ranges)} workers "
              f"({pages_per_worker} pages/worker)")

        t_start = time.perf_counter()
        all_elements = []

        with ProcessPoolExecutor(max_workers=len(ranges)) as pool:
            futures = {
                pool.submit(
                    _process_pdf_page_range,
                    file_path,
                    start,
                    end,
                    self.config.PDF_STRATEGY,
                    self.config.EXTRACT_IMAGES,
                ): (start, end)
                for start, end in ranges
            }

            # Collect results in page order
            results_by_start = {}
            for future in as_completed(futures):
                start, end = futures[future]
                try:
                    elements = future.result()
                    results_by_start[start] = elements
                    print(f"    Pages {start+1}-{end}: {len(elements)} elements")
                except Exception as e:
                    print(f"    Pages {start+1}-{end}: FAILED ({e})")
                    results_by_start[start] = []

        # Merge in page order
        for start in sorted(results_by_start.keys()):
            all_elements.extend(results_by_start[start])

        elapsed = time.perf_counter() - t_start
        print(f"  Parallel processing complete: {len(all_elements)} elements in {elapsed:.1f}s")
        return all_elements

    def _process_pdf_single(self, file_path: str) -> List:
        """Standard single-pass PDF processing."""
        pdf_kwargs = {
            "filename": file_path,
            "strategy": self.config.PDF_STRATEGY,
            "infer_table_structure": True,
        }
        if self.config.EXTRACT_IMAGES:
            pdf_kwargs["extract_image_block_type"] = ["Image"]
            pdf_kwargs["extract_image_block_to_payload"] = True
        return partition_pdf(**pdf_kwargs)

    def process_documents(self, file_paths: str) -> List:
        """Process documents and return a list of processed data."""
        file_extension = Path(file_paths).suffix.lower()
        file_name = Path(file_paths).name

        try:
            if file_extension == ".pdf":
                if self.config.PARALLEL_PDF_PAGES:
                    elements = self._process_pdf_parallel(file_paths)
                else:
                    elements = self._process_pdf_single(file_paths)

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

