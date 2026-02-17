import os
import json
import pickle
import sys
from typing import List,Dict,Any
from pathlib import Path
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

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_documents(self, file_paths: str) -> List:
        """Process documents and return a list of processed data."""
        cache_path = Path(file_paths).with_suffix('.pkl')
        file_extension = Path(file_paths).suffix.lower()
        file_name = Path(file_paths).name


        if cache_path.exists():
            try:
                print(f"loading cached eliments {file_paths}")
                with open(cache_path,'rb') as f:
                    return pickle.load(f)
            except Exception:
                print("cache corrupted.")    
        try:
            if file_extension == ".pdf":
                elements = partition_pdf(
                    filename= file_paths,                      #pdf path
                    strategy = 'hi_res',                       #most accurate strategy
                    infer_table_structure = True,               #table in html format
                    extract_image_block_type = ["Image"],      #image grab
                    extract_image_block_to_payload=True,       #store image as base64 in payload 
                )

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

            with open(cache_path,'wb') as f:
                pickle.dump(elements,f)
                
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
                overlap=self.config.CHUNK_OVERLAP,
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
                    "has_overlap": "..." in chunk_text,  
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

            # Simple check: if docs cache exists, load it and skip
            docs_cache_path = Path(file_path).with_suffix(".docs.pkl")
            if docs_cache_path.exists():
                try:
                    with open(docs_cache_path, "rb") as f:
                        cached_docs = pickle.load(f)
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
                    # Save docs cache so next run skips this file
                    with open(docs_cache_path, "wb") as f:
                        pickle.dump(chunks, f)
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
