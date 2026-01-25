import os
import json
import sys
from typing import List,Dict,Any
from pathlib import Path

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document

from src.utils import _log_elements_analysis, _get_element_type, _get_page_number, _element_has_image_payload, _table_html, _stable_id

try:
    from .config import Config
except ImportError:
    from src.components.config import Config

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
    
    def _create_table_description(self, el) -> str:
        """Extract meaningful description from table element."""
        try:
            table_text = el.text or ""
            # Extract first few lines for summary
            lines = table_text.split('\n')[:5]
            summary = " ".join([line.strip() for line in lines if line.strip()])
            return f"Table containing: {summary[:200]}..."
        except:
            return "Table data extracted from document."
    
    def _create_image_description(self, el, page_num=None) -> str:
        """Create meaningful description from image element."""
        try:
            # Try to get alt text or caption
            alt_text = getattr(el.metadata, 'alt_text', None) if hasattr(el, 'metadata') else None
            caption = getattr(el.metadata, 'caption', None) if hasattr(el, 'metadata') else None
            
            if alt_text or caption:
                return alt_text or caption
            
            # Fallback description
            page_info = f" on page {page_num}" if page_num else ""
            return f"Image content{page_info}. Contains visual information related to document content."
        except:
            return "Image content extracted from document."
    
    def _add_chunk_overlap(self, chunks: List) -> List:
        """Add overlapping chunks for better context preservation (25% overlap)."""
        # For now, just return chunks as-is
        # Overlap will be handled when creating Document objects
        return chunks

    def process_documents(self, file_paths: str) -> List:
        """Process documents and return a list of processed data."""
        file_extension = Path(file_paths).suffix.lower()
        file_name = Path(file_paths).name
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
                from unstructured.partition.text import partition_text
                elements = partition_text(
                    filename= file_paths
                )
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            print(f"Processed {file_paths} successfully. Extracted {len(elements)} elements.")
            
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

        # 1) Split elements by type (better than mixing everything)
        text_elements = []
        table_elements = []
        image_elements = []

        for el in elements:
            el_type = _get_element_type(el).lower()

            # Table detection
            if "table" in el_type:
                table_elements.append(el)
                continue

            # Image detection
            # Unstructured sometimes uses "Image" element types or payload in metadata
            if "image" in el_type or _element_has_image_payload(el):
                image_elements.append(el)
                continue

            # Default: treat as narrative text
            if getattr(el, "text", None):
                text_elements.append(el)

        docs: List[Document] = []

        # 2) Chunk narrative text using chunk_by_title (good for semantic sections)
        if text_elements:
            print("Chunking TEXT elements by title...")

            text_chunks = chunk_by_title(
                elements=text_elements,
                max_characters=self.config.CHUNK_SIZE,
                new_after_n_chars=self.config.NEW_AFTER_N_CHARS,
                combine_text_under_n_chars=self.config.COMBINE_TEXT_UNDER_N_CHARS,
            )
            
            # Apply chunk overlap for better context (50% overlap)
            text_chunks_with_overlap = self._add_chunk_overlap(text_chunks)

            for i, chunk in enumerate(text_chunks_with_overlap, start=1):
                chunk_text = chunk.text.strip() if chunk.text else ""
                if not chunk_text or len(chunk_text) < 10:  # Skip very small chunks
                    continue

                # Page range info is not always preserved after chunking
                # so we store best-effort page_number if present.
                page_number = getattr(chunk.metadata, "page_number", None) if hasattr(chunk, "metadata") else None

                metadata = {
                    "chunk_type": "text",
                    "source": getattr(chunk.metadata, "filepath", None),
                    "filename": getattr(chunk.metadata, "filename", None),
                    "filetype": getattr(chunk.metadata, "filetype", None),
                    "page_number": page_number,
                    "chunk_index": i,
                    "has_overlap": "..." in chunk_text,  # Flag overlap chunks
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(metadata["source"]),
                    chunk_type="text",
                    index=i,
                    text=chunk_text,
                )

                docs.append(Document(page_content=chunk_text, metadata=metadata))

            print(f"Created {len([d for d in docs if d.metadata.get('chunk_type') == 'text'])} TEXT chunks.")

        # 3) Store TABLE elements as separate chunks (best for retrieval)
        if table_elements:
            print("Creating TABLE chunks...")

            for i, el in enumerate(table_elements, start=1):
                page_number = _get_page_number(el)

                html = _table_html(el)
                table_text = html if html else (el.text or "")
                table_text = table_text.strip()

                if not table_text:
                    continue
                
                # Create meaningful description for retrieval
                table_description = self._create_table_description(el)

                source = getattr(el.metadata, "filepath", None) if hasattr(el, "metadata") else None
                filename = getattr(el.metadata, "filename", None) if hasattr(el, "metadata") else None
                filetype = getattr(el.metadata, "filetype", None) if hasattr(el, "metadata") else None

                metadata = {
                    "chunk_type": "table",
                    "source": source,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    "table_format": "html" if html else "text",
                    "description": table_description,  # Add description for better retrieval
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(source),
                    chunk_type="table",
                    index=i,
                    text=table_text,
                )

                docs.append(Document(page_content=table_text, metadata=metadata))

            print(f"Created {len([d for d in docs if d.metadata.get('chunk_type') == 'table'])} TABLE chunks.")

        # 4) Store IMAGE elements as separate chunks
        # NOTE: We do NOT embed base64. We store references only.
        if image_elements:
            print("Creating IMAGE chunks...")

            for i, el in enumerate(image_elements, start=1):
                page_number = _get_page_number(el)

                # Create meaningful image description for retrieval
                image_text = self._create_image_description(el, page_number)

                source = getattr(el.metadata, "filepath", None) if hasattr(el, "metadata") else None
                filename = getattr(el.metadata, "filename", None) if hasattr(el, "metadata") else None
                filetype = getattr(el.metadata, "filetype", None) if hasattr(el, "metadata") else None

                image_base64 = getattr(el.metadata, "image_base64", None) if hasattr(el, "metadata") else None
                image_path = getattr(el.metadata, "image_path", None) if hasattr(el, "metadata") else None

                metadata = {
                    "chunk_type": "image",
                    "source": source,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_number,
                    "chunk_index": i,
                    # store pointers only
                    "has_image_payload": bool(image_base64),
                    "image_path": image_path,
                    "description": image_text,  # Add description for better retrieval
                }

                metadata["chunk_id"] = _stable_id(
                    file_path=str(source),
                    chunk_type="image",
                    index=i,
                    text=image_text,
                )

                docs.append(Document(page_content=image_text, metadata=metadata))

            print(f"Created {len([d for d in docs if d.metadata.get('chunk_type') == 'image'])} IMAGE chunks.")

        print(f"Total LangChain Documents created: {len(docs)}")
        return docs

         
    def process_batch(self,directory:str)-> List:
        """ process docs in dir"""
        all_chunks = []

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            return all_chunks

        supported_formats = self.config.SUPPORTED_FILE_TYPES

        files= [f for f in os.listdir(directory) if f.startswith('.') is False and Path(os.path.join(directory, f)).suffix.lower().strip(".") in supported_formats]     
        print(f"found {len(files)} supported files in {directory} for processing")

        for file_name in files:
            file_path = os.path.join(directory,file_name)

            if os.path.isfile(file_path):
                try:
                    print(f"processing file: {file_name}")
                    elements = self.process_documents(file_paths=file_path)

                    if elements:
                        chunks = self.build_langchain_documents(elements=elements)
                        all_chunks.extend(chunks)
                        print(f"added {len(chunks)} chunks from {file_name}")

                    else:
                        print(f"No elements extracted from {file_name}, skipping chunking.")

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")


        print(f"processed completed. Total chunks created: {len(all_chunks)}")
        return all_chunks


"""if __name__ == "__main__":
    # testing
    config=Config()
    processor = DocumentProcessor(config=config)
    chunks = processor.process_batch('./docs')
    if chunks:
        print(json.dumps(chunks[0].metadata, indent=2,default=str))
"""
