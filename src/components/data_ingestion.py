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

try:
    from .config import Config
except ImportError:
    from src.components.config import Config

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config

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
            self._log_elements_analysis(elements)

            # source metadata
            for element in elements:
                element.metadata.filename = file_name
                element.metadata.filetype = file_extension.strip(".")
                element.metadata.filepath = file_paths
                
            return elements
        
        except Exception as e:
            print(f"Error processing {file_paths}: {e}")
            return []

    def _log_elements_analysis(self, elements: List) -> None:
        element_types={}
        for el in elements:
            el_type=el.type if hasattr(el, 'type') else type(el).__name__
            element_types[el_type]=element_types.get(el_type,0)+1

        print(f"element breakdown: {element_types}")

    def chunk_elements_by_title(self, elements: List) -> List:
        """title-based chunking of elements"""
        
        print("chunking ......")

        chunks = chunk_by_title(
            elements=elements,
            max_characters= self.config.CHUNK_SIZE,
            new_after_n_chars=self.config.NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=self.config.COMBINE_TEXT_UNDER_N_CHARS
        )

        # add rich and IMP metadata to each chunk
        for i,chunk in enumerate(chunks):
            
            #element type
            element_type = chunk.type if hasattr(chunk, 'type') else "Text"

            # metadata dict - ensure metadata exists
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                from types import SimpleNamespace
                chunk.metadata = SimpleNamespace()

            #metadata add
            chunk.metadata.chunk_id = f"chunk_{i+1}"
            chunk.metadata.element_type = element_type
            chunk.metadata.char_count = len(chunk.text)
            chunk.metadata.chunk_index = i+1
            chunk.metadata.total_chunks = len(chunks)

            #store pointer of images and tables in metadata for genration use
            chunk.metadata.content_info = self._extract_content_info(chunk)     


        print(f"created {len(chunks)} chunks with metadata")
        return chunks
    

    def _extract_content_info(self, chunk) -> Dict[str,Any]:
        """ Extract info about images and tables in chunk"""

        content_info = {
            "has_images": False,
            "has_tables": False,
            "has_text": True,
            "table_count": 0,
            "image_count": 0
        }

        lower_text = chunk.text.lower()

        if "<table" in lower_text or "<tr>" in lower_text:
            content_info["has_tables"] = True
            content_info["table_count"] = lower_text.count("<table")

        # Check for base64 images in metadata
        if hasattr(chunk, 'metadata') and chunk.metadata:
            # Check if image_base64 has actual content
            image_base64 = getattr(chunk.metadata, 'image_base64', None)
            if image_base64:
                content_info["has_images"] = True
                # Count base64 images
                if isinstance(image_base64, (list, tuple)):
                    content_info["image_count"] = len(image_base64)
                else:
                    content_info["image_count"] = 1
        
        # Also check for image tags in text
        if '<image' in lower_text:
            content_info["has_images"] = True
            content_info["image_count"] += lower_text.count("<image")

        return content_info
         
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
                        chunks = self.chunk_elements_by_title(elements=elements)
                        all_chunks.extend(chunks)
                        print(f"added {len(chunks)} chunks from {file_name}")

                    else:
                        print(f"No elements extracted from {file_name}, skipping chunking.")

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")


        print(f"processed completed. Total chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # testing
    config=Config()
    processor = DocumentProcessor(config=config)
    chunks = processor.process_batch('./docs')
    if chunks:
        print(json.dumps(chunks[0].metadata.__dict__, indent=2,default=str))
