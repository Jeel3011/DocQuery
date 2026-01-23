import os
from typing import List
from pathlib import Path

from unstructured.partition.pdf import partion_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx

from unstructured.chunking.title import partition_title

from src.components.config import config

#def doc_ingestion(file_path:str) -> list[str]:
    