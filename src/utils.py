import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.exception import CustomException
import hashlib

def _stable_id(file_path: str, chunk_type: str, index: int, text: str) -> str:
    
    raw = f"{file_path}::{chunk_type}::{index}::{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _get_element_type(el) -> str:
    
    if hasattr(el, "category") and el.category:
        return str(el.category)
    if hasattr(el, "type") and el.type:
        return str(el.type)
    
    return type(el).__name__
def _get_page_number(el) -> Optional[int]:
   
    if hasattr(el, "metadata") and el.metadata:
        return getattr(el.metadata, "page_number", None)
    return None

def _element_has_image_payload(el) -> bool:
   
    if not hasattr(el, "metadata") or not el.metadata:
        return False
    image_base64 = getattr(el.metadata, "image_base64", None)
    image_path = getattr(el.metadata, "image_path", None)
    return bool(image_base64 or image_path)

def _table_html(el) -> Optional[str]:
    
    if not hasattr(el, "metadata") or not el.metadata:
        return None
    return getattr(el.metadata, "text_as_html", None)

def _log_elements_analysis(elements: List) -> None:
    element_types = {}
    for el in elements:
        el_type = _get_element_type(el)
        element_types[el_type] = element_types.get(el_type, 0) + 1

    print(f"Element breakdown: {element_types}")

def _create_table_description(el) -> str:
        try:
            table_text = el.text or ""
            lines = table_text.split('\n')[:5]
            summary = " ".join([line.strip() for line in lines if line.strip()])
            return f"Table containing: {summary[:200]}..."
        except Exception:
            return "Table data extracted from document."

def _create_image_description(el, page_num=None) -> str:
    try:
            
        alt_text = getattr(el.metadata, 'alt_text', None) if hasattr(el, 'metadata') else None
        caption = getattr(el.metadata, 'caption', None) if hasattr(el, 'metadata') else None
            
        if alt_text or caption:
            return alt_text or caption
            
        page_info = f" on page {page_num}" if page_num else ""
        return f"Image content{page_info}. Contains visual information related to document content."
    except Exception:
        return "Image content extracted from document."        

        