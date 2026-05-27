"""
DocQuery — Document Comparison (Phase 5)

Compare two documents side-by-side to identify similarities,
differences, and key distinctions using LLM analysis.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.api.dependencies import get_current_user, get_user_config, get_retrieval_mgr, get_generator
from src.components.config import Config
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str
    focus: Optional[str] = None  # e.g., "legal terms", "pricing", "methodology"


class ComparisonResult(BaseModel):
    document_a: str  # filename
    document_b: str  # filename
    similarities: list[str]
    differences: list[str]
    summary: str
    focus_area: Optional[str] = None


@router.post("/documents/compare", response_model=ComparisonResult)
async def compare_documents(
    body: CompareRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """Compare two documents to identify similarities and differences.

    Retrieves top chunks from each document and uses the LLM to produce
    a structured comparison.
    """
    import asyncio
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Verify documents exist and belong to user
    doc_a = sb.get_document(body.document_id_a)
    doc_b = sb.get_document(body.document_id_b)
    if not doc_a:
        raise HTTPException(status_code=404, detail="Document A not found.")
    if not doc_b:
        raise HTTPException(status_code=404, detail="Document B not found.")

    filename_a = doc_a["filename"]
    filename_b = doc_b["filename"]

    # Retrieve representative chunks from each document
    query = body.focus or f"key content and main topics of the document"

    chunks_a = await asyncio.to_thread(
        retrieval_mgr.retrieve, query, filename_a, None
    )
    chunks_b = await asyncio.to_thread(
        retrieval_mgr.retrieve, query, filename_b, None
    )

    if not chunks_a and not chunks_b:
        raise HTTPException(
            status_code=422,
            detail="No content found in either document. Both may still be processing."
        )

    text_a = "\n\n".join([c.page_content for c in chunks_a[:5]])
    text_b = "\n\n".join([c.page_content for c in chunks_b[:5]])

    # LLM comparison
    focus_instruction = f"\nFocus specifically on: {body.focus}" if body.focus else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a document comparison expert. Compare the two documents and identify:
1. Key SIMILARITIES (3-5 bullet points)
2. Key DIFFERENCES (3-5 bullet points)
3. A brief SUMMARY of the comparison (2-3 sentences){focus_instruction}

Format your response EXACTLY as:
SIMILARITIES:
- point 1
- point 2
...

DIFFERENCES:
- point 1
- point 2
...

SUMMARY:
Your summary here."""),
        ("user", f"""Document A: "{filename_a}"
{text_a[:3000]}

---

Document B: "{filename_b}"
{text_b[:3000]}"""),
    ])

    llm = ChatOpenAI(
        model=user_config.LLM_MODEL_NAME,
        temperature=0.0,
        api_key=user_config.OPENAI_API_KEY,
        request_timeout=30,
    )

    try:
        raw = await asyncio.to_thread(
            lambda: (prompt | llm | StrOutputParser()).invoke({})
        )
    except Exception as exc:
        logger.exception("LLM comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(exc)}")

    # Parse the structured response
    similarities = []
    differences = []
    summary = ""

    section = None
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("SIMILARITIES"):
            section = "sim"
            continue
        elif line.upper().startswith("DIFFERENCES"):
            section = "diff"
            continue
        elif line.upper().startswith("SUMMARY"):
            section = "sum"
            continue

        if line.startswith("- ") or line.startswith("• "):
            point = line[2:].strip()
            if section == "sim":
                similarities.append(point)
            elif section == "diff":
                differences.append(point)
        elif section == "sum" and line:
            summary += (" " + line if summary else line)

    return ComparisonResult(
        document_a=filename_a,
        document_b=filename_b,
        similarities=similarities or ["No significant similarities identified"],
        differences=differences or ["No significant differences identified"],
        summary=summary or "Comparison completed.",
        focus_area=body.focus,
    )
