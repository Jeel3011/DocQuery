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

    # Pull representative text straight from each document's stored chunks
    # (ordered by chunk_index). Why not semantic search? A vague query like
    # "key content and main topics" routinely scores below SIMILARITY_THRESHOLD,
    # so retrieve() filtered everything out and the endpoint 422'd ("No content
    # found") even though both documents were fully processed. The stored chunks
    # always exist for a ready document, so reading them directly is reliable
    # (and avoids an embed + Pinecone round-trip).
    rows_a = await asyncio.to_thread(sb.get_document_chunks, body.document_id_a)
    rows_b = await asyncio.to_thread(sb.get_document_chunks, body.document_id_b)

    if not rows_a and not rows_b:
        raise HTTPException(
            status_code=422,
            detail="No content found in either document. Both may still be processing."
        )

    def _doc_text(rows: list, max_chunks: int = 8, max_chars: int = 3000) -> str:
        return "\n\n".join(r.get("content", "") for r in rows[:max_chunks])[:max_chars]

    text_a = _doc_text(rows_a)
    text_b = _doc_text(rows_b)

    # LLM comparison.
    # IMPORTANT: the document text is passed as TEMPLATE VARIABLES (via invoke),
    # never baked into the template string. ChatPromptTemplate parses "{...}" in the
    # template as input variables — and document text routinely contains "{" / "}"
    # (code, JSON, math, citations), which made from_messages treat them as missing
    # variables and raise, surfacing as a 500 "Comparison failed". Values handed to
    # invoke() are NOT re-parsed, so braces inside the documents are now safe.
    focus_instruction = f"\nFocus specifically on: {body.focus}" if body.focus else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document comparison expert. Compare the two documents and identify:
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
        ("user", 'Document A: "{filename_a}"\n{text_a}\n\n---\n\nDocument B: "{filename_b}"\n{text_b}'),
    ])

    llm = ChatOpenAI(
        model=user_config.LLM_MODEL_NAME,
        temperature=0.0,
        api_key=user_config.OPENAI_API_KEY,
        request_timeout=30,
    )

    try:
        chain = prompt | llm | StrOutputParser()
        raw = await asyncio.to_thread(
            chain.invoke,
            {
                "focus_instruction": focus_instruction,
                "filename_a": filename_a,
                "text_a": text_a[:3000],
                "filename_b": filename_b,
                "text_b": text_b[:3000],
            },
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
