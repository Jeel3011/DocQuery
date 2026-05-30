"""
DocQuery — Document Comparison (Phase 5)

Compare two documents side-by-side to identify similarities,
differences, and key distinctions using LLM analysis.
"""

import json
import re
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.api.dependencies import get_current_user, get_user_config, get_retrieval_mgr, get_generator
from src.api.schemas import CompareMultiRequest, MultiComparisonResult, DocSummary, ComparisonMatrixRow
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


def _loads_lenient(s: str) -> dict:
    """Parse JSON, stripping leading/trailing code fences if present."""
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return {}


@router.post("/documents/compare-multi", response_model=MultiComparisonResult)
async def compare_multi(
    body: CompareMultiRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
):
    """N-document comparison using map-reduce LLM synthesis."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    try:
        # 1. Resolve document IDs
        if body.document_ids:
            doc_ids = list(dict.fromkeys(body.document_ids))
        elif body.collection_id:
            doc_ids = sb.get_collection_document_ids(body.collection_id)
            if not doc_ids:
                raise HTTPException(status_code=422, detail="Collection is empty.")
        else:
            raise HTTPException(status_code=422, detail="Provide document_ids or collection_id.")

        if len(doc_ids) > user_config.COMPARE_MAX_DOCS:
            raise HTTPException(
                status_code=422,
                detail=f"Too many documents (max {user_config.COMPARE_MAX_DOCS})."
            )
        if len(doc_ids) < 2:
            raise HTTPException(status_code=422, detail="Provide at least 2 documents.")

        # 2. Fetch document records and filter to ready ones
        doc_records = []
        for doc_id in doc_ids:
            doc = sb.get_document(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
            doc_records.append(doc)

        ready_docs = [d for d in doc_records if d.get("status") == "ready"]
        if len(ready_docs) < 2:
            raise HTTPException(
                status_code=422,
                detail="Fewer than 2 documents are ready. Some may still be processing."
            )

        focus_query = body.focus or "key content, main topics, claims, and conclusions"

        # 3. Retrieve representative chunks per doc in parallel
        def _get_chunks(doc: dict) -> tuple[str, str, str]:
            fname = doc["filename"]
            doc_id = doc["id"]
            chunks = retrieval_mgr.retrieve(
                focus_query,
                filename_filter=fname,
                page_filter=None,
            )[:user_config.COMPARE_CHUNKS_PER_DOC]
            text = "\n\n".join(c.page_content for c in chunks)[:3000] if chunks else ""
            return doc_id, fname, text

        max_workers = min(len(ready_docs), user_config.COMPARE_MAP_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            chunk_results = list(pool.map(_get_chunks, ready_docs))

        llm = ChatOpenAI(
            model=user_config.LLM_MODEL_NAME,
            temperature=0.0,
            api_key=user_config.OPENAI_API_KEY,
            request_timeout=30,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        per_doc_summaries: list[DocSummary] = []

        if len(ready_docs) <= user_config.COMPARE_SINGLE_CALL_MAX_DOCS:
            # Single synthesis call — feed all docs at once
            docs_payload = json.dumps([
                {"document_id": did, "filename": fname, "text": text}
                for did, fname, text in chunk_results
            ])
            focus_line = f"\nFocus specifically on: {body.focus}" if body.focus else ""
            single_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a document comparison expert. Return valid JSON only."),
                ("user", (
                    "Compare the following documents and return JSON with keys: "
                    "similarities (list), differences (list), "
                    "per_document (list of {{document_id, filename, summary, key_points}}), "
                    "matrix (list of {{dimension, values: {{filename: value}}}}), summary (string).{focus_line}\n\n"
                    "Documents: {docs_json}"
                )),
            ])
            raw = await asyncio.to_thread(
                (single_prompt | llm | StrOutputParser()).invoke,
                {"focus_line": focus_line, "docs_json": docs_payload},
            )
            parsed = _loads_lenient(raw)

            for item in parsed.get("per_document", []):
                per_doc_summaries.append(DocSummary(
                    document_id=item.get("document_id", ""),
                    filename=item.get("filename", ""),
                    summary=item.get("summary", ""),
                    key_points=item.get("key_points", []),
                ))

            filenames = [fname for _, fname, _ in chunk_results]
            return MultiComparisonResult(
                documents=filenames,
                focus_area=body.focus,
                similarities=parsed.get("similarities", []),
                differences=parsed.get("differences", []),
                per_document=per_doc_summaries,
                matrix=[
                    ComparisonMatrixRow(dimension=r["dimension"], values=r.get("values", {}))
                    for r in parsed.get("matrix", [])
                ],
                summary=parsed.get("summary", "Comparison completed."),
            )

        # Map step: parallel per-doc summaries
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a document analysis expert. Return valid JSON only."),
            ("user", (
                'Summarize document "{filename}" focusing on: {focus_query}. '
                "Return JSON with keys: summary (string), key_points (list of strings).\n\n"
                "Document text: {doc_text}"
            )),
        ])

        def _map_one(args: tuple) -> DocSummary:
            doc_id, fname, text = args
            try:
                raw = (map_prompt | llm | StrOutputParser()).invoke({
                    "filename": fname,
                    "focus_query": focus_query,
                    "doc_text": text or "(no content retrieved)",
                })
                parsed = _loads_lenient(raw)
                return DocSummary(
                    document_id=doc_id,
                    filename=fname,
                    summary=parsed.get("summary", raw[:500]),
                    key_points=parsed.get("key_points", []),
                )
            except Exception as exc:
                logger.warning("Map step failed for %s: %s", fname, exc)
                return DocSummary(
                    document_id=doc_id,
                    filename=fname,
                    summary="(summary unavailable)",
                    key_points=[],
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            per_doc_summaries = list(pool.map(_map_one, chunk_results))

        # Reduce step: one synthesis call
        summaries_payload = json.dumps([
            {
                "filename": s.filename,
                "summary": s.summary,
                "key_points": s.key_points,
            }
            for s in per_doc_summaries
        ])
        focus_line = f"\nFocus specifically on: {body.focus}" if body.focus else ""
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a document comparison expert. Return valid JSON only."),
            ("user", (
                "Given these per-document summaries, return JSON with keys: "
                "similarities (list), differences (list), "
                "matrix (list of {{dimension, values: {{filename: value}}}}), summary (string).{focus_line}\n\n"
                "Summaries: {summaries_json}"
            )),
        ])
        raw_reduce = await asyncio.to_thread(
            (reduce_prompt | llm | StrOutputParser()).invoke,
            {"focus_line": focus_line, "summaries_json": summaries_payload},
        )
        parsed_reduce = _loads_lenient(raw_reduce)

        filenames = [s.filename for s in per_doc_summaries]
        return MultiComparisonResult(
            documents=filenames,
            focus_area=body.focus,
            similarities=parsed_reduce.get("similarities", []),
            differences=parsed_reduce.get("differences", []),
            per_document=per_doc_summaries,
            matrix=[
                ComparisonMatrixRow(dimension=r["dimension"], values=r.get("values", {}))
                for r in parsed_reduce.get("matrix", [])
            ],
            summary=parsed_reduce.get("summary", "Comparison completed."),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("compare-multi failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")
