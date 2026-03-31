"""
Chat endpoints — query, streaming, conversations, messages.
Uses lazy imports for heavy RAG components.
"""

import re
import json

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    CreateConversationRequest,
    ConversationResponse,
    ConversationListResponse,
    SendMessageRequest,
    MessageResponse,
    MessageListResponse,
)
from src.api.dependencies import (
    get_current_user,
    get_user_config,
    get_retrieval_mgr,
    get_generator,
)
from src.components.config import Config

router = APIRouter()

# Import shared limiter (created in server.py at startup)
def _get_limiter():
    from src.api.server import limiter
    return limiter


# ─────────────────────────────────────────
# DIRECT QUERY (no conversation context)
# ─────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    sb=Depends(get_current_user),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Query documents and get an answer.
    Non-streaming — returns the full response at once.
    """
    docs = retrieval_mgr.retrieve(
        body.question,
        filename_filter=body.filename_filter,
        page_filter=body.page_filter,
    )

    if not docs:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents for this question.",
            sources=[],
            num_sources_used=0,
        )

    result = generator.generate(query=body.question, retrieved_docs=docs)

    sources = [
        SourceInfo(
            source_id=s["source_id"],
            filename=s.get("filename"),
            page=s.get("page"),
            chunk_type=s.get("chunk_type"),
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        num_sources_used=result["num_sources_used"],
    )


@router.post("/query/stream")
async def query_stream(
    request: Request,          # P1: required by slowapi
    body: QueryRequest,
    sb=Depends(get_current_user),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Query documents with SSE streaming.
    Returns a stream of tokens followed by a final sources event.
    """
    chat_history = []
    if body.conversation_id:
        existing_msgs = sb.get_messages(body.conversation_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in existing_msgs
        ]

    # Contextualize ambiguous follow-up questions for the retriever
    search_query = generator.rewrite_query(body.question, chat_history)

    docs = retrieval_mgr.retrieve(
        search_query,
        filename_filter=body.filename_filter,
        page_filter=body.page_filter,
    )

    if not docs:
        async def no_results():
            yield f"data: {json.dumps({'type': 'token', 'content': 'I could not find relevant information in your documents for this question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(no_results(), media_type="text/event-stream")

    # Pass the generator directly to StreamingResponse
    return StreamingResponse(
        generator.generate_stream(
            query=body.question, 
            retrieved_docs=docs,
            chat_history=chat_history
        ),
        media_type="text/event-stream"
    )


# ─────────────────────────────────────────
# CONVERSATIONS
# ─────────────────────────────────────────

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    body: CreateConversationRequest,
    sb=Depends(get_current_user),
):
    """Create a new conversation thread."""
    conv = sb.create_conversation(body.title)
    if not conv:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return ConversationResponse(
        id=conv["id"],
        title=conv["title"],
        created_at=conv.get("created_at"),
        updated_at=conv.get("updated_at"),
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(sb=Depends(get_current_user)):
    """List all conversations for the current user, newest first."""
    convs = sb.get_conversations()
    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c["id"],
                title=c["title"],
                created_at=c.get("created_at"),
                updated_at=c.get("updated_at"),
            )
            for c in convs
        ],
        total=len(convs),
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    sb=Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    try:
        sb.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


# ─────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────

@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conversation_id: str,
    sb=Depends(get_current_user),
):
    """Get all messages for a conversation, oldest first."""
    msgs = sb.get_messages(conversation_id)
    return MessageListResponse(
        messages=[
            MessageResponse(
                id=m.get("id"),
                role=m["role"],
                content=m["content"],
                sources=m.get("sources", []),
                created_at=m.get("created_at"),
            )
            for m in msgs
        ],
        conversation_id=conversation_id,
    )


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: str,
    body: SendMessageRequest,
    sb=Depends(get_current_user),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Send a message in a conversation:
    1. Save user message to DB
    2. Retrieve relevant docs
    3. Generate AI response
    4. Save assistant message to DB
    5. Return the assistant message
    """
    existing_msgs = sb.get_messages(conversation_id)
    if len(existing_msgs) == 0:
        sb.auto_title_conversation(conversation_id, body.question)

    sb.save_message(conversation_id, "user", body.question)

    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in existing_msgs
    ]

    page_filter = body.page_filter
    if not page_filter:
        page_match = re.search(r'page\s+(\d+)', body.question.lower())
        if page_match:
            page_filter = int(page_match.group(1))

    docs = retrieval_mgr.retrieve(
        body.question,
        filename_filter=body.filename_filter,
        page_filter=page_filter,
    )

    if not docs:
        answer = "I couldn't find relevant information in your documents for this question."
        msg = sb.save_message(conversation_id, "assistant", answer)
        return MessageResponse(
            id=msg.get("id"),
            role="assistant",
            content=answer,
            sources=[],
            created_at=msg.get("created_at"),
        )

    result = generator.generate(query=body.question, retrieved_docs=docs, chat_history=chat_history)

    serializable_sources = [
        {
            "filename": s.get("filename"),
            "page": s.get("page"),
            "chunk_type": s.get("chunk_type"),
        }
        for s in result["sources"]
    ]

    msg = sb.save_message(
        conversation_id, "assistant", result["answer"], serializable_sources
    )

    return MessageResponse(
        id=msg.get("id"),
        role="assistant",
        content=result["answer"],
        sources=serializable_sources,
        created_at=msg.get("created_at"),
    )
