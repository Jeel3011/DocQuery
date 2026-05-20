"""
Chat endpoints — query, streaming, conversations, messages.
Uses lazy imports for heavy RAG components.
"""

import os
import asyncio
import re
import json
import time

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
    limiter,
)
from src.components.config import Config
from src.components.metrics import queries_total, retrieval_docs, cache_hits, cache_misses, cache_latency

router = APIRouter()

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _get_cache(user_id: str):
    """Return a SemanticCache scoped to this user. Non-fatal if Redis is down."""
    from src.components.semantic_cache import SemanticCache
    return SemanticCache(redis_url=_REDIS_URL, namespace=user_id)


async def _embed_query(query: str, model: str, api_key: str) -> list:
    """Embed a query string for cache lookup. Runs in thread pool to avoid blocking."""
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(model=model, api_key=api_key)
    return await asyncio.to_thread(embedder.embed_query, query)


# -----------------------------------------
# DIRECT QUERY (no conversation context)
# -----------------------------------------

@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    request: Request,          # required by slowapi
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Query documents and get an answer.
    Non-streaming — returns the full response at once.
    Checks semantic cache before running RAG pipeline (Phase 2).
    """
    # ── Phase 2: Semantic cache check ──
    t_cache = time.perf_counter()
    try:
        query_embedding = await _embed_query(
            body.question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
        cache = _get_cache(sb.user_id)
        cached = cache.get(body.question, query_embedding)
    except Exception:
        cached = None
        query_embedding = []
    cache_latency.observe(time.perf_counter() - t_cache)

    if cached:
        cache_hits.labels(tier=cached.get("tier", "unknown")).inc()
        sources = [
            SourceInfo(
                source_id=s.get("source_id", 0),
                filename=s.get("filename"),
                page=s.get("page"),
                chunk_type=s.get("chunk_type"),
            )
            for s in cached.get("sources", [])
        ]
        return QueryResponse(
            answer=cached["answer"],
            sources=sources,
            num_sources_used=len(sources),
        )
    cache_misses.inc()

    if user_config.USE_MULTI_QUERY:
        variants = await asyncio.to_thread(
            generator.generate_query_variants, body.question, user_config.MULTI_QUERY_COUNT
        )
        variants = [body.question] + variants
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            body.page_filter,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            body.question,
            body.filename_filter,
            body.page_filter,
        )

    # ── Prometheus metrics ──
    queries_total.labels(endpoint="query", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

    if not docs:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents for this question.",
            sources=[],
            num_sources_used=0,
        )

    result = await asyncio.to_thread(
        generator.generate, query=body.question, retrieved_docs=docs
    )

    sources = [
        SourceInfo(
            source_id=s["source_id"],
            filename=s.get("filename"),
            page=s.get("page"),
            chunk_type=s.get("chunk_type"),
        )
        for s in result["sources"]
    ]

    # ── Phase 2: Store in cache after successful generation ──
    if query_embedding and result.get("answer"):
        try:
            cache.set(
                body.question,
                query_embedding,
                result["answer"],
                [s.model_dump() for s in sources],
            )
        except Exception:
            pass  # cache write failure is non-fatal

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        num_sources_used=result["num_sources_used"],
    )


@router.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(
    request: Request,          # P1: required by slowapi
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Query documents with SSE streaming.
    Phase 2: Checks cache first — on hit, serves cached answer as SSE tokens (sub-50ms).
    On miss, runs full streaming RAG pipeline.
    """
    chat_history = []
    if body.conversation_id:
        existing_msgs = sb.get_messages(body.conversation_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in existing_msgs
        ]

    # ── Phase 2: Cache check before doing any retrieval ──
    t_cache = time.perf_counter()
    cached = None
    query_embedding = []
    try:
        query_embedding = await _embed_query(
            body.question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
        cache = _get_cache(sb.user_id)
        cached = cache.get(body.question, query_embedding)
    except Exception:
        pass
    cache_latency.observe(time.perf_counter() - t_cache)

    if cached:
        cache_hits.labels(tier=cached.get("tier", "unknown")).inc()
        async def cached_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', [])})}\n\n"
            # Stream the cached answer token by token (words) for natural UX
            words = cached["answer"].split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cache_hit': True, 'similarity': cached.get('similarity', 1.0)})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    cache_misses.inc()

    # Contextualize ambiguous follow-up questions for the retriever
    search_query = await asyncio.to_thread(
        generator.rewrite_query, body.question, chat_history
    )

    if user_config.USE_MULTI_QUERY:
        variants = await asyncio.to_thread(
            generator.generate_query_variants, search_query, user_config.MULTI_QUERY_COUNT
        )
        variants = [search_query] + variants
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            body.page_filter,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            search_query,
            body.filename_filter,
            body.page_filter,
        )

    # ── Prometheus metrics ──
    queries_total.labels(endpoint="query_stream", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

    if not docs:
        async def no_results():
            yield f"data: {json.dumps({'type': 'token', 'content': 'I could not find relevant information in your documents for this question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    # B3 FIX: Pass search_query (rewritten standalone query) to the generator
    return StreamingResponse(
        generator.generate_stream(
            query=search_query,
            retrieved_docs=docs,
            chat_history=chat_history
        ),
        media_type="text/event-stream"
    )


# -----------------------------------------
# AGENTIC QUERY (Phase 6)
# -----------------------------------------

@router.post("/query/agent", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_agent(
    request: Request,
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Agentic query endpoint (Phase 6) — Harvey-level accuracy.

    Slower than /query but higher accuracy for complex questions:
      1. Decompose the query into 2-4 atomic sub-questions.
      2. Retrieve relevant docs for each sub-question.
      3. Deduplicate and merge retrieved docs.
      4. Generate with self-critique loop (flags unsupported claims, revises if needed).

    Rate limit: 10/min (2x LLM calls per request vs standard 1x).
    """
    from src.components.agentic_retrieval import AgenticRetriever

    # Step 1+2+3: Agentic retrieval
    agentic = AgenticRetriever(config=user_config, retrieval_mgr=retrieval_mgr)
    retrieval_result = await asyncio.to_thread(
        agentic.retrieve_and_synthesize,
        body.question,
        body.filename_filter,
        body.page_filter,
    )
    docs = retrieval_result["docs"]

    queries_total.labels(endpoint="query_agent", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

    if not docs:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents for this question.",
            sources=[],
            num_sources_used=0,
        )

    # Step 4: Self-critique generation
    result = await asyncio.to_thread(
        generator.generate_with_self_review,
        query=body.question,
        retrieved_docs=docs,
        user_id=sb.user_id,
    )

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


# -----------------------------------------
# CONVERSATIONS
# -----------------------------------------

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


# -----------------------------------------
# MESSAGES
# -----------------------------------------

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
@limiter.limit("30/minute")
async def send_message(
    request: Request,          # required by slowapi
    conversation_id: str,
    body: SendMessageRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Send a message in a conversation:
    1. Save user message to DB
    2. Retrieve relevant docs (optionally via multi-query)
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

    # Rewrite query for standalone context
    search_query = await asyncio.to_thread(
        generator.rewrite_query, body.question, chat_history
    )

    if user_config.USE_MULTI_QUERY:
        variants = await asyncio.to_thread(
            generator.generate_query_variants, search_query, user_config.MULTI_QUERY_COUNT
        )
        variants = [search_query] + variants
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            page_filter,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            search_query,
            body.filename_filter,
            page_filter,
        )

    # -- Prometheus metrics --
    queries_total.labels(endpoint="send_message", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

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

    # B3 FIX: Pass search_query (rewritten) to generator so the LLM
    # sees a standalone question, not an ambiguous follow-up.
    result = await asyncio.to_thread(
        generator.generate,
        query=search_query,
        retrieved_docs=docs,
        chat_history=chat_history,
    )

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
