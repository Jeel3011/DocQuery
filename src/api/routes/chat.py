"""
Chat endpoints — query, streaming, conversations, messages.
Uses lazy imports for heavy RAG components.
"""

import os
import asyncio
import re
import json
import time
import threading

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
from src.api.routes.audit import log_audit
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _resolve_collection_filters(
    sb,
    collection_id: str = None,
    query: str = None,
    query_embedding: list = None,
    user_config=None,
) -> list[str] | None:
    """Resolve a collection_id to a filename list for Pinecone filtering.

    Phase 3+: when a DocumentRouter is available, uses Stage-1 routing to return
    only the top-N most relevant docs (caps fan-out per Invariant R1).  Falls back
    to the full filename list for small collections or when routing data is absent.

    Returns:
        None  — no collection scope (search all docs).
        []    — empty collection.
        list  — filenames to scope retrieval to.
    """
    if not collection_id:
        return None

    # Resolve full filename list (needed for fallback + size check)
    doc_ids = sb.get_collection_document_ids(collection_id)
    if not doc_ids:
        return []
    docs_in_coll = sb.client.table("documents").select("filename").in_(
        "id", doc_ids
    ).eq("user_id", sb.user_id).execute()
    all_filenames = [d["filename"] for d in (docs_in_coll.data or [])]

    if not all_filenames:
        return []

    # Small collection: skip routing overhead, fanout is already safe
    max_fanout = getattr(user_config, "ROUTING_MAX_FANOUT", 8) if user_config else 8
    if len(all_filenames) <= max_fanout:
        return all_filenames

    # Large collection: use Stage-1 router to narrow to top-N docs
    if query and user_config:
        try:
            from src.components.document_router import DocumentRouter
            router = DocumentRouter(user_config)
            routed = router.route(
                query=query,
                collection_id=collection_id,
                user_id=sb.user_id,
                query_embedding=query_embedding,
            )
            if routed:
                logger.info(
                    "Stage-1 router: collection %s has %d docs → routed to top %d",
                    collection_id, len(all_filenames), len(routed),
                )
                return routed
        except Exception as exc:
            logger.warning("Stage-1 router failed, falling back to full list: %s", exc)

    # Fallback: return full list (ROUTING_MAX_FANOUT cap applied inside retrieve_across_files)
    return all_filenames


def _saving_stream_wrapper(
    inner_gen, sb, conversation_id: str, question: str,
    is_agentic: bool = False, is_cache_hit: bool = False,
    retrieval_docs_count: int = 0, web_search_used: bool = False,
    history_len: int | None = None,
):
    """Wrap an SSE generator to capture tokens and save messages to DB after streaming.

    Intercepts 'token' events to collect the full answer text.
    After the stream ends ([DONE]), saves both the user question and
    assistant response to the conversation in the database.
    Also logs the query to query_logs for analytics.
    """
    collected_tokens = []
    collected_sources = []
    t_start = time.perf_counter()

    for chunk in inner_gen:
        yield chunk  # Forward to client immediately

        # Parse the SSE line to capture tokens
        line = chunk.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[6:])
            if data.get("type") == "token" and data.get("content"):
                collected_tokens.append(data["content"])
            elif data.get("type") == "sources":
                collected_sources = data.get("sources", [])
            elif data.get("type") == "meta" and data.get("cache_hit"):
                is_cache_hit = True
            elif data.get("type") == "web_search":
                web_search_used = True
        except (json.JSONDecodeError, KeyError):
            pass

    # Stream finished — save messages to DB
    full_answer = "".join(collected_tokens)
    latency_ms = int((time.perf_counter() - t_start) * 1000)

    if conversation_id and full_answer:
        try:
            # B4: one bulk INSERT + one updated_at bump instead of 2×(insert+update).
            sb.save_messages(conversation_id, [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer, "sources": collected_sources},
            ])
            # Auto-title on the first turn. Use the caller-provided history length
            # so we don't re-read the whole message list just to count it.
            is_first_turn = (
                history_len == 0 if history_len is not None
                else len(sb.get_messages(conversation_id)) <= 2
            )
            if is_first_turn:
                sb.auto_title_conversation(conversation_id, question)
        except Exception as e:
            logger.warning("Failed to save streamed messages: %s", e)

    # B4: analytics (query_logs + audit) are non-critical and would otherwise hold
    # the streaming connection/thread open after [DONE]. Fire-and-forget them.
    def _write_analytics():
        try:
            sb.client.table("query_logs").insert({
                "user_id": sb.user_id,
                "conversation_id": conversation_id,
                "question": question[:2000],
                "answer_length": len(full_answer),
                "sources_count": len(collected_sources),
                "retrieval_docs_count": retrieval_docs_count,
                "latency_ms": latency_ms,
                "cache_hit": is_cache_hit,
                "agentic": is_agentic,
                "web_search_used": web_search_used,
            }).execute()
        except Exception as e:
            logger.warning("Failed to log query analytics: %s", e)

        action = "query.agentic" if is_agentic else "query.ask"
        log_audit(sb, action, "conversation", conversation_id, {
            "question": question[:500],
            "latency_ms": latency_ms,
            "cache_hit": is_cache_hit,
            "web_search_used": web_search_used,
        })

    threading.Thread(target=_write_analytics, daemon=True).start()

# ── Module-level caches ───────────────────────────────────────────────────────
# SemanticCache: pool one instance per user — avoids opening a new Redis TCP
# connection + ping on every single request (~5ms saved per request).
_cache_pool: dict[str, object] = {}

# OpenAIEmbeddings: cache per (model, api_key) pair — the client object is
# lightweight but recreating it on every request is unnecessary.
_embedder_cache: dict[str, object] = {}


def _get_cache(user_id: str):
    """Return a cached SemanticCache scoped to this user. Non-fatal if Redis is down."""
    if user_id not in _cache_pool:
        from src.components.semantic_cache import SemanticCache
        _cache_pool[user_id] = SemanticCache(redis_url=_REDIS_URL, namespace=user_id)
    return _cache_pool[user_id]


async def _embed_query(query: str, model: str, api_key: str) -> list:
    """Embed a query string. Caches the embedder object; runs in thread pool."""
    from langchain_openai import OpenAIEmbeddings
    cache_key = f"{model}:{api_key[:12]}"
    if cache_key not in _embedder_cache:
        _embedder_cache[cache_key] = OpenAIEmbeddings(model=model, api_key=api_key)
    embedder = _embedder_cache[cache_key]
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
        cached = await asyncio.to_thread(cache.get, body.question, query_embedding)
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

    # ── Smart query routing based on complexity ──
    # classify_query_complexity() uses pure regex rules — no LLM call.
    # simple   → direct vector retrieval using pre-computed embedding (~1.0s total)
    # moderate → multi-query variants, skip self-review (~1.5-2.0s total)
    # complex  → full pipeline (handled by /query/agent endpoint)
    # Phase 1: Collection-scoped retrieval
    filename_filters = _resolve_collection_filters(sb, getattr(body, "collection_id", None), query=getattr(body, "question", None), query_embedding=query_embedding, user_config=user_config)

    complexity = generator.classify_query_complexity(body.question)
    logger.info("Query complexity: %s for: %.60s", complexity, body.question)

    if complexity == "simple" and query_embedding:
        # Fast path: reuse the pre-computed cache embedding — skip the second API call
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_by_vector,
            query_embedding,
            body.question,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
        )
    elif user_config.USE_MULTI_QUERY:
        variants = await asyncio.to_thread(
            generator.generate_query_variants, body.question, user_config.MULTI_QUERY_COUNT
        )
        variants = [body.question] + variants
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
            # B5: variants[0] == body.question, so the cache embedding fits — reuse it.
            primary_embedding=query_embedding or None,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            body.question,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
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
        generator.generate, query=body.question, retrieved_docs=docs,
        user_name=sb.preferred_name,
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
        cached = await asyncio.to_thread(cache.get, body.question, query_embedding)
    except Exception:
        pass
    cache_latency_s = time.perf_counter() - t_cache
    cache_latency.observe(cache_latency_s)

    if cached:
        cache_hits.labels(tier=cached.get("tier", "unknown")).inc()
        def cached_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', [])})}\n\n"
            # Stream the cached answer token by token (words) for natural UX
            words = cached["answer"].split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cache_hit': True, 'similarity': cached.get('similarity', 1.0)})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _saving_stream_wrapper(
                cached_stream(), sb, body.conversation_id, body.question,
                is_cache_hit=True, history_len=len(chat_history),
            ),
            media_type="text/event-stream"
        )

    cache_misses.inc()

    # ── B2: run the rewrite and variant LLM calls concurrently ──
    # rewrite_query no-ops (no LLM) when there's no chat history. We route on the
    # RAW question (regex, no LLM) so we can decide up front whether variants are
    # needed and fire them in parallel with the rewrite instead of after it.
    # Routing is unchanged from the prior rewritten-query classification; the only
    # trade-offs are: complexity is judged on the raw question, and variants are
    # generated from the raw question (the rewritten query still goes in as primary).
    t_prep = time.perf_counter()
    complexity = generator.classify_query_complexity(body.question)
    logger.info("Query stream complexity: %s for: %.60s", complexity, body.question)

    rewrite_task = asyncio.create_task(
        asyncio.to_thread(generator.rewrite_query, body.question, chat_history)
    )
    want_variants = user_config.USE_MULTI_QUERY and not (
        complexity == "simple" and query_embedding
    )
    variants_task = (
        asyncio.create_task(
            asyncio.to_thread(
                generator.generate_query_variants, body.question, user_config.MULTI_QUERY_COUNT
            )
        )
        if want_variants
        else None
    )

    search_query = await rewrite_task
    t_llm_prep = time.perf_counter() - t_prep

    # Phase 1: Collection-scoped retrieval
    filename_filters = _resolve_collection_filters(sb, getattr(body, "collection_id", None), query=getattr(body, "question", None), query_embedding=query_embedding, user_config=user_config)

    t_retrieve = time.perf_counter()
    if complexity == "simple" and query_embedding:
        # Fast path: reuse the pre-computed cache embedding — skip the second API call
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_by_vector,
            query_embedding,
            search_query,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
        )
    elif variants_task is not None:
        raw_variants = await variants_task
        variants = [search_query] + raw_variants
        # B5: query_embedding is for body.question; it only matches variants[0]
        # (search_query) when the rewrite was a no-op. Reuse it only then.
        primary_emb = query_embedding if (query_embedding and search_query == body.question) else None
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
            primary_embedding=primary_emb,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            search_query,
            body.filename_filter,
            body.page_filter,
            filename_filters=filename_filters,
        )
    t_retrieve = time.perf_counter() - t_retrieve

    # ── Per-stage timing (instrumentation) ──
    logger.info(
        "query_stream stages: embed+cache=%dms llm_prep=%dms retrieve=%dms docs=%d complexity=%s",
        int(cache_latency_s * 1000), int(t_llm_prep * 1000), int(t_retrieve * 1000),
        len(docs), complexity,
    )

    # ── Prometheus metrics ──
    queries_total.labels(endpoint="query_stream", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

    # ── Web search fallback (standard path) ──
    web_search_used = False
    if not docs and user_config.USE_WEB_FALLBACK:
        try:
            from src.components.web_search import WebSearcher
            searcher = WebSearcher(user_config)
            if searcher.is_available():
                web_docs = searcher.search(search_query, max_results=user_config.WEB_SEARCH_MAX_RESULTS)
                if web_docs:
                    docs = web_docs
                    web_search_used = True
        except Exception as _e:
            logger.warning("Web search fallback failed (non-fatal): %s", _e)

    if not docs:
        def no_results():
            yield f"data: {json.dumps({'type': 'token', 'content': 'I could not find relevant information in your documents for this question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _saving_stream_wrapper(
                no_results(), sb, body.conversation_id, body.question,
                history_len=len(chat_history),
            ),
            media_type="text/event-stream"
        )

    def standard_stream():
        if web_search_used:
            web_count = sum(1 for d in docs if d.metadata.get("source") == "web")
            yield f"data: {json.dumps({'type': 'web_search', 'results_count': web_count})}\n\n"
        yield from generator.generate_stream(
            query=search_query,
            retrieved_docs=docs,
            chat_history=chat_history,
            user_name=sb.preferred_name,
        )

    return StreamingResponse(
        _saving_stream_wrapper(
            standard_stream(),
            sb, body.conversation_id, body.question,
            retrieval_docs_count=len(docs),
            web_search_used=web_search_used,
            history_len=len(chat_history),
        ),
        media_type="text/event-stream"
    )


# -----------------------------------------
# AGENTIC QUERY (Phase 6)
# -----------------------------------------

def _make_agentic_retriever(user_config, retrieval_mgr, multi_hop_override=None):
    """Pick the agentic retriever for this request.

    Both retrievers expose the same ``retrieve_and_synthesize`` contract, so the agent
    endpoints are agnostic. Selection: ``multi_hop_override`` (the per-request
    ``body.multi_hop`` flag) wins when not None; otherwise the server default
    ``Config.USE_MULTIHOP``. True = the sequential ReAct/Self-RAG loop (Phase 4.5);
    False = the proven parallel-decompose path. Per-request override lets the UI A/B
    the two on the same question without an env change or restart.
    """
    use_multihop = (
        multi_hop_override if multi_hop_override is not None
        else getattr(user_config, "USE_MULTIHOP", False)
    )
    if use_multihop:
        from src.components.multi_hop import MultiHopRetriever
        return MultiHopRetriever(config=user_config, retrieval_mgr=retrieval_mgr)
    from src.components.agentic_retrieval import AgenticRetriever
    return AgenticRetriever(config=user_config, retrieval_mgr=retrieval_mgr)


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
      1. Semantic cache check (repeated complex queries served in <50ms).
      2. Decompose the query into 2-4 atomic sub-questions.
      3. Retrieve relevant docs for each sub-question IN PARALLEL.
      4. Deduplicate and merge retrieved docs.
      5. Generate with self-critique loop (flags unsupported claims, revises if needed).

    Rate limit: 10/min (2x LLM calls per request vs standard 1x).
    """
    # ── Cache check (agent queries can be expensive — cache hits are very valuable) ──
    t_cache = time.perf_counter()
    cached = None
    query_embedding = []
    try:
        query_embedding = await _embed_query(
            body.question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
        cache = _get_cache(sb.user_id)
        cached = await asyncio.to_thread(cache.get, body.question, query_embedding)
    except Exception:
        pass
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

    # Phase 1: Collection-scoped retrieval
    filename_filters = _resolve_collection_filters(sb, getattr(body, "collection_id", None), query=getattr(body, "question", None), query_embedding=query_embedding, user_config=user_config)

    # ── Step 1+2+3: Agentic retrieval (parallel sub-queries, or multi-hop loop) ──
    agentic = _make_agentic_retriever(user_config, retrieval_mgr, getattr(body, "multi_hop", None))
    retrieval_result = await asyncio.to_thread(
        agentic.retrieve_and_synthesize,
        body.question,
        body.filename_filter,
        body.page_filter,
        filename_filters=filename_filters,
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

    # ── Step 4: Self-critique generation ──
    result = await asyncio.to_thread(
        generator.generate_with_self_review,
        query=body.question,
        retrieved_docs=docs,
        user_id=sb.user_id,
        user_name=sb.preferred_name,
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

    # ── Store in cache for future identical/similar agent queries ──
    if query_embedding and result.get("answer"):
        try:
            cache.set(
                body.question,
                query_embedding,
                result["answer"],
                [s.model_dump() for s in sources],
            )
        except Exception:
            pass

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        num_sources_used=result["num_sources_used"],
    )


@router.post("/query/agent/stream")
@limiter.limit("10/minute")
async def agent_query_stream(
    request: Request,
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """
    Agentic query with SSE streaming (Phase 6).

    Same decompose → parallel retrieve → self-review pipeline as /query/agent,
    but streams the answer token-by-token via SSE for better UX.
    Emits: sub_queries → sources → token* → done
    """
    chat_history = []
    if body.conversation_id:
        existing_msgs = sb.get_messages(body.conversation_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in existing_msgs
        ]

    query_embedding: list = []
    # Phase 1: Collection-scoped retrieval
    filename_filters = _resolve_collection_filters(sb, getattr(body, "collection_id", None), query=getattr(body, "question", None), query_embedding=query_embedding, user_config=user_config)

    # Agentic retrieval: decompose → parallel retrieve → dedup, or sequential multi-hop
    agentic = _make_agentic_retriever(user_config, retrieval_mgr, getattr(body, "multi_hop", None))
    retrieval_result = await asyncio.to_thread(
        agentic.retrieve_and_synthesize,
        body.question,
        body.filename_filter,
        body.page_filter,
        filename_filters=filename_filters,
    )
    docs = retrieval_result["docs"]
    sub_queries = retrieval_result.get("sub_queries", [])
    web_search_used = retrieval_result.get("web_search_used", False)

    queries_total.labels(endpoint="query_agent_stream", has_results=str(len(docs) > 0)).inc()
    retrieval_docs.observe(len(docs))

    if not docs:
        def no_results():
            yield f"data: {json.dumps({'type': 'sub_queries', 'queries': sub_queries})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'I could not find relevant information in your documents for this question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _saving_stream_wrapper(
                no_results(), sb, body.conversation_id, body.question,
                history_len=len(chat_history),
            ),
            media_type="text/event-stream"
        )

    def agentic_stream():
        # 1. Emit sub-queries so the UI can show decomposition
        yield f"data: {json.dumps({'type': 'sub_queries', 'queries': sub_queries})}\n\n"

        # 2. Emit web_search event if fallback was used
        if web_search_used:
            web_count = sum(1 for d in docs if d.metadata.get("source") == "web")
            yield f"data: {json.dumps({'type': 'web_search', 'results_count': web_count})}\n\n"

        # 3. Emit sources
        source_infos = []
        for i, doc in enumerate(docs):
            source_infos.append({
                "source_id": i + 1,
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page_number"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content[:300] if doc.page_content else None,
            })
        yield f"data: {json.dumps({'type': 'sources', 'sources': source_infos})}\n\n"

        # 4. Stream answer via generate_stream (same as /query/stream)
        for chunk in generator.generate_stream(
            query=body.question,
            retrieved_docs=docs,
            chat_history=chat_history,
            user_name=sb.preferred_name,
        ):
            yield chunk

    return StreamingResponse(
        _saving_stream_wrapper(
            agentic_stream(), sb, body.conversation_id, body.question,
            is_agentic=True, retrieval_docs_count=len(docs), web_search_used=web_search_used,
            history_len=len(chat_history),
        ),
        media_type="text/event-stream"
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
    log_audit(sb, "conversation.create", "conversation", conv["id"], {"title": conv["title"]})
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
        log_audit(sb, "conversation.delete", "conversation", conversation_id, {})
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    body: dict,
    sb=Depends(get_current_user),
):
    """Rename a conversation."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    result = (
        sb.client
        .table("conversations")
        .update({"title": title})
        .eq("id", conversation_id)
        .eq("user_id", sb.user_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {"id": conversation_id, "title": title}


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

    query_embedding: list = []  # not pre-computed in this path; routing uses text query

    page_filter = body.page_filter
    if not page_filter:
        page_match = re.search(r'page\s+(\d+)', body.question.lower())
        if page_match:
            page_filter = int(page_match.group(1))

    # B2: rewrite (standalone context) and variant generation are independent —
    # run them concurrently. Variants come from the raw question; the rewritten
    # query still goes in as the primary variant.
    rewrite_task = asyncio.create_task(
        asyncio.to_thread(generator.rewrite_query, body.question, chat_history)
    )
    variants_task = (
        asyncio.create_task(
            asyncio.to_thread(
                generator.generate_query_variants, body.question, user_config.MULTI_QUERY_COUNT
            )
        )
        if user_config.USE_MULTI_QUERY
        else None
    )
    search_query = await rewrite_task

    # Phase 1: Collection-scoped retrieval
    filename_filters = _resolve_collection_filters(sb, getattr(body, "collection_id", None), query=getattr(body, "question", None), query_embedding=query_embedding, user_config=user_config)

    if variants_task is not None:
        variants = [search_query] + await variants_task
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve_multi_query,
            variants,
            body.filename_filter,
            page_filter,
            filename_filters=filename_filters,
        )
    else:
        docs = await asyncio.to_thread(
            retrieval_mgr.retrieve,
            search_query,
            body.filename_filter,
            page_filter,
            filename_filters=filename_filters,
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


# ── Brain endpoint (Phase 4) ─────────────────────────────────────────────────

@router.post("/query/brain/stream")
@limiter.limit("5/minute")
async def brain_query_stream(
    request: Request,
    body: QueryRequest,
    sb=Depends(get_current_user),
    user_config: Config = Depends(get_user_config),
    retrieval_mgr=Depends(get_retrieval_mgr),
    generator=Depends(get_generator),
):
    """Stage-2 Brain: map-reduce synthesis over a collection with SSE step streaming.

    Requires body.collection_id.  Returns brain_start → brain_map (per doc) →
    brain_verify → brain_reduce → sources → token* → brain_meta → [DONE] events.

    This endpoint is the 'synthesis / multi-doc' path.  It:
      1. Uses Stage-1 router to pick top-N docs.
      2. Runs MAP in parallel (one LLM call per doc, cheap model).
      3. Runs VERIFY (independent model) claim-by-claim — before synthesis.
      4. Runs REDUCE (one call, strong model) over the verified claims.
      5. Streams all events back to the client.

    Non-regression: single-doc / simple queries should use /query/stream instead.
    """
    from src.components.brain.map_reduce import Brain

    # Opt-in gate (Phase 4): the Brain path is off by default.  Set USE_BRAIN=true.
    if not getattr(user_config, "USE_BRAIN", False):
        raise HTTPException(
            status_code=403,
            detail="Brain path is disabled. Set USE_BRAIN=true to enable /query/brain/stream.",
        )

    collection_id = getattr(body, "collection_id", None)
    if not collection_id:
        raise HTTPException(
            status_code=400,
            detail="collection_id is required for the Brain endpoint.",
        )

    query_embedding: list = []
    try:
        query_embedding = await _embed_query(
            body.question, user_config.EMBEDDING_MODEL_NAME, user_config.OPENAI_API_KEY
        )
    except Exception:
        pass

    filename_filters = _resolve_collection_filters(
        sb, collection_id,
        query=body.question,
        query_embedding=query_embedding,
        user_config=user_config,
    )
    if not filename_filters:
        def _empty():
            import json as _j
            yield f"data: {_j.dumps({'type': 'token', 'content': 'No documents found in this collection.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Retrieve each routed doc's chunks in parallel (bounded by the routed top-N).
    # Read many chunks per doc (BRAIN_CHUNKS_PER_DOC) so MAP can find needles in big
    # filings, and bound each retrieval with a timeout so a network/DNS blip fails fast
    # instead of hanging on Pinecone retries for minutes.
    per_doc_k = getattr(user_config, "BRAIN_CHUNKS_PER_DOC", 8)
    retrieve_timeout = getattr(user_config, "BRAIN_RETRIEVE_TIMEOUT_S", 20)
    _retrieve_sem = asyncio.Semaphore(3)  # bound parallel Pinecone to 3

    async def _retrieve_one(fname: str):
        async with _retrieve_sem:
            try:
                chunks = await asyncio.wait_for(
                    asyncio.to_thread(
                        retrieval_mgr.retrieve,
                        body.question,
                        fname,            # filename_filter = single file
                        body.page_filter,
                        None,             # filename_filters
                        per_doc_k,        # top_k
                        False,            # apply_threshold=False — the Brain's
                                          # MAP+VERIFY handles precision; the 0.30
                                          # similarity threshold drops valid chunks
                                          # for cross-doc synthesis queries.
                        False,            # use_reranker=False — the local CPU
                                          # CrossEncoder (~18s/batch under concurrent
                                          # load) caused per-doc retrieval TIMEOUTS
                                          # that silently dropped whole docs (e.g. AWS
                                          # net-sales never reaching MAP). MAP+VERIFY
                                          # already handle precision.
                    ),
                    timeout=retrieve_timeout,
                )
                return fname, chunks
            except asyncio.TimeoutError:
                logger.warning("[brain] retrieval timed out for %s after %ss", fname, retrieve_timeout)
                return fname, []
            except Exception as exc:
                logger.warning("[brain] retrieval failed for %s: %s", fname, exc)
                return fname, []

    retrieved = await asyncio.gather(*[_retrieve_one(fn) for fn in filename_filters])

    doc_chunks: dict[str, tuple[str, list]] = {}
    for fname, chunks in retrieved:
        if chunks:
            doc_id = chunks[0].metadata.get("doc_id", fname)
            doc_chunks[doc_id] = (fname, chunks)

    if not doc_chunks:
        def _no_chunks():
            import json as _j
            yield f"data: {_j.dumps({'type': 'token', 'content': 'No relevant content found in the collection for this question.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_chunks(), media_type="text/event-stream")

    brain = Brain(user_config)

    # ── Phase 4.3: deterministic Analyst (§4b) — intent-gated so non-numeric
    # questions pay ZERO added latency and the path is identical to before.
    # Wrapped in asyncio.to_thread so the blocking Supabase queries + LLM call
    # don't freeze the event loop (the Analyst is additive / non-critical).
    analyst_block = None
    analyst_count = 0

    def _run_analyst_sync():
        """Run the Analyst pipeline off the event loop (sync, in a worker thread)."""
        _block = None
        _count = 0
        try:
            from src.components.brain.table_intent import has_numeric_intent, load_grids_for_docs
            if not has_numeric_intent(body.question):
                return _block, _count
            from src.components.brain.analyst import analyze, render_markdown
            filename_by_doc = {did: fn for did, (fn, _ch) in doc_chunks.items()}
            grids = load_grids_for_docs(
                sb, list(doc_chunks.keys()),
                question=body.question, filename_by_doc=filename_by_doc,
            )
            if not grids:
                return _block, _count
            from langchain_openai import ChatOpenAI
            spec_llm = ChatOpenAI(
                model=user_config.LLM_MODEL_NAME, temperature=0.0,
                api_key=user_config.OPENAI_API_KEY, request_timeout=30,
            )

            # ── Phase 4.6 executive spine (§5.6), opt-in behind USE_EXEC_SPINE ──
            # For a pivot/bridge/compare question the coordinator binds the pivot and
            # self-monitors the read (WRONG→ABSTAIN). Its block leads the Analyst output
            # (higher authority for bridges); a monitor abstain surfaces a clean refusal.
            # When the flag is off OR the question isn't a spine shape, this is a no-op and
            # the Analyst path below is byte-identical to today.
            _spine_block = None
            _spine_abstain = None
            if getattr(user_config, "USE_EXEC_SPINE", False):
                try:
                    from src.components.brain.meta_reasoner import run_executive_spine
                    # comprehend is the spine's single LLM call and its measured live
                    # bottleneck — run it on the strong tier (one call per pivot/bridge
                    # question; the Analyst spec-writer below stays on the cheap model).
                    comprehend_llm = ChatOpenAI(
                        model=getattr(user_config, "COMPREHEND_LLM_MODEL", None)
                              or user_config.LLM_MODEL_NAME,
                        temperature=0.0, api_key=user_config.OPENAI_API_KEY,
                        request_timeout=30,
                    )
                    outcome = run_executive_spine(body.question, grids, comprehend_llm)
                    if outcome.applied and outcome.block:
                        _spine_block = outcome.block
                        logger.info("[brain] executive spine applied (pivot=%s)", outcome.binding)
                    elif outcome.abstained:
                        # C4 enforcement: the spine RESOLVED a figure but its self-monitor
                        # flagged the reasoning → carry the reason so REDUCE binary-WITHHOLDS
                        # instead of letting the old path ship the wrong number (§5.5/§4a).
                        _spine_abstain = outcome.reason or "self-monitor flagged the reasoning"
                        logger.info("[brain] executive spine WITHHELD: %s", _spine_abstain)
                except Exception as exc:
                    logger.warning("[brain] executive spine skipped: %s", exc)

            results = analyze(body.question, grids, spec_llm)
            from src.components.brain.analyst import corroborate_with_prose
            prose = " ".join(
                getattr(ch, "page_content", "")
                for _fn, chunks in doc_chunks.values() for ch in chunks
            )
            results = corroborate_with_prose(results, prose)
            ok = [r for r in results if r.ok]
            if ok:
                _block = render_markdown(ok)
                _count = len(ok)
                logger.info("[brain] Analyst computed %d figures for the question", _count)
            # the spine block LEADS (it bound the pivot + self-monitored the bridge); the
            # Analyst table, if any, follows as corroborating detail. Either alone is fine.
            if _spine_block:
                _block = _spine_block + ("\n\n" + _block if _block else "")
                _count = max(_count, 1)
        except Exception as exc:
            logger.warning("[brain] Analyst step skipped: %s", exc)
        return _block, _count, _spine_abstain

    try:
        analyst_block, analyst_count, spine_abstain = await asyncio.to_thread(_run_analyst_sync)
    except Exception as exc:
        logger.warning("[brain] Analyst thread failed: %s", exc)
        analyst_block, analyst_count, spine_abstain = None, 0, None

    # Pass the sync generator straight to StreamingResponse — Starlette iterates it
    # in a worker thread, so the blocking MAP/REDUCE/VERIFY work stays off the event
    # loop (matches /query/stream and the other streaming endpoints in this file).
    # Wrapped in _saving_stream_wrapper so the Q&A persists to the conversation
    # (collects token/sources from the stream; brain_* events pass through untouched).
    conversation_id = getattr(body, "conversation_id", None)
    return StreamingResponse(
        _saving_stream_wrapper(
            brain.run_stream(
                body.question,
                doc_chunks,
                user_id=sb.user_id,
                collection_id=collection_id,
                conversation_id=conversation_id,
                analyst_block=analyst_block,
                analyst_count=analyst_count,
                spine_abstain=spine_abstain,
            ),
            sb,
            conversation_id,
            body.question,
            retrieval_docs_count=len(doc_chunks),
        ),
        media_type="text/event-stream",
    )
