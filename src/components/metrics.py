"""
DocQuery — Custom Prometheus Metrics

RAG-specific counters and histograms for observability.
Auto-HTTP metrics (latency, status codes, in-flight) are handled by
prometheus-fastapi-instrumentator in server.py.

Phase 2 additions: semantic cache hit/miss counters + latency.
Phase 4 additions: per-user LLM cost tracking (token counts by user_id).
"""

from prometheus_client import Counter, Histogram

# ── Query metrics ──
queries_total = Counter(
    "docquery_queries_total",
    "Total queries processed",
    ["endpoint", "has_results"],
)

retrieval_docs = Histogram(
    "docquery_retrieval_docs",
    "Number of documents retrieved per query",
    buckets=[0, 1, 2, 3, 5, 10, 15],
)

# ── LLM metrics ──
llm_tokens = Counter(
    "docquery_llm_tokens_total",
    "Approximate LLM token usage",
    ["model"],
)

# ── Upload metrics ──
uploads_total = Counter(
    "docquery_uploads_total",
    "File uploads by outcome",
    ["status"],
)

# ── Reranker metrics ──
rerank_latency = Histogram(
    "docquery_rerank_latency_seconds",
    "Cross-encoder reranker processing time",
)

# ── Phase 2: Semantic cache metrics ──
# Track cache hit rate per tier (exact vs semantic) and lookup latency.
# cache_hit_rate = cache_hits_total / (cache_hits_total + cache_misses_total)
cache_hits = Counter(
    "docquery_cache_hits_total",
    "Semantic cache hits",
    ["tier"],   # 'exact' or 'semantic'
)

cache_misses = Counter(
    "docquery_cache_misses_total",
    "Semantic cache misses",
)

cache_latency = Histogram(
    "docquery_cache_latency_seconds",
    "Time to check semantic cache (hit + miss)",
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250],
)

# ── Phase 4: Per-user LLM cost tracking ──
# Tracks approximate input token counts per user/model/operation.
# Surfaces in Grafana: which users are burning your OpenAI budget.
# operation labels: 'embed', 'generate', 'rewrite_query', 'self_review', 'decompose'
user_llm_cost = Counter(
    "docquery_user_llm_cost_tokens",
    "Approximate input token usage per user (len(text)//4 heuristic)",
    ["user_id", "model", "operation"],
)
