"""
DocQuery — Custom Prometheus Metrics

RAG-specific counters and histograms for observability.
Auto-HTTP metrics (latency, status codes, in-flight) are handled by
prometheus-fastapi-instrumentator in server.py.
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

# ── Reranker metrics (Feature 2) ──
rerank_latency = Histogram(
    "docquery_rerank_latency_seconds",
    "Cross-encoder reranker processing time",
)
