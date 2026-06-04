"""
Stage-1 Document Router — Phase 3.

Implements the two halves of document routing:

  INGEST half  (called from tasks.py after chunking):
    compute_doc_summary_and_embedding(chunks, doc_id, ...) →
        stores a one-sentence extractive summary + the chunk-centroid
        topic embedding in the `document_summaries` Supabase table.

  QUERY half  (called from retrieval path or chat.py when collection scope):
    route_collection(query, collection_id, ...) →
        returns the top-N most relevant doc_ids / filenames for this query,
        capping per-file fan-out at ROUTING_TOP_N (default 12) so
        retrieve_across_files never sees the full 200-doc collection.

This is what fixes MISTAKE 1 from the Brain plan:
  Before: retrieve_across_files fans out to ALL files in the collection.
  After:  it only fans out to the ≤N docs the router selected.

The router uses cosine similarity on the stored topic_embeddings (one per doc),
which is a fast in-process numpy operation — no Pinecone query, no LLM call.
Optional BM25-over-summaries fallback when embeddings aren't yet computed.
"""

from __future__ import annotations

import json
import time
import logging
import traceback
from typing import Optional, List

import numpy as np

from src.components.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


# ── Cosine similarity helpers ─────────────────────────────────────────────────

def _cosine(a: list | np.ndarray, b: list | np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _centroid(vectors: list[list]) -> list:
    """Mean of a list of embedding vectors (topic/centroid embedding)."""
    arr = np.array(vectors, dtype=np.float32)
    return arr.mean(axis=0).tolist()


def _mmr_select(
    candidates: list[tuple[float, str, Optional[list]]],
    top_n: int,
    lambda_relevance: float = 0.7,
) -> list[str]:
    """Maximal Marginal Relevance selection — diversity-aware document routing.

    Plain top-N cosine over-concentrates: near-duplicate docs (e.g. three years
    of the *same* company's 10-K) cluster tightly and crowd out the docs a
    cross-entity question actually needs (Entry 7 finding — the cross-company
    comparison kept the 3 Microsoft docs and dropped Google/Amazon FY23).

    MMR fixes that by picking docs one at a time, each maximizing:
        λ · relevance(doc, query)  −  (1−λ) · max similarity(doc, already_picked)
    so a candidate that's highly relevant *but* near-identical to something already
    chosen is penalised — naturally spreading selection across distinct documents
    (and therefore across companies/topics) without hard-coding "company".

    Args:
        candidates:        list of (relevance_score, filename, topic_vec | None).
                           topic_vec is the doc's embedding (for redundancy calc);
                           None falls back to pure relevance for that doc.
        top_n:             how many to select.
        lambda_relevance:  1.0 = pure relevance (old behaviour); lower = more diverse.

    Returns:
        filenames, ordered by selection.
    """
    if top_n <= 0 or not candidates:
        return []
    # Sort by relevance once; the first pick is always the most relevant doc.
    pool = sorted(candidates, key=lambda c: c[0], reverse=True)
    if len(pool) <= top_n:
        return [fn for _, fn, _ in pool]

    selected: list[tuple[float, str, Optional[list]]] = [pool.pop(0)]
    while pool and len(selected) < top_n:
        best_idx, best_mmr = 0, -1e9
        for i, (rel, _fn, vec) in enumerate(pool):
            if vec is None:
                redundancy = 0.0  # no embedding → can't measure overlap, don't penalise
            else:
                redundancy = max(
                    (_cosine(vec, s_vec) for _, _, s_vec in selected if s_vec is not None),
                    default=0.0,
                )
            mmr = lambda_relevance * rel - (1.0 - lambda_relevance) * redundancy
            if mmr > best_mmr:
                best_mmr, best_idx = mmr, i
        selected.append(pool.pop(best_idx))
    return [fn for _, fn, _ in selected]


# ── Extractive summary ────────────────────────────────────────────────────────

def _extractive_summary(chunks, max_chars: int = 400) -> str:
    """Return a short extractive summary from the first few chunks of a doc.

    Cheap — no LLM call.  Good enough for routing (the embedding does the work).
    """
    texts = []
    total = 0
    for chunk in chunks:
        text = getattr(chunk, "page_content", "") or ""
        text = text.strip()
        if not text:
            continue
        texts.append(text[:max_chars - total])
        total += len(text)
        if total >= max_chars:
            break
    return " ".join(texts)[:max_chars]


# ── Ingest half ───────────────────────────────────────────────────────────────

def compute_and_store_doc_routing_data(
    chunks: list,
    doc_id: str,
    user_id: str,
    collection_id: Optional[str],
    config: Config,
    db_client=None,
) -> bool:
    """Compute topic embedding + summary for a document and persist to document_summaries.

    Args:
        chunks:        LangChain Documents produced during ingest.
        doc_id:        Supabase document UUID.
        user_id:       Owner user UUID.
        collection_id: Supabase collection UUID (may be None for uncollected docs).
        config:        Config instance (holds embedding model / keys).
        db_client:     Optional SupabaseManager (created internally if None).

    Returns:
        True on success, False on non-fatal failure.
    """
    if not chunks:
        logger.warning("[doc_router] No chunks for doc %s — skipping summary", doc_id)
        return False

    try:
        # 1. Extractive summary (cheap, no LLM)
        summary = _extractive_summary(chunks)

        # 2. Embed each chunk text, then compute centroid
        from langchain_openai import OpenAIEmbeddings
        embed_model = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY,
        )
        texts = [c.page_content for c in chunks if c.page_content.strip()]
        if not texts:
            return False

        # Embed in one batch (chunks already embedded for Pinecone, but we need
        # individual vectors here; re-embedding is cheap relative to ingest total)
        embeddings = embed_model.embed_documents(texts[:50])  # cap at 50 chunks for cost
        topic_embedding = _centroid(embeddings)

        # 3. Persist to document_summaries
        if db_client is None:
            from src.components.db import SupabaseManager
            db_client = SupabaseManager(use_service_role=True)
            db_client._user = type("User", (), {"id": user_id})()

        row = {
            "document_id": doc_id,
            "user_id": user_id,
            "collection_id": collection_id,
            "summary": summary,
            "topic_embedding": json.dumps(topic_embedding),  # pgvector expects JSON array
            "model_used": config.EMBEDDING_MODEL_NAME,
        }
        db_client.client.table("document_summaries").upsert(
            row, on_conflict="document_id"
        ).execute()

        logger.info(
            "[doc_router] Stored summary+embedding for doc %s (%d chunks → centroid dim=%d)",
            doc_id, len(texts), len(topic_embedding),
        )
        return True

    except Exception as exc:
        logger.warning(
            "[doc_router] compute_and_store_doc_routing_data failed for doc %s (non-fatal): %s",
            doc_id, exc,
        )
        logger.debug(traceback.format_exc())
        return False


# ── Query half ────────────────────────────────────────────────────────────────

class DocumentRouter:
    """Stage-1 router: given a query + collection, return the top-N most relevant docs.

    Usage:
        router = DocumentRouter(config)
        top_filenames = router.route(query, collection_id, user_id, top_n=12)
        # now pass top_filenames to retrieve_across_files (capped fan-out)

    Falls back to returning all filenames (unordered) if no summaries/embeddings
    are stored yet — degrading gracefully to the pre-router behaviour.
    """

    def __init__(self, config: Config):
        self.config = config

    def route(
        self,
        query: str,
        collection_id: str,
        user_id: str,
        top_n: Optional[int] = None,
        query_embedding: Optional[list] = None,
    ) -> list[str]:
        """Return filenames of the top-N most relevant documents for this query.

        Args:
            query:            User query string.
            collection_id:    Supabase collection UUID.
            user_id:          Owner user UUID (for RLS scoping).
            top_n:            How many docs to return (default: config.ROUTING_TOP_N).
            query_embedding:  Pre-computed query embedding (skips an OpenAI call).

        Returns:
            List of filenames (strings), ordered most-relevant first.
            Returns [] if the collection is empty.
            Falls back to unranked filenames if no routing data is available.
        """
        top_n = top_n or self.config.ROUTING_TOP_N
        t0 = time.perf_counter()

        try:
            from src.components.db import SupabaseManager
            db = SupabaseManager(use_service_role=True)
            db._user = type("User", (), {"id": user_id})()

            # 1. Resolve collection membership.  Documents join collections through
            #    the many-to-many `collection_documents` table (a doc can belong to
            #    several collections), so `collection_id` is NOT denormalised onto
            #    document_summaries — we resolve the member doc_ids here (ownership
            #    checked inside get_collection_documents) and join to summaries on
            #    document_id.  This is the fix for the router being dead-on-arrival:
            #    filtering document_summaries by collection_id matched zero rows
            #    because that column is never populated at ingest.
            coll_docs = db.get_collection_documents(collection_id)
            id_to_filename = {
                d["id"]: d.get("filename", "")
                for d in (coll_docs or [])
                if d.get("id") and d.get("filename")
            }
            if not id_to_filename:
                return []
            doc_ids = list(id_to_filename.keys())

            # 2. Fetch summaries + topic embeddings for exactly those docs.
            rows = db.client.table("document_summaries").select(
                "document_id, summary, topic_embedding"
            ).in_("document_id", doc_ids).eq("user_id", user_id).execute()
            summaries = {r["document_id"]: r for r in (rows.data or [])}

            # 3. Embed the query once — only needed if any doc has a topic embedding.
            #    Treat an empty list the same as None (some callers pass []), so we
            #    re-embed instead of silently degrading to keyword-only scoring.
            have_embeddings = any(r.get("topic_embedding") for r in summaries.values())
            if have_embeddings and not query_embedding:
                try:
                    from langchain_openai import OpenAIEmbeddings
                    embed_model = OpenAIEmbeddings(
                        model=self.config.EMBEDDING_MODEL_NAME,
                        openai_api_key=self.config.OPENAI_API_KEY,
                    )
                    query_embedding = embed_model.embed_query(query)
                except Exception as exc:
                    logger.warning(
                        "[doc_router] query embed failed, keyword fallback: %s", exc
                    )
                    query_embedding = None

            # 4. Score EVERY collection doc so a doc without routing data is still a
            #    candidate (recall-first): cosine on the topic embedding when present,
            #    else keyword overlap on the summary, else on the filename. Keep each
            #    doc's topic vector so MMR (step 5) can measure redundancy.
            scored: list[tuple[float, str, Optional[list]]] = []
            for did, filename in id_to_filename.items():
                row = summaries.get(did)
                emb_raw = row.get("topic_embedding") if row else None
                topic_vec: Optional[list] = None
                if emb_raw and query_embedding:
                    try:
                        topic_vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                        score = _cosine(query_embedding, topic_vec)
                    except Exception:
                        topic_vec = None
                        score = _keyword_score(query, (row or {}).get("summary", "") or filename)
                elif row:
                    score = _keyword_score(query, row.get("summary", "") or filename)
                else:
                    score = _keyword_score(query, filename)
                scored.append((score, filename, topic_vec))

            # 5. Diversity-aware selection (MMR). When more docs than top_n,
            #    penalise candidates near-identical to ones already picked so a
            #    cross-entity query gets a spread of distinct docs instead of N
            #    near-duplicates from one cluster (Entry 7 fix). λ=1 reproduces the
            #    old pure-relevance ordering.
            lam = getattr(self.config, "ROUTING_MMR_LAMBDA", 0.7)
            result = _mmr_select(scored, top_n, lambda_relevance=lam)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "[doc_router] Routed collection %s: %d docs (%d with summaries) → top %d (MMR λ=%.2f) in %.1fms",
                collection_id, len(scored), len(summaries), len(result), lam, elapsed,
            )
            return result

        except Exception as exc:
            logger.warning(
                "[doc_router] route() failed for collection %s (falling back): %s",
                collection_id, exc,
            )
            return []


def _keyword_score(query: str, text: str) -> float:
    """Fallback BM25-ish scoring when topic_embedding is missing: word-overlap ratio."""
    if not text:
        return 0.0
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)
