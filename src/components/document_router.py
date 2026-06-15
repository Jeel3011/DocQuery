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
        metadata_filter: Optional[dict] = None,
    ) -> list[str]:
        """Return FILENAMES of the top-N most relevant documents for this query.

        Thin wrapper over `route_ranked` (which does the work and returns
        (doc_id, filename) pairs). Kept for the existing chat.py callers that scope the
        retriever by filename `$in`. New callers that want the stable `doc_id` scope
        axis (G3) should call `route_doc_ids`.

        Args:
            query:            User query string.
            collection_id:    Supabase collection UUID.
            user_id:          Owner user UUID (for RLS scoping).
            top_n:            How many docs to return (default: config.ROUTING_TOP_N).
            query_embedding:  Pre-computed query embedding (skips an OpenAI call).
            metadata_filter:  G3 Step D — pre-narrow the candidate set by doc_type /
                              fiscal_year BEFORE the vector fan-out (null-safe).

        Returns:
            List of filenames (strings), ordered most-relevant first.
            Returns [] if the collection is empty. Degrades gracefully when no routing
            data is available.
        """
        return [fn for _did, fn in self.route_ranked(
            query, collection_id, user_id,
            top_n=top_n, query_embedding=query_embedding, metadata_filter=metadata_filter,
        )]

    def route_doc_ids(
        self,
        query: str,
        collection_id: str,
        user_id: str,
        top_n: Optional[int] = None,
        query_embedding: Optional[list] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[str]:
        """Return DOC_IDS of the top-N most relevant documents (G3 scope axis).

        The agent-core / review-grid routes scope retrieval by `doc_id $in` (vault
        isolation as a DATA property), so they need the routed doc_ids, not filenames.
        """
        return [did for did, _fn in self.route_ranked(
            query, collection_id, user_id,
            top_n=top_n, query_embedding=query_embedding, metadata_filter=metadata_filter,
        )]

    def route_ranked(
        self,
        query: str,
        collection_id: str,
        user_id: str,
        top_n: Optional[int] = None,
        query_embedding: Optional[list] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[tuple[str, str]]:
        """Core router: return selected (doc_id, filename) pairs, most-relevant first.

        G3 Step D adds a metadata PRE-NARROW: for a vault with more than
        ``ROUTER_PRENARROW_THRESHOLD`` docs, the candidate set is filtered by the active
        ``metadata_filter`` (doc_type / fiscal_year) BEFORE any embedding work, so a
        1000-doc vault embed-searches only the matching subset, not all 1000 (CDB §7.3).
        The filter is null-safe (a doc with an unknown filtered value is kept, never
        guessed away) and never *widens* scope — it only drops known non-matches from a
        set that is already vault-scoped via collection membership.

        Returns [] on empty collection / failure (the caller falls back to the full list).
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
            #    because that column is never populated at ingest. We keep each doc's
            #    persisted doc_type / fiscal_year here so Step D can pre-narrow on them.
            coll_docs = db.get_collection_documents(collection_id)
            doc_meta = {
                d["id"]: {
                    "filename": d.get("filename", ""),
                    "doc_type": d.get("doc_type"),
                    "fiscal_year": d.get("fiscal_year"),
                }
                for d in (coll_docs or [])
                if d.get("id") and d.get("filename")
            }
            if not doc_meta:
                return []

            # 2. G3 Step D — metadata pre-narrow for big vaults (BEFORE any embedding).
            #    Only kicks in past the threshold: below it, scoring the whole (small)
            #    set is cheap and recall-first. The pre-narrow only DROPS known
            #    non-matches from the vault-scoped set — it can never add a doc outside
            #    the collection, so it cannot leak across vaults.
            full_count = len(doc_meta)
            if metadata_filter and full_count > ROUTER_PRENARROW_THRESHOLD:
                narrowed = {
                    did: m for did, m in doc_meta.items()
                    if _doc_matches_filter(m, metadata_filter)
                }
                if narrowed:  # never narrow to nothing — empty filter result keeps the vault
                    logger.info(
                        "[doc_router] pre-narrow collection %s: %d docs → %d match %s",
                        collection_id, full_count, len(narrowed), metadata_filter,
                    )
                    doc_meta = narrowed
            id_to_filename = {did: m["filename"] for did, m in doc_meta.items()}
            doc_ids = list(id_to_filename.keys())

            # 3. Fetch summaries + topic embeddings for exactly the (pre-narrowed) docs.
            rows = db.client.table("document_summaries").select(
                "document_id, summary, topic_embedding"
            ).in_("document_id", doc_ids).eq("user_id", user_id).execute()
            summaries = {r["document_id"]: r for r in (rows.data or [])}

            # 4. Embed the query once — only needed if any doc has a topic embedding.
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

            # 5. Score EVERY (pre-narrowed) doc so a doc without routing data is still a
            #    candidate (recall-first): cosine on the topic embedding when present,
            #    else keyword overlap on the summary, else on the filename. Keep each
            #    doc's topic vector AND doc_id so MMR (step 6) can measure redundancy and
            #    the caller can scope by doc_id.
            scored: list[tuple[float, str, Optional[list]]] = []
            id_by_filename: dict[str, str] = {}
            for did, filename in id_to_filename.items():
                id_by_filename[filename] = did
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

            # 6. Diversity-aware selection (MMR). When more docs than top_n,
            #    penalise candidates near-identical to ones already picked so a
            #    cross-entity query gets a spread of distinct docs instead of N
            #    near-duplicates from one cluster (Entry 7 fix). λ=1 reproduces the
            #    old pure-relevance ordering.
            lam = getattr(self.config, "ROUTING_MMR_LAMBDA", 0.7)
            result_fns = _mmr_select(scored, top_n, lambda_relevance=lam)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "[doc_router] Routed collection %s: %d docs (%d with summaries) → top %d (MMR λ=%.2f) in %.1fms",
                collection_id, len(scored), len(summaries), len(result_fns), lam, elapsed,
            )
            return [(id_by_filename[fn], fn) for fn in result_fns if fn in id_by_filename]

        except Exception as exc:
            logger.warning(
                "[doc_router] route_ranked() failed for collection %s (falling back): %s",
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


# ── Metadata pre-narrow (G3 Step D) ───────────────────────────────────────────

# >this many docs in a vault → pre-narrow by metadata BEFORE the vector fan-out, so a
# 1000-doc vault doesn't embed-search all 1000 (CDB §7.3). Below it, the candidate set
# is already small enough that pre-narrowing buys nothing — score them all (recall-first).
ROUTER_PRENARROW_THRESHOLD = 20


def _doc_matches_filter(meta: dict, metadata_filter: Optional[dict]) -> bool:
    """Does a doc's persisted metadata satisfy the active filter? (null-safe).

    `metadata_filter` mirrors the Pinecone shape used downstream:
        {"doc_type": "legal_contract", "fiscal_year": {"$in": [2023]}}  (scalar or $in).

    NULL-SAFE rule (G3 §5 risk #3): a doc whose value for a filtered key is **unknown**
    (None / missing) is NEVER excluded — we don't guess. Only a doc with a KNOWN,
    NON-matching value is dropped. This keeps a mis- or un-derived fiscal_year from
    silently hiding the right document (the corrupting-filter failure class).
    """
    if not metadata_filter:
        return True
    for key, cond in metadata_filter.items():
        val = meta.get(key)
        if val is None:
            continue  # unknown → don't exclude (never guess)
        if isinstance(cond, dict) and "$in" in cond:
            if val not in cond["$in"]:
                return False
        elif val != cond:
            return False
    return True
