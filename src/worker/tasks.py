"""
DocQuery — Celery Tasks

Heavy PDF processing (ingest -> chunk -> embed) runs here instead of
inside FastAPI's BackgroundTasks, freeing the API event loop.

Progress tracking: updates Supabase document status at each stage so
the frontend can show real progress instead of just "processing".

File transfer: The API uploads the raw file to Supabase Storage, then
passes the storage_path to this task.  The worker downloads the file
from Storage into its own /tmp — this is necessary because the API
and worker run as separate ECS containers with isolated filesystems.
The finally block always cleans up the local temp file.
"""

import os
import time
import logging

from src.worker.celery_app import celery

logger = logging.getLogger(__name__)

# Fix #2: reuse one EmbeddingManager per worker process. Constructing it builds the
# OpenAIEmbeddings client; recreating that on every task is wasted setup. Each forked
# child runs one task at a time, so a process-global is safe — only the Pinecone
# namespace varies per task, so we point the cached manager's config at the current
# task before use (the embedding model itself is namespace-independent).
_embed_mgr = None


def _get_embed_manager(config):
    global _embed_mgr
    from src.components.embeddings import EmbeddingManager
    if _embed_mgr is None:
        _embed_mgr = EmbeddingManager(config=config)
    else:
        _embed_mgr.config = config
    return _embed_mgr


@celery.task(bind=True, max_retries=2, default_retry_delay=30,
             acks_late=True, reject_on_worker_lost=True)
def process_document_task(
    self,
    filename: str,
    doc_id: str,
    storage_path: str,
    user_id: str,
    pinecone_namespace: str,
    collection_id: str | None = None,
    local_path: str | None = None,
):
    """
    Celery task: ingest -> chunk -> embed -> save chunks.

    Reports progress to Supabase at each stage:
      10% - started
      30% - elements extracted (unstructured parsing done)
      50% - chunks built
      80% - embeddings upserted to Pinecone
     100% - ready
    """
    from src.components.data_ingestion import DocumentProcessor
    from src.components.embeddings import EmbeddingManager
    from src.components.db import SupabaseManager
    from src.components.config import Config
    from src.components.metrics import uploads_total

    # Build a service-role Supabase client scoped to this user
    sb = SupabaseManager(use_service_role=True)
    sb._user = type("User", (), {"id": user_id})()

    config = Config()
    config.PINECONE_NAMESPACE = pinecone_namespace

    tmp_path = None        # the local file the worker reads to process
    spool_to_clean = None  # the API-spooled file we own and must delete at the end
    try:
        # PREFERRED PATH: the API spooled the file to a shared local path and did NOT
        # block the user on the slow Supabase upload. Read bytes from there directly
        # (zero storage round-trip on the request path), and upload to Storage HERE,
        # off the user's critical path, so the blob is still durable for re-download.
        if local_path and os.path.exists(local_path):
            tmp_path = local_path
            spool_to_clean = local_path
            logger.info("[%s] Using locally-spooled file: %s", doc_id, local_path)
            # Upload to Storage in the background (best-effort: a failure here only
            # costs durability/re-download, NOT processing — the user still gets a
            # queryable doc). Never fails the task on a slow/broken Storage upload.
            try:
                t_up = time.perf_counter()
                sb.upload_file_from_path(local_path, filename)
                logger.info("[%s] Background storage upload done in %.1fs",
                            doc_id, time.perf_counter() - t_up)
            except Exception as up_exc:
                logger.warning("[%s] Background storage upload failed (non-fatal, "
                               "doc still processes): %s", doc_id, up_exc)
        else:
            # FALLBACK (e.g. requeue after the spool file was cleaned, or a remote
            # worker that doesn't share the API's filesystem): download from Storage.
            logger.info("[%s] No local spool; downloading from storage: %s", doc_id, storage_path)
            suffix = os.path.splitext(filename)[1]  # e.g. ".pdf"
            try:
                tmp_path = sb.download_file_to_temp(storage_path, suffix=suffix)
            except Exception as dl_exc:
                logger.error("[%s] Failed to download from storage: %s", doc_id, dl_exc)
                sb.update_document_status(doc_id, "failed")
                uploads_total.labels(status="failed").inc()
                return {"status": "failed", "reason": f"storage download failed: {dl_exc}"}

        # -- Stage 1: Parse document (the slowest step) --
        sb.update_document_status(doc_id, "processing", progress_pct=10)
        logger.info("[%s] Stage 1/4: Parsing document %s", doc_id, filename)
        t_start = time.perf_counter()

        # C6: map page-level parse completion into the 10→30% band so the UI shows
        # continuous progress during the slow parse. Throttle to whole-percent jumps.
        _last_pct = [10]
        def _parse_progress(pages_done: int, total_pages: int):
            if not total_pages:
                return
            pct = 10 + int(20 * min(pages_done, total_pages) / total_pages)
            if pct > _last_pct[0]:
                _last_pct[0] = pct
                sb.update_document_status(doc_id, "processing", progress_pct=pct)

        processor = DocumentProcessor(config=config)
        elements = processor.process_documents(file_paths=tmp_path, progress_cb=_parse_progress)

        t_parse = time.perf_counter() - t_start
        logger.info("[%s] Parsing complete: %d elements in %.1fs", doc_id, len(elements), t_parse)

        # Fix filename metadata: process_documents uses the temp file name
        # (e.g. "tmpijcn_7da.pdf") but we need the original user-facing filename
        # so Pinecone metadata filters work correctly.
        for el in elements:
            if hasattr(el, 'metadata') and el.metadata:
                el.metadata.filename = filename

        if not elements:
            sb.update_document_status(doc_id, "failed")
            uploads_total.labels(status="failed").inc()
            return {"status": "failed", "reason": "no elements extracted"}

        sb.update_document_status(doc_id, "processing", progress_pct=30)

        # -- Stage 2: Build LangChain documents (chunking) --
        logger.info("[%s] Stage 2/4: Building chunks", doc_id)
        chunks = processor.build_langchain_documents(elements=elements, pdf_path=tmp_path)
        sb.update_document_status(doc_id, "processing", progress_pct=50)

        # -- Stamp workspace/doc/collection IDs on every chunk so Stage-1 routing
        # (Phase 3) can filter by collection_id without a huge $in filename list.
        # workspace_id == user_id for now; migrated to a real workspace table in Phase 7.
        for chunk in chunks:
            chunk.metadata["workspace_id"] = user_id
            chunk.metadata["doc_id"] = doc_id
            if collection_id:
                chunk.metadata["collection_id"] = collection_id

        # -- Stage 3: Embed and upsert to Pinecone --
        logger.info("[%s] Stage 3/4: Embedding %d chunks", doc_id, len(chunks))
        t_embed_start = time.perf_counter()

        embed_mgr = _get_embed_manager(config)
        embed_mgr.create_vector_store(chunks)

        t_embed = time.perf_counter() - t_embed_start
        logger.info("[%s] Embedding complete in %.1fs", doc_id, t_embed)

        # -- Done: the vectors are now in Pinecone, so the document is queryable.
        # Mark it ready immediately (Fix #3) — the user shouldn't wait on the Supabase
        # chunk bookkeeping below, which adds ~2s and nothing reads for retrieval.
        sb.update_document_status(doc_id, "ready", len(chunks), progress_pct=100)
        uploads_total.labels(status="success").inc()

        total_time = time.perf_counter() - t_start
        logger.info("[%s] Document ready: %d chunks in %.1fs (parse=%.1fs, embed=%.1fs)",
                    doc_id, len(chunks), total_time, t_parse, t_embed)

        # -- Post-ready bookkeeping: persist chunk text to Supabase. Nothing reads this
        # content for retrieval (that's Pinecone); only an analytics row-count touches
        # the table. So a failure here must NOT fail the document — log and move on.
        try:
            logger.info("[%s] Saving %d chunks to Supabase (analytics bookkeeping)", doc_id, len(chunks))
            sb.save_document_chunks(doc_id, chunks)
        except Exception as save_exc:
            logger.warning("[%s] Chunk save failed (non-fatal): %s", doc_id, save_exc)

        # -- Stage-1 routing data: summary + topic embedding per doc (Phase 3).
        # Non-fatal — the document is already queryable without it; routing just
        # falls back to unranked fanout until the row is present.
        try:
            from src.components.document_router import compute_and_store_doc_routing_data
            compute_and_store_doc_routing_data(
                chunks=chunks,
                doc_id=doc_id,
                user_id=user_id,
                collection_id=collection_id,
                config=config,
                db_client=sb,
            )
        except Exception as router_exc:
            logger.warning("[%s] Doc routing data failed (non-fatal): %s", doc_id, router_exc)

        return {"status": "ready", "chunks": len(chunks), "time_s": round(total_time, 1)}

    except Exception as exc:
        retries_left = self.max_retries - self.request.retries
        logger.exception(
            "[%s] Task failed (attempt %d/%d, retries_left=%d): %s",
            doc_id, self.request.retries + 1, self.max_retries + 1, retries_left, exc,
        )
        if self.request.retries >= self.max_retries:
            # All retries exhausted → DLQ path:
            # 1. Mark document as failed in DB so the user sees it (not stuck on "processing")
            # 2. Log at ERROR level for alerting
            # 3. Return cleanly — don't re-raise, task is done
            logger.error(
                "[%s] DLQ: document moved to failed after %d retries. "
                "Manual review required. File: %s",
                doc_id, self.max_retries, filename,
            )
            sb.update_document_status(doc_id, "failed")
            uploads_total.labels(status="failed").inc()
            return {"status": "dlq", "doc_id": doc_id, "reason": str(exc)}

        # Still have retries — update DB status and re-raise for Celery retry
        sb.update_document_status(doc_id, "failed")
        uploads_total.labels(status="failed").inc()
        raise self.retry(exc=exc)

    finally:
        # Always clean up the temp file — whether success, retry, or DLQ
        try:
            if tmp_path and os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError as e:
            logger.warning("[%s] Could not remove temp file %s: %s", doc_id, tmp_path, e)


@celery.task(bind=True)
def run_evaluation_task(
    self,
    questions_path: str = "eval_questions_v2.json",
    output_path: str = "eval_results.json",
    user_id: str = None,
):
    """
    Phase 5: Async RAGAS evaluation task.

    Triggered by POST /api/v1/admin/eval/run. Runs the full RAGAS evaluation
    pipeline in the background so the API stays responsive. Results are written
    to output_path for GET /admin/eval/results to serve.

    Why async via Celery?
    - 6 questions x 4 RAGAS metrics = 24 LLM calls at ~1-3s/call = 24-72s.
    - Running this synchronously would time out the HTTP request.
    """
    from src.components.config import Config
    from src.components.evaluation import RAGASEvaluator

    logger.info("[eval] Starting RAGAS evaluation from %s", questions_path)
    try:
        config = Config()
        if user_id:
            config.PINECONE_NAMESPACE = user_id

        evaluator = RAGASEvaluator(config)
        results = evaluator.evaluate(
            eval_dataset_path=questions_path,
            output_path=output_path,
            mode="baseline",
        )
        logger.info(
            "[eval] Complete. avg_score=%.4f, nan_metrics=%s",
            results.get("average_score", 0),
            results.get("nan_metrics", []),
        )
        return {"status": "complete", "average_score": results.get("average_score")}
    except Exception as exc:
        logger.exception("[eval] Evaluation task failed: %s", exc)
        return {"status": "failed", "error": str(exc)}
