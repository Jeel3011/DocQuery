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


@celery.task(bind=True, max_retries=2, default_retry_delay=30,
             acks_late=True, reject_on_worker_lost=True)
def process_document_task(
    self,
    filename: str,
    doc_id: str,
    storage_path: str,
    user_id: str,
    pinecone_namespace: str,
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

    tmp_path = None  # will be set after download
    try:
        # Download the file from Supabase Storage into worker's local /tmp
        logger.info("[%s] Downloading file from storage: %s", doc_id, storage_path)
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

        processor = DocumentProcessor(config=config)
        elements = processor.process_documents(file_paths=tmp_path)

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
        chunks = processor.build_langchain_documents(elements=elements)
        sb.update_document_status(doc_id, "processing", progress_pct=50)

        # -- Stage 3: Embed and upsert to Pinecone --
        logger.info("[%s] Stage 3/4: Embedding %d chunks", doc_id, len(chunks))
        t_embed_start = time.perf_counter()

        embed_mgr = EmbeddingManager(config=config)
        embed_mgr.create_vector_store(chunks)

        t_embed = time.perf_counter() - t_embed_start
        logger.info("[%s] Embedding complete in %.1fs", doc_id, t_embed)
        sb.update_document_status(doc_id, "processing", progress_pct=80)

        # -- Stage 4: Save chunks to Supabase --
        logger.info("[%s] Stage 4/4: Saving %d chunks to Supabase", doc_id, len(chunks))
        sb.save_document_chunks(doc_id, chunks)

        # -- Done --
        sb.update_document_status(doc_id, "ready", len(chunks), progress_pct=100)
        uploads_total.labels(status="success").inc()

        total_time = time.perf_counter() - t_start
        logger.info("[%s] Document ready: %d chunks in %.1fs (parse=%.1fs, embed=%.1fs)",
                    doc_id, len(chunks), total_time, t_parse, t_embed)
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
