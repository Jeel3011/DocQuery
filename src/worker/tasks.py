"""
DocQuery — Celery Tasks

Heavy PDF processing (ingest -> chunk -> embed) runs here instead of
inside FastAPI's BackgroundTasks, freeing the API event loop.

Progress tracking: updates Supabase document status at each stage so
the frontend can show real progress instead of just "processing".
"""

import os
import time
import logging

from src.worker.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, max_retries=2, default_retry_delay=30)
def process_document_task(
    self,
    filename: str,
    doc_id: str,
    tmp_path: str,
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

    try:
        # Verify the file exists on disk
        if not os.path.isfile(tmp_path):
            logger.error("File not found at %s for doc %s", tmp_path, doc_id)
            sb.update_document_status(doc_id, "failed")
            uploads_total.labels(status="failed").inc()
            return {"status": "failed", "reason": "file not found on disk"}

        # -- Stage 1: Parse document (the slowest step) --
        sb.update_document_status(doc_id, "processing", progress_pct=10)
        logger.info("[%s] Stage 1/4: Parsing document %s", doc_id, filename)
        t_start = time.perf_counter()

        processor = DocumentProcessor(config=config)
        elements = processor.process_documents(file_paths=tmp_path)

        t_parse = time.perf_counter() - t_start
        logger.info("[%s] Parsing complete: %d elements in %.1fs", doc_id, len(elements), t_parse)

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
        logger.exception("Celery task failed for doc %s", doc_id)
        sb.update_document_status(doc_id, "failed")
        uploads_total.labels(status="failed").inc()
        raise self.retry(exc=exc)  # retry up to max_retries

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
