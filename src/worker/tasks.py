"""
DocQuery — Celery Tasks

Heavy PDF processing (ingest → chunk → embed) runs here instead of
inside FastAPI's BackgroundTasks, freeing the API event loop.
"""

import os
import logging

from src.worker.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, max_retries=2, default_retry_delay=30)
def process_document_task(
    self,
    file_bytes_hex: str,
    filename: str,
    doc_id: str,
    tmp_path: str,
    user_id: str,
    pinecone_namespace: str,
):
    """
    Celery task: ingest → chunk → embed → save chunks.

    file_bytes are hex-encoded because Celery's JSON serializer
    cannot handle raw bytes.
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
        file_bytes = bytes.fromhex(file_bytes_hex)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        processor = DocumentProcessor(config=config)
        elements = processor.process_documents(file_paths=tmp_path)

        if not elements:
            sb.update_document_status(doc_id, "failed")
            uploads_total.labels(status="failed").inc()
            return {"status": "failed", "reason": "no elements extracted"}

        chunks = processor.build_langchain_documents(elements=elements)

        embed_mgr = EmbeddingManager(config=config)
        embed_mgr.create_vector_store(chunks)

        sb.save_document_chunks(doc_id, chunks)
        sb.update_document_status(doc_id, "ready", len(chunks))

        uploads_total.labels(status="success").inc()
        return {"status": "ready", "chunks": len(chunks)}

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
