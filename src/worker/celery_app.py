"""
DocQuery — Celery Application Factory

Broker & backend: Redis (configurable via REDIS_URL env var).

Queue structure:
  documents.fast   — txt / small PDFs (<500KB) — concurrency=4
  documents.normal — medium PDFs (<5MB)        — concurrency=2  [default]
  documents.heavy  — large PDFs (≥5MB)         — concurrency=1
  documents.dlq    — failed after max_retries  — manual review / alerting

Start the worker with:
    celery -A src.worker.celery_app worker --loglevel=info -Q documents.fast,documents.normal,documents.heavy
"""

import os
from celery import Celery
from kombu import Queue, Exchange
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "docquery",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Default exchange for all document queues
_doc_exchange = Exchange("documents", type="direct")

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,             # re-deliver if worker crashes mid-task
    worker_prefetch_multiplier=1,    # one task at a time per worker (CPU-heavy)
    task_reject_on_worker_lost=True, # re-queue task if worker is killed mid-task

    # ── Priority queues ──────────────────────────────────────────────────────
    # Separate queues prevent a 500-page PDF from blocking a 2-page txt.
    task_queues=(
        Queue("documents.fast",   _doc_exchange, routing_key="fast"),
        Queue("documents.normal", _doc_exchange, routing_key="normal"),
        Queue("documents.heavy",  _doc_exchange, routing_key="heavy"),
        Queue("documents.dlq",    _doc_exchange, routing_key="dlq"),   # dead letter
    ),
    task_default_queue="documents.normal",
    task_default_exchange="documents",
    task_default_routing_key="normal",

    # Route process_document_task to documents.normal by default.
    # The upload endpoint overrides the queue per file size (see documents.py).
    task_routes={
        "src.worker.tasks.process_document_task": {"queue": "documents.normal"},
    },
)

# Auto-discover tasks in src/worker/tasks.py
celery.autodiscover_tasks(["src.worker"])


# \u2500\u2500 Model pre-warming at worker startup (Layer 1) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# When this worker container boots, create the persistent PDF pool and submit
# warm-up tasks to each child process.
#
# Models are warmed INSIDE pool child processes, not in the parent Celery
# process. This is safe for PyTorch: no PyTorch state crosses the fork()
# boundary, eliminating file descriptor leaks and potential deadlocks.
#
# Cost: ~8-10s at container boot (paid once).
# Saves: ~20s on EVERY subsequent PDF processed by this worker.
from celery.signals import worker_init


@worker_init.connect
def on_worker_init(sender, **kwargs):
    import logging
    logger = logging.getLogger(__name__)
    try:
        from src.components.config import Config
        from src.components.data_ingestion import warm_pdf_pool
        cfg = Config()
        logger.info(
            "Worker startup: pre-warming PDF pool (%d workers)...",
            cfg.PDF_PARALLEL_WORKERS,
        )
        warm_pdf_pool(n_workers=cfg.PDF_PARALLEL_WORKERS, timeout=90)
        logger.info("Worker startup: PDF pool ready.")

        # Fix #1: pre-import the heavy task dependencies in the worker PARENT so the
        # prefork children inherit them via copy-on-write. Without this, the FIRST
        # upload handled by each child paid ~6s importing langchain/supabase inside
        # the task (data_ingestion is already imported above via warm_pdf_pool, which
        # is why parsing was fast but the pre-parse gap was not). The API never runs
        # worker_init, so its process stays lean.
        import importlib
        for _mod in ("src.components.embeddings", "src.components.db"):
            try:
                importlib.import_module(_mod)
            except Exception as _imp_exc:
                logger.warning("Pre-import of %s failed (non-fatal): %s", _mod, _imp_exc)
        logger.info("Worker startup: task deps pre-imported.")
    except Exception as exc:
        # Non-fatal \u2014 first PDF upload will still work, just slower
        logging.getLogger(__name__).warning(
            "Worker startup warm-up failed (non-fatal, first PDF will be slower): %s", exc
        )

    # \u2500\u2500 Sweep zombie ingests \u2500\u2500
    # A worker crash (e.g. SIGSEGV in native parsing) kills the task process
    # before its except-block can mark the document "failed", so the doc is stuck
    # in "processing" forever and the UI spins indefinitely. On every worker boot,
    # flip any doc that has been "processing" longer than the longest plausible
    # ingest to "failed", so the user sees a clear error instead of a hang.
    try:
        from datetime import datetime, timedelta, timezone
        from src.components.db import SupabaseManager
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        _db = SupabaseManager(use_service_role=True)
        _res = (
            _db.client.table("documents")
            .update({"status": "failed"})
            .eq("status", "processing")
            .lt("created_at", cutoff)
            .execute()
        )
        _n = len(_res.data or [])
        if _n:
            logger.warning("Worker startup: swept %d stale 'processing' doc(s) \u2192 'failed'", _n)
        else:
            logger.info("Worker startup: no stale 'processing' docs to sweep.")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Worker startup: stale-processing sweep failed (non-fatal): %s", exc
        )
