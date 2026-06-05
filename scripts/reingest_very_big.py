"""One-off: re-ingest the 'very big' collection so Phase 4.3 table chunks land.

Enqueues process_document_task for each of the 8 docs using their existing
storage paths + the collection owner's namespace. Idempotent on text chunks
(stable content-hash ids overwrite in place); adds new chunk_type=table chunks.

Run with the stack up (Redis + worker --pool=solo). Usage:
    python scripts/reingest_very_big.py
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")

from src.components.db import get_supabase_client

COLLECTION_ID = "1c485387-3fd0-4608-a677-64d724323f90"


def main():
    svc = get_supabase_client(use_service_role=True)
    owner = svc.table("collections").select("user_id").eq("id", COLLECTION_ID).single().execute().data
    user_id = owner["user_id"]
    namespace = user_id  # app namespaces Pinecone by user_id (Entry 5)

    doc_ids = [
        d["document_id"]
        for d in svc.table("collection_documents")
        .select("document_id").eq("collection_id", COLLECTION_ID).execute().data
    ]

    from src.worker.tasks import process_document_task

    print(f"Re-ingesting {len(doc_ids)} docs in collection {COLLECTION_ID}")
    for did in doc_ids:
        d = svc.table("documents").select("*").eq("id", did).single().execute().data
        size = d.get("file_size_bytes") or 0
        queue = ("documents.fast" if size < 500_000
                 else "documents.normal" if size < 5_000_000
                 else "documents.heavy")
        process_document_task.apply_async(
            kwargs=dict(
                filename=d["filename"],
                doc_id=did,
                storage_path=d["storage_path"],
                user_id=user_id,
                pinecone_namespace=namespace,
                collection_id=COLLECTION_ID,
            ),
            queue=queue,
        )
        print(f"  queued {d['filename']:<32} ({size//1024} KB → {queue})")
    print("All tasks enqueued. Watch .devlogs/worker.log for progress.")


if __name__ == "__main__":
    main()
