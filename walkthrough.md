# DocQuery Production Readiness — Walkthrough

## What Was Implemented

### ✅ Phase 1.1 — FastAPI Architecture Restored

**File:** [frontend/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/frontend/chat.py)

Streamlit is now a **thin API client** — all RAG components removed from imports.

| Before | After |
|---|---|
| Imported `DocumentProcessor`, [EmbeddingManager](file:///Users/jeelthummar/Desktop/DocQuery/src/components/embeddings.py#12-113), [RetrievalManager](file:///Users/jeelthummar/Desktop/DocQuery/src/components/retrieval.py#7-58), [AnswerGenration](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py#15-145) | Only imports [SupabaseManager](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#40-285) (auth) + `httpx` (API calls) |
| Ran ML pipeline inline, blocking session | All operations proxied to FastAPI backend |
| Uploaded files processed synchronously (app frozen) | Upload → `POST /api/v1/documents/upload` → 202 immediately |
| Chat responses generated inline | Streamed via `POST /api/v1/query/stream` (SSE) |

**Auth flow preserved:** Streamlit still handles login/logout directly via Supabase. Bearer token is passed as `Authorization: Bearer <access_token>` on every API call. FastAPI's existing [get_current_user](file:///Users/jeelthummar/Desktop/DocQuery/src/api/dependencies.py#47-78) dependency validates it.

**No double LLM call:** After SSE streaming completes, the exact streamed answer is saved to Supabase directly via `sb.save_message()` — the message endpoint is NOT called, avoiding re-running the LLM.

---

### ✅ Phase 1.2 — Chat History Token Limits

**Files:** [utils.py](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py), [genration.py](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py), [src/api/routes/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/chat.py)

[format_chat_history()](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116) now applies two safety layers:
1. **Sliding window:** Keeps only the last 6 messages (3 user + 3 assistant turns)
2. **Token ceiling:** Uses `tiktoken` (cl100k_base) to count tokens. Drops oldest messages until total history ≤ 2000 tokens. Gracefully falls back to character-based truncation if `tiktoken` isn't installed.

[generate()](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py#73-102) method now also accepts [chat_history](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116) (previously only [generate_stream()](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py#103-145) did).

---

### ✅ Phase 1.3 — Pickle Files Eliminated

**Files:** [db.py](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py), [documents.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/documents.py)

| Removed | Added |
|---|---|
| `upload_pkl()` | [save_document_chunks(document_id, chunks)](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#127-154) |
| `download_pkl()` | [get_document_chunks(document_id)](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#155-166) |

Extracted text chunks now stored in the **[document_chunks](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#155-166) Supabase table** as plain text + JSONB metadata. No binary deserialization, no RCE risk.

> [!IMPORTANT]
> Run the SQL migration before uploading any new documents:
> [docs/migrations/001_document_chunks.sql](file:///Users/jeelthummar/Desktop/DocQuery/docs/migrations/001_document_chunks.sql)

**Deduplication is preserved:** The [EmbeddingManager](file:///Users/jeelthummar/Desktop/DocQuery/src/components/embeddings.py#12-113) SHA-256 content hash dedup still works — it checks the vector store (ChromaDB) for existing `chunk_id`s, not the pkl files.

---

### ✅ Phase 2.2 — Asynchronous Uploads

**File:** [documents.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/documents.py)

`POST /api/v1/documents/upload` now:
1. Validates file type + size
2. Uploads raw file to Supabase Storage immediately
3. Creates a [documents](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/documents.py#141-160) DB record with `status=processing`
4. **Returns `202 Accepted` immediately** (no more 504 timeouts!)
5. Background task runs: Unstructured → chunk → embed → [save_document_chunks](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#127-154) → `status=ready`

Frontend polls `GET /api/v1/documents` to check when status changes from `processing` → `ready`.

---

### ✅ Phase 2.3 — Contextual Follow-Up Retrieval

**Files:** [src/components/genration.py](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py), [src/api/routes/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/chat.py), [frontend/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/frontend/chat.py)

Follow-up questions (e.g., "what is the purpose of it?") were originally passed directly to the vector DB, yielding no matches because context was missing.
1. [frontend/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/frontend/chat.py) now passes `conversation_id` in the `POST /query/stream` payload.
2. The FastAPI backend fetches the relevant [chat_history](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116).
3. `AnswerGenration.rewrite_query()` uses a fast LLM chain to rewrite the ambiguous question into a standalone query (e.g., "What is the purpose of the Transformer Encoder?").
4. The vector DB searches for this standalone query, successfully finding and citing the relevant documents.

---

## How to Run

```bash
# Terminal 1 — FastAPI backend
source venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit frontend
source venv/bin/activate
streamlit run frontend/chat.py
```

Set `API_BASE_URL` in [.env](file:///Users/jeelthummar/Desktop/DocQuery/.env) if FastAPI runs on a different port/host:
```
API_BASE_URL=http://localhost:8000
```

---

## What Still Needs to Be Done

| Phase | Item |
|---|---|
| 2.1 | Migrate ChromaDB → Pinecone (cloud vector DB so embeddings survive restarts) |
| 3.1 | Dockerize (multi-stage Dockerfile + docker-compose) |
| 3.2 | CI/CD GitHub Actions |
| 4.x | Celery, Hybrid Search, Reranking, Caching, GraphRAG, LangGraph, Telemetry |

## Files Changed

| File | Change |
|---|---|
| [frontend/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/frontend/chat.py) | Complete rewrite — thin httpx API client |
| [src/utils.py](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py) | Tiktoken sliding window in [format_chat_history()](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116) |
| [src/components/genration.py](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py) | Added [chat_history](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116) param to [generate()](file:///Users/jeelthummar/Desktop/DocQuery/src/components/genration.py#73-102) |
| [src/components/db.py](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py) | Removed pkl methods, added `save/get_document_chunks()` |
| [src/api/routes/documents.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/documents.py) | Async upload with BackgroundTasks, returns 202 |
| [src/api/routes/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/chat.py) | [send_message](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/chat.py#206-278) now passes [chat_history](file:///Users/jeelthummar/Desktop/DocQuery/src/utils.py#71-116) to generator |
| [requirements.txt](file:///Users/jeelthummar/Desktop/DocQuery/requirements.txt) | Added `httpx`, `tiktoken` |
| [docs/migrations/001_document_chunks.sql](file:///Users/jeelthummar/Desktop/DocQuery/docs/migrations/001_document_chunks.sql) | **NEW** — Supabase migration for [document_chunks](file:///Users/jeelthummar/Desktop/DocQuery/src/components/db.py#155-166) table |
