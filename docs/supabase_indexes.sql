-- DocQuery — Supabase Performance Indexes (Phase 4)
-- Run these in Supabase SQL Editor: https://supabase.com/dashboard → SQL Editor
--
-- These cost nothing and deliver significant query speedup at scale:
--   - document list view: idx_documents_user_status
--   - chunk retrieval:    idx_document_chunks_document_user + idx_document_chunks_user_id
--   - message history:   idx_messages_conversation_created
--
-- Run once. IF NOT EXISTS makes them idempotent (safe to re-run).

-- ── Speed up document list view (GET /documents) ──────────────────────────────
-- Covers: SELECT * FROM documents WHERE user_id = $1 ORDER BY created_at DESC
CREATE INDEX IF NOT EXISTS idx_documents_user_status
ON documents (user_id, status, created_at DESC);

-- ── Speed up document chunk retrieval ─────────────────────────────────────────
-- Covers: SELECT * FROM document_chunks WHERE document_id = $1 AND user_id = $2
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_user
ON document_chunks (document_id, user_id);

-- Covers: DELETE FROM document_chunks WHERE user_id = $1 (on doc delete)
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id
ON document_chunks (user_id);

-- ── Speed up conversation message retrieval ────────────────────────────────────
-- Covers: SELECT * FROM messages WHERE conversation_id = $1 ORDER BY created_at
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
ON messages (conversation_id, created_at);

-- ── Speed up conversation list view ───────────────────────────────────────────
-- Covers: SELECT * FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
ON conversations (user_id, updated_at DESC);
