-- DocQuery — Supabase SQL Hardening
-- Run this in the Supabase SQL Editor: https://app.supabase.com → SQL Editor
-- Safe to run multiple times (uses IF NOT EXISTS / DO NOTHING patterns)

-- ============================================================
-- 1. FORCE ROW LEVEL SECURITY
-- Prevents service-role key from bypassing RLS on these tables.
-- This means even internal code MUST pass user_id filters.
-- ============================================================

ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE messages FORCE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE conversations FORCE ROW LEVEL SECURITY;

-- ============================================================
-- 2. PERFORMANCE INDEXES
-- Free performance wins. Each takes <1s on empty/small tables.
-- ============================================================

-- Documents: most common query pattern = "get user's docs by status or recency"
CREATE INDEX IF NOT EXISTS idx_documents_user_status
  ON documents (user_id, status);

CREATE INDEX IF NOT EXISTS idx_documents_user_created
  ON documents (user_id, created_at DESC);

-- Document chunks: lookup by document or user for RAG retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_document_user
  ON document_chunks (document_id, user_id);

CREATE INDEX IF NOT EXISTS idx_chunks_user_id
  ON document_chunks (user_id);

-- Messages: most common = "get messages for a conversation in order"
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
  ON messages (conversation_id, created_at);

-- Conversations: "get user's conversations sorted by last activity"
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
  ON conversations (user_id, updated_at DESC);

-- ============================================================
-- Done! Verify with:
-- SELECT indexname, tablename FROM pg_indexes
-- WHERE schemaname = 'public'
-- ORDER BY tablename, indexname;
-- ============================================================
