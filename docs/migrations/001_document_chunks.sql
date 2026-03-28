-- =============================================================
-- DocQuery — Supabase SQL Migration
-- Run this in the Supabase SQL editor to create the
-- document_chunks table required for Phase 1.3 (pickle elimination).
-- =============================================================

-- 1. Create document_chunks table
--    Stores extracted text chunks from documents in plain text + JSONB metadata.
--    This replaces the insecure binary .pkl caching approach.
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Enforce unique chunk positions per document
    UNIQUE (document_id, chunk_index)
);

-- 2. Index for fast lookups by document and user
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id
    ON document_chunks (user_id);

-- 3. Row Level Security (RLS) — users can only see their own chunks
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage their own chunks" ON document_chunks;
CREATE POLICY "Users can manage their own chunks"
    ON document_chunks
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =============================================================
-- VERIFICATION QUERIES (run after migration to confirm setup)
-- =============================================================

-- Check table was created:
-- SELECT table_name FROM information_schema.tables WHERE table_name = 'document_chunks';

-- Check RLS is enabled:
-- SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'document_chunks';
