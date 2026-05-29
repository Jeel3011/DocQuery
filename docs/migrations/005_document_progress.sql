-- 005_document_progress.sql
-- C6: real ingest progress. The `documents` table had no progress column, so the
-- worker's progress_pct updates were silently dropped and the UI bar was stuck at
-- its fallback. Add a 0–100 percentage the worker updates at page-level granularity.
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to run more than once.

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS processing_progress smallint NOT NULL DEFAULT 0;

COMMENT ON COLUMN documents.processing_progress IS
  'Ingest progress 0–100, updated during processing (parse pages → embed → save).';
