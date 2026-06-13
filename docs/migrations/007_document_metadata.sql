-- 007_document_metadata.sql
-- G2 Step F: surface G1d document metadata in the UI (doc table type-chip + fidelity dot).
-- G1d (classify_document) computes a structural doc class at ingest and the extraction
-- fidelity self-report computes per-page coverage — but both were only written to chunk
-- metadata / logs, never to the document row, so the API and UI couldn't read them.
-- Add two columns the worker persists when a doc flips to 'ready':
--   doc_type  — G1d class: financial_filing | legal_contract | mixed | generic
--   fidelity  — coarse extraction-fidelity grade: good | partial  (NULL = unknown/legacy)
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to run more than once.
-- Code degrades gracefully if this migration isn't applied yet (forward-compat: the
-- worker drops these keys on column-missing error; the API/UI show neutral placeholders).

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS doc_type text,
  ADD COLUMN IF NOT EXISTS fidelity text;

COMMENT ON COLUMN documents.doc_type IS
  'G1d structural document class: financial_filing | legal_contract | mixed | generic.';
COMMENT ON COLUMN documents.fidelity IS
  'Coarse extraction-fidelity grade at ingest: good | partial. NULL = unknown/legacy.';
