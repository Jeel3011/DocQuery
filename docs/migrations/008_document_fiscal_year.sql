-- 008_document_fiscal_year.sql
-- G3 Step C: persist the structurally-derived fiscal year on the document row so the
-- vault's FY filter chip can read it and narrow retrieval scope.
-- derive_fiscal_year (data_ingestion.py) computes this STRUCTURALLY at ingest ($0, no
-- LLM) from the filename's period-end date / FY tag, falling back to the latest year in
-- the extracted statements' period headers. It is NULL when not structurally certain —
-- the FY filter treats NULL as "unknown, don't exclude" (never hides the right doc).
--
-- Mirrors migration 007 (doc_type / fidelity). Apply via the Supabase SQL editor (or
-- `psql`). Safe to run more than once. Code degrades gracefully if not yet applied (the
-- worker drops the key on column-missing error; the API/UI omit the FY chip).

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS fiscal_year integer;

COMMENT ON COLUMN documents.fiscal_year IS
  'G3 structural fiscal year (int) derived at ingest from filename period-end / FY tag, '
  'else the latest period-header year. NULL = unknown/legacy (FY filter does not exclude).';
