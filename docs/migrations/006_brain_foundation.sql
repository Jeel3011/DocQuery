-- Phase 1 & 3: Brain foundation tables
-- Run after 002_collections.sql.

-- ── document_summaries ──────────────────────────────────────────────────────
-- Stores one summary + topic embedding per document.
-- Used by Stage-1 document router (Phase 3) to pick the top-N relevant docs
-- for a query without fanning out a Pinecone query to every file in a collection.
-- Also stores the doc-level centroid embedding (mean of chunk embeddings).

CREATE TABLE IF NOT EXISTS document_summaries (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  collection_id   UUID REFERENCES collections(id) ON DELETE SET NULL,
  summary         TEXT NOT NULL,              -- short extractive/LLM summary
  -- Centroid of chunk embeddings, stored as a JSON array string. We score cosine
  -- similarity in Python (document_router._cosine), NOT in Postgres, so we do NOT
  -- need the pgvector extension here. If you later want in-DB ANN search, enable
  -- pgvector (`create extension if not exists vector with schema extensions;`) and
  -- change this to VECTOR(1536).
  topic_embedding TEXT,
  model_used      TEXT,                       -- which embedding model produced it
  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now(),
  UNIQUE (document_id)
);

ALTER TABLE document_summaries ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own summaries" ON document_summaries
  FOR ALL USING (user_id = auth.uid());

-- Fast lookup: all summaries for a user's collection (Stage-1 routing query)
CREATE INDEX IF NOT EXISTS idx_docsummaries_collection
  ON document_summaries(collection_id, user_id);

-- Fast lookup: a single doc's summary
CREATE INDEX IF NOT EXISTS idx_docsummaries_doc
  ON document_summaries(document_id);

-- ── runs ────────────────────────────────────────────────────────────────────
-- Audit ledger for every Brain/harness run (Phase 4+).
-- Records which documents were consulted, per-doc coverage, and cost.
-- "Coverage is itself a sellable feature in regulated work." — CDB plan §3.

CREATE TABLE IF NOT EXISTS runs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  collection_id   UUID REFERENCES collections(id) ON DELETE SET NULL,
  conversation_id UUID,                       -- links to chat conversation
  query_text      TEXT NOT NULL,
  run_type        TEXT NOT NULL DEFAULT 'fast_path',  -- fast_path | map_reduce | agentic
  docs_routed     INT  DEFAULT 0,             -- how many docs Stage-1 selected
  docs_read       INT  DEFAULT 0,             -- how many docs Stage-2 MAP processed
  docs_relevant   INT  DEFAULT 0,             -- how many returned evidence
  docs_failed     INT  DEFAULT 0,             -- MAP failures (non-fatal)
  confidence      FLOAT,                      -- REDUCE confidence score (0-1)
  token_cost      INT  DEFAULT 0,             -- rough token estimate for the run
  wall_ms         INT,                        -- total wall-clock milliseconds
  created_at      TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own runs" ON runs
  FOR ALL USING (user_id = auth.uid());

CREATE INDEX IF NOT EXISTS idx_runs_user_created
  ON runs(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_collection
  ON runs(collection_id, created_at DESC);
