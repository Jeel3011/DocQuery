-- 009_knowledge_base.sql
-- G8 §G8.0: the relational source-of-truth for the shared Indian legal Knowledge Base.
--
-- WHY relational, not vectors-only: the §G8.4 completeness gate (every section/Article
-- present, or the gap is listed) and the §G8.5 as-of gate (version-in-force) must be
-- COMPUTABLE from a ground truth — not inferred from what happened to embed. So every
-- ingested provision is BOTH a vector (in the `kb_in` Pinecone namespace, for retrieval)
-- AND a row here (for the gates + the tool's filters).
--
-- These tables MIRROR the dataclasses in src/components/knowledge/provision.py and the
-- row mappings in src/components/knowledge/store.py::SupabaseKnowledgeStore. Keep them in
-- lockstep — the store maps columns explicitly (no reflection) so a drift is a loud error
-- at the boundary, not a silent wrong column.
--
-- READ-ONLY-SHARED model (distinct from the per-user `documents` table, which is RLS'd to
-- the owner): the KB is one shared corpus of public legal authority. Every authenticated
-- user READS it; only the KB ingest CLI WRITES it (via the service role, bypassing RLS).
-- There is no user_id and no per-user isolation — isolation from user vaults is by
-- NAMESPACE (kb_in vs the per-user vector namespaces), enforced in the ingest adapter.
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS throughout).
-- Code degrades gracefully if this migration isn't applied yet (the live store is only
-- constructed on the $ ingest path; the offline gates run on InMemoryKnowledgeStore).

-- ── knowledge_sources — per-instrument metadata + the authoritative table of contents ──
-- Mirrors SourceMeta. `toc_ids` is the ground truth the completeness gate diffs the
-- ingested provisions against; without it captured at ingest, completeness is a hope.
CREATE TABLE IF NOT EXISTS knowledge_sources (
  source_key       text PRIMARY KEY,                 -- stable key ("constitution_of_india")
  title            text NOT NULL,                    -- "The Constitution of India"
  instrument_type  text NOT NULL,                    -- act|regulation|circular|judgment|article|schedule|rule
  jurisdiction     text NOT NULL DEFAULT 'IN',
  citation_prefix  text NOT NULL DEFAULT '',          -- "Constitution" → provisions cite "Constitution Art.N"
  enacted_date     date,
  as_of_date       date,                              -- the snapshot horizon for the whole source
  toc_ids          jsonb NOT NULL DEFAULT '[]'::jsonb, -- every expected section/article id (completeness ground truth)
  source_url       text,                              -- provenance (legislative.gov.in / India Code)
  partial          boolean NOT NULL DEFAULT false,    -- set by the completeness gate when the ToC isn't fully covered
  metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE knowledge_sources IS
  'G8 §G8.0: per-instrument KB metadata + authoritative ToC. Read-only-shared (no per-user RLS); written only by the KB ingest CLI. Mirrors SourceMeta.';
COMMENT ON COLUMN knowledge_sources.toc_ids IS
  'JSON array of every expected section/article id — the §G8.4 completeness ground truth, captured at ingest.';
COMMENT ON COLUMN knowledge_sources.as_of_date IS
  'Snapshot horizon: the date through which this source is known in force. A question dated past this → abstain (§G8.5).';

-- ── knowledge_provisions — one addressable unit of legal authority per row ──────────────
-- Mirrors Provision. `chunk_id` is the stable id shared by the row and its vector (in the
-- kb_in namespace) so the completeness gate joins ingested vectors to expected provisions
-- without a vector round-trip.
CREATE TABLE IF NOT EXISTS knowledge_provisions (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_key            text NOT NULL REFERENCES knowledge_sources(source_key) ON DELETE CASCADE,
  instrument_type       text NOT NULL,                -- one of INSTRUMENT_TYPES
  title                 text NOT NULL DEFAULT '',     -- human title ("Right to equality")
  citation              text NOT NULL,                -- the addressable locator ("Constitution Art.14")
  section_or_article_id text NOT NULL,                -- the bare id within the source ("14", "149", "II")
  text                  text NOT NULL,                -- the verbatim provision text (what gets quoted)
  jurisdiction          text NOT NULL DEFAULT 'IN',
  enacted_date          date,                         -- when the provision came into force (if known)
  as_of_date            date NOT NULL,                -- date through which this snapshot is known in force
                                                      -- (NOT NULL: the version-in-force guarantee requires it;
                                                      --  the ingest adapter rejects a provision without one)
  repealed              boolean NOT NULL DEFAULT false,
  superseded_by         text,                         -- citation of the superseding provision, if any
  toc_path              text,                         -- path in the source's ToC ("Part III > Art.14")
  chunk_id              text UNIQUE,                  -- stable id shared with the vector (row↔vector join key)
  page                  integer,                      -- source page (provenance)
  metadata              jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at            timestamptz NOT NULL DEFAULT now(),
  updated_at            timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE knowledge_provisions IS
  'G8 §G8.0: provision-granular legal authority (Article/section/judgment). Read-only-shared; written only by the KB ingest CLI. Mirrors Provision.';
COMMENT ON COLUMN knowledge_provisions.chunk_id IS
  'Stable id shared by this row and its Pinecone vector (kb_in namespace) — the join key the §G8.4 completeness gate uses. UNIQUE so upsert(on_conflict=chunk_id) is idempotent.';
COMMENT ON COLUMN knowledge_provisions.as_of_date IS
  'Snapshot horizon for this provision (NOT NULL — the §G8.5 version-in-force guarantee needs it). A query dated past this → withhold/abstain, never guess.';

-- Indexes mirror the store's read paths: list_provisions(source_key), the search_knowledge
-- metadata filters (source/instrument_type/jurisdiction), and the chunk_id join.
CREATE INDEX IF NOT EXISTS idx_kb_provisions_source       ON knowledge_provisions(source_key);
CREATE INDEX IF NOT EXISTS idx_kb_provisions_instrument   ON knowledge_provisions(instrument_type);
CREATE INDEX IF NOT EXISTS idx_kb_provisions_jurisdiction ON knowledge_provisions(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_kb_provisions_chunk        ON knowledge_provisions(chunk_id);

-- ── RLS: read-only-shared ───────────────────────────────────────────────────────────────
-- Every authenticated user may SELECT the shared corpus; nobody writes through the anon/
-- authenticated roles. The ingest CLI uses the service-role key, which bypasses RLS, so no
-- write policy is granted here (writes are intentionally impossible for normal clients).
ALTER TABLE knowledge_sources    ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_provisions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Authenticated read knowledge_sources"    ON knowledge_sources;
DROP POLICY IF EXISTS "Authenticated read knowledge_provisions" ON knowledge_provisions;

CREATE POLICY "Authenticated read knowledge_sources" ON knowledge_sources
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Authenticated read knowledge_provisions" ON knowledge_provisions
  FOR SELECT USING (auth.role() = 'authenticated');
