-- 010_matter_vaults.sql
-- F1a (UNIFIED_FEATURES_PLAN F1 / plans/F1_VAULT_PLAN.md §1): promote the `collection`
-- into a first-class, typed MATTER/VAULT inside a MULTI-FIRM tenancy (firm → matter → vault).
--
-- WHY this slice: F1 makes "which documents" a hard, typed invariant every other feature
-- inherits. F1a is the DATA-MODEL spine only — NO vector/namespace changes (that is F1b),
-- so the live retrieval path is byte-identical after this migration. The columns added to
-- `collections` are all ADDITIVE and DEFAULTED/nullable ⇒ existing rows stay valid and the
-- pre-F1 read path (`db.py` create/get/list) is unchanged for callers that don't send them.
--
-- TENANCY (Jeel, 2026-06-22, resolves UNIFIED §10.1): MULTI-FIRM SaaS from day one. A firm
-- owns matters (vaults); a user belongs to a firm. F1a adds the `firms` + `firm_memberships`
-- substrate; F2 (Firm Console) extends `firm_memberships` with ROLES and owns the firm-scoped
-- RLS. F1a deliberately KEEPS the working per-user RLS on `collections` so nothing breaks —
-- the firm-scoped row policy lands with F2, not here (don't half-build it).
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS throughout).
-- Code degrades gracefully if this migration isn't applied yet: `db.py` only references the
-- new columns when a caller sends them, and the new tables are read lazily.

-- ── firms — the top tenancy axis (a law/finance firm or an org) ──────────────────────────
CREATE TABLE IF NOT EXISTS firms (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- ── firm_memberships — which firm a user belongs to (F2 will add a `role` column) ─────────
CREATE TABLE IF NOT EXISTS firm_memberships (
  user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  firm_id     UUID NOT NULL REFERENCES firms(id) ON DELETE CASCADE,
  created_at  TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, firm_id)
);

-- ── promote `collections` → a typed MATTER/VAULT (additive columns only) ──────────────────
-- matter_kind: the 9 kinds from F1 (litigation · m&a · lending · arbitration · ip · regulatory
--   · employment · advisory · compliance). Nullable ⇒ a legacy/untyped vault is still valid;
--   F1c's practice-template self-config keys off it.
-- status: the matter lifecycle (active → on_hold → closed → archived → legal_hold). Defaulted
--   so every existing row reads as 'active' without a backfill.
-- parties: named parties for the F1c metadata-only conflict scan. JSONB array of {name, role?}.
-- firm_id: the owning firm (nullable during transition — a user without a firm keeps working;
--   F1b/F2 tighten this once firm membership is enforced).
ALTER TABLE collections ADD COLUMN IF NOT EXISTS firm_id     UUID REFERENCES firms(id);
ALTER TABLE collections ADD COLUMN IF NOT EXISTS matter_kind TEXT;
ALTER TABLE collections ADD COLUMN IF NOT EXISTS status      TEXT NOT NULL DEFAULT 'active';
ALTER TABLE collections ADD COLUMN IF NOT EXISTS parties     JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Enforce the small closed sets at the DB boundary (a wrong kind/status is a loud error here,
-- not a silent bad row). DROP-then-ADD so the migration is re-runnable.
ALTER TABLE collections DROP CONSTRAINT IF EXISTS collections_matter_kind_chk;
ALTER TABLE collections ADD  CONSTRAINT collections_matter_kind_chk CHECK (
  matter_kind IS NULL OR matter_kind IN (
    'litigation','m&a','lending','arbitration','ip',
    'regulatory','employment','advisory','compliance'
  )
);

ALTER TABLE collections DROP CONSTRAINT IF EXISTS collections_status_chk;
ALTER TABLE collections ADD  CONSTRAINT collections_status_chk CHECK (
  status IN ('active','on_hold','closed','archived','legal_hold')
);

-- ── RLS on the new tables (per-user for now; F2 owns the firm-role policy) ─────────────────
-- A user sees firms they belong to, and their own memberships. `collections` RLS is UNCHANGED
-- (still `user_id = auth.uid()` from 002) — F1a does not alter who can see a vault.
ALTER TABLE firms            ENABLE ROW LEVEL SECURITY;
ALTER TABLE firm_memberships ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users see firms they belong to" ON firms;
CREATE POLICY "Users see firms they belong to" ON firms
  FOR SELECT USING (
    id IN (SELECT firm_id FROM firm_memberships WHERE user_id = auth.uid())
  );

DROP POLICY IF EXISTS "Users see own memberships" ON firm_memberships;
CREATE POLICY "Users see own memberships" ON firm_memberships
  FOR ALL USING (user_id = auth.uid());

-- ── indexes ───────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_firm_memberships_user ON firm_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_collections_firm      ON collections(firm_id);
-- matter_kind drives F1c's per-kind template + flagship routing; index the lookup.
CREATE INDEX IF NOT EXISTS idx_collections_kind      ON collections(firm_id, matter_kind);
