-- 015_matter_team_and_review.sql
-- F2e (plans/F2_FIRM_CONSOLE_PLAN.md §F2e — D0/D3/D5): MATTER STAFFING + the REVIEW CHAIN of
-- command — THE PRODUCTIVITY ENGINE. (Delegation + the abstain-override moment shipped in F2d on
-- migration 014; this migration is ONLY matter-staffing + the review chain. Named 015 because 014
-- is taken by delegations.)
--
-- WHY this slice: the product exists to make EVERYONE at the firm productive — juniors and
-- paralegals included (D0, the prime principle). Security controls WHAT LEAVES the firm and WHICH
-- MATTER you are on; it NEVER blocks a team member from doing the work. F2e is the slice that makes
-- the product *increase* productivity for the whole firm: a senior STAFFS their team onto a matter
-- (D3) → everyone staffed gets the FULL working toolkit ON THAT MATTER (explicitly NOT read-only) →
-- finished work flows UP a chain of command (D5) for review and, at the end, external release.
--
-- WHAT this migration is: the STORAGE shape for two things.
--   (1) matter_memberships — who is on a matter (vault). Being on a matter is what grants ACCESS to
--       that matter; the member's ROLE already grants the full working toolkit (ROLE_CAPS, F2b —
--       paralegals/assistants are never read-only). The matter boundary + the ethical wall (F2c)
--       are the only controls; this row is the productivity grant, not a restriction.
--   (2) review_requests + matter_review_config — the review chain of command. A review_request is a
--       state machine per piece of reviewable work; `current_owner` is ALWAYS set in a non-terminal
--       state (the anti-stall invariant — the #1 documented review failure is "work stalls because
--       no one owns the next step"). The chain routes UP by role rank among the matter's members by
--       default (zero setup), or follows matter_review_config.chain when a matter lead customizes it.
--
-- THREAT COVERAGE (§0.7): T2/T3 (IDOR / cross-firm confused-deputy) — `firm_id` + `vault_id` are
-- resolved SERVER-SIDE by the caller and the row is scoped to them; a body-supplied firm_id NEVER
-- reaches a write (the vault is asserted in the caller's firm via db.collection_in_firm). T7 (stale
-- capability) — matter-membership + review ownership are resolved per-request; a removed member
-- loses access on the NEXT request. T10 (audit gap) — every staffing add/remove and every review
-- transition is log_audit'd by the route (the row F3 later hash-chains).
--
-- PRIME RULE (inherited): flag-off / pre-F2e byte-identical. No matter has a team row or a review
-- request until a senior staffs one / a junior sends for review, so EMPTY tables ⇒ resolve_membership
-- adds an empty matter_member set ⇒ authorize()/the routes behave EXACTLY as F2d shipped them. The
-- code degrades gracefully when these tables are absent (the migration is not applied live — Jeel
-- applies migrations himself; an absent table ⇒ empty result, never a 500 — exactly like 013/014).
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS / idempotent
-- policies throughout). Jeel applies migrations himself (F1 convention) — get consent first.


-- ── matter_memberships — who is STAFFED on a matter (vault) — D3 ─────────────────────────────
-- One row = "user_id is on the team for vault_id" → they get the FULL toolkit on that matter (D0).
-- firm_id scopes the row to the firm that owns the matter (NEVER trusted from a request body — the
-- caller resolves it server-side, and the vault is asserted to belong to that firm before a write).
-- added_by = the senior (partner / senior associate on the matter) who staffed them. Removing a row
-- is an INSTANT revoke (the member loses matter access on their next request — T7).
CREATE TABLE IF NOT EXISTS matter_memberships (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id     UUID NOT NULL REFERENCES firms(id)       ON DELETE CASCADE,
  vault_id    UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,   -- the matter (vault)
  user_id     UUID NOT NULL REFERENCES auth.users(id)  ON DELETE CASCADE,   -- the staffed member
  added_by    UUID NOT NULL REFERENCES auth.users(id)  ON DELETE CASCADE,   -- the senior who staffed
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_matter_team_firm  ON matter_memberships(firm_id);
-- The hot path: "which matters is THIS user staffed on?" (resolve_membership, per request) and
-- "who is on THIS matter?" (the team list + the default chain-builder). Index both axes.
CREATE INDEX IF NOT EXISTS idx_matter_team_user  ON matter_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_matter_team_vault ON matter_memberships(vault_id);
-- At most ONE staffing row per (vault, user) — a re-staff is idempotent, not a duplicate row.
DROP INDEX IF EXISTS uq_matter_team;
CREATE UNIQUE INDEX uq_matter_team ON matter_memberships(vault_id, user_id);

-- ── RLS on matter_memberships — visible to the firm's members ─────────────────────────────────
-- Like screens (013) / delegations (014): the live request client is service-role (RLS-exempt) so
-- the app-layer guard (require_cap + server-resolved firm/vault) is the real control; we still set a
-- sane SELECT policy so the RLS read-client (defense-in-depth) can't fan a firm's team rows out
-- cross-firm. A user may SELECT a matter-team row of a firm they are a member of.
ALTER TABLE matter_memberships ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Matter team visible to firm members" ON matter_memberships;
CREATE POLICY "Matter team visible to firm members" ON matter_memberships
  FOR SELECT USING (
    firm_id IN (
      SELECT firm_id FROM firm_memberships WHERE user_id = auth.uid()
    )
  );


-- ── review_requests — a piece of work sent UP the review chain (the state machine) — D5 ───────
-- status ∈ {pending, approved, changes_requested, released}.
--   pending            — in review; current_owner is the reviewer who owns the next step.
--   changes_requested  — bounced back to the submitter (current_owner = submitted_by) to revise.
--   approved           — the chain cleared internal review; current_owner = the partner who may
--                        release_external at the end (still a non-terminal owner — anti-stall).
--   released           — terminal: the work left the firm (partner released externally). decided_at set.
-- current_owner = WHO OWNS THE NEXT STEP. It is ALWAYS set in any non-terminal state (the anti-stall
-- invariant — nothing is ever ownerless). chain = the ordered reviewer list (user_ids) the request
-- routes through; the builder fills it at submit time (rank-default or the matter's custom chain).
-- firm_id / vault_id scope the request to the matter; artifact_ref identifies WHAT is under review.
CREATE TABLE IF NOT EXISTS review_requests (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id       UUID NOT NULL REFERENCES firms(id)       ON DELETE CASCADE,
  vault_id      UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,   -- the matter
  artifact_ref  TEXT NOT NULL,                                                 -- what is under review
  submitted_by  UUID NOT NULL REFERENCES auth.users(id)  ON DELETE CASCADE,   -- who sent it up
  status        TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','approved','changes_requested','released')),
  current_owner UUID NOT NULL REFERENCES auth.users(id)  ON DELETE CASCADE,   -- who owns the next step
  chain         JSONB NOT NULL DEFAULT '[]'::jsonb,                            -- ordered reviewer list
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  decided_at    TIMESTAMPTZ                                                    -- set on release (terminal)
);

CREATE INDEX IF NOT EXISTS idx_review_firm  ON review_requests(firm_id);
CREATE INDEX IF NOT EXISTS idx_review_vault ON review_requests(vault_id);
-- The hot path: "my review queue" = the open requests I currently OWN. Index the open-by-owner rows.
CREATE INDEX IF NOT EXISTS idx_review_owner_open ON review_requests(current_owner)
  WHERE status IN ('pending','approved','changes_requested');

-- ── RLS on review_requests — visible to the firm's members ────────────────────────────────────
-- Same posture as the team table: app-layer guard is the real control; the SELECT policy keeps the
-- RLS read-client from fanning a firm's review requests cross-firm.
ALTER TABLE review_requests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Review requests visible to firm members" ON review_requests;
CREATE POLICY "Review requests visible to firm members" ON review_requests
  FOR SELECT USING (
    firm_id IN (
      SELECT firm_id FROM firm_memberships WHERE user_id = auth.uid()
    )
  );


-- ── matter_review_config — optional per-matter CUSTOM chain — D5 ──────────────────────────────
-- One row per matter that has a customized chain. chain = an ordered JSONB list of reviewer user_ids
-- (the matter lead's defined order, e.g. Associate → Senior Associate → Compliance → Partner). NULL
-- / no row ⇒ use the rank-based default (the chain-builder falls back). vault_id is the PK (one
-- config per matter). firm_id scopes it; set_by attributes who customized it.
CREATE TABLE IF NOT EXISTS matter_review_config (
  vault_id    UUID PRIMARY KEY REFERENCES collections(id) ON DELETE CASCADE,
  firm_id     UUID NOT NULL REFERENCES firms(id)          ON DELETE CASCADE,
  chain       JSONB,                                       -- NULL ⇒ rank-based default
  set_by      UUID REFERENCES auth.users(id)              ON DELETE SET NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_config_firm ON matter_review_config(firm_id);

ALTER TABLE matter_review_config ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Review config visible to firm members" ON matter_review_config;
CREATE POLICY "Review config visible to firm members" ON matter_review_config
  FOR SELECT USING (
    firm_id IN (
      SELECT firm_id FROM firm_memberships WHERE user_id = auth.uid()
    )
  );
