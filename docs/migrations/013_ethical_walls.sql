-- 013_ethical_walls.sql
-- F2c (plans/F2_FIRM_CONSOLE_PLAN.md §F2c): ETHICAL WALLS (conflict screens) — THE REGULATORY MOAT.
--
-- WHY this slice: a law firm has a professional + statutory DUTY to "screen off" a lawyer from a
-- matter when that lawyer (or their prior client) is adverse to the matter's client (a conflict of
-- interest). ABA Op. 512 requires supervising lawyers to understand "the vendor's conflicts-check
-- system … to screen for adversity among firm clients"; Indian Evidence Act §126/§127 extends
-- privilege to "agents of attorneys" (DocQuery is one) — so a wall that leaks is not a bug, it is a
-- breach of a legal duty transferred to us. The wall MUST be enforced in the DATA layer, never the
-- prompt (the F1 isolation lesson, re-applied with the same path-by-path rigor).
--
-- WHAT this migration is: the STORAGE shape for a screen — one row = "this member is walled off
-- from this matter (vault)". F2b's authorize() already checks the screen FIRST in its deny-overrides
-- precedence (so a screen beats ANY role grant, even a Managing Partner's); this migration + the
-- db.py writers + the retrieval-layer enforcement (P1–P9) is what POPULATES and ENFORCES it.
--
-- AUDITABLE + REVERSIBLE: a screen is timestamped (created_at), attributed (created_by + reason),
-- and SOFT-removable (removed_at IS NULL = active; removing sets removed_at, never deletes the row)
-- — a regulator can see the full history of when a wall went up and came down. Removing a screen
-- restores access on the NEXT request (caps + screens are resolved per-request — T7).
--
-- THREAT COVERAGE (§0.7): T5 (ethical-wall bypass) — the wall is the whole point of this slice;
-- T2/T3 (IDOR / cross-firm confused-deputy) — `firm_id` is resolved SERVER-SIDE by the caller and
-- the row is scoped to it; a body-supplied firm_id NEVER reaches a write; T10 (audit gap) — every
-- screen add/remove is log_audit'd by the route.
--
-- PRIME RULE (inherited): flag-off / pre-F2c byte-identical. No firm ever has a screen until a
-- manage_members holder creates one, so an empty `screens` table ⇒ screened_vault_ids is always the
-- empty set ⇒ authorize()/resolve_membership behave EXACTLY as F2b shipped them.
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS / idempotent
-- policies throughout). Jeel applies migrations himself (F1 convention) — get consent first.

-- ── screens — one active row = a member walled off from a matter (vault) ─────────────────────
-- (user_id, vault_id) is the conflict pair; firm_id scopes the row to the firm that owns the wall
-- (NEVER trusted from a request body — the caller resolves it server-side). reason is REQUIRED
-- (a wall must be justifiable to a regulator). created_by is the manage_members holder who raised
-- it. removed_at distinguishes an active wall (NULL) from a lifted one (timestamp) — soft removal
-- keeps the full audit history. vault_id references collections (the vault = an existing collection,
-- per [[unified-features-plan-and-codebase-anchors]]).
CREATE TABLE IF NOT EXISTS screens (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id     UUID NOT NULL REFERENCES firms(id)        ON DELETE CASCADE,
  user_id     UUID NOT NULL REFERENCES auth.users(id)   ON DELETE CASCADE,   -- the walled-off member
  vault_id    UUID NOT NULL REFERENCES collections(id)  ON DELETE CASCADE,   -- the matter they can't touch
  reason      TEXT NOT NULL,                                                  -- required justification
  created_by  UUID NOT NULL REFERENCES auth.users(id)   ON DELETE CASCADE,   -- who raised the wall
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  removed_at  TIMESTAMPTZ                                                     -- NULL = active; set on soft-remove
);

CREATE INDEX IF NOT EXISTS idx_screens_firm        ON screens(firm_id);
-- The hot path: "is THIS user screened off ANY vault?" (resolve_membership, per request) and
-- "is this user screened off THIS vault?" (the retrieval-layer guard). Index the active rows.
CREATE INDEX IF NOT EXISTS idx_screens_user_active ON screens(user_id)  WHERE removed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_screens_vault_active ON screens(vault_id) WHERE removed_at IS NULL;
-- At most ONE active screen per (firm, user, vault) — a re-screen after a soft-remove succeeds
-- (the removed row is excluded), and a duplicate active screen is a clean conflict, not a dup row.
DROP INDEX IF EXISTS uq_screens_active;
CREATE UNIQUE INDEX uq_screens_active ON screens(firm_id, user_id, vault_id)
  WHERE removed_at IS NULL;

-- ── RLS on screens — visible to manage_members holders of the firm ───────────────────────────
-- Like firm_invites (012): F2c has no role-aware policy engine in the DB yet (that is F2f), and the
-- live request client is service-role (RLS-exempt) so app-layer scoping (the route's require_cap +
-- server-resolved firm) is the real guard. We still set a sane SELECT policy so the RLS read-client
-- (defense-in-depth) can't fan a firm's screens out cross-firm: a user may SELECT screens of a firm
-- in which they hold the partner tier (manage_members). We approximate "manage_members holder" at
-- the row layer as the partner roles (the only roles ROLE_CAPS grants manage_members); the precise
-- cap check stays in authz.authorize at the app layer. A screened member must NOT be able to read
-- the screen that walls them — so the subject (user_id = auth.uid()) is deliberately NOT granted SELECT.
ALTER TABLE screens ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Screens visible to firm managers" ON screens;
CREATE POLICY "Screens visible to firm managers" ON screens
  FOR SELECT USING (
    firm_id IN (
      SELECT firm_id FROM firm_memberships
      WHERE user_id = auth.uid()
        AND role IN ('managing_partner','senior_partner','partner')
    )
  );
