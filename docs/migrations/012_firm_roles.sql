-- 012_firm_roles.sql
-- F2a (plans/F2_FIRM_CONSOLE_PLAN.md §F2a): FIRM POPULATION + ROLES + INVITES.
--
-- WHY this slice: migration 010 created the `firms` + `firm_memberships` tables but NOTHING
-- writes to them — signup has no firm logic, there is no `role` column, no authorization.
-- F2a is the DATA-MODEL + POPULATION spine only — NO enforcement yet (that is F2b's
-- `authorize()`/`require_cap`). It adds the role axis, the invite table, and a backfill so
-- every existing user becomes a Managing Partner of their own solo firm.
--
-- PRIME RULE (inherited): solo-firm default = BYTE-IDENTICAL. A firm-of-one whose sole member
-- is a Managing Partner has exactly the powers it has today. The `role` column DEFAULTS to
-- 'managing_partner' so an existing/backfilled membership keeps full power (legacy parity);
-- F2 only starts *subtracting* power when a second, lesser-role member is added.
--
-- THREAT COVERAGE (F2a, §0.7): T4 (invite abuse) — invites are hash-stored, single-use,
-- expiring, email-bound; T3 (cross-firm) — every write resolves the firm server-side, never
-- from a request body. The enforcement of those properties is in db.py/routes; this migration
-- gives them the storage shape (token_hash not token; accepted_at for single-use; expires_at).
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS / idempotent
-- backfill throughout). Jeel applies migrations himself (F1 convention) — get consent first.

-- ── the 9 roles (lockstep with src/api/schemas.py ROLES + src/components/authz.py) ─────────
-- Hierarchy (top→bottom): managing_partner · senior_partner · partner · senior_associate ·
-- associate · paralegal · assistant · client · guest. `client`/`guest` are EXTERNAL (F2b gives
-- them a deny-by-default capability set). Default = managing_partner so legacy rows keep power.
ALTER TABLE firm_memberships ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'managing_partner';

ALTER TABLE firm_memberships DROP CONSTRAINT IF EXISTS firm_memberships_role_chk;
ALTER TABLE firm_memberships ADD  CONSTRAINT firm_memberships_role_chk CHECK (
  role IN (
    'managing_partner','senior_partner','partner','senior_associate',
    'associate','paralegal','assistant','client','guest'
  )
);

-- ── firm_invites — admin-led, invite-only join (D1) ────────────────────────────────────────
-- A `manage_members` holder issues an invite carrying the joiner's email + role. The token is
-- single-use, expiring, HASH-STORED (never the raw token at rest — T4), and EMAIL-BOUND (the
-- accepting user's verified email must equal `email`). The joiner can NEVER pick their own role.
CREATE TABLE IF NOT EXISTS firm_invites (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id     UUID NOT NULL REFERENCES firms(id) ON DELETE CASCADE,
  email       TEXT NOT NULL,
  role        TEXT NOT NULL,
  token_hash  TEXT NOT NULL,                       -- sha256(raw_token); raw is shown ONCE at create
  invited_by  UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  expires_at  TIMESTAMPTZ NOT NULL,
  accepted_at TIMESTAMPTZ,                          -- NULL = still pending; set atomically on accept
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- The invited role must be a valid role too (same closed set).
ALTER TABLE firm_invites DROP CONSTRAINT IF EXISTS firm_invites_role_chk;
ALTER TABLE firm_invites ADD  CONSTRAINT firm_invites_role_chk CHECK (
  role IN (
    'managing_partner','senior_partner','partner','senior_associate',
    'associate','paralegal','assistant','client','guest'
  )
);

CREATE INDEX IF NOT EXISTS idx_firm_invites_firm       ON firm_invites(firm_id);
CREATE INDEX IF NOT EXISTS idx_firm_invites_token_hash ON firm_invites(token_hash);
-- At most ONE active (un-accepted, un-expired) invite per (firm, email). A partial unique index
-- on the pending rows lets a re-invite after expiry/acceptance succeed without a manual cleanup.
DROP INDEX IF EXISTS uq_firm_invites_active;
CREATE UNIQUE INDEX uq_firm_invites_active ON firm_invites(firm_id, lower(email))
  WHERE accepted_at IS NULL;

-- ── RLS on firm_invites (visible to manage_members holders of the firm, or the addressed email) ──
-- F2a has no role-aware policy engine yet (that's F2b/F2f), and the live request client is
-- service-role (RLS-exempt) so app-layer scoping is the real guard. We still set a sane SELECT
-- policy so the RLS read-client (defense-in-depth) can't fan out invites cross-firm: a user sees
-- invites of firms they belong to, or invites addressed to their own verified email.
ALTER TABLE firm_invites ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Invites visible to firm members or the invitee" ON firm_invites;
CREATE POLICY "Invites visible to firm members or the invitee" ON firm_invites
  FOR SELECT USING (
    firm_id IN (SELECT firm_id FROM firm_memberships WHERE user_id = auth.uid())
    OR lower(email) = lower((SELECT email FROM auth.users WHERE id = auth.uid()))
  );

-- ── BACKFILL — every firm-less user → a solo firm + Managing-Partner membership (idempotent) ──
-- Legacy parity: a Gmail login today has no firm → get_user_firm()={} → vaults firm_id=NULL.
-- This gives each such user their own firm and stamps their existing collections with it. The
-- WHERE NOT EXISTS guards make the whole block re-runnable with no duplicate firms/memberships.
DO $$
DECLARE
  u RECORD;
  new_firm_id UUID;
BEGIN
  FOR u IN
    SELECT au.id AS user_id, au.email
    FROM auth.users au
    WHERE NOT EXISTS (
      SELECT 1 FROM firm_memberships fm WHERE fm.user_id = au.id
    )
  LOOP
    -- one firm per legacy user; name it after their email local-part for legibility
    INSERT INTO firms (name)
    VALUES (COALESCE(split_part(u.email, '@', 1), 'My Firm') || '''s Firm')
    RETURNING id INTO new_firm_id;

    INSERT INTO firm_memberships (user_id, firm_id, role)
    VALUES (u.user_id, new_firm_id, 'managing_partner');

    -- stamp their existing untyped vaults with the new firm (only the firm-less ones)
    UPDATE collections
    SET firm_id = new_firm_id
    WHERE user_id = u.user_id AND firm_id IS NULL;
  END LOOP;
END $$;
