-- 014_delegations.sql
-- F2d (plans/F2_FIRM_CONSOLE_PLAN.md §F2d + §F2e/D6): AUTHORITY DELEGATION — a senior delegates a
-- bounded, time-boxed, revocable set of verbs to a delegate (the canonical case: a partner lets a
-- Personal Assistant triage their queue / act on their behalf — D6). The SENIOR stays accountable.
--
-- WHY this slice: real firms attach a PA to a senior; the PA must be able to act for them WITHOUT
-- being handed the senior's full role (least privilege). A delegation is a POSITIVE grant in
-- authorize()'s precedence (step 1.5) — but it STILL cannot beat a screen DENY (the ethical wall
-- wins; precedence holds, asserted in the gate), and a delegate can NEVER be granted a verb the
-- delegator does not themselves hold (bounded at the resolver — db.active_delegated_verbs).
--
-- WHAT this migration is: the STORAGE shape for a delegation. One active row = "delegate_id may
-- exercise `verbs` on behalf of delegator_id, within firm_id, until expires_at (unless revoked)."
-- The narrower per-matter PA staffing + the review-chain tables are F2e (migration folded there);
-- F2d only needs the firm-level delegation grant + the abstain-override audit seam (audit_log,
-- already live — no new table; the override writes the F3 hash-chain contract row there).
--
-- TIME-BOXED + REVOCABLE + AUDITABLE (T7): `expires_at` bounds the grant; `revoked_at` lifts it
-- early (soft — the history is preserved); `active = revoked_at IS NULL AND expires_at > now()`.
-- A revoked/expired delegation stops granting on the delegate's NEXT request (caps + delegations
-- are resolved per-request — T7; never cached in the JWT). Every grant/revoke is audit-logged by
-- the route (T10).
--
-- THREAT COVERAGE (§0.7): T1 (privilege escalation) — a delegate cannot gain a verb the delegator
-- lacks (bounded at the resolver) NOR cross a screen (precedence); T3 (cross-firm) — firm_id is
-- resolved SERVER-SIDE by the caller and the row is scoped to it, a body firm_id never reaches a
-- write; T6 (override bypass) — the override verb can be HELD via a delegation but is still
-- screen-beaten + reason-required + audited; T10 (audit gap) — grant/revoke/override all logged.
--
-- PRIME RULE (inherited): flag-off / pre-F2d byte-identical. No firm has a delegation until a
-- senior creates one, so an empty `delegations` table ⇒ active_delegated_verbs is always the empty
-- set ⇒ resolve_membership/authorize behave EXACTLY as F2c shipped them (delegated_verbs = ∅).
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS / idempotent
-- policies throughout). Jeel applies migrations himself (F1 convention) — get consent first.

-- ── delegations — one active row = a bounded, time-boxed grant from a senior to a delegate ──────
-- delegator_id = the accountable senior; delegate_id = who acts for them (the PA). verbs is the
-- bounded capability subset (a TEXT[] of CAPABILITIES names — the resolver intersects it with the
-- delegator's OWN caps so a delegate can never exceed the delegator). expires_at bounds it;
-- revoked_at lifts it early (soft-remove keeps the audit trail). firm_id scopes the row to the
-- firm that owns the grant (NEVER trusted from a request body — resolved server-side).
CREATE TABLE IF NOT EXISTS delegations (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id       UUID NOT NULL REFERENCES firms(id)      ON DELETE CASCADE,
  delegator_id  UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,   -- the accountable senior
  delegate_id   UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,   -- who acts for them (the PA)
  verbs         TEXT[] NOT NULL DEFAULT '{}',                                 -- bounded capability subset
  expires_at    TIMESTAMPTZ NOT NULL,                                         -- time-boxed (required)
  revoked_at    TIMESTAMPTZ,                                                  -- NULL = not revoked; set on revoke
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_delegations_firm     ON delegations(firm_id);
-- The hot path: "what verbs does THIS delegate currently hold, by delegation?" (resolve_membership,
-- per request). Index the active-by-delegate rows (an active filter is applied in the query).
CREATE INDEX IF NOT EXISTS idx_delegations_delegate ON delegations(delegate_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_delegations_delegator ON delegations(delegator_id) WHERE revoked_at IS NULL;

-- ── RLS on delegations — visible to the firm's managers + the two parties to the grant ──────────
-- Like screens (013) / firm_invites (012): the live request client is service-role (RLS-exempt) so
-- the app-layer guard (require_cap + server-resolved firm) is the real control; we still set a sane
-- SELECT policy so the RLS read-client (defense-in-depth) can't fan a firm's delegations out
-- cross-firm. A user may SELECT a delegation row if they manage the firm (partner tier) OR they are
-- the delegator or the delegate on that row (both parties may see their own grant).
ALTER TABLE delegations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Delegations visible to firm managers and parties" ON delegations;
CREATE POLICY "Delegations visible to firm managers and parties" ON delegations
  FOR SELECT USING (
    delegator_id = auth.uid()
    OR delegate_id = auth.uid()
    OR firm_id IN (
      SELECT firm_id FROM firm_memberships
      WHERE user_id = auth.uid()
        AND role IN ('managing_partner','senior_partner','partner')
    )
  );
