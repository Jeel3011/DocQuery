-- 016_firm_rls.sql
-- F2f (plans/F2_FIRM_CONSOLE_PLAN.md §F2f + §0.7 T2/T3/T5/T9; plans/F2_ARCHITECTURE.md §1/§6):
-- THE FIRM-SCOPED ROW-LAYER BACKSTOP — the third defense layer, UNDER the central guard
-- (require_cap → authorize) and the retrieval/read wall (assert_vault_not_screened, P1–P7).
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- WHY THIS SLICE EXISTS (T9): every prior F2 slice (a–e) is enforced in APPLICATION code —
-- require_cap 403s before a mutating handler, assert_vault_not_screened returns 0 chunks on a
-- walled vault. Those are load-bearing, but they are CONVENTIONS a call site must remember. If a
-- future endpoint forgets a `.eq(user_id)` filter, or a path resolves a vault_id without the wall
-- check, the app layer leaks. F2f puts a guard at the ROW so that a forgotten app-layer check
-- can't cross a firm boundary or a wall on the RLS-enforced read path. Defense in depth: the row
-- floor is the property the data layer enforces, not something a call site has to get right.
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- WHAT THIS MIGRATION CHANGES (and what it deliberately does NOT):
--   • collections SELECT — replaces F1/002's per-USER policy (`user_id = auth.uid()`) with a
--     firm-MEMBERSHIP + ethical-wall + legacy-owner policy. THIS IS THE ONE BEHAVIOR-CHANGING
--     POLICY in the slice (see the NON-REGRESSION section — it is dormant on today's data).
--   • collections write (INSERT/UPDATE/DELETE) — adds an explicit WITH CHECK so a cross-firm row
--     can't be written on the RLS-enforced path (T3). (The app's normal writes are service-role /
--     RLS-EXEMPT — see the T9 HONESTY note — so app-layer require_cap stays the load-bearing
--     write guard; this WITH CHECK is the backstop for the read-client path, a future
--     authenticated-role write, and direct SQL.)
--   • documents / conversations / messages / audit_log — these tables have NO firm_id column
--     (verified against the live schema 2026-06-25 — see the SHARED-TABLES section). They CANNOT
--     carry a firm-JOIN without a schema change, and firm-level sharing of them is an APP-LAYER /
--     F6 (Firm Brain) concern, not a row-RLS one. F2f therefore KEEPS them per-USER
--     (`auth.uid() = user_id`) as the row floor and only RE-ASSERTS that floor idempotently with
--     an explicit WITH CHECK on writes. Their visibility is UNCHANGED.
--
-- DELIBERATELY NOT IN THE ROW POLICY: matter-membership. Gating a row READ on matter_memberships
-- would lock a Managing Partner out of their own firm's UN-STAFFED vault at the row layer
-- (over-restrictive — it breaks the "an MP/partner sees the whole firm" model). Matter-membership
-- gates matter WORK at the app layer (resolve_membership.matter_vault_ids + the route caps); the
-- ROW floor is correctly the FIRM boundary + the WALL. (Raised in §F2f; this is the resolved
-- decision.) The screen (wall) IS in the row policy because a wall must beat even a firm member's
-- own-firm visibility — deny-overrides, at the row.
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- THREAT COVERAGE (§0.7):
--   T2 (IDOR / object-reference) — a non-member's SELECT of a firm's vault returns 0 rows at the
--       row layer EVEN IF the app `.eq(user_id)` filter is dropped (the live verifier proves this).
--   T3 (cross-firm confused-deputy) — the membership JOIN scopes reads to the caller's firm(s); the
--       WITH CHECK blocks a write that would place a row in another firm on the RLS-enforced path.
--   T5 (ethical-wall bypass) — `NOT EXISTS (active screen on this user+vault)` makes the wall hold
--       at the ROW: a screened member sees 0 rows for the walled vault, beating ANY role (an MP
--       included) — deny-overrides at the row layer, mirroring authorize()'s step-1 screen DENY.
--   T9 (write-path leak) — the WITH CHECK is the row backstop UNDER the app-layer write guard; the
--       app-layer require_cap + server-resolved firm remain load-bearing on service-role writes.
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- ⚠️ NON-REGRESSION — THE INVARIANT THAT MUST HOLD (legacy / solo-MP parity):
-- Unlike F2a–F2e (byte-identical on EMPTY tables), F2f CHANGES the collections row policy, so the
-- proof is about DATA, not emptiness. Verified live (project dwcfcfgdefwddzhazmbm, 2026-06-25):
--   • all collections have firm_id set (012 backfill ran; 0 NULL) → the firm-JOIN resolves them;
--   • every firm has exactly ONE member (all solo firms) → for every collection, the count of
--     OTHER members of its firm is 0 → switching per-user → firm-JOIN changes the visible-row set
--     for ZERO existing users ("my firm's vaults" == "my vaults" while firms are solo).
-- The behavior change is therefore DORMANT until a real multi-member firm exists — exactly when
-- firm-scoping is DESIRED. The legacy fallback `firm_id IS NULL → user_id = auth.uid()` keeps any
-- future un-stamped row visible to its owner so a firm-less/solo user is NEVER orphaned.
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- ⚠️ T9 HONESTY — DOES THE WITH CHECK ACTUALLY FIRE ON THE APP'S WRITES? Mostly NO, by design.
-- The live request client used for WRITES is the SERVICE-ROLE key (dependencies.py:195), which is
-- RLS-EXEMPT — so RLS (SELECT policy AND WITH CHECK) does NOT run on the app's normal mutations.
-- The RLS-enforced client is `sb.read_client` (F1 hardening) and is used for READS only. So:
--   • The collections SELECT policy IS the live backstop on every user-facing READ (read_client).
--   • The WITH CHECK is a backstop ONLY for: the read_client path, any FUTURE authenticated-role
--     write client, and direct SQL / psql. It is correct and worth having (defense in depth), but
--     it is NOT what guards today's writes. The LOAD-BEARING write guard remains the app layer:
--     require_cap(verb) + the server-resolved firm_id + the `.eq(user_id)` filter. We state this
--     plainly rather than imply RLS secures the write path when it does not.
--
-- ────────────────────────────────────────────────────────────────────────────────────────────
-- ⚠️ APPLY / ROLLBACK — Jeel applies migrations himself (F1 convention). Because this one is
-- BEHAVIOR-CHANGING on apply (it swaps an active policy, not just adds dormant tables), GET
-- EXPLICIT CONSENT first, and know the rollback:
--   ROLLBACK = restore the per-user collections policy (002):
--     DROP POLICY IF EXISTS "Firm members see firm vaults (wall-aware)" ON collections;
--     DROP POLICY IF EXISTS "Firm members write firm vaults (wall-aware)" ON collections;
--     CREATE POLICY "Users see own collections" ON collections FOR ALL USING (user_id = auth.uid());
--   The helper functions and the documents/conversations/messages/audit_log re-asserted per-user
--   policies are visibility-neutral, so they need no rollback (drop them only if desired).
-- No app-code change is required to apply this: read_client already runs RLS-enforced (F1), and
-- resolve_membership already resolves firm + screens server-side — the row policy simply mirrors,
-- at the row, the decision authorize() already makes at the app layer.
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (idempotent throughout:
-- CREATE OR REPLACE FUNCTION, DROP POLICY IF EXISTS before each CREATE).


-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 1. SECURITY DEFINER helpers — break the RLS recursion + cache the auth lookup.
-- ════════════════════════════════════════════════════════════════════════════════════════════
-- The collections policy must read firm_memberships and screens. Both of THOSE tables have their
-- own RLS (012/013), so a bare subquery from the collections policy would be evaluated under the
-- caller's RLS on the subtable — recursive and fragile (and firm_memberships' own policy is
-- `user_id = auth.uid()`, which is fine here but couples two policies). Wrapping the lookups in
-- SECURITY DEFINER functions (owned by the table owner, which bypasses RLS INSIDE the function)
-- makes the membership/screen check authoritative and non-recursive. We mark them STABLE (no
-- writes, same result within a statement) so the planner can cache them, and we pin search_path to
-- avoid a hijacked-schema attack on a SECURITY DEFINER function (a standard hardening).
--
-- Each takes the user_id explicitly (the policy passes `(SELECT auth.uid())`) so the function is
-- reusable and testable, and so the policy — not the function — owns the `auth.uid()` call (which
-- the planner caches once per statement via the scalar-subquery idiom).

-- Is `p_user` a member of the firm that owns this vault? (firm-boundary check, T2/T3)
CREATE OR REPLACE FUNCTION public.f2f_user_in_firm(p_user UUID, p_firm UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM firm_memberships fm
    WHERE fm.user_id = p_user
      AND fm.firm_id = p_firm
  );
$$;

-- Is there an ACTIVE ethical wall (screen) on this user for this vault? (wall check, T5)
-- removed_at IS NULL = active. A screen here means "deny at the row", beating firm membership.
CREATE OR REPLACE FUNCTION public.f2f_vault_screened(p_user UUID, p_vault UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM screens s
    WHERE s.user_id  = p_user
      AND s.vault_id = p_vault
      AND s.removed_at IS NULL
  );
$$;

-- The composite row predicate, expressed once so SELECT-USING and write-WITH-CHECK stay in lockstep
-- (a wall/firm rule that protects reads but not writes is the classic asymmetry bug). A row is
-- visible/writable to p_user iff:
--   (firm-member of the row's firm  AND  not walled off it)        -- the firm + wall floor
--   OR (row has no firm yet AND p_user owns it)                    -- legacy/solo NULL-firm fallback
-- Deny-by-default falls out: a row that matches neither branch (other firm, or someone else's
-- legacy row) yields FALSE → invisible / un-writable. NULL firm_id with a non-owner → FALSE.
CREATE OR REPLACE FUNCTION public.f2f_can_access_vault(p_user UUID, p_firm UUID, p_owner UUID, p_vault UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    CASE
      WHEN p_firm IS NOT NULL THEN
        public.f2f_user_in_firm(p_user, p_firm)
        AND NOT public.f2f_vault_screened(p_user, p_vault)
      ELSE
        -- legacy / solo: no firm stamped yet → the owner (and only the owner) sees it.
        p_owner = p_user
    END;
$$;


-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 2. collections — the firm + wall + legacy-owner row policy (REPLACES the per-user 002 policy).
-- ════════════════════════════════════════════════════════════════════════════════════════════
-- We split the old single `FOR ALL` into an explicit SELECT (read floor — the live backstop on
-- read_client) and a write policy (INSERT/UPDATE/DELETE) carrying the WITH CHECK (T3 row backstop).
-- (`auth.uid()` is wrapped as `(SELECT auth.uid())` so Postgres caches it once per statement —
-- the documented Supabase RLS performance idiom — instead of re-evaluating per row.)

ALTER TABLE collections ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users see own collections"                      ON collections;  -- the 002 policy
DROP POLICY IF EXISTS "Firm members see firm vaults (wall-aware)"      ON collections;
DROP POLICY IF EXISTS "Firm members write firm vaults (wall-aware)"    ON collections;

-- READ: a firm member sees their firm's vaults, minus any vault they are walled off; a legacy
-- NULL-firm row stays visible to its owner. Deny-by-default for everything else.
CREATE POLICY "Firm members see firm vaults (wall-aware)" ON collections
  FOR SELECT
  USING (
    public.f2f_can_access_vault((SELECT auth.uid()), firm_id, user_id, id)
  );

-- WRITE: same predicate for both the rows you may act on (USING) and the row you may write
-- (WITH CHECK — blocks placing/moving a row into a firm you're not a member of, or a walled vault).
-- On a service-role (RLS-exempt) connection this does not fire — app-layer require_cap is the
-- load-bearing write guard (T9); this is the row backstop on the RLS-enforced path.
CREATE POLICY "Firm members write firm vaults (wall-aware)" ON collections
  FOR ALL
  USING (
    public.f2f_can_access_vault((SELECT auth.uid()), firm_id, user_id, id)
  )
  WITH CHECK (
    public.f2f_can_access_vault((SELECT auth.uid()), firm_id, user_id, id)
  );

-- Index support: the membership JOIN keys on collections.firm_id (idx_collections_firm exists from
-- 010); the screen lookups key on screens(user_id)/(vault_id) WHERE removed_at IS NULL (the partial
-- indexes idx_screens_user_active / idx_screens_vault_active exist from 013); firm_memberships keys
-- on (user_id) (idx_firm_memberships_user from 010). Add the composite that f2f_user_in_firm hits
-- on (user_id, firm_id) so the EXISTS is an index-only probe, not a scan of the user's memberships.
CREATE INDEX IF NOT EXISTS idx_firm_memberships_user_firm
  ON firm_memberships(user_id, firm_id);
-- And the screen membership probe by exact (user_id, vault_id) among active rows.
CREATE INDEX IF NOT EXISTS idx_screens_user_vault_active
  ON screens(user_id, vault_id) WHERE removed_at IS NULL;


-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 3. collection_documents — keep scoped to collections the caller may access (now wall+firm aware
--    transitively, because the collections subquery it already keys on is now firm/wall-scoped).
-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 002's policy was `collection_id IN (SELECT id FROM collections WHERE user_id = auth.uid())`. With
-- the collections policy now firm+wall-aware, we re-express this so the join-row inherits the SAME
-- floor: a membership row is visible iff its collection is one the caller may access. We use the
-- SECURITY DEFINER predicate against the parent collection's columns (looked up here) so a screened
-- or cross-firm collection's join rows disappear too (no leak via the join table).
DROP POLICY IF EXISTS "Users see own collection_documents"        ON collection_documents;
DROP POLICY IF EXISTS "Firm members see firm collection_documents" ON collection_documents;

CREATE POLICY "Firm members see firm collection_documents" ON collection_documents
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM collections c
      WHERE c.id = collection_documents.collection_id
        AND public.f2f_can_access_vault((SELECT auth.uid()), c.firm_id, c.user_id, c.id)
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM collections c
      WHERE c.id = collection_documents.collection_id
        AND public.f2f_can_access_vault((SELECT auth.uid()), c.firm_id, c.user_id, c.id)
    )
  );


-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 4. SHARED USER-SCOPED TABLES WITHOUT A firm_id — documents / conversations / messages /
--    audit_log — KEEP PER-USER (the row floor); only re-assert it idempotently + add WITH CHECK.
-- ════════════════════════════════════════════════════════════════════════════════════════════
-- DECISION (verified against the live schema 2026-06-25): these tables have a user_id but NO
-- firm_id column. A firm-JOIN is therefore impossible without a schema change:
--   • documents reach a firm only TRANSITIVELY (collection_documents → collections.firm_id), and a
--     document can be in zero or many collections — there is no single authoritative firm for a
--     document row, so a firm-JOIN would be ambiguous (and a doc not yet in any collection would
--     have NO firm at all). The document's firm-scoping is enforced where it matters: at retrieval
--     (the F1 isolation floor scopes by vault), and the per-user RLS here blocks cross-USER reads.
--   • conversations / messages / audit_log have NO vault or firm linkage whatsoever — they are
--     inherently per-user artifacts. Firm-level sharing of a conversation is an APP-LAYER product
--     decision (F6 Firm Brain explicitly excludes privileged content, F1e), not a row-RLS rule.
-- So F2f does NOT firm-scope these; it RE-ASSERTS the per-user floor idempotently and adds an
-- explicit WITH CHECK so a write can't set user_id to someone else on the RLS-enforced path. Their
-- visibility is UNCHANGED (still exactly the owner) — this is a hardening + a consistency
-- statement, not a behavior change. (We do not collapse the existing split per-command policies on
-- documents/conversations/messages; we ensure a single canonical owner policy is present and add
-- WITH CHECK where one was missing, leaving the per-command ones in place — all express the SAME
-- `auth.uid() = user_id` rule, so they compose without widening access.)

-- documents — re-assert a canonical owner FOR ALL policy with WITH CHECK (alongside the existing
-- split _select/_insert/_update/_delete policies; all enforce auth.uid() = user_id).
DROP POLICY IF EXISTS "F2f documents owner only" ON documents;
CREATE POLICY "F2f documents owner only" ON documents
  FOR ALL
  USING       ((SELECT auth.uid()) = user_id)
  WITH CHECK  ((SELECT auth.uid()) = user_id);

-- conversations
DROP POLICY IF EXISTS "F2f conversations owner only" ON conversations;
CREATE POLICY "F2f conversations owner only" ON conversations
  FOR ALL
  USING       ((SELECT auth.uid()) = user_id)
  WITH CHECK  ((SELECT auth.uid()) = user_id);

-- messages
DROP POLICY IF EXISTS "F2f messages owner only" ON messages;
CREATE POLICY "F2f messages owner only" ON messages
  FOR ALL
  USING       ((SELECT auth.uid()) = user_id)
  WITH CHECK  ((SELECT auth.uid()) = user_id);

-- audit_log — KEEP per-user read, but lock writes down at the row layer too. The audit trail is a
-- compliance artifact (DPDP Rule 6 logging; the F3 hash-chain seam) — a user must not be able to
-- forge a row under another user_id. (The app writes audit rows on the service-role path, which is
-- RLS-exempt, so this WITH CHECK is, like the others, the RLS-enforced-path / direct-SQL backstop.)
DROP POLICY IF EXISTS "F2f audit_log owner only" ON audit_log;
CREATE POLICY "F2f audit_log owner only" ON audit_log
  FOR ALL
  USING       ((SELECT auth.uid()) = user_id)
  WITH CHECK  ((SELECT auth.uid()) = user_id);


-- ════════════════════════════════════════════════════════════════════════════════════════════
-- 5. Lock down the helper functions — EXECUTE to the app roles only (least privilege).
-- ════════════════════════════════════════════════════════════════════════════════════════════
-- The SECURITY DEFINER functions run as the owner (RLS-bypassing inside). Restrict who may call
-- them to the Supabase app roles. They are read-only and take explicit args (no privilege beyond
-- "answer a membership/screen question"), but least-privilege is the right default for any
-- SECURITY DEFINER function. (anon can call them too — it harmlessly returns FALSE for a null/
-- unknown user — but we keep the grant tight to authenticated + service_role.)
REVOKE ALL ON FUNCTION public.f2f_user_in_firm(UUID, UUID)            FROM PUBLIC;
REVOKE ALL ON FUNCTION public.f2f_vault_screened(UUID, UUID)         FROM PUBLIC;
REVOKE ALL ON FUNCTION public.f2f_can_access_vault(UUID, UUID, UUID, UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.f2f_user_in_firm(UUID, UUID)            TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.f2f_vault_screened(UUID, UUID)         TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.f2f_can_access_vault(UUID, UUID, UUID, UUID) TO authenticated, service_role;
