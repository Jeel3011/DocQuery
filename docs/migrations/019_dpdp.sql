-- 019_dpdp.sql
-- F2k (plans/F2_FIRM_CONSOLE_PLAN.md §F2k + F2_ARCHITECTURE.md §0.2/§6): DPDP DATA-PRINCIPAL RIGHTS
-- + the named grievance officer + the erasure ledger. DPDP Rules 2025 (notified 13-Nov-2025).
--
-- WHY this slice: under the DPDP Act 2023 + Rules 2025 the FIRM is the Data Fiduciary and DocQuery is
-- the Data Processor — and the compliance liability rests with the Fiduciary EVEN WHEN the Processor
-- does the work. So a firm will demand we make them compliant; that demand is the wedge, but only if we
-- actually deliver the rights machinery: a person can have their data EXPORTED (access, §11), ERASED
-- within 90 days (correction/erasure + consent-withdrawal, §12), and their GRIEVANCE routed to a NAMED
-- officer (§13). This migration is the data model for the named officer + the grievance record + the
-- erasure ledger; the routes (routes/dpdp.py) and the pure core (src/components/dpdp.py) are the rest.
--
-- THE LOAD-BEARING DISTINCTION (documented in code too — src/components/dpdp.py): a §12 ERASURE targets
-- a person's personal CONTENT (their documents / conversations / messages), NOT the immutable records of
-- processing. The audit_log (Rule 6.4/6.5, retained ≥ 1 year) and the F2i `signatures` hash-chain are
-- RECORDS OF PROCESSING / legal evidence — they are RETAINED, never deleted by an erasure request. A
-- signature chain you can break to honor an erasure is worthless (it would also break every OTHER firm
-- member's sign-off in that chain, and destroy the non-repudiation it exists to provide). So erasure is
-- a SOFT-DELETE / tombstone of content, and the `data_erasures` table below is the PROOF the request was
-- honored (who/when/what scope) WITHOUT retaining the erased content — a record of erasure, not of the
-- erased thing. This resolves the classic "right-to-erasure vs immutable-log" conflict explicitly.
--
-- WHAT this migration is:
--   (1) `firms` gains the NAMED GRIEVANCE OFFICER (DPDP §13 + Rule / Privacy-Policy requirement: a
--       data principal must be told WHO to complain to, by name + contact). Nullable, so a firm with no
--       officer set yet is dormant-compatible; a grievance can still be filed and sits unassigned until
--       an officer is named (surfaced to managers). On a firm record (not a config constant) because it
--       is PER-FIRM identity, set + changed by a firm manager, and must be queryable per request.
--   (2) `grievances` — a tracked complaint record (the data principal, the subject, the named officer at
--       filing time, status, the firm). §13 requires a response mechanism + (Rules) a 90-day completion
--       window; the record carries `due_at` (filed + 90d) so the firm can see what is overdue.
--   (3) `data_erasures` — the erasure LEDGER (tombstone). One row per honored §12 erasure: who was
--       erased, by whom, what scope (counts of docs/convos/messages soft-deleted), when. Append-only by
--       posture (the proof of compliance must itself be tamper-resistant) — NO update/delete policy.
--
-- SECURITY / posture (inherited from 013/015/018): the live write client is service-role (RLS-exempt),
-- so the app-layer authz (require_cap("manage_members") for admin-initiated / on-behalf actions; the
-- requester themselves for their OWN export/erase) is the real guard; this migration's RLS is the
-- read-client BACKSTOP only. Firm is ALWAYS resolved server-side (T2/T3 — the body never carries a
-- firm_id). Every rights action is audit-logged (T10) by the route.
--
-- PRIME RULE (inherited): DORMANT-ON-EMPTY ⇒ tables-unapplied / columns-absent = byte-identical to
-- pre-F2k. Every db.py method degrades to None/[]/{}/False on a missing relation or column
-- (db._is_missing_relation) — the export still returns the user's OWN content (which never needed these
-- tables), the erase still soft-deletes content (the tombstone write is best-effort), the grievance
-- route reports "officer not configured" instead of 500-ing. The HONEST framing (recorded in the route +
-- the runbook): the firm OWNS the DPDP duty; we ENABLE it. DPDP is enforceable ~mid-2027 — this ships
-- the machinery now (retrofitting compliance is the expensive path) without blocking anything today.
--
-- Apply via the Supabase SQL editor (or psql). Safe to re-run (IF NOT EXISTS / idempotent policies).
-- Jeel applies migrations himself (F1 convention) — get consent first.

-- ── (1) firms: the NAMED grievance officer (DPDP §13) ──────────────────────────────────────────────
-- A data principal must be told who to complain to. Nullable (a firm may not have named one yet); set
-- by a firm manager (manage_members). `grievance_officer_user_id` links the officer to a firm member
-- when they are one (the common case); name/email are also stored verbatim so an external DPO works too.
ALTER TABLE firms ADD COLUMN IF NOT EXISTS grievance_officer_name    TEXT;
ALTER TABLE firms ADD COLUMN IF NOT EXISTS grievance_officer_email   TEXT;
ALTER TABLE firms ADD COLUMN IF NOT EXISTS grievance_officer_user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- ── (2) grievances — a tracked §13 complaint routed to the named officer ───────────────────────────
-- principal_id = WHO complained (the data principal; a firm member or, later, an external client).
-- subject = the free-text complaint. officer_* = the named officer captured AT FILING TIME (so the
-- record is stable even if the firm later changes its officer). due_at = filed_at + 90 days (the §13
-- completion window made visible). status ∈ {open, acknowledged, resolved, rejected}.
CREATE TABLE IF NOT EXISTS grievances (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id         UUID NOT NULL REFERENCES firms(id)      ON DELETE CASCADE,   -- the per-firm boundary (T3)
  principal_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,  -- WHO complained (never orphan)
  subject         TEXT NOT NULL,                           -- the complaint
  officer_name    TEXT,                                    -- the named officer at filing time (§13)
  officer_email   TEXT,
  officer_user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  status          TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open','acknowledged','resolved','rejected')),
  resolution_note TEXT,                                    -- how it was resolved (filled on close)
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  due_at          TIMESTAMPTZ,                             -- created_at + 90d (the §13 window, made visible)
  resolved_at     TIMESTAMPTZ,
  metadata        JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_grievances_firm        ON grievances(firm_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_grievances_principal   ON grievances(principal_id);
CREATE INDEX IF NOT EXISTS idx_grievances_firm_open   ON grievances(firm_id) WHERE status IN ('open','acknowledged');

-- ── (3) data_erasures — the erasure LEDGER (the §12 tombstone, the PROOF of compliance) ────────────
-- One row per honored erasure. Records WHO was erased, by WHOM, the SCOPE (counts), and WHEN — WITHOUT
-- retaining the erased content. This is the record that lets a firm demonstrate to the Board that a
-- §12 request was honored within 90 days, while the immutable audit_log + the F2i signature chain
-- (records of PROCESSING) are RETAINED untouched. Append-only by posture: NO update/delete policy.
CREATE TABLE IF NOT EXISTS data_erasures (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id            UUID REFERENCES firms(id) ON DELETE SET NULL,    -- nullable: a firm-less/legacy user can erase
  principal_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,  -- WHOSE data was erased
  requested_by       UUID NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,  -- self, or an admin on-behalf
  reason             TEXT,                                 -- e.g. 'data-principal request §12', 'consent withdrawn'
  documents_erased   INTEGER NOT NULL DEFAULT 0,
  conversations_erased INTEGER NOT NULL DEFAULT 0,
  messages_erased    INTEGER NOT NULL DEFAULT 0,
  preserved          TEXT NOT NULL DEFAULT 'audit_log,signatures',  -- WHAT was deliberately RETAINED (the distinction)
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata           JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_data_erasures_principal ON data_erasures(principal_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_erasures_firm      ON data_erasures(firm_id, created_at DESC);

-- ── RLS — grievances + the erasure ledger are firm-internal evidence ───────────────────────────────
-- A data principal sees their OWN grievance + their OWN erasure record; a firm manager (partner-tier)
-- sees the firm's full set (to action it). The read-client backstop keeps it firm-internal; the live
-- writes are service-role. No UPDATE/DELETE policy on data_erasures (append-only by construction — the
-- proof of an erasure must not itself be erasable).
ALTER TABLE grievances ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Grievances readable within firm" ON grievances;
CREATE POLICY "Grievances readable within firm" ON grievances
  FOR SELECT USING (
    principal_id = (SELECT auth.uid())
    OR EXISTS (
      SELECT 1 FROM firm_memberships m
      WHERE m.firm_id = grievances.firm_id
        AND m.user_id = (SELECT auth.uid())
        AND m.role IN ('managing_partner','senior_partner','partner')
    )
  );

ALTER TABLE data_erasures ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Erasure records readable within firm" ON data_erasures;
CREATE POLICY "Erasure records readable within firm" ON data_erasures
  FOR SELECT USING (
    principal_id = (SELECT auth.uid())
    OR requested_by = (SELECT auth.uid())
    OR EXISTS (
      SELECT 1 FROM firm_memberships m
      WHERE m.firm_id = data_erasures.firm_id
        AND m.user_id = (SELECT auth.uid())
        AND m.role IN ('managing_partner','senior_partner','partner')
    )
  );

-- ── ROLLBACK (if ever needed) ──────────────────────────────────────────────────────────────────────
-- DROP TABLE IF EXISTS data_erasures;
-- DROP TABLE IF EXISTS grievances;
-- ALTER TABLE firms DROP COLUMN IF EXISTS grievance_officer_user_id;
-- ALTER TABLE firms DROP COLUMN IF EXISTS grievance_officer_email;
-- ALTER TABLE firms DROP COLUMN IF EXISTS grievance_officer_name;
-- (Dormant-on-empty, feeds nothing else; dropping reverts to byte-identical pre-F2k. The erasure
--  ledger is a compliance record — do NOT drop it once erasures exist in production.)
