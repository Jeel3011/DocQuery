-- 017_notifications.sql
-- F2j (plans/F2_FIRM_CONSOLE_PLAN.md §F2j): NOTIFICATIONS — make the review chain VISIBLE.
--
-- WHY this slice: the F2e review chain already CANNOT stall (every review_requests row always names a
-- current_owner — the anti-stall invariant). But that flow is INVISIBLE: when work lands in someone's
-- review queue, a member is staffed onto a matter, or their work is approved/returned/released, nobody
-- is TOLD. F2j adds the telling — the feedback loop that makes the engine feel alive — WITHOUT changing
-- the flow itself.
--
-- WHAT this migration is: the STORAGE shape. `notifications` is the INBOX + the OUTBOX + the
-- queue-ready seam (one table, three jobs — the transactional-outbox pattern: the notification row is
-- written in the SAME flow as the action so no notification is ever lost on a broker outage, and a
-- worker later drains the EMAIL channel by querying undelivered rows). `notification_preferences` is
-- the per-user anti-nag control (mute a category / quiet hours / daily digest).
--
-- THE TWO RESEARCH-GROUNDED DESIGN CHOICES baked into the columns:
--   1. dedup_key + a UNIQUE partial index = the idempotency guarantee (the industry "last_notified_at"
--      anti-duplicate rule): the same event to the same person about the same resource on the same day
--      collapses to ONE row. A re-emit / a restarted escalation sweep NEVER double-nags.
--   2. email_status = the outbox drain column. v1 ships IN-APP ONLY (email_status='skipped'); when an
--      email transport lands, the worker queries email_status='pending' and sends — no schema change.
--
-- SECURITY (the F2 posture): every notification is scoped to ONE recipient and respects BOTH the F2c
-- ethical wall AND the F2f firm boundary. The wall/firm filtering is enforced in the APP DATA PATH
-- (src/components/notifications.py emit(): user_in_firm + is_vault_screened BEFORE the row is written)
-- — a screened or cross-firm user is NEVER written a row. This migration's RLS is the read-client
-- BACKSTOP (the live write client is service-role / RLS-exempt, like 013): a user may SELECT/UPDATE
-- only their OWN notifications (recipient_id = auth.uid()) — no one can read another user's inbox (T2/T8).
--
-- PRIME RULE (inherited): DORMANT-ON-EMPTY ⇒ flag-off / table-unapplied = byte-identical to pre-F2j.
-- Every db.py method degrades to empty/None on a missing relation (db._is_missing_relation); the emit
-- is best-effort (try/except, never raises — same posture as log_audit) so it can NEVER break the
-- originating action. An empty table ⇒ no notifications ⇒ the app behaves EXACTLY as F2g shipped it.
--
-- Apply via the Supabase SQL editor (or `psql`). Safe to re-run (IF NOT EXISTS / idempotent policies
-- throughout). Jeel applies migrations himself (F1 convention) — get consent first.

-- ── notifications — one row = one thing to tell one person (inbox + outbox + queue seam) ──────────
-- recipient_id = WHO is told (the row is theirs; RLS scopes reads to them). actor_id = who CAUSED it
-- (nullable: a system/escalation notification has no human actor). event = the specific thing
-- (review.awaiting / review.approved / review.changes_requested / review.released / matter.staffed /
-- review.escalation / answer.overridden). category = the preference bucket (review / matter /
-- governance) the anti-nag mute toggles act on. vault_id (nullable) = the wall scope: if set, a
-- recipient screened off that vault is never written the row (enforced in emit()). dedup_key = the
-- idempotency key (sha256 of recipient+event+resource+day) — the UNIQUE index makes a duplicate emit a
-- clean no-op. email_status = the outbox drain state ('skipped' in v1; 'pending'/'sent'/'failed' once
-- email is on). read_at NULL = unread (the bell badge counts these).
CREATE TABLE IF NOT EXISTS notifications (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id       UUID NOT NULL REFERENCES firms(id)        ON DELETE CASCADE,   -- the firm boundary (T3)
  recipient_id  UUID NOT NULL REFERENCES auth.users(id)   ON DELETE CASCADE,   -- who is told
  actor_id      UUID REFERENCES auth.users(id)            ON DELETE SET NULL,  -- who caused it (nullable)
  event         TEXT NOT NULL,                                                  -- the specific event
  category      TEXT NOT NULL DEFAULT 'review',                                 -- preference bucket
  resource_type TEXT,                                                           -- 'review_request' / 'vault' / 'answer'
  resource_id   TEXT,                                                           -- the resource the notice is about
  vault_id      UUID REFERENCES collections(id)           ON DELETE CASCADE,    -- the wall scope (nullable)
  title         TEXT,                                                           -- rendered human title
  body          TEXT,                                                           -- rendered human body
  dedup_key     TEXT,                                                           -- idempotency key (sha256)
  read_at       TIMESTAMPTZ,                                                    -- NULL = unread
  email_status  TEXT NOT NULL DEFAULT 'skipped',                                -- outbox drain: skipped|pending|sent|failed
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The hot inbox query: "my unread notifications, newest first."
CREATE INDEX IF NOT EXISTS idx_notifications_recipient_unread
  ON notifications(recipient_id, created_at DESC) WHERE read_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_notifications_recipient
  ON notifications(recipient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_firm ON notifications(firm_id);
-- The future email-drain query (the outbox): the worker pulls these and sends.
CREATE INDEX IF NOT EXISTS idx_notifications_email_pending
  ON notifications(email_status) WHERE email_status = 'pending';
-- THE DEDUP GUARANTEE: at most one notification per dedup_key. A duplicate emit (a re-fired event, a
-- restarted escalation sweep) hits this and is swallowed as a clean no-op by db.create_notification —
-- never a duplicate row, never a double-nag (the industry "last_notified_at" idempotency rule).
DROP INDEX IF EXISTS uq_notifications_dedup;
CREATE UNIQUE INDEX uq_notifications_dedup ON notifications(dedup_key)
  WHERE dedup_key IS NOT NULL;

-- ── RLS on notifications — a user sees / marks ONLY their own (T2/T8) ─────────────────────────────
-- Like 013/015: the live request client is service-role (RLS-exempt) so the app-layer scoping (the
-- route always passes sb.user_id as recipient_id) is the real guard; this policy is the read-client
-- backstop so a notification can never fan out cross-user. A user may SELECT + UPDATE (mark-read) only
-- rows addressed to THEM. Inserts come from the service-role write client (the emit path), so no
-- INSERT policy is needed for the app; a restrictive default (no policy ⇒ deny) is correct for the
-- authenticated read-client.
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Notifications visible to recipient" ON notifications;
CREATE POLICY "Notifications visible to recipient" ON notifications
  FOR SELECT USING (recipient_id = auth.uid());

DROP POLICY IF EXISTS "Notifications updatable by recipient" ON notifications;
CREATE POLICY "Notifications updatable by recipient" ON notifications
  FOR UPDATE USING (recipient_id = auth.uid()) WITH CHECK (recipient_id = auth.uid());


-- ── notification_preferences — per-user anti-nag controls (one row per user) ──────────────────────
-- muted_categories = the categories this user has turned OFF (a muted category suppresses the in-app
-- row). quiet_start/quiet_end = an optional hour-of-day window (e.g. 21→7) during which NON-URGENT
-- notifications defer their EMAIL — the in-app row is STILL written (queued, not dropped); urgent
-- events (review.escalation) bypass it. digest_mode = batch low-signal events into a daily email
-- summary (honored at the email layer; the in-app inbox is itself a passive digest). A MISSING ROW ⇒
-- sane defaults (nothing muted, no quiet hours, no digest) ⇒ the empty/unapplied table is
-- byte-identical to "notify normally" (db.get_notification_preferences returns the default dict).
CREATE TABLE IF NOT EXISTS notification_preferences (
  user_id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  muted_categories TEXT[] NOT NULL DEFAULT '{}',   -- categories turned off
  quiet_start      INT,                            -- hour 0-23, NULL = no quiet window
  quiet_end        INT,                            -- hour 0-23
  digest_mode      BOOLEAN NOT NULL DEFAULT false, -- batch low-signal events (email layer)
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Preferences owned by user" ON notification_preferences;
CREATE POLICY "Preferences owned by user" ON notification_preferences
  FOR ALL USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());


-- ── ROLLBACK (if ever needed) ─────────────────────────────────────────────────────────────────────
-- DROP TABLE IF EXISTS notification_preferences;
-- DROP TABLE IF EXISTS notifications;
-- (Both are dormant-on-empty and feed nothing else; dropping them reverts to byte-identical pre-F2j.)
