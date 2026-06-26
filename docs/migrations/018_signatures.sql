-- 018_signatures.sql
-- F2i (plans/F2_FIRM_CONSOLE_PLAN.md §F2i): E-SIGNATURES — legally-valid sign-off (IT Act 2000).
--
-- WHY this slice: the F2e review chain ends when a partner APPROVES (internal sign-off) and RELEASES
-- (the work leaves the firm). Today those transitions are plain audit rows — they record that it
-- happened, but they are NOT a SIGNATURE: no signer-bound, tamper-evident, legally-citable artifact a
-- firm can stand behind if a release is later disputed. F2i adds that artifact: a "secure electronic
-- signature" in the sense of IT Act §3/§5/§10A — click-to-sign + the authenticated (JWT-verified)
-- member identity + a tamper-evident record (signer · timestamp · intent · ARTIFACT HASH · IP/device).
-- That tier is legally valid with NO external dependency (§85B presumption of integrity for a secure
-- e-record). Aadhaar-eSign / DSC (the §67A/§85B/§85C strong tier) is a SEAM (signature_method column),
-- plugged in for court-grade documents when a paying firm needs it — never forced on every click.
--
-- WHAT this migration is: ONE append-only, HASH-CHAINED table. This is the key design choice (Jeel's
-- call): F2i ships its OWN tamper-evidence rather than waiting on F3's general ledger.
--   * Each row stores `content_hash` = sha256 of the SIGNED PAYLOAD (signer + artifact_ref +
--     artifact_hash + intent + signed_at) → tampering the artifact or the record breaks verification.
--   * Each row also stores `prev_hash` (the previous row's `row_hash` for that firm) and `row_hash`
--     = sha256(prev_hash + content_hash) → an append-only CHAIN: deleting or reordering a signature
--     breaks every subsequent row's hash. This is exactly what F3 will generalize for the whole audit
--     log; F2i's chain is the first, self-contained instance of it (a clean fold-in later).
--   * The chain is PER-FIRM (chain_seq is a per-firm monotonic counter) so one firm's signatures can
--     be verified end-to-end independently, and a firm's chain can't be polluted by another's rows.
--
-- THE §1(4) EXCLUSION GUARD lives in CODE (src/components/esign.py: EXCLUDED_INSTRUMENTS + classify),
-- not the schema — a categorically non-e-signable instrument (negotiable instrument, power of
-- attorney, will, trust, immovable-property sale) is REFUSED before any row is written ("wet-ink
-- required"), because a wrong e-sign there is VOID under IT Act §1(4). The table records what WAS
-- signed; the guard decides what MAY be.
--
-- SECURITY / posture (inherited from 013/015/017): the write client is service-role (RLS-exempt) so
-- the app-layer authz (require_cap("release_external") / the review-chain owner check) is the real
-- guard; this migration's RLS is the read-client BACKSTOP — a firm member may SELECT only their own
-- firm's signatures (the signature record is firm-internal evidence, partner-tier+ to read the full
-- ledger; the signer always sees their own). No UPDATE / DELETE policy at all — the table is
-- APPEND-ONLY by construction (immutability is the whole point of a signature ledger; a correction is
-- a NEW counter-signing row, never an edit).
--
-- PRIME RULE (inherited): DORMANT-ON-EMPTY ⇒ table-unapplied = byte-identical to pre-F2i. Every db.py
-- method degrades to None/[]/False on a missing relation (db._is_missing_relation); the sign step is
-- best-effort at the AUDIT level but BLOCKING at the legal level — see esign.py: a release still
-- succeeds if signing fails to WRITE (the work isn't held hostage by an infra blip), but the absence
-- of a signature row is itself detectable (verify() reports "unsigned"). The §1(4) refusal, by
-- contrast, is HARD (a 422 before release) — a void signature is worse than none.
--
-- Apply via the Supabase SQL editor (or psql). Safe to re-run (IF NOT EXISTS / idempotent policies).
-- Jeel applies migrations himself (F1 convention) — get consent first.

-- ── signatures — one row = one tamper-evident, hash-chained sign-off ───────────────────────────────
-- signer_id = WHO signed (JWT-verified identity = the "secure" tier's authentication factor).
-- artifact_ref = WHAT was signed (a review_request id today; a draft/grid/document later — same shape).
-- artifact_hash = sha256 of the artifact's content AT SIGNING TIME (the tamper anchor: re-hash the
--   artifact later and compare; a mismatch = the signed thing changed). intent = the legal act
--   ('approve' internal sign-off / 'release' external release). signature_method = the IT Act tier
--   ('secure_eauth' = tier-2 default; 'aadhaar_esign' / 'dsc' = the strong-tier SEAM, not forced).
-- content_hash / prev_hash / row_hash / chain_seq = the per-firm append-only chain (see header).
CREATE TABLE IF NOT EXISTS signatures (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  firm_id         UUID NOT NULL REFERENCES firms(id)      ON DELETE CASCADE,  -- the per-firm chain + boundary (T3)
  signer_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT, -- WHO signed (never orphan a signature)
  vault_id        UUID REFERENCES collections(id)         ON DELETE SET NULL, -- the matter (wall scope; nullable)
  artifact_type   TEXT NOT NULL DEFAULT 'review_request', -- WHAT KIND of thing was signed
  artifact_ref    TEXT NOT NULL,                          -- WHICH artifact (review_request id today)
  artifact_hash   TEXT NOT NULL,                          -- sha256 of the artifact content at signing (tamper anchor)
  intent          TEXT NOT NULL,                          -- 'approve' | 'release' (the legal act)
  signature_method TEXT NOT NULL DEFAULT 'secure_eauth',  -- IT Act tier: secure_eauth | aadhaar_esign | dsc
  signer_name     TEXT,                                   -- the human handle at signing time (best-effort)
  ip_address      TEXT,                                   -- the device/network factor (secure-tier evidence)
  user_agent      TEXT,                                   -- the device factor
  signed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),     -- WHEN (the timestamp factor)
  -- ── the hash chain (append-only tamper-evidence) ──
  chain_seq       BIGINT NOT NULL,                        -- per-firm monotonic counter (1,2,3…)
  content_hash    TEXT NOT NULL,                          -- sha256 of the signed payload (record-tamper anchor)
  prev_hash       TEXT,                                   -- the prior row's row_hash for this firm (NULL for seq 1)
  row_hash        TEXT NOT NULL,                          -- sha256(prev_hash + content_hash) — the chain link
  metadata        JSONB NOT NULL DEFAULT '{}'             -- intent note, gate state, method-specific evidence
);

-- The per-firm chain order is unique + monotonic — no two signatures share a sequence in a firm, so
-- the chain can be walked deterministically (and a gap is itself detectable).
CREATE UNIQUE INDEX IF NOT EXISTS uq_signatures_firm_seq ON signatures(firm_id, chain_seq);
-- The hot verification query: "this firm's signatures in chain order."
CREATE INDEX IF NOT EXISTS idx_signatures_firm_seq ON signatures(firm_id, chain_seq);
-- "the signature(s) for this artifact" (the release/approve record lookup).
CREATE INDEX IF NOT EXISTS idx_signatures_artifact ON signatures(artifact_type, artifact_ref);
CREATE INDEX IF NOT EXISTS idx_signatures_signer ON signatures(signer_id);

-- ── RLS — a firm member reads their firm's signature ledger; NO edits ever (T2/T3) ────────────────
-- The signer always needs to see their own sign-off; partner-tier reads the full ledger as firm
-- evidence. The read-client backstop here keeps it firm-internal (the live writes are service-role).
-- Crucially: there is NO UPDATE and NO DELETE policy — the authenticated read-client can NEVER mutate
-- a signature. Append-only is enforced structurally, not by convention.
ALTER TABLE signatures ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Signatures readable within firm" ON signatures;
CREATE POLICY "Signatures readable within firm" ON signatures
  FOR SELECT USING (
    signer_id = (SELECT auth.uid())
    OR EXISTS (
      SELECT 1 FROM firm_memberships m
      WHERE m.firm_id = signatures.firm_id
        AND m.user_id = (SELECT auth.uid())
        AND m.role IN ('managing_partner','senior_partner','partner')
    )
  );

-- ── ROLLBACK (if ever needed) ─────────────────────────────────────────────────────────────────────
-- DROP TABLE IF EXISTS signatures;
-- (Dormant-on-empty, feeds nothing else; dropping reverts to byte-identical pre-F2i. The F3 ledger,
--  when built, will RE-HOME this chain — do not drop it once signatures exist in production.)
