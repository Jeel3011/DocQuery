-- 011_document_privilege.sql
-- F1e (plans/F1_VAULT_PLAN.md §1 — Privilege firewall): tag a document as
-- privileged / attorney work-product so it can be EXCLUDED from any shared / cross-vault
-- surface (F6 Firm Brain, future cross-matter reuse) and WATERMARKED in exports. The
-- exclusion is a data-layer property (the flag travels with the doc into the run scope), not a
-- prompt instruction.
--
-- WHY now: courts in 2026 hold that an AI tool retaining cross-matter context can break
-- privilege. The privilege firewall is the marketed answer — a privileged doc never crosses a
-- vault boundary and every export that touches one is watermarked. F1e ships the FLAG + the
-- exclusion guard + the watermark; F2 owns WHO may set the flag (partner-gating).
--
-- Additive + defaulted ⇒ existing rows read as privileged=false (the byte-identical legacy
-- state — nothing is privileged until a user marks it). Safe to re-run. Code degrades
-- gracefully if unapplied: the worker/API drop the key on a column-missing error and the UI
-- shows the neutral (non-privileged) state.

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS privileged boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN documents.privileged IS
  'F1e privilege firewall: true = attorney-client / work-product. Excluded from shared / '
  'cross-vault surfaces (F6) and watermarked in exports. Default false (nothing privileged '
  'until marked).';

-- Index the flag so a cross-vault/shared query can cheaply filter out privileged docs.
CREATE INDEX IF NOT EXISTS idx_documents_privileged ON documents(user_id, privileged);
