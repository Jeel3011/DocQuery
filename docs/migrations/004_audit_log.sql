-- Phase 6: Audit Trail
-- Tracks all significant user actions for compliance and security.

CREATE TABLE audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  action TEXT NOT NULL,         -- e.g., 'document.upload', 'collection.create', 'query.ask'
  resource_type TEXT,           -- e.g., 'document', 'collection', 'conversation'
  resource_id TEXT,             -- UUID of the affected resource
  metadata JSONB DEFAULT '{}', -- Additional context (filename, IP, etc.)
  ip_address TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own audit logs" ON audit_log
  FOR ALL USING (user_id = auth.uid());

CREATE INDEX idx_audit_user_date ON audit_log(user_id, created_at DESC);
CREATE INDEX idx_audit_action ON audit_log(user_id, action);
CREATE INDEX idx_audit_resource ON audit_log(user_id, resource_type, resource_id);
