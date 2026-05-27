-- Phase 4: Analytics — Query Logging
-- Tracks every query for usage analytics, performance monitoring, and audit.

CREATE TABLE query_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
  question TEXT NOT NULL,
  answer_length INT,
  sources_count INT,
  retrieval_docs_count INT,
  latency_ms INT,
  cache_hit BOOLEAN DEFAULT FALSE,
  agentic BOOLEAN DEFAULT FALSE,
  web_search_used BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own logs" ON query_logs
  FOR ALL USING (user_id = auth.uid());

CREATE INDEX idx_query_logs_user_date ON query_logs(user_id, created_at DESC);
CREATE INDEX idx_query_logs_agentic ON query_logs(user_id, agentic);
