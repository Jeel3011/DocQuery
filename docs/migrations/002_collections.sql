-- Phase 1: Collections for multi-document Q&A

CREATE TABLE collections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  name TEXT NOT NULL DEFAULT 'My Documents',
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE collection_documents (
  collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  added_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (collection_id, document_id)
);

ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own collections" ON collections
  FOR ALL USING (user_id = auth.uid());

ALTER TABLE collection_documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own collection_documents" ON collection_documents
  FOR ALL USING (
    collection_id IN (SELECT id FROM collections WHERE user_id = auth.uid())
  );

CREATE INDEX idx_collections_user ON collections(user_id, updated_at DESC);
CREATE INDEX idx_collection_docs ON collection_documents(collection_id);
