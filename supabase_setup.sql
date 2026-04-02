-- Run this in your Supabase SQL Editor (supabase.com → project → SQL Editor)

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table (embeddings + metadata)
CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,
  patient_id  TEXT,
  task        TEXT,
  question    TEXT,
  answer      TEXT,
  note        TEXT,
  embedding   vector(384)   -- all-MiniLM-L6-v2 outputs 384 dims
);

-- 3. Similarity search function (cosine distance)
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(384),
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  id          BIGINT,
  patient_id  TEXT,
  task        TEXT,
  question    TEXT,
  answer      TEXT,
  note        TEXT,
  similarity  FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    id, patient_id, task, question, answer, note,
    1 - (embedding <=> query_embedding) AS similarity
  FROM documents
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- 4. Query logs table
CREATE TABLE IF NOT EXISTS query_logs (
  id          BIGSERIAL PRIMARY KEY,
  query       TEXT,
  response    TEXT,
  latency_ms  INT,
  sources     JSONB,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);
