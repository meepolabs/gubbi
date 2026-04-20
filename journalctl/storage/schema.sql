-- journalctl PostgreSQL schema
-- Idempotent: safe to run on an existing database (CREATE TABLE IF NOT EXISTS)

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Topics ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_topics_updated ON topics (updated_at DESC);

-- ── Conversations ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    topic_id      INTEGER NOT NULL REFERENCES topics (id) ON DELETE RESTRICT,
    title         TEXT NOT NULL,
    slug          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'claude',
    summary       TEXT NOT NULL DEFAULT '',
    tags          TEXT[] NOT NULL DEFAULT '{}',
    participants  TEXT[] NOT NULL DEFAULT '{}',
    message_count INTEGER NOT NULL DEFAULT 0,
    json_path     TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(summary, ''))
    ) STORED,
    UNIQUE (topic_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_conv_fts     ON conversations USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_conv_topic   ON conversations (topic_id);
CREATE INDEX IF NOT EXISTS idx_conv_slug    ON conversations (topic_id, slug);
-- Needed for list_conversations ORDER BY and briefing/timeline date-range queries
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations (created_at DESC);

-- ── Entries ──────────────────────────────────────────────────────────────────
-- search_vector is GENERATED from search_text (post-02.12 baseline).
-- New writes populate content_encrypted/content_nonce + reasoning_encrypted/
-- reasoning_nonce + search_text; legacy content/reasoning columns remain
-- during the 02.13 -> 02.14 backfill window (dropped in migration 0007).
CREATE TABLE IF NOT EXISTS entries (
    id                  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    topic_id            INTEGER NOT NULL REFERENCES topics (id) ON DELETE RESTRICT,
    date                DATE NOT NULL DEFAULT CURRENT_DATE,
    content             TEXT NOT NULL,
    reasoning           TEXT,
    content_encrypted   BYTEA,
    content_nonce       BYTEA,
    reasoning_encrypted BYTEA,
    reasoning_nonce     BYTEA,
    search_text         TEXT,
    conversation_id     INTEGER REFERENCES conversations (id) ON DELETE RESTRICT,
    tags                TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at          TIMESTAMPTZ,
    indexed_at          TIMESTAMPTZ,
    search_vector       tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(search_text, ''))
    ) STORED,
    CONSTRAINT entries_content_nonce_len
        CHECK (content_nonce IS NULL OR octet_length(content_nonce) = 12),
    CONSTRAINT entries_reasoning_nonce_len
        CHECK (reasoning_nonce IS NULL OR octet_length(reasoning_nonce) = 12)
);

-- Composite partial index: covers read_entries (topic + date range, active only),
-- get_entries_by_date_range, and _get_topic_id join. Replaces the separate
-- idx_entries_topic and idx_entries_date indexes for queries that filter deleted_at IS NULL.
CREATE INDEX IF NOT EXISTS idx_entries_topic_date ON entries (topic_id, date DESC)
    WHERE deleted_at IS NULL;
-- Kept for queries that do NOT filter deleted_at (e.g. update/delete by id)
CREATE INDEX IF NOT EXISTS idx_entries_topic      ON entries (topic_id);
CREATE INDEX IF NOT EXISTS idx_entries_conv       ON entries (conversation_id);
-- Partial index: only indexes the rows get_unindexed_entries actually queries
CREATE INDEX IF NOT EXISTS idx_entries_indexed_at ON entries (id) WHERE indexed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_entries_fts        ON entries USING GIN (search_vector);

-- ── Messages ─────────────────────────────────────────────────────────────────
-- content_encrypted/content_nonce/search_text added in migration 0006;
-- legacy content column remains NOT NULL during the backfill window.
CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id   INTEGER NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    content_encrypted BYTEA,
    content_nonce     BYTEA,
    search_text       TEXT,
    timestamp         TIMESTAMPTZ,
    position          INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT messages_content_nonce_len
        CHECK (content_nonce IS NULL OR octet_length(content_nonce) = 12)
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, position);

-- ── Entry embeddings (pgvector) ───────────────────────────────────────────────
-- ON DELETE CASCADE: deleting an entry row automatically removes its embedding.
-- This replaces the hard_delete_by_entry_id() raw-SQL hack.
CREATE TABLE IF NOT EXISTS entry_embeddings (
    entry_id   INTEGER PRIMARY KEY REFERENCES entries (id) ON DELETE CASCADE,
    embedding  vector(384) NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index: no training step required, works well on initially small datasets.
-- m=32, ef_construction=128: higher recall at >50k entries vs defaults (m=16, ef=64).
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON entry_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 128);
