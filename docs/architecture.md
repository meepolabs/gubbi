# Architecture

## System overview

journalctl is a FastAPI application that exposes 13 MCP tools over streamable HTTP. Any MCP-compatible client connects via the MCP protocol, authenticates with Bearer tokens or OAuth 2.1, and reads/writes to a canonical PostgreSQL 17 database (`pgvector/pgvector:pg17`) that stores all journal data. Conversation transcripts are additionally archived as JSON files on disk for long-term backup.

![System architecture](diagrams/system-architecture.svg)

## Three-tier data model

The journal stores data in three tiers, each serving a different purpose:

![Data model](diagrams/data-model.svg)

**Tier 1 — Hot data** (entry.content, ~50-100 tokens). Headline/summary always loaded with briefings and searches. Stored in the `entries` table.

**Tier 2 — Warm data** (entry.reasoning, ~200-500 tokens). Reasoning and background context, loaded on-demand when reading a specific topic via `journal_read_topic`. Stored in the `entries` table alongside content.

**Tier 3 — Cold data** (conversation JSON, 5k-100k tokens). Full chat transcripts archived as `conversations_json/{uuid}.json` files on disk alongside the database. Messages are also stored in the `messages` table for in-database access. The JSON archives are the rebuildable source; designed for eventual S3 backup.

## Storage architecture

### Canonical storage: PostgreSQL 17

All data lives in one PostgreSQL database. Five tables:

```sql
topics            -- id, path, title, description,
                  --  created_at, updated_at

conversations     -- id, topic_id, title, slug, source, summary,
                  --  tags TEXT[], participants TEXT[], message_count,
                  --  json_path, created_at, updated_at,
                  --  search_vector tsvector GENERATED ALWAYS AS (...) STORED,
                  --  UNIQUE (topic_id, slug)

entries           -- id, topic_id, date, content, reasoning,
                  --  conversation_id, tags TEXT[],
                  --  created_at, updated_at, deleted_at, indexed_at,
                  --  search_vector tsvector GENERATED ALWAYS AS (...) STORED

messages          -- id, conversation_id, role, content, timestamp, position

entry_embeddings  -- entry_id PK FK (ON DELETE CASCADE),
                  --  embedding vector(384), indexed_at
```

Key schema decisions:

- `search_vector` is a **generated stored column** — PostgreSQL rewrites it on every content/reasoning (or title/summary) change. Zero application-side FTS sync.
- `entry_embeddings` is a separate table with `ON DELETE CASCADE` from `entries`. Deleting an entry (hard) automatically removes its embedding.
- `pgvector` HNSW index tuned for multi-tenant scale (`m = 32`, `ef_construction = 128`).
- `tsvector` GIN index powers `websearch_to_tsquery`, which handles natural language + boolean operators without crashing on trailing punctuation.
- `tags` and `participants` are `TEXT[]` — `= ANY(tags)` is enough for current queries; swap to JSONB if containment operators are needed later.
- `UNIQUE (topic_id, slug)` on `conversations` enables `ON CONFLICT DO UPDATE` upsert for idempotent re-saves.
- All timestamps are `TIMESTAMPTZ` except `entries.date` (day-level `DATE` is sufficient for the human-facing journal date).
- Foreign keys use `ON DELETE RESTRICT` for topic/entry/conversation links (referential integrity) and `ON DELETE CASCADE` for messages and embeddings (lifecycle tied to parent).

### Archival storage: JSON files

JSON files exist as **readable archival copies** inside `conversations_json/{uuid}.json`. Every saved conversation writes a UUID-named JSON file **before** the database transaction — failure modes are clean: a failed file write never opens a transaction, and a failed transaction leaves a harmless orphan UUID file that nothing references.

```json
{
  "meta": {
    "type": "conversation",
    "source": "claude",
    "title": "Half Marathon Training Plan",
    "topic": "hobbies/running",
    "tags": ["running", "training"],
    "created": "2026-02-10",
    "summary": "Designed a 12-week half marathon training plan...",
    "message_count": 12
  },
  "messages": [
    {
      "role": "user",
      "content": "I want to train for a half marathon in April. Can you help me plan?",
      "timestamp": "2026-02-10T14:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Let's build a 12-week plan based on your current fitness level...",
      "timestamp": "2026-02-10T14:30:15Z"
    }
  ]
}
```

The JSON files are the archival record — rebuildable source for the database. Designed to be shipped to S3 for long-term storage. Saving a conversation is idempotent — re-saving the same `(topic, slug)` updates the existing row in place via `ON CONFLICT DO UPDATE`.

## Data flow

![Data flow](diagrams/data-flow.svg)

### Write path (journal_append_entry)

Input is validated (path traversal prevention, topic/date format, freetext sanitization) → a single CTE inserts the entry and bumps `topics.updated_at` in one round-trip → the `search_vector` column is automatically refreshed by PostgreSQL → after commit, the ONNX embedding is generated via `asyncio.to_thread` (outside the DB connection) and upserted into `entry_embeddings` via pgvector, then `entries.indexed_at` is stamped.

### Read path (journal_read_topic)

The tool queries the `entries` table for the given topic (pre-filtered by `WHERE deleted_at IS NULL`), sorted by date → optionally filters by date_from/date_to → uses a window function (`COUNT(*) OVER()`) to return total and data rows in one query → capped at 500 entries with offset pagination → returns structured objects with id, date, content, reasoning, tags.

### Update path (journal_update_entry)

A transaction runs a CTE that updates the entry (setting `indexed_at = NULL` to mark re-embed needed) and bumps `topics.updated_at` → the committed `(content, reasoning)` is read back via `get_text` in the same transaction → after commit, the new embedding is generated outside the DB connection and upserted. `tsvector` refreshes automatically because `search_vector` is `GENERATED ALWAYS`.

### Delete path (journal_delete_entry)

A single CTE soft-deletes the entry (sets `deleted_at`), deletes its row in `entry_embeddings`, and bumps `topics.updated_at` — all in one round-trip. The entry is preserved in the database for audit but excluded from all reads and searches.

### Search path (journal_search)

The query embedding is generated via `asyncio.to_thread(embedding_service.encode, query)` **before** a DB connection is acquired, so the pool isn't pinned during inference → if a `topic_prefix` is set, topic IDs are resolved first → `fts_search` runs `websearch_to_tsquery` against both `entries` and `conversations` (two `conn.fetch` calls, negated `ts_rank` for ascending sort) with `ts_headline` snippets → `embedding_service.search_by_vector` runs pgvector cosine similarity pre-filtered by topic IDs → FTS and semantic results are merged by `source_key` deduplication → response includes `semantic_available` so clients can tell whether semantic degraded.

### Conversation save path (journal_save_conversation)

Message count validated (max 1000) → conversation JSON archived to `conversations_json/{uuid}.json` **first** (phase 1, writes a fresh UUID file) → a single DB transaction runs the `ON CONFLICT DO UPDATE` upsert, deletes + re-inserts messages if message count changed, and upserts a linked entry tagged `['conversation']` so the saved conversation shows up in the timeline. Failure modes are clean: if the file write fails, no transaction runs; if the transaction fails, the orphan UUID file is harmless.

### Briefing path (journal_briefing)

User profile is read from `knowledge/user-profile.md` → a canned key-facts query embedding is pre-encoded outside the pool → one acquired connection fetches this week's entries (most-recent-first, capped at 25), the top 20 recently-updated topics, topic count, entry stats, and semantic key-fact matches via `embedding_service.search_by_vector` → all returned as a single context payload.

## Concurrency model

The server runs multiple gunicorn workers against a shared PostgreSQL database:

| Mechanism | What | Why |
|-----------|------|-----|
| Per-worker asyncpg pool | Each gunicorn worker creates its own pool in its lifespan | asyncpg pools cannot survive `os.fork()` — no `--preload` flag in gunicorn |
| PostgreSQL MVCC | Native reader/writer concurrency | No WAL-mode quirks, no `busy_timeout` needed |
| `pg_try_advisory_lock(...)` | Cross-worker reindex lock | Prevents two workers from running semantic reindex concurrently; returns `already_running` if the lock can't be acquired |
| Shared-state cooldown | `MAX(indexed_at) FROM entries` as a reindex timestamp | Returns `cooldown` if a reindex ran in the last 60 seconds |
| ASGI middleware | Raw scope/receive/send passthrough | `BaseHTTPMiddleware` buffers responses, which breaks SSE streaming |
| `secrets.compare_digest` | Timing-safe token comparison | Prevents token-guessing via timing side channels |
| `statement_cache_size=0` | asyncpg setting | Required for pgbouncer transaction-pooling compatibility |

## Authentication

Dual-mode auth supports both static API keys and OAuth 2.1:

**API key mode** -- for CLI tools and desktop apps. Set the key as an environment variable, pass it as a Bearer token in the MCP client config.

**OAuth 2.1 mode** -- for browser and mobile clients. Full PKCE flow with RFC 7591 Dynamic Client Registration, bcrypt password verification, CSRF-protected login page, and token refresh. OAuth state lives in `oauth.db` (SQLite), deliberately separate from the PostgreSQL journal database so auth changes never touch user data. (This is the self-host OAuth path -- multi-tenant hosted deploys use Hydra+Kratos instead; see the private `journalctl-cloud` repo.)

```
Incoming request with Bearer token
    ├── Matches static API key? (secrets.compare_digest) → Allow
    ├── Valid OAuth access token? (check oauth.db expiry) → Allow
    └── Neither → 401 Unauthorized
```

## Semantic memory is internal

There is no separate "memory service" to orchestrate. Semantic search is just part of the journal: `journal_append_entry` auto-embeds the entry after commit, `journal_search` merges `tsvector` FTS with `pgvector` semantic results, and `journal_briefing` surfaces key life facts by running a canned semantic query against the same embeddings. No memory tools are exposed to the LLM — it just calls `journal_search`, `journal_briefing`, and `journal_read_topic` and gets both keyword and meaning-based results.

The `EmbeddingService` (`storage/embedding_service.py`) is a thin ONNX wrapper: synchronous `encode()` for CPU-bound inference, async `store_by_vector`/`search_by_vector` for pgvector upsert/search. Tool code encodes via `asyncio.to_thread` before acquiring a DB connection, so the pool is never pinned during inference.

![Journal vs Memory](diagrams/journal-vs-memory.svg)
