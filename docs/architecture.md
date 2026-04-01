# Architecture

## System overview

journalctl is a FastAPI application that exposes 13 MCP tools over streamable HTTP. Any MCP-compatible client connects via the MCP protocol, authenticates with Bearer tokens or OAuth 2.0, and reads/writes to a canonical SQLite database that stores all journal data with markdown files as archival record.

![System architecture](diagrams/system-architecture.svg)

## Three-tier data model

The journal stores data in three tiers, each serving a different purpose:

![Data model](diagrams/data-model.svg)

**Tier 1 — Hot data** (entry.content, ~50-100 tokens). Headline/summary always loaded with briefings and searches. Stored in the `entries` table.

**Tier 2 — Warm data** (entry.reasoning, ~200-500 tokens). Reasoning and background context, loaded on-demand when reading a specific topic via `journal_read_topic`. Stored in the `entries` table alongside content.

**Tier 3 — Cold data** (conversation JSON, 5k-100k tokens). Full chat transcripts archived in separate `conversations_json/{id}.json` files (flat by conversation ID). Messages stored in the `messages` table. Rarely accessed; designed for archival and eventual S3 backup.

## Storage architecture

### Canonical storage: SQLite database

All data is stored in `journal.db` (canonical source of truth):

```sql
topics        -- id, path, title, description, tags, created_at, updated_at
entries       -- id, topic_id, date, content, reasoning, conversation_id, tags, position,
              --  created_at, updated_at, deleted_at, indexed_at
conversations -- id, topic_id, title, slug, source, summary, tags, message_count,
              --  json_path, created_at, updated_at
messages      -- id, conversation_id, role, content, timestamp, position
```

### Archival storage: Markdown + JSON files

JSON files exist as **readable archival copies** inside `conversations_json/{id}.json` (keyed by conversation ID):

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

The JSON files are the archival record — rebuildable source for the database. Designed to be shipped to S3 for long-term storage. Saving a conversation is idempotent — re-saving overwrites the file.

## Data flow

![Data flow](diagrams/data-flow.svg)

### Write path (journal_append_entry)

Input is validated (path traversal prevention) → entry is inserted into the `entries` table with date, content, and reasoning → the FTS5 index is updated → both writes are committed atomically via SQLite WAL.

### Read path (journal_read_topic)

The tool queries the `entries` table for the given topic, sorted by date → optionally filters by date_from/date_to → optionally limits to last N entries with offset pagination (capped at 500) → returns as structured objects with id, date, content, reasoning, tags.

### Update path (journal_update_entry)

Entry ID is looked up in the `entries` table → content/reasoning/tags/date are updated → FTS5 index is refreshed and semantic embedding is re-stored.

### Delete path (journal_delete_entry)

Entry ID is marked as deleted (soft delete) in the `entries` table → excluded from future reads/searches but preserved in database for git backup.

### Search path (journal_search)

An FTS5 keyword query runs in parallel with semantic vector search → results are merged and deduplicated by entry ID → returned with snippets and relevance scores. Limit capped at 100 results.

### Conversation save path (journal_save_conversation)

Message count validated (max 1000) → conversation JSON archived to `conversations_json/{id}.json` → conversation record inserted into the `conversations` table → all messages inserted into the `messages` table → FTS5 index updated. Re-saving the same topic + title updates the existing record in place.

### Briefing path (journal_briefing)

User profile is read from `knowledge/user-profile.md` → key facts retrieved from semantic memory (top matches for identity/preferences) → this week's entries queried from the database (most-recent-first, capped at 25) → top 20 recently-active topics listed → document counts gathered → all returned as a single context payload.

## Concurrency model

The server runs multiple gunicorn workers sharing the same filesystem and SQLite database:

| Mechanism | What | Why |
|-----------|------|-----|
| WAL mode | SQLite Write-Ahead Logging | Allows concurrent readers alongside a single writer |
| `busy_timeout=5000` | 5-second retry on SQLITE_BUSY | Workers retry instead of crashing on lock contention |
| `asyncio.Lock` | Reindex lock (in-process) | Prevents concurrent reindex runs from corrupting the FTS5 virtual table |
| ASGI middleware | Raw scope/receive/send passthrough | BaseHTTPMiddleware buffers responses, which breaks SSE streaming |
| `secrets.compare_digest` | Timing-safe token comparison | Prevents token-guessing via timing side channels |

## Authentication

Dual-mode auth supports both static API keys and OAuth 2.0:

**API key mode** — for CLI tools and desktop apps. Set the key as an environment variable, pass it as a Bearer token in the MCP client config.

**OAuth 2.0 mode** — for browser and mobile clients. Full PKCE flow with bcrypt password verification, CSRF-protected login page, and token refresh. OAuth state lives in `oauth.db`, separate from the disposable FTS5 index.

```
Incoming request with Bearer token
    ├── Matches static API key? (secrets.compare_digest) → Allow
    ├── Valid OAuth access token? (check oauth.db expiry) → Allow
    └── Neither → 401 Unauthorized
```

## What the journal is not

The journal is a **ledger** — it records what happened. It doesn't store quick-recall facts, entity relationships, or current-state preferences. That's the job of the **memory service**, which uses a local ONNX embedding model for semantic fuzzy matching and runs on the same server. The LLM orchestrates between both services, routing "what happened?" to the journal and "what is?" to memory.

![Journal vs Memory](diagrams/journal-vs-memory.svg)
