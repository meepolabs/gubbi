# Architecture

## System overview

journalctl is a FastAPI application that exposes 12 MCP tools over streamable HTTP. Any MCP-compatible client connects via the MCP protocol, authenticates with Bearer tokens or OAuth 2.0, and reads/writes markdown files that form the journal.

![System architecture](diagrams/system-architecture.svg)

## Three-layer data model

The journal stores data in three layers, each serving a different purpose:

![Data model](diagrams/data-model.svg)

**Layer 1 — Curated entries** (`topics/**/*.md`). Dated entries organized by topic. Each entry is a decision, milestone, update, or reflection. This is what you browse when you want to see the progression of something.

**Layer 2 — Full transcripts** (`conversations/**/*.md`). Complete chat archives saved explicitly. Not every conversation gets saved — only substantive ones. Each saved conversation auto-generates a summary entry in the parent topic, so the topic file stays readable while the full transcript is available for deep reference.

**Layer 3 — Temporal views** (dynamic from FTS5). Timeline queries served on-demand — "show me everything from this week" or "what happened in March." These aren't static files; they're assembled at query time from the FTS5 index.

## Markdown file format

### Topic file

```markdown
---
topic: hobbies/running
title: Running Log
description: Training progress and race goals
tags: [running, fitness, outdoors]
created: 2025-01-15
updated: 2026-03-23
entry_count: 42
---
# Running Log

Training progress and race goals

---

## 2025-01-15

#milestone

First run of the year. 5K in 28 minutes. Starting slow.

---

## 2026-03-23

#milestone

Finished first half marathon! 1:52:30.
```

Entries are separated by `## YYYY-MM-DD` headers (not `---` dividers, which are ambiguous with YAML frontmatter). Inline tags use the `#tag` format within entry content.

### Conversation file

```markdown
---
type: conversation
source: claude
title: Half Marathon Training Plan
topic: hobbies/running
tags: [running, training]
created: 2026-02-10
summary: Designed a 12-week half marathon training plan...
message_count: 12
---
# Half Marathon Training Plan

---

### User (2026-02-10 14:30:00)

I want to train for a half marathon in April. Can you help me plan?

---

### Assistant (2026-02-10 14:30:15)

Let's build a 12-week plan based on your current fitness level...
```

Saving a conversation is idempotent — re-saving the same topic + title overwrites the file. Git (via daily cron) preserves every version.

## Data flow

![Data flow](diagrams/data-flow.svg)

### Write path

The LLM calls `journal_append` → input is validated (path traversal prevention, content sanitization) → a filelock is acquired → the entry is appended with a `## YYYY-MM-DD` header → the file is written and lock released → the FTS5 index is updated.

### Read path

The LLM calls `journal_read` → the markdown file is loaded → YAML frontmatter is parsed → the body is split on `## YYYY-MM-DD` headers → entries are returned as structured objects.

### Search path

The LLM calls `journal_search` → an FTS5 MATCH query is built with optional topic and date filters → results are returned with `<mark>` highlighted snippets and relevance scores.

### Conversation save path

The LLM calls `journal_save_conversation` → the transcript is written to `conversations/{topic}/{title}.md` → a summary entry is upserted in the parent topic file with a `[[wikilink]]` → both files are indexed in FTS5.

### Briefing path

The LLM calls `journal_briefing` → user profile is read from `knowledge/user-profile.md` → this week's timeline is queried from FTS5 → top 20 recently-active topics are listed → document counts are gathered → all returned as a single context payload.

## Concurrency model

The server runs multiple gunicorn workers sharing the same filesystem and SQLite database:

| Mechanism | What | Why |
|-----------|------|-----|
| `filelock` | Per-file write locks (`.{name}.lock`) | Multiple workers may write the same topic file simultaneously |
| WAL mode | SQLite Write-Ahead Logging | Allows concurrent readers alongside a single writer |
| `busy_timeout=5000` | 5-second retry on SQLITE_BUSY | Workers retry instead of crashing on lock contention |
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

The journal is a **ledger** — it records what happened. It doesn't store quick-recall facts, entity relationships, or current-state preferences. That's the job of a separate **memory service**, which would use semantic embeddings for fuzzy matching. The LLM orchestrates between both services, routing "what happened?" to the journal and "what is?" to memory.

![Journal vs Memory](diagrams/journal-vs-memory.svg)
