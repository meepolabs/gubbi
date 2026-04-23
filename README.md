# journalctl

A self-hosted [MCP](https://modelcontextprotocol.io/) server that gives any LLM a persistent, searchable journal — a personal memory infrastructure layer for AI, backed by PostgreSQL on your own infrastructure.

**Works with any MCP-compatible client** -- not tied to any specific LLM provider. Any chat app or CLI tool that supports MCP servers via Bearer token or OAuth 2.1 can connect.

```
You: "I just finished setting up the new home server"

LLM: *appends to projects/homelab topic, saves conversation transcript, updates timeline*

Next week, different device, different client:

You: "Where did I leave off with the homelab?"

LLM: *searches journal* → "On March 23rd you finished the initial setup..."
```

## Why this exists

If you use LLMs across multiple clients — CLI tools, desktop apps, browser, mobile — your conversations vanish between sessions. You lose the thread on ongoing projects, life decisions, and accumulated context.

journalctl solves this by providing a **persistent memory layer** accessible from any MCP-compatible client. Every client connects to the same journal, so you pick up exactly where you left off — whether that's a coding project, a hobby log, a fitness plan, or a reading list.

The journal is an **append-only ledger**, not a brain. It faithfully stores everything — decisions, conversations, milestones, research — and never compresses, forgets, or consolidates. Full-text search (`tsvector` + GIN) and semantic search (`pgvector` HNSW) are both built into the same PostgreSQL database. No data leaves your infrastructure.

## Quick start

**Prerequisites:** Docker + Docker Compose, any small VPS, a domain with DNS configured.

```bash
git clone https://github.com/user/journalctl.git && cd journalctl
cp .env.example .env   # fill in your secrets
docker compose up -d   # brings up postgres + journalctl
```

Then connect your MCP client:

```json
{
  "mcpServers": {
    "journal": {
      "type": "http",
      "url": "https://journal.yourdomain.com/mcp/",
      "headers": { "Authorization": "Bearer YOUR_API_KEY" }
    }
  }
}
```

For browser-based clients, the server supports OAuth 2.1 with PKCE and RFC 7591 Dynamic Client Registration -- connect via your client's MCP integrations settings.

## What the LLM can do

| Tool | What it does |
|------|-------------|
| `journal_briefing` | Loads your profile, this week's activity, and recent topics at conversation start |
| `journal_append_entry` | Adds a dated entry to any topic |
| `journal_read_topic` | Reads a topic's full history or last N entries |
| `journal_search` | Hybrid full-text (tsvector) + semantic (pgvector) search across everything |
| `journal_save_conversation` | Archives a full chat transcript with summary |
| `journal_timeline` | Shows all activity for a week, month, or year |

Plus 7 more tools for topic management, conversation browsing, entry editing, deletion, and index maintenance. See [docs/tools-reference.md](docs/tools-reference.md) for the complete reference.

## How data is organized

```
data/
├── postgres/                        # PostgreSQL cluster (WAL + tablespaces)
├── onnx/                            # Cached ONNX embedding model (~/.cache/journalctl)
└── journal/
    ├── oauth.db                     # OAuth tokens/clients (SQLite, separate from PG)
    ├── conversations_json/          # Archived conversation transcripts (JSON)
    │   └── {uuid}.json
    └── knowledge/
        └── user-profile.md          # Your identity profile, loaded by journal_briefing
```

All journal data lives in PostgreSQL 17 (`topics`, `conversations`, `entries`, `messages`, `entry_embeddings`). Full-text search is a `tsvector` generated column with a GIN index — auto-maintained by the database, no reindex needed. Semantic search is a `pgvector` HNSW index keyed to `entries.id` via `ON DELETE CASCADE`. A local ONNX model (`all-MiniLM-L6-v2`, ~24MB quantized) generates embeddings on the CPU. `journal_reindex` rebuilds the semantic embeddings only — `tsvector` stays in sync automatically.

OAuth state (clients, auth codes, tokens) stays in a separate SQLite file (`oauth.db`), intentionally independent from the journal database.

See [docs/taxonomy-guide.md](docs/taxonomy-guide.md) for guidance on organizing topics.

## Architecture

![System architecture](docs/diagrams/system-architecture.svg)

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Documentation

| Doc | What's in it |
|-----|-------------|
| [Architecture](docs/architecture.md) | System design, data model, deployment stack, data flow |
| [Tools Reference](docs/tools-reference.md) | All 13 MCP tools with parameters, return values, examples |
| [Deployment Guide](docs/deployment.md) | Docker, nginx, SSL, secrets, OAuth setup |
| [Design Philosophy](docs/philosophy.md) | Why append-only, why not RAG, why PostgreSQL |
| [Taxonomy Guide](docs/taxonomy-guide.md) | How to organize topics, naming conventions, migration strategy |

## Project structure

```
journalctl/
├── journalctl/                # Python package
│   ├── main.py                #   FastAPI app, MCP mount, OAuth wiring
│   ├── config.py              #   Pydantic settings (JOURNAL_* env vars)
│   ├── core/                  #   AppContext, structlog, validation
│   ├── middleware/            #   ASGI auth + path normalization
│   ├── storage/               #   asyncpg pool + pgvector EmbeddingService
│   │   └── repositories/      #     topics, entries, conversations, search
│   ├── models/                #   Pydantic models
│   ├── tools/                 #   13 MCP tool implementations
│   └── oauth/                 #   OAuth 2.1 + DCR provider for browser clients (self-host)
├── tests/                     # pytest-asyncio, session-scoped PG pool fixture
└── deployment/                # Dockerfile, entrypoint.sh, nginx.conf
```

## Key design decisions

- **PostgreSQL is the canonical store.** `topics`, `conversations`, `entries`, `messages`, and `entry_embeddings` all live in one database. `tsvector` generated columns give FTS with zero application-side index sync. `pgvector` HNSW powers semantic search on the same table.
- **Append-only.** No compaction, compression, or auto-deletion. Soft delete only.
- **Raw ASGI auth middleware.** BaseHTTPMiddleware buffers responses and breaks SSE streaming.
- **OAuth 2.1 + API keys.** CLI/desktop clients use Bearer tokens. Browser/mobile clients use OAuth (PKCE + RFC 7591 DCR via the MCP SDK routes). OAuth state stays in its own SQLite file, separate from the journal database.
- **Worker-owned pools.** Each gunicorn worker creates its own asyncpg pool in its lifespan (no `--preload`), because asyncpg pools cannot survive `os.fork()`.
- **No in-process git.** No GitPython, no cross-process locking. Backup strategy is your choice.
- **gosu Docker pattern.** Container starts as root, detects bind mount owner UID, drops to non-root.
- **Cross-platform neutral.** Any MCP-compatible client connects — Claude (CLI/Desktop/Web/Mobile), ChatGPT (Apps SDK over MCP), Gemini. Provider-neutral memory infrastructure.

## Stack

Python 3.12 · FastAPI · FastMCP · PostgreSQL 17 · pgvector · asyncpg · ONNX embeddings (all-MiniLM-L6-v2) · Docker · nginx

## License

AGPL-3.0-or-later
