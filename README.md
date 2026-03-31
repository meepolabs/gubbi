# journalctl

A self-hosted [MCP](https://modelcontextprotocol.io/) server that gives any LLM a persistent, searchable journal — a personal memory infrastructure layer for AI, stored in SQLite on your own infrastructure.

**Works with any MCP-compatible client** — not tied to any specific LLM provider. Any chat app or CLI tool that supports MCP servers via Bearer token or OAuth 2.0 can connect.

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

The journal is an **append-only ledger**, not a brain. It faithfully stores everything — decisions, conversations, milestones, research — and never compresses, forgets, or consolidates. No data leaves your infrastructure.

## Quick start

**Prerequisites:** Docker, a server (GCP, AWS, VPS, etc.), a domain with DNS configured.

```bash
git clone https://github.com/user/journalctl.git && cd journalctl
doppler setup    # or create a .env file
docker compose up -d
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

For browser-based clients, the server supports OAuth 2.0 with PKCE — connect via your client's MCP integrations settings.

## What the LLM can do

| Tool | What it does |
|------|-------------|
| `journal_briefing` | Loads your profile, this week's activity, and recent topics at conversation start |
| `journal_append` | Adds a dated entry to any topic |
| `journal_read` | Reads a topic's full history or last N entries |
| `journal_search` | Hybrid FTS5 + semantic search across everything |
| `journal_save_conversation` | Archives a full chat transcript with summary |
| `journal_timeline` | Shows all activity for a week, month, or year |

Plus 7 more tools for topic management, conversation browsing, entry editing, deletion, and index maintenance. See [docs/tools-reference.md](docs/tools-reference.md) for the complete reference.

## How data is organized

```
data/
├── journal.db                       # Canonical store — topics, entries, conversations, FTS5 index
├── memory.db                        # Semantic embeddings (ONNX, sqlite-vec)
├── oauth.db                         # OAuth tokens/clients (if OAuth enabled)
├── conversations_json/              # Archived conversation transcripts (JSON)
│   └── {id}.json
└── knowledge/
    └── user-profile.md              # Your identity profile, loaded by journal_briefing
```

All journal data lives in `journal.db` (SQLite — canonical store). Semantic search uses `memory.db` with a local ONNX embedding model. The FTS5 index is rebuildable at any time using `journal_reindex`.

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
| [Design Philosophy](docs/philosophy.md) | Why append-only, why not RAG, FTS5 + semantic search |
| [Taxonomy Guide](docs/taxonomy-guide.md) | How to organize topics, naming conventions, migration strategy |

## Project structure

```
journalctl/
├── journalctl/                # Python package
│   ├── main.py                #   FastAPI app, MCP mount, OAuth wiring
│   ├── config.py              #   Pydantic settings (JOURNAL_* env vars)
│   ├── middleware/             #   ASGI auth + path normalization
│   ├── storage/               #   SQLite canonical storage + FTS5 index
│   ├── models/                #   Pydantic models + input validation
│   ├── tools/                 #   13 MCP tool implementations
│   ├── memory/                #   ONNX embeddings + sqlite-vec integration
│   └── oauth/                 #   OAuth 2.0 provider for browser clients
├── tests/                     # 156 tests (pytest-asyncio)
└── deployment/                # Dockerfile, entrypoint.sh, nginx.conf
```

## Key design decisions

- **SQLite is the canonical store.** `journal.db` holds all topics, entries, and conversations. The FTS5 virtual table inside it is rebuildable with `journal_reindex`.
- **Append-only.** No compaction, compression, or auto-deletion. Git preserves every version.
- **Raw ASGI auth middleware.** BaseHTTPMiddleware buffers responses and breaks SSE streaming.
- **OAuth 2.0 + API keys.** CLI/desktop clients use Bearer tokens. Browser/mobile clients use OAuth.
- **No in-process git.** No GitPython, no cross-process locking. Backup strategy is your choice.
- **gosu Docker pattern.** Container starts as root, detects bind mount owner UID, drops to non-root.
- **Cross-platform neutral.** Any MCP-compatible client connects — Claude, Gemini, ChatGPT (via REST wrapper). Provider-neutral memory infrastructure.

## Stack

Python 3.12 · FastAPI · FastMCP · SQLite FTS5 (WAL mode) · ONNX embeddings · sqlite-vec · Docker · nginx

## License

MIT
