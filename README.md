# journalctl

A self-hosted [MCP](https://modelcontextprotocol.io/) server that gives any LLM a persistent, searchable journal — your life's chronological record, stored as plain markdown on your own infrastructure.

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

The journal is an **append-only ledger**, not a brain. It faithfully stores everything — decisions, conversations, milestones, research — and never compresses, forgets, or consolidates. Your data stays as readable markdown files in a git repo on your own server.

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
| `journal_search` | Full-text search across everything with keyword highlighting |
| `journal_save_conversation` | Archives a full chat transcript with auto-generated summary |
| `journal_timeline` | Shows all activity for a week, month, or year |

Plus 6 more tools for topic management, conversation browsing, entry editing, and index maintenance. See [docs/tools-reference.md](docs/tools-reference.md) for the complete reference.

## How data is organized

```
journal/content/
├── topics/                          # Curated entries
│   ├── hobbies/
│   │   ├── running.md               #   Dated entries, decisions, milestones
│   │   └── woodworking.md
│   ├── projects/
│   │   └── homelab.md
│   └── health/
│       └── fitness.md
│
├── conversations/                   # Full transcripts
│   ├── hobbies/running/
│   │   └── marathon-training-plan.md
│   └── projects/homelab/
│       └── network-setup-session.md
│
└── knowledge/                       # Reference docs
    └── user-profile.md
```

Everything is plain markdown with YAML frontmatter. The FTS5 search index is disposable — delete `journal.db` anytime and rebuild with `journal_reindex`.

See [docs/taxonomy-guide.md](docs/taxonomy-guide.md) for guidance on organizing topics.

## Architecture

![System architecture](docs/diagrams/system-architecture.svg)

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Documentation

| Doc | What's in it |
|-----|-------------|
| [Architecture](docs/architecture.md) | System design, data model, deployment stack, data flow |
| [Tools Reference](docs/tools-reference.md) | All 12 MCP tools with parameters, return values, examples |
| [Deployment Guide](docs/deployment.md) | Docker, nginx, SSL, secrets, OAuth setup |
| [Design Philosophy](docs/philosophy.md) | Why append-only markdown, why not RAG, journal vs memory |
| [Taxonomy Guide](docs/taxonomy-guide.md) | How to organize topics, naming conventions, migration strategy |

## Project structure

```
journalctl/
├── journalctl/                # Python package
│   ├── main.py                #   FastAPI app, MCP mount, stdio/http modes
│   ├── config.py              #   Pydantic settings (JOURNAL_* env vars)
│   ├── middleware/             #   ASGI auth + path normalization
│   ├── storage/               #   Markdown CRUD + FTS5 index
│   ├── models/                #   Pydantic models + input validation
│   ├── tools/                 #   All 12 MCP tool implementations
│   └── oauth/                 #   OAuth 2.0 provider for browser clients
├── tests/                     # 68+ tests (pytest-asyncio)
├── deployment/                # Dockerfile, entrypoint.sh, nginx.conf
└── scripts/                   # Daily git sync, timeline generation
```

## Key design decisions

- **Markdown is source of truth.** SQLite FTS5 is a disposable acceleration layer.
- **Append-only.** No compaction, compression, or auto-deletion. Git preserves every version.
- **Raw ASGI auth middleware.** BaseHTTPMiddleware buffers responses and breaks SSE streaming.
- **OAuth 2.0 + API keys.** CLI/desktop clients use Bearer tokens. Browser/mobile clients use OAuth.
- **External git only.** Daily cron handles commits. No in-process git, no cross-process locking.
- **gosu Docker pattern.** Container starts as root, detects bind mount owner UID, drops to non-root.
- **LLM-agnostic.** Any MCP-compatible client can connect.

## Stack

Python 3.12 · FastAPI · FastMCP · SQLite FTS5 (WAL mode) · Docker · nginx · MkDocs Material

## License

MIT
