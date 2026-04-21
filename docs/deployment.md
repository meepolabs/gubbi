# Deployment Guide

## Prerequisites

- A server with Docker and Docker Compose (any small VPS will do; see memory requirements below)
- A domain with DNS configured
- A secrets manager of your choice, or a local `.env` file

## Repository layout

One repo on the server:

```
~/journalctl/           # This repo — server code, docker-compose.yml, Dockerfile, data/
    └── data/           # Bind-mounted persistent data (see below)
        ├── postgres/   #   PostgreSQL WAL + tablespaces  → /var/lib/postgresql/data
        ├── journal/    #   oauth.db, knowledge/, conversations_json/  → /app/journal
        └── onnx/       #   Cached ONNX embedding model  → /home/appuser/.cache/journalctl
```

All persistent state lives under `./data/` in the repo root. Nothing lives in named Docker volumes — `docker compose down -v` is safe.

## Environment variables

All configuration is via `JOURNAL_*` environment variables. In production, use a secrets manager; for local dev, a `.env` file works.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `JOURNAL_API_KEY` | yes | Static Bearer token for API key auth. Must be ≥32 chars. | `sk-journal-...` |
| `JOURNAL_POSTGRES_PASSWORD` | yes | PostgreSQL bootstrap password (initdb + healthcheck). | (random) |
| `JOURNAL_APP_PASSWORD` | yes | Password for the runtime database role created by migration 0002. Set via `ALTER ROLE ... WITH PASSWORD '<value>'`. | (random, distinct) |
| `JOURNAL_ADMIN_PASSWORD` | yes | Password for the privileged database role created by migration 0002. | (random, distinct) |
| `JOURNAL_TIMEZONE` | yes | Timezone for "today" defaulting and briefing week math | `America/Los_Angeles` |
| `JOURNAL_SERVER_URL` | for OAuth | Public HTTPS URL used in OAuth metadata endpoints | `https://journal.yourdomain.com` |
| `JOURNAL_OWNER_PASSWORD_HASH` | for OAuth | Bcrypt hash for OAuth login. Empty string disables OAuth. | `$2b$12$...` |
| `JOURNAL_OAUTH_DB_PATH` | for OAuth | Path to OAuth SQLite DB inside container | `/app/journal/oauth.db` |

`JOURNAL_DATABASE_URL` and `JOURNAL_DATABASE_URL_ADMIN` are composed inside `docker-compose.yml` from the role passwords above -- configure the three password values in your secrets manager, not the full DSNs. Edit `data/journal/knowledge/user-profile.md` to set your identity profile.

Generate a bcrypt password hash via the built-in CLI:

```bash
docker compose exec journalctl python -m journalctl.oauth.crypto 'your-password'
```

## Docker Compose

The stack is two services: `postgres` (pgvector/pgvector:pg17) and `journalctl`. Both use bind mounts under `./data/`.

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: journalctl-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: journal
      POSTGRES_USER: journal
      POSTGRES_PASSWORD: ${JOURNAL_POSTGRES_PASSWORD}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./deployment/init.sql:/docker-entrypoint-initdb.d/01-extensions.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U journal -d journal"]
      interval: 10s

  journalctl:
    build:
      context: .
      dockerfile: deployment/Dockerfile
    container_name: journalctl
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "127.0.0.1:8100:8100"   # loopback only — nginx fronts it
    environment:
      - JOURNAL_API_KEY
      - JOURNAL_DATABASE_URL=postgresql://journal_app:${JOURNAL_APP_PASSWORD}@postgres:5432/journal
      - JOURNAL_DATABASE_URL_ADMIN=postgresql://journal_admin:${JOURNAL_ADMIN_PASSWORD}@postgres:5432/journal
      - JOURNAL_JOURNAL_ROOT=/app/journal
      - JOURNAL_TRANSPORT=streamable-http
      - JOURNAL_TIMEZONE=${JOURNAL_TIMEZONE:-America/Los_Angeles}
      - JOURNAL_SERVER_URL
      - JOURNAL_OWNER_PASSWORD_HASH
      - JOURNAL_OAUTH_DB_PATH
    volumes:
      - ./data/journal:/app/journal
      - ./logs:/app/logs
      - ./data/onnx:/home/appuser/.cache/journalctl
    deploy:
      resources:
        limits: { memory: 1024M }
        reservations: { memory: 512M }
```

Bring it up:

```bash
docker compose --env-file .env up -d
# or pipe env through your secrets manager of choice
```

PostgreSQL will come up first, then journalctl waits for the healthcheck before starting. First boot runs the idempotent `schema.sql` bootstrap inside the `setup_schema(pool)` lifespan step.

### Docker permissions (the gosu pattern)

The journalctl container starts as root, reads the UID/GID of the bind mount owner (`/app/journal`), then drops to a matching non-root user via `gosu`. Same pattern used by the official PostgreSQL and Redis images.

```
entrypoint.sh:
  1. Start as root
  2. Detect mount owner: stat -c '%u:%g' /app/journal
  3. usermod/groupmod appuser to match
  4. chown the ONNX cache, logs, and src
  5. Pre-download the ONNX model as appuser (serialized, prevents worker race)
  6. exec gosu appuser gunicorn ...
```

The ONNX pre-download is critical: without it, multiple gunicorn workers race on first boot and one can get a corrupted partial download. The entrypoint exits non-zero on model-load failure so Docker restarts instead of booting degraded.

### Why no `--preload` in gunicorn?

`asyncpg` pools cannot survive `os.fork()`. Each gunicorn worker creates its own pool during its lifespan startup.

## nginx configuration

nginx sits in front of the container and handles SSL termination, routing, rate limiting, and SSE passthrough.

Rate limits below are placeholders — tune them for your traffic and threat model, and keep the values out of version control if you can.

```nginx
# Tune these for your environment.
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=<CHOOSE_A_LOW_VALUE>;
limit_req_zone $binary_remote_addr zone=oauth_limit:10m rate=<CHOOSE_A_HIGHER_VALUE>;

upstream journalctl {
    server 127.0.0.1:8100;
}

server {
    server_name journal.yourdomain.com;

    # Security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # OAuth endpoints — auth handled by the app
    location ~ ^/(\.well-known/(oauth-authorization-server|oauth-protected-resource)|authorize|token|register|revoke) {
        limit_req zone=oauth_limit burst=<N> nodelay;
        proxy_pass http://journalctl;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Login endpoint — stricter rate limit than OAuth
    location = /login {
        limit_req zone=login_limit burst=<N> nodelay;
        proxy_pass http://journalctl/login;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # MCP endpoint — Bearer/OAuth auth handled by the app
    # Prefix match catches both /mcp and /mcp/ — path normalization
    # handled by MCPPathNormalizer middleware in the application
    location /mcp {
        proxy_pass http://journalctl;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Required for SSE streaming (MCP transport)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        chunked_transfer_encoding on;
    }

    # Health check (unprotected)
    location /health {
        proxy_pass http://journalctl/health;
    }

    # UI placeholder — a custom viewer will land in a later phase
    location / {
        return 404;
    }

    # SSL managed by certbot
}

server {
    server_name _;
    listen 80;
    return 301 https://$host$request_uri;
}
```

**Critical:** `proxy_buffering off` is required for the `/mcp` location. Without it, nginx buffers SSE responses and MCP streaming breaks.

## Connecting MCP clients

Any MCP-compatible client can connect using either method:

### API key (Bearer token)

For CLI tools and desktop apps, add the journal MCP server to your client's config:

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

If your client doesn't support `type: "http"` natively, use `mcp-remote` as a local proxy:

```json
{
  "mcpServers": {
    "journal": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://journal.yourdomain.com/mcp/",
        "--header",
        "Authorization: Bearer YOUR_API_KEY"
      ]
    }
  }
}
```

### OAuth 2.0 (browser/mobile clients)

For browser-based clients that require OAuth (rather than a static Bearer token), the server implements OAuth 2.0 with PKCE, bcrypt password verification, CSRF-protected login, and token refresh. Connect via your client's MCP integrations settings — you'll be redirected to a login page on your server.

OAuth state (clients, auth codes, tokens) lives in `oauth.db` (SQLite), deliberately separate from the PostgreSQL journal database.

**Hardening checklist** before exposing OAuth endpoints to the public internet:

- Put strict rate limits on `/login`, `/token`, and `/register` at the reverse proxy.
- Ensure your reverse proxy only accepts requests on the public hostname you actually own.
- Review the OAuth hardening backlog for items that matter at scale.

## Backup strategy

Two things to back up:

1. **PostgreSQL dumps.** `pg_dump -Fc` to an external host or object store on a cron.
   ```bash
   docker compose exec -T postgres pg_dump -U journal -Fc journal \
       > backups/journal_$(date +%Y%m%d).dump
   ```
2. **Conversation JSON archives.** `data/journal/conversations_json/` — rsync or S3 sync. These are the rebuildable source for `conversations` and `messages` tables.

The ONNX model cache (`data/onnx/`) is not worth backing up — it's re-downloaded automatically on boot if missing.

## Memory requirements

On a small VM (2GB RAM), PostgreSQL + journalctl + nginx run comfortably for a single user (hundreds of topics, thousands of entries). Sizing guidance:

- PostgreSQL idle: ~200MB. Add ~50MB per concurrent pool connection.
- journalctl idle: ~300MB, of which ~100MB is the loaded ONNX model.
- nginx: negligible.

For multi-tenant scale, plan for multiple vCPUs and 4+ GB RAM on a dedicated VPS.
