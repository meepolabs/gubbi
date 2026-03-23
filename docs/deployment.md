# Deployment Guide

## Prerequisites

- A server with Docker and Docker Compose (tested on GCP e2-small, 2GB RAM)
- A domain with DNS configured
- [Doppler](https://doppler.com/) for secrets management (or use `.env` files)
- A git repo for journal content backup

## Repository layout

You'll have two repos on the server:

```
~/journalctl/     # This repo — server code, Dockerfile, tests
~/journal/        # Content repo — markdown files, MkDocs config, .gitignore
```

The content repo is mounted into the Docker container as a bind volume. The server code repo contains the application and deployment config.

## Environment variables

All configuration is via `JOURNAL_*` environment variables, managed through Doppler in production:

| Variable | Description | Example |
|----------|-------------|---------|
| `JOURNAL_API_KEY` | Static Bearer token for API key auth | `sk-journal-...` |
| `JOURNAL_HOST_PATH` | Absolute path to journal content on host | `/home/user/journal` |
| `JOURNAL_JOURNAL_ROOT` | Content root inside container | `/app/journal/content` |
| `JOURNAL_DB_PATH` | FTS5 database path inside container | `/app/journal/journal.db` |
| `JOURNAL_TIMEZONE` | Timezone for date handling | `America/New_York` |
| `HOST_UID` | UID of the bind mount owner on host | `1000` |
| `HOST_GID` | GID of the bind mount owner on host | `1000` |
| `JOURNAL_OAUTH_DB_PATH` | OAuth database path | `/app/journal/oauth.db` |
| `JOURNAL_OWNER_PASSWORD_HASH` | Bcrypt hash for OAuth login | `$2b$12$...` |
| `JOURNAL_SERVER_URL` | Public HTTPS URL | `https://journal.yourdomain.com` |

## Docker Compose

```yaml
services:
  journalctl:
    build: .
    ports:
      - "127.0.0.1:8100:8100"
    volumes:
      - ${JOURNAL_HOST_PATH}:/app/journal
      - ./logs:/app/logs
    environment:
      - JOURNAL_JOURNAL_ROOT=/app/journal/content
      - JOURNAL_DB_PATH=/app/journal/journal.db
      - JOURNAL_OAUTH_DB_PATH=/app/journal/oauth.db
      # OAuth (optional)
      - JOURNAL_SERVER_URL
      - JOURNAL_OWNER_PASSWORD_HASH
    # env_file: .env  # or use Doppler: doppler run -- docker compose up

  mkdocs:
    image: squidfunk/mkdocs-material
    ports:
      - "8300:8000"
    volumes:
      - ${JOURNAL_HOST_PATH}:/docs
    command: serve --dev-addr 0.0.0.0:8000
```

### Docker permissions (the gosu pattern)

The container starts as root, reads the UID/GID of the bind mount owner, then drops to a matching non-root user via `gosu`. This is the same pattern used by the official PostgreSQL and Redis Docker images.

```
entrypoint.sh:
  1. Start as root
  2. Detect mount owner: stat -c '%u:%g' /app/journal
  3. Create appuser with matching UID:GID
  4. exec gosu appuser gunicorn ...
```

## nginx configuration

nginx sits in front of both containers and handles SSL termination, routing, rate limiting, and SSE passthrough.

```nginx
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=oauth_limit:10m rate=30r/m;

upstream journalctl {
    server 127.0.0.1:8100;
}

upstream journal_mkdocs {
    server 127.0.0.1:8300;
}

server {
    server_name journal.yourdomain.com;

    # Security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # OAuth endpoints — must come before MkDocs catch-all
    location ~ ^/(\.well-known/(oauth-authorization-server|oauth-protected-resource)|authorize|token|register|revoke) {
        limit_req zone=oauth_limit burst=10 nodelay;
        proxy_pass http://journalctl;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Login endpoint — stricter rate limit
    location = /login {
        limit_req zone=login_limit burst=3 nodelay;
        proxy_pass http://journalctl/login;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # MCP endpoint — Bearer/OAuth auth handled by app
    location /mcp {
        proxy_pass http://journalctl;
        proxy_set_header Host $host;
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

    # MkDocs — browsable journal site, basic auth
    location / {
        auth_basic "Journal";
        auth_basic_user_file /etc/nginx/.htpasswd_journal;
        proxy_pass http://journal_mkdocs/;
        proxy_set_header Host $host;
    }

    # SSL managed by certbot
}
```

**Critical:** `proxy_buffering off` is required for the MCP endpoint. Without it, nginx buffers SSE responses and MCP streaming breaks.

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

For browser-based clients, the server supports OAuth 2.0 with PKCE. Connect via your client's MCP integrations settings. You'll be redirected to a login page on your server.

Generate a bcrypt password hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"
```

Set the result as `JOURNAL_OWNER_PASSWORD_HASH` in your secrets manager.

The OAuth flow:

```
Client  → POST /register          → gets client_id
Client  → GET /authorize           → redirects to /login
User    → enters password           → bcrypt verified, CSRF protected
Server  → generates auth code       → redirects back to client
Client  → POST /token (code+PKCE)  → gets access + refresh tokens
Client  → MCP calls with Bearer    → works
```

OAuth state (clients, auth codes, tokens) lives in `oauth.db`, separate from the disposable FTS5 index.

## Daily sync cron

Set up a cron job to auto-commit and push journal changes:

```bash
# crontab -e
0 3 * * * /path/to/journalctl/scripts/daily_sync.sh >> /var/log/journal-sync.log 2>&1
```

`daily_sync.sh` runs:
1. `python scripts/generate_timeline.py` — generates timeline pages for MkDocs
2. `cd ~/journal && git add -A && git commit -m "daily sync $(date +%Y-%m-%d)" && git push`

## MkDocs (browsable journal)

MkDocs Material runs as a separate container, serving the same journal markdown as a searchable website. Access it at `https://journal.yourdomain.com/` behind HTTP basic auth.

**Current limitation:** The MkDocs dev server doesn't detect cross-container file changes via inotify. The planned fix is to replace the dev server with a static build (`mkdocs build`) served directly by nginx, rebuilt on a cron schedule.

## Memory requirements

On a small VM (2GB RAM), the two containers (journalctl + mkdocs) plus nginx run comfortably. If RAM becomes tight, replacing the MkDocs dev server with static builds reduces memory usage.
