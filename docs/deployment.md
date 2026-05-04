# Deployment Guide (self-host)

> **Hosted deploys:** if you are operating the multi-tenant hosted variant
> (Mode 3), see `gubbi-cloud/docs/RUNBOOK.md` in the private cloud
> repo. This document covers the two self-host shapes only.

Gubbi supports three mutually-exclusive deploy shapes, selected by
which of `JOURNAL_HYDRA_ADMIN_URL` and `JOURNAL_PASSWORD_HASH` are set.
Setting both is a configuration error and fails startup.

| Shape | `JOURNAL_HYDRA_ADMIN_URL` | `JOURNAL_PASSWORD_HASH` | API key | Hydra | Self-host OAuth |
|-------|---------------------------|-------------------------|---------|-------|-----------------|
| Mode 1: API-key-only self-host | empty | empty | yes | -- | -- |
| Mode 2: Full self-host         | empty | set   | yes | -- | yes |
| Mode 3: Multi-tenant hosted    | set   | empty | no  | yes | -- |

This document covers **Mode 1** and **Mode 2** only.

### Mode 1: API-key-only self-host

Simplest surface. `JOURNAL_API_KEY` is the only accepted credential.
Useful when every client supports static Bearer tokens (Claude Code,
Claude Desktop, Cursor, Aider).

### Mode 2: Full self-host (this document, primary path)

One operator, one user identity. Uses the MCP SDK's built-in OAuth 2.1 +
PKCE + RFC 7591 Dynamic Client Registration routes for OAuth clients
(claude.ai, ChatGPT) AND accepts the static API key for Bearer-token
clients. One container, one PostgreSQL.

Set `JOURNAL_PASSWORD_HASH` (bcrypt hash of the operator's password) to
activate self-host OAuth.

The rest of this document covers the **full self-host** path. The Mode 1
variant is identical except `JOURNAL_PASSWORD_HASH` stays unset and the
OAuth section below is irrelevant.

---

## Prerequisites

- A server with Docker and Docker Compose (any small VPS will do; see memory requirements below)
- A domain with DNS configured
- A secrets manager of your choice, or a local `.env` file

## Repository layout

One repo on the server:

```
~/gubbi/           # This repo -- server code, docker-compose.yml, Dockerfile, data/
    +-- data/           # Bind-mounted persistent data (see below)
        +-- postgres/   #   PostgreSQL WAL + tablespaces  -> /var/lib/postgresql/data
        +-- journal/    #   oauth.db, knowledge/, conversations_json/  -> /app/journal
        +-- onnx/       #   Cached ONNX embedding model  -> /home/appuser/.cache/gubbi
```

All persistent state lives under `./data/` in the repo root. Nothing lives in named Docker volumes -- `docker compose down -v` is safe.

## Environment variables

All configuration is via `JOURNAL_*` environment variables. In production, use a secrets manager; for local dev, a `.env` file works.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `JOURNAL_API_KEY` | yes (Mode 1 + Mode 2) | Static Bearer token for API key auth. Must be >=32 chars. | `sk-journal-...` |
| `JOURNAL_DB_SUPERUSER_PASSWORD` | yes | PostgreSQL superuser bootstrap password (initdb + healthcheck). The app never connects as this role. | (random) |
| `JOURNAL_DB_APP_PASSWORD` | yes | Password for the runtime database role (`journal_app`, RLS-enforced) created by migration 0002. Set via `ALTER ROLE ... WITH PASSWORD '<value>'`. | (random, distinct) |
| `JOURNAL_DB_ADMIN_PASSWORD` | yes | Password for the privileged database role (`journal_admin`, BYPASSRLS) created by migration 0002. | (random, distinct) |
| `JOURNAL_ENCRYPTION_MASTER_KEY_V1` | yes | Base64-encoded 32-byte master key for AES-256-GCM at-rest encryption of `entries.content`, `entries.reasoning`, `messages.content`, `conversations.title`, `conversations.summary`. Generate via `python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"`. Additional versions (`_V2`, `_V3`, ...) can coexist for key rotation. | (32 random bytes, base64) |
| `JOURNAL_TIMEZONE` | yes | Timezone for "today" defaulting and briefing week math | `America/Los_Angeles` |
| `JOURNAL_OPERATOR_EMAIL` | yes (Mode 1 + Mode 2) | Email of the single operator user. A `users` row is auto-scaffolded at startup, and the operator's UUID is bound to every API-key / self-host OAuth request. | `you@example.com` |
| `JOURNAL_SERVER_URL` | for self-host OAuth | Public HTTPS URL advertised in the OAuth + RFC 7591 DCR metadata endpoints | `https://journal.yourdomain.com` |
| `JOURNAL_PASSWORD_HASH` | for self-host OAuth (Mode 2 only) | Bcrypt hash of the single operator's password. Setting this activates the self-host OAuth server (authorize/token/register/revoke + login form). Empty keeps the server API-key-only (Mode 1). | `$2b$12$...` |

`JOURNAL_DB_APP_URL`, `JOURNAL_DB_ADMIN_URL`, and `JOURNAL_DB_MIGRATION_URL` are composed inside `docker-compose.yml` from the three password values above -- configure those in your secrets manager, not the full DSNs. The runtime pool uses `journal_app` (RLS-enforced, no DDL); alembic resolves its DSN from `JOURNAL_DB_MIGRATION_URL` (preferred) or `JOURNAL_DB_ADMIN_URL`, both pointing at the privileged `journal_admin` role. The OAuth SQLite file lives at `<JOURNAL_DATA_DIR>/oauth.db` and needs no separate config. Edit `data/journal/knowledge/user-profile.md` to set your identity profile.

Generate a bcrypt password hash via the built-in CLI:

```bash
docker compose exec gubbi python -m gubbi.oauth.crypto 'your-password'
```

### Operator provisioning

Operator row is auto-scaffolded by the app on Mode 1 / Mode 2 startup. No manual
provisioning step is needed -- after `alembic upgrade head`, just start the
server. The operator email from `JOURNAL_OPERATOR_EMAIL` is resolved to a
concrete UUID at startup in the app lifespan.

## Docker Compose

The stack is two services: `postgres` (pgvector/pgvector:pg17) and `gubbi`. Both use bind mounts under `./data/`.

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: gubbi-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: journal
      POSTGRES_USER: journal
      POSTGRES_PASSWORD: ${JOURNAL_DB_SUPERUSER_PASSWORD}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./deployment/scripts/init.sql:/docker-entrypoint-initdb.d/01-extensions.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U journal -d journal"]
      interval: 10s

  gubbi:
    build:
      context: .
      dockerfile: deployment/Dockerfile
    container_name: gubbi
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "127.0.0.1:8100:8100"   # loopback only -- nginx fronts it
    environment:
      - JOURNAL_API_KEY
      - JOURNAL_DB_APP_URL=postgresql://journal_app:${JOURNAL_DB_APP_PASSWORD}@postgres:5432/journal
      - JOURNAL_DB_ADMIN_URL=postgresql://journal_admin:${JOURNAL_DB_ADMIN_PASSWORD}@postgres:5432/journal
      - JOURNAL_DB_MIGRATION_URL=postgresql://journal_admin:${JOURNAL_DB_ADMIN_PASSWORD}@postgres:5432/journal
      - JOURNAL_ENCRYPTION_MASTER_KEY_V1
      - JOURNAL_DATA_DIR=/app/journal
      - JOURNAL_TRANSPORT=streamable-http
      - JOURNAL_TIMEZONE=${JOURNAL_TIMEZONE:-America/Los_Angeles}
      - JOURNAL_OPERATOR_EMAIL
      - JOURNAL_SERVER_URL
      - JOURNAL_PASSWORD_HASH
    volumes:
      - ./data/journal:/app/journal
      - ./logs:/app/logs
      - ./data/onnx:/home/appuser/.cache/gubbi
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

PostgreSQL will come up first, then gubbi waits for the healthcheck before starting. Before the very first `docker compose up`, run `alembic upgrade head` against the database to create the schema; alembic owns all schema changes. Subsequent restarts do not require re-running migrations unless a new revision has been added.

### Docker permissions (the gosu pattern)

The gubbi container starts as root, reads the UID/GID of the bind mount owner (`/app/journal`), then drops to a matching non-root user via `gosu`. Same pattern used by the official PostgreSQL and Redis images.

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

Rate limits below are placeholders -- tune them for your traffic and threat model, and keep the values out of version control if you can.

```nginx
# Tune these for your environment.
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=<CHOOSE_A_LOW_VALUE>;
limit_req_zone $binary_remote_addr zone=oauth_limit:10m rate=<CHOOSE_A_HIGHER_VALUE>;

upstream gubbi {
    server 127.0.0.1:8100;
}

server {
    server_name journal.yourdomain.com;

    # Security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # OAuth endpoints -- auth handled by the app
    location ~ ^/(\.well-known/(oauth-authorization-server|oauth-protected-resource)|authorize|token|register|revoke) {
        limit_req zone=oauth_limit burst=<N> nodelay;
        proxy_pass http://gubbi;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Login endpoint -- stricter rate limit than OAuth
    location = /login {
        limit_req zone=login_limit burst=<N> nodelay;
        proxy_pass http://gubbi/login;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # MCP endpoint -- Bearer/OAuth auth handled by the app
    # Prefix match catches both /mcp and /mcp/ -- path normalization
    # handled by MCPPathNormalizer middleware in the application
    location /mcp {
        proxy_pass http://gubbi;
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
        proxy_pass http://gubbi/health;
    }

    # UI placeholder -- a custom viewer will land in a later phase
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

### OAuth 2.1 + PKCE + RFC 7591 Dynamic Client Registration (browser/mobile clients, Mode 2 only)

For clients that require OAuth rather than a static Bearer token
(claude.ai, ChatGPT, custom web apps), the server implements OAuth 2.1
with PKCE, RFC 7591 Dynamic Client Registration, bcrypt password
verification, CSRF-protected login, and token refresh -- all via the
MCP SDK's built-in auth routes. Enable it by setting
`JOURNAL_PASSWORD_HASH` (bcrypt hash of the single operator's
password).

Clients that support MCP OAuth self-register via `POST /register`, then
walk the standard `/authorize` + `/token` flow against the login page
served by this container.

OAuth state (registered clients, auth codes, access + refresh tokens)
lives in `oauth.db` (SQLite), deliberately separate from the
PostgreSQL journal database.

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
2. **Conversation JSON archives.** `data/journal/conversations_json/` -- rsync or S3 sync. These are the rebuildable source for `conversations` and `messages` tables.

The ONNX model cache (`data/onnx/`) is not worth backing up -- it's re-downloaded automatically on boot if missing.

## Memory requirements

On a small VM (2GB RAM), PostgreSQL + gubbi + nginx run comfortably for a single user (hundreds of topics, thousands of entries). Sizing guidance:

- PostgreSQL idle: ~200MB. Add ~50MB per concurrent pool connection.
- gubbi idle: ~300MB, of which ~100MB is the loaded ONNX model.
- nginx: negligible.

## Restore procedure

Use this procedure when restoring from a PostgreSQL dump (disaster recovery,
data corruption repair, point-in-time recovery from a known-good backup). Two
tools handle the restore:

| Tool | Purpose |
|------|---------|
| `restore-db.sh` | Runs `pg_restore`, then verifies invariants, optionally replays grants. |
| `verify-db-invariants.sh` | Checks that tenant tables have correct privileges, RLS enabled, policies attached, and default privileges set. |

### Taking a compatible backup

Use `pg_dump` with the compress format (`-Fc`) from the `journal_admin` role:

```bash
docker compose exec -T postgres pg_dump -U journal_admin -Fc journal > backups/journal_$(date +%Y%m%d).dump
```

The `-Fc` format supports parallel restore and `--clean --if-exists` (idempotent re-runs).

**Important:** Do NOT pass `--no-acl` to `pg_dump`. The flag strips all access-control-list entries from the dump, which removes the GRANT statements set up by [migration 0002](../gubbi/alembic/versions/20260419_0002_create_db_roles.py) and migration 0009 (the `GRANT journal_app TO journal_admin` admin option). After a restore, all privileges are gone -- tables and RLS policies survive because they are part of the DDL, but the GRANTs are stripped. Gubbi will accept traffic on the first read but return `permission denied for table users`, causing 500 errors until grants are manually replayed.

`--no-owner` is safe and often necessary. On cross-restore scenarios (e.g., restoring a production dump into a dev environment), role names may differ; `--no-owner` avoids ownership conflicts on restored tables and roles.

### Running the restore

```bash
# Required: set the superuser DSN for pg_restore + grant replay
export JOURNAL_DB_SUPERUSER_URL=postgresql://journal_admin:password@localhost:5432/journal

# Standard restore with verification
./deployment/scripts/restore-db.sh backups/journal_20260419.dump
```

Flags:

| Flag | Effect |
|------|--------|
| `--repair-grants` | After restore, replays the canonical GRANT block (migration 0002 + 0009 grants for `journal_app` and `journal_admin`) inside a single transaction. Use this if your dump was taken with `--no-acl` or if you suspect grants were stripped. |
| `--no-verify` | Skip the post-restore invariant check. Discouraged -- use only when you have verified separately with external tooling. |

The restore script uses these flags against `pg_restore`: `--clean --if-exists --no-owner`. A re-run on an existing database drops all objects first (non-destructive of data in the sense that the incoming dump replaces everything cleanly) then loads the dump.

If verification fails **without** `--repair-grants`, the script exits with code 1 and prints which table, role, and privilege are missing:

```
FAIL: users: role=journal_app missing=SELECT
Remedy: run restore-db.sh --repair-grants <dump-file> to replay migration 0002 + 0009 grants.
```

With `--repair-grants`, the script attempts a full fix automatically by replaying grants and re-verifying. This is idempotent -- granting already-granted privileges on an already-healthy database is a no-op.

### When to use `--repair-grants` specifically

Use it when:
1. The source dump was created with `--no-acl`.
2. A manual `GRANT` was run outside migrations and you want the canonical form back.
3. After restoring from a very old backup, to modernize default privileges.

### Reference: who owns what

[Migration 0002](../gubbi/alembic/versions/20260419_0002_create_db_roles.py) sets up two database roles:

- **`journal_app`**: runtime role, RLS-enforced. Granted `SELECT, INSERT, UPDATE, DELETE` on all current and future tenant tables in the public schema (migration 0002 + ALTER DEFAULT PRIVILEGES).
- **`journal_admin`**: privileged admin role, BYPASSRLS. Granted `ALL PRIVILEGES` on database/schema/tables/sequences (migration 0002). Also granted membership on `journal_app` with ADMIN OPTION (migration 0009).

Restoring a dump strips these GRANTs if `--no-acl` was used during backup. The `--repair-grants` flag replays them exactly as migration 0002 + 0009 define them, ensuring idempotent correctness.
