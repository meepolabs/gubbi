# gubbi -- CODEMAP

> Module-level navigation guide for agents and humans working in this repo.
> For deep file-level detail read the modules directly. For self-host
> deployment see [`docs/deployment.md`](./docs/deployment.md).

## What this is

`gubbi` is the AGPL-3.0 MCP server + REST API behind gubbi.ai: a
journal-as-a-service for agents. Python 3.12, FastAPI + FastMCP,
PostgreSQL 17 + pgvector, ONNX-backed semantic search. Self-hostable as
a single container or co-deployable with the hosted control plane.

## Three deploy shapes

`gubbi/config.py` enforces these at startup via the `_validate_deploy_shape`
model validator:

| Mode | Trigger | Auth | Used by |
|---|---|---|---|
| 1 -- API-key self-host | API-key only | static Bearer | single operator |
| 2 -- OAuth self-host | bcrypt password hash | DCR + bcrypt login | single operator + multiple MCP clients |
| 3 -- Hosted (multi-tenant) | external IdP wired in | Hydra OAuth 2.1 | gubbi.ai prod |

Per-key env-var triggers and required combinations are catalogued in
the operator's deployment env registry; see [`docs/deployment.md`](./docs/deployment.md)
for the public Mode 1/2 self-host wiring.

Mode 3 sits behind a separate gateway (proprietary, not in this repo)
that forwards via the trust-gateway header path.

## Where it fits

- **Inbound:** MCP clients (Claude, Cursor, ChatGPT) hit `/mcp` as
  authenticated bearer tokens; REST callers hit `/api/v1/*`.
- **Outbound:** PostgreSQL (RLS-enforced runtime pool + admin pool),
  Redis (Arq queue for extraction), Anthropic API (extraction worker
  LLM calls).
- **Sibling:** [`gubbi-common`](https://github.com/meepolabs/gubbi-common)
  -- AGPL shared library, pinned via git tag. Owns audit Action enum,
  audit SQL helpers, HMAC gateway-signature envelope, telemetry
  allowlist, `user_scoped_connection`, bound logger.

## Repo layout

```
gubbi/
  pyproject.toml          Poetry, Python ~3.12
  alembic.ini             migrations under gubbi/alembic/
  docker-compose.yml      gubbi + postgres (bind-mounted ./data/)
  deployment/             Dockerfile (builds the gubbi MCP server image -- used by docker-compose.yml and the prod Kamal deploy), entrypoint, nginx snippet;
                          deployment/scripts/ holds init.sql, grants.sql,
                          restore-db.sh, verify-db-invariants.sh
  tools/                  standalone helpers (env-contract checker)
  gubbi/                  Python package (see below)
  tests/                  unit / integration / e2e
```

## Python package modules (`gubbi/`)

Lifespan-driven FastAPI app. `main.py` wires the modules below in a
fixed order. Tools share an `AppContext` dataclass (pools, embedding
service, settings, cipher, operator UUID).

| Module | What it does |
|---|---|
| `main.py` | FastAPI lifespan + MCP mount + ASGI middleware composition. Single entry point for both `streamable-http` and `stdio` transports. The deployment target is the module-level `server: ASGIApp = CorrelationIDMiddleware(MCPPathNormalizer(app))` wrapper; FastAPI's `user_middleware` list is kept empty so no `BaseHTTPMiddleware` can buffer SSE responses. `app.middleware_stack` is invalidated after `FastAPIInstrumentor.instrument_app(app)` so Kind=Server OTel spans fire correctly. |
| `mcp_validation.py` | `JournalFastMCP` subclass + `format_validation_error` helper. Maps pre-body pydantic `ValidationError` (raised by the MCP SDK before the tool body's try/except runs) to the canonical `validation_error` envelope so clients see a uniform `success=False, error_code=VALIDATION_ERROR` shape for both schema-validation and in-body failures. `main.create_mcp_server` instantiates this subclass instead of bare `FastMCP`. |
| `app_state.py` | Typed accessors for `request.app.state.*` (replaces the legacy `CustomFastAPI` subclass). `require_<field>` raises if lifespan didn't install; `get_optional_<field>` for graceful-degrade paths. |
| `app_context.py` | `AppContext` frozen dataclass (pool, admin_pool, embedding_service, settings, logger, cipher, operator_user_id). Captured by every tool's `register(mcp, app_ctx)` closure. |
| `auth_context.py` | `current_user_id` ContextVar. Set by middleware, read by `db.user_scoped_connection`. |
| `config.py` | pydantic-settings; nested config groups (`DbConfig`, `AuthConfig`, `ServerConfig`, `LLMConfig`) with flat-env compat shim. The 3-shape deploy validator lives here. |
| `validation.py` | All input validation: `validate_topic`, `validate_date`, `sanitize_freetext`, `slugify`, `local_today`, `reject_tool_call_syntax`. Called by every tool before storage. |
| `constants.py` | Cross-package constants (token TTLs, scope name, advisory-lock keys). |
| `bootstrap/` | Lifespan helpers extracted from `main.lifespan`: trust-gateway bind-address safety check, MCP middleware assembly, OAuth wiring, gateway HMAC secret decode. |
| `auth/` | `hydra.py` (introspection client, TTL cache, JIT email lookup). `strategies.py` (auth strategy Protocol + four impls: `TrustGatewayStrategy`, `ApiKeyStrategy`, `HydraStrategy`, `SelfHostStrategy`). `scope.py` (OAuth scope parser). |
| `oauth/` | Self-host OAuth (Mode 2): MCP Python SDK provider impl, SQLite-backed storage for clients/codes/tokens, bcrypt login form, Mode-3 disabled stubs, well-known metadata payload builders. |
| `middleware/` | ASGI middleware: `BearerAuthMiddleware` (strategies dispatched per-request), `MCPPathNormalizer`, `OriginValidationMiddleware`, `CorrelationIDMiddleware`. |
| `audit/` | Audit log writer. `record_audit(conn, ...)` + `@audited` decorator. Action constants re-exported from gubbi-common. Append-only by DB trigger. |
| `crypto/` | App-layer AES-256-GCM. `cipher.py` (`ContentCipher`, key-version-in-nonce, `load_master_keys_from_env`). `guard.py` (`require_cipher` fail-fast for tool entry points). |
| `core/` | PEP 562 deprecation shim. Old import paths under `core/*` still work but new code uses the canonical homes (`audit/`, `crypto/`, top-level `auth_context.py` etc). Will be removed; do not add new symbols here. |
| `users/bootstrap.py` | `scaffold_operator(admin_pool, email, tz)` -- idempotent operator users row in Modes 1/2. |
| `storage/` | `pg_setup.py` (asyncpg pool init + advisory locks), `embedding_service.py` (ONNX MiniLM-L6-v2 + pgvector), `knowledge.py` (filesystem reader for user-profile.md), `repositories/` (all SQL: topics, entries, conversations, search). Repo functions take `conn: asyncpg.Connection` first, encrypt/decrypt via injected cipher. |
| `models/` | Plain dataclasses: `TopicMeta`, `Entry`, `Message`, `ConversationMeta`, `SearchResult`. |
| `tools/` | MCP tool handlers grouped by surface. Each module exports `register(mcp, app_ctx)`. `registry.py` calls them in order: topics -> entries -> search -> conversations -> context. `admin.py` is a library function (not registered), used by future admin API. |
| `api/v1/` | REST endpoints for cloud-side gateway forwarding: `auth.py`, `ingest.py`, `extraction.py`. |
| `extraction/` | Arq worker for conversation -> entry extraction. `service.py` orchestration, `worker.py` Arq settings (provider factory registry keyed off the LLM-provider config field), `jobs/` (single-conversation job; emits `extraction.job` + `extraction.llm_call` OTel spans; `_resolve_provider_attrs` helper resolves provider/model names for span attributes), `llm/` (provider Protocol + `AnthropicProvider` real impl + `FakeLLMProvider` test stub for testbench D-tier; failure markers `FAKE_LLM_FAIL` / `FAKE_LLM_FAIL_PERMANENT`), `prompts/` (categorize + extract markdown templates). |
| `telemetry/` | OTel span/metric/log helpers. `attrs.py` (canonical attribute names including `SpanNames.EXTRACTION_JOB` + `SpanNames.EXTRACTION_LLM_CALL`), `metrics.py` (Prometheus counters/histograms + `rebind_metrics_after_configure` hook), `spans.py` (gubbi-common allowlist wrapper), `logger.py` (structlog -> OTel logs bridge). Three "orphan" counters (`orphan_cleanup`, `extract_conversation`, `anthropic_provider`) use `@lru_cache` factories deferred to first call; `rebind_metrics_after_configure` clears and re-primes all four counter factories after lifespan `configure_otel` so they bind to the SDK provider, not the import-time NoOp. Adding a new orphan counter requires wiring BOTH this rebind hook AND the autouse test fixture. |
| `scripts/` | Operational scripts shipped with the package: encryption-key rotation. |
| `alembic/` | DB migrations. Raw SQL via `op.execute`. Migration DSN resolved by env.py fallback chain (MIGRATION -> ADMIN -> APP). |
```

## Entry points

- **HTTP server:** `gubbi.main:server` -- gunicorn `--workers 2 --worker-class uvicorn.workers.UvicornWorker`. Each worker creates its own asyncpg pools (no `--preload`; asyncpg cannot survive `os.fork()`).
- **stdio:** `gubbi.main:main()` with the stdio transport selected via config -- builds the same AppContext and runs FastMCP over stdin/stdout.
- **Migrations:** `alembic upgrade head` against the migration DSN (resolved via the env.py fallback chain).
- **Reindex script:** `gubbi.scripts.rotate_encryption_key` (and the deployment-side backfills).

## Storage shape (live schema; see migrations for evolution)

Six tenant tables (RLS-enforced via `journal_app` role) + audit log
(append-only by trigger):

- `topics` -- categories, identity by `path`
- `entries` -- dated records; content + reasoning encrypted
- `conversations` -- saved transcripts; title + summary encrypted
- `messages` -- per-turn rows; content encrypted
- `entry_embeddings` -- pgvector(384) ON DELETE CASCADE
- `users` -- one row per tenant; partial UNIQUE on email WHERE NOT deleted
- `audit_log` -- append-only; UPDATE/DELETE blocked by trigger

Five column pairs are AES-256-GCM at the app layer
(`*_encrypted BYTEA NOT NULL`, `*_nonce BYTEA NOT NULL CHECK(=12)`).
Key version is encoded in nonce byte 0; `ContentCipher.encrypt` uses the
highest-numbered key, `decrypt` selects by nonce-byte. `search_vector`
is a regular tsvector populated inline by repo SQL via
`to_tsvector('english', $N)` from ephemeral plaintext binds -- prose
never lands in a column.

## Cross-repo deps

| Direction | Repo | What |
|---|---|---|
| imports | [`gubbi-common`](https://github.com/meepolabs/gubbi-common) | `audit.actions.Action`, `audit.sql.AUDIT_INSERT_*_SQL`, `auth.gateway_signature`, `auth.bearer_challenge`, `db.user_scoped_connection`, `middleware.correlation`, `telemetry.allowlist`, `telemetry.correlation_processor` (CorrelationSpanProcessor stamps correlation_id onto every span), `bootstrap.pg_log_probe` |
| trust-gateway producer | upstream cloud control plane | accepts an `X-Auth-User-Id` envelope verified by HMAC against a shared trust-gateway secret when trust-gateway mode is enabled; the strategy list in lifespan reduces to `TrustGatewayStrategy` only |

`gubbi` does NOT import the cloud control plane and does NOT write to
its tables.

## Deeper docs

- [`README.md`](./README.md) -- install, config, quick start
- [`docs/deployment.md`](./docs/deployment.md) -- self-host deploy guide (public AGPL surface)
- [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`CLA.md`](./CLA.md), [`LICENSE`](./LICENSE)
- Tests: `tests/unit/`, `tests/integration/`, `tests/e2e/`. Integration
  + e2e require `TEST_DATABASE_URL` (default
  `postgresql://journal:testpass@localhost:5433/journal_test`); skip
  gracefully if unreachable.
