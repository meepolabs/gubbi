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
  Redis (Arq queue for extraction), an OpenAI-compatible
  chat-completions endpoint at the configured LLM base URL (extraction
  worker LLM calls).
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
  Makefile                test-stack-up/down/status (tools/testdb/testdb.py
                          with gubbi's profile) and prepush (stack check,
                          then tools/run_db_suites.py; the pre-push hook)
  deployment/             Dockerfile (builds the gubbi MCP server image -- used by docker-compose.yml and the prod Kamal deploy), entrypoint, nginx snippet;
                          deployment/scripts/ holds init.sql, grants.sql,
                          restore-db.sh, verify-db-invariants.sh
  tools/                  standalone helpers (env-contract checker;
                          check_required_needs.py, the required-needs
                          checker the `required` CI job runs -- a byte
                          copy of gubbi-common's, sha256-pinned by
                          tests/unit/test_check_required_needs.py;
                          run_db_suites.py, the stdlib runner for every
                          DB-backed pytest stage CI runs: testdb reset, the
                          integration suites, the coverage gate);
                          tools/testdb/ holds testdb.env (pinned test DB
                          images), bootstrap.sql (test role bootstrap) and
                          testdb.py (stdlib controller: per-checkout docker
                          test stack up/down/status/env, reset (drop roles
                          and DBs, recreate, bootstrap, migrate; local or
                          --ci), check-env, the one testdb.env validator,
                          psql-path, and psql-plan) and
                          bin/psql (docker psql wrapper, put on PATH by
                          psql-path when no host psql of PG_MAJOR exists:
                          docker exec into this checkout's own pg container
                          when psql-plan verifies the target is one, else
                          a throwaway PGVECTOR_IMAGE container)
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
| `config.py` | pydantic-settings; nested config groups (`DbConfig`, `AuthConfig`, `ServerConfig`, `LLMConfig`) with flat-env compat shim. The 3-shape deploy validator lives here. `LLMConfig.base_url` must be https; plain http is accepted only for localhost, loopback, RFC 1918 or unique-local IPv6 hosts. |
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
| `storage/` | `pg_setup.py` (asyncpg pool init + advisory locks), `embedding_service.py` (ONNX MiniLM-L6-v2 + pgvector), `repositories/` (all SQL: topics, entries, conversations, search, plus `stats.py` dashboard aggregates and `setup_signals.py` onboarding EXISTS probes; topics carries the rename/delete-with-reassign/merge mutations and entries carries `list_entries`/`get_entry_by_id`/`move_entries_to_topic` for the web surface, plus the extraction links on insert (`client`, `source_conversation_id`, `duplicate_of_entry_id`), `link_to_conversation`, `lock_dedupe_days` (per user/day/client advisory locks under a shared per-user lock; a run spanning more days than the per-day limit takes the per-user lock exclusively instead) and `list_dedupe_neighbours` (same-day stored entries); conversations carries `lock_processed_at` and the extraction content-hash state; `extraction_jobs.create_pending` / `refresh_queued` write a job's lane, `run_after` and activity time; `mark_running` claims only an admitted pending row whose `run_after` has passed, under a row lock, and returns whether it claimed; `mark_completed` / `mark_failed` add the run's tokens, calls, chunks and cents to the row, and `mark_failed` sets the retry policy's `run_after` and returns the `FailureOutcome` written or None; `finished_attempt` / `create_follow_up` queue a follow-up job, its `activity_at` read from the conversation as saved now, when a conversation was re-saved during a completed run; `extraction_queue.py` is the scheduler's cross-tenant surface over the same table: `first_upload_at`, `supersede_failed`, `select_runnable` (`FOR UPDATE SKIP LOCKED`, the runnable predicate rechecked on the locked row, at most a given number per user), `admit`, `defer_lane`, `revive_failed`, `stale_admitted_pending`, `inflight_admitted_count` / `inflight_by_user`, `fail_enqueue_lost` (returns the `FailureOutcome` or None), and `queue_snapshot` for metrics; `wallet.py` is the per-user LLM wallet over `credit_grants` / `llm_charges`: `available` (monthly allowance left plus credit granted minus credit spent, per user, in one statement; negative credit is repaid first from each later month's allowance, derived by folding per-month totals, never written), `global_used` and `month_usage` (this UTC month's weighted and raw totals across users), `grant_credit` (the one grant write path, idempotent per key) and `charge` (one attempt's charge under a per-user advisory lock, monthly allowance first, then credit, which may go negative)). Repo functions take `conn: asyncpg.Connection` first, encrypt/decrypt via injected cipher. |
| `services/` | Stateless query/composition layer reused across surfaces. `search.py` (`run_journal_search` -- the hybrid FTS+semantic pipeline extracted from the MCP search tool so the tool and the web `/api/v1/search` endpoint share one implementation). |
| `models/` | Plain dataclasses: `TopicMeta`, `Entry`, `Message`, `ConversationMeta`, `SearchResult`. `conversation.keep_conversation_roles` is the user/assistant role filter extraction reads transcripts through. |
| `tools/` | MCP tool handlers grouped by surface. Each module exports `register(mcp, app_ctx)`. `registry.py` calls them in order: topics -> entries -> search -> conversations -> context. `context.py` (`journal_briefing`) emits `BRIEFING_SEED_INSTRUCTIONS` (constants.py) on a truly-empty journal so the calling LLM auto-onboards, and exports the public `resolve_period(period, today)` -- the authority for the period vocabulary, consumed by `storage/repositories/stats.py`, `scripts/dump_period_fixtures.py`, and (re-implemented in TypeScript, pinned to that generator's fixture) the web client. `admin.py` is a library function (not registered), used by future admin API. |
| `api/v1/` | REST endpoints reached via the cloud-api gateway (HMAC `X-Auth-*` envelope verified by `auth.require_scope`). `auth.py` (scope dependency), `ingest.py` (saves conversations and records one queued extraction job per save with its lane and `run_after`; it never enqueues or charges, and reports `budget_exhausted` from the user's wallet), `extraction.py`, `user.py` (`/user/setup-signals` onboarding booleans), and `web/` -- the web read/write REST API (`topics`, `entries` + `entries_admin`, `conversations`, `search`, `stats`, `timeline`, `topic_admin`) sharing `schemas.py` (`PaginatedList`, `DecryptableItem`), `decryption.py` (`decrypt_field` sentinel path -- a corrupt row yields `[decryption failed]` + a flag, never a 500), `pagination.py`, `responses.py` (`private, no-store`), and `errors.py`. Reads require `journal:read`, mutations `journal:write`. |
| `extraction/` | Postgres-authoritative extraction queue + Arq worker for conversation -> entry extraction. `scheduler.py` (`run_extraction_scheduler`, a loop in the HTTP server on the admin pool; each tick, serialized across replicas by a session advisory lock, supersedes replaced failed rows, selects runnable rows, decides admission against the wallet, the global ceiling, the running-job cap and capacity, admits / defers / revives, enqueues to Arq with job id `<job uuid>:<attempts>`, re-enqueues stale admitted-pending rows, fails rows whose enqueue raised as `enqueue_lost`, then records the queue gauges). `intake.py` (pure lane assignment `paste` / `live` / `backfill`, `next_nightly_window`), `admission.py` (pure `decide` over each user's available balance and running-job cap, the global ceiling and dispatch capacity; work is deferred with a `DeferReason` or left queued, never refused), `spend_rules.py` (`SPEND_RULES`, one `SpendRule` per lane: schedule, wallet stop line, share of the global ceiling; `WalletLimits` + `limits_for`, `global_ceiling`, `run_after_for` -- the one accessor for configured wallet policy), `wallet.py` (pure `weighted` usage in weighted tokens, `split_charge` between the monthly allowance and credit, and `balance`, the per-month fold that repays debt from the next allowance), `charging.py` (`settle_attempt` charges a finished attempt in the transaction that ended it and queues a follow-up for a conversation re-saved during the run; `record_charge` counts `extraction.charges.weighted_total` and `record_unmetered` counts `extraction.llm.unmetered_calls_total` by lane after commit), `retry_policy.py` (`ErrorCode` -> transient / deterministic / terminal `RetryClass`, `FailureOutcome` + `failure_outcome`, `next_run_after` backoff). `supervised_loop.py` (`run_every` / `start_supervised` / `stop_supervised` / `stop_supervised_before_teardown` / `report_not_started`: a raising tick or a stopped loop is logged and counted on `background_task.died_total`; `stop_supervised` returns the loops still running at its deadline, counted with `stage=stop_timeout`, and `stop_supervised_before_teardown` gives them one more bounded wait before pools close; runs the scheduler, the orphan sweep and the worker heartbeat). `orphan_cleanup.py` (`run_orphan_cleanup`: fails rows stuck in `running` past the job timeout as `worker_lost` with a backoff `run_after`; pending rows are never swept). `queue_metrics.py` (`record_queue_metrics`, the per-tick queue and budget gauges), `heartbeat.py` (`run_worker_heartbeat`, the `extraction.worker.heartbeat` counter), `failure_metrics.py` (`record_job_failures`, `extraction.jobs.failed_total` by `error_class` / `outcome`). `worker.py` Arq settings (provider factory registry keyed off the LLM-provider config field; `max_tries = 1`, no kept results, no Arq retries -- the scheduler owns every retry; starts the heartbeat at startup; shutdown stops it through `stop_supervised_before_teardown`, then drains `cancel_cleanups` with two bounded waits -- wait, cancel the rest, wait again -- before closing Redis and PG). `jobs/` (single-conversation job: claim via `mark_running` -> role-filtered load -> rule skips -> pipeline -> `sanitize_candidates` -> embed -> persist, with a row-locked `processed_at` recheck before writing; no DB connection is held during LLM calls or embedding; failures map to typed `ErrorCode`s `deadline_exceeded`, `llm_refused`, `llm_truncated`, `llm_schema`, `llm_rate_limited`, `llm_provider_error`, `internal_error` and are counted via `record_job_failures`; cancellation marks the job failed with the running spend; the completion, nothing-to-record and `extraction_job_failed` log lines carry `unmetered_calls`; each attempt is charged to the wallet through `settle_attempt` from the provider-reported usage; a conversation already processed has its row claimed and completed at no charge; emits `extraction.job` + `extraction.llm_call` OTel spans). `pipeline.py` (LLM phase: chunk, optional per-chunk triage, `single` or `two_call` extraction per chunk with prior entries forwarded, split-and-retry on truncation, deadline-bounded per-call time budget; failures raise `PipelineFailure` carrying usage; `run_pipeline(on_spend=...)` reports a `PipelineSpend` running total after every call, incl. unmetered calls). `chunking.py` (pure transcript chunker on estimated tokens), `chunk_extraction.py` (strict JSON schemas + pydantic parsing of chunk outputs), `dating.py` (event-date resolution: model date, then message dates, never the worker clock), `skips.py` (pure rule-based skips + conversation content hash), `dedupe.py` (deterministic exact/near classification against same-day stored entries and earlier candidates; never deletes), `persist.py` (`sanitize_candidates` sanitizes and caps model text as the MCP write path does, dropping empty or tool-call candidates -- the job calls it before embedding; `persist_extraction` writes topic + deduped entries + embeddings + completion in the caller's transaction, raises `ValueError` on unsanitized candidates, and takes the dedupe advisory locks). `service.py` (per-chunk calls `extract_chunk` / `categorize_chunk` / `extract_chunk_entries` / `triage`, each with a keyword-only `time_budget`; exposes `provider_name`, `model_name`, `estimate_cost_cents`). `llm/` (provider Protocol whose `complete` takes `time_budget`, error hierarchy incl. `LLMDeadlineError`; responses and errors carry `unmetered_attempts`, timed-out attempts that report no usage; `OpenAICompatibleProvider` real impl, `FakeLLMProvider` test stub for testbench D-tier with markers `FAKE_LLM_FAIL` / `FAKE_LLM_FAIL_PERMANENT` / `FAKE_LLM_REFUSE` / `FAKE_LLM_TRUNCATE`). `prompts/` (`extract_single`, `categorize_chunk`, `extract_chunk_entries`, `triage` markdown templates). |
| `telemetry/` | OTel span/metric/log helpers. `attrs.py` (canonical attribute names including `SpanNames.EXTRACTION_JOB` + `SpanNames.EXTRACTION_LLM_CALL`), `attrs.GUBBI_METRIC_ALLOWLIST` (per-metric allowed attribute keys for the extraction queue gauges, worker heartbeat, failed-job counter, charges counter and unmetered-calls counter), `metrics.py` (Prometheus counters/histograms/gauges, `record_gauge` / `record_counter` filtered through that allowlist, `record_background_task_died`, + `rebind_metrics_after_configure` hook), `spans.py` (gubbi-common allowlist wrapper), `logger.py` (structlog -> OTel logs bridge), `sanitized_errors.py` (`record_exception_sanitized` / `safe_error_fields` / `is_driver_caused` -- describe a database failure by exception type + shape-validated SQLSTATE so a driver message or DETAIL block never reaches a span or log line; classification walks a bounded `__cause__`/`__context__` chain so a wrapper like `DatabaseUnavailable` is caught too), `asyncpg_sanitized.py` (`instrument_asyncpg_sanitized` wraps `AsyncPGInstrumentor`'s tracer so its per-statement CLIENT spans record a bounded `exception` event instead of the driver's message/stacktrace; wired from `_wire_instrumentors`, and FAIL-CLOSED -- if the tracer cannot be replaced the instrumentation is removed and verified gone, raising `SanitizerUnsafeError` if a wrapper survives, so the process never runs with default leaking spans). Two "orphan" counters (`orphan_cleanup`, `openai_compatible`) use `@lru_cache` factories deferred to first call; `rebind_metrics_after_configure` clears and re-primes `initialize_metrics` and both counter factories after lifespan `configure_otel` so they bind to the SDK provider, not the import-time NoOp. Adding a new orphan counter requires wiring BOTH this rebind hook AND the autouse test fixture. |
| `scripts/` | Operational scripts shipped with the package: encryption-key rotation; `release_held_extraction_jobs` (`--dry-run` / `--release`: counts held rows at `run_after = 'infinity'` by status and error class, and makes them runnable except terminal failures). |
| `alembic/` | DB migrations. Raw SQL via `op.execute`. Migration DSN resolved by env.py fallback chain (MIGRATION -> ADMIN -> APP). Head: `0006_llm_wallet`. |
```

## Entry points

- **HTTP server:** `gubbi.main:server` -- gunicorn `--workers 2 --worker-class uvicorn.workers.UvicornWorker`. Each worker creates its own asyncpg pools (no `--preload`; asyncpg cannot survive `os.fork()`).
- **stdio:** `gubbi.main:main()` with the stdio transport selected via config -- builds the same AppContext and runs FastMCP over stdin/stdout.
- **Migrations:** `alembic upgrade head` against the migration DSN (resolved via the env.py fallback chain).
- **Reindex script:** `gubbi.scripts.rotate_encryption_key` (and the deployment-side backfills).
- **Held extraction jobs:** `python -m gubbi.scripts.release_held_extraction_jobs --dry-run | --release`.
- **Extraction worker:** `arq gubbi.extraction.worker.WorkerSettings`; the queue scheduler and orphan sweep run inside the HTTP server lifespan.

## Storage shape (live schema; see migrations for evolution)

Nine tenant tables (RLS-enforced via `journal_app` role) + audit log
(append-only by trigger):

- `topics` -- categories, identity by `path`
- `entries` -- dated records; content + reasoning encrypted; extracted
  rows carry `client`, `source_conversation_id` and (near-duplicates)
  `duplicate_of_entry_id`
- `conversations` -- saved transcripts; title + summary encrypted;
  `extracted_content_hash` + `extraction_version` record what was last
  extracted so an unchanged conversation is not extracted again
- `messages` -- per-turn rows; content encrypted
- `entry_embeddings` -- pgvector(384) ON DELETE CASCADE
- `extraction_jobs` -- one row per extraction job; `cents_spent` and the
  usage counters (`input_tokens`, `output_tokens`, `cached_input_tokens`,
  `reasoning_tokens`, `llm_calls`, `chunks`) sum every model call, failed
  attempts included; queue columns `lane`, `run_after` (`'infinity'` =
  held), `defer_reason`, `attempts`, `dispatch_failures`, `activity_at`,
  `admitted_at`
- `credit_grants` -- per-user LLM wallet credit, append-only for
  `journal_app` (SELECT + INSERT); unique `idempotency_key`; amount
  bounded per grant; a RESTRICTIVE policy limits `journal_app` inserts
  to the user's one onboarding grant (`onboarding:<user id>`), every
  other grant kind goes through `journal_admin`; cascades on user delete
- `llm_charges` -- one row per charged job attempt (raw and weighted
  tokens, monthly / credit split, UTC month), unique `(job_id,
  attempt)`, append-only for `journal_app`; `user_id` is SET NULL on
  user delete so the deployment-wide monthly total keeps the charge
- `users` -- one row per tenant; partial UNIQUE on email WHERE NOT deleted
- `audit_log` -- append-only; UPDATE/DELETE blocked by trigger. `journal_app`
  has INSERT plus column-level SELECT on the five columns the deduped INSERT's
  ON CONFLICT clause reads, scoped to its own rows by a self-only policy; no
  table-wide SELECT

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
  gracefully if unreachable. Shared extraction harnesses live in
  `tests/fixtures/extraction_job.py` (in-memory job harness) and
  `tests/fixtures/extraction_db.py` (DB seeding). The live extraction
  smoke `tests/integration/test_extraction_live_smoke.py` is skipped
  unless `JOURNAL_LLM_BASE_URL`, `JOURNAL_LLM_MODEL` and
  `GUBBI_LIVE_SMOKE_CONVERSATION` are set.
- CI gate (`.github/workflows/ci.yml`): the terminal `required` job is
  the one check branch protection names; it needs every blocking lane
  (lint + typecheck, the called `security-tests.yml` and `gitleaks.yml`)
  and fails unless all succeeded. Before the verdict it runs
  `tools/check_required_needs.py --self-test` (stdlib, on the runner's
  python3), which fails it when `needs` misses a job or the file leaves
  the one layout the check accepts.
- Tier-1 suite (`.github/workflows/security-tests.yml`, call-only from
  `ci.yml`): env-contract check, a no-service unit job, a
  `testdb-config` job that exports the `tools/testdb/testdb.env` pins
  through `testdb.py check-env`, and the DB job, whose two pgvector
  services take their image from those outputs and whose suites run in
  one `tools/run_db_suites.py --ci` call (reset with `bootstrap.sql`,
  migrate, integration suites, coverage gate). Selection is
  marker-driven -- `-m "not hosted_live"`; `hosted_live` tests (live
  Hydra + Kratos + cloud-api) run in Tier-2 on the testbench, not here.
  Its shape is pinned by `tests/unit/test_ci_workflow_contract.py`.
- Image closure gate (`image-closure` job in `ci.yml`, a lane of
  `required`): `tools/image_closure_needed.py` diffs against the PR base
  or push `before` and skips only when no watched path changed (an
  unreadable base runs it); otherwise installs the hash-pinned Poetry
  closure in the Dockerfile's base image and checks `POETRY_VERSION`
  matches the lock's `poetry` pin.
- Dependency scan (`.github/workflows/dependency-scan.yml`): advisory
  OSV scan of every lockfile via the digest-pinned scanner image; red on
  a finding, required by nothing.
