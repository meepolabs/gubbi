"""Named timeout and limit constants for the gubbi package."""

from typing import Final

# -- HTTP / network timeouts ---------------------------------------------------

HTTPX_CONNECT_TIMEOUT_SECS: Final = 5  # httpx client connect timeout (seconds)
DB_HEALTH_TIMEOUT_SECS: Final = 5  # DB health-probe timeout (seconds, forward-looking)
# /health/ready probe budget: split acquire vs query so a saturated
# pool fails the readiness check fast (asyncpg pool.acquire blocks
# until a slot is free, query timeout caps the SELECT 1 round-trip).
# Mirrors gubbi-cloud's ``DB_HEALTH_ACQUIRE_TIMEOUT_SECS`` /
# ``DB_HEALTH_QUERY_TIMEOUT_SECS`` so the readiness contract is
# uniform across the gateway and self-host paths.
DB_HEALTH_ACQUIRE_TIMEOUT_SECS: Final = 2
DB_HEALTH_QUERY_TIMEOUT_SECS: Final = 1
# /health/ready Redis ping budget. Mirrors the DB-side split so a
# saturated Redis client surfaces fast instead of holding the readiness
# slot. The PING is a cheap round-trip; 1s is generous against the
# in-cluster Redis we ping.
REDIS_HEALTH_PING_TIMEOUT_SECS: Final = 1
EMBEDDING_REQUEST_TIMEOUT_SECS: Final = (
    120  # Embedding API request timeout -- reserved; currently using ONNX inference
)

# -- Database pool & command timeouts -------------------------------------------

DB_COMMAND_TIMEOUT_SECS: Final = 30  # asyncpg connection-level command timeout (seconds)
APP_POOL_SIZE_MIN: Final = 2  # app-role pool minimum size
APP_POOL_SIZE_MAX: Final = 12  # Bumped (m-h5-h6) above worker max_jobs=10; leaves headroom for
# idempotency probes + audit writes during the connection-split extraction job.
ADMIN_POOL_SIZE_MIN: Final = 2  # admin-role pool minimum size (same bounds as app pool for now)
ADMIN_POOL_SIZE_MAX: Final = 5  # admin-role pool maximum size (same bounds as app pool for now)

# -- Worker / job timeouts ------------------------------------------------------

ARQ_JOB_TIMEOUT_SECS: Final = 600  # Arq worker job timeout (seconds)

# -- LLM retry budgets ----------------------------------------------------------

ANTHROPIC_MAX_RETRIES: Final = 5  # Anthropic SDK retry budget on RateLimitError
ANTHROPIC_REQUEST_TIMEOUT_SECS: Final = 120  # Anthropic SDK per-request timeout (seconds)

# -- Pagination -----------------------------------------------------------------

DEFAULT_PAGE_SIZE: Final = 10  # Default repository pagination size
