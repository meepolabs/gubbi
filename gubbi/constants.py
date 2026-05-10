"""Named timeout and limit constants for the gubbi package."""

from typing import Final

# -- HTTP / network timeouts ---------------------------------------------------

HTTPX_CONNECT_TIMEOUT_SECS: Final = 5  # httpx client connect timeout (seconds)
DB_HEALTH_TIMEOUT_SECS: Final = 5  # DB health-probe timeout (seconds, forward-looking)
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

# -- Pagination -----------------------------------------------------------------

DEFAULT_PAGE_SIZE: Final = 10  # Default repository pagination size
