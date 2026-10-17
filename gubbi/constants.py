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
APP_POOL_SIZE_MAX: Final = 12  # Bumped above worker max_jobs=10; leaves headroom for
# idempotency probes + audit writes during the connection-split extraction job.
ADMIN_POOL_SIZE_MIN: Final = 2  # admin-role pool minimum size (same bounds as app pool for now)
ADMIN_POOL_SIZE_MAX: Final = 5  # admin-role pool maximum size (same bounds as app pool for now)

# -- Worker / job timeouts ------------------------------------------------------

ARQ_JOB_TIMEOUT_SECS: Final = 600  # Arq worker job timeout (seconds)
# Extraction gets its own Arq timeout, sized as chunks x attempts x per-attempt
# LLM timeout: a >100k-token conversation splits into ~4 chunks, each call
# makes up to 3 attempts of ~120 s, so 4 x 3 x 120 s = 1440 s of worst-case
# LLM time. The job starts no new LLM call past the deadline (1440 s plus
# backoff headroom); the 300 s margin up to the Arq timeout holds the tail of
# the call in flight at the deadline and the persist step.
EXTRACTION_JOB_DEADLINE_SECS: Final = 1500
EXTRACTION_JOB_TIMEOUT_SECS: Final = 1800
# Concurrent jobs per extraction worker process.
EXTRACTION_WORKER_MAX_JOBS: Final = 10

# -- LLM retry budgets ----------------------------------------------------------

LLM_MAX_ATTEMPTS: Final = 3  # Total requests per completion, first attempt included
LLM_REQUEST_TIMEOUT_SECS: Final = 120  # Per-attempt timeout (seconds)
LLM_BACKOFF_BASE_SECS: Final = 1.0  # Backoff before the second attempt; doubles per retry
# Ceiling on any single wait. A Retry-After asking for longer than this ends
# the retries instead of stalling the job.
LLM_BACKOFF_CAP_SECS: Final = 60.0
LLM_MAX_OUTPUT_TOKENS: Final = 16_000  # Output-token cap sent with every request

# -- Message text limits -------------------------------------------------------

# Per-message character cap for stored conversation messages.
MAX_MSG_CHARS: Final = 20_000
# Appended inside MAX_MSG_CHARS when a message is cut.
MSG_TRUNCATION_MARKER: Final = "[truncated]"

# -- Pagination -----------------------------------------------------------------

DEFAULT_PAGE_SIZE: Final = 10  # Default repository pagination size

# -- Extraction chunking --------------------------------------------------------

# Characters per estimated token. Deliberately low (real tokenizers average
# closer to 4 for English) so the estimate overshoots rather than undershoots.
CHARS_PER_TOKEN: Final = 3.5
CHUNK_TARGET_TOKENS: Final = 50_000  # Max estimated tokens of transcript per LLM call
CHUNK_OVERLAP_TOKENS: Final = 2_000  # Whole messages repeated from the previous chunk
PRIOR_ENTRIES_MAX_TOKENS: Final = 4_000  # Budget for earlier chunks' entries sent forward
# Already-extracted messages loaded before the new ones as read-only context.
EXTRACTION_TAIL_MESSAGES: Final = 6
# Newest entries of the same conversation sent as prior entries to a re-extraction.
EXTRACTION_THREAD_ENTRIES_LIMIT: Final = 100

# -- Extraction skip rules ------------------------------------------------------

# Bump to re-extract conversations whose content is unchanged since their last
# extraction (stored hash matches but the stored version is older).
EXTRACTION_VERSION: Final = 2
# Conversations whose user turns total fewer stripped characters than this are
# skipped. Kept low enough that a short meaningful line ("I got married", 13
# chars) is still extracted; only greetings / acknowledgements fall below it.
EXTRACTION_MIN_USER_CHARS: Final = 8
# Conversation sources produced by automated or sub-agent sessions, which are
# never extracted. Empty: no ingest source is automated today.
AUTOMATED_EXTRACTION_SOURCES: Final[frozenset[str]] = frozenset()

# -- Extraction dedupe ----------------------------------------------------------

# Cosine similarity (all-MiniLM-L6-v2 embeddings) at or above which an extracted
# entry repeats a same-day entry and is not inserted.
DEDUPE_EXACT_COSINE: Final = 0.97
# At or above this (and below the exact threshold) the entry is inserted but
# flagged as a near-duplicate of the entry it resembles.
DEDUPE_NEAR_COSINE: Final = 0.90
# Above this many distinct event days an extraction run takes one per-user
# dedupe lock instead of a lock per day, so the advisory locks one transaction
# holds in the shared lock table stay bounded whatever dates the model chose.
DEDUPE_PER_DAY_LOCK_MAX_DAYS: Final = 32

# -- Request body limits --------------------------------------------------------

# Largest ingest request body accepted (25 MiB), the same cap the hosted gateway
# applies, so self-hosted servers bound the body before parsing it.
INGEST_MAX_BODY_BYTES: Final = 26_214_400
