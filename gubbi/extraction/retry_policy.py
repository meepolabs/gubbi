"""Retry classification for failed extraction jobs.

Pure functions only: no I/O, no logging. Maps every ``error_code`` a failed
job row can carry to a retry class, and computes when that row may run again.
"""

import random
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class ErrorCode(StrEnum):
    """Every ``error_code`` written to a failed extraction job row."""

    LLM_RATE_LIMITED = "llm_rate_limited"
    LLM_PROVIDER_ERROR = "llm_provider_error"
    WORKER_LOST = "worker_lost"
    ENQUEUE_LOST = "enqueue_lost"
    LLM_SCHEMA = "llm_schema"
    LLM_TRUNCATED = "llm_truncated"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL_ERROR = "internal_error"
    LLM_REFUSED = "llm_refused"


class RetryClass(StrEnum):
    """How a failure is retried."""

    TRANSIENT = "transient"
    DETERMINISTIC = "deterministic"
    TERMINAL = "terminal"


RETRY_CLASSES: Final[Mapping[ErrorCode, RetryClass]] = MappingProxyType(
    {
        ErrorCode.LLM_RATE_LIMITED: RetryClass.TRANSIENT,
        ErrorCode.LLM_PROVIDER_ERROR: RetryClass.TRANSIENT,
        ErrorCode.WORKER_LOST: RetryClass.TRANSIENT,
        ErrorCode.ENQUEUE_LOST: RetryClass.TRANSIENT,
        ErrorCode.LLM_SCHEMA: RetryClass.DETERMINISTIC,
        ErrorCode.LLM_TRUNCATED: RetryClass.DETERMINISTIC,
        ErrorCode.DEADLINE_EXCEEDED: RetryClass.DETERMINISTIC,
        ErrorCode.INTERNAL_ERROR: RetryClass.DETERMINISTIC,
        ErrorCode.LLM_REFUSED: RetryClass.TERMINAL,
    }
)

TRANSIENT_BASE_DELAY: Final = timedelta(seconds=60)
TRANSIENT_MAX_DELAY: Final = timedelta(hours=6)
DETERMINISTIC_DELAY: Final = timedelta(hours=6)
DETERMINISTIC_MAX_ATTEMPTS: Final = 3
# Each delay is shortened by up to this fraction, so jitter never pushes a
# delay past its cap and simultaneous failures do not retry in lockstep.
JITTER_RATIO: Final = 0.2
# asyncpg encodes exactly this naive value as PostgreSQL 'infinity' for
# timestamptz and decodes 'infinity' back to it.
RUN_AFTER_NEVER: Final = datetime.max

_MAX_BACKOFF_EXPONENT: Final = 30


def retry_class(code: ErrorCode) -> RetryClass:
    """Return the retry class of ``code``."""
    return RETRY_CLASSES[code]


def next_run_after(
    code: ErrorCode,
    attempts: int,
    now: datetime,
    *,
    rng: Callable[[], float] = random.random,
) -> datetime:
    """Return when a job that failed with ``code`` may run again.

    ``attempts`` counts runs started so far, including the one that failed.
    ``rng`` returns a float in ``[0, 1)`` and drives the jitter. Returns
    :data:`RUN_AFTER_NEVER` when the job must not be retried.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cls = retry_class(code)
    if cls is RetryClass.TERMINAL:
        return RUN_AFTER_NEVER
    if cls is RetryClass.DETERMINISTIC:
        if attempts >= DETERMINISTIC_MAX_ATTEMPTS:
            return RUN_AFTER_NEVER
        return now + _jittered(DETERMINISTIC_DELAY, rng)
    return now + _jittered(_transient_delay(attempts), rng)


def _transient_delay(attempts: int) -> timedelta:
    exponent = min(attempts - 1, _MAX_BACKOFF_EXPONENT)
    return min(TRANSIENT_BASE_DELAY * (1 << exponent), TRANSIENT_MAX_DELAY)


def _jittered(delay: timedelta, rng: Callable[[], float]) -> timedelta:
    return delay * (1 - JITTER_RATIO * rng())
