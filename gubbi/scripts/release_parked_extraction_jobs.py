"""Release extraction jobs held with ``run_after = 'infinity'``.

A held ("parked") job is a pending or failed row the scheduler never picks
up. The queue-columns migration holds every unfinished row present at
upgrade, and a failure with no retry left is held the same way.

``--dry-run`` reports held-row counts by status, error class and whether
the retry policy treats the row as terminal, inside a read-only transaction.
``--release`` makes every held row runnable now, except terminal failed rows
(a terminal error code, or a deterministic one past its attempt limit);
completed and running rows are never touched. A failed row with no or an
unrecognised error code is counted as ``unclassified`` and released: it retries
as a transient failure. Releasing twice is a no-op.

A row held by the queue-columns migration carries no token count, and the
scheduler never admits such a row. ``--release`` first counts its tokens from
the stored conversation with the estimator ingest uses; a row whose
conversation cannot be loaded or decrypted stays held and is counted as
``held_unhydrated``. ``--dry-run`` reports how many rows would be counted and
how many would stay held. Both modes need the content encryption keys
(``JOURNAL_ENCRYPTION_MASTER_KEY_V*``).

Output is one JSON log line per count group on stderr, counts only.

USAGE::

    JOURNAL_DB_ADMIN_URL="postgresql://admin@host:5432/journal" \\
    JOURNAL_ENCRYPTION_MASTER_KEY_V1="..." \\
        python -m gubbi.scripts.release_parked_extraction_jobs --dry-run

    Replace ``--dry-run`` with ``--release`` to release the held rows.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TextIO

import structlog
from pydantic import ValidationError

from gubbi.crypto.cipher import ContentCipher, DecryptionError, load_master_keys_from_env
from gubbi.extraction.intake import TOKEN_COUNT_VERSION, count_filtered_tokens
from gubbi.extraction.retry_policy import ErrorCode, retry_class
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.pg_setup import init_pool
from gubbi.storage.repositories.conversations import read_conversation_by_id
from gubbi.storage.repositories.extraction_queue import (
    terminal_failure_args,
    terminal_failure_sql,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from uuid import UUID

    import asyncpg

NO_ERROR: Final = "none"
UNCLASSIFIED_ERROR: Final = "unclassified"

_TERMINAL_FAILURE: Final = f"(status = 'failed' AND {terminal_failure_sql(1)})"

_COUNT_PARKED_SQL: Final = f"""
SELECT status, error_code, {_TERMINAL_FAILURE} AS terminal, count(*) AS n
FROM extraction_jobs
WHERE run_after = 'infinity' AND status IN ('pending', 'failed')
GROUP BY 1, 2, 3
"""  # noqa: S608 - fixed SQL fragments; values are bound

_RELEASABLE: Final = (
    f"run_after = 'infinity' AND status IN ('pending', 'failed') AND NOT {_TERMINAL_FAILURE}"
)

_UNCOUNTED_SQL: Final = f"""
SELECT id, conversation_id
FROM extraction_jobs
WHERE {_RELEASABLE}
  AND token_count_version IS NULL
ORDER BY conversation_id, id
"""  # noqa: S608 - fixed SQL fragments; values are bound

_HYDRATE_SQL: Final = """
UPDATE extraction_jobs
SET filtered_tokens = $2,
    token_count_version = $3
WHERE id = $1
  AND run_after = 'infinity'
  AND token_count_version IS NULL
"""

_RELEASE_SQL: Final = f"""
WITH released AS (
    UPDATE extraction_jobs
    SET run_after = now()
    WHERE {_RELEASABLE}
      AND token_count_version IS NOT NULL
    RETURNING status, error_code
)
SELECT status, error_code, false AS terminal, count(*) AS n
FROM released
GROUP BY status, error_code
"""  # noqa: S608 - fixed SQL fragments; values are bound

_COUNT_HELD_TERMINAL_SQL: Final = f"""
SELECT count(*)
FROM extraction_jobs
WHERE run_after = 'infinity'
  AND {_TERMINAL_FAILURE}
"""  # noqa: S608 - fixed SQL fragments; values are bound


class EventLogger(Protocol):
    """The logging calls this script makes."""

    def info(self, event: str, **fields: object) -> object:
        """Log ``event`` at info level."""
        ...

    def error(self, event: str, **fields: object) -> object:
        """Log ``event`` at error level."""
        ...


@dataclass(frozen=True)
class GroupCount:
    """Number of rows sharing one status, error class and terminal verdict.

    ``terminal`` rows are failed rows the retry policy never runs again;
    release leaves them held.
    """

    status: str
    error_class: str
    count: int
    terminal: bool = False


@dataclass(frozen=True)
class DryRunResult:
    """Held rows by group, and how many uncounted rows would be counted or stay held."""

    groups: tuple[GroupCount, ...]
    would_hydrate: int = 0
    held_unhydrated: int = 0


@dataclass(frozen=True)
class ReleaseResult:
    """Rows released, by group; rows whose token count was filled in; rows left held."""

    released: tuple[GroupCount, ...]
    held_terminal: int
    hydrated: int = 0
    held_unhydrated: int = 0


@dataclass(frozen=True)
class _TokenCounts:
    counted: dict[UUID, int]
    unloadable: frozenset[UUID]


def error_class_of(status: str, error_code: str | None) -> str:
    """Return the retry class name of ``error_code``.

    A row that is not ``failed`` and has no code maps to ``none``; any other
    missing or unrecognised code maps to ``unclassified``.
    """
    if error_code is None:
        return NO_ERROR if status != "failed" else UNCLASSIFIED_ERROR
    try:
        return retry_class(ErrorCode(error_code)).value
    except ValueError:
        return UNCLASSIFIED_ERROR


def group_counts(rows: Iterable[Mapping[str, object]]) -> tuple[GroupCount, ...]:
    """Fold ``(status, error_code, terminal, n)`` rows into counts per group."""
    totals: Counter[tuple[str, str, bool]] = Counter()
    for row in rows:
        error_code, terminal, n = row["error_code"], row["terminal"], row["n"]
        if not (
            (error_code is None or isinstance(error_code, str))
            and isinstance(terminal, bool)
            and isinstance(n, int)
        ):
            raise TypeError(f"unexpected count row types: {sorted(row)}")
        status = str(row["status"])
        totals[(status, error_class_of(status, error_code), terminal)] += n
    return tuple(
        GroupCount(status=status, error_class=error_class, count=count, terminal=terminal)
        for (status, error_class, terminal), count in sorted(totals.items())
    )


async def _count_uncounted_tokens(conn: asyncpg.Connection, cipher: ContentCipher) -> _TokenCounts:
    """Count the filtered tokens of every releasable held row that has no token count.

    Tokens come from the stored conversation, counted as ingest counts them.
    A row whose conversation is missing or does not decrypt is unloadable.
    """
    counted: dict[UUID, int] = {}
    unloadable: set[UUID] = set()
    by_conversation: dict[int, int | None] = {}
    for row in await conn.fetch(_UNCOUNTED_SQL, *terminal_failure_args()):
        conversation_id = int(row["conversation_id"])
        if conversation_id not in by_conversation:
            by_conversation[conversation_id] = await _conversation_tokens(
                conn, cipher, conversation_id
            )
        tokens = by_conversation[conversation_id]
        if tokens is None:
            unloadable.add(row["id"])
        else:
            counted[row["id"]] = tokens
    return _TokenCounts(counted=counted, unloadable=frozenset(unloadable))


async def _conversation_tokens(
    conn: asyncpg.Connection, cipher: ContentCipher, conversation_id: int
) -> int | None:
    try:
        _meta, messages, _total = await read_conversation_by_id(conn, cipher, conversation_id)
    except (ConversationNotFoundError, DecryptionError):
        return None
    return count_filtered_tokens(messages)


async def count_parked(pool: asyncpg.Pool, cipher: ContentCipher) -> DryRunResult:
    """Count held rows by status and error class without writing anything."""
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        rows = await conn.fetch(_COUNT_PARKED_SQL, *terminal_failure_args())
        tokens = await _count_uncounted_tokens(conn, cipher)
    return DryRunResult(
        groups=group_counts(rows),
        would_hydrate=len(tokens.counted),
        held_unhydrated=len(tokens.unloadable),
    )


async def release_parked(pool: asyncpg.Pool, cipher: ContentCipher) -> ReleaseResult:
    """Count tokens where missing, then make every non-terminal, counted held row runnable now."""
    args = terminal_failure_args()
    async with pool.acquire() as conn, conn.transaction():
        tokens = await _count_uncounted_tokens(conn, cipher)
        if tokens.counted:
            await conn.executemany(
                _HYDRATE_SQL,
                [(job_id, n, TOKEN_COUNT_VERSION) for job_id, n in tokens.counted.items()],
            )
        rows = await conn.fetch(_RELEASE_SQL, *args)
        held_terminal = await conn.fetchval(_COUNT_HELD_TERMINAL_SQL, *args)
    return ReleaseResult(
        released=group_counts(rows),
        held_terminal=int(held_terminal or 0),
        hydrated=len(tokens.counted),
        held_unhydrated=len(tokens.unloadable),
    )


def build_logger(stream: TextIO) -> EventLogger:
    """Return a JSON-lines structlog logger on ``stream``, independent of global config."""
    logger: EventLogger = structlog.wrap_logger(
        structlog.PrintLogger(stream),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
    )
    return logger


def log_dry_run(log: EventLogger, result: DryRunResult) -> None:
    """Emit one line per held group, then a summary."""
    groups = result.groups
    for group in groups:
        log.info(
            "parked_jobs",
            status=group.status,
            error_class=group.error_class,
            terminal=group.terminal,
            count=group.count,
        )
    terminal = sum(group.count for group in groups if group.terminal)
    parked = sum(group.count for group in groups)
    log.info(
        "dry_run_summary",
        parked=parked,
        releasable=parked - terminal - result.held_unhydrated,
        terminal=terminal,
        would_hydrate=result.would_hydrate,
        held_unhydrated=result.held_unhydrated,
    )


def log_release(log: EventLogger, result: ReleaseResult) -> None:
    """Emit one line per released group, then a summary."""
    for group in result.released:
        log.info(
            "released_jobs", status=group.status, error_class=group.error_class, count=group.count
        )
    log.info(
        "release_summary",
        released=sum(group.count for group in result.released),
        held_terminal=result.held_terminal,
        hydrated=result.hydrated,
        held_unhydrated=result.held_unhydrated,
    )


def _resolve_admin_dsn() -> str:
    """Return the BYPASSRLS admin DSN from the environment or settings."""
    dsn = os.environ.get("JOURNAL_DB_ADMIN_URL")
    if dsn:
        return dsn
    try:
        from gubbi.config import get_settings

        admin_url = get_settings().db.admin_url
    except ValidationError:
        admin_url = ""
    if admin_url:
        return str(admin_url)
    sys.exit(
        "FATAL: JOURNAL_DB_ADMIN_URL is required; it provides the BYPASSRLS role "
        "needed to read every tenant's jobs."
    )


def _build_cipher() -> ContentCipher:
    """Return the content cipher from the encryption key environment variables."""
    keys = load_master_keys_from_env()
    if not keys:
        sys.exit(
            "FATAL: JOURNAL_ENCRYPTION_MASTER_KEY_V* is required; it decrypts stored "
            "conversations to count the tokens of held jobs that have no count."
        )
    return ContentCipher(keys)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count or release extraction jobs held with run_after = 'infinity'.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Report held-row counts; write nothing."
    )
    mode.add_argument(
        "--release",
        action="store_true",
        help="Make held rows runnable now, except terminal failures and rows "
        "whose conversation cannot be read.",
    )
    return parser


async def _run(dsn: str, cipher: ContentCipher, *, release: bool, log: EventLogger) -> None:
    pool = await init_pool(dsn, min_size=1, max_size=1)
    try:
        if release:
            log_release(log, await release_parked(pool, cipher))
        else:
            log_dry_run(log, await count_parked(pool, cipher))
    finally:
        await pool.close()


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry-point."""
    args = _build_parser().parse_args(argv)
    dsn = _resolve_admin_dsn()
    cipher = _build_cipher()
    log = build_logger(sys.stderr)
    try:
        asyncio.run(_run(dsn, cipher, release=args.release, log=log))
    except Exception as exc:
        log.error("release_parked_jobs_failed", release=args.release, error_type=type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
