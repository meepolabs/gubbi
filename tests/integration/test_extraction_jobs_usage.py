"""Integration tests: job usage counters written by ``extraction_jobs.mark_completed``
and ``extraction_jobs.mark_failed`` under RLS."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories import extraction_jobs as jobs_repo

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

# Distinct values so a column/parameter swap cannot pass.
_USAGE: dict[str, int] = {
    "input_tokens": 101,
    "output_tokens": 202,
    "cached_input_tokens": 303,
    "reasoning_tokens": 404,
    "llm_calls": 5,
    "chunks": 6,
}
_ZERO_USAGE: dict[str, int] = dict.fromkeys(_USAGE, 0)
_CENTS = 17
_USAGE_SQL = (
    "SELECT status, cents_spent, input_tokens, output_tokens, cached_input_tokens,"
    " reasoning_tokens, llm_calls, chunks FROM extraction_jobs WHERE id = $1"
)


async def _create_job(conn: asyncpg.Connection, seed: TenantSeed) -> UUID:
    assert seed.conversation_id is not None
    return await jobs_repo.create_pending(
        conn, seed.user_id, seed.conversation_id, "chatgpt", period_start=date(2026, 10, 1)
    )


def _usage_of(row: asyncpg.Record | None) -> dict[str, int]:
    assert row is not None
    return {column: row[column] for column in _USAGE}


async def _complete(
    conn: asyncpg.Connection, job_id: UUID, usage: dict[str, int], cents: int = 0
) -> bool:
    return await jobs_repo.mark_completed(
        conn, job_id, topics_created=1, entries_created=2, cents_spent=cents, **usage
    )


async def _fail(
    conn: asyncpg.Connection, job_id: UUID, usage: dict[str, int], cents: int | None = None
) -> bool:
    if cents is None:
        return await jobs_repo.mark_failed(conn, job_id, error_code="llm_refused", **usage)
    return await jobs_repo.mark_failed(
        conn, job_id, error_code="llm_refused", cents_spent=cents, **usage
    )


_TERMINAL_WRITERS: list[Any] = [
    pytest.param(_complete, "completed", id="mark_completed"),
    pytest.param(_fail, "failed", id="mark_failed"),
]


@pytest.mark.parametrize(("write_terminal", "expected_status"), _TERMINAL_WRITERS)
async def test_terminal_transition_persists_all_usage_counters(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    write_terminal: Any,
    expected_status: str,
) -> None:
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        job_id = await _create_job(conn, seeded_a)
        transitioned = await write_terminal(conn, job_id, _USAGE, _CENTS)
        row = await conn.fetchrow(_USAGE_SQL, job_id)

    assert transitioned is True
    assert row is not None
    assert row["status"] == expected_status
    assert row["cents_spent"] == _CENTS
    assert _usage_of(row) == _USAGE


@pytest.mark.parametrize(("write_terminal", "expected_status"), _TERMINAL_WRITERS)
async def test_terminal_transition_without_usage_records_zero(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    write_terminal: Any,
    expected_status: str,
) -> None:
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        job_id = await _create_job(conn, seeded_a)
        transitioned = await write_terminal(conn, job_id, {})
        row = await conn.fetchrow(_USAGE_SQL, job_id)

    assert transitioned is True
    assert row is not None
    assert row["status"] == expected_status
    assert row["cents_spent"] == 0
    assert _usage_of(row) == _ZERO_USAGE


@pytest.mark.parametrize(("write_terminal", "expected_status"), _TERMINAL_WRITERS)
@pytest.mark.parametrize("first_terminal", [_complete, _fail], ids=["completed", "failed"])
async def test_terminal_job_keeps_its_status_and_usage(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    first_terminal: Any,
    write_terminal: Any,
    expected_status: str,
) -> None:
    del expected_status
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        job_id = await _create_job(conn, seeded_a)
        await first_terminal(conn, job_id, _ZERO_USAGE, 0)
        before = await conn.fetchrow(_USAGE_SQL, job_id)
        transitioned = await write_terminal(conn, job_id, _USAGE, _CENTS)
        after = await conn.fetchrow(_USAGE_SQL, job_id)

    assert transitioned is False
    assert after == before
