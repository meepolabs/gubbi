"""DB-backed tests for ``gubbi.scripts.release_parked_extraction_jobs``.

Skipped automatically when the RLS test database is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import asyncpg
import pytest

from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS
from gubbi.scripts import release_parked_extraction_jobs as release_script
from gubbi.scripts.release_parked_extraction_jobs import (
    GroupCount,
    count_parked,
    release_parked,
)
from tests.fixtures.tenants import seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


@dataclass(frozen=True)
class PlantedJob:
    """One extraction_jobs row the tests plant, keyed by a readable label."""

    label: str
    status: str
    error_code: str | None
    parked: bool
    attempts: int = 0


_PLANTED: tuple[PlantedJob, ...] = (
    PlantedJob("pending_held", "pending", None, parked=True),
    PlantedJob("failed_transient_held", "failed", "llm_rate_limited", parked=True),
    PlantedJob("failed_deterministic_held", "failed", "llm_schema", parked=True),
    PlantedJob("failed_without_code_held", "failed", None, parked=True),
    PlantedJob("failed_unknown_code_held", "failed", "legacy_code", parked=True),
    PlantedJob("failed_refused_held", "failed", "llm_refused", parked=True),
    PlantedJob(
        "failed_deterministic_exhausted_held",
        "failed",
        "llm_schema",
        parked=True,
        attempts=DETERMINISTIC_MAX_ATTEMPTS,
    ),
    PlantedJob("completed_held", "completed", None, parked=True),
    PlantedJob("running_held", "running", None, parked=True),
    PlantedJob("pending_scheduled", "pending", None, parked=False),
)

_RELEASABLE = {
    "pending_held",
    "failed_transient_held",
    "failed_deterministic_held",
    "failed_without_code_held",
    "failed_unknown_code_held",
}


async def _plant_jobs(admin_pool: asyncpg.Pool, user_id: UUID) -> dict[str, UUID]:
    seed = await seed_for(admin_pool, user_id, topic_path="release-parked", n_entries=0)
    assert seed.conversation_id is not None
    job_ids: dict[str, UUID] = {}
    async with admin_pool.acquire() as conn, conn.transaction():
        for job in _PLANTED:
            job_ids[job.label] = await conn.fetchval(
                """
                INSERT INTO extraction_jobs
                    (user_id, conversation_id, source, status, error_code, run_after, attempts)
                VALUES ($1, $2, $3, $4, $5,
                        CASE WHEN $6 THEN 'infinity'::timestamptz
                             ELSE now() + interval '1 day' END,
                        $7)
                RETURNING id
                """,
                user_id,
                seed.conversation_id,
                job.label,
                job.status,
                job.error_code,
                job.parked,
                job.attempts,
            )
    return job_ids


async def _snapshot(
    admin_pool: asyncpg.Pool, job_ids: dict[str, UUID]
) -> dict[str, tuple[str, str]]:
    """Map label -> (row version, run_after text); a changed version means a write."""
    rows = await admin_pool.fetch(
        "SELECT id, xmin::text AS version, run_after::text AS run_after "
        "FROM extraction_jobs WHERE id = ANY($1::uuid[])",
        list(job_ids.values()),
    )
    by_id = {row["id"]: (row["version"], row["run_after"]) for row in rows}
    return {label: by_id[job_id] for label, job_id in job_ids.items()}


async def test_dry_run_counts_held_rows_by_status_and_error_class_without_writing(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    before = await _snapshot(admin_pool, job_ids)

    groups = await count_parked(admin_pool)

    assert set(groups) == {
        GroupCount("failed", "deterministic", 1),
        GroupCount("failed", "deterministic", 1, terminal=True),
        GroupCount("failed", "none", 1),
        GroupCount("failed", "terminal", 1, terminal=True),
        GroupCount("failed", "transient", 1),
        GroupCount("failed", "unknown", 1),
        GroupCount("pending", "none", 1),
    }
    assert await _snapshot(admin_pool, job_ids) == before


async def test_dry_run_query_runs_in_a_read_only_transaction(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    before = await _snapshot(admin_pool, job_ids)
    monkeypatch.setattr(
        release_script,
        "_COUNT_PARKED_SQL",
        "UPDATE extraction_jobs SET run_after = now() "
        "WHERE run_after = 'infinity' AND $1::text[] IS NOT NULL "
        "AND $2::text[] IS NOT NULL AND $3::int IS NOT NULL "
        "RETURNING status, error_code, false AS terminal, 1 AS n",
    )

    with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
        await count_parked(admin_pool)

    assert await _snapshot(admin_pool, job_ids) == before


async def test_release_frees_held_rows_except_terminal_failures(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    before = await _snapshot(admin_pool, job_ids)

    result = await release_parked(admin_pool)

    after = await _snapshot(admin_pool, job_ids)
    changed = {label for label in job_ids if after[label] != before[label]}
    assert changed == _RELEASABLE
    assert all(after[label][1] != "infinity" for label in _RELEASABLE)
    assert after["failed_refused_held"][1] == "infinity"
    assert after["failed_deterministic_exhausted_held"][1] == "infinity"
    assert sum(group.count for group in result.released) == len(_RELEASABLE)
    assert result.held_terminal == 2


async def test_second_release_is_a_no_op(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    await release_parked(admin_pool)
    after_first = await _snapshot(admin_pool, job_ids)

    second = await release_parked(admin_pool)

    assert second.released == ()
    assert second.held_terminal == 2
    assert await _snapshot(admin_pool, job_ids) == after_first
