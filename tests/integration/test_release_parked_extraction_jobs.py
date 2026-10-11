"""DB-backed tests for ``gubbi.scripts.release_parked_extraction_jobs``.

Skipped automatically when the RLS test database is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import pytest

from gubbi.extraction import orphan_cleanup
from gubbi.extraction.intake import TOKEN_COUNT_VERSION, count_filtered_tokens
from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS
from gubbi.extraction.scheduler import ExtractionScheduler, SchedulerConfig
from gubbi.models.conversation import Message
from gubbi.scripts import release_parked_extraction_jobs as release_script
from gubbi.scripts.release_parked_extraction_jobs import (
    GroupCount,
    count_parked,
    release_parked,
)
from gubbi.storage.repositories import extraction_queue as queue_repo
from tests.fixtures.tenants import seed_for

if TYPE_CHECKING:
    from gubbi.crypto.cipher import ContentCipher

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

_HYDRATABLE = _RELEASABLE | {"pending_scheduled", "running_held"}


async def _plant_jobs(
    admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str = "release-parked"
) -> dict[str, UUID]:
    seed = await seed_for(admin_pool, user_id, topic_path=topic_path, n_entries=0)
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
    cipher: ContentCipher,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    before = await _snapshot(admin_pool, job_ids)

    result = await count_parked(admin_pool, cipher)

    assert set(result.groups) == {
        GroupCount("failed", "deterministic", 1),
        GroupCount("failed", "deterministic", 1, terminal=True),
        GroupCount("failed", "terminal", 1, terminal=True),
        GroupCount("failed", "transient", 1),
        GroupCount("failed", "unclassified", 2),
        GroupCount("pending", "none", 1),
    }
    assert (result.would_hydrate, result.would_mark_unreadable) == (len(_HYDRATABLE), 0)
    assert await _snapshot(admin_pool, job_ids) == before


async def test_dry_run_query_runs_in_a_read_only_transaction(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
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
        await count_parked(admin_pool, cipher)

    assert await _snapshot(admin_pool, job_ids) == before


async def test_release_frees_held_rows_except_terminal_failures(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    before = await _snapshot(admin_pool, job_ids)

    result = await release_parked(admin_pool, cipher)

    after = await _snapshot(admin_pool, job_ids)
    changed = {label for label in job_ids if after[label] != before[label]}
    assert changed == _HYDRATABLE
    assert all(after[label][1] != "infinity" for label in _RELEASABLE)
    assert after["pending_scheduled"][1] == before["pending_scheduled"][1]
    assert after["failed_refused_held"][1] == "infinity"
    assert after["failed_deterministic_exhausted_held"][1] == "infinity"
    assert after["running_held"][1] == "infinity"
    assert sum(group.count for group in result.released) == len(_RELEASABLE)
    assert result.held_terminal == 2
    assert result.hydrated == len(_HYDRATABLE)


async def test_a_running_row_counted_by_release_is_retried_once_swept_as_worker_lost(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    job_id = (await _plant_jobs(admin_pool, tenant_a, "release-running"))["running_held"]
    # Every planted row shares one conversation; any sibling would block or supersede this one.
    await admin_pool.execute(
        "DELETE FROM extraction_jobs WHERE user_id = $1 AND id <> $2", tenant_a, job_id
    )
    expected = await _stored_tokens(
        admin_pool, (await _job(admin_pool, job_id))["conversation_id"], cipher
    )

    await release_parked(admin_pool, cipher)
    counted = await admin_pool.fetchrow(
        "SELECT status, filtered_tokens, token_count_version FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    await admin_pool.execute(
        "UPDATE extraction_jobs SET created_at = now() - interval '1 day' WHERE id = $1", job_id
    )
    await orphan_cleanup._sweep_running(admin_pool)
    await admin_pool.execute(
        "UPDATE extraction_jobs SET run_after = now() - interval '1 minute' WHERE id = $1", job_id
    )
    async with admin_pool.acquire() as conn, conn.transaction():
        runnable = await queue_repo.select_runnable(conn, limit=1000)

    assert expected > 0
    assert counted is not None
    assert dict(counted) == {
        "status": "running",
        "filtered_tokens": expected,
        "token_count_version": TOKEN_COUNT_VERSION,
    }
    swept = [job for job in runnable if job.job_id == job_id]
    assert [(job.status, job.filtered_tokens) for job in swept] == [("failed", expected)]


async def test_second_release_is_a_no_op(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    job_ids = await _plant_jobs(admin_pool, tenant_a)
    await release_parked(admin_pool, cipher)
    after_first = await _snapshot(admin_pool, job_ids)

    second = await release_parked(admin_pool, cipher)

    assert (second.released, second.hydrated) == ((), 0)
    assert second.held_terminal == 2
    assert await _snapshot(admin_pool, job_ids) == after_first


async def _plant_legacy_pending(admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str) -> UUID:
    """One pending row as the queue-columns migration leaves it: held, no token count."""
    job_ids = await _plant_jobs(admin_pool, user_id, topic_path)
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM extraction_jobs WHERE id <> ALL($1::uuid[])"
            " AND conversation_id = (SELECT conversation_id FROM extraction_jobs WHERE id = $2)",
            [job_ids["pending_held"]],
            job_ids["pending_held"],
        )
    return job_ids["pending_held"]


async def _job(admin_pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    row = await admin_pool.fetchrow(
        "SELECT conversation_id, filtered_tokens, token_count_version,"
        " run_after = 'infinity' AS held FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    return row


async def _stored_tokens(admin_pool: asyncpg.Pool, conversation_id: int, cipher: Any) -> int:
    rows = await admin_pool.fetch(
        "SELECT role, content_encrypted, content_nonce FROM messages"
        " WHERE conversation_id = $1 ORDER BY position",
        conversation_id,
    )
    return count_filtered_tokens(
        Message(
            role=row["role"], content=cipher.decrypt(row["content_encrypted"], row["content_nonce"])
        )
        for row in rows
    )


class _RecordingArq:
    async def enqueue_job(
        self, function: str, *args: Any, _job_id: str | None = None, **kwargs: Any
    ) -> object:
        return object()


async def test_released_legacy_work_is_charged_its_stored_token_count(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    job_id = await _plant_legacy_pending(admin_pool, tenant_a, "release-legacy")
    expected = await _stored_tokens(
        admin_pool, (await _job(admin_pool, job_id))["conversation_id"], cipher
    )

    result = await release_parked(admin_pool, cipher)
    hydrated = await _job(admin_pool, job_id)
    config = SchedulerConfig(paused=False, budget_enabled=True, dispatch_bound=5)
    await ExtractionScheduler(admin_pool, _RecordingArq(), config).tick()
    async with admin_pool.acquire() as conn:
        usage = await queue_repo.user_period_usage(
            conn, [tenant_a], datetime.now(UTC).date().replace(day=1)
        )

    assert expected > 0
    assert (result.hydrated, result.marked_unreadable) == (1, 0)
    assert (hydrated["filtered_tokens"], hydrated["token_count_version"]) == (
        expected,
        TOKEN_COUNT_VERSION,
    )
    assert (usage[tenant_a].period_conversations, usage[tenant_a].period_tokens) == (1, expected)


async def _ended(admin_pool: asyncpg.Pool, job_id: UUID) -> tuple[str, str | None, bool]:
    row = await admin_pool.fetchrow(
        "SELECT status, error_code, run_after = 'infinity' AS held FROM extraction_jobs"
        " WHERE id = $1",
        job_id,
    )
    assert row is not None
    return row["status"], row["error_code"], row["held"]


async def _make_unreadable(admin_pool: asyncpg.Pool, job_id: UUID) -> None:
    await admin_pool.execute(
        "UPDATE messages SET content_encrypted = 'not ciphertext'::bytea"
        " WHERE conversation_id = (SELECT conversation_id FROM extraction_jobs WHERE id = $1)",
        job_id,
    )


async def _plant_lone(admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str, label: str) -> UUID:
    """One planted row alone on its conversation, as the queue-columns migration leaves it."""
    job_ids = await _plant_jobs(admin_pool, user_id, topic_path)
    await admin_pool.execute(
        "DELETE FROM extraction_jobs WHERE id <> $1 AND conversation_id ="
        " (SELECT conversation_id FROM extraction_jobs WHERE id = $1)",
        job_ids[label],
    )
    return job_ids[label]


@pytest.mark.parametrize(
    "label", ["pending_held", "failed_transient_held", "pending_scheduled"], ids=str
)
async def test_release_ends_a_row_whose_conversation_does_not_decrypt(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
    label: str,
) -> None:
    readable = await _plant_legacy_pending(admin_pool, tenant_a, "release-readable")
    unreadable = await _plant_lone(
        admin_pool, tenant_a, f"release-unreadable-{label.replace('_', '-')}", label
    )
    await _make_unreadable(admin_pool, unreadable)

    dry_run = await count_parked(admin_pool, cipher)
    result = await release_parked(admin_pool, cipher)
    after_first = await _snapshot(admin_pool, {"unreadable": unreadable})
    second = await release_parked(admin_pool, cipher)

    assert (dry_run.would_hydrate, dry_run.would_mark_unreadable) == (1, 1)
    assert (result.hydrated, result.marked_unreadable) == (1, 1)
    assert sum(group.count for group in result.released) == 1
    assert (await _job(admin_pool, readable))["held"] is False
    assert await _ended(admin_pool, unreadable) == ("failed", "conversation_unreadable", True)
    async with admin_pool.acquire() as conn, conn.transaction():
        runnable = {job.job_id for job in await queue_repo.select_runnable(conn, limit=1000)}
    assert unreadable not in runnable
    assert (second.released, second.hydrated, second.marked_unreadable) == ((), 0, 0)
    assert await _snapshot(admin_pool, {"unreadable": unreadable}) == after_first


async def test_an_unreadable_running_row_is_left_to_the_sweep_then_ended_on_a_later_release(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    job_id = await _plant_lone(admin_pool, tenant_a, "release-running-unreadable", "running_held")
    await _make_unreadable(admin_pool, job_id)

    dry_run = await count_parked(admin_pool, cipher)
    first = await release_parked(admin_pool, cipher)
    left_running = await _ended(admin_pool, job_id)
    await admin_pool.execute(
        "UPDATE extraction_jobs SET created_at = now() - interval '1 day' WHERE id = $1", job_id
    )
    await orphan_cleanup._sweep_running(admin_pool)
    second = await release_parked(admin_pool, cipher)

    assert (dry_run.would_mark_unreadable, dry_run.unreadable_running) == (0, 1)
    assert (first.marked_unreadable, first.unreadable_running) == (0, 1)
    assert left_running == ("running", None, True)
    assert (second.marked_unreadable, second.unreadable_running) == (1, 0)
    assert await _ended(admin_pool, job_id) == ("failed", "conversation_unreadable", True)


async def _plant_uncounted_pending(
    admin_pool: asyncpg.Pool, job_user_id: UUID, conversation_id: int, run_after: str
) -> UUID:
    job_id: UUID = await admin_pool.fetchval(
        """
        INSERT INTO extraction_jobs (user_id, conversation_id, source, status, run_after)
        VALUES ($1, $2, 'uncounted', 'pending', $3::text::timestamptz)
        RETURNING id
        """,
        job_user_id,
        conversation_id,
        run_after,
    )
    return job_id


async def test_release_counts_a_scheduled_row_left_without_a_token_count(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
) -> None:
    seed = await seed_for(admin_pool, tenant_a, topic_path="release-deploy-window", n_entries=0)
    assert seed.conversation_id is not None
    job_id = await _plant_uncounted_pending(
        admin_pool, tenant_a, seed.conversation_id, "2000-01-01T00:00:00Z"
    )
    expected = await _stored_tokens(admin_pool, seed.conversation_id, cipher)

    dry_run = await count_parked(admin_pool, cipher)
    result = await release_parked(admin_pool, cipher)
    async with admin_pool.acquire() as conn, conn.transaction():
        runnable = await queue_repo.select_runnable(conn, limit=10)

    assert expected > 0
    assert (dry_run.would_hydrate, dry_run.would_mark_unreadable) == (1, 0)
    assert (result.hydrated, result.released) == (1, ())
    assert [(job.job_id, job.filtered_tokens) for job in runnable] == [(job_id, expected)]


async def test_a_row_whose_conversation_has_another_owner_is_never_decrypted(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
    cipher: ContentCipher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = await seed_for(admin_pool, tenant_a, topic_path="release-owner", n_entries=0)
    assert seed.conversation_id is not None
    job_id = await _plant_uncounted_pending(admin_pool, tenant_b, seed.conversation_id, "infinity")
    decrypted: list[int] = []
    real_tokens = release_script._conversation_tokens

    async def recording_tokens(conn: Any, cipher: Any, conversation_id: int) -> int | None:
        decrypted.append(conversation_id)
        return await real_tokens(conn, cipher, conversation_id)

    monkeypatch.setattr(release_script, "_conversation_tokens", recording_tokens)

    dry_run = await count_parked(admin_pool, cipher)
    result = await release_parked(admin_pool, cipher)

    assert decrypted == []
    assert (dry_run.would_hydrate, dry_run.would_mark_unreadable) == (0, 1)
    assert (result.hydrated, result.marked_unreadable, result.released) == (0, 1, ())
    assert await _ended(admin_pool, job_id) == ("failed", "conversation_unreadable", True)


@pytest.mark.parametrize(
    ("concurrent_change", "version_after"),
    [
        pytest.param("token_count_version = $2", TOKEN_COUNT_VERSION, id="counted"),
        pytest.param("status = 'completed', completed_at = now()", None, id="completed"),
    ],
)
async def test_hydrated_count_reports_only_rows_actually_written(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    cipher: ContentCipher,
    monkeypatch: pytest.MonkeyPatch,
    concurrent_change: str,
    version_after: int | None,
) -> None:
    seed = await seed_for(admin_pool, tenant_a, topic_path="release-race", n_entries=0)
    assert seed.conversation_id is not None
    job_id = await _plant_uncounted_pending(admin_pool, tenant_a, seed.conversation_id, "infinity")
    real_count = release_script._count_uncounted_tokens

    async def count_then_race(conn: Any, cipher: Any) -> Any:
        tokens = await real_count(conn, cipher)
        args = (TOKEN_COUNT_VERSION,) if "$2" in concurrent_change else ()
        await conn.execute(
            f"UPDATE extraction_jobs SET {concurrent_change} WHERE id = $1",  # noqa: S608 - parametrize literal
            job_id,
            *args,
        )
        return tokens

    monkeypatch.setattr(release_script, "_count_uncounted_tokens", count_then_race)

    result = await release_parked(admin_pool, cipher)

    assert result.hydrated == 0
    row = await _job(admin_pool, job_id)
    assert (row["filtered_tokens"], row["token_count_version"]) == (0, version_after)
