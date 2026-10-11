"""Schema, data and constraint contract for the extraction queue columns revision.

Every case runs on a throwaway database on the working cluster (the migrations
grant to journal_app and journal_admin, which exist there), so a downgrade never
disturbs the shared ``journal_test`` other suites rely on.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from tests.conftest import TEST_DATABASE_URL
from tests.fixtures.cluster_roles import maintenance_dsn

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATABASE_PREFIX = "journal_queue_cols_"
_PREVIOUS_REVISION = "0004_extraction_tracking_columns"


@dataclass(frozen=True)
class ColumnSpec:
    data_type: str
    is_nullable: str
    column_default: str | None


_TIMESTAMPTZ = "timestamp with time zone"
_PERIOD_START_DEFAULT = "(date_trunc('month'::text, (now() AT TIME ZONE 'UTC'::text)))::date"

EXPECTED_COLUMNS: dict[str, ColumnSpec] = {
    "lane": ColumnSpec("text", "NO", "'backfill'::text"),
    "run_after": ColumnSpec(_TIMESTAMPTZ, "NO", "now()"),
    "defer_reason": ColumnSpec("text", "YES", None),
    "attempts": ColumnSpec("integer", "NO", "0"),
    "activity_at": ColumnSpec(_TIMESTAMPTZ, "YES", None),
    "filtered_tokens": ColumnSpec("integer", "NO", "0"),
    "token_count_version": ColumnSpec("smallint", "YES", None),
    "admitted_at": ColumnSpec(_TIMESTAMPTZ, "YES", None),
    "budget_period_start": ColumnSpec("date", "YES", None),
}

EXPECTED_CONSTRAINTS: dict[str, str] = {
    "extraction_jobs_lane_check": (
        "CHECK ((lane = ANY (ARRAY['backfill'::text, 'live'::text, 'paste'::text])))"
    ),
    "extraction_jobs_defer_reason_check": (
        "CHECK ((defer_reason = ANY (ARRAY['budget'::text, 'global'::text, 'provider'::text])))"
    ),
}

EXPECTED_INDEXES: dict[str, str] = {
    "idx_extraction_jobs_runnable": (
        "CREATE INDEX idx_extraction_jobs_runnable ON public.extraction_jobs"
        " USING btree (status, run_after)"
        " WHERE (status = ANY (ARRAY['pending'::text, 'failed'::text]))"
    ),
    "idx_extraction_jobs_user_period_admitted": (
        "CREATE INDEX idx_extraction_jobs_user_period_admitted ON public.extraction_jobs"
        " USING btree (user_id, budget_period_start) WHERE (admitted_at IS NOT NULL)"
    ),
    "idx_extraction_jobs_period_admitted": (
        "CREATE INDEX idx_extraction_jobs_period_admitted ON public.extraction_jobs"
        " USING btree (budget_period_start) INCLUDE (filtered_tokens)"
        " WHERE (admitted_at IS NOT NULL)"
    ),
}


@dataclass(frozen=True)
class SchemaSnapshot:
    columns: dict[str, ColumnSpec]
    constraints: dict[str, str]
    indexes: dict[str, str]
    period_start_default: str | None


@dataclass(frozen=True)
class LegacyJob:
    """One extraction job planted at the previous revision."""

    status: str
    source: str


# The partial unique index allows one active job per (user, conversation,
# source), so each status gets its own source.
LEGACY_JOBS = (
    LegacyJob(status="pending", source="claude"),
    LegacyJob(status="failed", source="chatgpt"),
    LegacyJob(status="running", source="gemini"),
    LegacyJob(status="completed", source="perplexity"),
)


@dataclass(frozen=True)
class RoundTrip:
    dsn: str
    after_first_upgrade: SchemaSnapshot
    after_downgrade: SchemaSnapshot
    after_second_upgrade: SchemaSnapshot
    legacy_jobs: dict[str, asyncpg.Record]
    upgraded_at: datetime


def _with_database(dsn: str, name: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{name}"))


def _run_alembic(dsn: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- argv entries are literals
        [sys.executable, "-m", "alembic", *args],
        cwd=_PROJECT_ROOT,
        env={**os.environ, "JOURNAL_DB_MIGRATION_URL": dsn},
        capture_output=True,
        text=True,
        check=False,
    )


def _alembic(dsn: str, *args: str) -> None:
    result = _run_alembic(dsn, *args)
    if result.returncode != 0:
        pytest.fail(
            f"alembic {' '.join(args)} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


async def _alembic_off_the_loop(dsn: str, *args: str) -> None:
    await asyncio.to_thread(_alembic, dsn, *args)


async def _snapshot(dsn: str) -> SchemaSnapshot:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        column_rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'extraction_jobs'
              AND column_name = ANY($1::text[])
            """,
            list(EXPECTED_COLUMNS),
        )
        constraint_rows = await conn.fetch(
            """
            SELECT conname, pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conrelid = 'public.extraction_jobs'::regclass
              AND conname = ANY($1::text[])
            """,
            list(EXPECTED_CONSTRAINTS),
        )
        index_rows = await conn.fetch(
            """
            SELECT indexname, indexdef FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = 'extraction_jobs'
              AND indexname = ANY($1::text[])
            """,
            list(EXPECTED_INDEXES),
        )
        period_start_default = await conn.fetchval(
            """
            SELECT column_default FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'extraction_jobs'
              AND column_name = 'period_start'
            """
        )
    finally:
        await conn.close()
    return SchemaSnapshot(
        columns={
            r["column_name"]: ColumnSpec(r["data_type"], r["is_nullable"], r["column_default"])
            for r in column_rows
        },
        constraints={r["conname"]: r["definition"] for r in constraint_rows},
        indexes={r["indexname"]: r["indexdef"] for r in index_rows},
        period_start_default=period_start_default,
    )


async def _seed_user_and_conversation(conn: asyncpg.Connection) -> tuple[UUID, int]:
    user_id = await conn.fetchval(
        "INSERT INTO users (id, email) VALUES (gen_random_uuid(), $1) RETURNING id",
        f"queue-{uuid.uuid4().hex}@test.local",
    )
    topic_id = await conn.fetchval(
        "INSERT INTO topics (path, title, user_id) VALUES ('queue/topic', 'Queue', $1) "
        "RETURNING id",
        user_id,
    )
    nonce = b"\x00" * 12
    conversation_id = await conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, slug, title_encrypted, title_nonce,
             summary_encrypted, summary_nonce)
        VALUES ($1, $2, 'queue-conv', '\\x00', $3, '\\x00', $3)
        RETURNING id
        """,
        topic_id,
        user_id,
        nonce,
    )
    return user_id, int(conversation_id)


async def _plant_legacy_jobs(dsn: str) -> dict[str, UUID]:
    """Insert one extraction job per status under the previous schema."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        async with conn.transaction():
            user_id, conversation_id = await _seed_user_and_conversation(conn)
            job_ids: dict[str, UUID] = {}
            for job in LEGACY_JOBS:
                job_ids[job.status] = await conn.fetchval(
                    """
                    INSERT INTO extraction_jobs
                        (user_id, conversation_id, source, status, period_start)
                    VALUES ($1, $2, $3, $4, CURRENT_DATE)
                    RETURNING id
                    """,
                    user_id,
                    conversation_id,
                    job.source,
                    job.status,
                )
    finally:
        await conn.close()
    return job_ids


async def _read_legacy_jobs(dsn: str, job_ids: dict[str, UUID]) -> dict[str, asyncpg.Record]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        rows = await conn.fetch(
            f"SELECT id, {', '.join(EXPECTED_COLUMNS)}, "  # noqa: S608 -- column names are module literals
            "isfinite(run_after) AS run_after_finite, "
            "run_after = 'infinity'::timestamptz AS run_after_infinity "
            "FROM extraction_jobs WHERE id = ANY($1::uuid[])",
            list(job_ids.values()),
        )
    finally:
        await conn.close()
    by_id = {row["id"]: row for row in rows}
    return {status: by_id[job_id] for status, job_id in job_ids.items()}


@asynccontextmanager
async def _scratch_database() -> AsyncIterator[str]:
    """Yield the DSN of a fresh database with pgvector, dropped on exit."""
    name = f"{_DATABASE_PREFIX}{uuid.uuid4().hex[:12]}"
    dsn = _with_database(TEST_DATABASE_URL, name)
    try:
        admin = await asyncpg.connect(maintenance_dsn(TEST_DATABASE_URL), timeout=5)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"PostgreSQL not reachable -- infrastructure missing: {exc}")

    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        try:
            setup = await asyncpg.connect(dsn, timeout=5)
            try:
                await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await setup.close()
            yield dsn
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def round_trip() -> AsyncIterator[RoundTrip]:
    """Upgrade a fresh database to head, step back one revision, and upgrade again.

    Jobs planted at the previous revision, one per status, show what the
    re-applied revision does to pre-existing rows.
    """
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_first_upgrade = await _snapshot(dsn)
        await _alembic_off_the_loop(dsn, "downgrade", _PREVIOUS_REVISION)
        after_downgrade = await _snapshot(dsn)
        job_ids = await _plant_legacy_jobs(dsn)
        upgraded_at = datetime.now(UTC)
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        yield RoundTrip(
            dsn=dsn,
            after_first_upgrade=after_first_upgrade,
            after_downgrade=after_downgrade,
            after_second_upgrade=await _snapshot(dsn),
            legacy_jobs=await _read_legacy_jobs(dsn, job_ids),
            upgraded_at=upgraded_at,
        )


@pytest.mark.parametrize("stage", ["after_first_upgrade", "after_second_upgrade"])
def test_upgrade_creates_the_columns_constraints_and_indexes(
    round_trip: RoundTrip, stage: str
) -> None:
    snapshot: SchemaSnapshot = getattr(round_trip, stage)

    assert snapshot == SchemaSnapshot(
        columns=EXPECTED_COLUMNS,
        constraints=EXPECTED_CONSTRAINTS,
        indexes=EXPECTED_INDEXES,
        period_start_default=_PERIOD_START_DEFAULT,
    )


def test_downgrade_removes_every_added_object(round_trip: RoundTrip) -> None:
    assert round_trip.after_downgrade == SchemaSnapshot(
        columns={}, constraints={}, indexes={}, period_start_default=None
    )


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_upgrade_holds_unfinished_jobs_until_released(round_trip: RoundTrip, status: str) -> None:
    assert round_trip.legacy_jobs[status]["run_after_infinity"] is True


@pytest.mark.parametrize("status", ["running", "completed"])
def test_upgrade_leaves_other_jobs_runnable_from_the_upgrade(
    round_trip: RoundTrip, status: str
) -> None:
    job = round_trip.legacy_jobs[status]

    assert job["run_after_finite"] is True
    assert round_trip.upgraded_at <= job["run_after"] <= datetime.now(UTC)


@pytest.mark.parametrize("status", [job.status for job in LEGACY_JOBS])
def test_pre_existing_jobs_read_the_backfill_lane_and_no_charge(
    round_trip: RoundTrip, status: str
) -> None:
    job = dict(round_trip.legacy_jobs[status])

    assert {
        key: value
        for key, value in job.items()
        if key not in {"id", "run_after", "run_after_finite", "run_after_infinity"}
    } == {
        "lane": "backfill",
        "defer_reason": None,
        "attempts": 0,
        "activity_at": None,
        "filtered_tokens": 0,
        "token_count_version": None,
        "admitted_at": None,
        "budget_period_start": None,
    }


@asynccontextmanager
async def _rolled_back_tenant(dsn: str) -> AsyncIterator[tuple[asyncpg.Connection, UUID, int]]:
    """Yield a connection inside a transaction with a fresh user and conversation, rolled back."""
    conn = await asyncpg.connect(dsn, timeout=5)
    transaction = conn.transaction()
    await transaction.start()
    try:
        user_id, conversation_id = await _seed_user_and_conversation(conn)
        yield conn, user_id, conversation_id
    finally:
        await transaction.rollback()
        await conn.close()


@pytest.mark.parametrize("column", ["lane", "defer_reason"])
@pytest.mark.asyncio(loop_scope="session")
async def test_insert_with_an_unknown_value_is_rejected(round_trip: RoundTrip, column: str) -> None:
    async with _rolled_back_tenant(round_trip.dsn) as (conn, user_id, conversation_id):
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO extraction_jobs "  # noqa: S608 -- column is a parametrize literal
                f"(user_id, conversation_id, source, status, period_start, {column}) "
                "VALUES ($1, $2, 'claude', 'pending', CURRENT_DATE, 'x')",
                user_id,
                conversation_id,
            )


@pytest.mark.parametrize(
    ("lane", "defer_reason"),
    [
        pytest.param("backfill", None, id="backfill-undeferred"),
        pytest.param("live", "budget", id="live-budget"),
        pytest.param("paste", "global", id="paste-global"),
        pytest.param("live", "provider", id="live-provider"),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_insert_with_a_known_value_is_accepted(
    round_trip: RoundTrip, lane: str, defer_reason: str | None
) -> None:
    async with _rolled_back_tenant(round_trip.dsn) as (conn, user_id, conversation_id):
        inserted = await conn.fetchrow(
            "INSERT INTO extraction_jobs "
            "(user_id, conversation_id, source, status, period_start, lane, defer_reason) "
            "VALUES ($1, $2, 'claude', 'pending', CURRENT_DATE, $3, $4) "
            "RETURNING lane, defer_reason",
            user_id,
            conversation_id,
            lane,
            defer_reason,
        )

    assert inserted is not None
    assert (inserted["lane"], inserted["defer_reason"]) == (lane, defer_reason)


@pytest.mark.asyncio(loop_scope="session")
async def test_insert_without_period_start_gets_the_first_of_the_utc_month(
    round_trip: RoundTrip,
) -> None:
    async with _rolled_back_tenant(round_trip.dsn) as (conn, user_id, conversation_id):
        inserted = await conn.fetchrow(
            "INSERT INTO extraction_jobs (user_id, conversation_id, source, status) "
            "VALUES ($1, $2, 'claude', 'pending') RETURNING period_start, created_at",
            user_id,
            conversation_id,
        )

    assert inserted is not None
    created_utc: datetime = inserted["created_at"].astimezone(UTC)
    assert inserted["period_start"] == created_utc.date().replace(day=1)


@pytest.mark.parametrize(
    "drift",
    [
        pytest.param(
            "ALTER TABLE extraction_jobs ADD COLUMN lane integer", id="column-of-another-type"
        ),
        pytest.param(
            "CREATE INDEX idx_extraction_jobs_runnable ON extraction_jobs (created_at)",
            id="index-on-another-column",
        ),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_upgrade_fails_and_records_nothing_over_a_drifted_object(drift: str) -> None:
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", _PREVIOUS_REVISION)
        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            await conn.execute(drift)
        finally:
            await conn.close()

        result = await asyncio.to_thread(_run_alembic, dsn, "upgrade", "head")

        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            version = await conn.fetchval("SELECT version_num FROM alembic_version")
            added = await conn.fetchval(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'extraction_jobs' "
                "AND column_name = 'admitted_at'"
            )
        finally:
            await conn.close()

    assert result.returncode != 0, "upgrade must refuse a pre-existing object it owns"
    assert version == _PREVIOUS_REVISION
    assert added == 0, "a failed upgrade must roll back every column it added"
