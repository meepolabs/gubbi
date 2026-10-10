"""Schema and foreign-key contract for the extraction tracking columns revision.

The round trip runs on a throwaway database on the working cluster (the
migrations grant to journal_app and journal_admin, which exist there), so a
downgrade never disturbs the shared ``journal_test`` other suites rely on. The
foreign-key behaviour is exercised on the shared RLS database through the
admin pool, which is how every other suite seeds rows.
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
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from tests.conftest import TEST_DATABASE_URL
from tests.fixtures.cluster_roles import maintenance_dsn
from tests.fixtures.tenants import TenantSeed, seed_for

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATABASE_PREFIX = "journal_extr_cols_"
_PREVIOUS_REVISION = "0003_audit_log_app_dedup_read"


@dataclass(frozen=True)
class ColumnSpec:
    data_type: str
    is_nullable: str
    column_default: str | None


_COUNTER = ColumnSpec("integer", "NO", "0")

EXPECTED_COLUMNS: dict[tuple[str, str], ColumnSpec] = {
    ("entries", "duplicate_of_entry_id"): ColumnSpec("integer", "YES", None),
    ("entries", "client"): ColumnSpec("text", "YES", None),
    ("entries", "source_conversation_id"): ColumnSpec("integer", "YES", None),
    ("extraction_jobs", "input_tokens"): _COUNTER,
    ("extraction_jobs", "output_tokens"): _COUNTER,
    ("extraction_jobs", "cached_input_tokens"): _COUNTER,
    ("extraction_jobs", "reasoning_tokens"): _COUNTER,
    ("extraction_jobs", "llm_calls"): _COUNTER,
    ("extraction_jobs", "chunks"): _COUNTER,
    ("conversations", "extracted_content_hash"): ColumnSpec("text", "YES", None),
    ("conversations", "extraction_version"): ColumnSpec("integer", "YES", None),
}

# (column on entries) -> (referenced table, referenced column, delete rule)
EXPECTED_FOREIGN_KEYS: dict[str, tuple[str, str, str]] = {
    "duplicate_of_entry_id": ("entries", "id", "SET NULL"),
    "source_conversation_id": ("conversations", "id", "SET NULL"),
}

EXPECTED_INDEXES: dict[str, str] = {
    "idx_entries_duplicate_of_entry_id": (
        "CREATE INDEX idx_entries_duplicate_of_entry_id"
        " ON public.entries USING btree (duplicate_of_entry_id)"
    ),
    "idx_entries_source_conversation_id": (
        "CREATE INDEX idx_entries_source_conversation_id"
        " ON public.entries USING btree (source_conversation_id)"
    ),
    "idx_entries_user_date_client": (
        "CREATE INDEX idx_entries_user_date_client"
        " ON public.entries USING btree (user_id, date, client) WHERE (deleted_at IS NULL)"
    ),
}


@dataclass(frozen=True)
class SchemaSnapshot:
    columns: dict[tuple[str, str], ColumnSpec]
    foreign_keys: dict[str, tuple[str, str, str]]
    indexes: dict[str, str]


@dataclass(frozen=True)
class LegacyRows:
    """Row ids planted at the previous revision, before this one is applied."""

    entry_id: int
    conversation_id: int
    job_id: UUID


@dataclass(frozen=True)
class RoundTrip:
    after_first_upgrade: SchemaSnapshot
    after_downgrade: SchemaSnapshot
    after_second_upgrade: SchemaSnapshot
    legacy_entry: asyncpg.Record
    legacy_conversation: asyncpg.Record
    legacy_job: asyncpg.Record


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
            SELECT table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND (table_name, column_name) IN (SELECT * FROM unnest($1::text[], $2::text[]))
            """,
            [table for table, _ in EXPECTED_COLUMNS],
            [column for _, column in EXPECTED_COLUMNS],
        )
        fk_rows = await conn.fetch(
            """
            SELECT kcu.column_name, ccu.table_name AS ref_table,
                   ccu.column_name AS ref_column, rc.delete_rule
            FROM information_schema.referential_constraints rc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_schema = rc.constraint_schema
             AND kcu.constraint_name = rc.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_schema = rc.unique_constraint_schema
             AND ccu.constraint_name = rc.unique_constraint_name
            WHERE kcu.table_schema = 'public' AND kcu.table_name = 'entries'
              AND kcu.column_name = ANY($1::text[])
            """,
            list(EXPECTED_FOREIGN_KEYS),
        )
        index_rows = await conn.fetch(
            """
            SELECT indexname, indexdef FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = 'entries'
              AND indexname = ANY($1::text[])
            """,
            list(EXPECTED_INDEXES),
        )
    finally:
        await conn.close()
    return SchemaSnapshot(
        columns={
            (r["table_name"], r["column_name"]): ColumnSpec(
                r["data_type"], r["is_nullable"], r["column_default"]
            )
            for r in column_rows
        },
        foreign_keys={
            r["column_name"]: (r["ref_table"], r["ref_column"], r["delete_rule"]) for r in fk_rows
        },
        indexes={r["indexname"]: r["indexdef"] for r in index_rows},
    )


async def _plant_legacy_rows(dsn: str) -> LegacyRows:
    """Insert one entry, conversation and extraction job under the previous schema."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        async with conn.transaction():
            user_id = await conn.fetchval(
                "INSERT INTO users (id, email) VALUES (gen_random_uuid(), 'legacy@test.local') "
                "RETURNING id"
            )
            topic_id = await conn.fetchval(
                "INSERT INTO topics (path, title, user_id) VALUES ('legacy/topic', 'Legacy', $1) "
                "RETURNING id",
                user_id,
            )
            nonce = b"\x00" * 12
            conversation_id = await conn.fetchval(
                """
                INSERT INTO conversations
                    (topic_id, user_id, slug, title_encrypted, title_nonce,
                     summary_encrypted, summary_nonce)
                VALUES ($1, $2, 'legacy-conv', '\\x00', $3, '\\x00', $3)
                RETURNING id
                """,
                topic_id,
                user_id,
                nonce,
            )
            entry_id = await conn.fetchval(
                """
                INSERT INTO entries (topic_id, user_id, content_encrypted, content_nonce)
                VALUES ($1, $2, '\\x00', $3)
                RETURNING id
                """,
                topic_id,
                user_id,
                nonce,
            )
            job_id = await conn.fetchval(
                """
                INSERT INTO extraction_jobs
                    (user_id, conversation_id, source, status, period_start)
                VALUES ($1, $2, 'claude', 'completed', CURRENT_DATE)
                RETURNING id
                """,
                user_id,
                conversation_id,
            )
    finally:
        await conn.close()
    return LegacyRows(entry_id=int(entry_id), conversation_id=int(conversation_id), job_id=job_id)


async def _read_legacy_rows(
    dsn: str, rows: LegacyRows
) -> tuple[asyncpg.Record, asyncpg.Record, asyncpg.Record]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        entry = await conn.fetchrow(
            "SELECT duplicate_of_entry_id, client, source_conversation_id FROM entries "
            "WHERE id = $1",
            rows.entry_id,
        )
        conversation = await conn.fetchrow(
            "SELECT extracted_content_hash, extraction_version FROM conversations WHERE id = $1",
            rows.conversation_id,
        )
        job = await conn.fetchrow(
            "SELECT input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, "
            "llm_calls, chunks FROM extraction_jobs WHERE id = $1",
            rows.job_id,
        )
    finally:
        await conn.close()
    assert entry is not None
    assert conversation is not None
    assert job is not None
    return entry, conversation, job


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

    Rows planted at the previous revision show the re-applied revision leaves
    pre-existing data readable with the new columns at their defaults.
    """
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_first_upgrade = await _snapshot(dsn)
        await _alembic_off_the_loop(dsn, "downgrade", _PREVIOUS_REVISION)
        after_downgrade = await _snapshot(dsn)
        legacy = await _plant_legacy_rows(dsn)
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_second_upgrade = await _snapshot(dsn)
        entry, conversation, job = await _read_legacy_rows(dsn, legacy)
        yield RoundTrip(
            after_first_upgrade=after_first_upgrade,
            after_downgrade=after_downgrade,
            after_second_upgrade=after_second_upgrade,
            legacy_entry=entry,
            legacy_conversation=conversation,
            legacy_job=job,
        )


@pytest.mark.parametrize("stage", ["after_first_upgrade", "after_second_upgrade"])
def test_upgrade_creates_the_columns_foreign_keys_and_indexes(
    round_trip: RoundTrip, stage: str
) -> None:
    snapshot: SchemaSnapshot = getattr(round_trip, stage)

    assert snapshot.columns == EXPECTED_COLUMNS
    assert snapshot.foreign_keys == EXPECTED_FOREIGN_KEYS
    assert snapshot.indexes == EXPECTED_INDEXES


def test_downgrade_removes_every_added_column_foreign_key_and_index(
    round_trip: RoundTrip,
) -> None:
    assert round_trip.after_downgrade == SchemaSnapshot(columns={}, foreign_keys={}, indexes={})


def test_rows_from_before_the_upgrade_read_the_new_column_defaults(
    round_trip: RoundTrip,
) -> None:
    assert dict(round_trip.legacy_entry) == {
        "duplicate_of_entry_id": None,
        "client": None,
        "source_conversation_id": None,
    }
    assert dict(round_trip.legacy_conversation) == {
        "extracted_content_hash": None,
        "extraction_version": None,
    }
    assert dict(round_trip.legacy_job) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "llm_calls": 0,
        "chunks": 0,
    }


@pytest.mark.parametrize(
    "drift",
    [
        pytest.param("ALTER TABLE entries ADD COLUMN client integer", id="column-of-another-type"),
        pytest.param(
            "CREATE INDEX idx_entries_source_conversation_id ON entries (created_at)",
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
                "WHERE table_schema = 'public' AND table_name = 'conversations' "
                "AND column_name = 'extraction_version'"
            )
        finally:
            await conn.close()

    assert result.returncode != 0, "upgrade must refuse a pre-existing object it owns"
    assert version == _PREVIOUS_REVISION
    assert added == 0, "a failed upgrade must roll back every column it added"


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_the_referenced_entry_nulls_duplicate_of_entry_id(
    admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    seed: TenantSeed = await seed_for(admin_pool, tenant_a, topic_path="fk/dup", n_entries=2)
    original_id, duplicate_id = seed.entry_ids

    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE entries SET duplicate_of_entry_id = $1 WHERE id = $2",
            original_id,
            duplicate_id,
        )
        linked = await conn.fetchval(
            "SELECT duplicate_of_entry_id FROM entries WHERE id = $1", duplicate_id
        )
        await conn.execute("DELETE FROM entries WHERE id = $1", original_id)
        after = await conn.fetchrow(
            "SELECT duplicate_of_entry_id FROM entries WHERE id = $1", duplicate_id
        )

    assert linked == original_id
    assert after is not None, "the duplicate entry must survive deletion of its original"
    assert after["duplicate_of_entry_id"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_the_source_conversation_nulls_source_conversation_id(
    admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    seed: TenantSeed = await seed_for(admin_pool, tenant_a, topic_path="fk/src", n_entries=1)
    assert seed.conversation_id is not None
    (entry_id,) = seed.entry_ids

    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE entries SET source_conversation_id = $1 WHERE id = $2",
            seed.conversation_id,
            entry_id,
        )
        linked = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", entry_id
        )
        await conn.execute("DELETE FROM conversations WHERE id = $1", seed.conversation_id)
        after = await conn.fetchrow(
            "SELECT source_conversation_id FROM entries WHERE id = $1", entry_id
        )

    assert linked == seed.conversation_id
    assert after is not None, "the extracted entry must survive deletion of its conversation"
    assert after["source_conversation_id"] is None
