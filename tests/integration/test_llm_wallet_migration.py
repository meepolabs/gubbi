"""Schema, grant, seed and round-trip contract for the LLM wallet revision.

Every case runs on a throwaway database on the working cluster, so a downgrade
never disturbs the shared ``journal_test`` other suites rely on.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from tests.integration.test_extraction_queue_columns_migration import (
    _alembic_off_the_loop,
    _scratch_database,
)

pytestmark = [pytest.mark.integration]

_PREVIOUS_REVISION = "0005_extraction_queue_columns"
_TABLES = ("credit_grants", "llm_charges")
_ONBOARDING_AMOUNT = 25_000_000

EXPECTED_INDEXES = {
    "credit_grants_pkey",
    "credit_grants_idempotency_key_key",
    "idx_credit_grants_user",
    "llm_charges_pkey",
    "llm_charges_job_attempt_key",
    "idx_llm_charges_user_month",
    "idx_llm_charges_month",
}

EXPECTED_POLICIES = {
    ("credit_grants", "credit_grants_user_isolation", "{journal_app}"),
    ("credit_grants", "credit_grants_app_insert_onboarding_only", "{journal_app}"),
    ("llm_charges", "llm_charges_user_isolation", "{journal_app}"),
}

# (role, table, privilege) -> held
EXPECTED_GRANTS = {
    (role, table, privilege): role == "journal_admin" or privilege in ("SELECT", "INSERT")
    for role in ("journal_app", "journal_admin")
    for table in _TABLES
    for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE")
}


@dataclass(frozen=True)
class WalletSchema:
    tables: dict[str, tuple[bool, bool]]
    indexes: set[str]
    policies: set[tuple[str, str, str]]


@dataclass(frozen=True)
class RoundTrip:
    dsn: str
    after_first_upgrade: WalletSchema
    after_downgrade: WalletSchema
    after_second_upgrade: WalletSchema
    grants: dict[tuple[str, str, str], bool]
    seeded: dict[UUID, list[asyncpg.Record]]
    users: dict[str, UUID]


async def _wallet_schema(dsn: str) -> WalletSchema:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        tables = await conn.fetch(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class"
            " WHERE relnamespace = 'public'::regnamespace AND relkind = 'r'"
            " AND relname = ANY($1::text[])",
            list(_TABLES),
        )
        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            " AND tablename = ANY($1::text[])",
            list(_TABLES),
        )
        policies = await conn.fetch(
            "SELECT tablename, policyname, roles::text AS roles FROM pg_policies"
            " WHERE schemaname = 'public' AND tablename = ANY($1::text[])",
            list(_TABLES),
        )
    finally:
        await conn.close()
    return WalletSchema(
        tables={r["relname"]: (r["relrowsecurity"], r["relforcerowsecurity"]) for r in tables},
        indexes={r["indexname"] for r in indexes},
        policies={(r["tablename"], r["policyname"], r["roles"]) for r in policies},
    )


async def _grants(dsn: str) -> dict[tuple[str, str, str], bool]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        return {
            key: await conn.fetchval("SELECT has_table_privilege($1, $2, $3)", *key)
            for key in EXPECTED_GRANTS
        }
    finally:
        await conn.close()


async def _plant_users(dsn: str) -> dict[str, UUID]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        users: dict[str, UUID] = {}
        for label, deleted in (("active", False), ("second", False), ("deleted", True)):
            users[label] = await conn.fetchval(
                "INSERT INTO users (id, email, deleted_at) VALUES (gen_random_uuid(), $1,"
                " CASE WHEN $2 THEN now() END) RETURNING id",
                f"wallet-{label}-{uuid.uuid4().hex[:8]}@test.local",
                deleted,
            )
    finally:
        await conn.close()
    return users


async def _grants_by_user(dsn: str, users: dict[str, UUID]) -> dict[UUID, list[asyncpg.Record]]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        rows = await conn.fetch(
            "SELECT user_id, amount, reason, idempotency_key FROM credit_grants"
            " WHERE user_id = ANY($1::uuid[])",
            list(users.values()),
        )
    finally:
        await conn.close()
    by_user: dict[UUID, list[asyncpg.Record]] = {user_id: [] for user_id in users.values()}
    for row in rows:
        by_user[row["user_id"]].append(row)
    return by_user


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def round_trip() -> AsyncIterator[RoundTrip]:
    """Upgrade to head, step back one revision, plant users, and upgrade again."""
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_first_upgrade = await _wallet_schema(dsn)
        await _alembic_off_the_loop(dsn, "downgrade", _PREVIOUS_REVISION)
        after_downgrade = await _wallet_schema(dsn)
        users = await _plant_users(dsn)
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        yield RoundTrip(
            dsn=dsn,
            after_first_upgrade=after_first_upgrade,
            after_downgrade=after_downgrade,
            after_second_upgrade=await _wallet_schema(dsn),
            grants=await _grants(dsn),
            seeded=await _grants_by_user(dsn, users),
            users=users,
        )


@pytest.mark.parametrize("stage", ["after_first_upgrade", "after_second_upgrade"])
def test_upgrade_creates_both_tables_with_forced_rls_indexes_and_policies(
    round_trip: RoundTrip, stage: str
) -> None:
    assert getattr(round_trip, stage) == WalletSchema(
        tables=dict.fromkeys(_TABLES, (True, True)),
        indexes=EXPECTED_INDEXES,
        policies=EXPECTED_POLICIES,
    )


def test_downgrade_removes_every_added_object(round_trip: RoundTrip) -> None:
    assert round_trip.after_downgrade == WalletSchema(tables={}, indexes=set(), policies=set())


def test_the_app_role_may_only_select_and_insert(round_trip: RoundTrip) -> None:
    assert round_trip.grants == EXPECTED_GRANTS


@pytest.mark.parametrize("label", ["active", "second"])
def test_upgrade_grants_each_existing_user_onboarding_credit_once(
    round_trip: RoundTrip, label: str
) -> None:
    user_id = round_trip.users[label]

    assert [dict(row) for row in round_trip.seeded[user_id]] == [
        {
            "user_id": user_id,
            "amount": _ONBOARDING_AMOUNT,
            "reason": "onboarding",
            "idempotency_key": f"onboarding:{user_id}",
        }
    ]


def test_upgrade_grants_nothing_to_a_deleted_user(round_trip: RoundTrip) -> None:
    assert round_trip.seeded[round_trip.users["deleted"]] == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_charge_whose_parts_do_not_sum_to_weighted_is_rejected(
    round_trip: RoundTrip,
) -> None:
    conn = await asyncpg.connect(round_trip.dsn, timeout=5)
    try:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO llm_charges (user_id, job_kind, job_id, attempt, input_tokens,"
                " cached_input_tokens, output_tokens, weighted, monthly_part, credit_part, month)"
                " VALUES ($1, 'live', gen_random_uuid(), 1, 0, 0, 0, 10, 4, 5, CURRENT_DATE)",
                round_trip.users["active"],
            )
    finally:
        await conn.close()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_non_positive_grant_is_rejected(round_trip: RoundTrip) -> None:
    conn = await asyncpg.connect(round_trip.dsn, timeout=5)
    try:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO credit_grants (user_id, amount, reason, idempotency_key)"
                " VALUES ($1, 0, 'test', 'zero-grant')",
                round_trip.users["active"],
            )
    finally:
        await conn.close()
