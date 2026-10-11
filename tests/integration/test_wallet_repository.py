"""Integration tests for the per-user LLM wallet repository against real PostgreSQL."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import date
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories import wallet as wallet_repo
from gubbi.storage.repositories.wallet import AttemptUsage

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_MONTHLY = 1_000


def _attempt(user_id: UUID, weighted: int, *, job_id: UUID | None = None) -> AttemptUsage:
    return AttemptUsage(
        user_id=user_id,
        job_kind="live",
        job_id=job_id or uuid4(),
        attempt=1,
        input_tokens=weighted,
        cached_input_tokens=0,
        output_tokens=0,
        weighted=weighted,
    )


async def _charge(pool: asyncpg.Pool, attempt: AttemptUsage) -> wallet_repo.Charge | None:
    async with pool.acquire() as conn, conn.transaction():
        return await wallet_repo.charge(conn, attempt, _MONTHLY)


async def _parts(pool: asyncpg.Pool, user_id: UUID) -> list[tuple[int, int]]:
    rows = await pool.fetch(
        "SELECT monthly_part, credit_part FROM llm_charges WHERE user_id = $1 ORDER BY created_at",
        user_id,
    )
    return [(row["monthly_part"], row["credit_part"]) for row in rows]


async def _available(pool: asyncpg.Pool, user_id: UUID) -> int:
    async with pool.acquire() as conn:
        return (await wallet_repo.available(conn, {user_id: _MONTHLY}))[user_id]


async def _move_charges_to_last_month(pool: asyncpg.Pool, user_id: UUID) -> None:
    await pool.execute(
        "UPDATE llm_charges SET month ="
        " (date_trunc('month', now() AT TIME ZONE 'UTC') - interval '1 month')::date"
        " WHERE user_id = $1",
        user_id,
    )


async def _grant(pool: asyncpg.Pool, user_id: UUID, amount: int) -> None:
    async with pool.acquire() as conn:
        assert await wallet_repo.grant_credit(conn, user_id, amount, "test", f"test:{uuid4()}")


# ---------------------------------------------------------------------------
# charge: the monthly/credit split
# ---------------------------------------------------------------------------


async def test_a_charge_spends_the_monthly_allowance_first_then_credit(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 5_000)

    await _charge(clean_rls_db, _attempt(tenant_a, 600))
    await _charge(clean_rls_db, _attempt(tenant_a, 600))
    await _charge(clean_rls_db, _attempt(tenant_a, 300))

    assert await _parts(clean_rls_db, tenant_a) == [(600, 0), (400, 200), (0, 300)]
    assert await _available(clean_rls_db, tenant_a) == 5_000 - 500


async def test_credit_goes_negative_when_a_charge_exceeds_everything_left(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 100)

    charge = await _charge(clean_rls_db, _attempt(tenant_a, 1_500))

    assert charge is not None
    assert (charge.split.monthly_part, charge.split.credit_part) == (1_000, 500)
    assert await _available(clean_rls_db, tenant_a) == -400


async def test_a_new_month_brings_a_fresh_allowance_and_keeps_credit_spent(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 1_000)
    await _charge(clean_rls_db, _attempt(tenant_a, 1_200))
    await _move_charges_to_last_month(clean_rls_db, tenant_a)

    charge = await _charge(clean_rls_db, _attempt(tenant_a, 700))

    assert charge is not None
    assert (charge.split.monthly_part, charge.split.credit_part) == (700, 0)
    assert await _available(clean_rls_db, tenant_a) == (1_000 - 700) + (1_000 - 200)


async def test_a_charge_is_written_to_the_current_utc_month(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _charge(clean_rls_db, _attempt(tenant_a, 10))

    month, current = await clean_rls_db.fetchrow(
        "SELECT month, date_trunc('month', now() AT TIME ZONE 'UTC')::date AS current"
        " FROM llm_charges WHERE user_id = $1",
        tenant_a,
    )

    assert isinstance(month, date)
    assert month == current


async def test_a_zero_charge_writes_nothing(clean_rls_db: asyncpg.Pool, tenant_a: UUID) -> None:
    assert await _charge(clean_rls_db, _attempt(tenant_a, 0)) is None
    assert await _parts(clean_rls_db, tenant_a) == []


async def test_the_same_job_attempt_is_charged_once(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    attempt = _attempt(tenant_a, 300)

    first = await _charge(clean_rls_db, attempt)
    second = await _charge(clean_rls_db, attempt)

    assert first is not None
    assert second is None
    assert await _parts(clean_rls_db, tenant_a) == [(300, 0)]


async def test_concurrent_charges_of_one_user_never_both_spend_the_same_allowance(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 10_000)
    attempts = [_attempt(tenant_a, 800) for _ in range(2)]
    # Each charge pauses after reading this month's use until the other has
    # read it too, or a second passes. Serialized charges never meet there;
    # unserialized ones both read 0 used and both take 800 from the allowance.
    original_fetchrow = asyncpg.connection.Connection.fetchrow
    both_read = asyncio.Barrier(2)

    async def _fetchrow_then_wait(self: asyncpg.Connection, query: str, *args: object) -> object:
        row = await original_fetchrow(self, query, *args)
        if "monthly_used" in query:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(both_read.wait(), timeout=1)
        return row

    with patch.object(asyncpg.connection.Connection, "fetchrow", _fetchrow_then_wait):
        await asyncio.gather(*(_charge(clean_rls_db, a) for a in attempts))

    parts = await _parts(clean_rls_db, tenant_a)
    assert sum(monthly for monthly, _ in parts) == _MONTHLY
    assert sorted(parts) == [(200, 600), (800, 0)]


# ---------------------------------------------------------------------------
# available / global_used
# ---------------------------------------------------------------------------


async def test_available_reads_every_requested_user_in_one_call(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID, tenant_b: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 250)
    await _charge(clean_rls_db, _attempt(tenant_a, 100))
    await _charge(clean_rls_db, _attempt(tenant_b, 1_300))
    untouched = uuid4()

    async with clean_rls_db.acquire() as conn:
        balances = await wallet_repo.available(
            conn, {tenant_a: _MONTHLY, tenant_b: _MONTHLY, untouched: 2_000}
        )

    assert balances == {tenant_a: 900 + 250, tenant_b: -300, untouched: 2_000}


async def test_available_under_rls_sees_only_the_users_own_rows(
    clean_rls_db: asyncpg.Pool, app_pool: asyncpg.Pool, tenant_a: UUID, tenant_b: UUID
) -> None:
    await _charge(clean_rls_db, _attempt(tenant_a, 100))
    await _charge(clean_rls_db, _attempt(tenant_b, 700))

    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        balances = await wallet_repo.available(conn, {tenant_a: _MONTHLY, tenant_b: _MONTHLY})

    assert balances == {tenant_a: 900, tenant_b: _MONTHLY}


async def test_global_used_sums_this_months_weighted_charges_of_every_user(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID, tenant_b: UUID
) -> None:
    await _charge(clean_rls_db, _attempt(tenant_a, 400))
    await _move_charges_to_last_month(clean_rls_db, tenant_a)
    await _charge(clean_rls_db, _attempt(tenant_a, 30))
    await _charge(clean_rls_db, _attempt(tenant_b, 1_500))

    async with clean_rls_db.acquire() as conn:
        total = await wallet_repo.global_used(conn)

    assert total == 1_530


# ---------------------------------------------------------------------------
# grant_credit
# ---------------------------------------------------------------------------


async def test_grant_credit_is_idempotent_per_key(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    key = f"onboarding:{tenant_a}"
    async with clean_rls_db.acquire() as conn:
        first = await wallet_repo.grant_credit(conn, tenant_a, 25_000, "onboarding", key)
        second = await wallet_repo.grant_credit(conn, tenant_a, 25_000, "onboarding", key)
        other = await wallet_repo.grant_credit(conn, tenant_a, 7, "top-up", "payment:abc")

    assert (first, second, other) == (True, False, True)
    assert await _available(clean_rls_db, tenant_a) == _MONTHLY + 25_000 + 7


async def test_the_app_role_grants_credit_to_its_own_user_under_rls(
    clean_rls_db: asyncpg.Pool, app_pool: asyncpg.Pool, tenant_a: UUID, tenant_b: UUID
) -> None:
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        assert await wallet_repo.grant_credit(conn, tenant_a, 50, "onboarding", "own")

    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
            await wallet_repo.grant_credit(conn, tenant_b, 50, "onboarding", "other")

    assert await _available(clean_rls_db, tenant_a) == _MONTHLY + 50
    assert await _available(clean_rls_db, tenant_b) == _MONTHLY


async def test_the_app_role_cannot_rewrite_or_delete_wallet_rows(
    clean_rls_db: asyncpg.Pool, app_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    await _grant(clean_rls_db, tenant_a, 10)
    await _charge(clean_rls_db, _attempt(tenant_a, 10))

    for statement in (
        "UPDATE credit_grants SET amount = 1",
        "DELETE FROM credit_grants",
        "UPDATE llm_charges SET weighted = 0",
        "DELETE FROM llm_charges",
    ):
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
                await conn.execute(statement)
