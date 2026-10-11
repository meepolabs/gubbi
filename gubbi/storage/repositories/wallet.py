"""Repository module for the per-user LLM wallet: credit grants and charges.

Both tables are append-only; a balance is always a sum over them. Every amount
is in weighted tokens. A month is the first day of a UTC calendar month, read
from the database clock, so every reader and the charge writer agree on it.

A user's available balance is what remains of this month's allowance plus
credit: ``max(monthly_amount - monthly use this month, 0) + granted credit -
credit spent``. Credit may be negative, so the balance may be too.

Reads work on an RLS-scoped connection for the user's own id, and on the
admin pool across users.

Public surface
--------------
available       -- available balance of each requested user
global_used     -- weighted tokens charged this month across every user
month_usage     -- this month's weighted and raw token totals across every user
grant_credit    -- add credit once per idempotency key; the only grant write path
charge          -- write one attempt's charge, monthly allowance first, then credit
AttemptUsage    -- dataclass charge takes
Charge          -- dataclass returned by charge
MonthUsage      -- dataclass returned by month_usage
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

from gubbi.extraction.wallet import ChargeSplit, split_charge

if TYPE_CHECKING:
    from collections.abc import Mapping

    import asyncpg

__all__: list[str] = [
    "CURRENT_MONTH_SQL",
    "AttemptUsage",
    "Charge",
    "MonthUsage",
    "available",
    "charge",
    "global_used",
    "grant_credit",
    "month_usage",
]

# First day of the current UTC month by the database clock.
CURRENT_MONTH_SQL: Final = "(date_trunc('month', now() AT TIME ZONE 'UTC'))::date"

_LOCK_NAMESPACE: Final = "llm-wallet"


@dataclass(frozen=True, slots=True)
class Charge:
    """One written charge: the attempt's job kind and how its amount split."""

    job_kind: str
    split: ChargeSplit


@dataclass(frozen=True, slots=True)
class AttemptUsage:
    """What one attempt of a job used: raw provider counts and their weighted total."""

    user_id: UUID
    job_kind: str
    job_id: UUID
    attempt: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    weighted: int


@dataclass(frozen=True, slots=True)
class MonthUsage:
    """Charges of the current month across every user."""

    weighted: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


def _wallet_lock_key(user_id: UUID) -> int:
    digest = hashlib.blake2b(f"{_LOCK_NAMESPACE}:{user_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def available(
    conn: asyncpg.Connection, monthly_amounts: Mapping[UUID, int]
) -> dict[UUID, int]:
    """Return the available balance of every user in ``monthly_amounts``.

    ``monthly_amounts`` maps each user to their monthly allowance. A user with
    no charges and no grants has the whole allowance available.
    """
    ids = list(monthly_amounts)
    rows = await conn.fetch(
        f"""
        WITH charges AS (
            SELECT user_id,
                   sum(monthly_part) FILTER (WHERE month = {CURRENT_MONTH_SQL}) AS monthly_used,
                   sum(credit_part) AS credit_spent
            FROM llm_charges
            WHERE user_id = ANY($1::uuid[])
            GROUP BY user_id
        ), grants AS (
            SELECT user_id, sum(amount) AS granted
            FROM credit_grants
            WHERE user_id = ANY($1::uuid[])
            GROUP BY user_id
        )
        SELECT u.user_id,
               coalesce(c.monthly_used, 0) AS monthly_used,
               coalesce(c.credit_spent, 0) AS credit_spent,
               coalesce(g.granted, 0) AS granted
        FROM (SELECT unnest($1::uuid[]) AS user_id) AS u
        LEFT JOIN charges AS c ON c.user_id = u.user_id
        LEFT JOIN grants AS g ON g.user_id = u.user_id
        """,  # noqa: S608 - the interpolated expression is fixed SQL; values are bound
        ids,
    )
    # Overspend is recorded in credit_part, so monthly use past a (lowered) allowance is no debt.
    return {
        UUID(str(r["user_id"])): max(
            monthly_amounts[UUID(str(r["user_id"]))] - int(r["monthly_used"]), 0
        )
        + int(r["granted"])
        - int(r["credit_spent"])
        for r in rows
    }


async def global_used(conn: asyncpg.Connection) -> int:
    """Return the weighted tokens charged this month across every user."""
    total = await conn.fetchval(
        f"SELECT coalesce(sum(weighted), 0) FROM llm_charges WHERE month = {CURRENT_MONTH_SQL}"  # noqa: S608 - fixed SQL
    )
    return int(total)


async def month_usage(conn: asyncpg.Connection) -> MonthUsage:
    """Return this month's weighted and raw provider token totals across every user."""
    row = cast(
        "asyncpg.Record",
        await conn.fetchrow(
            f"""
            SELECT coalesce(sum(weighted), 0) AS weighted,
                   coalesce(sum(input_tokens), 0) AS input_tokens,
                   coalesce(sum(cached_input_tokens), 0) AS cached_input_tokens,
                   coalesce(sum(output_tokens), 0) AS output_tokens
            FROM llm_charges
            WHERE month = {CURRENT_MONTH_SQL}
            """  # noqa: S608 - the interpolated expression is fixed SQL
        ),
    )
    return MonthUsage(
        weighted=int(row["weighted"]),
        input_tokens=int(row["input_tokens"]),
        cached_input_tokens=int(row["cached_input_tokens"]),
        output_tokens=int(row["output_tokens"]),
    )


async def grant_credit(
    conn: asyncpg.Connection,
    user_id: UUID,
    amount: int,
    reason: str,
    idempotency_key: str,
) -> bool:
    """Grant ``amount`` of credit to ``user_id``; return whether a grant was written.

    A grant with the same ``idempotency_key`` already present writes nothing,
    so a retried grant source credits once. ``amount`` must be positive; the
    table's CHECK rejects anything else.
    """
    inserted = await conn.fetchval(
        """
        INSERT INTO credit_grants (user_id, amount, reason, idempotency_key)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING true
        """,
        user_id,
        amount,
        reason,
        idempotency_key,
    )
    return bool(inserted)


async def charge(
    conn: asyncpg.Connection, attempt: AttemptUsage, monthly_amount: int
) -> Charge | None:
    """Write the charge of one attempt; return it, or None when nothing was written.

    Must run inside the caller's transaction. A per-user advisory transaction
    lock serializes charges of one user, so two attempts finishing at once
    never both spend the same remaining allowance. The charge goes to the
    current month, read from the database clock once. A zero ``weighted``
    writes nothing, and so does an attempt already charged.
    """
    if attempt.weighted <= 0:
        return None
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1::bigint)", _wallet_lock_key(attempt.user_id)
    )
    row = cast(
        "asyncpg.Record",
        await conn.fetchrow(
            f"""
            SELECT m.month,
                   coalesce((
                       SELECT sum(c.monthly_part)
                       FROM llm_charges AS c
                       WHERE c.user_id = $1 AND c.month = m.month
                   ), 0) AS monthly_used
            FROM (SELECT {CURRENT_MONTH_SQL} AS month) AS m
            """,  # noqa: S608 - the interpolated expression is fixed SQL; values are bound
            attempt.user_id,
        ),
    )
    split = split_charge(monthly_amount - int(row["monthly_used"]), attempt.weighted)
    written = await conn.fetchval(
        """
        INSERT INTO llm_charges
            (user_id, job_kind, job_id, attempt, input_tokens, cached_input_tokens,
             output_tokens, weighted, monthly_part, credit_part, month)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (job_id, attempt) DO NOTHING
        RETURNING true
        """,
        attempt.user_id,
        attempt.job_kind,
        attempt.job_id,
        attempt.attempt,
        attempt.input_tokens,
        attempt.cached_input_tokens,
        attempt.output_tokens,
        attempt.weighted,
        split.monthly_part,
        split.credit_part,
        row["month"],
    )
    return Charge(job_kind=attempt.job_kind, split=split) if written else None
