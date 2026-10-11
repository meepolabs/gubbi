"""Repository module for the per-user LLM wallet: credit grants and charges.

Both tables are append-only; a balance is always a sum over them. Every amount
is in weighted tokens. A month is the first day of a UTC calendar month, read
from the database clock, so every reader and the charge writer agree on it.

A user's available balance is what remains of this month's allowance plus
credit. Credit may go negative; each month's allowance first pays off the debt
carried into that month, and only the rest is spendable. Repayment is derived,
never written: ``gubbi.extraction.wallet.balance`` folds the user's per-month
totals into the current balance.

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
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

from gubbi.extraction.wallet import Balance, ChargeSplit, MonthTotals, balance, split_charge

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

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


# One row per (user, month with activity), plus one row with a NULL month for
# a user with none; every row carries the current month. A grant counts in the
# UTC month it was created in.
_MONTH_TOTALS_SQL: Final = f"""
    WITH activity AS (
        SELECT user_id, month, monthly_part, credit_part, 0::bigint AS granted
        FROM llm_charges
        WHERE user_id = ANY($1::uuid[])
        UNION ALL
        SELECT user_id, (date_trunc('month', created_at AT TIME ZONE 'UTC'))::date,
               0, 0, amount
        FROM credit_grants
        WHERE user_id = ANY($1::uuid[])
    ), monthly AS (
        SELECT user_id, month,
               sum(monthly_part) AS monthly_part,
               sum(credit_part) AS credit_part,
               sum(granted) AS granted
        FROM activity
        GROUP BY user_id, month
    )
    SELECT u.user_id, {CURRENT_MONTH_SQL} AS current_month,
           m.month, m.monthly_part, m.credit_part, m.granted
    FROM unnest($1::uuid[]) AS u(user_id)
    LEFT JOIN monthly AS m ON m.user_id = u.user_id
"""  # noqa: S608 - the interpolated expression is fixed SQL; values are bound


async def _balances(
    conn: asyncpg.Connection, monthly_amounts: Mapping[UUID, int]
) -> tuple[date, dict[UUID, Balance]]:
    """Return the current month and each user's balance in it, from one statement."""
    rows: Sequence[asyncpg.Record] = await conn.fetch(_MONTH_TOTALS_SQL, list(monthly_amounts))
    months: dict[UUID, list[MonthTotals]] = defaultdict(list)
    for r in rows:
        if r["month"] is not None:
            months[UUID(str(r["user_id"]))].append(
                MonthTotals(
                    month=r["month"],
                    monthly_part=int(r["monthly_part"]),
                    credit_part=int(r["credit_part"]),
                    granted=int(r["granted"]),
                )
            )
    current_month: date = rows[0]["current_month"]
    # Every month is folded with today's allowance; a per-month plan history
    # would change this to the allowance in force in each month.
    return current_month, {
        user_id: balance(amount, months[user_id], current_month)
        for user_id, amount in monthly_amounts.items()
    }


async def available(
    conn: asyncpg.Connection, monthly_amounts: Mapping[UUID, int]
) -> dict[UUID, int]:
    """Return the available balance of every user in ``monthly_amounts``.

    ``monthly_amounts`` maps each user to their monthly allowance. A user with
    no charges and no grants has the whole allowance available. Debt carried
    into this month has already taken its share of the allowance. Anonymous
    charges belong to no user and count in no balance.
    """
    if not monthly_amounts:
        return {}
    _, balances = await _balances(conn, monthly_amounts)
    return {user_id: b.available for user_id, b in balances.items()}


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
    so a retried grant source credits once. ``amount`` must be positive and at
    most the table's single-grant bound; its CHECK rejects anything else. On a
    journal_app connection the policies admit only the scoped user's
    onboarding grant (reason ``onboarding``, key ``onboarding:<user id>``);
    any other grant kind needs a journal_admin connection.
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
    current month, read from the database clock once, and spends what debt
    repayment left of that month's allowance before credit. A zero ``weighted``
    writes nothing, and so does an attempt already charged.
    """
    if attempt.weighted <= 0:
        return None
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1::bigint)", _wallet_lock_key(attempt.user_id)
    )
    month, balances = await _balances(conn, {attempt.user_id: monthly_amount})
    split = split_charge(balances[attempt.user_id].monthly_left, attempt.weighted)
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
        month,
    )
    return Charge(job_kind=attempt.job_kind, split=split) if written else None
