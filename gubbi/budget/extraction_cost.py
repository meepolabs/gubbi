"""Worker-side budget delta writer.

Mirrors the previously-cloud-side function. Worker calls this after
extraction completes to write a delta against the user's Redis budget
counter; the gubbi-cloud reconciler then flushes Redis -> PG.

Budget delta failure is best-effort: the extraction itself succeeded,
the row write succeeded, and the pre-charge already protected the cap.
Worst case on missed delta: user pays slightly over by the pre-charge
amount until the next reconciler run. Callers MUST NOT roll back their
persistence transaction on Redis failure here.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol
from uuid import UUID


class _BudgetRedisClient(Protocol):
    async def hincrby(self, key: str, field: str, amount: int) -> int: ...
    async def sadd(self, key: str, *values: str) -> int: ...
    async def expire(self, key: str, time: int) -> bool: ...


async def record_extraction_cost(
    user_id: UUID,
    period_start: date,
    *,
    actual_cents: int,
    estimated_cents: int,
    redis: _BudgetRedisClient | None,
) -> None:
    """Write the actual-vs-estimated delta to the user's Redis budget counter."""
    if redis is None:
        return
    delta = actual_cents - estimated_cents
    key = f"budget:{user_id}:{period_start}"
    member = f"{user_id}:{period_start}"
    await redis.hincrby(key, "used_cents", delta)
    await redis.sadd("budget:dirty", member)
    await redis.expire(key, 3600)
