"""Scaffold an operator row in the users table.

Idempotent function. Safe to call on every startup.

Usage:
    from journalctl.users.bootstrap import scaffold_operator
    # called during app lifespan before first request
    await scaffold_operator(admin_pool, settings.operator_email, settings.timezone)

Requires a pre-existing asyncpg pool (usually the admin pool).
Connects directly -- no alembic dependency.

Inserts ON CONFLICT DO NOTHING on the partial unique index
(users.email WHERE deleted_at IS NULL). If a row already exists, this is a
no-op and proceeds to verify the row exists.
"""

from __future__ import annotations

import asyncpg


async def scaffold_operator(
    admin_pool: asyncpg.Pool,
    email: str,
    timezone: str = "UTC",
) -> None:
    """Ensure the operator row exists in users table (idempotent).

    Inserts when absent; raises RuntimeError if no active user row is
    found after the insert attempt.
    """
    try:
        async with admin_pool.acquire() as conn:
            # Insert operator row if it does not exist.
            insert_sql = (
                "INSERT INTO users (id, email, timezone) "
                "VALUES (gen_random_uuid(), $1, $2) "
                "ON CONFLICT (email) WHERE deleted_at IS NULL DO NOTHING"
            )
            await conn.execute(insert_sql, email, timezone)

            # Look up the row -- will exist regardless of whether INSERT created it.
            user_id = await conn.fetchval(
                "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
                email,
            )

            if user_id is None:
                raise RuntimeError(f"No active user row found after provisioning for {email}")

    except asyncpg.PostgresError as exc:
        raise RuntimeError(f"PostgreSQL error during scaffold: {exc}") from exc
