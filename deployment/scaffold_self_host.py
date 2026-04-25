"""Provision an operator row in the users table.

Idempotent one-shot script. Safe to re-run.

Usage:
    poetry run python -m journalctl.scripts.provision_operator
    poetry run python -m journalctl.scripts.provision_operator --email you@example.com
    JOURNAL_OPERATOR_EMAIL=you@example.com \\
        POETRY_RUN python -m journalctl.scripts.provision_operator

Requires ``JOURNAL_OPERATOR_EMAIL`` (env or ``--email``) and a PostgreSQL
DSN from  ``JOURNAL_DB_ADMIN_URL`` (fetched first) then ``JOURNAL_DB_APP_URL``.
Connects directly via asyncpg -- no pool, no alembic dependency.

Inserts ON CONFLICT DO NOTHING on the partial unique index
(users.email WHERE deleted_at IS NULL). If a row already exists, prints
the existing UUID and exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _fail(msg: str) -> None:
    """Print error to stderr and exit 1."""
    sys.stderr.write(f"{msg}\n")
    await asyncio.sleep(0)
    sys.exit(1)


async def main() -> None:
    """Provision operator row in users table."""
    parser = argparse.ArgumentParser(
        description="Provision an operator row in the users table (idempotent)."
    )
    env_email = os.environ.get("JOURNAL_OPERATOR_EMAIL", "")
    env_tz = os.environ.get("JOURNAL_TIMEZONE", "UTC")
    parser.add_argument("--email", default=env_email or None, help="Operator email address")
    parser.add_argument(
        "--timezone",
        default=env_tz or "UTC",
        help="Default timezone (default: UTC)",
    )
    args = parser.parse_args()

    if not args.email:
        await _fail("Email required. Set --email or JOURNAL_OPERATOR_EMAIL env var.")

    db_url = os.environ.get("JOURNAL_DB_ADMIN_URL") or os.environ.get("JOURNAL_DB_APP_URL")
    if not db_url:
        await _fail("No DATABASE URL found. Set JOURNAL_DB_ADMIN_URL or JOURNAL_DB_APP_URL.")

    import asyncpg  # noqa: PLC0415 -- stdlib-only until actual usage, keeps top-level clean

    conn = None
    try:
        conn = await asyncpg.connect(db_url)

        # Insert operator row if it does not exist.
        insert_sql = (
            "INSERT INTO users (id, email, timezone) "
            "VALUES (gen_random_uuid(), $1, $2) "
            "ON CONFLICT (email) WHERE deleted_at IS NULL DO NOTHING"
        )
        insert_tag = await conn.execute(insert_sql, args.email, args.timezone)
        # asyncpg returns a tag string like "INSERT 0 1" (created) or "INSERT 0 0" (conflict).
        newly_created = insert_tag.endswith(" 1")

        # Look up the row -- will exist regardless of whether INSERT created it.
        user_id = await conn.fetchval(
            "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
            args.email,
        )

        if user_id is None:
            await _fail(f"No active user row found after provisioning for {args.email}")

        if newly_created:
            print(f"Operator row provisioned. id={user_id} email={args.email}")  # noqa: T201
        else:
            print(f"Operator row already exists. id={user_id} email={args.email}")  # noqa: T201

    except asyncpg.PostgresError as exc:
        await _fail(f"PostgreSQL error: {exc}")
    except OSError as exc:
        await _fail(f"Cannot connect to PostgreSQL at {db_url}: {exc}")
    finally:
        if conn is not None:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
