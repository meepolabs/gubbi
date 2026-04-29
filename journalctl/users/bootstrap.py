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
no-op and proceeds to verify the row exists. A user.created audit row is
written ONLY when the INSERT actually creates a row (RETURNING id is None
on no-op).
"""

from __future__ import annotations

import logging

import asyncpg

from journalctl.audit import Action, record_audit

logger = logging.getLogger(__name__)


async def scaffold_operator(
    admin_pool: asyncpg.Pool,
    email: str,
    timezone: str = "UTC",
) -> None:
    """Ensure the operator row exists in users table (idempotent).

    Inserts when absent; raises RuntimeError if no active user row is
    found after the insert attempt.

    User-row write paths (M2 review #6):

    * **This path (Mode 1/2 self-host only)** -- runs once at journalctl
      startup to ensure the founder row exists. ON CONFLICT DO NOTHING
      keyed on the partial unique index ``idx_users_email_active``
      (email WHERE deleted_at IS NULL). Disjoint from Mode 3 paths.
    * **Kratos webhook (Mode 3 fast path)** --
      ``journalctl-cloud/journalctl_cloud/webhooks/kratos.py``
      ``_upsert_user``; never fires in Mode 1/2 (no Kratos).
    * **JIT (Mode 3 self-heal)** --
      ``journalctl/middleware/auth.py`` ``_pre_context_jwt_provision``;
      never fires in Mode 1/2 (no Hydra introspection path).

    Mode 3 hosted deploys do NOT call scaffold_operator -- the founder
    is just another user provisioned via the webhook + JIT pair.
    """
    try:
        async with admin_pool.acquire() as conn:
            # INSERT and detect whether we actually created a row -- RETURNING
            # returns the id on insert, NULL on the ON CONFLICT DO NOTHING path.
            inserted_id = await conn.fetchval(
                "INSERT INTO users (id, email, timezone) "
                "VALUES (gen_random_uuid(), $1, $2) "
                "ON CONFLICT (email) WHERE deleted_at IS NULL DO NOTHING "
                "RETURNING id",
                email,
                timezone,
            )

            if inserted_id is not None:
                # New row -- audit the creation. Best-effort: a failure to
                # write the audit row must not break startup.
                try:
                    await record_audit(
                        conn,
                        actor_type="system",
                        actor_id="scaffold_operator",
                        action=Action.IDENTITY_CREATED,
                        target_type="user",
                        target_id=str(inserted_id),
                        metadata={"provision_path": "scaffold"},
                    )
                except Exception:
                    logger.exception("scaffold_operator: audit write failed")

            # Verify a row exists (covers both newly-created and pre-existing).
            user_id = await conn.fetchval(
                "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
                email,
            )

            if user_id is None:
                raise RuntimeError(f"No active user row found after provisioning for {email}")

    except asyncpg.PostgresError as exc:
        raise RuntimeError(f"PostgreSQL error during scaffold: {exc}") from exc
