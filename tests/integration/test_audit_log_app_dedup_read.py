"""Live-role contract for the narrow read capability the deduped audit INSERT needs.

``record_audit_deduped_async`` routes through ``AUDIT_INSERT_DEDUPED_SQL``,
whose ``ON CONFLICT (actor_id, target_kind, target_id, action,
(metadata->>'content_hash'))`` inference clause makes PostgreSQL read those
five columns. Executed against PostgreSQL 17 both of these are required and
neither works alone:

- column-level SELECT for ``journal_app`` on exactly those five columns; and
- a self-only SELECT policy on ``audit_log`` for ``journal_app`` mirroring the
  existing ``audit_log_app_insert_self_only`` WITH CHECK predicate.

Table-wide SELECT stays denied, so every other audit column -- including
``actor_type``, ``occurred_at``, ``reason``, ``ip_address`` and ``user_agent``
-- remains unreadable through the app role, and ``SELECT *`` fails outright.

The denied-column set is derived from ``information_schema.columns`` minus the
five conflict-target columns rather than hardcoded, so a future column added to
``audit_log`` is asserted denied without editing this file.
"""

from __future__ import annotations

import hashlib
import re
import time
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.audit.actions import Action
from gubbi_common.audit.sql import record_audit_async, record_audit_deduped_async
from gubbi_common.audit.targets import TargetKind
from gubbi_common.db.user_scoped import user_scoped_connection

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_APP_ROLE = "journal_app"
_SELECT_POLICY_NAME = "audit_log_app_select_self_only"
_INSERT_POLICY_NAME = "audit_log_app_insert_self_only"

# The five columns the dedup partial-index inference clause reads.
CONFLICT_TARGET_COLUMNS = ("actor_id", "target_kind", "target_id", "action", "metadata")

_RLS_OR_PRIVILEGE_ERROR_RE = re.compile(
    r"row-level security|permission denied",
    re.IGNORECASE,
)


def _normalize_predicate(expr: str | None) -> str:
    """Strip whitespace and pg's ``::text`` casts so the two policies compare.

    ``pg_get_expr`` reprints a USING and a WITH CHECK of the same expression in
    forms that differ cosmetically, so both sides are normalized identically.
    """
    if expr is None:
        return ""
    return re.sub(r"\s+|::text|[()]", "", expr)


async def _audit_log_columns(conn: asyncpg.Connection) -> tuple[str, ...]:
    """Every column of ``public.audit_log``, read from the catalog."""
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'audit_log'
        ORDER BY column_name
        """
    )
    return tuple(str(row["column_name"]) for row in rows)


def _content_hash(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Privilege shape (catalog assertions through the admin pool)
# ---------------------------------------------------------------------------


async def test_app_role_reads_exactly_the_five_conflict_target_columns(
    admin_pool: asyncpg.Pool,
) -> None:
    # Arrange
    async with admin_pool.acquire() as conn:
        columns = await _audit_log_columns(conn)
        assert set(CONFLICT_TARGET_COLUMNS) <= set(columns), (
            f"conflict-target columns missing from audit_log: {columns!r}"
        )

        # Act
        readable = {
            column: await conn.fetchval(
                "SELECT has_column_privilege($1, 'public.audit_log', $2, 'SELECT')",
                _APP_ROLE,
                column,
            )
            for column in columns
        }

    # Assert
    granted = {column for column, ok in readable.items() if ok}
    assert granted == set(CONFLICT_TARGET_COLUMNS), (
        "journal_app readable column set diverges from the dedup conflict target: "
        f"expected={sorted(CONFLICT_TARGET_COLUMNS)} got={sorted(granted)}"
    )


async def test_app_role_still_has_no_table_wide_select_on_audit_log(
    admin_pool: asyncpg.Pool,
) -> None:
    # Arrange / Act
    async with admin_pool.acquire() as conn:
        has_table_select = await conn.fetchval(
            "SELECT has_table_privilege($1, 'public.audit_log', 'SELECT')", _APP_ROLE
        )

    # Assert
    assert has_table_select is False, (
        "journal_app must not hold table-wide SELECT on audit_log -- "
        "only the five dedup conflict-target columns are readable"
    )


async def test_self_only_select_policy_is_the_only_select_surface(
    admin_pool: asyncpg.Pool,
) -> None:
    """The complete SELECT-applicable policy set, not just the expected policy's existence.

    Permissive policies for the same command are OR-ed, so a second permissive
    SELECT (or ``FOR ALL``) policy would widen journal_app's read surface to
    every audit row while every existence check on the expected policy still
    passed. ``polroles = {0}`` is ``TO PUBLIC``, which also applies to
    journal_app, so it is part of the matched set.
    """
    # Arrange / Act
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT polname,
                   polcmd::text AS cmd,
                   polpermissive,
                   ARRAY(
                       SELECT rolname::text FROM pg_roles
                       WHERE oid = ANY (polroles) ORDER BY rolname
                   ) AS roles,
                   pg_get_expr(polqual, polrelid) AS using_expr
            FROM pg_policy
            WHERE polrelid = 'public.audit_log'::regclass
              AND polcmd IN ('r', '*')
              AND ($1::regrole = ANY (polroles) OR 0 = ANY (polroles))
            ORDER BY polname
            """,
            _APP_ROLE,
        )
        insert_expr = await conn.fetchval(
            """
            SELECT pg_get_expr(polwithcheck, polrelid)
            FROM pg_policy
            WHERE polname = $1 AND polrelid = 'public.audit_log'::regclass
            """,
            _INSERT_POLICY_NAME,
        )

    # Assert
    assert [row["polname"] for row in rows] == [_SELECT_POLICY_NAME], (
        "journal_app's SELECT-applicable policy set on audit_log must be exactly "
        f"{_SELECT_POLICY_NAME!r}, got {[row['polname'] for row in rows]!r}"
    )
    policy = rows[0]
    assert policy["cmd"] == "r", f"expected polcmd SELECT, got {policy['cmd']!r}"
    assert policy["polpermissive"] is True, (
        "policy must be PERMISSIVE -- a RESTRICTIVE one ANDs and denies the dedup read"
    )
    assert list(policy["roles"]) == [_APP_ROLE], (
        f"policy must be scoped TO journal_app only, got {policy['roles']!r}"
    )
    assert insert_expr, f"missing {_INSERT_POLICY_NAME} WITH CHECK predicate"
    assert _normalize_predicate(policy["using_expr"]) == _normalize_predicate(insert_expr), (
        "SELECT policy USING must mirror the INSERT policy WITH CHECK:\n"
        f"select={policy['using_expr']!r}\ninsert={insert_expr!r}"
    )


# ---------------------------------------------------------------------------
# Canonical deduped write through the user-scoped app connection
# ---------------------------------------------------------------------------


async def test_user_scoped_deduped_insert_succeeds_then_retry_deduplicates(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    # Arrange
    payload = {
        "actor_type": "user",
        "actor_id": str(tenant_a),
        "action": Action.ENTRY_CREATED,
        "target_kind": TargetKind.ENTRY,
        "target_type": "entry",
        "target_id": "entry-1",
        "metadata": {"content_hash": _content_hash(str(tenant_a), "entry-1")},
    }

    # Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        first = await record_audit_deduped_async(conn, **payload)
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        retry = await record_audit_deduped_async(conn, **payload)

    # Assert
    assert first is True, "canonical user-scoped deduped insert must record a row"
    assert retry is False, "re-delivery must collapse at the partial unique index"


async def test_unscoped_app_connection_deduped_insert_is_rejected(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    # Arrange / Act / Assert
    async with app_pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_OR_PRIVILEGE_ERROR_RE):
            await record_audit_deduped_async(
                conn,
                actor_type="user",
                actor_id=str(tenant_a),
                action=Action.ENTRY_CREATED,
                target_kind=TargetKind.ENTRY,
                target_type="entry",
                target_id="entry-unscoped",
                metadata={"content_hash": _content_hash("unscoped")},
            )


async def test_cross_user_deduped_insert_is_rejected(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    # Arrange / Act / Assert
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_OR_PRIVILEGE_ERROR_RE):
            await record_audit_deduped_async(
                conn,
                actor_type="user",
                actor_id=str(tenant_b),
                action=Action.ENTRY_CREATED,
                target_kind=TargetKind.ENTRY,
                target_type="entry",
                target_id="entry-cross",
                metadata={"content_hash": _content_hash("cross")},
            )


async def test_admin_pool_user_attributed_deduped_insert_is_rejected(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """The admin cross-attribution trigger still refuses actor_type='user'."""
    # Arrange / Act / Assert
    async with admin_pool.acquire() as conn:
        with pytest.raises(
            asyncpg.PostgresError,
            match="journal_admin cannot insert audit_log row with actor_type=user",
        ):
            await record_audit_deduped_async(
                conn,
                actor_type="user",
                actor_id=str(tenant_a),
                action=Action.ENTRY_CREATED,
                target_kind=TargetKind.ENTRY,
                target_type="entry",
                target_id="entry-admin",
                metadata={"content_hash": _content_hash("admin")},
            )


async def test_non_deduped_user_scoped_insert_still_records(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """The canonical non-dedup writer is unaffected by the new read capability."""
    # Arrange / Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await record_audit_async(
            conn,
            actor_type="user",
            actor_id=str(tenant_a),
            action=Action.ENTRY_UPDATED,
            target_type="entry",
            target_kind=TargetKind.ENTRY,
            target_id="entry-plain",
        )

    # Assert
    async with admin_pool.acquire() as conn:
        recorded = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE actor_id = $1 AND action = $2",
            str(tenant_a),
            str(Action.ENTRY_UPDATED),
        )
    assert recorded == 1


async def test_billing_shaped_user_attributed_deduped_write_records(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """The existing user-attributed billing audit write lands on this schema.

    The wire shape (action, target_kind, target_type, target_id, and the
    ``endpoint`` + bucketed ``content_hash`` metadata pair) mirrors the
    checkout-session billing guard's emission. Reproduced here through the
    canonical common writer so this repo's test does not import the cloud
    package.
    """
    # Arrange
    bucket = int(time.time() // 60)
    metadata = {
        "endpoint": "checkout_session",
        "content_hash": _content_hash(str(tenant_a), "checkout_session", str(bucket)),
    }

    # Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        inserted = await record_audit_deduped_async(
            conn,
            actor_type="user",
            actor_id=str(tenant_a),
            action=Action.BILLING_EMAIL_UNVERIFIED_BLOCKED,
            target_type="user",
            target_id=str(tenant_a),
            target_kind=TargetKind.USER,
            metadata=metadata,
        )

    # Assert
    assert inserted is True, "billing-shaped user-attributed deduped write must record"
    async with admin_pool.acquire() as conn:
        endpoint = await conn.fetchval(
            "SELECT metadata->>'endpoint' FROM audit_log WHERE actor_id = $1 AND action = $2",
            str(tenant_a),
            str(Action.BILLING_EMAIL_UNVERIFIED_BLOCKED),
        )
    assert endpoint == "checkout_session"


# ---------------------------------------------------------------------------
# Read surface through the app role
# ---------------------------------------------------------------------------


async def test_own_row_is_readable_on_the_five_columns(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    # Arrange
    target_id = "entry-readback"
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await record_audit_deduped_async(
            conn,
            actor_type="user",
            actor_id=str(tenant_a),
            action=Action.ENTRY_CREATED,
            target_kind=TargetKind.ENTRY,
            target_type="entry",
            target_id=target_id,
            metadata={"content_hash": _content_hash("readback")},
        )

    # Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        row = await conn.fetchrow(
            "SELECT actor_id, target_kind, target_id, action, metadata "
            "FROM audit_log WHERE target_id = $1",
            target_id,
        )

    # Assert
    assert row is not None, "author must be able to read back its own audit row"
    assert row["actor_id"] == str(tenant_a)
    assert row["target_kind"] == str(TargetKind.ENTRY)


async def test_select_star_on_audit_log_is_rejected(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    # Arrange / Act / Assert
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch("SELECT * FROM audit_log")


async def test_every_non_conflict_target_column_is_denied_to_the_app_role(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """A live SELECT of any other column fails, including the sensitive ones.

    ``has_column_privilege`` is asserted separately; this exercises the real
    statement path so a grant that looked narrow in the catalog but was widened
    by some other means still trips.

    Each column gets its own scoped connection: the first denial aborts the
    surrounding transaction, so a shared one would report
    ``InFailedSQLTransactionError`` for every later column and prove nothing
    about their privileges.
    """
    # Arrange
    async with admin_pool.acquire() as conn:
        columns = await _audit_log_columns(conn)
    denied = tuple(c for c in columns if c not in CONFLICT_TARGET_COLUMNS)
    assert denied, "expected audit_log to carry columns outside the conflict target"
    # The sensitive surface must be part of what we prove denied.
    assert {"actor_type", "occurred_at", "reason", "ip_address", "user_agent"} <= set(denied)

    # Act
    readable: list[str] = []
    for column in denied:
        async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
            try:
                await conn.fetch(f'SELECT "{column}" FROM audit_log')  # noqa: S608
            except asyncpg.InsufficientPrivilegeError:
                continue
            readable.append(column)

    # Assert
    assert not readable, f"journal_app can read columns it must not: {readable}"


async def test_other_user_rows_are_invisible(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    # Arrange
    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        await record_audit_deduped_async(
            conn,
            actor_type="user",
            actor_id=str(tenant_b),
            action=Action.ENTRY_CREATED,
            target_kind=TargetKind.ENTRY,
            target_type="entry",
            target_id="entry-of-b",
            metadata={"content_hash": _content_hash("b")},
        )

    # Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        visible = await conn.fetch("SELECT actor_id, target_id FROM audit_log")
        own_view_of_b = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE actor_id = $1", str(tenant_b)
        )

    # Assert
    assert own_view_of_b == 0, "user A must not see rows authored by user B"
    assert all(row["actor_id"] == str(tenant_a) for row in visible), (
        f"cross-user rows leaked into user A's view: {visible!r}"
    )


async def test_system_and_admin_rows_about_the_user_are_invisible(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """A non-user actor_type row naming user A stays outside A's read surface.

    The self-only SELECT policy predicates on ``actor_type = 'user'`` as well as
    the actor id, so a system or admin row whose ``target_id`` is user A is not
    readable by A even though A's id appears in it.
    """
    # Arrange
    async with admin_pool.acquire() as conn:
        for actor_type, actor_id in (("system", "system:probe"), ("admin", "admin:probe")):
            await conn.execute(
                """
                INSERT INTO audit_log
                    (actor_type, actor_id, action, target_kind, target_type, target_id)
                VALUES ($1, $2, $3, 'user', 'user', $4)
                """,
                actor_type,
                actor_id,
                str(Action.IDENTITY_DELETED),
                str(tenant_a),
            )
        planted = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE target_id = $1", str(tenant_a)
        )
    assert planted == 2, "fixture failed to plant the system/admin rows"

    # Act
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        visible = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE target_id = $1", str(tenant_a)
        )

    # Assert
    assert visible == 0, "system/admin rows about user A must be invisible to A"
