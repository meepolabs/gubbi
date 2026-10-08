"""Live posture of the roles tools/testdb/bootstrap.sql provisions.

Every database-backed suite runs on a cluster that ``testdb.py reset`` rebuilt:
roles from bootstrap.sql, then the migrations. These cases read what that
produced, on the server itself, rather than the SQL text:

- journal_app holds no BYPASSRLS (and no SUPERUSER), or row-level security is
  advisory for every user-facing connection and the RLS suites pass vacuously.
- journal_admin holds BYPASSRLS, which the cross-tenant maintenance paths need.
- bootstrap.sql hands journal_app no mutating DML, neither on tables that
  already exist when it runs nor through default privileges on tables created
  after it. The table grants journal_app does hold come from the migrations.
- the superuser that runs bootstrap.sql owns schema public, and neither
  deployed role is, or is a member of, that owner: the migrations and the
  ownership-reassigning suites run as that superuser.

The bootstrap cases run bootstrap.sql in a scratch database on the working
cluster. The roles are cluster-global, so the fixture holds the cluster role
lock the other role-mutating fixtures take for as long as the scratch database
lives. Under that lock it snapshots both deployed roles (existence, the
attributes and memberships the harness tracks, password presence) before
bootstrap.sql runs and restores exactly that snapshot afterwards, dropping a
role the run created, even when a step in between fails.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
import pytest_asyncio

from tests.conftest import TEST_DATABASE_URL
from tests.fixtures.cluster_roles import (
    ClusterRoleState,
    capture_cluster_state,
    cluster_role_lock,
    maintenance_dsn,
    quote_role,
    restore_cluster_state,
)
from tests.fixtures.db_invariants import REPO_ROOT, psql_bin

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

BOOTSTRAP_SQL = REPO_ROOT / "tools" / "testdb" / "bootstrap.sql"
# bootstrap.sql reads role passwords from these; unset, it leaves passwords alone.
_PASSWORD_ENVS = ("JOURNAL_DB_APP_PASSWORD", "JOURNAL_DB_ADMIN_PASSWORD", "PG_OTEL_RO_PASSWORD")

_APP_ROLE = "journal_app"
_ADMIN_ROLE = "journal_admin"
_DEPLOYED_ROLES = (_APP_ROLE, _ADMIN_ROLE)
_MUTATING_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE")
_PROBE_TABLES = ("probe_before_bootstrap", "probe_after_bootstrap")
_SCRATCH_PREFIX = "bootstrap_posture_"

CleanupStep = tuple[str, Callable[[], Awaitable[None]]]


async def _connect(dsn: str) -> asyncpg.Connection:
    try:
        return await asyncpg.connect(dsn, timeout=5)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"PostgreSQL not reachable for the bootstrap posture cases: {exc}")


async def _role_flags(role: str) -> tuple[bool, bool]:
    """Return ``(rolsuper, rolbypassrls)`` for ``role``; fail when it is absent."""
    conn = await _connect(TEST_DATABASE_URL)
    try:
        row = await conn.fetchrow(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = $1", role
        )
    finally:
        await conn.close()
    assert row is not None, f"{role} does not exist; reset runs bootstrap.sql to create it"
    return bool(row["rolsuper"]), bool(row["rolbypassrls"])


def _with_database(dsn: str, database: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{database}"))


def _run_bootstrap(dsn: str, *, admin_createrole: bool) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key not in _PASSWORD_ENVS}
    createrole = f"admin_createrole={str(admin_createrole).lower()}"
    return subprocess.run(  # noqa: S603 -- argv of the resolved psql and repo paths
        [
            psql_bin(),
            "-v",
            "ON_ERROR_STOP=1",
            "-v",
            createrole,
            "-X",
            "-q",
            "-f",
            str(BOOTSTRAP_SQL),
            "-d",
            dsn,
        ],
        cwd=Path(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass(frozen=True)
class _DeployedRoleSnapshot:
    """The deployed roles' cluster-global state, as of before bootstrap.sql ran."""

    roles: ClusterRoleState
    has_password: dict[str, bool]


async def _password_presence(conn: asyncpg.Connection) -> dict[str, bool]:
    rows = await conn.fetch(
        "SELECT rolname, rolpassword IS NOT NULL AS has_password"
        " FROM pg_authid WHERE rolname = ANY($1::text[])",
        list(_DEPLOYED_ROLES),
    )
    return {str(row["rolname"]): bool(row["has_password"]) for row in rows}


async def _snapshot_deployed_roles(conn: asyncpg.Connection) -> _DeployedRoleSnapshot:
    return _DeployedRoleSnapshot(
        roles=await capture_cluster_state(conn, _DEPLOYED_ROLES),
        has_password=await _password_presence(conn),
    )


async def _restore_password_presence(
    conn: asyncpg.Connection, before: _DeployedRoleSnapshot
) -> None:
    """Clear a password the run added; fail on one it removed, which cannot be recovered."""
    current = await _password_presence(conn)
    for role in sorted(before.roles.present & set(current)):
        had, has = before.has_password.get(role, False), current[role]
        if has and not had:
            await conn.execute(f"ALTER ROLE {quote_role(role)} WITH PASSWORD NULL")
        elif had and not has:
            pytest.fail(f"bootstrap.sql removed the password of {role}; it cannot be restored")


async def _restore_deployed_roles(conn: asyncpg.Connection, before: _DeployedRoleSnapshot) -> None:
    created = [role for role in _DEPLOYED_ROLES if role not in before.roles.present]
    await restore_cluster_state(conn, before.roles, _DEPLOYED_ROLES, disposable=created)
    await _restore_password_presence(conn, before)


async def _bootstrap_between_probes(dsn: str, before: _DeployedRoleSnapshot) -> None:
    """Create one probe table, run bootstrap.sql, then create the other."""
    admin_attributes = before.roles.attributes.get(_ADMIN_ROLE)
    # An absent journal_admin is created by the run and dropped by the restore, so
    # bootstrap.sql's own default is as good as any value for it.
    admin_createrole = True if admin_attributes is None else admin_attributes.can_create_role
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        await conn.execute(f"CREATE TABLE public.{_PROBE_TABLES[0]} (id int)")
        result = _run_bootstrap(dsn, admin_createrole=admin_createrole)
        assert result.returncode == 0, (
            f"bootstrap.sql failed in a scratch database:\n{result.stderr}"
        )
        await conn.execute(f"CREATE TABLE public.{_PROBE_TABLES[1]} (id int)")
    finally:
        await conn.close()


async def _run_cleanup_steps(steps: tuple[CleanupStep, ...]) -> list[BaseException]:
    failures: list[BaseException] = []
    for label, step in steps:
        try:
            await step()
        except BaseException as exc:  # pytest.fail raises a BaseException subclass
            exc.add_note(f"raised by cleanup step: {label}")
            failures.append(exc)
    return failures


def _note_cleanup_failures(target: BaseException, failures: list[BaseException]) -> None:
    for failure in failures:
        target.add_note(f"cleanup also failed: {type(failure).__name__}: {failure}")


@asynccontextmanager
async def cleanup_after(*steps: CleanupStep) -> AsyncIterator[None]:
    """Run every step in order after the body, without masking the body's error.

    A failing body is re-raised with each cleanup failure attached as a note. A
    clean body with a failing step raises that step's error, later step failures
    attached as notes. A failing step never stops the steps after it.
    """
    try:
        yield
    except BaseException as body_failure:
        _note_cleanup_failures(body_failure, await _run_cleanup_steps(steps))
        raise
    failures = await _run_cleanup_steps(steps)
    if failures:
        first, *later = failures
        _note_cleanup_failures(first, later)
        raise first


@pytest_asyncio.fixture(scope="module")
async def bootstrapped_db() -> AsyncIterator[str]:
    """A scratch database where bootstrap.sql ran between two probe-table creations.

    The scratch database is dropped and the deployed roles restored before the
    cluster role lock is released, whichever step fails.
    """
    name = f"{_SCRATCH_PREFIX}{uuid.uuid4().hex[:8]}"
    dsn = _with_database(TEST_DATABASE_URL, name)
    probe = await _connect(maintenance_dsn(TEST_DATABASE_URL))
    await probe.close()
    async with cluster_role_lock(TEST_DATABASE_URL) as lock_conn:
        before = await _snapshot_deployed_roles(lock_conn)

        async def drop_scratch_db() -> None:
            await lock_conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

        async def restore_roles() -> None:
            await _restore_deployed_roles(lock_conn, before)

        async with cleanup_after(
            ("drop scratch database", drop_scratch_db),
            ("restore deployed roles", restore_roles),
        ):
            await lock_conn.execute(f'CREATE DATABASE "{name}"')
            await _bootstrap_between_probes(dsn, before)
            yield dsn


@pytest.mark.parametrize(
    ("role", "expected", "contract"),
    [
        pytest.param(
            _APP_ROLE,
            (False, False),
            "NOSUPERUSER NOBYPASSRLS, or RLS does not apply to it",
            id="app_role_neither_bypassrls_nor_superuser",
        ),
        pytest.param(
            _ADMIN_ROLE,
            (False, True),
            "NOSUPERUSER BYPASSRLS",
            id="admin_role_bypasses_rls_without_superuser",
        ),
    ],
)
async def test_the_deployed_role_holds_its_posture_flags(
    role: str, expected: tuple[bool, bool], contract: str
) -> None:
    is_superuser, bypasses_rls = await _role_flags(role)

    assert (is_superuser, bypasses_rls) == expected, (
        f"{role} must be {contract}: rolsuper={is_superuser} rolbypassrls={bypasses_rls}"
    )


async def test_bootstrap_grants_the_app_role_no_mutating_dml(bootstrapped_db: str) -> None:
    conn = await asyncpg.connect(bootstrapped_db, timeout=5)
    try:
        rows = await conn.fetch(
            "SELECT t.name AS table_name, p.name AS privilege"
            " FROM unnest($1::text[]) AS t(name), unnest($2::text[]) AS p(name)"
            " WHERE has_table_privilege($3, 'public.' || t.name, p.name)",
            list(_PROBE_TABLES),
            list(_MUTATING_PRIVILEGES),
            _APP_ROLE,
        )
        can_create = await conn.fetchval(
            "SELECT has_schema_privilege($1, 'public', 'CREATE')", _APP_ROLE
        )
    finally:
        await conn.close()

    held = sorted((str(row["table_name"]), str(row["privilege"])) for row in rows)
    assert held == [], f"bootstrap.sql hands {_APP_ROLE} mutating DML: {held}"
    assert can_create is False, f"bootstrap.sql lets {_APP_ROLE} create objects in public"


async def test_the_bootstrap_superuser_owns_the_schema(bootstrapped_db: str) -> None:
    conn = await asyncpg.connect(bootstrapped_db, timeout=5)
    try:
        row = await conn.fetchrow(
            "SELECT owner.rolname AS owner, owner.rolsuper AS is_superuser,"
            " current_user AS bootstrap_user"
            " FROM pg_namespace n"
            " JOIN pg_database d ON d.datname = current_database()"
            " JOIN pg_roles owner ON owner.oid = CASE"
            "   WHEN n.nspowner = 'pg_database_owner'::regrole THEN d.datdba"
            "   ELSE n.nspowner END"
            " WHERE n.nspname = 'public'"
        )
        assert row is not None, "schema public is missing from the scratch database"
        members = await conn.fetchval(
            "SELECT array_agg(r ORDER BY r) FROM unnest($1::text[]) AS r"
            " WHERE pg_has_role(r, $2, 'MEMBER')",
            list(_DEPLOYED_ROLES),
            str(row["owner"]),
        )
    finally:
        await conn.close()

    assert (row["owner"], row["is_superuser"]) == (row["bootstrap_user"], True), (
        f"schema public must be owned by the superuser that ran bootstrap.sql "
        f"({row['bootstrap_user']}), got {row['owner']} (superuser={row['is_superuser']})"
    )
    assert members is None, f"deployed roles own or are members of the schema owner: {members}"
