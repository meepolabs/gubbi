"""The post-deploy verifier pins the superuser / BYPASSRLS posture of both roles.

Four catalog booleans make up the role posture the data plane depends on:

    journal_app    rolsuper = false, rolbypassrls = false
    journal_admin  rolsuper = false, rolbypassrls = true

``journal_app`` is the role every user-facing connection authenticates as, so
either boolean turning true makes row-level security advisory rather than
enforced. ``journal_admin`` needs BYPASSRLS for the cross-tenant maintenance
paths but never superuser, which would make every other privilege assertion in
the verifier vacuous.

Each boolean is asserted independently, in the direction that matters: the two
``journal_app`` booleans and ``journal_admin.rolsuper`` fail when BROADENED,
``journal_admin.rolbypassrls`` fails when WEAKENED. ``rolsuper`` and
``rolbypassrls`` are separate catalog columns -- a superuser role still reports
``rolbypassrls = false`` -- so each mutation moves exactly one boolean and each
test names the single tag it expects, with the sibling tag asserted absent.

Those four describe journal_app's OWN row, and role attributes are not
inherited. So a separate arm covers REACHABLE posture: whether journal_app can
come to execute as some other role that holds either attribute, over INHERIT,
SET and ADMIN edges, including the two-edge INHERIT-then-ADMIN chain. A
``pg_has_role(..., 'USAGE')`` test sees only the INHERIT closure.

CLUSTER-GLOBAL STATE, DATABASE-SCOPED LOCKS. Role attributes and memberships
live on the cluster, but advisory locks do not: a lock taken in the working
database does not contend with one taken in a sibling database, while both
sessions mutate the same ``pg_authid`` rows. Every fixture here therefore locks
the cluster's MAINTENANCE database (see ``tests/fixtures/cluster_roles.py``),
which makes one fixed key cluster-wide in effect, with a bounded
``lock_timeout`` so contention names the blocking session instead of hanging.

Restore is exact and narrow: only the attributes and membership options the
harness mutates are snapshotted and rewritten. A pre-existing shared role is
never dropped and never recreated -- four attributes cannot reconstitute a
role's password, config or connection limit -- so absence is exercised with
fixture-owned disposable roles.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
import pytest_asyncio
from _pytest.outcomes import Failed

from tests.conftest import RLS_BOOTSTRAP_URL
from tests.fixtures.cluster_roles import (
    LOCK_TIMEOUT_MS,
    MAINTENANCE_DATABASE,
    POSTURE_ADVISORY_LOCK_KEY,
    ClusterRoleState,
    PostureLockUnavailableError,
    assert_required_roles_present,
    capture_cluster_state,
    cluster_role_lock,
    maintenance_dsn,
    restore_cluster_state,
)
from tests.fixtures.db_invariants import (
    assert_clean_of,
    psql_bin,
    reached_report_block,
    run_argv,
    verifier_failure_tags,
    verifier_run,
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# A throwaway PostgreSQL 17 superuser DSN. Required only by the cases that need a
# role to be genuinely ABSENT: roles are cluster-global, so absence cannot be
# staged on the working cluster without destroying shared state.
DISPOSABLE_CLUSTER_ENV = "TEST_DISPOSABLE_CLUSTER_URL"

# Per-case database names this lane creates, and how many to name in a refusal.
_ABSENT_DATABASE_PREFIX = "posture_absent_"
_UNEXPECTED_DATABASE_SAMPLE = 5
_UNEXPECTED_ROLE_SAMPLE = 5

_APP_ROLE = "journal_app"
_ADMIN_ROLE = "journal_admin"
_OTEL_ROLE = "otel_ro"

# Roles the deployment contract requires. Their absence on a REACHABLE server is
# a harness defect, never a skip.
_REQUIRED_ROLES = (_APP_ROLE, _ADMIN_ROLE)

# The verifier's failure tags for the four booleans.
_APP_SUPERUSER_TAG = "journal_app_not_superuser"
_APP_BYPASSRLS_TAG = "journal_app_no_bypassrls"
_ADMIN_SUPERUSER_TAG = "journal_admin_not_superuser"
_ADMIN_BYPASSRLS_TAG = "journal_admin_bypassrls"
_REACHABLE_TAG = "reachable_posture journal_app"
_OTEL_EXISTS_TAG = "otel_ro_exists"
_ROLE_EXISTS_TAG = "role_exists"

_POSTURE_TAGS = (
    _APP_SUPERUSER_TAG,
    _APP_BYPASSRLS_TAG,
    _ADMIN_SUPERUSER_TAG,
    _ADMIN_BYPASSRLS_TAG,
)

# The exact diagnostic each of the four booleans produces once broken. The
# verifier reports the value it OBSERVED, so a broadened boolean reads 'f' where
# the assertion wanted true, and the weakened admin BYPASSRLS reads 'f' likewise.
_EXPECTED_POSTURE_LINES = {
    _APP_SUPERUSER_TAG: "journal_app_not_superuser: expected true, got 'f'",
    _APP_BYPASSRLS_TAG: "journal_app_no_bypassrls: expected true, got 'f'",
    _ADMIN_SUPERUSER_TAG: "journal_admin_not_superuser: expected true, got 'f'",
    _ADMIN_BYPASSRLS_TAG: "journal_admin_bypassrls: expected true, got 'f'",
}

# Mirrors REACHABLE_POSTURE_SAMPLE_LIMIT in the verifier. A drift here shows up as
# a failing count assertion in the >limit test rather than silently.
_REACHABLE_SAMPLE_LIMIT = 5

# Captured before any monkeypatch replaces it, so the cleanup-precedence tests can
# delegate to the real implementation for every statement except the one they fail.
_ORIGINAL_EXECUTE = asyncpg.Connection.execute

# Mirrors GUBBI_NON_AUDIT_TABLES in the verifier. Both roles' expected table lists
# derive from it there, INDEPENDENTLY of each other -- which is what keeps an
# absent journal_app from silently emptying the journal_admin loop.
_NON_AUDIT_TABLES = (
    "topics",
    "conversations",
    "entries",
    "messages",
    "entry_embeddings",
    "extraction_jobs",
)

# Roles the fixture OWNS: it creates them, and it is the only thing that drops
# them. Named distinctly from anything a deployment or another test module
# provisions. otel_ro is deliberately NOT in this list -- it is a shared role in
# every real topology, so absence is exercised with a disposable stand-in.
_INHERIT_PROBE_ROLE = "posture_probe_inherit"
_SET_PROBE_ROLE = "posture_probe_set"
_ADMIN_PROBE_ROLE = "posture_probe_admin"
_CHAIN_MID_ROLE = "posture_probe_chain_mid"
_CHAIN_TARGET_ROLE = "posture_probe_chain_target"
_BULK_PROBE_PREFIX = "posture_probe_bulk_"
_HOSTILE_QUOTE_ROLE = 'posture_probe_"quoted'
_HOSTILE_NEWLINE_ROLE = "posture_probe_nl\nFAIL: forged"
_HOSTILE_TAB_ROLE = "posture_probe_tab\tsplit"
_HOSTILE_CR_ROLE = "posture_probe_cr\roverwrite"
_HOSTILE_BACKSLASH_ROLE = "posture_probe_bs\\x0A"
_HOSTILE_LINE_SEP_ROLE = "posture_probe_ls\u2028sep"
# U+0085 NEXT LINE is a line break for str.splitlines() and many log readers, yet
# ascii() reports 133 for it -- outside a `< 32 OR = 127` control test. It gets its
# own case because that is exactly the gap an ASCII-only condition leaves open.
_HOSTILE_NEL_ROLE = "posture_probe_nel\u0085break"
_HOSTILE_C1_ROLE = "posture_probe_c1\u009fend"

# The EXACT single-line rendering each hostile name must produce. Encoding is
# reversible on purpose: a reader can tell \x0A from \x09, which a single
# placeholder would destroy. A pre-existing literal backslash is doubled, so
# ``bs\\x0A`` stays distinguishable from a real newline.
_EXPECTED_HOSTILE_RENDERING = {
    _HOSTILE_NEWLINE_ROLE: "posture_probe_nl\\x0AFAIL: forged",
    _HOSTILE_TAB_ROLE: "posture_probe_tab\\x09split",
    _HOSTILE_CR_ROLE: "posture_probe_cr\\x0Doverwrite",
    _HOSTILE_BACKSLASH_ROLE: "posture_probe_bs\\\\x0A",
    _HOSTILE_LINE_SEP_ROLE: "posture_probe_ls\\u2028sep",
    _HOSTILE_NEL_ROLE: "posture_probe_nel\\x85break",
    _HOSTILE_C1_ROLE: "posture_probe_c1\\x9Fend",
    _HOSTILE_QUOTE_ROLE: 'posture_probe_"quoted',
}

_DISPOSABLE_ROLES = (
    _INHERIT_PROBE_ROLE,
    _SET_PROBE_ROLE,
    _ADMIN_PROBE_ROLE,
    _CHAIN_MID_ROLE,
    _CHAIN_TARGET_ROLE,
    _HOSTILE_QUOTE_ROLE,
    _HOSTILE_NEWLINE_ROLE,
    _HOSTILE_TAB_ROLE,
    _HOSTILE_CR_ROLE,
    _HOSTILE_BACKSLASH_ROLE,
    _HOSTILE_LINE_SEP_ROLE,
    _HOSTILE_NEL_ROLE,
    _HOSTILE_C1_ROLE,
    *(f"{_BULK_PROBE_PREFIX}{n:02d}" for n in range(_REACHABLE_SAMPLE_LIMIT + 3)),
)

_TRACKED_ROLES = (*_REQUIRED_ROLES, _OTEL_ROLE, *_DISPOSABLE_ROLES)


def _quoted(role: str) -> str:
    """``role`` as a SQL identifier, doubling embedded quotes.

    Needed because two probe roles carry a quote and a newline on purpose -- the
    diagnostics tests plant them to prove the verifier cannot be made to emit a
    forged log line.
    """
    escaped = role.replace('"', '""')
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def _required_roles_present() -> None:
    """Fail hard, before any migration runs, if a required role is missing.

    Ordered BEFORE ``_rls_provisioned`` in every consumer's signature on purpose:
    that fixture runs ``alembic upgrade head``, whose failure it converts into a
    ``pytest.skip``. A missing journal_app or journal_admin makes the baseline
    migration fail on a GRANT, so without this gate a required-role defect would
    silently become a skip -- reporting green for a posture contract nothing
    measured. Only an UNREACHABLE server skips, and that is decided here too:
    reachability is established by connecting to the maintenance database, which
    no migration has to have touched.
    """
    try:
        conn = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"cannot reach PostgreSQL to verify required roles: {exc}")
    try:
        await assert_required_roles_present(conn, _REQUIRED_ROLES)
    finally:
        await conn.close()


@asynccontextmanager
async def _posture_lock(
    key: int = POSTURE_ADVISORY_LOCK_KEY,
) -> AsyncIterator[asyncpg.Connection]:
    """Hold the cluster role lock on the maintenance database for the body."""
    async with cluster_role_lock(RLS_BOOTSTRAP_URL, key) as conn:
        yield conn


@asynccontextmanager
async def _posture_session(
    lock_conn: asyncpg.Connection | None = None,
) -> AsyncIterator[str]:
    """Converge prerequisites under the cluster role lock, restoring on exit.

    ``lock_conn`` lets a caller that already holds the lock reuse it, so the hold
    stays unbroken across its own baseline read and the session's lifecycle.
    """
    if lock_conn is not None:
        async with _converged_cluster(lock_conn) as dsn:
            yield dsn
        return
    async with _posture_lock() as own_conn, _converged_cluster(own_conn) as dsn:
        yield dsn


@asynccontextmanager
async def _converged_cluster(lock_conn: asyncpg.Connection) -> AsyncIterator[str]:
    """Snapshot, converge the prerequisite roles, and restore on exit.

    The ``try``/``finally`` opens IMMEDIATELY after the snapshot and before any
    convergence statement runs, so a convergence that fails halfway -- one GRANT
    applied, the next raising -- still restores. Putting convergence outside the
    guard would leak exactly the partially-applied state it creates.
    """
    before = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
    # otel_ro is CONDITIONALLY disposable: on a migrated cluster the baseline
    # migration created it, so it is shared state this harness must preserve; on a
    # cluster where it is genuinely absent the convergence below creates it, which
    # makes it harness-owned and therefore the harness's to remove. The decision is
    # made from the snapshot, before anything is written, so it can never be based
    # on a role the harness itself just created. ``restore_cluster_state`` refuses
    # to drop anything present in ``before`` regardless, so this is belt and braces.
    disposable = _disposable_roles_for(before)
    try:
        await _converge_prerequisites(lock_conn)
        yield RLS_BOOTSTRAP_URL
    finally:
        await restore_cluster_state(lock_conn, before, _TRACKED_ROLES, disposable=disposable)


def _disposable_roles_for(before: ClusterRoleState) -> tuple[str, ...]:
    """The roles teardown may drop, given what existed at snapshot time.

    A role in ``before`` is shared state and is never disposable, whatever list it
    appears on. otel_ro joins the disposable set exactly when it was absent.
    """
    conditional = () if _OTEL_ROLE in before.present else (_OTEL_ROLE,)
    return (*_DISPOSABLE_ROLES, *conditional)


async def _converge_prerequisites(conn: asyncpg.Connection) -> None:
    """Bring the shared roles to the state the verifier assumes a deploy left.

    Only ALTERs and GRANTs shared roles -- never creates or drops one. otel_ro is
    created only if genuinely absent, which on a migrated topology it never is
    (the baseline migration creates it); when the harness does create it, teardown
    drops it because :func:`_disposable_roles_for` classified it as harness-owned
    from the pre-convergence snapshot.
    """
    await conn.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_OTEL_ROLE}') THEN
                CREATE ROLE {_OTEL_ROLE} LOGIN;
            END IF;
        END $$;
        """  # noqa: S608 -- interpolates a module-private role-name literal
    )
    await conn.execute(f"ALTER ROLE {_OTEL_ROLE} WITH LOGIN NOSUPERUSER NOBYPASSRLS")
    await conn.execute(f"GRANT pg_monitor TO {_OTEL_ROLE}")
    await conn.execute(f"GRANT {_APP_ROLE} TO {_ADMIN_ROLE} WITH ADMIN OPTION")


@pytest_asyncio.fixture
async def posture_dsn(_required_roles_present: None, _rls_provisioned: None) -> AsyncIterator[str]:
    """The migrated DSN the verifier runs against, with all cluster state restored.

    The verifier reads catalog state only, so the session-scoped RLS database is a
    valid target. What needs undoing is the cluster-global state, which
    :func:`_posture_session` owns under the maintenance-database lock.
    """
    async with _posture_session() as dsn:
        yield dsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _execute(dsn: str, statement: str) -> None:
    """Run one statement against ``dsn`` on its own short-lived connection."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        await conn.execute(statement)
    finally:
        await conn.close()


async def _role_booleans(dsn: str, role: str) -> tuple[bool, bool]:
    """``(rolsuper, rolbypassrls)`` as the catalog currently reports them."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        row = await conn.fetchrow(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = $1", role
        )
    finally:
        await conn.close()
    assert row is not None, f"{role} does not exist"
    return bool(row["rolsuper"]), bool(row["rolbypassrls"])


async def _assert_still_mutated(dsn: str, role: str, expected: tuple[bool, bool]) -> None:
    """Re-read the mutated booleans AFTER the verifier ran.

    Reading the catalog again proves the verifier observed the broken posture,
    rather than an already-restored clean one -- which would make an absent tag
    look like a real verifier gap, or a present tag unattributable.
    """
    actual = await _role_booleans(dsn, role)
    assert actual == expected, (
        f"{role}'s posture changed while the verifier ran: expected the mutation "
        f"{expected} to still be in place, found {actual}. The verifier therefore "
        "did not necessarily observe the state this test induced."
    )


async def _create_probe_role(dsn: str, role: str, attribute: str) -> None:
    """Create a NOLOGIN, fixture-owned probe role holding ``attribute``.

    ``role`` comes from ``_DISPOSABLE_ROLES`` and ``attribute`` from a parametrize
    list; role names and attribute keywords are identifiers and cannot be bound
    parameters in DDL.
    """
    ident = _quoted(role)
    literal = role.replace("'", "''")
    await _execute(
        dsn,
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{literal}') THEN
                CREATE ROLE {ident} NOLOGIN;
            END IF;
        END $$;
        ALTER ROLE {ident} WITH NOLOGIN {attribute};
        """,  # noqa: S608 -- module-private role literals and attribute keywords
    )


async def _has_usage(dsn: str, member: str, granted: str) -> bool:
    """Whether ``pg_has_role(member, granted, 'USAGE')`` -- the INHERIT closure."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        return bool(await conn.fetchval("SELECT pg_has_role($1, $2, 'USAGE')", member, granted))
    finally:
        await conn.close()


async def _role_exists(dsn: str, role: str) -> bool:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        return bool(
            await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)", role)
        )
    finally:
        await conn.close()


def _reachable_tags(run_tags: frozenset[str]) -> list[str]:
    return sorted(tag for tag in run_tags if _REACHABLE_TAG in tag)


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------


async def test_invariant_script_passes_every_role_posture_check_on_an_intact_database(
    posture_dsn: str,
) -> None:
    # Arrange
    app_booleans = await _role_booleans(posture_dsn, _APP_ROLE)
    admin_booleans = await _role_booleans(posture_dsn, _ADMIN_ROLE)
    assert app_booleans == (False, False), (
        "journal_app posture is already broadened before the test mutated anything"
    )
    assert admin_booleans == (False, True), (
        "journal_admin posture does not match the deployment contract before mutation"
    )

    # Act
    tags = verifier_failure_tags(posture_dsn)

    # Assert
    flagged = sorted(tag for tag in tags if any(t in tag for t in _POSTURE_TAGS))
    assert not flagged, f"verifier flagged role posture on an intact database: {flagged}"
    assert not _reachable_tags(tags), (
        f"verifier flagged reachable posture on an intact database: {sorted(tags)}"
    )
    assert not [tag for tag in tags if _ROLE_EXISTS_TAG in tag], (
        f"verifier reported a required role missing on an intact database: {sorted(tags)}"
    )


# ---------------------------------------------------------------------------
# One mutation per boolean
# ---------------------------------------------------------------------------


async def test_invariant_script_flags_a_superuser_app_role(posture_dsn: str) -> None:
    """SUPERUSER on journal_app makes every grant and policy assertion advisory."""
    # Arrange
    assert_clean_of(posture_dsn, _APP_SUPERUSER_TAG)
    await _execute(posture_dsn, f"ALTER ROLE {_APP_ROLE} WITH SUPERUSER")
    assert await _role_booleans(posture_dsn, _APP_ROLE) == (True, False), (
        "the mutation must move rolsuper alone, leaving rolbypassrls false"
    )

    # Act
    degraded = verifier_failure_tags(posture_dsn)

    # Assert
    await _assert_still_mutated(posture_dsn, _APP_ROLE, (True, False))
    assert _EXPECTED_POSTURE_LINES[_APP_SUPERUSER_TAG] in degraded, (
        f"verifier did not emit the exact superuser-app diagnostic: {sorted(degraded)}"
    )
    assert not any(_APP_BYPASSRLS_TAG in tag for tag in degraded), (
        "rolbypassrls is untouched here, so its check must stay green -- otherwise "
        f"neither boolean is independently pinned: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_bypassrls_app_role(posture_dsn: str) -> None:
    """BYPASSRLS on journal_app turns every tenant-isolation policy into a no-op."""
    # Arrange
    assert_clean_of(posture_dsn, _APP_BYPASSRLS_TAG)
    await _execute(posture_dsn, f"ALTER ROLE {_APP_ROLE} WITH BYPASSRLS")
    assert await _role_booleans(posture_dsn, _APP_ROLE) == (False, True), (
        "the mutation must move rolbypassrls alone, leaving rolsuper false"
    )

    # Act
    degraded = verifier_failure_tags(posture_dsn)

    # Assert
    await _assert_still_mutated(posture_dsn, _APP_ROLE, (False, True))
    assert _EXPECTED_POSTURE_LINES[_APP_BYPASSRLS_TAG] in degraded, (
        f"verifier did not emit the exact bypassrls-app diagnostic: {sorted(degraded)}"
    )
    assert not any(_APP_SUPERUSER_TAG in tag for tag in degraded), (
        "rolsuper is untouched here, so its check must stay green -- otherwise "
        f"neither boolean is independently pinned: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_superuser_admin_role(posture_dsn: str) -> None:
    """journal_admin needs BYPASSRLS, never SUPERUSER -- the latter voids every ACL check."""
    # Arrange
    assert_clean_of(posture_dsn, _ADMIN_SUPERUSER_TAG)
    await _execute(posture_dsn, f"ALTER ROLE {_ADMIN_ROLE} WITH SUPERUSER")
    assert await _role_booleans(posture_dsn, _ADMIN_ROLE) == (True, True), (
        "the mutation must move rolsuper alone, leaving rolbypassrls as the contract has it"
    )

    # Act
    degraded = verifier_failure_tags(posture_dsn)

    # Assert
    await _assert_still_mutated(posture_dsn, _ADMIN_ROLE, (True, True))
    assert _EXPECTED_POSTURE_LINES[_ADMIN_SUPERUSER_TAG] in degraded, (
        f"verifier did not emit the exact superuser-admin diagnostic: {sorted(degraded)}"
    )
    assert not any(_ADMIN_BYPASSRLS_TAG in tag for tag in degraded), (
        "the admin BYPASSRLS check is satisfied here, so it must stay green -- "
        f"otherwise the two admin booleans are not independently pinned: {sorted(degraded)}"
    )


async def test_invariant_script_flags_an_admin_role_without_bypassrls(posture_dsn: str) -> None:
    """This boolean fails when WEAKENED: without BYPASSRLS the maintenance paths stall.

    The other three posture booleans are broadening failures; this one is the
    inverse, so a verifier that only ever asserted ``NOT rolbypassrls`` for both
    roles would pass all the broadening tests and still be wrong here.
    """
    # Arrange
    assert_clean_of(posture_dsn, _ADMIN_BYPASSRLS_TAG)
    await _execute(posture_dsn, f"ALTER ROLE {_ADMIN_ROLE} WITH NOBYPASSRLS")
    assert await _role_booleans(posture_dsn, _ADMIN_ROLE) == (False, False), (
        "the mutation must move rolbypassrls alone, leaving rolsuper false"
    )

    # Act
    degraded = verifier_failure_tags(posture_dsn)

    # Assert
    await _assert_still_mutated(posture_dsn, _ADMIN_ROLE, (False, False))
    assert _EXPECTED_POSTURE_LINES[_ADMIN_BYPASSRLS_TAG] in degraded, (
        f"verifier did not emit the exact admin-bypassrls diagnostic: {sorted(degraded)}"
    )
    assert not any(_ADMIN_SUPERUSER_TAG in tag for tag in degraded), (
        "rolsuper is untouched here, so its check must stay green -- otherwise "
        f"neither admin boolean is independently pinned: {sorted(degraded)}"
    )


# ---------------------------------------------------------------------------
# REACHABLE posture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attribute", ["SUPERUSER", "BYPASSRLS"])
async def test_invariant_script_flags_a_set_only_reachable_defeating_role(
    posture_dsn: str, attribute: str
) -> None:
    """A SET-only edge grants no privileges by inheritance, yet one SET ROLE defeats RLS.

    ``pg_has_role(journal_app, probe, 'USAGE')`` is FALSE across a SET-only edge,
    so a USAGE-only reachability test reports clean here while journal_app can
    still assume the role at will. This is the case the shortcut misses.
    """
    # Arrange
    assert_clean_of(posture_dsn, _REACHABLE_TAG)
    await _create_probe_role(posture_dsn, _SET_PROBE_ROLE, attribute)
    await _execute(
        posture_dsn,
        f"GRANT {_SET_PROBE_ROLE} TO {_APP_ROLE} WITH INHERIT FALSE, SET TRUE",
    )
    assert not await _has_usage(posture_dsn, _APP_ROLE, _SET_PROBE_ROLE), (
        "a SET-only edge must NOT report USAGE -- otherwise this case is not "
        "exercising what a pg_has_role(USAGE) shortcut misses"
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert flagged, f"verifier did not flag the SET-only reachable role: {sorted(run.tags)}"
    assert all(_SET_PROBE_ROLE in tag and attribute in tag for tag in flagged), (
        f"the diagnostic must name the reachable role and its capability: {flagged}"
    )


@pytest.mark.parametrize("attribute", ["SUPERUSER", "BYPASSRLS"])
async def test_invariant_script_flags_an_admin_only_reachable_defeating_role(
    posture_dsn: str, attribute: str
) -> None:
    """ADMIN authority lets journal_app GRANT itself SET, so an ADMIN edge is assumable.

    Neither inheritance nor a SET option is present here: the only thing
    journal_app holds is the right to administer the membership, which is enough
    to give itself the SET option and then assume the role.
    """
    # Arrange
    assert_clean_of(posture_dsn, _REACHABLE_TAG)
    await _create_probe_role(posture_dsn, _ADMIN_PROBE_ROLE, attribute)
    await _execute(
        posture_dsn,
        f"GRANT {_ADMIN_PROBE_ROLE} TO {_APP_ROLE} WITH ADMIN TRUE, INHERIT FALSE, SET FALSE",
    )
    assert not await _has_usage(posture_dsn, _APP_ROLE, _ADMIN_PROBE_ROLE), (
        "an ADMIN-only edge must NOT report USAGE -- otherwise this case is not "
        "exercising what a pg_has_role(USAGE) shortcut misses"
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert flagged, f"verifier did not flag the ADMIN-only reachable role: {sorted(run.tags)}"
    assert all(_ADMIN_PROBE_ROLE in tag and attribute in tag for tag in flagged), (
        f"the diagnostic must name the reachable role and its capability: {flagged}"
    )


@pytest.mark.parametrize("attribute", ["SUPERUSER", "BYPASSRLS"])
async def test_invariant_script_flags_an_inherit_reachable_defeating_role(
    posture_dsn: str, attribute: str
) -> None:
    """The default GRANT carries INHERIT and SET, so the role is assumable and flagged.

    This is the edge a USAGE-only test DOES see. It is asserted anyway so the
    three edge kinds are pinned independently -- a regression that narrowed the
    walk to SET and ADMIN edges only would otherwise go unnoticed.
    """
    # Arrange
    assert_clean_of(posture_dsn, _REACHABLE_TAG)
    await _create_probe_role(posture_dsn, _INHERIT_PROBE_ROLE, attribute)
    await _execute(posture_dsn, f"GRANT {_INHERIT_PROBE_ROLE} TO {_APP_ROLE}")
    assert await _has_usage(posture_dsn, _APP_ROLE, _INHERIT_PROBE_ROLE), (
        "the default GRANT must report USAGE -- this case is the inherit arm"
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert flagged, f"verifier did not flag the inherited reachable role: {sorted(run.tags)}"
    assert all(_INHERIT_PROBE_ROLE in tag and attribute in tag for tag in flagged), (
        f"the diagnostic must name the reachable role and its capability: {flagged}"
    )


async def test_invariant_script_flags_a_privileged_role_reached_by_inherit_then_admin(
    posture_dsn: str,
) -> None:
    """Two edges: INHERIT-only to an intermediate, whose ADMIN authority reaches the target.

    This is the case the ``OR m.inherit_option`` arm of the recursive walk exists
    for, and the ONLY case that pins it. journal_app cannot assume the
    intermediate (inherit-only carries no attributes and no SET), but it INHERITS
    the intermediate's ADMIN authority over the target -- so it can grant itself
    SET on the target and assume THAT. Neither a SET-only nor an ADMIN-only
    single-edge case reaches through an inherit edge, so without this test the
    inherit arm could be deleted with the suite still green.

    The escalation is EXECUTED, not merely derived from the catalog: the
    self-GRANT and the SET ROLE run inside one transaction that is then rolled
    back, so the proof is a real privilege change rather than a claim about one.
    """
    # Arrange
    assert_clean_of(posture_dsn, _REACHABLE_TAG)
    await _create_probe_role(posture_dsn, _CHAIN_TARGET_ROLE, "BYPASSRLS")
    await _create_probe_role(posture_dsn, _CHAIN_MID_ROLE, "NOSUPERUSER NOBYPASSRLS")
    # mid holds ADMIN over target, but cannot itself be assumed by app.
    await _execute(
        posture_dsn,
        f"GRANT {_CHAIN_TARGET_ROLE} TO {_CHAIN_MID_ROLE} "
        "WITH ADMIN TRUE, INHERIT FALSE, SET FALSE",
    )
    await _execute(
        posture_dsn,
        f"GRANT {_CHAIN_MID_ROLE} TO {_APP_ROLE} WITH ADMIN FALSE, INHERIT TRUE, SET FALSE",
    )
    assert not await _has_usage(posture_dsn, _APP_ROLE, _CHAIN_TARGET_ROLE), (
        "the app role must not reach the target by plain USAGE -- otherwise this "
        "case collapses into the single-edge inherit arm"
    )
    await _assert_escalation_is_executable(posture_dsn)

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert flagged, (
        f"verifier did not flag the inherit-then-admin reachable target: {sorted(run.tags)}"
    )
    assert any(_CHAIN_TARGET_ROLE in tag and "BYPASSRLS" in tag for tag in flagged), (
        f"the diagnostic must name the reachable target and its capability: {flagged}"
    )


async def _assert_escalation_is_executable(dsn: str) -> None:
    """Execute the self-GRANT and SET ROLE, proving the chain is a real escalation.

    Run as journal_app's own privileges would allow, inside a transaction that is
    rolled back, so the cluster is left untouched. Without this the test would
    assert only that the verifier agrees with a catalog walk, not that the walk
    describes something an attacker can actually do.
    """
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(f"SET LOCAL ROLE {_APP_ROLE}")
            # Inherited ADMIN authority over the target is what makes this legal.
            await conn.execute(f"GRANT {_CHAIN_TARGET_ROLE} TO {_APP_ROLE} WITH SET TRUE")
            await conn.execute(f"SET LOCAL ROLE {_CHAIN_TARGET_ROLE}")
            reached = await conn.fetchval("SELECT current_user")
            bypasses = await conn.fetchval(
                "SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
        finally:
            await transaction.rollback()
    finally:
        await conn.close()
    assert str(reached) == _CHAIN_TARGET_ROLE, (
        f"the app role did not actually reach the target by SET ROLE, so the chain "
        f"is not an executable escalation: landed on {reached!r}"
    )
    assert bypasses is True, (
        "the reached role does not hold BYPASSRLS, so assuming it would not defeat "
        "row-level security and the finding would be spurious"
    )


async def test_invariant_script_accepts_a_reachable_role_with_no_defeating_attribute(
    posture_dsn: str,
) -> None:
    """Clean control: reachability alone is not the finding -- a defeating ATTRIBUTE is.

    Without this, every reachability test above would also pass on a check that
    flagged any assumable role whatsoever, which would fire on the deployment's
    own journal_admin-to-journal_app edge.
    """
    # Arrange
    await _create_probe_role(posture_dsn, _SET_PROBE_ROLE, "NOSUPERUSER NOBYPASSRLS")
    await _execute(
        posture_dsn,
        f"GRANT {_SET_PROBE_ROLE} TO {_APP_ROLE} WITH INHERIT FALSE, SET TRUE",
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    assert not _reachable_tags(run.tags), (
        f"an assumable role holding NEITHER attribute must not be flagged: {sorted(run.tags)}"
    )


async def test_invariant_script_ignores_an_inherit_only_edge_to_a_defeating_role(
    posture_dsn: str,
) -> None:
    """Measured on PostgreSQL 17: role ATTRIBUTES do not cross an inherit-only edge.

    With ``SET FALSE`` journal_app cannot assume the role, and SUPERUSER /
    BYPASSRLS apply only to the role that holds them -- so the privileges flow
    but the attributes do not, and there is nothing to flag. This pins the
    boundary of the walk: a check that flagged every INHERIT-reachable role
    regardless of assumability would fail here.
    """
    # Arrange
    await _create_probe_role(posture_dsn, _INHERIT_PROBE_ROLE, "BYPASSRLS")
    await _execute(
        posture_dsn,
        f"GRANT {_INHERIT_PROBE_ROLE} TO {_APP_ROLE} WITH ADMIN FALSE, INHERIT TRUE, SET FALSE",
    )
    assert await _has_usage(posture_dsn, _APP_ROLE, _INHERIT_PROBE_ROLE), (
        "the inherit-only edge must still report USAGE, or this case proves nothing"
    )
    _, app_bypasses = await _role_booleans(posture_dsn, _APP_ROLE)
    assert app_bypasses is False, (
        "journal_app must not itself gain BYPASSRLS across an inherit edge -- "
        "that premise is what makes the expected verdict 'clean'"
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    assert not _reachable_tags(run.tags), (
        "an inherit-only edge does not carry role attributes, so there is nothing "
        f"assumable to flag: {sorted(run.tags)}"
    )


# ---------------------------------------------------------------------------
# Bounded, single-line reachable diagnostics
# ---------------------------------------------------------------------------

# "<n> role(s) ... (showing <k>, omitted <m>)" -- the counts the diagnostic must
# always carry even when the sample is truncated.
_REACHABLE_COUNTS = re.compile(
    r"can assume (\d+) role\(s\) holding SUPERUSER or BYPASSRLS "
    r"\(showing (\d+), omitted (\d+)\)"
)


async def test_reachable_diagnostic_bounds_the_sample_and_keeps_exact_counts(
    posture_dsn: str,
) -> None:
    """Above the sample limit, the line truncates but the totals stay exact.

    A cluster can have hundreds of reachable roles. An unbounded diagnostic would
    emit all of them into a deploy log; a bounded one that dropped the count would
    hide the scale of the finding. So both are asserted: at most the limit is
    named, and total/shown/omitted are exact.
    """
    # Arrange
    over_limit = _REACHABLE_SAMPLE_LIMIT + 3
    names = [f"{_BULK_PROBE_PREFIX}{n:02d}" for n in range(over_limit)]
    for name in names:
        await _create_probe_role(posture_dsn, name, "BYPASSRLS")
        await _execute(posture_dsn, f"GRANT {name} TO {_APP_ROLE} WITH INHERIT FALSE, SET TRUE")

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert len(flagged) == 1, f"the finding must be exactly one line: {flagged}"
    match = _REACHABLE_COUNTS.search(flagged[0])
    assert match is not None, f"the diagnostic lost its count preamble: {flagged[0]!r}"
    total, shown, omitted = (int(g) for g in match.groups())
    assert total == over_limit, (
        f"the exact total must survive truncation: reported {total}, planted {over_limit}"
    )
    assert shown == _REACHABLE_SAMPLE_LIMIT, (
        f"the sample must be bounded at {_REACHABLE_SAMPLE_LIMIT}, showed {shown}"
    )
    assert omitted == over_limit - _REACHABLE_SAMPLE_LIMIT, (
        f"the omitted count must reconcile: total={total} shown={shown} omitted={omitted}"
    )
    named = [name for name in names if name in flagged[0]]
    assert len(named) == _REACHABLE_SAMPLE_LIMIT, (
        f"exactly {_REACHABLE_SAMPLE_LIMIT} names may appear, found {len(named)}: {named}"
    )
    assert named == sorted(names)[:_REACHABLE_SAMPLE_LIMIT], (
        f"the sample must be the ordered prefix, so it is reproducible: {named}"
    )


@pytest.mark.parametrize(
    "hostile_role",
    list(_EXPECTED_HOSTILE_RENDERING),
    ids=[
        "embedded_quote",
        "embedded_newline",
        "embedded_tab",
        "embedded_carriage_return",
        "literal_backslash_lookalike",
        "unicode_line_separator",
        "unicode_next_line",
        "unicode_c1_control",
    ],
)
async def test_reachable_diagnostic_neutralizes_hostile_role_names(
    posture_dsn: str, hostile_role: str
) -> None:
    """A hostile role name must render as its EXACT single-line encoding.

    Role names are operator-supplied text and the diagnostic lands in a log parsed
    line by line. A raw newline would split the finding across lines and could
    forge a second ``FAIL:`` entry.

    The assertion is EXACT, not merely "no newline survived": a placeholder that
    collapsed every control byte to one token would also pass a
    newline-absence check while destroying the reader's ability to tell a newline
    from a tab. So each name's expected rendering is spelled out, including the
    doubled backslash that keeps an encoded value unambiguous.
    """
    # Arrange
    await _create_probe_role(posture_dsn, hostile_role, "BYPASSRLS")
    await _execute(
        posture_dsn,
        f"GRANT {_quoted(hostile_role)} TO {_APP_ROLE} WITH INHERIT FALSE, SET TRUE",
    )

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    flagged = _reachable_tags(run.tags)
    assert len(flagged) == 1, f"the finding must remain exactly one line: {flagged}"
    assert "\n" not in flagged[0], (
        f"a newline in the diagnostic would split the finding across log lines: {flagged[0]!r}"
    )
    assert "\r" not in flagged[0], (
        f"a carriage return can overwrite a log line in a terminal: {flagged[0]!r}"
    )
    # UNICODE-AWARE: str.splitlines() breaks on U+0085, U+2028, U+2029 and the C1
    # range as well as \n and \r, so checking only the two ASCII characters would
    # pass a diagnostic that a log reader still splits in two.
    assert len(flagged[0].splitlines()) == 1, (
        "the diagnostic must be ONE line under Unicode-aware line splitting, not "
        f"merely free of \\n and \\r: {flagged[0]!r} -> {flagged[0].splitlines()!r}"
    )
    expected = _EXPECTED_HOSTILE_RENDERING[hostile_role]
    assert expected in flagged[0], (
        "the diagnostic must carry the role name's exact reversible encoding -- a "
        "lossy placeholder would hide which control byte was present:\n"
        f"want fragment={expected!r}\ngot line={flagged[0]!r}"
    )
    forged = [tag for tag in run.tags if tag.startswith("forged")]
    assert not forged, f"a role name must not be able to forge an additional FAIL line: {forged}"
    assert "FAIL: forged" not in run.output.replace(flagged[0], ""), (
        "the hostile name's payload must not appear outside its encoded diagnostic, "
        "or it has forged a log line elsewhere in the output"
    )
    # The finding still has to be actionable: enough of the name to identify it.
    assert "posture_probe" in flagged[0], (
        f"escaping must not erase the role identity: {flagged[0]!r}"
    )


# ---------------------------------------------------------------------------
# Required-role fail-hard, and the existence-gate continuation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", list(_REQUIRED_ROLES))
async def test_verifier_exits_nonzero_naming_a_missing_required_role(
    posture_dsn: str, missing: str
) -> None:
    """A missing required role is a named nonzero failure, never a silent pass.

    The role is not actually dropped -- a shared role cannot be recreated
    faithfully. Instead the verifier runs against a DISPOSABLE CLUSTER where the
    role was never created, which is the only safe way to observe true absence.
    """
    # Arrange / Act
    async with _cluster_missing(missing) as absent_dsn:
        run = verifier_run(absent_dsn, require_report=False)

    # Assert
    assert run.exit_code != 0, (
        f"a missing {missing} must exit nonzero: exit={run.exit_code}\n{run.output}"
    )
    assert reached_report_block(run.stdout, run.stderr), (
        f"the verifier must reach its report block, not abort: {run.output}"
    )
    assert f"{_ROLE_EXISTS_TAG} {missing}" in "\n".join(run.tags), (
        f"the failure must NAME the missing role: {sorted(run.tags)}"
    )


async def test_missing_required_role_still_reaches_posture_and_aggregation(
    posture_dsn: str,
) -> None:
    """Section 7 must not abort on a missing app role: posture and report still run.

    ``has_table_privilege`` raises on a missing role, and under ``set -e`` that
    killed the script mid-section -- so a deployment missing the app role reported
    an EMPTY failure list with every later invariant unchecked. Continuation is
    proven positively: the labeled unverified failures appear AND the admin
    posture checks, which execute after section 7, still produce their verdicts.
    """
    # Arrange / Act
    async with _cluster_missing(_APP_ROLE) as absent_dsn:
        run = verifier_run(absent_dsn, require_report=False)

    # Assert
    assert reached_report_block(run.stdout, run.stderr), (
        f"the verifier must reach its report block: {run.output}"
    )
    assert any("unverified" in tag for tag in run.tags), (
        f"checks skipped for the missing role must be labeled unverified: {sorted(run.tags)}"
    )
    assert any(_APP_ROLE in tag and "unverified" in tag for tag in run.tags), (
        f"the unverified labels must name the absent role: {sorted(run.tags)}"
    )
    # Sections 8b-ii and 9 both run AFTER section 7. Their verdicts appearing is
    # what distinguishes "continued" from "aborted politely" -- a clean section
    # emits nothing, so the marker has to be a check that reports on this
    # partially-migrated database either way.
    assert any("alembic_version" in tag for tag in run.tags), (
        "section 9 runs after section 7; its verdict proves execution continued "
        f"past the missing role rather than aborting: {sorted(run.tags)}"
    )
    assert any(f"reachable_posture {_APP_ROLE}" in tag for tag in run.tags), (
        "the reachable-posture check also runs after section 7 and must report "
        f"itself unverified rather than being skipped silently: {sorted(run.tags)}"
    )
    # An absent journal_app must not silently erase the journal_admin grant checks.
    # When the canonical table list lived INSIDE the journal_app arm, the
    # journal_admin loop iterated an EMPTY array -- so every admin grant went
    # unchecked while the report showed no failure for them, which reads exactly
    # like "all admin grants are correct". The admin verdicts are therefore
    # required to be PRESENT, per table, and named.
    admin_verdicts = {
        table
        for table in (*_NON_AUDIT_TABLES, "users")
        if any(f"grant {table}: {_ADMIN_ROLE}" in tag for tag in run.tags)
    }
    assert admin_verdicts == {*_NON_AUDIT_TABLES, "users"}, (
        "every journal_admin table grant must still produce a verdict when "
        "journal_app is absent -- an empty loop would look identical to a clean "
        f"result:\nmissing={sorted({*_NON_AUDIT_TABLES, 'users'} - admin_verdicts)}"
    )


async def test_a_missing_otel_ro_records_a_labeled_failure_and_continues(
    posture_dsn: str,
) -> None:
    """Without otel_ro the verifier must reach its report, not abort mid-script.

    Exercised on a disposable cluster where otel_ro was never created, rather than
    by dropping the shared role: dropping one is unrecoverable, since the harness
    cannot restore a password or rolconfig it never captured.
    """
    # Arrange / Act
    async with _cluster_missing(_OTEL_ROLE) as absent_dsn:
        run = verifier_run(absent_dsn, require_report=False)

    # Assert
    assert reached_report_block(run.stdout, run.stderr), (
        f"verifier did not reach its report block: {run.output}"
    )
    assert any(_OTEL_EXISTS_TAG in tag for tag in run.tags), (
        f"the missing otel_ro must be reported under its own label: {sorted(run.tags)}"
    )
    assert any(_OTEL_ROLE in tag and "unverified" in tag for tag in run.tags), (
        f"otel_ro-dependent checks must be reported as unverified: {sorted(run.tags)}"
    )
    # Continuation must be proven by a check that runs AFTER the otel_ro block and
    # emits a verdict regardless of posture. A clean posture section emits nothing,
    # which is indistinguishable from never running -- so the section-9 grants,
    # which are last and always report on this partially-migrated database, are the
    # marker.
    assert any("alembic_version" in tag for tag in run.tags), (
        "section 9 runs after the otel_ro block; its verdict is what distinguishes "
        f"'continued' from 'aborted politely': {sorted(run.tags)}"
    )
    assert not any(_ADMIN_BYPASSRLS_TAG in tag for tag in run.tags), (
        "journal_admin is correctly configured on this cluster, so its posture "
        "check must stay green -- a failure here would mean the continuation "
        f"marker above is measuring a broken posture instead: {sorted(run.tags)}"
    )


# ---------------------------------------------------------------------------
# Cross-database contention
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _cluster_missing(
    missing_role: str, *, lock_conn: asyncpg.Connection | None = None
) -> AsyncIterator[str]:
    """A migrated database on the disposable cluster where ``missing_role`` is absent.

    Roles are cluster-global, so genuine absence cannot be staged on the working
    cluster: dropping a shared role there is unrecoverable, because the harness
    captures four attributes and not the password, rolconfig or connection limit it
    would need to put one back. So this needs a SEPARATE cluster, supplied out of
    band via ``TEST_DISPOSABLE_CLUSTER_URL``.

    THE DISPOSABLE CLUSTER'S ROLE STATE IS ALSO RESTORED. Being throwaway makes
    dropping a role SAFE, not unnecessary: two local suites can share one such
    cluster, and a case that leaves journal_app dropped makes the next case's
    "the role exists" premise false. So this snapshots roles and memberships BEFORE
    the reset and restores them in ``finally``, dropping only roles absent from that
    snapshot -- the same discipline the working cluster gets, applied to a cluster
    where the drops are permitted.

    Every role mutation runs under ``cluster_role_lock(cluster)``, held across the
    snapshot, the reset, the body and the restore, so concurrent local suites
    serialize on this cluster exactly as they do on the working one.

    ``lock_conn`` lets a caller that ALREADY holds that key pass its connection in.
    Reacquiring on a second connection would self-deadlock -- pg_advisory_lock is
    re-entrant only within one session -- and it is also what a meta-test needs in
    order to read its baseline inside the same hold the lane runs under.
    """
    cluster = os.environ.get(DISPOSABLE_CLUSTER_ENV)
    if not cluster:
        pytest.skip(
            f"{DISPOSABLE_CLUSTER_ENV} is not set. Genuine role absence needs a "
            "separate cluster: roles are cluster-global, and dropping a shared "
            "role on the working cluster is unrecoverable because the harness "
            "does not capture the password, rolconfig or connection limit needed "
            "to restore one. Point this at a throwaway PostgreSQL 17 superuser "
            "DSN to gate these cases."
        )
    suffix = re.sub(r"[^a-z0-9]+", "_", missing_role.lower())
    name = f"{_ABSENT_DATABASE_PREFIX}{suffix}_{uuid.uuid4().hex[:8]}"
    target = urlunparse(urlparse(cluster)._replace(path=f"/{name}"))
    tracked = (*_REQUIRED_ROLES, _OTEL_ROLE)

    if lock_conn is not None:
        async with _cluster_missing_under_lock(
            lock_conn, cluster, missing_role, name, target, tracked
        ) as dsn:
            yield dsn
        return
    async with (
        cluster_role_lock(cluster) as own_conn,
        _cluster_missing_under_lock(own_conn, cluster, missing_role, name, target, tracked) as dsn,
    ):
        yield dsn


@asynccontextmanager
async def _cluster_missing_under_lock(
    lock_conn: asyncpg.Connection,
    cluster: str,
    missing_role: str,
    name: str,
    target: str,
    tracked: tuple[str, ...],
) -> AsyncIterator[str]:
    """The lane's lifecycle, on a connection that already holds the cluster lock."""
    # PRECONDITION, checked before anything is mutated. This lane creates and
    # drops the tracked roles, so it must start from a cluster where none of them
    # pre-exists. Classifying a pre-existing role as "absent from my baseline,
    # therefore mine to drop" would PERMANENTLY destroy a role the lane did not
    # create -- including a credential and memberships it never captured -- and it
    # would do so silently. CI provisions this service fresh, so the pristine
    # state is the norm and a violation means the DSN points somewhere it must
    # not. Failing here, before the reset, is what keeps that recoverable.
    await _assert_tracked_roles_absent(lock_conn, tracked)
    await _assert_cluster_is_disposable(cluster)
    # Captured AFTER the precondition proved every tracked role absent, so this snapshot
    # is known-empty rather than merely observed: nothing in it can be a role a
    # concurrent session owns.
    before = await capture_cluster_state(lock_conn, tracked)
    created_roles: frozenset[str] = frozenset()
    try:
        created_roles = await _reset_disposable_roles(lock_conn, missing_role)
        await lock_conn.execute(f'CREATE DATABASE "{name}"')
        try:
            await _prepare_absent_role_database(target, missing_role)
            yield target
        finally:
            await _drop_disposable_database(cluster, name)
    finally:
        # Disposable is exactly the set whose own CREATE succeeded above -- not "absent
        # from the snapshot", which is the same presence inference by another name: a role
        # a concurrent session creates after the snapshot reads as absent-then-present and
        # would be destroyed.
        await restore_cluster_state(
            lock_conn, before, tracked, disposable=tuple(sorted(created_roles))
        )


async def _assert_tracked_roles_absent(conn: asyncpg.Connection, tracked: tuple[str, ...]) -> None:
    """Fail fast when the disposable cluster already carries a tracked role.

    The message names a BOUNDED list of the offending roles -- enough to act on,
    without enumerating a catalog into a CI log. It is deliberately a hard failure
    rather than a skip: a populated cluster here means the configured DSN is not the
    throwaway one, and continuing would drop roles somewhere it must not.
    """
    present = sorted(
        str(row["rolname"])
        for row in await conn.fetch(
            "SELECT rolname FROM pg_roles WHERE rolname = ANY($1::text[]) ORDER BY rolname",
            list(tracked),
        )
    )
    if present:
        shown = present[:_UNEXPECTED_ROLE_SAMPLE]
        omitted = len(present) - len(shown)
        pytest.fail(
            f"{DISPOSABLE_CLUSTER_ENV} points at a cluster that already has "
            f"{len(present)} of the role(s) this lane creates and drops "
            f"(showing {len(shown)}, omitted {omitted}): {shown}. This lane would "
            "otherwise classify them as its own and DROP them permanently, losing "
            "credentials and memberships it never captured. Point it at a pristine "
            "throwaway cluster."
        )


async def _prepare_absent_role_database(target: str, missing_role: str) -> None:
    """Install pgvector and migrate ``target``, tolerating the omitted role's effect."""
    setup = await asyncpg.connect(target, timeout=5)
    try:
        await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await setup.close()

    upgrade = run_argv(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        JOURNAL_DB_MIGRATION_URL=target,
        JOURNAL_OPERATOR_EMAIL="operator@test.local",
    )
    if upgrade.returncode != 0 and missing_role not in _REQUIRED_ROLES:
        pytest.fail(
            "alembic upgrade head failed on the disposable cluster for a reason "
            f"unrelated to the omitted role:\n{upgrade.stdout}\n{upgrade.stderr}"
        )
    # A required role being absent makes the baseline fail on a GRANT. That is
    # expected here: the verifier's behaviour on the resulting partially migrated
    # database is exactly what these cases measure.


async def _assert_cluster_is_disposable(cluster_dsn: str) -> None:
    """Refuse to drop roles unless the cluster is demonstrably a throwaway.

    This lane drops shared roles, which is catastrophic on a real cluster. So the
    target must LOOK disposable: it may carry only the template databases, the
    maintenance database, and databases this lane itself creates. Anything else --
    a ``journal_rls_test``, a production database -- means the DSN points somewhere
    it must not, and the drop is refused.

    The failure names a BOUNDED list of the unexpected databases rather than the
    server's raw error detail, so a misconfiguration is actionable without dumping
    catalog contents into the log.
    """
    admin = await asyncpg.connect(maintenance_dsn(cluster_dsn), timeout=5)
    try:
        names = [
            str(row["datname"])
            for row in await admin.fetch(
                "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname"
            )
        ]
    finally:
        await admin.close()
    unexpected = [
        name
        for name in names
        if name != MAINTENANCE_DATABASE and not name.startswith(_ABSENT_DATABASE_PREFIX)
    ]
    if unexpected:
        shown = unexpected[:_UNEXPECTED_DATABASE_SAMPLE]
        omitted = len(unexpected) - len(shown)
        pytest.fail(
            f"{DISPOSABLE_CLUSTER_ENV} points at a cluster carrying "
            f"{len(unexpected)} database(s) this lane did not create "
            f"(showing {len(shown)}, omitted {omitted}): {shown}. This lane DROPS "
            "shared roles, so it refuses to run anywhere that is not demonstrably "
            "a throwaway cluster."
        )


async def _reset_disposable_roles(
    lock_conn: asyncpg.Connection, missing_role: str
) -> frozenset[str]:
    """Create every required role EXCEPT ``missing_role``, returning the ones made here.

    Runs on the RETAINED lock connection, so every create is serialized against the same
    advisory hold the caller took, and each role's ownership is the success of its own
    exact ``CREATE ROLE`` -- returned so the caller drops nothing else.

    ``missing_role`` is never dropped to manufacture its absence. The lane's precondition
    has already established that no tracked role exists on this cluster, so the role is
    absent because nothing created it. A ``DROP`` here would have been the destructive
    shortcut: on a cluster that was NOT pristine it removes a role, a credential and
    memberships this lane never captured, and the presence check that used to guard it
    cannot tell that role from one the lane made itself.
    """
    created: set[str] = set()
    for role in (*_REQUIRED_ROLES, _OTEL_ROLE):
        if role == missing_role:
            continue
        await _create_owned_role_or_refuse(lock_conn, f"CREATE ROLE {role} LOGIN", role)
        created.add(role)

    if missing_role != _ADMIN_ROLE:
        await lock_conn.execute(f"ALTER ROLE {_ADMIN_ROLE} WITH BYPASSRLS NOCREATEROLE")
    # Every GRANT below names a role that must still exist, so they are skipped when that
    # role is the one being left absent.
    if missing_role not in (_APP_ROLE, _ADMIN_ROLE):
        await lock_conn.execute(f"GRANT {_APP_ROLE} TO {_ADMIN_ROLE} WITH ADMIN OPTION")
    if missing_role != _OTEL_ROLE:
        await lock_conn.execute(f"GRANT pg_monitor TO {_OTEL_ROLE}")
    return frozenset(created)


async def _drop_disposable_database(cluster_dsn: str, name: str) -> None:
    """Drop a per-case database, first clearing the grants that pin its roles.

    Only genuinely-absent objects are tolerated, and each is checked FIRST rather
    than discovered by catching an exception: a role that does not exist is skipped,
    a database that does not exist returns early. Every other failure -- an
    unreachable server, a permission denial, a dependency this cleanup did not
    anticipate -- propagates with the database and role named, because swallowing
    those turns a broken disposable lane into silent leftovers that make the next
    case's "role is absent" premise false.
    """
    admin = await asyncpg.connect(maintenance_dsn(cluster_dsn), timeout=5)
    try:
        exists = bool(
            await admin.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name
            )
        )
    finally:
        await admin.close()
    if not exists:
        return

    target = urlunparse(urlparse(cluster_dsn)._replace(path=f"/{name}"))
    inner = await asyncpg.connect(target, timeout=5)
    try:
        present = {
            str(row["rolname"])
            for row in await inner.fetch(
                "SELECT rolname FROM pg_roles WHERE rolname = ANY($1::text[])",
                [*_REQUIRED_ROLES, _OTEL_ROLE],
            )
        }
        for role in (*_REQUIRED_ROLES, _OTEL_ROLE):
            if role not in present:
                continue
            await inner.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            await inner.execute(f"DROP OWNED BY {role} CASCADE")
    except asyncpg.PostgresError as exc:
        # ``from None`` suppresses the chain deliberately. A PostgresError carries the
        # server's DETAIL, which can quote catalog contents and identifiers; letting
        # it chain would print all of it in the CI failure this assertion renders.
        # The exception TYPE is kept, which is what makes the failure diagnosable.
        raise AssertionError(
            f"failed to clear role-owned objects in disposable database {name!r} "
            f"before dropping it: {type(exc).__name__}. Leaving this unresolved "
            "would leave the database in place and make the next case's "
            "role-absence premise false."
        ) from None
    finally:
        await inner.close()

    admin = await asyncpg.connect(maintenance_dsn(cluster_dsn), timeout=5)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_the_posture_lock_blocks_a_session_working_in_another_database() -> None:
    """The production key must serialize across DATABASES, not just within one.

    This is the finding the maintenance-database lock exists for: advisory locks
    are database-scoped while roles are cluster-global, so a lock taken in the
    working database leaves a session working in a scratch database entirely
    unserialized -- and both mutate the same ``pg_authid`` rows.

    Proven end to end with the real helper and the real key: while
    ``_posture_lock()`` holds it, a connection to a DIFFERENT database cannot take
    it, and contention surfaces as the bounded, named failure rather than a hang.
    Acquisition succeeds only after the full snapshot/mutation/restore lifecycle
    has exited.
    """
    # Arrange
    other_working_database = _another_database_in_the_same_cluster()
    assert other_working_database != RLS_BOOTSTRAP_URL, (
        "the two sessions must work in genuinely different databases, or this test "
        "measures same-database contention and proves nothing about the fix"
    )
    naive_blocked_before = await _naive_same_database_lock_is_contended()

    # Act. The lock-holding connection is taken explicitly so its backend pid is
    # known: release is then verified by that pid's absence from the lock table,
    # not by re-acquiring -- which a concurrent suite could legitimately prevent.
    async with _posture_lock() as lock_conn:
        holder_pid = int(await lock_conn.fetchval("SELECT pg_backend_pid()"))
        async with _posture_session(lock_conn) as dsn:
            await _execute(dsn, f"ALTER ROLE {_APP_ROLE} WITH SUPERUSER")
            acquired_during = await _try_posture_lock_from(other_working_database)
            naive_blocked_during = await _naive_same_database_lock_is_contended()
            # A short explicit timeout: this probe only needs to observe that the
            # acquisition is REFUSED and names the holder. Waiting the production bound
            # here would block for ten minutes on a result already known at two seconds.
            with pytest.raises(PostureLockUnavailableError, match="backend pid="):
                async with cluster_role_lock(
                    other_working_database, POSTURE_ADVISORY_LOCK_KEY, timeout_ms=2_000
                ):
                    pass
    released_after = await _posture_lock_is_free_of(holder_pid)

    # Assert
    assert acquired_during is False, (
        "a session whose working database differs took the posture lock while this "
        "one held it, so the lock serializes nothing cluster-wide -- exactly the "
        "defect routing it to the maintenance database closes"
    )
    assert released_after is True, (
        "this session still held the posture lock after its lifecycle exited, so "
        "the 'blocked' result above could be explained by something other than the "
        "hold. Release is checked by PID rather than by re-acquiring: a concurrent "
        "suite legitimately taking the lock in between would make a re-acquire "
        "attempt fail for a reason unrelated to this session"
    )
    # The control that makes the result meaningful rather than tautological: a lock
    # taken on the WORKING database -- the naive implementation -- is NOT contended
    # even while the posture lock is held. Same key, same cluster, same
    # cluster-global rows at risk; different database, so no serialization.
    assert naive_blocked_during is False, (
        "a working-database lock appeared contended while the maintenance-database "
        "lock was held, which would mean this test cannot distinguish the two "
        "designs and its primary assertion is not evidence"
    )
    assert naive_blocked_before is False, (
        "the working-database lock was already contended before the test started, "
        "so its 'uncontended' reading during the hold is not attributable"
    )


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_posture_lock_contention_fails_fast_instead_of_hanging() -> None:
    """Contention raises a bounded, named error well inside the lock timeout.

    An unbounded ``pg_advisory_lock`` would hang the suite on a stuck holder. The
    timeout turns that into a failure, and the message identifies the blocking
    backend so an operator can act on it.
    """
    # Arrange. A SHORT explicit timeout is used, not the production default: the
    # behaviour under test is "bounded and named rather than hanging", and waiting
    # the production bound to observe it would add ten minutes per run.
    other_database = _another_database_in_the_same_cluster()
    probe_timeout_ms = 2_000

    # Act / Assert
    async with _posture_lock():
        with pytest.raises(PostureLockUnavailableError) as caught:
            async with cluster_role_lock(
                other_database, POSTURE_ADVISORY_LOCK_KEY, timeout_ms=probe_timeout_ms
            ):
                pass

    message = str(caught.value)
    assert str(probe_timeout_ms) in message, (
        f"the failure must state the bound it actually waited: {message}"
    )
    assert probe_timeout_ms < LOCK_TIMEOUT_MS, (
        "the production default must be the generous one; a default as short as "
        "this probe would fail legitimate concurrent suites"
    )
    assert "backend pid=" in message, f"the failure must name the blocking backend: {message}"
    assert "database=" in message, (
        f"the failure must name the blocking session's database: {message}"
    )
    assert "testpass" not in message, (
        f"the failure must not echo a credential while naming the holder: {message}"
    )


async def _posture_lock_is_free_of(pid: int) -> bool:
    """Whether NO advisory lock on the posture key is held by this process's session.

    Checked by pid, not by re-acquiring: under concurrency a sibling suite holds
    the lock legitimately, and a failed re-acquire would then say nothing about
    whether THIS session released.
    """
    conn = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    try:
        holders = [
            int(row["pid"])
            for row in await conn.fetch(
                """
                SELECT l.pid
                FROM pg_locks l
                WHERE l.locktype = 'advisory'
                  AND l.granted
                  AND ((l.classid::bigint << 32) | (l.objid::bigint & 4294967295)) = $1
                """,
                POSTURE_ADVISORY_LOCK_KEY,
            )
        ]
    finally:
        await conn.close()
    return pid not in holders


async def _naive_same_database_lock_is_contended() -> bool:
    """Whether the posture key is already taken IN THE WORKING DATABASE.

    This models the naive implementation the fix replaced: locking the database
    the test happens to be working in. Advisory locks are database-scoped, so this
    reads "uncontended" even while a sibling session holds the same key in the
    maintenance database -- which is precisely why it could not protect
    cluster-global role state.
    """
    conn = await asyncpg.connect(RLS_BOOTSTRAP_URL, timeout=5)
    try:
        acquired = bool(
            await conn.fetchval("SELECT pg_try_advisory_lock($1)", POSTURE_ADVISORY_LOCK_KEY)
        )
        if acquired:
            await conn.execute("SELECT pg_advisory_unlock($1)", POSTURE_ADVISORY_LOCK_KEY)
        return not acquired
    finally:
        await conn.close()


def _another_database_in_the_same_cluster() -> str:
    """A working DSN for a DIFFERENT database in the same cluster.

    The maintenance database is used as the second working database: it always
    exists, and it is a genuinely different database from the RLS test database
    the first session works in. That difference is the whole point -- a
    database-scoped lock taken in one would not contend with a lock taken in the
    other, which is the defect the maintenance-database lock closes.
    """
    return maintenance_dsn(RLS_BOOTSTRAP_URL)


async def _try_posture_lock_from(dsn: str) -> bool:
    """Whether the production posture key can be taken, right now, from ``dsn``."""
    conn = await asyncpg.connect(maintenance_dsn(dsn), timeout=5)
    try:
        acquired = bool(
            await conn.fetchval("SELECT pg_try_advisory_lock($1)", POSTURE_ADVISORY_LOCK_KEY)
        )
        if acquired:
            await conn.execute("SELECT pg_advisory_unlock($1)", POSTURE_ADVISORY_LOCK_KEY)
        return acquired
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Teardown exactness
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_teardown_restores_mutated_state_and_touches_nothing_else() -> None:
    """Teardown restores exactly what it mutated and leaves everything else alone.

    Two halves, both necessary. The mutated attributes and membership options must
    come back after a raising body -- that is the leak-prevention half. And the
    state the harness never writes must be BIT-IDENTICAL: password hash,
    ``rolconfig``, connection limit, validity, and every membership grantor. A
    restore that recreated a role would pass the first half and fail the second,
    which is why recreation is forbidden rather than merely discouraged.
    """
    async with _posture_lock() as lock_conn:
        # Arrange
        before = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
        untouched_before = await _untouched_role_facts(lock_conn)

        # Act
        mutated = await _mutate_then_fail_inside_a_posture_session(lock_conn)

        # Assert
        assert mutated == (True, True), (
            "the induced mutation did not take, so restore proves nothing"
        )
        after = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
        untouched_after = await _untouched_role_facts(lock_conn)

    assert after.attributes == before.attributes, (
        f"teardown left mutated attributes changed:\n"
        f"before={before.attributes}\nafter={after.attributes}"
    )
    assert after.memberships == before.memberships, (
        f"teardown left memberships changed:\n"
        f"added={sorted(after.memberships - before.memberships)}\n"
        f"removed={sorted(before.memberships - after.memberships)}"
    )
    assert untouched_after == untouched_before, (
        "teardown modified state the harness never mutates -- a credential, "
        "rolconfig, connection limit, validity window, or membership grantor:\n"
        f"before={untouched_before}\nafter={untouched_after}"
    )
    assert set(_REQUIRED_ROLES) <= set(after.attributes), (
        "teardown must never drop a role the deployment contract requires"
    )


async def _untouched_role_facts(conn: asyncpg.Connection) -> dict[str, object]:
    """Role state the harness must never write, for a bit-identical comparison.

    Deliberately includes the password HASH: it is the single most destructive
    thing a recreate-on-restore would silently discard, and comparing it is how
    this suite proves no such recreation happens. It is compared, never logged --
    the assertion prints the whole mapping only on failure, which by construction
    means the hashes already match the pre-test value.
    """
    rows = await conn.fetch(
        """
        SELECT r.rolname,
               a.rolpassword IS NOT NULL AS has_password,
               md5(coalesce(a.rolpassword, '')) AS password_fingerprint,
               r.rolconfig,
               r.rolconnlimit,
               r.rolvaliduntil,
               r.rolinherit,
               r.rolcreatedb,
               r.rolreplication
        FROM pg_roles r
        LEFT JOIN pg_authid a ON a.oid = r.oid
        WHERE r.rolname = ANY($1::text[])
        ORDER BY r.rolname
        """,
        list(_TRACKED_ROLES),
    )
    grantors = await conn.fetch(
        """
        SELECT g.rolname AS granted, m.rolname AS member, gr.rolname AS grantor
        FROM pg_auth_members am
        JOIN pg_roles g ON g.oid = am.roleid
        JOIN pg_roles m ON m.oid = am.member
        LEFT JOIN pg_roles gr ON gr.oid = am.grantor
        WHERE g.rolname = ANY($1::text[]) OR m.rolname = ANY($1::text[])
        ORDER BY 1, 2
        """,
        list(_TRACKED_ROLES),
    )
    return {
        "roles": [dict(row) for row in rows],
        "grantors": [dict(row) for row in grantors],
    }


async def _mutate_then_fail_inside_a_posture_session(
    lock_conn: asyncpg.Connection,
) -> tuple[bool, bool]:
    """Mutate cluster state inside a posture session, then raise out of it.

    Returns the booleans as they stood at the moment of the raise, so the caller
    can confirm the mutation actually took -- a restore that "worked" because
    nothing had changed would prove nothing.
    """
    mutated: tuple[bool, bool] | None = None
    try:
        async with _posture_session(lock_conn) as dsn:
            await _execute(dsn, f"ALTER ROLE {_APP_ROLE} WITH SUPERUSER BYPASSRLS")
            await _execute(dsn, f"ALTER ROLE {_ADMIN_ROLE} WITH CREATEROLE")
            await _create_probe_role(dsn, _SET_PROBE_ROLE, "BYPASSRLS")
            await _execute(dsn, f"GRANT {_SET_PROBE_ROLE} TO {_APP_ROLE}")
            await _execute(dsn, f"REVOKE {_APP_ROLE} FROM {_ADMIN_ROLE}")
            mutated = await _role_booleans(dsn, _APP_ROLE)
            raise RuntimeError("induced")
    except RuntimeError as exc:
        if str(exc) != "induced":
            raise
    assert mutated is not None, "the session body did not run"
    return mutated


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_teardown_drops_only_roles_the_fixture_created() -> None:
    """A disposable role the fixture made is dropped; a shared role is not.

    Cleanup of created roles is exercised with a fixture-OWNED role, never by
    dropping a shared one. The shared roles are asserted to survive in the same
    pass, so "cleans up after itself" and "does not destroy shared state" are one
    measurement rather than two hopes.
    """
    async with _posture_lock() as lock_conn:
        # Arrange
        before = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
        assert _SET_PROBE_ROLE not in before.present, (
            "the disposable probe role must not pre-exist, or its cleanup is unproven"
        )

        # Act
        async with _posture_session(lock_conn) as dsn:
            await _create_probe_role(dsn, _SET_PROBE_ROLE, "BYPASSRLS")
            created = await _role_exists(dsn, _SET_PROBE_ROLE)

        # Assert
        after = await capture_cluster_state(lock_conn, _TRACKED_ROLES)

    assert created is True, "the fixture-owned role was not created, so cleanup proves nothing"
    assert _SET_PROBE_ROLE not in after.present, (
        "a role the fixture created must be dropped on teardown"
    )
    assert set(_REQUIRED_ROLES) <= after.present, (
        "teardown must never drop a shared role it did not create"
    )
    assert after.present == before.present, (
        f"teardown changed which roles exist:\nbefore={sorted(before.present)}\n"
        f"after={sorted(after.present)}"
    )


# ---------------------------------------------------------------------------
# Diagnostics hygiene
# ---------------------------------------------------------------------------

# A posture tag: <role>_<label words> followed by the assert_true verdict.
_POSTURE_TAG_SHAPE = re.compile(
    r"^(journal_app|journal_admin)_[a-z_]+: expected true, got '[tf]?'$"
)


async def test_role_posture_diagnostics_are_exact_and_leak_free(posture_dsn: str) -> None:
    """One verifier run backs the report-reached, exact-tag and leak assertions.

    All three views come from a SINGLE execution. Running the verifier once per
    assertion would let them describe different runs, so a leak found in one could
    coexist with a clean verdict from another and neither would be wrong.

    All four booleans are broken at once, so the assertion sees the complete set of
    posture diagnostics rather than one arm of it, and each line is compared to its
    exact expected text.
    """
    # Arrange
    psql_bin()
    await _execute(posture_dsn, f"ALTER ROLE {_APP_ROLE} WITH SUPERUSER BYPASSRLS")
    await _execute(posture_dsn, f"ALTER ROLE {_ADMIN_ROLE} WITH SUPERUSER NOBYPASSRLS")

    # Act
    run = verifier_run(posture_dsn)

    # Assert
    await _assert_still_mutated(posture_dsn, _APP_ROLE, (True, True))
    await _assert_still_mutated(posture_dsn, _ADMIN_ROLE, (True, False))
    assert reached_report_block(run.stdout, run.stderr), (
        f"verifier did not reach its report block: {run.output}"
    )
    posture_lines = {tag for tag in run.tags if any(t in tag for t in _POSTURE_TAGS)}
    assert posture_lines == set(_EXPECTED_POSTURE_LINES.values()), (
        "the four induced failures must each appear as its exact expected line:\n"
        f"missing={sorted(set(_EXPECTED_POSTURE_LINES.values()) - posture_lines)}\n"
        f"unexpected={sorted(posture_lines - set(_EXPECTED_POSTURE_LINES.values()))}"
    )
    malformed = [line for line in sorted(posture_lines) if not _POSTURE_TAG_SHAPE.match(line)]
    assert not malformed, (
        f"posture diagnostics may carry only a role name and a boolean label: {malformed}"
    )
    assert run.output.strip(), "empty verifier output would make the leak search vacuous"
    findings = _sensitive_findings(run.output, posture_dsn)
    assert not findings, (
        f"verifier output carried connection data or error detail: {sorted(findings)}"
    )


_LEAK_PROBE_DSN = "postgresql://someuser:somesecret@db.example:6543/somedb"


@pytest.mark.parametrize(
    ("planted", "expected_label"),
    [
        (f"psql: could not connect to {_LEAK_PROBE_DSN}", "dsn"),
        ("authentication failed for user with password somesecret", "password"),
        ("could not translate host name db.example to address", "host"),
        ('FATAL: database "somedb" does not exist', "database"),
        ("ERROR: broken\nDETAIL:  key (id)=(7) already exists", "sql_detail"),
    ],
    ids=["dsn", "password", "host", "database", "sql_detail"],
)
async def test_connection_data_detector_attributes_each_planted_leak_class(
    planted: str, expected_label: str
) -> None:
    """Positive control for the leak search, one independently attributed class at a time.

    "No connection data found" is free on output containing nothing the detector
    can match, so the same detector is run against text that DOES carry each
    class. Each case asserts its own label is the ONLY one reported, so a detector
    arm that had stopped working could not hide behind a sibling arm firing on the
    same text.
    """
    # Arrange / Act
    findings = _sensitive_findings(planted, _LEAK_PROBE_DSN)

    # Assert
    assert findings == {expected_label}, (
        f"expected exactly the {expected_label!r} class from this text, got {sorted(findings)}"
    )


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://journal:testpass@localhost:5433/journal_rls_test",
        "postgresql://journal:testpass@localhost:5432/journal",
        "postgresql://journal_app:pw@db:5432/journal_prod",
    ],
    ids=["ci_shaped", "database_named_journal", "app_role_as_username"],
)
async def test_connection_data_detector_ignores_legitimate_role_labels(dsn: str) -> None:
    """The role labels the verifier must print are not a leak, on deploy-shaped DSNs.

    ``journal`` is both a plausible database name and a prefix of the
    ``journal_app`` / ``journal_admin`` labels every posture diagnostic carries.
    A substring search for the database name therefore fires on correct output --
    so the database detector is WORD-BOUNDED, and these cases are what pin that.
    The third case is the nastiest: the DSN username IS the app role name.
    """
    # Arrange
    legitimate = (
        "FAIL: journal_app_not_superuser: expected true, got 'f'\n"
        "FAIL: journal_admin_bypassrls: expected true, got 'f'\n"
        "FAIL: role_exists journal_admin: expected true, got 'f'\n"
        "FAIL: reachable_posture journal_app: can assume 1 role(s) holding "
        "SUPERUSER or BYPASSRLS (showing 1, omitted 0): posture_probe_set holds BYPASSRLS\n"
        "verify-db-invariants: OK -- journal_app + journal_admin posture verified"
    )

    # Act
    findings = _sensitive_findings(legitimate, dsn)

    # Assert
    assert not findings, (
        f"legitimate role-labeled diagnostics were misreported as leaks: {sorted(findings)}"
    )


async def test_database_name_detector_still_catches_a_real_database_leak() -> None:
    """Word-bounding the database needle must not make it unable to fire.

    The boundary rule exists so ``journal`` does not match ``journal_app``. If it
    also stopped matching a genuine standalone mention, the detector arm would be
    inert -- so a real leak of the same name is asserted to be caught.
    """
    # Arrange
    dsn = "postgresql://journal:testpass@localhost:5432/journal"

    # Act
    caught = _sensitive_findings('FATAL: database "journal" does not exist', dsn)
    ignored = _sensitive_findings("FAIL: journal_app_not_superuser: expected true, got 'f'", dsn)

    # Assert
    assert caught == {"database"}, f"a standalone database mention must be caught: {caught}"
    assert not ignored, f"a role label sharing the prefix must not be caught: {ignored}"


def _mentions_standalone(output: str, needle: str) -> bool:
    r"""Whether ``needle`` appears in ``output`` as a whole word.

    ``[\w-]`` is the boundary class, so neither an identifier character nor a
    hyphen may sit next to a match. That is what separates a genuine leak of
    ``journal`` or ``db`` from the ``journal_app`` label and the
    ``verify-db-invariants`` banner the verifier is required to emit.
    """
    return re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", output) is not None


def _sensitive_findings(output: str, dsn: str) -> frozenset[str]:
    """Labels for every class of sensitive value present in ``output``.

    The DATABASE name is matched on WORD BOUNDARIES rather than as a bare
    substring: on this harness and in deploy-shaped DSNs it is ``journal`` or
    ``journal_rls_test``, both of which are prefixes or substrings of the
    ``journal_app`` / ``journal_admin`` labels the verifier is REQUIRED to print.
    An unbounded search would flag correct output.

    The DSN USERNAME is not a needle at all: it can BE a role name (a deploy that
    connects as ``journal_app``), so no boundary rule can separate a leak of it
    from a legitimate label.

    Password, host and the full DSN are matched directly -- none can collide with
    a role label.
    """
    parsed = urlparse(dsn)
    findings: set[str] = set()

    core = f"{parsed.netloc}{parsed.path}"
    if core and core in output:
        findings.add("dsn")
    if parsed.password and parsed.password in output:
        findings.add("password")
    # HOST and DATABASE are both matched on WORD BOUNDARIES. Neither can be a
    # bare substring search: a short host like ``db`` occurs inside the
    # verifier's own banner ("verify-db-invariants"), and the database name is a
    # prefix of the journal_app / journal_admin labels the verifier must print.
    if parsed.hostname and _mentions_standalone(output, parsed.hostname):
        findings.add("host")

    database = parsed.path.lstrip("/")
    if database and _mentions_standalone(output, database):
        findings.add("database")
    if "DETAIL:" in output:
        findings.add("sql_detail")

    # A full-DSN echo necessarily contains the host, password and database too;
    # reporting it as one finding keeps each case's attribution unambiguous.
    if "dsn" in findings:
        findings -= {"host", "password", "database"}
    return frozenset(findings)


# ---------------------------------------------------------------------------
# Credential fingerprints.
#
# A role password is a CLUSTER-GLOBAL secret stored as a hash the harness never
# captures, so overwriting one is unrecoverable: it invalidates every
# session-scoped pool in the run and, on a shared or deployed cluster, destroys a
# credential nothing here could put back. The harness therefore never writes one
# -- it reads the credential the test environment provisioned.
#
# "Never writes one" is a claim about absence, which is free unless something can
# detect the write. These controls fingerprint rolpassword for both required roles
# and require byte-identity across every path that touches cluster-global state.
# ---------------------------------------------------------------------------


async def _credential_fingerprints(conn: asyncpg.Connection) -> dict[str, str]:
    """An md5 fingerprint of each required role's stored password hash.

    The FINGERPRINT is compared, never the hash: a mismatch means the credential
    changed, which is all a test needs to know, and the digest cannot be used to
    authenticate. ``NULL`` (no password set) is distinguished from any hash, so
    clearing a credential is caught as readily as replacing one.
    """
    rows = await conn.fetch(
        """
        SELECT r.rolname,
               CASE WHEN a.rolpassword IS NULL THEN 'unset'
                    ELSE md5(a.rolpassword) END AS fingerprint
        FROM pg_roles r
        LEFT JOIN pg_authid a ON a.oid = r.oid
        WHERE r.rolname = ANY($1::text[])
        ORDER BY r.rolname
        """,
        list(_REQUIRED_ROLES),
    )
    return {str(row["rolname"]): str(row["fingerprint"]) for row in rows}


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_posture_fixture_never_rewrites_a_role_credential() -> None:
    """The posture lifecycle leaves both required roles' credentials byte-identical.

    Exercised across a mutation AND an exception, because the restore path runs in
    both cases and a credential rewrite hiding in either would be equally fatal.
    """
    async with _posture_lock() as lock_conn:
        # Arrange
        before = await _credential_fingerprints(lock_conn)
        assert set(before) == set(_REQUIRED_ROLES), (
            f"both required roles must have a fingerprint to compare: {sorted(before)}"
        )
        assert all(value != "unset" for value in before.values()), (
            "the required roles must have credentials provisioned, or an unchanged "
            f"'unset' reading would prove nothing: {before}"
        )

        # Act -- mutate role state and then raise out of the session.
        await _mutate_then_fail_inside_a_posture_session(lock_conn)

        # Assert
        after = await _credential_fingerprints(lock_conn)

    assert after == before, (
        "the posture fixture changed a role credential. Passwords are "
        "cluster-global and the harness does not capture the hash, so a rewrite is "
        f"unrecoverable:\nbefore={before}\nafter={after}"
    )


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_credentials_survive_the_scratch_fixture_lifecycle() -> None:
    """Bracket the whole scratch lifecycle: credentials in, credentials out.

    Driven directly rather than requested, so the fingerprints are taken OUTSIDE
    the fixture and can observe the before/after pair.
    """
    # Arrange
    probe = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    try:
        before = await _credential_fingerprints(probe)
    finally:
        await probe.close()
    assert all(value != "unset" for value in before.values()), (
        f"credentials must be provisioned before the exercise: {before}"
    )

    # Act
    await _drive_dedup_scratch_fixture_once()

    # Assert
    probe = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    try:
        after = await _credential_fingerprints(probe)
    finally:
        await probe.close()
    assert after == before, (
        "the scratch fixture changed a role credential across its lifecycle:\n"
        f"before={before}\nafter={after}"
    )


async def _drive_dedup_scratch_fixture_once() -> None:
    """Run the dedup module's scratch fixture setup/teardown with an empty body."""
    from tests.integration.test_audit_log_dedup_read_paths import scratch_dsn as dedup_scratch

    generator = dedup_scratch.__wrapped__()  # type: ignore[attr-defined]
    await anext(generator)
    with pytest.raises(StopAsyncIteration):
        await anext(generator)


async def test_credentials_survive_concurrent_posture_sessions() -> None:
    """Two overlapping posture sessions leave credentials byte-identical.

    Concurrency is the case where a credential rewrite would be most damaging and
    hardest to attribute: the second session's restore could write back a value the
    first had changed. The maintenance-database lock serializes them, and this pins
    the outcome rather than the mechanism.
    """
    # Arrange
    probe = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    try:
        before = await _credential_fingerprints(probe)
    finally:
        await probe.close()

    # Act -- two full lifecycles, launched together.
    async def one_session() -> None:
        async with _posture_session() as dsn:
            await _execute(dsn, f"ALTER ROLE {_APP_ROLE} WITH SUPERUSER")

    await asyncio.gather(one_session(), one_session())

    # Assert
    probe = await asyncpg.connect(maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    try:
        after = await _credential_fingerprints(probe)
    finally:
        await probe.close()
    assert after == before, (
        f"concurrent posture sessions changed a role credential:\nbefore={before}\nafter={after}"
    )


# ---------------------------------------------------------------------------
# Membership row identity includes the grantor.
#
# On PostgreSQL 16+ one (granted, member) pair can have SEVERAL pg_auth_members
# rows, one per grantor, each with its own option triple. A bare
# ``REVOKE granted FROM member`` removes only the row granted by the current user,
# and a bare GRANT creates a row attributed to the current user. So a restore that
# keyed on the pair alone would revoke one row and leave another, or add a
# duplicate -- which is why identity is the full triple and every statement names
# GRANTED BY.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_restore_preserves_exact_multi_grantor_membership_rows() -> None:
    """Two grantors, one pair: restore must reproduce both rows and add none.

    journal_admin holds ADMIN on journal_app, so it can grant the probe role
    membership that the bootstrap superuser also grants -- producing two rows for
    the same pair with different grantors. The test then perturbs memberships and
    requires the restore to reproduce the exact row set, grantors included.
    """
    async with _posture_lock() as lock_conn:
        # Arrange -- a probe role, granted to journal_app by TWO different grantors.
        #
        # Created with a BARE statement on the connection that holds the posture lock,
        # not `CREATE ROLE IF NOT EXISTS` through a second connection. The conditional
        # form silently adopts a role another session owns and the teardown below then
        # drops it; issuing the create directly here means the success of that one
        # statement is the ownership fact, and it is serialized against the same hold
        # every other statement in this test uses.
        owns_probe = await _create_owned_role(
            lock_conn, f"CREATE ROLE {_SET_PROBE_ROLE} LOGIN NOINHERIT"
        )
        assert owns_probe, (
            f"a concurrent session already owns {_SET_PROBE_ROLE}, so this test cannot "
            "own the probe role its teardown would drop"
        )
        await lock_conn.execute(f"GRANT {_SET_PROBE_ROLE} TO {_APP_ROLE} WITH ADMIN OPTION")
        await lock_conn.execute(f"GRANT {_SET_PROBE_ROLE} TO {_ADMIN_ROLE} WITH ADMIN TRUE")
        # journal_admin now re-grants the same pair, creating a SECOND row whose
        # grantor is journal_admin rather than the bootstrap user.
        await _grant_as_role(
            RLS_BOOTSTRAP_URL,
            _ADMIN_ROLE,
            f"GRANT {_SET_PROBE_ROLE} TO {_APP_ROLE} WITH INHERIT FALSE, SET TRUE",
        )
        baseline = await capture_cluster_state(lock_conn, (*_TRACKED_ROLES, _SET_PROBE_ROLE))
        pair_rows = [
            edge
            for edge in baseline.memberships
            if (edge.granted, edge.member) == (_SET_PROBE_ROLE, _APP_ROLE)
        ]
        assert len(pair_rows) == 2, (
            "the fixture must produce TWO rows for one pair, or this test is not "
            f"exercising multi-grantor identity at all: {sorted(pair_rows)}"
        )
        assert len({edge.grantor for edge in pair_rows}) == 2, (
            f"the two rows must have DIFFERENT grantors: {sorted(pair_rows)}"
        )

        # Act -- perturb BOTH rows, then restore to the captured baseline.
        #
        # Revoking only the connection user's row would leave the journal_admin-
        # granted row in place, so restore would never have to RECREATE a row whose
        # grantor differs from the connection user -- and a GRANT that silently
        # omitted GRANTED BY would still pass. Both rows are therefore removed, each
        # named by its own grantor, which forces restore to reconstruct the foreign
        # attribution rather than merely leave it alone.
        try:
            for edge in sorted(pair_rows):
                await lock_conn.execute(
                    f"REVOKE {_SET_PROBE_ROLE} FROM {_APP_ROLE} GRANTED BY {_quoted(edge.grantor)}"
                )
            emptied = await capture_cluster_state(lock_conn, (*_TRACKED_ROLES, _SET_PROBE_ROLE))
            assert not [
                edge
                for edge in emptied.memberships
                if (edge.granted, edge.member) == (_SET_PROBE_ROLE, _APP_ROLE)
            ], (
                "both rows must be gone before restore runs, or restore is not "
                "required to recreate the foreign-grantor row"
            )
            # Also plant an EXTRA row attributed to journal_admin, which restore
            # must REVOKE. Its grantor is not the connection user, so a revoke
            # without GRANTED BY silently removes nothing and the extra row
            # survives -- which is the revoke-side half of the contract.
            await _grant_as_role(
                RLS_BOOTSTRAP_URL,
                _ADMIN_ROLE,
                f"GRANT {_SET_PROBE_ROLE} TO {_ADMIN_ROLE} WITH INHERIT FALSE, SET TRUE",
            )
            planted_extra = await capture_cluster_state(
                lock_conn, (*_TRACKED_ROLES, _SET_PROBE_ROLE)
            )
            assert any(
                edge.grantor == _ADMIN_ROLE and edge.member == _ADMIN_ROLE
                for edge in planted_extra.memberships
                if edge.granted == _SET_PROBE_ROLE
            ), (
                "the extra foreign-grantor row was not planted, so restore is not "
                "required to revoke one"
            )

            await restore_cluster_state(
                lock_conn,
                baseline,
                (*_TRACKED_ROLES, _SET_PROBE_ROLE),
                disposable=(),
            )
            after = await capture_cluster_state(lock_conn, (*_TRACKED_ROLES, _SET_PROBE_ROLE))
        finally:
            await _teardown_multi_grantor_probe(lock_conn, owns_probe=owns_probe)

    # Assert
    restored_pair = [
        edge
        for edge in after.memberships
        if (edge.granted, edge.member)
        == (
            _SET_PROBE_ROLE,
            _APP_ROLE,
        )
    ]
    assert sorted(restored_pair) == sorted(pair_rows), (
        "restore did not reproduce the exact multi-grantor row set. Keying on the "
        "(granted, member) pair alone loses one row or attributes it to the wrong "
        f"grantor:\nbefore={sorted(pair_rows)}\nafter={sorted(restored_pair)}"
    )
    assert len(restored_pair) == 2, (
        f"exactly two rows must survive -- no extra grantor: {sorted(restored_pair)}"
    )
    foreign = [edge for edge in restored_pair if edge.grantor == _ADMIN_ROLE]
    assert len(foreign) == 1, (
        "restore must have RECREATED the row attributed to journal_admin, not to "
        "the connection user. A GRANT without GRANTED BY would produce a row "
        f"attributed to the connection user instead: {sorted(restored_pair)}"
    )
    assert foreign[0].options == (False, False, True), (
        "the recreated foreign-grantor row must carry its ORIGINAL option triple, "
        f"not the defaults a bare GRANT would apply: {foreign[0]}"
    )
    assert after.memberships == baseline.memberships, (
        "restore changed the wider membership set. An EXTRA row whose grantor is "
        "not the connection user survives a revoke that omits GRANTED BY, and a "
        "recreated row lands under the wrong grantor when the grant omits it:\n"
        f"added={sorted(after.memberships - baseline.memberships)}\n"
        f"removed={sorted(baseline.memberships - after.memberships)}"
    )


async def _grant_as_role(dsn: str, role: str, statement: str) -> None:
    """Execute ``statement`` with ``SET LOCAL ROLE role``, so the grantor is ``role``."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(f"SET LOCAL ROLE {role}")
            await conn.execute(statement)
        except BaseException:
            await transaction.rollback()
            raise
        else:
            await transaction.commit()
    finally:
        await conn.close()


async def _teardown_multi_grantor_probe(conn: asyncpg.Connection, *, owns_probe: bool) -> None:
    """Remove the multi-grantor probe role and every edge pinning it.

    Gated on ``owns_probe`` -- the success of this test's own ``CREATE ROLE`` -- and issued
    on the same connection that ran it. Ungated, a run that had REFUSED the role would
    still revoke another session's memberships and drop its role.
    """
    if not owns_probe:
        return
    for granted, member in (
        (_SET_PROBE_ROLE, _APP_ROLE),
        (_SET_PROBE_ROLE, _ADMIN_ROLE),
    ):
        await conn.execute(f"REVOKE {granted} FROM {member} CASCADE")
    await conn.execute(f"DROP ROLE IF EXISTS {_SET_PROBE_ROLE}")


# ---------------------------------------------------------------------------
# Fresh-cluster teardown and required-role ordering, on a disposable cluster.
#
# Two behaviours can only be observed where a role is genuinely ABSENT, which on a
# cluster-global catalog means a separate cluster:
#   * a role the harness CREATED because it was absent must be dropped on teardown
#     (whereas the same role, pre-existing, must survive -- covered on the working
#     cluster);
#   * a reachable cluster missing a required role must FAIL, never skip, and must
#     do so before a migration failure can be converted into a skip.
# ---------------------------------------------------------------------------


async def test_teardown_drops_an_otel_role_it_created_on_a_fresh_cluster() -> None:
    """Where otel_ro was absent, the harness created it and must remove it.

    The mirror of the working-cluster case, which requires a PRE-EXISTING otel_ro
    to survive. Both directions are needed: a teardown that always dropped would
    destroy shared state, and one that never dropped would leak on a fresh cluster.
    Only a disposable cluster can present genuine absence, because roles are
    cluster-global.
    """
    cluster = os.environ.get(DISPOSABLE_CLUSTER_ENV)
    if not cluster:
        pytest.skip(
            f"{DISPOSABLE_CLUSTER_ENV} is not set; genuine role absence needs a "
            "throwaway cluster because roles are cluster-global"
        )
    await _assert_cluster_is_disposable(cluster)

    # Every role mutation on this cluster runs under its lock, so a concurrent local
    # suite sharing the same throwaway cluster cannot interleave with the convergence
    # or the restore.
    async with cluster_role_lock(cluster) as lock_conn:
        # The absence this test needs is ASSERTED, never manufactured. The lane's
        # precondition establishes that no tracked role exists here, which is what makes
        # otel_ro's absence a property of a pristine cluster rather than something a
        # `DROP ROLE` produced -- and that drop was the destructive step: on a cluster
        # that was not pristine it removed a role, a credential and memberships this
        # test never captured, and nothing could tell that role from one it made itself.
        await _assert_tracked_roles_absent(lock_conn, _TRACKED_ROLES)
        outer_baseline = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
        companions: set[str] = set()
        try:
            # The convergence GRANTs to the required roles, so they must exist. Each is
            # created with its own bare statement and recorded on success, so the outer
            # restore drops exactly these and nothing else.
            for role in _REQUIRED_ROLES:
                await _create_owned_role_or_refuse(lock_conn, f"CREATE ROLE {role} LOGIN", role)
                companions.add(role)

            absent_before = not bool(
                await lock_conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)", _OTEL_ROLE
                )
            )
            assert absent_before, (
                "otel_ro must be absent for 'created then dropped' to be provable. The "
                "precondition above should have refused a cluster carrying it"
            )

            # Act -- one converge/restore cycle against this cluster.
            before = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
            disposable = _disposable_roles_for(before)
            assert _OTEL_ROLE in disposable, (
                "an absent otel_ro must be classified disposable, or teardown cannot "
                f"clean up what convergence creates: {disposable}"
            )
            try:
                await _converge_prerequisites(lock_conn)
                created = bool(
                    await lock_conn.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)",
                        _OTEL_ROLE,
                    )
                )
            finally:
                await restore_cluster_state(
                    lock_conn, before, _TRACKED_ROLES, disposable=disposable
                )
            after = await capture_cluster_state(lock_conn, _TRACKED_ROLES)
        finally:
            # Exactly the companions whose own CREATE succeeded, never "absent from a
            # snapshot" -- which would also name a role a concurrent session created
            # after the snapshot was taken.
            await restore_cluster_state(
                lock_conn,
                outer_baseline,
                _TRACKED_ROLES,
                disposable=tuple(sorted(companions)),
            )

    # Assert
    assert created, "convergence did not create otel_ro, so its cleanup is unproven"
    assert _OTEL_ROLE not in after.present, (
        "a role the harness created because it was absent must be dropped on "
        f"teardown: {sorted(after.present)}"
    )
    assert set(_REQUIRED_ROLES) <= after.present, (
        f"teardown must never drop a shared role it did not create: {sorted(after.present)}"
    )


@pytest.mark.parametrize("missing", list(_REQUIRED_ROLES))
async def test_a_reachable_cluster_missing_a_required_role_fails_hard(missing: str) -> None:
    """The required-role guard FAILS on a reachable cluster, never skips.

    Skipping would report green for every posture contract. The guard is exercised
    directly against a cluster where the role was never created, which is the only
    way to observe true absence without destroying shared state.
    """
    cluster = os.environ.get(DISPOSABLE_CLUSTER_ENV)
    if not cluster:
        pytest.skip(
            f"{DISPOSABLE_CLUSTER_ENV} is not set; genuine role absence needs a "
            "throwaway cluster because roles are cluster-global"
        )
    async with _cluster_missing(missing):
        conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
        try:
            # Act / Assert -- Failed is pytest's own failure exception, so catching
            # it here is what distinguishes "failed" from "skipped". A Skipped
            # exception would propagate and mark this test skipped, which is the
            # outcome under test.
            with pytest.raises(Failed) as caught:
                await assert_required_roles_present(conn, _REQUIRED_ROLES)
        finally:
            await conn.close()

    message = str(caught.value)
    assert missing in message, (
        f"the failure must NAME the missing role so it is actionable: {message}"
    )
    assert "reachable" in message, (
        "the failure must state that the server was reachable, which is what makes "
        f"this a hard failure rather than a skip: {message}"
    )


# ---------------------------------------------------------------------------
# Lock-cleanup exception precedence.
#
# A bare ``finally: await unlock()`` has two defects: an unlock that raises
# REPLACES the body's exception, so the real failure is lost and the report blames
# the lock; and the close never runs, leaking the connection and its advisory hold
# for the rest of the session. Both steps must always be attempted, the body's
# exception must win, and a cleanup failure may surface only when the body
# succeeded.
# ---------------------------------------------------------------------------


class _BodyFailure(RuntimeError):
    """Raised from a lock body, so its survival through cleanup is identifiable."""


class _CleanupFailure(RuntimeError):
    """Raised from an injected cleanup step, standing in for a real unlock failure."""


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_lock_cleanup_preserves_the_body_exception_when_unlock_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The body's exception wins, and the close is still attempted.

    Both failures happen at once, which is the case a naive ``finally`` gets
    wrong: it would surface the unlock error and never close the connection.
    """
    # Arrange
    closed: list[bool] = []
    real_close = asyncpg.Connection.close

    async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
        if "pg_advisory_unlock" in query:
            raise _CleanupFailure("unlock refused")
        await _ORIGINAL_EXECUTE(self, query, *args)

    async def tracking_close(self: asyncpg.Connection, **kwargs: object) -> None:
        closed.append(True)
        await real_close(self, **kwargs)

    monkeypatch.setattr(asyncpg.Connection, "execute", failing_execute)
    monkeypatch.setattr(asyncpg.Connection, "close", tracking_close)

    # Act / Assert
    with pytest.raises(_BodyFailure) as caught:
        async with cluster_role_lock(RLS_BOOTSTRAP_URL):
            raise _BodyFailure("the real problem")

    assert closed, (
        "the connection must still be closed when the unlock fails, or the session "
        "leaks the connection and its advisory hold"
    )
    notes = getattr(caught.value, "__notes__", [])
    assert any("cleanup also failed" in note for note in notes), (
        "the unlock failure must remain VISIBLE as a note on the body's exception, "
        f"not be silently discarded: {notes}"
    )


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_lock_cleanup_raises_its_own_error_only_when_the_body_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no body exception to protect, a cleanup failure must surface.

    The mirror of the test above: suppressing cleanup errors unconditionally would
    hide a genuinely broken unlock, so the rule is precedence, not suppression.
    """

    # Arrange
    async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
        if "pg_advisory_unlock" in query:
            raise _CleanupFailure("unlock refused")
        await _ORIGINAL_EXECUTE(self, query, *args)

    monkeypatch.setattr(asyncpg.Connection, "execute", failing_execute)

    # Act / Assert
    with pytest.raises(_CleanupFailure, match="unlock refused"):
        async with cluster_role_lock(RLS_BOOTSTRAP_URL):
            pass


@pytest.mark.usefixtures("_required_roles_present", "_rls_provisioned")
async def test_lock_cleanup_attempts_the_unlock_even_when_the_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing close must not prevent the unlock, and must not mask the body error.

    Ordering matters: releasing the advisory lock is what unblocks other sessions,
    so it has to happen even when tearing down the connection goes wrong.
    """
    # Arrange
    unlocked: list[bool] = []

    async def tracking_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
        if "pg_advisory_unlock" in query:
            unlocked.append(True)
        await _ORIGINAL_EXECUTE(self, query, *args)

    async def failing_close(self: asyncpg.Connection, **kwargs: object) -> None:
        raise _CleanupFailure("close refused")

    monkeypatch.setattr(asyncpg.Connection, "execute", tracking_execute)
    monkeypatch.setattr(asyncpg.Connection, "close", failing_close)

    # Act / Assert
    with pytest.raises(_BodyFailure):
        async with cluster_role_lock(RLS_BOOTSTRAP_URL):
            raise _BodyFailure("the real problem")

    assert unlocked, (
        "the advisory lock must be released even when the close fails -- otherwise "
        "a broken close blocks every other session on this key"
    )


# ---------------------------------------------------------------------------
# Cleanup-failure sanitization.
#
# The disposable-database cleanup wraps a PostgresError in an AssertionError. A
# PostgresError carries the server's DETAIL, which can quote catalog contents and
# identifiers, so the chain is suppressed with ``from None`` and only the exception
# TYPE is kept. "The DETAIL is absent" is free unless something plants a DETAIL and
# renders the failure, which is what this control does.
# ---------------------------------------------------------------------------

_SENTINEL_DETAIL = "SENTINEL_DETAIL_MUST_NOT_REACH_CI_OUTPUT"


async def test_cleanup_failure_omits_the_original_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PRODUCTION cleanup path must not print the server's DETAIL.

    Exercises :func:`_drop_disposable_database` itself rather than a local replica of
    its ``raise ... from None``: a replica would keep passing after the production
    code dropped the suppression. A real ``asyncpg.PostgresError`` carrying a sentinel
    in its DETAIL is injected into the role-owned-object cleanup branch, the
    AssertionError the helper raises is RENDERED the way pytest renders it, and the
    sentinel must be absent from that rendering.

    OWNERSHIP IS ESTABLISHED BEFORE ANYTHING IS CREATED. The pristineness
    precondition runs first, under the cluster lock, so this test can only ever drop
    a role it created itself -- and ``created_role`` records that fact rather than
    inferring it later, when the answer would already be contaminated by its own
    CREATE. Cleanup is nested so the role is removed even when the database drop
    raises the injected error, which is the path under test.
    """
    cluster = _disposable_cluster_or_skip()
    tracked = (*_REQUIRED_ROLES, _OTEL_ROLE)
    planted = _postgres_error_with_sentinel_detail()
    rendered = ""

    async with cluster_role_lock(cluster) as lock_conn:
        # Refuse a populated cluster BEFORE creating anything: afterwards this test's
        # own probe role would make the cluster look populated to its own check.
        name, created_role = await _prepare_cleanup_probe(lock_conn)
        try:

            async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
                if "REASSIGN OWNED BY" in query:
                    raise planted
                await _ORIGINAL_EXECUTE(self, query, *args)

            monkeypatch.setattr(asyncpg.Connection, "execute", failing_execute)
            try:
                # Act -- the production helper, rendered as CI would render its failure.
                try:
                    await _drop_disposable_database(cluster, name)
                except AssertionError:
                    rendered = traceback.format_exc()
            finally:
                monkeypatch.undo()
        finally:
            # Nested, so a raising database drop cannot strand the role. The database
            # drop is retried unpatched; the role goes only if this test made it.
            try:
                await _drop_disposable_database(cluster, name)
            finally:
                if created_role:
                    await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")

        # Assert the SANITIZED production exception, after cleanup has run, and while
        # the lock still guarantees no sibling lane has touched this cluster.
        assert rendered, (
            "the production helper did not raise, so the sanitization claim is untested"
        )
        assert _SENTINEL_DETAIL not in rendered, (
            "the server's DETAIL reached the rendered failure. `raise ... from None` in "
            f"_drop_disposable_database is what keeps catalog contents out of CI:\n{rendered}"
        )
        assert "During handling of the above exception" not in rendered, (
            "the implicit chain is visible, which means the original PostgresError -- and "
            f"its DETAIL -- is being printed:\n{rendered}"
        )
        # The concrete class, not the base: PostgresError.new dispatches on SQLSTATE, so
        # 42501 arrives as InsufficientPrivilegeError. Asserting the concrete name is
        # what proves the helper reports the ACTUAL failure class rather than a generic
        # label -- which is the part that makes a sanitized failure diagnosable.
        assert type(planted).__name__ in rendered, (
            f"the exception TYPE ({type(planted).__name__}) must survive sanitization, "
            f"or the failure is not diagnosable:\n{rendered}"
        )
        assert name in rendered, (
            "the sanitized context must still name the database being cleaned up, or an "
            f"operator cannot act on the failure:\n{rendered}"
        )

        # And the lane owns nothing it did not create.
        after = await capture_cluster_state(lock_conn, tracked)
        assert not after.present, (
            "the failing-cleanup path left a tracked role behind, so the next lane's "
            f"precondition would trip: {sorted(after.present)}"
        )
        assert not await _database_exists(lock_conn, name), (
            f"the probe database survived the nested cleanup: {name}"
        )


async def _database_exists(conn: asyncpg.Connection, name: str) -> bool:
    """Whether ``name`` is still present, read on the already-held lock connection."""
    return bool(
        await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name)
    )


def _postgres_error_with_sentinel_detail() -> asyncpg.PostgresError:
    """A real ``asyncpg.PostgresError`` whose DETAIL carries the sentinel.

    Built through ``PostgresError.new`` so the instance carries genuine server
    attributes -- a hand-constructed exception would not exercise the same rendering
    path that prints DETAIL.
    """
    error = asyncpg.PostgresError.new(
        {
            "C": "42501",
            "M": "permission denied",
            "D": _SENTINEL_DETAIL,
            "S": "ERROR",
            "V": "ERROR",
        }
    )
    assert isinstance(error, asyncpg.PostgresError)
    assert _SENTINEL_DETAIL in str(getattr(error, "detail", "")), (
        "the planted DETAIL did not attach, so this fixture cannot demonstrate a leak"
    )
    return error


async def _prepare_cleanup_probe(lock_conn: asyncpg.Connection) -> tuple[str, bool]:
    """The cleanup regression's setup: refuse a populated cluster, THEN build the probe.

    One helper so the ordering is a single fact rather than a convention repeated at
    each call site. Both the real cleanup test and the populated-cluster ordering
    regression drive THIS function, which is what makes the regression a test of
    production setup rather than of its own copy of it.

    The precondition must precede every create: afterwards the probe role this helper
    adds would itself make the cluster look populated, and a genuinely pre-existing
    role could be mistaken for the helper's own and dropped.
    """
    await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
    return await _create_probe_database(lock_conn)


async def _create_probe_database(lock_conn: asyncpg.Connection) -> tuple[str, bool]:
    """A throwaway database plus, if needed, the role that makes the REASSIGN run.

    Returns ``(database_name, created_role)``. The helper skips ``REASSIGN OWNED BY``
    for roles that do not exist, so at least one tracked role must be present or the
    injected failure is never reached and the sanitization claim goes untested.

    OWNERSHIP COMES FROM THE CREATE, NOT FROM A PRECHECK. A ``SELECT`` followed by a
    ``CREATE ROLE IF NOT EXISTS`` block is a time-of-check/time-of-use race: between
    the two, a concurrent session can create the role, and the block then silently
    does nothing while this helper reports ``created_role=True`` -- so a role it never
    made would be dropped on rollback, destroying a credential and memberships it
    never captured. The ``CREATE ROLE`` is therefore issued DIRECTLY and
    ``created_role`` is set only after that exact statement returns.

    PostgreSQL 17 produces TWO distinct verdicts for the loser, both measured:
      * the role was ALREADY COMMITTED when the statement ran -> ``DuplicateObjectError``,
        SQLSTATE 42710, from the catalog pre-check;
      * a competitor created it in a still-OPEN transaction -> the statement BLOCKS on
        the uncommitted index entry and, once the competitor commits, fails with
        ``UniqueViolationError``, SQLSTATE 23505, on ``pg_authid_rolname_index``.
    Only the second needs qualifying: a unique violation on any other constraint is
    not a role-name collision and must not be silently reclassified as one, so the
    constraint name is required. Both are classified as NOT owned and refused, leaving
    the existing role untouched.

    A ``CREATE DATABASE`` that fails after this call created the role would otherwise
    strand it -- the caller never receives ``created_role``, so it cannot know a
    rollback is owed. The role is dropped here before re-raising, and ONLY when this
    call created it. The drop is nested so the original create failure wins: a failure
    to drop is attached as a note rather than displacing the cause, matching the
    precedence the cluster lock uses.
    """
    name = f"{_ABSENT_DATABASE_PREFIX}sanitize_{uuid.uuid4().hex[:8]}"
    created_role = False
    try:
        await lock_conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN")
    except (asyncpg.DuplicateObjectError, asyncpg.UniqueViolationError) as exc:
        # Scoped narrowly: only around the CREATE ROLE, and for a unique violation only
        # when it is the role-name index. Anything else is a different fault and must
        # propagate rather than be reported as "the role already exists".
        if isinstance(exc, asyncpg.UniqueViolationError) and not _is_role_name_collision(exc):
            raise
        _refuse_unowned_probe_role(exc)
    else:
        created_role = True

    try:
        await lock_conn.execute(f'CREATE DATABASE "{name}"')
    except BaseException as create_error:
        if created_role:
            try:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
            except BaseException as drop_error:
                # repr, never str: a PostgresError's str() can carry the server's
                # DETAIL, which may quote catalog contents into a CI log. repr() of
                # these exception types renders the class and the primary message only.
                create_error.add_note(f"rolling back the probe role also failed: {drop_error!r}")
        raise
    return name, created_role


# ---------------------------------------------------------------------------
# Ownership is a LOCAL flag set by a successful CREATE, never a catalog read.
#
# A presence check answers "does this role exist?", which is true whether this test
# made it or a concurrent session did; gating a DROP on that answer destroys a role the
# test refused. Setting the flag on the create's own success is the only signal that
# distinguishes the two, so every plant below goes through _create_owned_role and every
# cleanup is gated on its return.
# ---------------------------------------------------------------------------


async def _settle_task(task: asyncio.Task[Any] | None) -> None:
    """Leave ``task`` finished and its outcome retrieved, whatever state it was in.

    Three distinct states, and only one of them needs cancelling:

    * ``None`` -- nothing was ever started; the only case that returns early.
    * already finished -- it must STILL be awaited. A task that completed with an
      exception nobody retrieved emits "Task exception was never retrieved" at
      interpreter shutdown, and returning early on ``done()`` is exactly how that
      happens. Awaiting a finished task is cheap and retrieves the outcome.
    * unfinished -- cancel, then await. Cancellation only REQUESTS a stop; the await is
      what guarantees the task has actually stopped before a caller closes its
      connection.

    The suppression names exactly three expected outcomes -- ``CancelledError`` from the
    cancel above, ``Failed`` from a refusal under test, and ordinary ``Exception`` from a
    server error. ``Failed`` must be listed explicitly because it derives from
    ``BaseException``, not ``Exception``: a suppression without it lets a re-awaited
    refusal escape the ``finally`` that was only trying to retrieve it, replacing the
    verdict the test had already asserted (measured: one escape produced 14 failures).

    ``KeyboardInterrupt`` and ``SystemExit`` deliberately propagate. Swallowing them here
    would make a cleanup path silently ignore an operator's interrupt or an interpreter
    shutdown, turning a requested stop into a hang.
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Failed, Exception):
        await task


async def _create_gatekeeper_role(conn: asyncpg.Connection) -> bool:
    """Create the NOCREATEDB gatekeeper, refusing rather than adopting a collision.

    Extracted so the on-wire rejection test and the classifier controls drive the SAME
    code: a copy in each would let one diverge while the other kept passing. The 23505
    qualification is the app role's exact rule -- an unrelated unique violation is a
    different fault and propagates, and the refusal names the GATEKEEPER so an operator
    can tell which of the lane's two created roles collided.
    """
    try:
        await conn.execute(
            f"CREATE ROLE {_GATEKEEPER_ROLE} LOGIN CREATEROLE NOCREATEDB "
            f"PASSWORD '{_GATEKEEPER_PASSWORD}'"
        )
    except (asyncpg.DuplicateObjectError, asyncpg.UniqueViolationError) as exc:
        if isinstance(exc, asyncpg.UniqueViolationError) and not _is_role_name_collision(exc):
            raise
        _refuse_unowned_probe_role(exc, role=_GATEKEEPER_ROLE)
    return True


async def _create_owned_role(conn: asyncpg.Connection, statement: str) -> bool:
    """Run ``statement`` and report whether THIS call created the role.

    Returns True only when the CREATE succeeded here. A duplicate verdict means something
    else owns the role, so the caller must not clean it up -- but only the two verdicts
    that genuinely mean "this role name is taken" count as duplicates:

    * ``DuplicateObjectError`` (42710) -- the role was already committed.
    * ``UniqueViolationError`` (23505) -- ONLY when the strict classifier confirms both
      the role-name index and the role catalog. Any other 23505 is a different fault and
      re-raises, because reporting it as "not owned" would both hide the real error and
      silently disarm the caller's cleanup for a role the caller may well have created.
    """
    try:
        await conn.execute(statement)
    except asyncpg.DuplicateObjectError:
        return False
    except asyncpg.UniqueViolationError as exc:
        if not _is_role_name_collision(exc):
            raise
        return False
    return True


async def _create_owned_role_or_refuse(conn: asyncpg.Connection, statement: str, role: str) -> None:
    """Create ``role`` via ``statement``, refusing with the server's own verdict on a clash.

    For the setup paths that cannot continue without owning the role. The refusal carries
    the REAL exception, so its class and SQLSTATE distinguish the already-committed case
    from the lost-race one -- a synthetic stand-in would report the wrong verdict and make
    the message misleading about what actually happened.
    """
    try:
        await conn.execute(statement)
    except (asyncpg.DuplicateObjectError, asyncpg.UniqueViolationError) as exc:
        if isinstance(exc, asyncpg.UniqueViolationError) and not _is_role_name_collision(exc):
            raise
        _refuse_unowned_probe_role(exc, role=role)


async def _resolve_race_create(task: asyncio.Task[Any]) -> tuple[bool, BaseException | None]:
    """Await a blocked ``CREATE ROLE`` task to a concrete verdict, never a cancellation.

    Returns ``(created_here, verdict)``: whether THAT connection's create succeeded, and
    the exception it produced if it did not.

    Cancelling such a task cannot answer the ownership question. Measured on
    PostgreSQL 17, 8/8: cancelling a task blocked in ``CREATE ROLE`` stops the await but
    NOT the statement, which lands anyway once the blocker resolves -- so the task
    reports ``cancelled()`` while the role it created exists, owned by nobody any local
    flag knows about. Awaiting it to a verdict is what makes ownership a fact: the await
    returns only once the server has answered that statement, so a success here is a
    create the server has already applied and a failure is a create that never took.
    Measured 10/10 in both directions: the verdict and another session's view agree.

    ``CancelledError`` and the process-control exceptions are re-raised rather than
    captured. A cancellation reaching here means the surrounding test is being torn down,
    and returning it as a "verdict" would let a caller treat teardown as an ownership
    answer -- exactly the inference this function exists to remove.
    """
    verdict: BaseException | None = None
    try:
        await task
    except (Failed, Exception) as exc:
        verdict = exc
    created_here = verdict is None
    return created_here, verdict


# How long a contended CREATE ROLE is given to reach its WAITING state before the block
# is confirmed. Generous on purpose: a short wait that has not yet blocked would report
# "not blocked" and silently downgrade the test to the already-committed path.
_RACE_BLOCK_SETTLE_SECONDS = 0.5

_ROLE_NAME_INDEX = "pg_authid_rolname_index"
_ROLE_NAME_TABLE = "pg_authid"


def _is_role_name_collision(exc: asyncpg.UniqueViolationError) -> bool:
    """Whether ``exc`` is specifically a duplicate ROLE NAME, not some other clash.

    asyncpg exposes the server's own constraint metadata, so this is the server's
    classification rather than a guess from message text.

    BOTH fields are required. ``pg_authid`` carries a second unique index --
    ``pg_authid_oid_index``, confirmed present on PostgreSQL 17 -- so matching the table
    alone would classify an OID collision as a duplicate role name. And matching the
    constraint alone would accept that index name from any table. Requiring the
    conjunction means a 23505 is adopted only when the server says both "this table"
    and "this index"; anything else propagates as the distinct fault it is.
    """
    constraint = str(getattr(exc, "constraint_name", "") or "")
    table = str(getattr(exc, "table_name", "") or "")
    return constraint == _ROLE_NAME_INDEX and table == _ROLE_NAME_TABLE


def _refuse_unowned_probe_role(exc: asyncpg.PostgresError, *, role: str = _APP_ROLE) -> None:
    """Fail with a bounded, sanitized account of why ``role`` cannot be owned.

    Carries the exception CLASS, the SQLSTATE and the ATTEMPTED role name -- enough to
    diagnose, to distinguish the already-committed case from the lost-race case, and to
    tell which of the lane's two created roles collided. Deliberately omits the server's
    DETAIL, which quotes the offending key and can surface catalog contents in a CI log,
    and uses ``from None`` so the original exception cannot chain that DETAIL back into
    the rendered failure.
    """
    # Raised explicitly rather than via pytest.fail(), because `from None` attaches to
    # a raise statement and not to a call -- and suppressing the chain is the whole
    # point: the original exception's DETAIL must not reach the rendered failure.
    raise Failed(
        f"{role} already exists on the disposable cluster, so this helper cannot "
        f"own it ({type(exc).__name__}, SQLSTATE {getattr(exc, 'sqlstate', 'unknown')}). "
        "Adopting it would mean dropping a role -- and a credential and memberships -- "
        "that this call never created. Ensure the lane runs against a pristine "
        "throwaway cluster.",
        pytrace=False,
    ) from None


async def test_the_sentinel_detail_control_can_actually_fail() -> None:
    """Positive control: the same search DOES find the DETAIL when the chain is kept.

    Without this, the test above could pass against a renderer that never shows
    exception detail at all, and would prove nothing about ``from None``.
    """
    # Arrange
    planted = _postgres_error_with_sentinel_detail()

    # Act -- the UNSANITIZED wrap, chaining the original.
    rendered = ""
    try:
        try:
            raise planted
        except asyncpg.PostgresError as exc:
            raise AssertionError("cleanup failed") from exc
    except AssertionError:
        rendered = traceback.format_exc()

    # Assert
    assert _SENTINEL_DETAIL in rendered, (
        "a chained PostgresError must expose its DETAIL, or the sanitization test "
        f"above is not measuring anything:\n{rendered}"
    )
    assert "The above exception was the direct cause" in rendered, (
        f"the chain must be visible in the unsanitized rendering:\n{rendered}"
    )


# ---------------------------------------------------------------------------
# Disposable-cluster ordering.
#
# This lane creates and drops the tracked roles. Deciding what to drop from "absent
# in MY baseline" is only safe if the cluster started with none of them: otherwise a
# pre-existing role is classified as the lane's own and destroyed permanently, along
# with a credential and memberships the harness never captured. The precondition
# runs BEFORE any mutation, which is what makes a misconfigured DSN recoverable.
# ---------------------------------------------------------------------------


def _disposable_cluster_or_skip() -> str:
    cluster = os.environ.get(DISPOSABLE_CLUSTER_ENV)
    if not cluster:
        pytest.skip(
            f"{DISPOSABLE_CLUSTER_ENV} is not set; the disposable-cluster ordering "
            "contract has nothing to exercise"
        )
    return cluster


# The NOCREATEDB gatekeeper the on-wire rejection control needs. It is TRACKED so an
# interrupted run's residue produces the lane's bounded refusal rather than a raw
# DuplicateObject out of its own CREATE, and so the pristineness precondition reports it
# by name like any other unexpected role.
_GATEKEEPER_ROLE = "posture_probe_nocreatedb"
# One literal for the CREATE and the DSN that authenticates as it. Two copies would let
# the login silently diverge from the role, and the rejection this gates would then come
# from a failed authentication rather than the missing CREATEDB privilege under test.
_GATEKEEPER_PASSWORD = "gatepass"
_COMPETITOR_PASSWORD = "competitorsecret"
_DISPOSABLE_TRACKED_ROLES = (*_REQUIRED_ROLES, _OTEL_ROLE, _GATEKEEPER_ROLE)


async def _password_fingerprint(conn: asyncpg.Connection, role: str) -> str:
    """An md5 of ``role``'s stored verifier, read on an already-held connection."""
    return str(
        await conn.fetchval(
            """
            SELECT CASE WHEN a.rolpassword IS NULL THEN 'unset'
                        ELSE md5(a.rolpassword) END
            FROM pg_roles r LEFT JOIN pg_authid a ON a.oid = r.oid
            WHERE r.rolname = $1
            """,
            role,
        )
    )


async def test_the_disposable_lane_starts_and_ends_pristine() -> None:
    """A full lane run on a pristine cluster leaves it pristine.

    The baseline half of the contract: if the lane did not return the cluster to its
    starting state, the fail-fast precondition would trip on the NEXT case and the
    suite could not run twice.

    Both reads happen on the HELD lock connection. Reading them on a separate
    connection outside the lock would let a sibling lane mutate the cluster between
    the observation and the run, so a drift this test reported could belong to the
    other lane -- and a drift it missed could be hidden by the other lane's restore.
    The lane under test reuses the same hold rather than reacquiring the key.
    """
    # Arrange
    cluster = _disposable_cluster_or_skip()
    async with cluster_role_lock(cluster) as lock_conn:
        before = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        assert not before.present, (
            "the disposable cluster must start with none of the tracked roles, or this "
            f"test is not measuring the pristine path: {sorted(before.present)}"
        )

        # Act
        async with _cluster_missing(_OTEL_ROLE, lock_conn=lock_conn):
            pass

        # Assert
        after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)

    assert not after.present, (
        "the lane left tracked roles behind, so the next case's precondition would "
        f"trip and the suite could not run twice: {sorted(after.present)}"
    )
    assert after.attributes == before.attributes, (
        f"role attributes drifted:\nbefore={before.attributes}\nafter={after.attributes}"
    )
    assert after.memberships == before.memberships, (
        f"memberships drifted:\n"
        f"added={sorted(after.memberships - before.memberships)}\n"
        f"removed={sorted(before.memberships - after.memberships)}"
    )


async def test_the_disposable_lane_refuses_a_cluster_with_a_preexisting_tracked_role() -> None:
    """A pre-created tracked role fails the lane BEFORE it mutates anything.

    The role is planted with a distinctive attribute, a password AND a membership,
    then asserted byte-identical afterwards: the point is not merely that the lane
    refused, but that it refused EARLY enough to have destroyed nothing.

    The plant, the observation and the check all run on the held lock connection, so
    a sibling lane cannot interleave and be blamed for -- or mask -- the outcome.
    """
    cluster = _disposable_cluster_or_skip()
    planted_role = _OTEL_ROLE

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(
            lock_conn, f"CREATE ROLE {planted_role} LOGIN CREATEROLE PASSWORD 'plantedsecret'"
        )
        assert owns_role, (
            f"a concurrent session already owns {planted_role}, so this test cannot "
            "plant the fixture it needs"
        )
        try:
            # A membership too: an early refusal must preserve edges as well as
            # attributes, and a role with no edges could not show that.
            await lock_conn.execute(f"GRANT pg_monitor TO {planted_role}")
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            planted_fingerprint = await _password_fingerprint(lock_conn, planted_role)
            assert planted_role in planted.present, "the plant did not take"
            assert planted_fingerprint != "unset", (
                "the planted role must carry a credential, or its preservation is "
                "not being measured"
            )

            # Act / Assert -- Failed, not Skipped: a populated cluster is a hard error.
            with pytest.raises(Failed) as caught:
                async with _cluster_missing(_APP_ROLE, lock_conn=lock_conn):
                    pass

            message = str(caught.value)
            assert planted_role in message, (
                f"the refusal must NAME the unexpected role so it is actionable: {message}"
            )
            assert "showing" in message, (
                f"the refusal must state how many roles it is naming: {message}"
            )
            assert "omitted" in message, (
                "the refusal must state how many it suppressed, so truncation cannot "
                f"hide the scale of the misconfiguration: {message}"
            )

            # And nothing was destroyed.
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert after.attributes == planted.attributes, (
                "the lane mutated role state before refusing, which is exactly the "
                f"destruction the precondition exists to prevent:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the lane altered memberships before refusing:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, planted_role) == planted_fingerprint, (
                "the planted role's credential changed, so the refusal came too late"
            )
        finally:
            # Gated on THIS test's create, not on the role's presence.
            if owns_role:
                await lock_conn.execute(f"REVOKE pg_monitor FROM {planted_role}")
                await lock_conn.execute(f"DROP ROLE IF EXISTS {planted_role}")


async def test_concurrent_disposable_lanes_stay_serialized() -> None:
    """Two lanes against the same disposable cluster do not interleave.

    Each creates and drops the same cluster-global roles, so without serialization
    one would drop a role the other had just created and both would see a state
    neither produced. The lock is held across precondition, reset, body and restore,
    so the two runs queue instead.

    The OBSERVER cannot hold the lock while the lanes run -- they need it -- so each
    read takes the lock for itself. That is what removes the race the earlier version
    had: an unlocked read could land mid-lane and report the other lane's transient
    state as this one's drift.
    """
    # Arrange
    cluster = _disposable_cluster_or_skip()
    async with cluster_role_lock(cluster) as observer:
        before = await capture_cluster_state(observer, _DISPOSABLE_TRACKED_ROLES)
        assert not before.present, f"the cluster must start pristine: {sorted(before.present)}"

    # Act -- two full lanes launched together, each taking the lock for itself.
    async def one_lane(missing: str) -> None:
        async with _cluster_missing(missing):
            pass

    await asyncio.gather(one_lane(_OTEL_ROLE), one_lane(_APP_ROLE))

    # Assert -- read under the lock, so both lanes have provably finished.
    async with cluster_role_lock(cluster) as observer:
        after = await capture_cluster_state(observer, _DISPOSABLE_TRACKED_ROLES)

    assert not after.present, (
        "concurrent lanes left tracked roles behind, so one lane's restore ran "
        f"against the other's mutations: {sorted(after.present)}"
    )
    assert after.attributes == before.attributes, (
        f"concurrent lanes left attribute drift:\n"
        f"before={before.attributes}\nafter={after.attributes}"
    )
    assert after.memberships == before.memberships, (
        f"concurrent lanes left membership drift:\n"
        f"added={sorted(after.memberships - before.memberships)}\n"
        f"removed={sorted(before.memberships - after.memberships)}"
    )


async def test_cleanup_regression_refuses_a_populated_cluster_before_creating_anything() -> None:
    """The cleanup regression's own precondition fires before it creates a probe role.

    Ordering is the finding: if the precondition ran AFTER the probe role was created,
    it would see a populated cluster and refuse on the lane's own artefact -- or, worse,
    a genuinely pre-existing journal_app would be classified as the test's and dropped.
    A credentialed journal_app with a membership is planted, and the refusal must come
    early enough that all of it survives byte-identically.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(
            lock_conn,
            f"CREATE ROLE {_APP_ROLE} LOGIN CREATEROLE PASSWORD 'preexistingsecret'",
        )
        assert owns_role, (
            "a concurrent session already owns the app role, so this test cannot plant "
            "its own fixture; the lane's precondition should have refused first"
        )
        try:
            await lock_conn.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            planted_fingerprint = await _password_fingerprint(lock_conn, _APP_ROLE)
            databases_before = await _absent_lane_databases(lock_conn)

            # Act / Assert -- drive the PRODUCTION setup helper, not a local copy of
            # its precondition. That is what makes this a regression against the real
            # ordering: moving the precondition after the create inside
            # _prepare_cleanup_probe turns this test red.
            with pytest.raises(Failed) as caught:
                await _prepare_cleanup_probe(lock_conn)

            message = str(caught.value)
            assert _APP_ROLE in message, f"the refusal must name the pre-existing role: {message}"
            assert "showing" in message, f"the refusal must bound its list: {message}"

            # Nothing created, nothing destroyed.
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert after.attributes == planted.attributes, (
                "attributes changed despite an early refusal:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "memberships changed despite an early refusal:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == planted_fingerprint, (
                "the pre-existing credential was rewritten, so the refusal came too late"
            )
            assert await _absent_lane_databases(lock_conn) == databases_before, (
                "the setup helper created a probe database before refusing, so its "
                "precondition is not running before its create operations"
            )
        finally:
            # Gated on THIS test's create, not on the role's presence: if an external
            # session had won the create, the role would still be present and an
            # ungated drop would destroy it.
            if owns_role:
                await lock_conn.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")


async def _absent_lane_databases(conn: asyncpg.Connection) -> list[str]:
    """This lane's per-case databases, read on an already-held connection."""
    return sorted(
        str(row["datname"])
        for row in await conn.fetch(
            "SELECT datname FROM pg_database WHERE datname LIKE $1",
            f"{_ABSENT_DATABASE_PREFIX}%",
        )
    )


async def test_probe_setup_refuses_a_preexisting_app_role_without_touching_it() -> None:
    """A role the helper did not create must be refused, not adopted.

    Ownership now comes from the ``CREATE ROLE`` itself, so a pre-existing role makes
    that statement raise ``DuplicateObjectError`` and the helper refuses. That is
    strictly safer than the old adopt-and-report-False behaviour: the helper can no
    longer be in a position where it holds a role it did not make.

    The role is planted with a credential AND a membership, and both are asserted
    byte-identical afterwards -- refusing is only useful if it refuses without
    altering anything.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(
            lock_conn, f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD 'preexistingsecret'"
        )
        assert owns_role, (
            "a concurrent session already owns the app role, so this test cannot plant "
            "its own fixture"
        )
        try:
            await lock_conn.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            fingerprint_before = await _password_fingerprint(lock_conn, _APP_ROLE)
            databases_before = await _absent_lane_databases(lock_conn)

            # Act -- the helper must refuse rather than adopt.
            with pytest.raises(Failed) as caught:
                await _create_probe_database(lock_conn)

            message = str(caught.value)
            assert _APP_ROLE in message, (
                f"the refusal must name the role it declined to own: {message}"
            )
            assert "DuplicateObjectError" in message, (
                "the refusal must report the server's own verdict, which is what makes "
                f"the classification trustworthy rather than inferred: {message}"
            )

            # Assert -- nothing altered, nothing created.
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert after.attributes == planted.attributes, (
                "the refused role's attributes changed:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the refused role's memberships changed:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == fingerprint_before, (
                "the refused role's credential was rewritten"
            )
            assert await _absent_lane_databases(lock_conn) == databases_before, (
                "a probe database was created despite the refusal"
            )
        finally:
            # Gated on THIS test's create, not on the role's presence: if an external
            # session had won the create, the role would still be present and an
            # ungated drop would destroy it.
            if owns_role:
                await lock_conn.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")


async def test_probe_setup_refuses_a_role_planted_by_a_concurrent_session() -> None:
    """The race the precheck could not close: a role created AFTER the precondition.

    Deterministic by construction. The outer pristineness precondition runs and
    passes; a SECOND connection then plants a credentialed journal_app with a
    membership; only then does the helper issue its ``CREATE ROLE``. A
    check-then-create helper would have passed its own precheck before the plant and
    reported ``created_role=True`` for a role it never made -- and dropped it on
    rollback. Because ownership now comes from the CREATE statement, PostgreSQL
    serialises the two attempts and the helper loses, refuses, and touches nothing.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        # The precondition passes here -- the cluster IS pristine at this instant.
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        databases_before = await _absent_lane_databases(lock_conn)

        # A genuinely separate session, as a concurrent suite would be.
        intruder = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
        owns_role = False
        try:
            owns_role = await _create_owned_role(
                intruder, f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD 'intrudersecret'"
            )
            assert owns_role, (
                "a third session already owns the app role, so this test cannot stage "
                "the plant it measures"
            )
            await intruder.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            fingerprint_before = await _password_fingerprint(lock_conn, _APP_ROLE)
            assert _APP_ROLE in planted.present, "the intruder's plant did not take"

            # Act -- the helper's CREATE now loses the race.
            with pytest.raises(Failed) as caught:
                await _create_probe_database(lock_conn)

            assert "DuplicateObjectError" in str(caught.value), (
                "the refusal must come from the server's duplicate verdict, which is "
                f"what makes it race-proof rather than precheck-dependent: {caught.value}"
            )

            # Assert -- the intruder's role is intact, and no probe database exists.
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert after.attributes == planted.attributes, (
                "the concurrently-created role's attributes changed:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the concurrently-created role's memberships changed:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == fingerprint_before, (
                "the concurrently-created role's credential was rewritten -- exactly "
                "the destruction a precheck-derived ownership flag would cause"
            )
            assert await _absent_lane_databases(lock_conn) == databases_before, (
                "a probe database survived the refusal"
            )
        finally:
            # Only the intruder removes what the intruder created.
            if owns_role:
                await intruder.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
                await intruder.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
            await intruder.close()


async def test_probe_setup_rolls_back_its_own_role_when_create_database_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing CREATE DATABASE must not strand the role the helper just created.

    This is the one path where the caller cannot clean up: it never receives
    ``created_role``, so it has no way to know a role was made. The helper therefore
    rolls back its own creation before re-raising, and the ORIGINAL create failure is
    what propagates -- a rollback problem is attached as a note rather than displacing
    the cause, matching the precedence the cluster lock uses.
    """
    cluster = _disposable_cluster_or_skip()
    create_failure = RuntimeError("CREATE DATABASE refused")

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        databases_before = await _absent_lane_databases(lock_conn)

        async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
            if "CREATE DATABASE" in query:
                raise create_failure
            await _ORIGINAL_EXECUTE(self, query, *args)

        monkeypatch.setattr(asyncpg.Connection, "execute", failing_execute)
        try:
            # Act / Assert -- the ORIGINAL failure propagates, not a rollback error.
            with pytest.raises(RuntimeError, match="CREATE DATABASE refused") as caught:
                await _prepare_cleanup_probe(lock_conn)
        finally:
            monkeypatch.undo()

        assert caught.value is create_failure, (
            "the original CREATE DATABASE failure must be the exception that "
            f"propagates, not one raised while rolling back: {caught.value!r}"
        )

        # Nothing stranded: no tracked role, no probe database.
        leftover = (await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)).present
        assert not leftover, (
            "the helper created a role and then failed to create the database, leaving "
            "the role behind. The caller never receives created_role on this path, so "
            f"it cannot clean up -- the helper must: {sorted(leftover)}"
        )
        assert await _absent_lane_databases(lock_conn) == databases_before, (
            "a probe database survived a failing CREATE DATABASE"
        )


async def test_probe_setup_note_keeps_the_primary_error_and_leaks_no_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both steps fail: the create error stays primary and the note is repr-safe.

    Two distinct injections, so the precedence is observable rather than assumed: a
    recognisable ``CREATE DATABASE`` error, and a real ``PostgresError`` carrying a
    sentinel in its DETAIL raised by the rollback ``DROP ROLE``.

    Three things must hold. The propagating exception is the create failure OBJECT, not
    the rollback's. The rollback failure survives as a note, so it is not silently
    discarded. And that note uses ``repr``, not ``str`` -- a PostgresError's ``str()``
    can carry the server's DETAIL, which would put catalog contents into a CI log.

    An outer finally removes the role the injected rollback could not, so later cases
    still see a pristine cluster.
    """
    cluster = _disposable_cluster_or_skip()
    create_failure = RuntimeError("CREATE DATABASE refused")
    drop_failure = _postgres_error_with_sentinel_detail()
    # Records that the helper's CREATE ROLE actually reached the server, observed from
    # inside the injection rather than inferred from a later catalog read. That is what
    # makes the cleanup gate answer "did WE create it?" instead of "is it there?".
    created_by_helper: list[bool] = []

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        try:

            async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
                if "CREATE ROLE" in query:
                    await _ORIGINAL_EXECUTE(self, query, *args)
                    created_by_helper.append(True)
                    return
                if "CREATE DATABASE" in query:
                    raise create_failure
                if "DROP ROLE" in query:
                    raise drop_failure
                await _ORIGINAL_EXECUTE(self, query, *args)

            monkeypatch.setattr(asyncpg.Connection, "execute", failing_execute)
            try:
                # Act
                with pytest.raises(RuntimeError, match="CREATE DATABASE refused") as caught:
                    await _prepare_cleanup_probe(lock_conn)
            finally:
                monkeypatch.undo()

            # Assert -- primary exception, note present, note sanitized.
            assert caught.value is create_failure, (
                "the rollback failure displaced the original create failure; the cause "
                f"of the problem must remain the exception that propagates: {caught.value!r}"
            )
            notes = getattr(caught.value, "__notes__", [])
            assert any("rolling back the probe role also failed" in note for note in notes), (
                "the rollback failure must survive as a note rather than be discarded "
                f"-- a silently swallowed rollback hides a stranded role: {notes}"
            )
            joined = "\n".join(notes)
            assert _SENTINEL_DETAIL not in joined, (
                "the note carried the server's DETAIL into the failure output. repr() "
                f"of the exception is what keeps catalog contents out of a CI log: {joined}"
            )
            assert type(drop_failure).__name__ in joined, (
                "the note must still identify the rollback failure's class, or it is "
                f"not actionable: {joined}"
            )
        finally:
            # Gated on the helper's own create having reached the server: the injected
            # DROP ROLE never ran, so the role this call made is still here and is ours
            # to remove. Had the create instead lost to an external session, there would
            # be a role present that is NOT ours, and dropping it would destroy it.
            if created_by_helper:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")

    # Outer proof: the next lane sees a pristine cluster.
    async with cluster_role_lock(cluster) as verifier:
        leftover = (await capture_cluster_state(verifier, _DISPOSABLE_TRACKED_ROLES)).present
        assert not leftover, f"a role survived the injected rollback failure: {sorted(leftover)}"


async def test_probe_setup_cleans_up_when_postgres_itself_rejects_create_database() -> None:
    """A REAL server-side rejection, not an injected exception.

    Every other rollback test fakes the failure by patching ``execute``, which proves
    the Python path but not that the same thing happens when PostgreSQL refuses on the
    wire -- different exception class, different transaction state, no monkeypatch in
    sight. Here the lock connection is replaced by one authenticated as a role WITHOUT
    ``CREATEDB``, so the server rejects ``CREATE DATABASE`` with
    ``InsufficientPrivilegeError`` (SQLSTATE 42501, measured on PostgreSQL 17) and the
    helper's rollback runs for real.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        databases_before = await _absent_lane_databases(lock_conn)
        # A login role that may CREATEROLE (so the helper's CREATE ROLE succeeds) but
        # not CREATEDB (so the server rejects the database).
        # Ownership of the gatekeeper is derived from ITS create too: residue from an
        # interrupted run must be refused, not adopted and later dropped.
        owns_gatekeeper = await _create_gatekeeper_role(lock_conn)
        try:
            parsed = urlparse(maintenance_dsn(cluster))
            host = parsed.hostname or "localhost"
            port = f":{parsed.port}" if parsed.port else ""
            limited_dsn = urlunparse(
                parsed._replace(netloc=f"{_GATEKEEPER_ROLE}:{_GATEKEEPER_PASSWORD}@{host}{port}")
            )
            limited = await asyncpg.connect(limited_dsn, timeout=5)
            try:
                # Act -- no monkeypatch; PostgreSQL does the rejecting.
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await _create_probe_database(limited)
            finally:
                await limited.close()

            # Assert -- the role the helper created is gone, and no database exists.
            # The gatekeeper is deliberately still present: this test created it and
            # removes it in its own finally. What must be gone is the role the HELPER
            # created, so it is excluded rather than asserting a set that can never be
            # empty here.
            present = (await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)).present
            assert _GATEKEEPER_ROLE in present, (
                "the gatekeeper vanished, so the rejection may not have come from the "
                "missing privilege this test relies on"
            )
            leftover = present - {_GATEKEEPER_ROLE}
            assert not leftover, (
                "the helper's role survived a genuine server-side CREATE DATABASE "
                f"rejection, so its rollback does not cover the real path: {sorted(leftover)}"
            )
            assert await _absent_lane_databases(lock_conn) == databases_before, (
                "a probe database exists after the server rejected its creation"
            )
        finally:
            # The app role is dropped ONLY because the helper's rollback is what this
            # test asserts already removed it; re-issuing the drop unconditionally
            # would delete a role a concurrent session had planted and this test had
            # refused. So it is gated on the same catalog read the assertions used.
            # NOT presence-gated. The helper's own rollback is what this test asserts
            # removed the app role, so nothing is owed here; a role still present would
            # belong to a concurrent session, and dropping it on the strength of "it
            # exists" is exactly the destruction these gates prevent. The gatekeeper IS
            # dropped, because this test's own create is what made it.
            if owns_gatekeeper:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_GATEKEEPER_ROLE}")


@dataclass
class _RaceOutcome:
    """The concrete result of the contended create: who made the role, and what failed.

    ``created_here`` is the ONLY ownership signal a caller may act on, and it comes from
    the contended statement's own verdict rather than from a catalog read or a task state.
    """

    created_here: bool
    verdict: BaseException | None
    blocked_on_create: bool


class _RaceLane:
    """The contended side of a staged role-name collision.

    A caller starts its statement with :meth:`start`, releases the blocker with
    :meth:`release`, and reads the concrete outcome. Every step runs inside the context
    manager below, whose exit resolves the blocker's transaction and awaits the statement
    to a verdict before any connection closes.
    """

    def __init__(
        self,
        conn: asyncpg.Connection,
        commit_blocker: Callable[[], Awaitable[None]],
        rollback_blocker: Callable[[], Awaitable[None]],
    ) -> None:
        self.conn = conn
        self.commit_blocker = commit_blocker
        self.rollback_blocker = rollback_blocker
        self._task: asyncio.Task[Any] | None = None
        self._blocked_on_create = False
        self._outcome: _RaceOutcome | None = None
        self._blocker_resolved = False
        self._blocker_committed = False

    def start(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run ``coro`` as the contended statement. It will block on the open blocker."""
        assert self._task is None, "the contended statement was already started"
        self._task = asyncio.create_task(coro)

    async def confirm_blocked(self) -> None:
        """Record that the statement is genuinely WAITING, not already resolved.

        Without this the test could be exercising the already-committed path, whose verdict
        is a different SQLSTATE carrying no DETAIL.
        """
        assert self._task is not None, "nothing was started, so nothing can be blocked"
        await asyncio.sleep(_RACE_BLOCK_SETTLE_SECONDS)
        self._blocked_on_create = not self._task.done()

    async def release(self, *, commit: bool) -> _RaceOutcome:
        """Resolve the blocker, then await the contended statement to a concrete verdict.

        ``commit=True`` makes the blocker's role real, so the contended statement loses;
        ``commit=False`` rolls it back, so the contended statement WINS and its own session
        owns the resulting role. Either way the answer is the statement's verdict -- never
        a cancellation, and never a catalog read.
        """
        assert self._task is not None, "nothing was started, so there is nothing to release"
        await (self.commit_blocker() if commit else self.rollback_blocker())
        self._blocker_committed = commit
        self._blocker_resolved = True
        created_here, verdict = await _resolve_race_create(self._task)
        self._outcome = _RaceOutcome(
            created_here=created_here, verdict=verdict, blocked_on_create=self._blocked_on_create
        )
        return self._outcome

    @property
    def blocker_owns_role(self) -> bool:
        """Whether the BLOCKER's create survived -- true only if its transaction committed.

        A rolled-back ``CREATE ROLE`` never existed, so the blocker owns nothing and must
        not drop: at that point the role present under that name belongs to whichever
        contended statement then won it. Reading this instead of "the create statement
        succeeded" is the difference between ownership and a statement that merely ran.
        """
        return self._blocker_committed

    async def finalize(self) -> bool:
        """Resolve the blocker, leave nothing in flight, and report what THIS lane made.

        Reached from the context manager's ``finally``, including the paths where an
        assertion tripped before :meth:`release` -- or before the contended statement was
        ever started.

        THE BLOCKER'S TRANSACTION IS ALWAYS RESOLVED HERE, even when nothing was started.
        An open transaction makes every later statement on that connection part of it, so
        a cleanup issued inside one is discarded when the connection closes and rolls it
        back: the drops appear to run and nothing is actually removed. Rolling back first
        is also what releases a contended statement, so the verdict below can be read --
        and it revokes the blocker's own ownership, which :attr:`blocker_owns_role` then
        reports.

        The statement is awaited to a verdict, never cancelled: a cancelled ``CREATE
        ROLE`` still lands on the server, leaving a role no local flag claims.
        """
        if not self._blocker_resolved:
            with contextlib.suppress(asyncpg.PostgresError, asyncpg.InterfaceError):
                await self.rollback_blocker()
            self._blocker_committed = False
            self._blocker_resolved = True
        if self._outcome is not None:
            return self._outcome.created_here
        if self._task is None:
            return False
        created_here, _ = await _resolve_race_create(self._task)
        return created_here


@asynccontextmanager
async def _a_true_role_create_race(cluster: str) -> AsyncIterator[_RaceLane]:
    """Stage the genuine overlap: a competitor's CREATE ROLE held UNCOMMITTED.

    Measured on PostgreSQL 17: while the competitor's transaction is open, a second
    session's ``CREATE ROLE`` BLOCKS on the uncommitted index entry rather than failing,
    and only after the commit does it fail with ``UniqueViolationError`` (23505) on
    ``pg_authid_rolname_index`` -- the one verdict whose DETAIL quotes the role name.
    That makes this the only staging that can exercise the 23505 branch or its
    sanitization.

    THE EXIT IS THE POINT. It resolves the blocker's transaction, awaits the contended
    statement to a concrete verdict, and only then drops -- each side dropping strictly
    what its own successful CREATE produced, on the connection that ran it. Nothing here
    cancels an in-flight ``CREATE ROLE`` and nothing infers ownership from a task state or
    a catalog read, so a role an external session owns survives this exit untouched.
    """
    competitor = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    transaction = competitor.transaction()
    lane = _RaceLane(
        await asyncpg.connect(maintenance_dsn(cluster), timeout=5),
        commit_blocker=transaction.commit,
        rollback_blocker=transaction.rollback,
    )
    competitor_owns_role = False
    try:
        await transaction.start()
        # Refuses with the server's own verdict rather than adopting the role: an
        # external owner here means the race cannot be staged, and the exit below must
        # then leave that owner's role completely untouched.
        await _create_owned_role_or_refuse(
            competitor,
            f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD '{_COMPETITOR_PASSWORD}'",
            _APP_ROLE,
        )
        competitor_owns_role = True
        await competitor.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
        yield lane
    finally:
        lane_owns_role = await lane.finalize()
        if lane_owns_role:
            await lane.conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
        # The blocker owns the role only if its transaction COMMITTED. A rolled-back
        # create never existed, so dropping on the strength of the statement having run
        # would destroy whatever the contended side then won -- which is exactly the
        # role the branch above has already cleaned up or deliberately left alone.
        if competitor_owns_role and lane.blocker_owns_role:
            with contextlib.suppress(asyncpg.PostgresError):
                await competitor.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
            with contextlib.suppress(asyncpg.PostgresError):
                await competitor.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
        await competitor.close()
        await lane.conn.close()


async def test_probe_setup_refuses_the_loser_of_a_truly_overlapping_create() -> None:
    """The genuine race: a competitor's CREATE ROLE is still UNCOMMITTED when ours runs.

    Distinct from the already-committed case, and the reason 42710 alone is not enough.
    Measured on PostgreSQL 17: when the competitor's transaction is still open, our
    ``CREATE ROLE`` does not fail immediately -- it BLOCKS on the uncommitted index
    entry, and only once the competitor commits does it fail with
    ``UniqueViolationError`` (23505) on ``pg_authid_rolname_index``. A handler that
    caught only ``DuplicateObjectError`` would let that escape unclassified.

    The overlap is deterministic, not timing-hopeful: the helper runs as a retained
    task, and the test PROVES the task is still blocked before committing the
    competitor. Its refusal is then required, and the competitor-owned role's
    attributes, credential and membership must all survive byte-identically.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        databases_before = await _absent_lane_databases(lock_conn)

        async with _a_true_role_create_race(cluster) as lane:
            # Act -- the helper blocks inside CREATE ROLE while the competitor holds it.
            lane.start(_create_probe_database(lane.conn))
            await lane.confirm_blocked()
            outcome = await lane.release(commit=True)

            # Assert -- the overlap was real, and the verdict is the 23505 one.
            assert outcome.blocked_on_create, (
                "the helper did not block on the competitor's uncommitted CREATE ROLE, "
                "so this test exercised the already-committed path and says nothing "
                "about the true race"
            )
            assert not outcome.created_here, (
                "the helper won the race, so there is no refusal to inspect"
            )
            assert isinstance(outcome.verdict, Failed), (
                f"the helper must refuse rather than propagate a raw error: {outcome.verdict!r}"
            )
            message = str(outcome.verdict)
            assert "UniqueViolationError" in message, (
                "the true race loser must be classified from its UniqueViolation "
                f"verdict; a 42710-only handler would not have caught it: {message}"
            )
            assert "23505" in message, (
                f"the refusal must carry the SQLSTATE that identifies the case: {message}"
            )
            assert "already exists" in message, (
                f"the refusal must state the cause in operator terms: {message}"
            )
            assert _SENTINEL_KEY_PHRASE not in message, (
                "the refusal must not carry the server's DETAIL, which quotes the "
                f"offending key: {message}"
            )

            # The competitor's role is untouched.
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert _APP_ROLE in after.present, (
                "the helper dropped a role a concurrent session owned -- exactly the "
                "destruction create-derived ownership exists to prevent"
            )
            assert after.attributes[_APP_ROLE].can_login is True, (
                f"the competitor's role attributes changed: {after.attributes[_APP_ROLE]}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) != "unset", (
                "the competitor's credential was cleared"
            )
            assert any(
                edge.granted == "pg_monitor" and edge.member == _APP_ROLE
                for edge in after.memberships
            ), f"the competitor's membership was revoked: {sorted(after.memberships)}"
            assert await _absent_lane_databases(lock_conn) == databases_before, (
                "a probe database survived the refusal"
            )


# The literal the server puts in a duplicate-key DETAIL. Asserting its ABSENCE is what
# proves the refusal is sanitized rather than merely differently worded.
_SENTINEL_KEY_PHRASE = "Key (rolname)"


async def test_no_caller_finally_destroys_a_concurrently_planted_role() -> None:
    """A refused role must survive the CALLER's cleanup, not just the helper's.

    The helper refusing is only half the protection: an outer ``finally`` that issues an
    unconditional ``DROP ROLE IF EXISTS`` would destroy the same role a moment later,
    and the destruction would be invisible because the test already failed. This drives
    the full caller path -- ``_prepare_cleanup_probe`` inside the same
    ``try``/``finally`` shape the real tests use -- against a planted role, and requires
    the role to be intact afterwards.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        planter = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
        owns_plant = False
        try:
            owns_plant = await _create_owned_role(
                planter, f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD 'plantersecret'"
            )
            assert owns_plant, (
                "a third session already owns the app role, so this test cannot stage "
                "the plant its caller-gate assertion measures"
            )
            await planter.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            fingerprint_before = await _password_fingerprint(lock_conn, _APP_ROLE)

            # Act -- the caller shape: attempt setup, then run its own cleanup.
            created_role = False
            try:
                _, created_role = await _prepare_cleanup_probe(lock_conn)
            except Failed:
                pass
            finally:
                # This is the gate under test. Ungated, it would drop the planted role.
                if created_role:
                    await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")

            # Assert
            assert created_role is False, (
                "the helper reported ownership of a role it did not create, so the "
                "caller's gate would drop it"
            )
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert _APP_ROLE in after.present, (
                "the caller's finally destroyed a concurrently planted role. Gating the "
                "drop on the helper's ownership flag is what prevents it"
            )
            assert after.attributes == planted.attributes, (
                f"the planted role's attributes changed:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the planted role's memberships changed:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == fingerprint_before, (
                "the planted role's credential was rewritten"
            )
        finally:
            if owns_plant:
                await planter.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
                await planter.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
            await planter.close()


@pytest.mark.parametrize("residue", [_APP_ROLE, _GATEKEEPER_ROLE])
async def test_interrupted_run_residue_produces_a_bounded_refusal(residue: str) -> None:
    """Leftovers from a killed run must refuse with a bounded message, not a raw error.

    A run interrupted between a create and its cleanup -- a mutation run killed
    mid-test, a CI cancellation -- leaves a tracked role behind. The next run must say
    what it found and why it refused, in a message an operator can act on, rather than
    surfacing a raw ``DuplicateObjectError`` from somewhere inside setup. Both roles the
    lane creates are exercised, because the gatekeeper was added to the tracked set for
    exactly this reason.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        # Simulate the residue a killed run leaves.
        owns_residue = await _create_owned_role(
            lock_conn, f"CREATE ROLE {residue} LOGIN PASSWORD 'residuesecret'"
        )
        assert owns_residue, (
            f"a concurrent session already owns {residue}, so this test cannot plant "
            "the residue it simulates"
        )
        try:
            fingerprint_before = await _password_fingerprint(lock_conn, residue)

            # Act -- the lane's own precondition is what must speak first.
            with pytest.raises(Failed) as caught:
                await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)

            message = str(caught.value)
            assert residue in message, f"the refusal must name the residual role: {message}"
            assert "showing" in message, f"the refusal must bound its list: {message}"
            assert "omitted" in message, f"the refusal must report what it suppressed: {message}"
            assert "DuplicateObject" not in message, (
                "residue must be reported by the lane's own bounded precondition, not "
                f"as a raw server error escaping from a create: {message}"
            )

            # And the residue is untouched -- refusing is not licence to clean up.
            assert await _password_fingerprint(lock_conn, residue) == fingerprint_before, (
                "the residual role's credential was rewritten by a refusal"
            )
        finally:
            if owns_residue:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {residue}")


async def test_an_unrelated_unique_violation_is_not_misclassified_as_a_role_collision() -> None:
    """A unique violation on some OTHER constraint must propagate, not be adopted.

    The 23505 handler exists for one specific collision: a duplicate role name. If it
    accepted every ``UniqueViolationError``, an unrelated failure -- a different
    catalog, a different index, a genuine bug -- would be reported as "the role already
    exists" and the real fault would be lost. The classifier is therefore driven
    directly with the server's own metadata for a non-role violation and must reject it.
    """
    cluster = _disposable_cluster_or_skip()

    # Arrange -- a real UniqueViolationError from a table that is not pg_authid.
    conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    unrelated: asyncpg.UniqueViolationError | None = None
    try:
        await conn.execute("CREATE TABLE IF NOT EXISTS posture_probe_unique (k text PRIMARY KEY)")
        await conn.execute("INSERT INTO posture_probe_unique VALUES ('dup')")
        try:
            await conn.execute("INSERT INTO posture_probe_unique VALUES ('dup')")
        except asyncpg.UniqueViolationError as exc:
            unrelated = exc
    finally:
        await conn.execute("DROP TABLE IF EXISTS posture_probe_unique")
        await conn.close()

    # Act / Assert
    assert unrelated is not None, (
        "the fixture did not produce a UniqueViolationError, so the classifier is not "
        "being exercised against a real one"
    )
    assert unrelated.sqlstate == "23505", (
        f"the fixture must produce SQLSTATE 23505: {unrelated.sqlstate}"
    )
    assert str(getattr(unrelated, "table_name", "")) != _ROLE_NAME_TABLE, (
        "the fixture's violation must come from a table other than pg_authid, or it "
        "cannot distinguish a correct classifier from one that accepts everything"
    )
    assert _is_role_name_collision(unrelated) is False, (
        "an unrelated unique violation was classified as a duplicate role name. That "
        "would report someone else's bug as 'the role already exists' and lose the "
        f"real fault: constraint={getattr(unrelated, 'constraint_name', None)!r} "
        f"table={getattr(unrelated, 'table_name', None)!r}"
    )


async def test_a_real_role_collision_is_classified_as_one() -> None:
    """The positive half: the classifier must still recognise the case it exists for.

    Paired with the test above, which only proves it says no. A classifier that said no
    to everything would pass that one and silently stop protecting the race path, so
    the genuine collision is asserted to be recognised from the same metadata.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _a_true_role_create_race(cluster) as lane:
            lane.start(lane.conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN"))
            await lane.confirm_blocked()
            outcome = await lane.release(commit=True)

            assert not outcome.created_here, (
                "the contended create succeeded, so the competitor's was not the one that "
                "committed first and this is not the collision under test"
            )
            assert isinstance(outcome.verdict, asyncpg.UniqueViolationError), (
                f"the verdict must be the 23505 collision, not {outcome.verdict!r}"
            )
            collision = outcome.verdict

    assert _is_role_name_collision(collision) is True, (
        "a genuine concurrent role-name collision was NOT classified as one, so the "
        f"race path would surface a raw error: constraint="
        f"{getattr(collision, 'constraint_name', None)!r} "
        f"table={getattr(collision, 'table_name', None)!r}"
    )


async def test_the_refusal_suppresses_the_servers_duplicate_key_detail() -> None:
    """The already-committed refusal renders with no exception chain at all.

    Measured on PostgreSQL 17: this path's verdict is ``DuplicateObjectError`` whose
    ``detail`` is None, so the absence of the server's quoted key is free here and proves
    nothing on its own -- the DETAIL claim is pinned on the race verdict further down.
    What this path DOES pin is that no chain is rendered: the original exception, whose
    message quotes the role name, must not appear at all.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(lock_conn, f"CREATE ROLE {_APP_ROLE} LOGIN")
        assert owns_role, (
            "a concurrent session owns the app role, so this test cannot plant the "
            "collision it needs"
        )
        try:
            # Act -- render the refusal as a failure report would.
            rendered = ""
            try:
                await _create_probe_database(lock_conn)
            except Failed:
                rendered = traceback.format_exc()

            # Assert
            assert rendered, "the helper did not refuse, so there is nothing to inspect"
            assert _SENTINEL_KEY_PHRASE not in rendered, (
                "the server's duplicate-key DETAIL reached the rendered failure. "
                "`from None` on the refusal is what keeps the offending key out of a "
                f"CI log:\n{rendered}"
            )
            assert "During handling of the above exception" not in rendered, (
                f"the implicit chain is visible, so the original error is printed:\n{rendered}"
            )
            assert "The above exception was the direct cause" not in rendered, (
                f"the explicit chain is visible, so the original error is printed:\n{rendered}"
            )
            assert "SQLSTATE" in rendered, (
                f"the sanitized refusal must still carry the SQLSTATE:\n{rendered}"
            )
        finally:
            if owns_role:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")


# ---------------------------------------------------------------------------
# Classifier precision: BOTH metadata fields, never one.
#
# pg_authid carries a second unique index (pg_authid_oid_index, confirmed present on
# PostgreSQL 17), so a table-only rule would classify an OID collision as a duplicate
# role name; a constraint-only rule would accept that index name from any table. The
# conjunction is what makes adoption safe, and these cases are what pin it.
# ---------------------------------------------------------------------------


class _FakeUniqueViolation(asyncpg.UniqueViolationError):
    """A UniqueViolationError with chosen metadata, for classifier table cases.

    Constructed rather than provoked because the point is to vary the two fields
    independently, including combinations PostgreSQL would not naturally produce on this
    schema -- which is exactly where a one-field rule goes wrong.
    """

    def __init__(self, constraint: str | None, table: str | None) -> None:
        super().__init__("duplicate key value violates unique constraint")
        self.constraint_name = constraint  # type: ignore[misc]
        self.table_name = table  # type: ignore[misc]


@pytest.mark.parametrize(
    ("constraint", "table", "expected"),
    [
        (_ROLE_NAME_INDEX, _ROLE_NAME_TABLE, True),
        ("pg_authid_oid_index", _ROLE_NAME_TABLE, False),
        (_ROLE_NAME_INDEX, "pg_database", False),
        (_ROLE_NAME_INDEX, None, False),
        (None, _ROLE_NAME_TABLE, False),
        (None, None, False),
        ("", "", False),
    ],
    ids=[
        "both_match",
        "oid_index_same_table",
        "role_index_other_table",
        "constraint_only",
        "table_only",
        "neither",
        "empty_strings",
    ],
)
async def test_the_role_collision_classifier_requires_both_metadata_fields(
    constraint: str | None, table: str | None, expected: bool
) -> None:
    """Only the exact pair is a role-name collision; every one-field case is rejected.

    ``oid_index_same_table`` is the case that matters most: a real second unique index on
    the same catalog, so a table-only rule would adopt an OID collision as "the role
    already exists" and lose the actual fault.
    """
    # Arrange / Act
    verdict = _is_role_name_collision(_FakeUniqueViolation(constraint, table))

    # Assert
    assert verdict is expected, (
        f"constraint={constraint!r} table={table!r} classified as {verdict}, expected "
        f"{expected}. Adoption requires BOTH {_ROLE_NAME_INDEX!r} and "
        f"{_ROLE_NAME_TABLE!r}; anything else is a different fault and must propagate."
    )


async def test_the_second_unique_index_the_strict_rule_guards_against_really_exists() -> None:
    """The real-server control for ``oid_index_same_table``.

    That table case asserts an OID collision on ``pg_authid`` is rejected, which only
    matters if such an index exists -- otherwise the strictness guards a combination
    PostgreSQL can never produce and the case is decoration. Read from the live catalog
    rather than declared, because a declaration about the server is not evidence about it.
    """
    cluster = _disposable_cluster_or_skip()

    conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    try:
        # Act -- every unique index on the role-name catalog, named by the server.
        indexes = sorted(
            str(row["indexname"])
            for row in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = $1", _ROLE_NAME_TABLE
            )
        )
    finally:
        await conn.close()

    # Assert
    assert _ROLE_NAME_INDEX in indexes, (
        f"the classifier's constraint name is not an index on {_ROLE_NAME_TABLE}, so it "
        f"could never match a real verdict: {indexes}"
    )
    others = [name for name in indexes if name != _ROLE_NAME_INDEX]
    assert others, (
        f"{_ROLE_NAME_TABLE} carries only one index, so requiring BOTH fields guards "
        "nothing a table-only rule would get wrong -- the strict rule's premise is stale"
    )


async def test_the_already_committed_collision_is_a_separate_sqlstate() -> None:
    """42710 is its own verdict, not a 23505, so the classifier never sees it.

    The strict 23505 rule would be a liability if the common already-committed case also
    arrived as 23505: every such refusal would then have to satisfy the conjunction.
    Measured against the live server -- the two cases are genuinely distinct SQLSTATEs,
    which is why the handler catches both types and qualifies only one.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(lock_conn, f"CREATE ROLE {_APP_ROLE} LOGIN")
        assert owns_role, "a concurrent session owns the app role, so this cannot be staged"
        try:
            # Act -- a second create against an ALREADY-COMMITTED role.
            verdict: BaseException | None = None
            try:
                await lock_conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN")
            except asyncpg.PostgresError as exc:
                verdict = exc

            # Assert
            assert isinstance(verdict, asyncpg.DuplicateObjectError), (
                f"the already-committed case must be a DuplicateObjectError: {verdict!r}"
            )
            assert verdict.sqlstate == "42710", (
                f"the already-committed case must be SQLSTATE 42710: {verdict.sqlstate}"
            )
            assert not isinstance(verdict, asyncpg.UniqueViolationError), (
                "the already-committed case arrived as a unique violation, so it would "
                "have to satisfy the strict 23505 conjunction to be refused at all"
            )
        finally:
            if owns_role:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")


async def test_an_unrelated_unique_violation_propagates_out_of_the_probe_helper() -> None:
    """A 23505 that is not a role collision must escape, not become a refusal.

    Drives the real helper with an injected non-role unique violation and requires the
    ORIGINAL exception out, unchanged. A helper that refused here would report someone
    else's bug as "the role already exists" and the true fault would be lost.
    """
    cluster = _disposable_cluster_or_skip()
    unrelated = _FakeUniqueViolation("pg_authid_oid_index", _ROLE_NAME_TABLE)

    conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    try:
        original_execute = asyncpg.Connection.execute

        async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
            if "CREATE ROLE" in query:
                raise unrelated
            await original_execute(self, query, *args)

        asyncpg.Connection.execute = failing_execute  # type: ignore[method-assign]
        try:
            # Act / Assert -- the same object, not a Failed refusal.
            with pytest.raises(asyncpg.UniqueViolationError) as caught:
                await _create_probe_database(conn)
        finally:
            asyncpg.Connection.execute = original_execute  # type: ignore[method-assign]
    finally:
        await conn.close()

    assert caught.value is unrelated, (
        "an unrelated unique violation was converted into something else, so the real "
        f"fault is no longer visible: {caught.value!r}"
    )


# ---------------------------------------------------------------------------
# Caller cleanup must never target a role an external session created.
#
# Each formerly presence-gated or unconditional site is exercised with a competitor
# plant made OUTSIDE this lane's lock, which is the shape a concurrent suite has.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _competitor_owned_app_role(cluster: str) -> AsyncIterator[str]:
    """A credentialed journal_app with a membership, owned by a SEPARATE connection.

    Created and removed on its own connection, so this lane can never be the thing that
    cleans it up. Yields an md5 fingerprint of the credential, so a caller can prove it
    was not rewritten.

    Its own create is gated the same way every other plant is: if a genuinely external
    session already owns the role, this helper must refuse rather than adopt and later
    drop it.
    """
    owner = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    owns_role = False
    try:
        owns_role = await _create_owned_role(
            owner, f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD 'competitoronly'"
        )
        assert owns_role, (
            "a concurrent session already owns the app role, so this fixture cannot be "
            "the competitor it claims to be"
        )
        await owner.execute(f"GRANT pg_monitor TO {_APP_ROLE}")
        yield await _password_fingerprint(owner, _APP_ROLE)
    finally:
        if owns_role:
            with contextlib.suppress(asyncpg.PostgresError):
                await owner.execute(f"REVOKE pg_monitor FROM {_APP_ROLE}")
            with contextlib.suppress(asyncpg.PostgresError):
                await owner.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")
        await owner.close()


async def test_an_unrelated_unique_violation_propagates_out_of_the_ownership_helper() -> None:
    """A non-role 23505 must ESCAPE the ownership helper, not read as "not owned".

    The strict arm of the helper's own classification. Reporting an unrelated unique
    violation as not-owned does double damage: the real fault vanishes, and the caller's
    cleanup is silently disarmed for a role its statement may well have created. The
    helper is driven with an injected OID-index violation and the original object must
    come back out.
    """
    cluster = _disposable_cluster_or_skip()
    unrelated = _FakeUniqueViolation("pg_authid_oid_index", _ROLE_NAME_TABLE)

    conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    try:
        original_execute = asyncpg.Connection.execute

        async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
            if "CREATE ROLE" in query:
                raise unrelated
            await original_execute(self, query, *args)

        asyncpg.Connection.execute = failing_execute  # type: ignore[method-assign]
        try:
            # Act / Assert -- the same object, not a False verdict.
            with pytest.raises(asyncpg.UniqueViolationError) as caught:
                await _create_owned_role(conn, f"CREATE ROLE {_APP_ROLE} LOGIN")
        finally:
            asyncpg.Connection.execute = original_execute  # type: ignore[method-assign]
    finally:
        await conn.close()

    assert caught.value is unrelated, (
        "an unrelated unique violation was converted into an ownership verdict, so the "
        f"real fault is invisible and the caller's cleanup is disarmed: {caught.value!r}"
    )


async def test_an_unrelated_unique_violation_propagates_out_of_the_refusing_helper() -> None:
    """The refusing variant applies the same strict rule as the reporting one.

    It has its OWN except clause, so the reporting helper's qualification says nothing
    about it. A lax version would convert an unrelated 23505 into "the role already
    exists" and the setup paths that depend on it -- the lane reset, the race staging --
    would report the wrong cause for a fault they did not diagnose.
    """
    cluster = _disposable_cluster_or_skip()
    unrelated = _FakeUniqueViolation("pg_authid_oid_index", _ROLE_NAME_TABLE)

    conn = await asyncpg.connect(maintenance_dsn(cluster), timeout=5)
    try:
        original_execute = asyncpg.Connection.execute

        async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
            if "CREATE ROLE" in query:
                raise unrelated
            await original_execute(self, query, *args)

        asyncpg.Connection.execute = failing_execute  # type: ignore[method-assign]
        try:
            # Act / Assert -- the original object, not a Failed refusal.
            with pytest.raises(asyncpg.UniqueViolationError) as caught:
                await _create_owned_role_or_refuse(
                    conn, f"CREATE ROLE {_APP_ROLE} LOGIN", _APP_ROLE
                )
        finally:
            asyncpg.Connection.execute = original_execute  # type: ignore[method-assign]
    finally:
        await conn.close()

    assert caught.value is unrelated, (
        "an unrelated unique violation became a refusal, so the real fault is lost: "
        f"{caught.value!r}"
    )


async def test_the_refusing_helper_names_the_attempted_role_on_a_real_collision() -> None:
    """The positive half, with the ATTEMPTED role named rather than a default.

    Without it the strict rule above could be satisfied by a variant that re-raised
    everything, removing the refusal the setup paths rely on. The role label matters
    because these paths create several different roles: a refusal naming the wrong one
    sends an operator after the wrong residue.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_role = await _create_owned_role(lock_conn, f"CREATE ROLE {_OTEL_ROLE} LOGIN")
        assert owns_role, "a concurrent session owns the companion role"
        try:
            fingerprint_before = await _password_fingerprint(lock_conn, _OTEL_ROLE)

            # Act
            with pytest.raises(Failed) as caught:
                await _create_owned_role_or_refuse(
                    lock_conn, f"CREATE ROLE {_OTEL_ROLE} LOGIN", _OTEL_ROLE
                )

            # Assert
            message = str(caught.value)
            assert _OTEL_ROLE in message, (
                f"the refusal must name the ATTEMPTED role, not a default: {message}"
            )
            assert _APP_ROLE not in message, (
                f"the refusal named a role this call never attempted: {message}"
            )
            assert "SQLSTATE" in message, f"the refusal must carry the SQLSTATE: {message}"
            assert await _password_fingerprint(lock_conn, _OTEL_ROLE) == fingerprint_before, (
                "the refused role's credential was rewritten"
            )
        finally:
            if owns_role:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_OTEL_ROLE}")


async def test_the_ownership_helper_refuses_to_claim_a_role_it_did_not_create() -> None:
    """``_create_owned_role`` reports False for a role an external session already owns.

    Every cleanup in this file is gated on this one return value, so a helper that
    over-claimed would arm every gate at once. The downstream tests assert the gate HELD;
    this asserts the signal feeding it is correct, against both duplicate verdicts.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _competitor_owned_app_role(cluster):
            # Act -- the same statement a test would use to plant its own fixture.
            claimed = await _create_owned_role(lock_conn, f"CREATE ROLE {_APP_ROLE} LOGIN")

            # Assert
            assert claimed is False, (
                "the helper claimed a role a separate session created. Every cleanup gate "
                "in this file reads this value, so a True here arms all of them against "
                "a role none of them made"
            )

        # And the positive half, on a now-pristine cluster: it must still say True for a
        # create that genuinely succeeded, or the gates would never fire and every test's
        # own fixture would leak.
        owns_role = await _create_owned_role(lock_conn, f"CREATE ROLE {_APP_ROLE} LOGIN")
        try:
            assert owns_role is True, (
                "the helper denied ownership of a role it just created, so no cleanup "
                "would ever run and every fixture would leak"
            )
        finally:
            if owns_role:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")


@pytest.mark.parametrize(
    "drive",
    ["prepare_cleanup_probe", "create_probe_database"],
)
async def test_no_caller_destroys_a_competitor_owned_role_at_any_entry_point(
    drive: str,
) -> None:
    """Both setup entry points, against a competitor-owned role, destroy nothing.

    The helper refusing is half the protection; the other half is that no caller's
    ``finally`` then drops the same role. Both entry points are driven, because each has
    its own callers and a gate fixed at one would say nothing about the other.

    Survival is asserted on the credential and the membership, not merely on presence: a
    ``DROP`` then re-``CREATE`` would restore presence while destroying both.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _competitor_owned_app_role(cluster) as fingerprint_before:
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)

            # Act -- the caller shape: attempt setup, run its own gated cleanup.
            owns_role = False
            try:
                if drive == "prepare_cleanup_probe":
                    _, owns_role = await _prepare_cleanup_probe(lock_conn)
                else:
                    _, owns_role = await _create_probe_database(lock_conn)
            except Failed:
                pass
            finally:
                if owns_role:
                    await lock_conn.execute(f"DROP ROLE IF EXISTS {_APP_ROLE}")

            # Assert
            assert owns_role is False, (
                "the helper claimed ownership of a competitor's role, so the caller's "
                "gate would drop it"
            )
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert _APP_ROLE in after.present, (
                f"the competitor's role was destroyed via the {drive} path"
            )
            assert after.attributes == planted.attributes, (
                f"the competitor's role attributes changed:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the competitor's memberships changed:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == fingerprint_before, (
                "the competitor's credential was rewritten, which a DROP-then-CREATE "
                "would do while leaving presence intact"
            )


async def test_a_blocked_create_that_wins_after_a_rollback_is_owned_by_its_own_session() -> None:
    """When the blocker rolls back, the blocked CREATE succeeds and its session owns it.

    The abandoned-statement hazard, made deterministic instead of cancelled. Measured on
    PostgreSQL 17: cancelling the task does NOT cancel the statement -- it lands anyway
    once the blocker resolves, leaving a role present that no local flag claims, which is
    unowned residue by construction. Resolving the statement to its verdict instead makes
    ownership a fact: the create either succeeded on THAT connection or it did not, and
    only a success authorizes the drop.

    The role must be gone afterwards, asserted on a separate hold, because residue here
    trips the next lane's pristineness precondition.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _a_true_role_create_race(cluster) as lane:
            lane.start(lane.conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN"))
            await lane.confirm_blocked()

            # Act -- the blocker ROLLS BACK, so the blocked create is released and WINS.
            outcome = await lane.release(commit=False)

            # Assert
            assert outcome.blocked_on_create, (
                "the create was not blocked, so this test is not exercising the contended path"
            )
            assert outcome.created_here, (
                "the released create did not succeed, so the ownership this test pins "
                f"never transferred: {outcome.verdict!r}"
            )
            assert (
                _APP_ROLE
                in (await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)).present
            ), (
                "the create reported success but the role is absent, so the verdict is "
                "not a usable ownership signal"
            )

    # Asserted on a fresh hold, OUTSIDE the one above: pg_advisory_lock is re-entrant
    # only within a session, so reacquiring it while the outer hold is open self-deadlocks.
    async with cluster_role_lock(cluster) as verifier:
        leaked = (await capture_cluster_state(verifier, _DISPOSABLE_TRACKED_ROLES)).present
        assert not leaked, (
            "the winning create's role was left behind, so the next lane's pristineness "
            f"precondition would refuse: {sorted(leaked)}"
        )


# ---------------------------------------------------------------------------
# _settle_task lifecycle, exercised without a server.
#
# Its three states are pure asyncio, so they are pinned directly: returning early on a
# FINISHED task would leave a completed exception unretrieved, and not awaiting an
# unfinished one would leave it running past its connection's close.
# ---------------------------------------------------------------------------


def _would_warn_about_unretrieved_exception(task: asyncio.Task[Any]) -> bool:
    """Whether ``task`` still owes a retrieval, read from asyncio's own bookkeeping.

    ``asyncio`` sets this flag when a future completes with an exception and clears it the
    moment something reads that exception; at ``__del__`` a still-set flag is what emits
    "Task exception was never retrieved". Reading the flag is how the warning can be
    asserted deterministically -- waiting for garbage collection and capturing the
    loop's exception handler would make the check depend on collection timing.

    Deliberately NOT ``task.exception()``: calling that would itself clear the flag, so
    the check would report success regardless of what the code under test did.
    """
    return bool(getattr(task, "_log_traceback", False))


async def test_settling_a_finished_failed_task_retrieves_its_exception() -> None:
    """A task that already FAILED must still be awaited, or its exception goes unobserved.

    The state an early ``done()`` return would skip. Retrieval is observed through
    asyncio's own flag, so the assertion fails for a settle that returns without awaiting
    -- which is exactly the defect, and one no other check in this file can see.
    """

    async def fail_immediately() -> None:
        raise RuntimeError("the statement failed")

    # Arrange -- finished, with an exception nobody has looked at yet.
    task = asyncio.create_task(fail_immediately())
    await asyncio.sleep(0)
    assert task.done(), f"the task must have completed: {task}"
    assert not task.cancelled(), f"the task must have FAILED, not been cancelled: {task}"
    assert _would_warn_about_unretrieved_exception(task), (
        "the task's exception is already retrieved before the act, so this test cannot "
        "observe whether settling retrieves it"
    )

    # Act
    await _settle_task(task)

    # Assert
    assert not _would_warn_about_unretrieved_exception(task), (
        "settling left the exception unretrieved, so this task emits 'Task exception was "
        "never retrieved' at shutdown. Returning early on a FINISHED task is what does it"
    )
    assert not task.cancelled(), (
        "a task that had already FAILED must not be reported as cancelled: settling "
        f"cancelled a finished task, changing its outcome: {task}"
    )
    # A second settle is also safe, which is what a nested cleanup path does.
    await _settle_task(task)


async def test_settling_an_unfinished_task_cancels_and_awaits_it() -> None:
    """An unfinished task must be cancelled AND awaited, not merely asked to stop.

    Cancellation is a request; without the await the task can still be running when its
    caller closes the connection it is using.
    """
    started = asyncio.Event()

    async def block_forever() -> None:
        started.set()
        await asyncio.sleep(3600)

    # Arrange
    task = asyncio.create_task(block_forever())
    await started.wait()
    assert not task.done(), "the task must be unfinished, or this is the other state"

    # Act
    await _settle_task(task)

    # Assert
    assert task.done(), "the task was not awaited to completion, so it could outlive its connection"
    assert task.cancelled(), f"an unfinished task must be cancelled: {task}"


async def test_settling_a_task_that_failed_with_a_base_exception_does_not_escape() -> None:
    """A retrieved ``Failed`` must not propagate out of the settle, displacing the verdict.

    ``_pytest.outcomes.Failed`` -- what the refusal under test raises -- derives from
    ``BaseException``, not ``Exception``. A settle whose suppression names ``Exception``
    re-raises it from the ``finally`` that was only trying to retrieve it, and the
    refusal a test had already asserted resurfaces as that test's failure. Measured: this
    turned one escape into 14 failures across the file.
    """

    async def refuse() -> None:
        raise Failed("the helper refused", pytrace=False)

    # Arrange -- finished, carrying a BaseException-derived outcome.
    task = asyncio.create_task(refuse())
    await asyncio.sleep(0)
    assert task.done(), f"the task must have completed: {task}"
    assert _would_warn_about_unretrieved_exception(task), (
        "the outcome is already retrieved, so this test cannot observe the settle"
    )

    # Act / Assert -- no exception escaping is the whole contract.
    await _settle_task(task)

    assert not _would_warn_about_unretrieved_exception(task), (
        "the BaseException-derived outcome was not retrieved, so it warns at shutdown"
    )


async def test_settling_nothing_is_a_no_op() -> None:
    """``None`` is the only state that returns without awaiting: nothing was started."""
    # Act / Assert -- no exception is the whole contract.
    await _settle_task(None)


def _task_completed_with(exc: BaseException) -> asyncio.Task[Any]:
    """A finished awaitable carrying ``exc``, without ever raising it inside the loop.

    A coroutine that raises ``KeyboardInterrupt`` propagates it through the event loop and
    aborts the whole pytest session before any assertion runs -- measured. Setting the
    exception on a future instead produces the identical state for the code under test (a
    done awaitable whose exception is unretrieved) while keeping it inside the ``await``
    the helper performs.
    """
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    future.set_exception(exc)
    return cast("asyncio.Task[Any]", future)


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
async def test_settling_re_raises_process_control_exceptions(
    control: type[BaseException],
) -> None:
    """An interrupt or an exit must NOT be swallowed by a cleanup helper.

    Both derive from ``BaseException`` and both mean "stop now". A settle that suppressed
    them would turn an operator's Ctrl-C, or an interpreter shutdown, into a cleanup path
    that quietly continues -- the failure mode where a test run refuses to die. The
    refusal type IS suppressed and these are not, so the distinction is asserted rather
    than inherited from the class hierarchy.
    """
    # Arrange
    task = _task_completed_with(control())

    # Act / Assert
    with pytest.raises(control):
        await _settle_task(task)


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
async def test_resolving_a_race_re_raises_process_control_exceptions(
    control: type[BaseException],
) -> None:
    """The race resolver must not report an interrupt as an ownership verdict.

    ``created_here`` is the signal every cleanup gate reads. If a ``KeyboardInterrupt``
    came back as a ``verdict``, the caller would read ``created_here=False`` and skip the
    drop for a role its statement may well have created -- turning an interrupt into
    leaked residue.
    """
    # Arrange
    task = _task_completed_with(control())

    # Act / Assert
    with pytest.raises(control):
        await _resolve_race_create(task)


async def test_resolving_a_race_captures_an_ordinary_failure_as_its_verdict() -> None:
    """The positive half: an ordinary error IS a verdict, so it must be captured.

    Without this, the re-raise controls above could be satisfied by a resolver that
    re-raised everything -- which would break every caller that reads the verdict.
    """
    # Arrange
    failure = RuntimeError("the statement failed")

    # Act
    created_here, verdict = await _resolve_race_create(_task_completed_with(failure))

    # Assert
    assert created_here is False, "a failed create must never report ownership"
    assert verdict is failure, f"the verdict must be the original exception: {verdict!r}"


async def test_resolving_a_race_re_raises_cancellation_rather_than_reporting_it() -> None:
    """A cancelled task is teardown, not a verdict, so it must never become one.

    The inference this whole model exists to remove: a cancellation says nothing about
    whether the statement created the role, so returning it as a ``verdict`` would let a
    caller treat teardown as an ownership answer.
    """
    started = asyncio.Event()

    async def block_forever() -> None:
        started.set()
        await asyncio.sleep(3600)

    # Arrange
    task = asyncio.create_task(block_forever())
    await started.wait()
    task.cancel()

    # Act / Assert
    with pytest.raises(asyncio.CancelledError):
        await _resolve_race_create(task)


@pytest.mark.parametrize("abandon_after", ["start", "confirm_blocked", "nothing_started"])
async def test_the_race_helper_exit_leaves_the_cluster_pristine_from_any_state(
    abandon_after: str,
) -> None:
    """Whatever state the body abandons the lane in, the exit leaves no role behind.

    Three abandonment points, because the exit has to cope with all of them: before
    anything was started, after a statement is in flight, and after it is confirmed
    blocked. Residue from any of them trips the NEXT lane's pristineness precondition,
    which is how a leak here surfaces as an unrelated failure much later.

    The blocker's transaction must be resolved by the exit itself. An open transaction
    absorbs every later statement on that connection, so drops issued inside one are
    discarded when the connection closes and rolls back -- the cleanup appears to run and
    removes nothing. Asserted on a fresh hold, after both connections are gone, because
    that is the only state in which the rollback's effect is visible.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _a_true_role_create_race(cluster) as lane:
            if abandon_after != "nothing_started":
                lane.start(lane.conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN"))
            if abandon_after == "confirm_blocked":
                await lane.confirm_blocked()
            # Act -- leave without releasing. The exit is the code under test.

    # Assert -- on a fresh hold, outside the one above: pg_advisory_lock is re-entrant
    # only within a session, so reacquiring it while the outer hold is open self-deadlocks.
    async with cluster_role_lock(cluster) as verifier:
        leaked = (await capture_cluster_state(verifier, _DISPOSABLE_TRACKED_ROLES)).present
        assert not leaked, (
            f"abandoning the lane after {abandon_after} left a role behind, so the next "
            f"lane's pristineness precondition would refuse: {sorted(leaked)}"
        )


async def test_the_race_helper_exit_never_touches_an_externally_owned_role() -> None:
    """The lane's own exit, against a role a THIRD session owns, destroys nothing.

    The helper is the single cleanup path for five race tests, so its exit is the one
    place a sibling drop would do the most damage. An external owner plants the role
    AFTER the outer precondition has passed -- the shape a concurrent suite has -- and the
    lane is then driven to a refusal and closed. Its credential and membership must
    survive: a ``DROP`` followed by a re-``CREATE`` would restore presence while
    destroying both, so presence alone is not asserted.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        # Planted AFTER the precondition, by a session neither side of the lane owns.
        async with _competitor_owned_app_role(cluster) as fingerprint_before:
            planted = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)

            # Act -- the lane cannot stage its own competitor, so it refuses; the refusal
            # is the path whose exit is under test.
            with pytest.raises(Failed):
                async with _a_true_role_create_race(cluster):
                    pytest.fail("the lane staged a race against an externally-owned role")

            # Assert
            after = await capture_cluster_state(lock_conn, _DISPOSABLE_TRACKED_ROLES)
            assert _APP_ROLE in after.present, (
                "the lane's exit destroyed a role a third session owned -- the sibling "
                "drop this model removes"
            )
            assert after.attributes == planted.attributes, (
                f"the external owner's attributes changed:\n"
                f"planted={planted.attributes}\nafter={after.attributes}"
            )
            assert after.memberships == planted.memberships, (
                "the external owner's memberships changed:\n"
                f"added={sorted(after.memberships - planted.memberships)}\n"
                f"removed={sorted(planted.memberships - after.memberships)}"
            )
            assert await _password_fingerprint(lock_conn, _APP_ROLE) == fingerprint_before, (
                "the external owner's credential was rewritten, which a DROP-then-CREATE "
                "would do while leaving presence intact"
            )


async def test_an_unrelated_unique_violation_propagates_from_the_gatekeeper_create() -> None:
    """The gatekeeper's create gets the same strict rule as the app role's.

    Its ``CREATE ROLE`` has its own except clause, so the app role's qualification says
    nothing about it: a lax gatekeeper would convert an unrelated 23505 into "already
    exists", losing the real fault, and no other test would notice. The helper is driven
    with an injected non-role unique violation on the GATEKEEPER statement specifically,
    and the original exception must come back out.
    """
    cluster = _disposable_cluster_or_skip()
    unrelated = _FakeUniqueViolation("pg_authid_oid_index", _ROLE_NAME_TABLE)

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        original_execute = asyncpg.Connection.execute

        async def failing_execute(self: asyncpg.Connection, query: str, *args: object) -> None:
            if _GATEKEEPER_ROLE in query and "CREATE ROLE" in query:
                raise unrelated
            await original_execute(self, query, *args)

        asyncpg.Connection.execute = failing_execute  # type: ignore[method-assign]
        try:
            # Act / Assert -- the original object, not a Failed refusal.
            with pytest.raises(asyncpg.UniqueViolationError) as caught:
                await _create_gatekeeper_role(lock_conn)
        finally:
            asyncpg.Connection.execute = original_execute  # type: ignore[method-assign]

    assert caught.value is unrelated, (
        "an unrelated unique violation on the gatekeeper create was converted into "
        f"something else, so the real fault is no longer visible: {caught.value!r}"
    )


async def test_a_gatekeeper_role_collision_is_refused_by_name() -> None:
    """A genuine gatekeeper collision refuses, and the message names the GATEKEEPER.

    The positive half. Without it the strict rule above could be satisfied by a
    classifier that rejected everything, silently removing the refusal the residue path
    depends on -- and the refusal has to name the right role, or an operator cannot tell
    which of the lane's two created roles collided.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        owns_gatekeeper = await _create_owned_role(
            lock_conn, f"CREATE ROLE {_GATEKEEPER_ROLE} LOGIN PASSWORD 'plantedgate'"
        )
        assert owns_gatekeeper, "a concurrent session owns the gatekeeper role"
        try:
            fingerprint_before = await _password_fingerprint(lock_conn, _GATEKEEPER_ROLE)

            # Act
            with pytest.raises(Failed) as caught:
                await _create_gatekeeper_role(lock_conn)

            # Assert
            message = str(caught.value)
            assert _GATEKEEPER_ROLE in message, (
                f"the refusal must name the gatekeeper, not the app role: {message}"
            )
            assert "SQLSTATE" in message, (
                f"the refusal must carry the SQLSTATE that identifies the case: {message}"
            )
            assert await _password_fingerprint(lock_conn, _GATEKEEPER_ROLE) == (
                fingerprint_before
            ), "the refused gatekeeper's credential was rewritten"
        finally:
            if owns_gatekeeper:
                await lock_conn.execute(f"DROP ROLE IF EXISTS {_GATEKEEPER_ROLE}")


# ---------------------------------------------------------------------------
# DETAIL suppression, measured on the ONLY verdict that carries a DETAIL.
#
# Measured on PostgreSQL 17: the already-committed refusal arrives as
# DuplicateObjectError with detail None, so asserting the absence of "Key (rolname)"
# against that path is free and would pass with the suppression removed. Only the true
# overlapping race yields the 23505 whose DETAIL quotes the key, so the suppression
# claim is exercised there -- with a positive control proving the same rendering DOES
# surface the phrase when the chain is kept.
# ---------------------------------------------------------------------------


async def test_the_refusal_suppresses_the_detail_of_a_true_race_verdict() -> None:
    """The 23505 refusal's rendering must not carry the server's quoted key.

    Driven through the REAL helper, whose ``except`` block raises the refusal ``from
    None`` with the original 23505 as active context. ``from None`` is invisible in the
    message string and shows up only in the RENDERED traceback, which is what a CI log
    contains -- so the captured exception is rendered with its chain, exactly as a
    failure report would render it, and the server's DETAIL phrase must be absent.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _a_true_role_create_race(cluster) as lane:
            # Act -- the real helper loses the race and refuses.
            lane.start(_create_probe_database(lane.conn))
            await lane.confirm_blocked()
            outcome = await lane.release(commit=True)

            # Assert
            assert outcome.blocked_on_create, (
                "the helper did not block on the uncommitted create, so this is the "
                "already-committed path and its verdict carries no DETAIL"
            )
            assert not outcome.created_here, "the helper won the race, so there is no refusal here"
            verdict = outcome.verdict
            assert isinstance(verdict, Failed), (
                f"the helper must refuse rather than propagate a raw error: {verdict!r}"
            )
            context = verdict.__context__
            assert isinstance(context, asyncpg.UniqueViolationError), (
                f"the refusal's active context must be the 23505 verdict: {context!r}"
            )
            assert _SENTINEL_KEY_PHRASE in str(getattr(context, "detail", "")), (
                "the context's DETAIL does not quote the key, so this rendering cannot "
                "demonstrate a leak"
            )

            rendered = "".join(traceback.format_exception(verdict))
            assert _SENTINEL_KEY_PHRASE not in rendered, (
                "the server's duplicate-key DETAIL reached the rendered failure. "
                f"`from None` is what keeps the offending key out of a CI log:\n{rendered}"
            )
            assert "During handling of the above exception" not in rendered, (
                f"the implicit chain is visible, so the original error is printed:\n{rendered}"
            )
            assert "23505" in rendered, (
                f"the sanitized refusal must still carry the SQLSTATE:\n{rendered}"
            )


async def test_a_chained_race_verdict_does_leak_its_detail() -> None:
    """Positive control: the same rendering DOES surface the key when the chain is kept.

    Without it, the suppression test above would pass against a renderer that never
    prints exception detail, or against a verdict whose DETAIL is empty -- proving
    nothing about ``from None``.
    """
    cluster = _disposable_cluster_or_skip()

    async with cluster_role_lock(cluster) as lock_conn:
        await _assert_tracked_roles_absent(lock_conn, _DISPOSABLE_TRACKED_ROLES)
        async with _a_true_role_create_race(cluster) as lane:
            lane.start(lane.conn.execute(f"CREATE ROLE {_APP_ROLE} LOGIN"))
            await lane.confirm_blocked()
            outcome = await lane.release(commit=True)
            assert not outcome.created_here, "the contended create won, so there is no collision"
            verdict = outcome.verdict
            assert isinstance(verdict, asyncpg.UniqueViolationError), (
                f"the verdict must be the 23505 collision: {verdict!r}"
            )

            # Act -- the UNSANITIZED wrap, chaining the original.
            rendered = ""
            try:
                try:
                    raise verdict
                except asyncpg.UniqueViolationError as exc:
                    raise AssertionError("cannot own the role") from exc
            except AssertionError:
                rendered = traceback.format_exc()

            # Assert
            assert _SENTINEL_KEY_PHRASE in rendered, (
                "a chained race verdict must expose its DETAIL, or the suppression test "
                f"above is not measuring anything:\n{rendered}"
            )
            assert "The above exception was the direct cause" in rendered, (
                f"the chain must be visible in the unsanitized rendering:\n{rendered}"
            )
