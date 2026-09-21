"""Equivalence between the migration chain, grants.sql repair, and the contract.

Three paths must describe the same end state for the audit_log dedup read
capability:

1. ``alembic upgrade head`` on a fresh database.
2. ``psql -f deployment/scripts/grants.sql`` replayed against a live instance
   (the ``restore-db.sh --repair-grants`` path). Its ``REVOKE ALL ON TABLE
   public.audit_log FROM journal_app`` plus the per-column REVOKE loop clear
   every column-level SELECT, so a repair run that forgot to re-grant the
   columns and re-create the policy would silently strip the capability.
3. ``deployment/scripts/verify-db-invariants.sh``, the post-deploy assertion.

Each test provisions its own throwaway database so it cannot disturb the
session-scoped RLS fixture DB.

Exposure is evaluated EFFECTIVELY, not only through grants and policies that
name ``journal_app``: its privileges include everything held by PUBLIC and by
every role it is a member of. The fixture provisions a parent role for exactly
that reason, and the repair path is asserted to either remove the exposure or
abort -- never to report success while cross-user audit rows stay readable.

Prerequisites are AUTHORITATIVE here, not best-effort: ``psql`` on PATH and the
``journal_app`` / ``journal_admin`` / ``otel_ro`` roles are what the deployed
database has, so their absence is a harness defect that fails these tests rather
than skipping them. The only skip is a positively-detected unreachable Postgres,
established before any migration runs.
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
import pytest_asyncio

from tests.conftest import RLS_ADMIN_URL, RLS_APP_PASSWORD, RLS_BOOTSTRAP_URL
from tests.fixtures.cluster_roles import (
    ClusterRoleState,
    assert_required_roles_present,
    capture_cluster_state,
    cluster_role_lock,
    restore_cluster_state,
)
from tests.fixtures.db_invariants import (
    REPO_ROOT as _REPO_ROOT,
)
from tests.fixtures.db_invariants import (
    assert_clean_of as _assert_clean_of,
)
from tests.fixtures.db_invariants import (
    psql_bin as _psql_bin,
)
from tests.fixtures.db_invariants import (
    run_argv as _run,
)
from tests.fixtures.db_invariants import (
    tags_mentioning as _tags_mentioning,
)
from tests.fixtures.db_invariants import (
    verifier_failure_tags as _verifier_failure_tags,
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_GRANTS_SQL = _REPO_ROOT / "deployment" / "scripts" / "grants.sql"

_APP_ROLE = "journal_app"
_SELECT_POLICY_NAME = "audit_log_app_select_self_only"
_INSERT_POLICY_NAME = "audit_log_app_insert_self_only"
CONFLICT_TARGET_COLUMNS = ("actor_id", "target_kind", "target_id", "action", "metadata")

# A role journal_app is a member of. journal_app's effective privileges include
# everything this role holds, and no REVOKE naming journal_app or PUBLIC clears a
# grant made to it -- that is the exposure the fail-closed path exists for.
_PARENT_ROLE = "journal_app_parent_probe"

# The literal self-only contract, as written in the migration and in grants.sql.
# Pinning to it -- rather than only to equality between the two policies -- is
# what makes a mutation widening BOTH predicates to `true` fail.
_SELF_ONLY_CONTRACT = (
    "actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')) "
    "AND actor_id <> '' AND actor_type = 'user'"
)

# Cloud-chain tables that grants.sql names in GRANT / REVOKE statements. gubbi's
# own chain does not create them; the scratch fixture stubs them so the repair
# script can run to completion under ON_ERROR_STOP.
_CLOUD_GRANT_TARGETS = ("tenants", "subscriptions", "llm_budgets", "stripe_events")

# Roles this fixture touches, and the subset it OWNS. The parent probe role
# exists only for these tests, so the fixture creates and drops it. The shared
# roles are ALTERed and GRANTed but never created blindly and never dropped: four
# attributes cannot reconstitute a role's password, rolconfig or connection limit.
_PARENT_PROBE_ROLE = "journal_app_parent_probe"
_OTEL_ROLE = "otel_ro"
# Unconditionally fixture-owned: created here, dropped here.
_FIXTURE_OWNED_ROLES = (_PARENT_PROBE_ROLE,)
_SHARED_ROLES = ("journal_app", "journal_admin", _OTEL_ROLE)
_SCRATCH_TRACKED_ROLES = (*_SHARED_ROLES, _PARENT_PROBE_ROLE)

# Convergence the migration chain, grants.sql and the verifier all assume a deploy
# already performed. Roles are cluster-global, so this runs under the
# maintenance-database lock and is undone by an exact restore on teardown --
# without which this fixture leaks otel_ro, the parent-role membership and
# journal_admin's NOCREATEROLE into every later test in the session.
_REQUIRED_ROLES_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro') THEN
        CREATE ROLE otel_ro LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'journal_app_parent_probe') THEN
        CREATE ROLE journal_app_parent_probe NOLOGIN;
    END IF;
END $$;
ALTER ROLE journal_app  WITH LOGIN;
ALTER ROLE journal_admin WITH LOGIN BYPASSRLS NOCREATEROLE;
ALTER ROLE otel_ro WITH LOGIN NOBYPASSRLS;
GRANT journal_app TO journal_admin WITH ADMIN OPTION;
GRANT pg_monitor TO otel_ro;
GRANT journal_app_parent_probe TO journal_app;
"""

# This fixture NEVER writes a role credential. Passwords are cluster-global
# secrets: overwriting one invalidates every session-scoped pool in the run and,
# on a shared or deployed cluster, destroys a credential the harness cannot
# restore (rolpassword is a hash it never captured). The app connection below
# authenticates with the credential the test environment already provisioned --
# tests/conftest.py owns that, under the same maintenance-database lock -- and a
# credential that does not work is reported as an environment prerequisite.


def _scratch_disposable_roles(before: ClusterRoleState) -> tuple[str, ...]:
    """The roles this fixture's teardown may drop, given the pre-convergence snapshot.

    otel_ro is CONDITIONALLY disposable: the baseline migration creates it, so on a
    migrated cluster it is shared state to preserve, while on a cluster where it is
    genuinely absent the convergence creates it and the harness owns it. The
    decision reads the snapshot taken BEFORE any write, so it can never classify a
    role the harness just created as pre-existing.
    """
    conditional = () if _OTEL_ROLE in before.present else (_OTEL_ROLE,)
    return (*_FIXTURE_OWNED_ROLES, *conditional)


def _with_database(dsn: str, name: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{name}"))


def _maintenance_dsn(dsn: str) -> str:
    return _with_database(dsn, "postgres")


def _psql(dsn: str, sql: str) -> str:
    """Return the unadorned single-value result of ``sql`` against ``dsn``."""
    result = _run([_psql_bin(), "-v", "ON_ERROR_STOP=1", "-tAc", sql, dsn])
    if result.returncode != 0:
        pytest.fail(f"psql failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def _replay_grants(dsn: str) -> subprocess.CompletedProcess[str]:
    return _run([_psql_bin(), "-v", "ON_ERROR_STOP=1", "-f", str(_GRANTS_SQL), dsn])


@pytest_asyncio.fixture
async def scratch_dsn() -> AsyncIterator[str]:
    """A freshly-migrated throwaway database, dropped on teardown.

    Uses a per-test random name so a parallel or crashed run cannot collide
    with, or inherit state from, another.

    grants.sql is a whole-database repair script for the SHARED journal
    database: it GRANTs on the cloud-owned tables too, and ON_ERROR_STOP aborts
    the run if any are missing. gubbi's alembic chain does not create them, so
    the fixture materializes them as bare privilege-target placeholders. They
    carry no column shape on purpose -- nothing here asserts anything about the
    cloud schema, and grants.sql only ever names them in GRANT / REVOKE. This
    is the same prerequisite the deployed database satisfies by having run the
    cloud chain first.

    CLUSTER-GLOBAL ROLE STATE. The convergence below ALTERs shared roles and
    GRANTs memberships, all of which outlive this database. Advisory locks are
    database-scoped while roles are not, so the lock is taken on the cluster's
    MAINTENANCE database -- the same fixed key the posture suite uses -- and held
    across the snapshot, the convergence, the test body and the restore. The
    ``try``/``finally`` opens immediately after the snapshot, so a convergence
    that fails halfway still restores.

    Reaching Postgres at all is the ONE prerequisite whose absence skips: it is
    positively detected before any migration starts. Everything after that point
    -- role provisioning, the alembic upgrade -- fails loudly, because past that
    line a failure is a real defect in what these tests cover.
    """
    name = f"journal_scratch_{uuid.uuid4().hex[:12]}"
    dsn = _with_database(RLS_BOOTSTRAP_URL, name)

    try:
        probe = await asyncpg.connect(_maintenance_dsn(RLS_BOOTSTRAP_URL), timeout=5)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"cannot reach Postgres to provision a scratch DB: {exc}")
    await probe.close()

    async with cluster_role_lock(RLS_BOOTSTRAP_URL) as lock_conn:
        await assert_required_roles_present(lock_conn, ("journal_app", "journal_admin"))
        before = await capture_cluster_state(lock_conn, _SCRATCH_TRACKED_ROLES)
        try:
            await lock_conn.execute(_REQUIRED_ROLES_SQL)
            await lock_conn.execute(f'CREATE DATABASE "{name}"')
            try:
                setup = await asyncpg.connect(dsn, timeout=5)
                try:
                    await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
                finally:
                    await setup.close()

                upgrade = _run(
                    [sys.executable, "-m", "alembic", "upgrade", "head"],
                    JOURNAL_DB_MIGRATION_URL=dsn,
                    JOURNAL_OPERATOR_EMAIL="operator@test.local",
                )
                if upgrade.returncode != 0:
                    pytest.fail(
                        "alembic upgrade head failed for the scratch DB -- Postgres was "
                        "reachable and the required roles were provisioned, so this is a "
                        f"migration-chain failure, not a missing prerequisite:\n"
                        f"stdout:\n{upgrade.stdout}\nstderr:\n{upgrade.stderr}"
                    )

                stubs = await asyncpg.connect(dsn, timeout=5)
                try:
                    for table in _CLOUD_GRANT_TARGETS:
                        await stubs.execute(f'CREATE TABLE IF NOT EXISTS public."{table}" ()')
                    await stubs.execute(
                        "CREATE TABLE IF NOT EXISTS public.outbox_events (id bigserial PRIMARY KEY)"
                    )
                    await stubs.execute(
                        "CREATE TABLE IF NOT EXISTS public.alembic_version_cloud (version_num text)"
                    )
                    await stubs.execute("ALTER TABLE public.outbox_events OWNER TO journal_admin")
                finally:
                    await stubs.close()

                yield dsn
            finally:
                await lock_conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await restore_cluster_state(
                lock_conn,
                before,
                _SCRATCH_TRACKED_ROLES,
                disposable=_scratch_disposable_roles(before),
            )


@dataclass(frozen=True)
class SelectPolicy:
    """One policy that applies to a SELECT statement on audit_log."""

    name: str
    permissive: bool
    cmd: str
    roles: tuple[str, ...]
    predicate: str


@dataclass(frozen=True)
class CapabilityState:
    """The dedup read capability's observable end state on one database.

    ``select_policies`` is the COMPLETE set of policies that apply to a SELECT
    on audit_log, not just the expected one. Permissive policies for the same
    command are OR-ed, so an extra one widens the read surface while every
    existence check on the expected policy still passes. Applicability is
    resolved through role membership, so a policy scoped TO PUBLIC or TO a
    parent role of journal_app is in the set.

    ``readable_columns`` is EFFECTIVE (``has_column_privilege`` folds in PUBLIC
    and inherited roles), while ``direct_column_grants`` is what journal_app
    holds in its own name. The two differ exactly when exposure arrives through
    a grantee a REVOKE naming journal_app cannot reach, which is why
    ``inherited_select_grants`` records the grantee.
    """

    readable_columns: frozenset[str]
    direct_column_grants: frozenset[str]
    inherited_select_grants: frozenset[str]
    table_select: bool
    direct_table_select: bool
    select_policies: tuple[SelectPolicy, ...]
    insert_predicate: str | None
    force_rls: bool
    triggers: tuple[str, ...]


_SELECT_POLICY_QUERY = """
    SELECT polname,
           polpermissive,
           polcmd::text AS cmd,
           ARRAY(
               SELECT rolname::text FROM pg_roles
               WHERE oid = ANY (polroles) ORDER BY rolname
           ) AS roles,
           pg_get_expr(polqual, polrelid) AS predicate
    FROM pg_policy p
    WHERE p.polrelid = 'public.audit_log'::regclass
      AND p.polcmd IN ('r', '*')
      AND EXISTS (
          SELECT 1 FROM unnest(p.polroles) AS r(oid)
          WHERE r.oid = 0 OR pg_has_role($1, r.oid, 'USAGE')
      )
    ORDER BY polname
"""

# SELECT reaching journal_app from a grantee other than itself: PUBLIC (grantee
# oid 0, which pg_has_role does not accept) or any role whose privileges
# journal_app holds. Each row names the grantee, so a failure is actionable.
_INHERITED_SELECT_QUERY = """
    SELECT format('table SELECT via %s',
                  CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                       ELSE a.grantee::regrole::text END) AS descr
    FROM pg_class c, aclexplode(c.relacl) AS a
    WHERE c.oid = 'public.audit_log'::regclass
      AND a.privilege_type = 'SELECT'
      AND a.grantee <> $1::regrole
      AND (a.grantee = 0 OR pg_has_role($1, a.grantee, 'USAGE'))
    UNION ALL
    SELECT format('column SELECT on %I via %s', at.attname,
                  CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                       ELSE a.grantee::regrole::text END) AS descr
    FROM pg_attribute at, aclexplode(at.attacl) AS a
    WHERE at.attrelid = 'public.audit_log'::regclass
      AND at.attnum > 0
      AND a.privilege_type = 'SELECT'
      AND a.grantee <> $1::regrole
      AND (a.grantee = 0 OR pg_has_role($1, a.grantee, 'USAGE'))
"""

# Column-level SELECT held by journal_app in its own name, read off attacl
# rather than has_column_privilege so it excludes anything inherited.
_DIRECT_COLUMN_GRANT_QUERY = """
    SELECT at.attname
    FROM pg_attribute at, aclexplode(at.attacl) AS a
    WHERE at.attrelid = 'public.audit_log'::regclass
      AND at.attnum > 0
      AND a.privilege_type = 'SELECT'
      AND a.grantee = $1::regrole
"""

_DIRECT_TABLE_SELECT_QUERY = """
    SELECT EXISTS (
        SELECT 1 FROM pg_class c, aclexplode(c.relacl) AS a
        WHERE c.oid = 'public.audit_log'::regclass
          AND a.privilege_type = 'SELECT'
          AND a.grantee = $1::regrole
    )
"""


async def _capability_state(dsn: str) -> CapabilityState:
    """Read the dedup read capability's observable end state."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        columns = await conn.fetch(
            """
            SELECT column_name,
                   has_column_privilege($1, 'public.audit_log', column_name, 'SELECT') AS readable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'audit_log'
            """,
            _APP_ROLE,
        )
        table_select = await conn.fetchval(
            "SELECT has_table_privilege($1, 'public.audit_log', 'SELECT')", _APP_ROLE
        )
        policies = await conn.fetch(_SELECT_POLICY_QUERY, _APP_ROLE)
        inherited = await conn.fetch(_INHERITED_SELECT_QUERY, _APP_ROLE)
        direct_columns = await conn.fetch(_DIRECT_COLUMN_GRANT_QUERY, _APP_ROLE)
        direct_table_select = await conn.fetchval(_DIRECT_TABLE_SELECT_QUERY, _APP_ROLE)
        insert_predicate = await conn.fetchval(
            """
            SELECT pg_get_expr(polwithcheck, polrelid)
            FROM pg_policy
            WHERE polname = $1 AND polrelid = 'public.audit_log'::regclass
            """,
            _INSERT_POLICY_NAME,
        )
        force_rls = await conn.fetchval(
            "SELECT relforcerowsecurity FROM pg_class WHERE oid = 'audit_log'::regclass"
        )
        triggers = await conn.fetch(
            """
            SELECT tgname FROM pg_trigger
            WHERE tgrelid = 'audit_log'::regclass AND NOT tgisinternal AND tgenabled = 'O'
            ORDER BY tgname
            """
        )
    finally:
        await conn.close()

    return CapabilityState(
        readable_columns=frozenset(str(row["column_name"]) for row in columns if row["readable"]),
        direct_column_grants=frozenset(str(row["attname"]) for row in direct_columns),
        inherited_select_grants=frozenset(str(row["descr"]) for row in inherited),
        table_select=bool(table_select),
        direct_table_select=bool(direct_table_select),
        select_policies=tuple(
            SelectPolicy(
                name=str(row["polname"]),
                permissive=bool(row["polpermissive"]),
                cmd=str(row["cmd"]),
                roles=tuple(str(r) for r in row["roles"]),
                predicate=_normalize_predicate(row["predicate"]),
            )
            for row in policies
        ),
        insert_predicate=None if insert_predicate is None else str(insert_predicate),
        force_rls=bool(force_rls),
        triggers=tuple(str(row["tgname"]) for row in triggers),
    )


def _normalize_predicate(expr: str | None) -> str:
    """Normalize a predicate so a reprint compares to the source it came from.

    ``pg_get_expr`` reprints in its own canonical form: a USING and a WITH CHECK
    of the same expression differ cosmetically, and the subquery gains an
    ``AS "nullif"`` output alias absent from the source text. Strip whitespace,
    ``::text`` casts, parens and that alias.
    """
    if expr is None:
        return ""
    stripped = re.sub(r"\s+|::text|[()]", "", expr)
    return stripped.replace('AS"nullif"', "")


def _expected_select_policies(state: CapabilityState) -> tuple[SelectPolicy, ...]:
    """The policy set a correct database has: exactly the self-only SELECT policy."""
    return (
        SelectPolicy(
            name=_SELECT_POLICY_NAME,
            permissive=True,
            cmd="r",
            roles=(_APP_ROLE,),
            predicate=_normalize_predicate(state.insert_predicate),
        ),
    )


def _assert_intact(state: CapabilityState) -> None:
    """Assert the complete observable capability state, field by field."""
    assert state.readable_columns == frozenset(CONFLICT_TARGET_COLUMNS), (
        f"readable column set diverges from the dedup conflict target: "
        f"{sorted(state.readable_columns)}"
    )
    # Direct and effective must AGREE. They diverge exactly when exposure arrives
    # via PUBLIC or an inherited role, which is the case a journal_app-scoped
    # REVOKE cannot reach -- so asserting only the effective set would pass a
    # database whose narrow appearance is a coincidence of two wider grants.
    assert state.direct_column_grants == frozenset(CONFLICT_TARGET_COLUMNS), (
        f"journal_app's OWN column grants diverge from the dedup conflict target: "
        f"{sorted(state.direct_column_grants)}"
    )
    assert not state.inherited_select_grants, (
        "audit_log SELECT reaches journal_app through PUBLIC or an inherited role: "
        f"{sorted(state.inherited_select_grants)}"
    )
    assert state.table_select is False, "journal_app must not hold table-wide SELECT"
    assert state.direct_table_select is False, (
        "journal_app must not hold a direct table-wide SELECT grant on audit_log"
    )
    assert state.insert_predicate, f"missing {_INSERT_POLICY_NAME} WITH CHECK predicate"
    # Both predicates are pinned to the LITERAL contract, independently. Checking
    # only that they mirror each other passes a mutation widening both to `true`.
    contract = _normalize_predicate(_SELF_ONLY_CONTRACT)
    assert _normalize_predicate(state.insert_predicate) == contract, (
        f"{_INSERT_POLICY_NAME} WITH CHECK is not the self-only contract:\n"
        f"got={state.insert_predicate!r}\nwant={_SELF_ONLY_CONTRACT!r}"
    )
    assert [p.predicate for p in state.select_policies] == [contract], (
        f"{_SELECT_POLICY_NAME} USING is not the self-only contract:\n"
        f"got={[p.predicate for p in state.select_policies]!r}\nwant={contract!r}"
    )
    assert state.select_policies == _expected_select_policies(state), (
        "SELECT-applicable policy set on audit_log is not exactly the self-only "
        f"policy mirroring {_INSERT_POLICY_NAME}:\n"
        f"got={state.select_policies!r}\ninsert_predicate={state.insert_predicate!r}"
    )


# ---------------------------------------------------------------------------
# Migration upgrade path
# ---------------------------------------------------------------------------


async def test_migration_upgrade_grants_exactly_the_conflict_target_columns(
    scratch_dsn: str,
) -> None:
    # Arrange / Act
    state = await _capability_state(scratch_dsn)

    # Assert
    assert state.readable_columns == frozenset(CONFLICT_TARGET_COLUMNS)
    assert state.table_select is False
    assert [p.name for p in state.select_policies] == [_SELECT_POLICY_NAME]


async def test_migration_upgrade_preserves_force_rls_and_immutable_triggers(
    scratch_dsn: str,
) -> None:
    # Arrange / Act
    state = await _capability_state(scratch_dsn)

    # Assert
    assert state.force_rls is True, "FORCE ROW LEVEL SECURITY must survive the migration"
    assert state.triggers == (
        "trg_audit_log_admin_no_user_actor",
        "trg_audit_log_no_delete",
        "trg_audit_log_no_update",
    )


async def test_migration_upgrade_yields_exactly_the_expected_policy_set(
    scratch_dsn: str,
) -> None:
    """The read surface is exactly the write surface -- one permissive policy, same predicate."""
    # Arrange / Act
    state = await _capability_state(scratch_dsn)

    # Assert
    _assert_intact(state)


# ---------------------------------------------------------------------------
# grants.sql repair path
# ---------------------------------------------------------------------------


async def test_grants_repair_preserves_the_complete_capability_state(scratch_dsn: str) -> None:
    """Replaying grants.sql leaves the migrated end state intact, field for field.

    The REVOKE ALL plus the per-column REVOKE loop inside grants.sql clear every
    column grant, so this passes only because the file re-grants the columns and
    re-creates the SELECT policy.
    """
    # Arrange
    before = await _capability_state(scratch_dsn)
    _assert_intact(before)

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode == 0, f"grants.sql failed:\n{repair.stdout}\n{repair.stderr}"
    after = await _capability_state(scratch_dsn)
    _assert_intact(after)
    assert after == before, f"grant repair changed the capability state:\n{before!r}\n{after!r}"


async def test_grants_repair_restores_a_stripped_capability(scratch_dsn: str) -> None:
    """Repair is a fix, not just a no-op: it re-establishes a removed capability."""
    # Arrange
    _psql(
        scratch_dsn,
        "REVOKE SELECT (actor_id, target_kind, target_id, action, metadata) "
        "ON TABLE public.audit_log FROM journal_app",
    )
    _psql(scratch_dsn, f"DROP POLICY IF EXISTS {_SELECT_POLICY_NAME} ON public.audit_log")
    stripped = await _capability_state(scratch_dsn)
    assert stripped.readable_columns == frozenset()
    assert stripped.select_policies == ()

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode == 0, f"grants.sql failed:\n{repair.stdout}\n{repair.stderr}"
    _assert_intact(await _capability_state(scratch_dsn))


async def test_grants_repair_narrows_a_broadened_column_grant(scratch_dsn: str) -> None:
    """A hand-granted extra column is revoked by the repair run, not carried forward.

    ``ip_address`` is granted directly, outside the five-column contract, the way
    an incident-time ``GRANT SELECT (...)`` would leave it.

    This asserts the OUTCOME, not which statement produces it. Measured on
    PostgreSQL 17, three statements in grants.sql each independently narrow it:
    the schema-wide ``REVOKE ALL ON ALL TABLES``, the ``REVOKE ALL ON TABLE
    public.audit_log``, and the explicit per-column REVOKE loop -- removing any
    two still leaves this test green. The explicit loop is what keeps the outcome
    true if a future server release stops clearing column ACLs on a table-level
    REVOKE, or if a later edit narrows either broad REVOKE's scope.
    """
    # Arrange
    _psql(
        scratch_dsn,
        "GRANT SELECT (ip_address, actor_type) ON TABLE public.audit_log TO journal_app",
    )
    broadened = await _capability_state(scratch_dsn)
    assert {"ip_address", "actor_type"} <= broadened.readable_columns, (
        "fixture failed to broaden the column grant"
    )

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode == 0, f"grants.sql failed:\n{repair.stdout}\n{repair.stderr}"
    after = await _capability_state(scratch_dsn)
    assert after.readable_columns == frozenset(CONFLICT_TARGET_COLUMNS), (
        f"grant repair left a broadened column readable: {sorted(after.readable_columns)}"
    )
    _assert_intact(after)


async def test_grants_repair_is_idempotent(scratch_dsn: str) -> None:
    """A second repair run succeeds and converges on the identical state."""
    # Arrange
    first = _replay_grants(scratch_dsn)
    assert first.returncode == 0, f"grants.sql run 1 failed:\n{first.stdout}\n{first.stderr}"
    after_first = await _capability_state(scratch_dsn)

    # Act
    second = _replay_grants(scratch_dsn)

    # Assert
    assert second.returncode == 0, f"grants.sql run 2 failed:\n{second.stdout}\n{second.stderr}"
    after_second = await _capability_state(scratch_dsn)
    _assert_intact(after_second)
    assert after_second == after_first, (
        f"second repair run changed state:\n{after_first!r}\n{after_second!r}"
    )


# ---------------------------------------------------------------------------
# Repair against exposure that does not arrive through journal_app's own ACLs.
#
# journal_app's effective privileges include everything held by PUBLIC and by
# every role it is a member of. Measured on PostgreSQL 17:
#   * a table- or column-level grant to PUBLIC IS cleared by a REVOKE naming
#     PUBLIC, so grants.sql removes it and reports success;
#   * a grant to a PARENT role is cleared by no REVOKE grants.sql could issue
#     without mutating a role it does not own, so grants.sql aborts instead.
# The one outcome that must be impossible is a rc=0 repair leaving journal_app
# reading anything beyond the five conflict-target columns of its own rows.
#
# Exposure has TWO independent axes, and a widening moves only one of them:
#   * a GRANT widens the readable COLUMN set -- row-level security still filters
#     rows, so a grant-only widening leaves row visibility at zero;
#   * a POLICY widens the visible ROW set -- the column ACL still applies, so a
#     policy-only widening leaves the readable column set unchanged.
# Measuring only rows would make every grant mutation's positive control
# vacuous, and vice versa. So the probe below reports both.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppReadSurface:
    """What a journal_app session can actually read, measured as journal_app.

    Measured through a journal_app connection with NO ``app.current_user_id``
    set: the role whose exposure is in question, in the state where a foreign
    row must be invisible. An admin connection is BYPASSRLS and would report
    every row regardless, proving nothing.
    """

    visible_rows: int
    readable_columns: frozenset[str]

    @property
    def is_narrow(self) -> bool:
        """True when neither axis exceeds the self-only, five-column contract."""
        return self.visible_rows == 0 and self.readable_columns <= frozenset(
            CONFLICT_TARGET_COLUMNS
        )


async def _app_read_surface(dsn: str) -> AppReadSurface:
    """Probe both exposure axes as journal_app, with no actor identity set.

    Logs in with the password the session RLS fixture already assigned. Roles are
    CLUSTER-GLOBAL: setting a fresh password here would invalidate every
    session-scoped pool in the run, so the credential is read, never rotated.
    """
    admin = await asyncpg.connect(dsn, timeout=5)
    try:
        planted = await admin.fetchval("SELECT COUNT(*) FROM public.audit_log")
        all_columns = [
            str(row["column_name"])
            for row in await admin.fetch(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'audit_log'
                ORDER BY column_name
                """
            )
        ]
    finally:
        await admin.close()
    assert planted, "no audit rows exist, so a row-invisibility result proves nothing"

    app = await _connect_as_provisioned_app_role(dsn)
    try:
        # actor_id is readable under the contract, so this counts rows the
        # POLICY lets through rather than tripping the column ACL first.
        visible = int(await app.fetchval("SELECT COUNT(*) FROM public.audit_log"))
        readable: set[str] = set()
        for column in all_columns:
            try:
                # Each probe is its own implicit transaction, so a denial does
                # not poison the ones after it.
                await app.fetch(f'SELECT "{column}" FROM public.audit_log LIMIT 0')  # noqa: S608
            except asyncpg.InsufficientPrivilegeError:
                continue
            readable.add(column)
    finally:
        await app.close()

    return AppReadSurface(visible_rows=visible, readable_columns=frozenset(readable))


async def _connect_as_provisioned_app_role(dsn: str) -> asyncpg.Connection:
    """Connect to ``dsn`` as journal_app using the ALREADY-PROVISIONED credential.

    The credential is READ from the test environment, never written: a role
    password is cluster-global, and overwriting one invalidates every
    session-scoped pool in the run while destroying a secret the harness cannot
    restore (``rolpassword`` is a hash it does not capture).

    A credential that does not authenticate is therefore an ENVIRONMENT
    PREREQUISITE failure, not something to repair by assignment. The message names
    what to fix without echoing the credential itself.
    """
    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    app_dsn = urlunparse(parsed._replace(netloc=f"{_APP_ROLE}:{RLS_APP_PASSWORD}@{host}{port}"))
    try:
        return await asyncpg.connect(app_dsn, timeout=5)
    except asyncpg.InvalidPasswordError as exc:
        pytest.fail(
            f"cannot authenticate as {_APP_ROLE} with the credential the test "
            "environment provisioned. This is an environment prerequisite: the "
            "harness deliberately does NOT assign role passwords, because they are "
            "cluster-global and a reassignment would invalidate every "
            "session-scoped pool in the run and destroy a secret it cannot "
            "restore. Re-provision the role credential to match "
            f"tests.conftest.RLS_APP_PASSWORD, then re-run. ({type(exc).__name__})"
        )


async def _plant_foreign_audit_row(dsn: str) -> None:
    """One user-authored audit row belonging to nobody the reader could be."""
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        await conn.execute(
            """
            INSERT INTO public.audit_log
                (actor_type, actor_id, action, target_kind, target_type, target_id)
            VALUES ('system', 'system:probe', 'identity.deleted', 'user', 'user', 'someone-else')
            """
        )
    finally:
        await conn.close()


async def test_grants_repair_removes_a_public_broadened_column_grant(scratch_dsn: str) -> None:
    """A column granted to PUBLIC is removed by the repair, not silently carried.

    ``has_column_privilege('journal_app', ...)`` reports true for a PUBLIC grant
    and no REVOKE naming journal_app clears it, so before the per-column PUBLIC
    REVOKE this case survived a repair run that reported success.
    """
    # Arrange
    _psql(scratch_dsn, "GRANT SELECT (ip_address) ON TABLE public.audit_log TO PUBLIC")
    broadened = await _capability_state(scratch_dsn)
    assert "ip_address" in broadened.readable_columns, "fixture failed to broaden via PUBLIC"
    assert "ip_address" not in broadened.direct_column_grants, (
        "fixture must broaden through PUBLIC, not through a direct journal_app grant"
    )

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode == 0, f"grants.sql failed:\n{repair.stdout}\n{repair.stderr}"
    _assert_intact(await _capability_state(scratch_dsn))


async def test_grants_repair_removes_a_public_table_wide_select(scratch_dsn: str) -> None:
    """A table-wide SELECT to PUBLIC exposes every column; the repair removes it."""
    # Arrange
    _psql(scratch_dsn, "GRANT SELECT ON TABLE public.audit_log TO PUBLIC")
    broadened = await _capability_state(scratch_dsn)
    assert broadened.table_select is True, "fixture failed to broaden table SELECT via PUBLIC"
    assert broadened.direct_table_select is False, (
        "fixture must broaden through PUBLIC, not through a direct journal_app grant"
    )

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode == 0, f"grants.sql failed:\n{repair.stdout}\n{repair.stderr}"
    _assert_intact(await _capability_state(scratch_dsn))


@pytest.mark.parametrize(
    ("mutation", "description"),
    [
        (
            f"GRANT SELECT (ip_address) ON TABLE public.audit_log TO {_PARENT_ROLE}",
            "column SELECT on ip_address",
        ),
        (
            f"GRANT SELECT ON TABLE public.audit_log TO {_PARENT_ROLE}",
            "table SELECT",
        ),
    ],
)
async def test_grants_repair_fails_closed_on_an_inherited_role_grant(
    scratch_dsn: str, mutation: str, description: str
) -> None:
    """Repair aborts rather than reporting success it cannot deliver.

    Clearing this would mean revoking from, or unmembering, a role grants.sql
    does not own -- a wider blast radius than the exposure. So the run raises,
    the transaction rolls back, and the operator gets the grantee by name.
    """
    # Arrange
    _psql(scratch_dsn, mutation)
    broadened = await _capability_state(scratch_dsn)
    assert broadened.inherited_select_grants, f"fixture failed to broaden via {_PARENT_ROLE}"

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode != 0, (
        f"grants.sql reported success while {description} reached journal_app "
        f"through {_PARENT_ROLE}:\n{repair.stdout}\n{repair.stderr}"
    )
    combined = repair.stdout + repair.stderr
    assert _PARENT_ROLE in combined, (
        f"the abort must name the offending grantee so it can be resolved:\n{combined}"
    )


@pytest.mark.parametrize(
    ("policy_sql", "policy_name"),
    [
        (
            f"CREATE POLICY audit_log_probe_parent_select ON public.audit_log "
            f"FOR SELECT TO {_PARENT_ROLE} USING (true)",
            "audit_log_probe_parent_select",
        ),
        (
            f"CREATE POLICY audit_log_probe_parent_all ON public.audit_log "
            f"FOR ALL TO {_PARENT_ROLE} USING (true)",
            "audit_log_probe_parent_all",
        ),
        (
            "CREATE POLICY audit_log_probe_public_select ON public.audit_log "
            "FOR SELECT USING (true)",
            "audit_log_probe_public_select",
        ),
    ],
)
async def test_grants_repair_fails_closed_on_an_extra_applicable_policy(
    scratch_dsn: str, policy_sql: str, policy_name: str
) -> None:
    """A policy reaching journal_app via a parent role or PUBLIC aborts the repair.

    Measured on PostgreSQL 17, a policy scoped TO a parent role applies to
    journal_app even when journal_app is NOINHERIT, and permissive policies for
    the same command are OR-ed -- so this widens the read surface to every row.
    """
    # Arrange
    _psql(scratch_dsn, policy_sql)
    broadened = await _capability_state(scratch_dsn)
    assert policy_name in [p.name for p in broadened.select_policies], (
        f"fixture policy {policy_name} is not in the SELECT-applicable set: "
        f"{[p.name for p in broadened.select_policies]}"
    )

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode != 0, (
        f"grants.sql reported success while {policy_name} widened journal_app's "
        f"read surface:\n{repair.stdout}\n{repair.stderr}"
    )
    assert policy_name in repair.stdout + repair.stderr, (
        f"the abort must name the offending policy:\n{repair.stdout}\n{repair.stderr}"
    )


async def test_grants_repair_fails_closed_when_both_predicates_widen_together(
    scratch_dsn: str,
) -> None:
    """Widening both predicates to ``true`` keeps them mirrored and must still abort.

    This is the mutation an equality-only check cannot see: the SELECT USING still
    equals the INSERT WITH CHECK, so every mirror assertion stays green while
    every audit row becomes readable through journal_app.
    """
    # Arrange
    _psql(
        scratch_dsn,
        f"ALTER POLICY {_SELECT_POLICY_NAME} ON public.audit_log USING (true); "
        f"ALTER POLICY {_INSERT_POLICY_NAME} ON public.audit_log WITH CHECK (true)",
    )
    mirrored = _psql(
        scratch_dsn,
        # Both interpolated policy names are module constants, not input.
        "SELECT (SELECT pg_get_expr(polqual, polrelid) FROM pg_policy "  # noqa: S608
        f"WHERE polrelid = 'public.audit_log'::regclass AND polname = '{_SELECT_POLICY_NAME}') "
        "= (SELECT pg_get_expr(polwithcheck, polrelid) FROM pg_policy "
        f"WHERE polrelid = 'public.audit_log'::regclass AND polname = '{_INSERT_POLICY_NAME}')",
    )
    assert mirrored == "t", "fixture must leave the two predicates mirrored to be the right test"

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    assert repair.returncode != 0, (
        "grants.sql reported success while both self-only predicates were widened "
        f"to true:\n{repair.stdout}\n{repair.stderr}"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "GRANT SELECT (ip_address) ON TABLE public.audit_log TO PUBLIC",
        "GRANT SELECT ON TABLE public.audit_log TO PUBLIC",
        f"GRANT SELECT (ip_address) ON TABLE public.audit_log TO {_PARENT_ROLE}",
        f"GRANT SELECT ON TABLE public.audit_log TO {_PARENT_ROLE}",
        f"CREATE POLICY audit_log_probe_wide ON public.audit_log "
        f"FOR SELECT TO {_PARENT_ROLE} USING (true)",
        "CREATE POLICY audit_log_probe_pub ON public.audit_log FOR SELECT USING (true)",
        f"ALTER POLICY {_SELECT_POLICY_NAME} ON public.audit_log USING (true); "
        f"ALTER POLICY {_INSERT_POLICY_NAME} ON public.audit_log WITH CHECK (true)",
    ],
)
async def test_no_successful_repair_leaves_a_widened_app_read_surface(
    scratch_dsn: str, mutation: str
) -> None:
    """The invariant across every widening: rc=0 implies the narrow read surface.

    Either the repair removes the exposure (rc=0, and a journal_app session with
    no actor identity set is back to zero rows and the five columns), or it
    aborts (rc!=0). A rc=0 run whose end state is still wider on either axis is
    the failure this asserts against, independent of WHICH statement in
    grants.sql produces the outcome.
    """
    # Arrange
    await _plant_foreign_audit_row(scratch_dsn)
    _psql(scratch_dsn, mutation)
    widened = await _app_read_surface(scratch_dsn)
    assert not widened.is_narrow, (
        f"fixture did not widen either exposure axis, so a later narrow result "
        f"proves nothing: {widened!r}"
    )

    # Act
    repair = _replay_grants(scratch_dsn)

    # Assert
    if repair.returncode == 0:
        after = await _app_read_surface(scratch_dsn)
        assert after.is_narrow, (
            "grants.sql reported success while journal_app's read surface was "
            f"still wider than the self-only five-column contract: {after!r}, "
            f"after: {mutation}"
        )
        _assert_intact(await _capability_state(scratch_dsn))


# ---------------------------------------------------------------------------
# Post-deploy invariant script. The failure-tag diffing rationale lives in
# tests/fixtures/db_invariants.py, which owns the harness.
# ---------------------------------------------------------------------------


async def test_invariant_script_flags_a_dropped_select_policy(scratch_dsn: str) -> None:
    # Arrange
    _assert_clean_of(scratch_dsn, _SELECT_POLICY_NAME)
    _psql(scratch_dsn, f"DROP POLICY {_SELECT_POLICY_NAME} ON public.audit_log")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, _SELECT_POLICY_NAME), (
        f"verifier did not flag the missing SELECT policy: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_revoked_conflict_target_column(scratch_dsn: str) -> None:
    # Arrange
    _assert_clean_of(scratch_dsn, "column_grant audit_log")
    _psql(scratch_dsn, "REVOKE SELECT (metadata) ON TABLE public.audit_log FROM journal_app")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    flagged = _tags_mentioning(degraded, "column_grant audit_log.metadata")
    assert flagged, f"verifier did not flag the revoked column: {sorted(degraded)}"


async def test_invariant_script_flags_an_extra_granted_column(scratch_dsn: str) -> None:
    # Arrange
    _assert_clean_of(scratch_dsn, "column_grant audit_log")
    _psql(scratch_dsn, "GRANT SELECT (ip_address) ON TABLE public.audit_log TO journal_app")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    flagged = _tags_mentioning(degraded, "ip_address")
    assert flagged, f"verifier did not flag the extra granted column: {sorted(degraded)}"


async def test_invariant_script_flags_a_predicate_broadened_to_using_true(
    scratch_dsn: str,
) -> None:
    """``USING (true)`` keeps the name, cmd, role and posture -- only the surface widens.

    This is the mutation an existence-only check cannot see: every row in
    audit_log becomes readable through journal_app while the policy still looks
    correct in ``pg_policies.policyname``.
    """
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_predicate")
    _psql(scratch_dsn, f"ALTER POLICY {_SELECT_POLICY_NAME} ON public.audit_log USING (true)")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_predicate"), (
        f"verifier did not flag the broadened USING predicate: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_second_permissive_select_policy(
    scratch_dsn: str,
) -> None:
    """A second permissive SELECT policy OR-widens the read surface to every row."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_set audit_log")
    _psql(
        scratch_dsn,
        "CREATE POLICY audit_log_app_select_wide ON public.audit_log "
        "FOR SELECT TO journal_app USING (true)",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_set audit_log"), (
        f"verifier did not flag the extra permissive SELECT policy: {sorted(degraded)}"
    )


async def test_invariant_script_flags_an_extra_for_all_policy(scratch_dsn: str) -> None:
    """A ``FOR ALL`` policy applies to SELECT too, so it widens the read surface."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_set audit_log")
    _psql(
        scratch_dsn,
        "CREATE POLICY audit_log_app_all_wide ON public.audit_log "
        "FOR ALL TO journal_app USING (true)",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_set audit_log"), (
        f"verifier did not flag the extra FOR ALL policy: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_restrictive_select_policy(scratch_dsn: str) -> None:
    """Recreated AS RESTRICTIVE the policy ANDs instead of ORs -- the dedup read dies."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_permissive")
    _psql(
        scratch_dsn,
        f"DROP POLICY {_SELECT_POLICY_NAME} ON public.audit_log; "
        f"CREATE POLICY {_SELECT_POLICY_NAME} ON public.audit_log AS RESTRICTIVE "
        "FOR SELECT TO journal_app USING ("
        "actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')) "
        "AND actor_id <> '' AND actor_type = 'user')",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_permissive"), (
        f"verifier did not flag the restrictive policy posture: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_wrong_role_scoped_policy(scratch_dsn: str) -> None:
    """Recreated ``TO PUBLIC`` the policy exposes the read surface beyond journal_app."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_roles")
    _psql(
        scratch_dsn,
        f"DROP POLICY {_SELECT_POLICY_NAME} ON public.audit_log; "
        f"CREATE POLICY {_SELECT_POLICY_NAME} ON public.audit_log "
        "FOR SELECT USING ("
        "actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')) "
        "AND actor_id <> '' AND actor_type = 'user')",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_roles"), (
        f"verifier did not flag the TO PUBLIC policy scope: {sorted(degraded)}"
    )


async def test_invariant_script_flags_a_wrong_command_policy(scratch_dsn: str) -> None:
    """Recreated ``FOR ALL`` the policy no longer pins the read surface to SELECT."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_cmd")
    _psql(
        scratch_dsn,
        f"DROP POLICY {_SELECT_POLICY_NAME} ON public.audit_log; "
        f"CREATE POLICY {_SELECT_POLICY_NAME} ON public.audit_log "
        "FOR ALL TO journal_app USING ("
        "actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')) "
        "AND actor_id <> '' AND actor_type = 'user')",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_cmd"), (
        f"verifier did not flag the wrong policy command: {sorted(degraded)}"
    )


# ---------------------------------------------------------------------------
# Verifier: effective exposure through PUBLIC and inherited roles.
#
# Direct-polroles matching and journal_app-scoped REVOKEs both miss these: the
# grant or policy never names journal_app, yet reaches it through role
# membership. The verifier resolves applicability with pg_has_role (plus an
# explicit arm for PUBLIC, which pg_has_role does not accept as an argument).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grantee",
    ["PUBLIC", _PARENT_ROLE],
    ids=["public", "inherited_role"],
)
async def test_invariant_script_flags_an_extra_column_grant_reaching_the_app_role(
    scratch_dsn: str, grantee: str
) -> None:
    """An extra column readable via PUBLIC or a parent role is flagged with its grantee."""
    # Arrange
    _assert_clean_of(scratch_dsn, "inherited_grant audit_log")
    _psql(scratch_dsn, f"GRANT SELECT (ip_address) ON TABLE public.audit_log TO {grantee}")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    flagged = _tags_mentioning(degraded, "inherited_grant audit_log")
    assert flagged, f"verifier did not flag the {grantee} column grant: {sorted(degraded)}"
    assert any(grantee in tag and "ip_address" in tag for tag in flagged), (
        f"the flagged tag must name the grantee and the column: {sorted(flagged)}"
    )


@pytest.mark.parametrize(
    "grantee",
    ["PUBLIC", _PARENT_ROLE],
    ids=["public", "inherited_role"],
)
async def test_invariant_script_flags_a_table_wide_grant_reaching_the_app_role(
    scratch_dsn: str, grantee: str
) -> None:
    """Table-wide SELECT via PUBLIC or a parent role exposes every audit column."""
    # Arrange
    _assert_clean_of(scratch_dsn, "inherited_grant audit_log")
    _psql(scratch_dsn, f"GRANT SELECT ON TABLE public.audit_log TO {grantee}")

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    flagged = _tags_mentioning(degraded, "inherited_grant audit_log")
    assert flagged, f"verifier did not flag the {grantee} table grant: {sorted(degraded)}"
    assert any(grantee in tag and "table SELECT" in tag for tag in flagged), (
        f"the flagged tag must name the grantee: {sorted(flagged)}"
    )


@pytest.mark.parametrize(
    "policy_sql",
    [
        f"CREATE POLICY audit_log_probe_parent_select ON public.audit_log "
        f"FOR SELECT TO {_PARENT_ROLE} USING (true)",
        f"CREATE POLICY audit_log_probe_parent_all ON public.audit_log "
        f"FOR ALL TO {_PARENT_ROLE} USING (true)",
    ],
    ids=["select_policy", "for_all_policy"],
)
async def test_invariant_script_flags_a_policy_scoped_to_an_inherited_role(
    scratch_dsn: str, policy_sql: str
) -> None:
    """A policy TO a parent role applies to journal_app and OR-widens its read surface.

    Measured on PostgreSQL 17 this holds even with journal_app NOINHERIT, so
    matching polroles against journal_app alone would report green.
    """
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_set audit_log")
    _psql(scratch_dsn, policy_sql)

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert _tags_mentioning(degraded, "policy_set audit_log"), (
        f"verifier did not flag the inherited-role policy: {sorted(degraded)}"
    )


async def test_invariant_script_flags_both_predicates_widened_together(
    scratch_dsn: str,
) -> None:
    """Both predicates widened to ``true`` stay mirrored -- each literal pin must fail.

    The mirror comparison passes here by construction. Only pinning each predicate
    to the literal self-only contract catches it, so BOTH pins are asserted red.
    """
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_predicate")
    _psql(
        scratch_dsn,
        f"ALTER POLICY {_SELECT_POLICY_NAME} ON public.audit_log USING (true); "
        f"ALTER POLICY {_INSERT_POLICY_NAME} ON public.audit_log WITH CHECK (true)",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    flagged = _tags_mentioning(degraded, "policy_predicate")
    assert any(_SELECT_POLICY_NAME in tag and "USING" in tag for tag in flagged), (
        f"verifier did not flag the widened SELECT USING against the contract: {sorted(degraded)}"
    )
    assert any(_INSERT_POLICY_NAME in tag and "WITH CHECK" in tag for tag in flagged), (
        f"verifier did not flag the widened INSERT WITH CHECK against the contract: "
        f"{sorted(degraded)}"
    )
    # The mirror check cannot see this mutation -- that is precisely why the two
    # literal pins above exist.
    assert not _tags_mentioning(degraded, "mirrors"), (
        "the mirror check should stay green here; if it fails, this test is no "
        f"longer exercising the paired-widening case: {sorted(degraded)}"
    )


async def test_invariant_script_flags_only_the_insert_predicate_widened(
    scratch_dsn: str,
) -> None:
    """The INSERT WITH CHECK is pinned in its own right, not only via the mirror."""
    # Arrange
    _assert_clean_of(scratch_dsn, "policy_predicate")
    _psql(
        scratch_dsn,
        f"ALTER POLICY {_INSERT_POLICY_NAME} ON public.audit_log WITH CHECK (true)",
    )

    # Act
    degraded = _verifier_failure_tags(scratch_dsn)

    # Assert
    assert any(
        _INSERT_POLICY_NAME in tag and "WITH CHECK" in tag
        for tag in _tags_mentioning(degraded, "policy_predicate")
    ), f"verifier did not flag the widened INSERT WITH CHECK: {sorted(degraded)}"


# ---------------------------------------------------------------------------
# The scratch fixture's own cluster-state hygiene.
#
# scratch_dsn CONVERGES cluster-global role state (otel_ro, the parent-probe
# membership, journal_admin's NOCREATEROLE) so the migration chain and the
# verifier see what a deploy would have left. All of that outlives the throwaway
# database, so the fixture must restore it -- otherwise every later test in the
# session inherits the drift, and a leaked BYPASSRLS or NOCREATEROLE silently
# voids assertions in unrelated modules.
#
# Both topologies are exercised, because they restore differently: on a FRESH
# cluster otel_ro and the parent role do not exist and the fixture created them,
# so they must be dropped; on a PRECONFIGURED one otel_ro already exists as a
# shared role and must SURVIVE with its attributes intact.
# ---------------------------------------------------------------------------


async def _scratch_tracked_state(dsn: str) -> ClusterRoleState:
    conn = await asyncpg.connect(_maintenance_dsn(dsn), timeout=5)
    try:
        return await capture_cluster_state(conn, _SCRATCH_TRACKED_ROLES)
    finally:
        await conn.close()


async def _role_present(dsn: str, role: str) -> bool:
    conn = await asyncpg.connect(_maintenance_dsn(dsn), timeout=5)
    try:
        return bool(
            await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)", role)
        )
    finally:
        await conn.close()


async def test_scratch_fixture_leaves_no_cluster_role_drift(scratch_dsn: str) -> None:
    """Inside the fixture, the convergence it promises is actually in place.

    The paired teardown assertion lives in the test below, which reads the
    cluster AFTER the fixture has torn down. This half exists so that test cannot
    pass vacuously: if convergence never happened, "no drift afterwards" would be
    free.
    """
    # Arrange / Act
    state = await _scratch_tracked_state(scratch_dsn)

    # Assert
    assert _PARENT_PROBE_ROLE in state.attributes, (
        "the fixture must have provisioned the parent probe role, or the teardown "
        "assertion in the sibling test proves nothing"
    )
    edges = {(edge.granted, edge.member) for edge in state.memberships}
    assert (_PARENT_PROBE_ROLE, _APP_ROLE) in edges, (
        f"the parent-role membership the exposure tests need is missing: {sorted(edges)}"
    )
    assert state.attributes["journal_admin"].can_create_role is False, (
        "the fixture converges journal_admin to NOCREATEROLE; without that this "
        "test is not observing the state whose restoration matters"
    )


@pytest.mark.usefixtures("_rls_provisioned")
async def test_scratch_fixture_restores_cluster_state_on_a_fresh_topology() -> None:
    """On a cluster without otel_ro or the parent role, teardown drops both.

    "Fresh" is established by observation, not assumed: if the roles already
    exist this is the preconfigured topology and the sibling test covers it.
    """
    # Arrange
    before = await _scratch_tracked_state(RLS_BOOTSTRAP_URL)
    if _PARENT_PROBE_ROLE in before.attributes:
        pytest.skip(
            "the parent probe role already exists, so this cluster is not in the "
            "fresh topology; the preconfigured case is covered separately"
        )

    # Act
    await _drive_scratch_fixture_once()

    # Assert
    after = await _scratch_tracked_state(RLS_BOOTSTRAP_URL)
    assert _PARENT_PROBE_ROLE not in after.attributes, (
        "a role the fixture created must be dropped on teardown"
    )
    assert after.attributes == before.attributes, (
        f"teardown left role attributes changed:\nbefore={before.attributes}\n"
        f"after={after.attributes}"
    )
    assert after.memberships == before.memberships, (
        f"teardown left memberships changed:\n"
        f"added={sorted(after.memberships - before.memberships)}\n"
        f"removed={sorted(before.memberships - after.memberships)}"
    )


@pytest.mark.usefixtures("_rls_provisioned")
async def test_scratch_fixture_preserves_a_preexisting_shared_role() -> None:
    """On a cluster where otel_ro already exists, teardown must NOT drop it.

    A deployment's otel_ro carries a password and a pg_monitor membership the
    harness never captures, so dropping and recreating it would destroy state.
    The role is planted with a distinctive, non-default attribute set, and both
    its survival and its exact attributes are asserted afterwards.
    """
    # Arrange -- plant otel_ro as a pre-existing shared role with CREATEROLE set,
    # which the fixture's convergence does not touch and must not clear. The plant
    # and its removal each take the lock briefly and RELEASE it, because the
    # fixture under test acquires the same key on the same maintenance database:
    # holding it across the drive would self-deadlock on a separate connection.
    pristine = await _plant_preexisting_otel_ro()
    try:
        before = await _scratch_tracked_state(RLS_BOOTSTRAP_URL)
        assert before.attributes["otel_ro"].can_create_role is True, (
            "the planted attribute did not take, so its preservation is unproven"
        )

        # Act
        await _drive_scratch_fixture_once()

        # Assert
        after = await _scratch_tracked_state(RLS_BOOTSTRAP_URL)
    finally:
        await _restore_after_plant(pristine)

    # OUTER assertion: this test's own plant must be fully undone. The inner
    # before/after pair proves the FIXTURE preserved a pre-existing role; this pair
    # proves the TEST left no drift of its own -- without it, a plant that leaked
    # would silently become the next test's baseline.
    outer_after = await _scratch_tracked_state(RLS_BOOTSTRAP_URL)
    assert outer_after.attributes == pristine.attributes, (
        "the test's own plant was not undone, so it leaks cluster-global drift into "
        f"the rest of the session:\npristine={pristine.attributes}\n"
        f"after={outer_after.attributes}"
    )
    assert outer_after.memberships == pristine.memberships, (
        "the test's own plant left membership drift:\n"
        f"added={sorted(outer_after.memberships - pristine.memberships)}\n"
        f"removed={sorted(pristine.memberships - outer_after.memberships)}"
    )

    assert "otel_ro" in after.attributes, (
        "a pre-existing shared role must survive the fixture -- the harness cannot "
        "recreate its password or memberships"
    )
    assert after.attributes["otel_ro"] == before.attributes["otel_ro"], (
        "the shared role's attributes must be restored exactly, including the "
        f"CREATEROLE the fixture's convergence never set:\n"
        f"before={before.attributes['otel_ro']}\nafter={after.attributes['otel_ro']}"
    )
    assert after.memberships == before.memberships, (
        f"teardown left memberships changed:\n"
        f"added={sorted(after.memberships - before.memberships)}\n"
        f"removed={sorted(before.memberships - after.memberships)}"
    )


async def _plant_preexisting_otel_ro() -> ClusterRoleState:
    """Give otel_ro a distinctive attribute the fixture must preserve.

    Returns the pre-plant snapshot so the caller can undo it. CREATEROLE is chosen
    because the fixture's convergence never writes it, so preserving it can only
    happen by genuinely restoring rather than by coincidentally re-converging.
    """
    async with cluster_role_lock(RLS_BOOTSTRAP_URL) as lock_conn:
        pristine = await capture_cluster_state(lock_conn, _SCRATCH_TRACKED_ROLES)
        await lock_conn.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='otel_ro') "
            "THEN CREATE ROLE otel_ro LOGIN; END IF; END $$;"
        )
        await lock_conn.execute("ALTER ROLE otel_ro WITH LOGIN CREATEROLE")
    return pristine


async def _restore_after_plant(pristine: ClusterRoleState) -> None:
    """Undo the plant, so this test leaks none of the drift it exists to detect.

    The disposable set is derived from ``pristine`` -- the snapshot taken before the
    plant -- not from the unconditional owned list. Passing the unconditional list
    would refuse to drop an otel_ro the PLANT created, leaking it into the rest of
    the session; deriving it here drops exactly that case and still preserves an
    otel_ro that already existed.
    """
    async with cluster_role_lock(RLS_BOOTSTRAP_URL) as lock_conn:
        await restore_cluster_state(
            lock_conn,
            pristine,
            _SCRATCH_TRACKED_ROLES,
            disposable=_scratch_disposable_roles(pristine),
        )


async def _drive_scratch_fixture_once() -> None:
    """Run the scratch fixture's full setup/teardown cycle with an empty body.

    The fixture is driven directly rather than requested, so the calling test can
    snapshot the cluster OUTSIDE it and observe the before/after pair.

    The caller must NOT hold the posture lock across this call on a separate
    connection: the fixture takes the same key on the same maintenance database,
    and pg_advisory_lock is re-entrant only within one session. Callers that need a
    surrounding hold use the same connection, which this helper does not touch.
    """
    generator = scratch_dsn.__wrapped__()  # type: ignore[attr-defined]
    await anext(generator)
    with pytest.raises(StopAsyncIteration):
        await anext(generator)


# ---------------------------------------------------------------------------
# Credential hygiene.
#
# This fixture used to ALTER journal_app's password on every setup, reasoning that
# it assigned the same value. That still rewrote a CLUSTER-GLOBAL secret on a
# cluster it does not own, raced every concurrent session on the same pg_authid
# row, and would have destroyed a real credential on a shared or deployed cluster
# -- rolpassword is a hash the harness never captures, so there is nothing to
# restore from. The credential is now READ, never written.
#
# "Never written" is a claim about absence, free unless something can detect a
# write. These controls fingerprint the stored hash and require byte-identity, and
# separately require that the credential still AUTHENTICATES -- a fingerprint match
# alone would miss a rewrite to a different valid hash of the same password.
# ---------------------------------------------------------------------------

_CREDENTIAL_ROLES = ("journal_app", "journal_admin")


async def _credential_fingerprints(dsn: str) -> dict[str, str]:
    """An md5 fingerprint of each shared role's stored password hash.

    The FINGERPRINT is compared, never the hash: a mismatch is all a test needs,
    and the digest cannot be used to authenticate. ``unset`` distinguishes "no
    password" from any hash, so clearing a credential is caught as readily as
    replacing one.

    This is a STRICT detector, which is what makes it worth having. Under
    ``password_encryption = scram-sha-256`` the stored verifier embeds a random
    salt, so re-assigning the SAME password produces a DIFFERENT hash -- measured
    on PostgreSQL 17. A fixture that "harmlessly" re-asserts the expected password
    therefore cannot hide behind an unchanged fingerprint.
    """
    conn = await asyncpg.connect(_maintenance_dsn(dsn), timeout=5)
    try:
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
            list(_CREDENTIAL_ROLES),
        )
    finally:
        await conn.close()
    return {str(row["rolname"]): str(row["fingerprint"]) for row in rows}


async def test_scratch_fixture_leaves_credentials_usable(scratch_dsn: str) -> None:
    """Inside the fixture, the provisioned credential still authenticates.

    Paired with the bracketing test below: this half proves the credential is
    present and working DURING the fixture, so a before/after fingerprint match
    cannot pass on two identically-broken readings.
    """
    # Arrange
    during = await _credential_fingerprints(scratch_dsn)

    # Act
    app = await _connect_as_provisioned_app_role(scratch_dsn)
    await app.close()

    # Assert
    assert all(value != "unset" for value in during.values()), (
        f"the shared roles must still have credentials after setup: {during}"
    )


@pytest.mark.usefixtures("_rls_provisioned")
async def test_scratch_fixture_never_rewrites_a_shared_credential() -> None:
    """Bracket the whole scratch lifecycle: credentials in, credentials out.

    Driven directly rather than requested, so the fingerprints are taken OUTSIDE
    the fixture and can observe the before/after pair.

    SCOPE. This measures the SCRATCH FIXTURE only. The session fixture's
    ``_set_role_passwords`` legitimately converges the test-environment credential
    once per session, and under scram the stored verifier is salted, so that write
    moves the fingerprint by design. The bracket here therefore excludes session
    setup -- ``_rls_provisioned`` has already run -- so any movement it observes is
    attributable to the fixture under test and to nothing else. The companion
    assertion about the environment's convergence being CORRECT is logical
    authentication, covered by the sibling test below.
    """
    # Arrange
    before = await _credential_fingerprints(RLS_BOOTSTRAP_URL)
    assert all(value != "unset" for value in before.values()), (
        f"credentials must be provisioned before the exercise: {before}"
    )

    # Act
    await _drive_scratch_fixture_once()

    # Assert
    after = await _credential_fingerprints(RLS_BOOTSTRAP_URL)
    assert after == before, (
        "the scratch fixture changed a shared role credential. Passwords are "
        "cluster-global and the harness does not capture the hash, so a rewrite is "
        f"unrecoverable:\n before={before}\n after={after}"
    )


@pytest.mark.usefixtures("_rls_provisioned")
async def test_scratch_fixture_preserves_credentials_through_an_exception() -> None:
    """A body that raises must not cost a credential either.

    The restore path runs on both the happy and the failing path, so a rewrite
    hiding in the exception route would be just as fatal and far less visible.
    """
    # Arrange
    before = await _credential_fingerprints(RLS_BOOTSTRAP_URL)

    # Act
    generator = scratch_dsn.__wrapped__()  # type: ignore[attr-defined]
    await anext(generator)
    with pytest.raises(RuntimeError, match="induced"):
        await generator.athrow(RuntimeError("induced"))

    # Assert
    after = await _credential_fingerprints(RLS_BOOTSTRAP_URL)
    assert after == before, (
        "an exception inside the scratch fixture cost a shared role credential:\n"
        f" before={before}\n after={after}"
    )


@pytest.mark.usefixtures("_rls_provisioned")
async def test_provisioned_credentials_authenticate_after_the_scratch_lifecycle() -> None:
    """LOGICAL authentication survives, which is the property that actually matters.

    Kept separate from the fingerprint bracket on purpose. Under
    ``password_encryption = scram-sha-256`` the stored verifier is salted, so the
    session fixture's legitimate one-time convergence changes the hash while leaving
    the PASSWORD unchanged. A fingerprint comparison alone therefore cannot
    distinguish "someone rewrote the credential" from "the environment converged it
    as designed" unless the measurement window excludes that convergence.

    This test asks the question the fingerprint cannot: after the scratch lifecycle,
    do both roles still authenticate with the credentials the environment
    provisioned? That is unaffected by salting and is what every pool in the run
    depends on.
    """
    # Arrange
    await _drive_scratch_fixture_once()

    # Act / Assert -- an actual login per role, not a catalog read.
    app = await _connect_as_provisioned_app_role(RLS_BOOTSTRAP_URL)
    try:
        assert str(await app.fetchval("SELECT current_user")) == "journal_app", (
            "the app connection authenticated as an unexpected role"
        )
    finally:
        await app.close()

    admin = await asyncpg.connect(RLS_ADMIN_URL, timeout=5)
    try:
        assert str(await admin.fetchval("SELECT current_user")) == "journal_admin", (
            "the admin connection authenticated as an unexpected role"
        )
    finally:
        await admin.close()
