"""Cluster-global role state: the maintenance-database lock, snapshot, and restore.

Roles are CLUSTER-GLOBAL but PostgreSQL advisory locks are DATABASE-SCOPED: two
sessions holding "the same" key in two different databases do not contend at all.
A role-state fixture that locks its own working database therefore serializes
nothing against a session working in a sibling database, while both mutate the
same ``pg_authid`` rows.

So every lock here is taken on ONE fixed maintenance database -- the cluster's
``postgres`` database, derived from the working DSN with every other connection
setting preserved -- which makes the key cluster-wide in effect.

Contention is BOUNDED. ``lock_timeout`` turns a lock another session already
holds into a prompt error rather than an unbounded hang, and that error is
re-raised as a pytest failure naming the blocking session's backend, database and
application name.

RESTORE IS EXACT AND NARROW. The harness snapshots and restores precisely the
attributes and membership options it mutates, and nothing else: no password, no
``rolconfig``, no connection limit, no ``rolvaliduntil``, no grantor provenance is
read or written. A pre-existing shared role is NEVER dropped and never recreated
-- three attributes cannot reconstitute a role's complete state, so a test that
needs a role to be absent uses a fixture-owned disposable role instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest

# One fixed key for the production posture lock, so every session contends on the
# same one rather than each picking its own. Session-scoped: released on
# disconnect even if the process dies mid-test.
POSTURE_ADVISORY_LOCK_KEY = 8_090_473_310_114_552

# Bounded wait. Sized against the real worst case rather than a guess: a single
# holder runs the verifier (several seconds of psql round trips) inside the hold,
# and a suite of ~35 such tests can be queued behind two siblings doing the same.
# Measured on PostgreSQL 17 with three concurrent suites, holds queued up past 30s
# and produced spurious "another session is mutating" failures on work that was
# progressing normally. Still bounded, so a genuinely stuck holder fails the suite
# instead of hanging it.
LOCK_TIMEOUT_MS = 600_000

# The cluster's maintenance database. Always present, never the target of the
# migrations these suites run, so locking here cannot interact with the work.
MAINTENANCE_DATABASE = "postgres"

# Attributes the harness mutates, and therefore the exact set it restores.
# Deliberately NOT included: rolpassword, rolconfig, rolconnlimit, rolvaliduntil,
# rolreplication, rolcreatedb, rolinherit -- the harness never writes them, so
# reading them would invite a restore that rewrites state it never broke.
MUTATED_ATTRIBUTES = ("rolsuper", "rolbypassrls", "rolcanlogin", "rolcreaterole")


def quote_role(role: str) -> str:
    """``role`` as a SQL identifier, doubling embedded double-quotes.

    Every role name this module writes into DDL goes through here. Role names are
    operator-supplied text -- a diagnostics test deliberately plants ones carrying
    a quote and a newline -- and a role name cannot be a bound parameter in DDL,
    so quoting is the only correct handling.
    """
    escaped = role.replace('"', '""')
    return f'"{escaped}"'


def maintenance_dsn(dsn: str) -> str:
    """``dsn`` pointed at the cluster's maintenance database.

    Every other component -- scheme, credentials, host, port, query parameters --
    is preserved, so the lock connection authenticates exactly as the caller's own
    connection does and needs no separate credential.
    """
    return urlunparse(urlparse(dsn)._replace(path=f"/{MAINTENANCE_DATABASE}"))


@dataclass(frozen=True)
class RoleAttributes:
    """Exactly the role attributes this harness mutates."""

    is_superuser: bool
    bypasses_rls: bool
    can_login: bool
    can_create_role: bool

    def as_alter_clause(self) -> str:
        """The ``ALTER ROLE ... WITH`` clause that restores exactly this state."""
        return " ".join(
            (
                "SUPERUSER" if self.is_superuser else "NOSUPERUSER",
                "BYPASSRLS" if self.bypasses_rls else "NOBYPASSRLS",
                "LOGIN" if self.can_login else "NOLOGIN",
                "CREATEROLE" if self.can_create_role else "NOCREATEROLE",
            )
        )


@dataclass(frozen=True, order=True)
class MembershipEdge:
    """One ``pg_auth_members`` row, reduced to what the harness can change.

    Ordered so a failure diff lists edges deterministically; field order puts the
    granted role first, which reads naturally in a sorted report.

    GRANTOR IS PART OF THE IDENTITY. On PostgreSQL 16+ a single (granted, member)
    pair can have SEVERAL rows, one per grantor, each with its own option triple --
    and ``REVOKE granted FROM member`` without ``GRANTED BY`` removes only the row
    whose grantor is the current user. Treating the pair as the identity therefore
    collapsed distinct rows: a restore would revoke one and leave another, or
    re-grant under the wrong grantor and leave an extra row behind. So the identity
    is the full triple, and restore always names ``GRANTED BY``.
    """

    granted: str
    member: str
    grantor: str
    admin_option: bool
    inherit_option: bool
    set_option: bool

    @property
    def identity(self) -> tuple[str, str, str]:
        """The triple that identifies this ``pg_auth_members`` row exactly."""
        return (self.granted, self.member, self.grantor)

    @property
    def options(self) -> tuple[bool, bool, bool]:
        return (self.admin_option, self.inherit_option, self.set_option)


@dataclass(frozen=True)
class ClusterRoleState:
    """The cluster-global facts about a set of roles, as of one moment.

    Restricted to what the harness mutates. ``present`` is the set of roles that
    EXISTED at snapshot time; a role absent then and present later was created by
    the harness and is the only kind it may drop.
    """

    attributes: dict[str, RoleAttributes]
    memberships: frozenset[MembershipEdge]

    @property
    def present(self) -> frozenset[str]:
        return frozenset(self.attributes)

    def edges_by_identity(self) -> dict[tuple[str, str, str], MembershipEdge]:
        """Every captured row, keyed by its exact (granted, member, grantor) triple."""
        return {edge.identity: edge for edge in self.memberships}


_ATTRIBUTE_SQL = f"""
    SELECT rolname, {", ".join(MUTATED_ATTRIBUTES)}
    FROM pg_roles WHERE rolname = ANY($1::text[])
"""  # noqa: S608 -- interpolates only the MUTATED_ATTRIBUTES literal tuple; the role list is a bound parameter

# One row per pg_auth_members row -- NOT one per (granted, member) pair. On
# PostgreSQL 16+ the same pair can appear once per grantor, and each row carries its
# own option triple. The join to pg_roles for the grantor is INNER, because a row
# whose grantor cannot be resolved cannot be restored with GRANTED BY and is
# reported rather than silently reduced.
_MEMBERSHIP_SQL = """
    SELECT g.rolname AS granted, m.rolname AS member, gr.rolname AS grantor,
           am.admin_option, am.inherit_option, am.set_option
    FROM pg_auth_members am
    JOIN pg_roles g ON g.oid = am.roleid
    JOIN pg_roles m ON m.oid = am.member
    JOIN pg_roles gr ON gr.oid = am.grantor
    WHERE g.rolname = ANY($1::text[]) OR m.rolname = ANY($1::text[])
"""


async def capture_cluster_state(conn: asyncpg.Connection, roles: Sequence[str]) -> ClusterRoleState:
    """Read the mutated-attribute and membership state of ``roles``.

    Membership edges are captured for the tracked roles on EITHER side, so an edge
    into or out of a tracked role is in scope.
    """
    tracked = list(roles)
    attribute_rows = await conn.fetch(_ATTRIBUTE_SQL, tracked)
    membership_rows = await conn.fetch(_MEMBERSHIP_SQL, tracked)
    return ClusterRoleState(
        attributes={
            str(row["rolname"]): RoleAttributes(
                is_superuser=bool(row["rolsuper"]),
                bypasses_rls=bool(row["rolbypassrls"]),
                can_login=bool(row["rolcanlogin"]),
                can_create_role=bool(row["rolcreaterole"]),
            )
            for row in attribute_rows
        },
        memberships=frozenset(
            MembershipEdge(
                granted=str(row["granted"]),
                member=str(row["member"]),
                grantor=str(row["grantor"]),
                admin_option=bool(row["admin_option"]),
                inherit_option=bool(row["inherit_option"]),
                set_option=bool(row["set_option"]),
            )
            for row in membership_rows
        ),
    )


async def restore_cluster_state(
    conn: asyncpg.Connection,
    before: ClusterRoleState,
    roles: Sequence[str],
    *,
    disposable: Sequence[str],
) -> None:
    """Return ``roles`` to ``before``, dropping only harness-created disposables.

    A role present in ``before`` is NEVER dropped and never recreated: the four
    attributes captured here cannot reconstitute a role's password, config, or
    connection limit, so a restore that recreated one would silently destroy
    state. Only a role in ``disposable`` that did NOT exist at snapshot time is
    dropped -- the harness made it, so the harness owns it.

    Order: revoke added edges, restore removed ones, reconcile option drift,
    rewrite attributes, then drop harness-created disposables.
    """
    disposable_set = set(disposable)
    current = await capture_cluster_state(conn, roles)

    leaked_shared = (before.present - current.present) - disposable_set
    if leaked_shared:
        pytest.fail(
            "a pre-existing shared role vanished during the test: "
            f"{sorted(leaked_shared)}. The harness cannot recreate one -- its "
            "password, rolconfig and connection limit are not captured -- so "
            "dropping a shared role is forbidden, not something to repair."
        )

    before_edges = before.edges_by_identity()
    current_edges = current.edges_by_identity()

    # Every statement names GRANTED BY. Without it, REVOKE removes only the row
    # whose grantor is the current user and GRANT creates a row attributed to the
    # current user -- either of which leaves a DIFFERENT row than the one intended
    # when several grantors exist for the same pair.
    for identity in sorted(set(current_edges) - set(before_edges)):
        await _revoke_membership(conn, current_edges[identity])

    for identity in sorted(set(before_edges) - set(current_edges)):
        edge = before_edges[identity]
        if edge.granted not in current.present or edge.member not in current.present:
            # Its endpoint is a harness-created role about to be dropped, or a
            # shared role whose disappearance already failed above.
            continue
        await _apply_membership_options(conn, edge)

    for identity in sorted(set(before_edges) & set(current_edges)):
        wanted, found = before_edges[identity], current_edges[identity]
        if wanted.options != found.options:
            await _apply_membership_options(conn, wanted)

    for role, attributes in sorted(before.attributes.items()):
        await conn.execute(f"ALTER ROLE {quote_role(role)} WITH {attributes.as_alter_clause()}")

    for role in sorted((current.present - before.present) & disposable_set):
        await conn.execute(f"DROP ROLE IF EXISTS {quote_role(role)}")


async def _apply_membership_options(conn: asyncpg.Connection, edge: MembershipEdge) -> None:
    """Create or converge ``edge`` exactly: all three options, under its own grantor.

    ``GRANTED BY`` is mandatory. Omitting it attributes the row to the current user,
    which on PostgreSQL 16+ produces an ADDITIONAL row rather than converging the
    intended one -- so a restore would leave both the original and a duplicate.
    """
    options = ", ".join(
        (
            f"ADMIN {str(edge.admin_option).upper()}",
            f"INHERIT {str(edge.inherit_option).upper()}",
            f"SET {str(edge.set_option).upper()}",
        )
    )
    await conn.execute(
        f"GRANT {quote_role(edge.granted)} TO {quote_role(edge.member)} "
        f"WITH {options} GRANTED BY {quote_role(edge.grantor)}"
    )


async def _revoke_membership(conn: asyncpg.Connection, edge: MembershipEdge) -> None:
    """Remove exactly ``edge``'s row, identified by its grantor.

    A bare ``REVOKE`` drops only the row granted by the current user, so an edge
    added by a different grantor would survive it.
    """
    await conn.execute(
        f"REVOKE {quote_role(edge.granted)} FROM {quote_role(edge.member)} "
        f"GRANTED BY {quote_role(edge.grantor)}"
    )


class PostureLockUnavailableError(AssertionError):
    """Another session holds the posture lock; raised instead of hanging."""


@asynccontextmanager
async def cluster_role_lock(
    working_dsn: str,
    key: int = POSTURE_ADVISORY_LOCK_KEY,
    *,
    timeout_ms: int = LOCK_TIMEOUT_MS,
) -> AsyncIterator[asyncpg.Connection]:
    """Hold ``key`` on the MAINTENANCE database for the body, with a bounded wait.

    Locking the maintenance database rather than the working one is what makes the
    lock cluster-wide: advisory locks are database-scoped, so a lock taken in
    ``journal_rls_test`` does not contend with one taken in a scratch database,
    even though both sessions mutate the same cluster-global roles.

    The yielded connection is the LOCK HOLDER, on the maintenance database. Role
    statements are cluster-global, so it is also the right connection to run the
    snapshot, mutation and restore on -- keeping the hold unbroken across all
    three.
    """
    conn = await asyncpg.connect(maintenance_dsn(working_dsn), timeout=5)
    closed = False
    try:
        await conn.execute(f"SET lock_timeout = {timeout_ms}")
        try:
            await conn.execute("SELECT pg_advisory_lock($1)", key)
        except asyncpg.LockNotAvailableError as exc:
            blocker = await _describe_lock_holder(conn, key)
            raise PostureLockUnavailableError(
                f"could not acquire the cluster role lock within "
                f"{timeout_ms}ms: {blocker}. Another session is mutating "
                "cluster-global role state; this is a bounded failure rather than "
                "an unbounded hang so the blocking session is identifiable."
            ) from exc
        body_error: BaseException | None = None
        try:
            yield conn
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            # EVERY cleanup step is attempted, and the BODY's exception wins.
            #
            # A bare `finally: await unlock()` has two defects. If the unlock itself
            # raises, it REPLACES the body's exception -- so the real failure is lost
            # and the report blames the lock -- and the close below never runs, which
            # leaks the connection and its advisory hold for the rest of the session.
            #
            # So both steps always run, their own failures are collected rather than
            # propagated, and a cleanup failure is raised only when the body
            # succeeded. When the body failed, cleanup errors are attached to it as
            # __notes__ so they are still visible without displacing the cause.
            cleanup_errors = await _release_lock_and_close(conn, key)
            if cleanup_errors and body_error is None:
                raise cleanup_errors[0]
            for error in cleanup_errors:
                body_error.add_note(  # type: ignore[union-attr]
                    f"cluster role lock cleanup also failed: {error!r}"
                )
            closed = True
    finally:
        # Reached when ACQUISITION failed, before the block above existed. The
        # success and body-failure paths already closed the connection there, and
        # re-entering close() would re-raise a close failure the block above has
        # already recorded -- displacing the body's exception, which is the whole
        # thing this structure exists to prevent.
        if not closed:
            await _close_quietly(conn)


async def _release_lock_and_close(conn: asyncpg.Connection, key: int) -> list[BaseException]:
    """Unlock and close, attempting BOTH regardless of either failing.

    Returns the failures in the order they happened, so the caller decides whether
    any of them should displace an exception the body already raised.
    """
    errors: list[BaseException] = []
    try:
        await conn.execute("SELECT pg_advisory_unlock($1)", key)
    except BaseException as exc:
        errors.append(exc)
    try:
        await conn.close()
    except BaseException as exc:
        errors.append(exc)
    return errors


async def _close_quietly(conn: asyncpg.Connection) -> None:
    """Close ``conn``, tolerating any failure.

    Used only on the acquisition-failure path, where a ``PostureLockUnavailableError``
    is already propagating and is the more informative failure. Suppressing broadly
    here is deliberate: the alternative is displacing that error with a close
    failure, which tells the reader nothing about why the lock could not be taken.
    """
    with contextlib.suppress(Exception):
        await conn.close()


async def _describe_lock_holder(conn: asyncpg.Connection, key: int) -> str:
    """Name the session holding ``key``, without echoing any credential.

    Only backend pid, database name and application name are read: enough to
    identify which session to look at, nothing that could carry a secret.
    """
    try:
        rows = await conn.fetch(
            """
            SELECT a.pid, a.datname, coalesce(a.application_name, '') AS app
            FROM pg_locks l
            JOIN pg_stat_activity a ON a.pid = l.pid
            WHERE l.locktype = 'advisory'
              AND l.granted
              AND ((l.classid::bigint << 32) | (l.objid::bigint & 4294967295)) = $1
            ORDER BY a.pid
            """,
            key,
        )
    except asyncpg.PostgresError:
        return "holder could not be identified"
    if not rows:
        return "holder is no longer visible in pg_stat_activity"
    return "; ".join(
        f"backend pid={row['pid']} database={row['datname']} application={row['app']!r}"
        for row in rows
    )


async def assert_required_roles_present(conn: asyncpg.Connection, roles: Sequence[str]) -> None:
    """Fail hard when a REACHABLE cluster lacks a role the contract describes.

    Called on the maintenance connection, which proves the server is reachable, so
    a missing role here cannot be confused with an unreachable server. Skipping
    would report green for a contract nothing measured.
    """
    present = {
        str(row["rolname"])
        for row in await conn.fetch(
            "SELECT rolname FROM pg_roles WHERE rolname = ANY($1::text[])", list(roles)
        )
    }
    missing = sorted(set(roles) - present)
    if missing:
        pytest.fail(
            "PostgreSQL is reachable but roles the deployment contract requires do "
            f"not exist: {missing}. They are provisioned by the deploy init script "
            "and by the CI role bootstrap, so their absence is a harness or "
            "environment defect -- skipping here would report green for a contract "
            "nothing measured."
        )
