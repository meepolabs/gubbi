"""Run a per-checkout test database stack in docker, and validate testdb.env.

Standard library only, so it runs on any python3 >= 3.12 with nothing
installed. Every docker call is an argument array; nothing goes through a shell.

Usage::

    testdb.py up        PROFILE
    testdb.py down      PROFILE
    testdb.py status    PROFILE [--require-ready]
    testdb.py env       PROFILE
    testdb.py check-env [--file PATH] [--github-output PATH]
    testdb.py psql-path [--file PATH]
    testdb.py psql-plan [PSQL-ARG ...]
    testdb.py reset     PROFILE [--ci] [--bootstrap-var NAME=VALUE ...]
                        [--migrate-dsn-env NAME ...] [--migrate DIR ARGV... ...]

PROFILE is ``--repo NAME --role ROLE [--role ROLE ...] [--db NAME ...]``:

``--repo``  the consumer repo name (``[a-z][a-z0-9-]*``), part of every
            container name.
``--role``  one container: ``pg`` (runs PGVECTOR_IMAGE) or ``redis`` (runs
            REDIS_IMAGE), optionally with one suffix such as ``pg-disposable``
            for a second cluster. Repeat for each container.
``--db``    a database ``up`` creates on the ``pg`` role (which must then be in
            the profile). Repeat for each database. Suffixed pg roles only get
            the ``postgres`` maintenance database.

For example ``--repo gubbi --role pg --role pg-disposable --db journal_test
--db journal_rls_test``, or ``--repo app --role pg --role redis --db journal``.

The stack belongs to the git checkout the command runs in (``git rev-parse
--show-toplevel`` of the working directory), never to the checkout holding
this script. Each container is named
``testdb-<repo>-<first 8 hex of sha256(toplevel)>-<role>`` and carries the
labels ``ai.gubbi.testdb.{repo,role,owner-uid,checkout,env-sha256}``: checkout
is the full sha256 of the toplevel path, env-sha256 the sha256 of the raw
testdb.env bytes it was started from. A pg container also carries
``ai.gubbi.testdb.auth=scram-sha-256``: it is created with
``POSTGRES_HOST_AUTH_METHOD=scram-sha-256`` and ``POSTGRES_INITDB_ARGS`` setting
scram for local and host connections, so every connection to it, from inside the
container over its unix socket or loopback as much as through the published
port, must authenticate with a password. Ports are published on 127.0.0.1 only,
with a docker-assigned host port read back from ``docker container inspect``.
Ownership is checked on the inspected container, and every later docker call
that changes or execs into it addresses that container's full ID, never the
name, so a container recreated under the name in between is never touched.
testdb.env image pins must be ``repo[:tag]@sha256:<digest>``.

``up``      starts any missing container, restarts a stopped one, waits for
            readiness (pg: ``pg_isready -h 127.0.0.1``; redis: ``redis-cli
            ping``) with bounded attempts (a docker-level failure such as a
            vanished container stops the wait at once), creates the ``--db``
            databases, and
            writes ``.testdb.env`` at the checkout root. It refuses to reuse a
            container whose labels differ in any way, a stale env-sha256
            included; run ``down`` then ``up`` after testdb.env changes. A pg
            container without the current auth label (one created before every
            connection required a password) is refused the same way, by ``up``,
            ``status``, ``env``, ``reset`` and ``psql-plan`` alike: run ``down``
            then ``up`` to recreate it.
``down``    removes this checkout's containers and ``.testdb.env``. It only
            removes a container whose repo, role, owner-uid and checkout labels
            all match; any other container under the name is a refusal and
            nothing is removed. A stale env-sha256 or auth label does not block
            ``down``, and ``down`` never reads testdb.env, so a broken pin file
            cannot block teardown.
``status``  prints ``<role> <container> <state>`` per role. With
            ``--require-ready`` it prints nothing on success, and on any miss
            (container missing, stopped, not ready, foreign labels, stale
            env-sha256 or auth label, docker unavailable, ``.testdb.env``
            missing or not naming exactly the live containers and ports, as
            after a container restart moved its port) prints exactly this line
            to stderr:

                test stack not running: make test-stack-up

``env``     prints the ``.testdb.env`` content for the running stack.
``check-env`` validates testdb.env (default: the one next to this script) and
            prints its ``KEY=value`` lines, or appends them to
            ``--github-output PATH``. Every other subcommand that needs the
            pins loads them through the same validator. Each output line is
            ``KEY=value`` with the value drawn from the validated printable
            charset, which still admits shell metacharacters: consumers must
            never eval or source it. Read specific keys instead (CI appends it
            to ``$GITHUB_OUTPUT``; make and bash consumers pick keys by name).
``psql-path`` prints the absolute path of ``bin/`` next to this script, which
            holds a ``psql`` wrapper running PGVECTOR_IMAGE's psql in docker,
            when no ``psql`` on PATH (``bin/`` itself excluded) reports major
            PG_MAJOR in ``psql --version``; otherwise prints nothing. Consumers
            prepend a non-empty result to PATH.
``psql-plan`` decides how ``bin/psql`` runs one psql command line, and prints
            the decision as NUL-terminated fields. ``exec ID FILE ARG...``: run
            ``psql ARG...`` in the running pg container ID with ``docker exec``,
            stdin read from FILE when FILE is not empty. ``run IMAGE``: run the
            unchanged command line in a throwaway IMAGE (PGVECTOR_IMAGE)
            container. exec is chosen only when every argument is a psql option
            it knows, the target is ``127.0.0.1`` or ``localhost`` on an
            explicit port (``-h``/``-p``, or the host of a ``postgresql://``
            URL with no query; a conninfo may name only dbname and user), no
            ``-o``/``-L`` is given, PGHOSTADDR/PGSERVICE/PGSERVICEFILE are
            unset, and that port is published by a running pg container that
            passes the same label check as ``env`` for the working directory's
            checkout, current env-sha256 and auth label, with exactly one IPv4
            network address. The target is then rewritten to that address on
            port 5432. That rewrite covers only the first connection: SQL on
            stdin or in a file may ``\\connect`` anywhere the container
            reaches, its own unix socket and loopback included, which is why
            every pg_hba rule of a stack container requires scram-sha-256 and
            none trusts. One ``-f PATH`` becomes ``-f -`` with FILE the resolved
            PATH, which must be a regular file under ``deployment/scripts`` or
            ``tools/testdb`` of the repository holding this script (exit 2
            otherwise). Anything else is ``run``. Takes psql's arguments as
            given; no ``--`` separator. Under exec, psql meta-commands that
            name a file (``\\i``, ``\\copy``, ``\\o``) see the container's
            filesystem, not the host's.

``reset``   returns every pg cluster of the profile to a fresh state, in order:
            drops the ``--db`` databases (``WITH (FORCE)``); drops every role
            that is not a superuser, not ``pg_*`` and not the user it connects
            as, after ``REASSIGN OWNED`` to that user and one ``DROP OWNED`` per
            role in every surviving database but ``template0`` (roles are
            cluster-global; a database with ``datallowconn`` false is opened
            for the cleanup and closed again afterwards); recreates the
            ``--db`` databases; runs ``bootstrap.sql`` (``psql -v
            ON_ERROR_STOP=1 -f``) in each of them, since the vector extension
            is per database; then runs each migration against each of them.
            Only ``--db`` databases are dropped or bootstrapped, so a suffixed
            pg role's cluster is left with no non-superuser roles at all. It
            refuses a ``--db`` of ``postgres``, ``template0`` or ``template1``.
            It never starts a container. It prints each step's name and
            duration and never a password: failure output is redacted.

            Targets. Locally (the default) every container in the profile must
            be running with this checkout's full label set and current
            env-sha256, as ``env`` requires; host and port come from ``docker
            container inspect`` of those containers, never from ``.testdb.env``
            or the environment. With ``--ci`` the targets come from the
            environment instead, and only when ``GITHUB_ACTIONS=true``: per pg
            role R, ``TESTDB_R_URL`` (the same names ``.testdb.env`` uses, e.g.
            ``TESTDB_PG_URL``, ``TESTDB_PG_DISPOSABLE_URL``) holds a
            ``postgresql://user[:password]@host[:port]/db`` URL of a superuser
            on that cluster. A host other than ``127.0.0.1`` or ``localhost``
            is a refusal. reset does not check that such a server belongs to
            this checkout or to CI: any loopback server the URL names is
            wiped, so ``--ci`` must never be used outside CI.

            psql is the first ``psql`` on PATH when it reports PG_MAJOR, else
            the ``bin/psql`` wrapper (as ``psql-path`` decides). psql and the
            migrations run with an environment stripped of every ``PG*``
            variable (bootstrap.sql's password variables excepted) and of every
            variable holding a postgres URL. Every psql ``-d`` is a
            ``dbname='...'`` conninfo, so a database name is never read as
            connection settings.

            ``--bootstrap-var NAME=VALUE`` passes a boolean bootstrap.sql
            variable (``admin_createrole``, ``grant_app_to_admin``,
            ``with_otel_ro``); omitted ones take bootstrap.sql's defaults.
            Role passwords reach bootstrap.sql only through its environment
            variables (``JOURNAL_DB_APP_PASSWORD``, ``JOURNAL_DB_ADMIN_PASSWORD``,
            ``PG_OTEL_RO_PASSWORD``), never on an argv.

            ``--migrate DIR ARGV...`` adds one migration command, run with
            working directory DIR (relative to the checkout root) and no
            shell. DIR may lie outside the checkout (a sibling repo's chain);
            it is not a confinement boundary. Repeat it for several chains;
            they run in the order given.
            Every ``--migrate`` must come after all other options, since each
            one runs to the next ``--migrate`` or the end of the arguments.
            Each command sees the database's superuser DSN in every
            ``--migrate-dsn-env NAME`` (at least one is required with
            ``--migrate``). For example::

                testdb.py reset --repo gubbi --role pg --role pg-disposable \\
                    --db journal_test --db journal_rls_test \\
                    --migrate-dsn-env JOURNAL_DB_MIGRATION_URL \\
                    --migrate . poetry run alembic upgrade head

            and two chains, the second from a sibling checkout::

                ... --migrate-dsn-env JOURNAL_DB_ADMIN_URL \\
                    --migrate ../gubbi poetry run alembic upgrade head \\
                    --migrate . poetry run alembic upgrade head

``.testdb.env`` is bash-sourceable (``set -a; . ./.testdb.env``); every value
is single-quoted and contains no quote, space or control character. Per role R
(upper-cased, ``-`` as ``_``): ``TESTDB_R_CONTAINER``, ``TESTDB_R_HOST``,
``TESTDB_R_PORT``, ``TESTDB_R_URL`` (pg: the ``postgres`` database; redis: db
0), and for each ``--db`` D on ``pg``: ``TESTDB_PG_URL_<D>``.

Exit codes: 0 success; 1 the stack is missing, not ready, foreign or stale,
or docker failed; 2 usage error or an invalid testdb.env (the message names
the offending line).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

TESTDB_ENV = Path(__file__).resolve().parent / "testdb.env"
PSQL_WRAPPER_DIR = Path(__file__).resolve().parent / "bin"
STACK_ENV_NAME = ".testdb.env"
NOT_RUNNING = "test stack not running: make test-stack-up"

EXIT_OK = 0
EXIT_STACK = 1
EXIT_INVALID = 2

LABEL_PREFIX = "ai.gubbi.testdb."
REPO_LABEL = f"{LABEL_PREFIX}repo"
ROLE_LABEL = f"{LABEL_PREFIX}role"
UID_LABEL = f"{LABEL_PREFIX}owner-uid"
CHECKOUT_LABEL = f"{LABEL_PREFIX}checkout"
ENV_DIGEST_LABEL = f"{LABEL_PREFIX}env-sha256"
AUTH_LABEL = f"{LABEL_PREFIX}auth"
PG_AUTH_METHOD = "scram-sha-256"
_PG_INITDB_AUTH = f"--auth-local={PG_AUTH_METHOD} --auth-host={PG_AUTH_METHOD}"

LOOPBACK = "127.0.0.1"
PG_USER = "journal"
PG_PASSWORD = "testpass"  # noqa: S105 - throwaway loopback test cluster credential
PG_MAINTENANCE_DB = "postgres"
PRIMARY_PG_ROLE = "pg"

EXIT_TIMED_OUT = 124
# docker exec could not run the probe at all: docker itself failed, the
# command was not executable, or it was not found.
_DOCKER_EXEC_FAILURES = frozenset({125, 126, 127})
_DOCKER_FAILURE_STDERR = re.compile(
    r"no such container|is not running|error response from daemon"
    r"|cannot connect to the docker daemon",
    re.IGNORECASE,
)
# pg_isready: 1 = rejecting connections (starting up), 2 = no response.
_PG_NOT_READY_EXITS = frozenset({1, 2})

READY_ATTEMPTS = 60
READY_INTERVAL_S = 1.0
PROBE_TIMEOUT_S = 5.0
DOCKER_TIMEOUT_S = 60.0

_REPO_RULE = re.compile(r"[a-z][a-z0-9-]{0,39}", re.ASCII)
_ROLE_RULE = re.compile(r"(pg|redis)(-[a-z0-9]{1,20})?", re.ASCII)
_DB_RULE = re.compile(r"[a-z_][a-z0-9_]{0,62}", re.ASCII)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}", re.ASCII)
_HOST_PORT = re.compile(r"[1-9][0-9]{0,4}", re.ASCII)
_SHELL_SAFE_VALUE = re.compile(r"[!-&(-~]+", re.ASCII)
_STACK_ENV_LINE = re.compile(r"([A-Z][A-Z0-9_]*)='([!-&(-~]+)'", re.ASCII)
# A digest is required and the first character cannot be "-", so a pin can
# never reach docker as an option.
_IMAGE_RULE = re.compile(r"[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}", re.ASCII)
# Substrings other tooling on a shared docker host sweeps by; a name carrying
# one could be removed by, or remove, a stack this controller does not own.
_FORBIDDEN_NAME_PARTS = ("gubbi-db-", "-disp-pg")

# testdb.env line rule: comment, blank, or KEY=value with printable ASCII values.
_LINE_RULE = re.compile(r"[A-Z][A-Z0-9_]*=[!-~]+", re.ASCII)
_CONTROL_CHAR = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]", re.ASCII)
_PG_MAJOR_RULE = re.compile(r"[1-9][0-9]*", re.ASCII)
_PSQL_VERSION = re.compile(r"psql \(PostgreSQL\) ([0-9]+)", re.ASCII)

BOOTSTRAP_SQL = Path(__file__).resolve().parent / "bootstrap.sql"
_REPO_ROOT = Path(__file__).resolve().parents[2]
# The only files bin/psql feeds a container: this repository's SQL inputs.
PSQL_SQL_DIRS = (_REPO_ROOT / "deployment" / "scripts", _REPO_ROOT / "tools" / "testdb")
PSQL_PLAN = "psql-plan"
MIGRATE_FLAG = "--migrate"
BOOTSTRAP_BOOL_VARS = frozenset({"admin_createrole", "grant_app_to_admin", "with_otel_ro"})
# bootstrap.sql reads role passwords from these with \getenv; reset redacts them.
BOOTSTRAP_PASSWORD_ENVS = (
    "JOURNAL_DB_APP_PASSWORD",
    "JOURNAL_DB_ADMIN_PASSWORD",
    "PG_OTEL_RO_PASSWORD",
)
PROTECTED_DBS = frozenset({PG_MAINTENANCE_DB, "template0", "template1"})
CI_HOSTS = frozenset({LOOPBACK, "localhost"})
PSQL_TIMEOUT_S = 300.0
MIGRATE_TIMEOUT_S = 1800.0
ROLE_LIST_SQL = (
    "SELECT coalesce(json_agg(json_build_object('name', rolname, 'super', rolsuper)"
    " ORDER BY rolname), '[]') FROM pg_roles"
)
DB_LIST_SQL = (
    "SELECT coalesce(json_agg(json_build_object('name', datname, 'allowconn', datallowconn)"
    " ORDER BY datname), '[]') FROM pg_database WHERE datname <> 'template0'"
)
_BOOL_VALUE = re.compile(r"true|false|on|off|yes|no|1|0", re.ASCII)
_ENV_NAME_RULE = re.compile(r"[A-Z][A-Z0-9_]{0,63}", re.ASCII)
# libpq connection settings (PGHOST, PGSERVICE, PGCONNECT_TIMEOUT, ...) and
# anything else shaped like one.
_LIBPQ_ENV = re.compile(r"PG[A-Z0-9_]*", re.ASCII)
_PG_DSN_VALUE = re.compile(r"\s*postgres(ql)?(\+[a-z0-9]+)?://", re.ASCII | re.IGNORECASE)
_URL_PASSWORD = re.compile(r"(://[^:/@\s]*):[^@\s]*@")


class ConfigError(Exception):
    """Invalid arguments or testdb.env; exits 2."""


class StackError(Exception):
    """The stack is missing, foreign, stale or docker failed; exits 1."""


# ---------------------------------------------------------------------------
# testdb.env: the one validator
# ---------------------------------------------------------------------------


def read_testdb_env(path: Path) -> bytes:
    """Return the file's raw bytes; read_text() would translate CRLF before validation."""
    return path.read_bytes()


def _is_ignorable(line: str) -> bool:
    return line == "" or line.startswith("#")


def _lines(text: str) -> list[str]:
    # Split on LF only: splitlines() would also eat CR, VT, FF and other
    # separators, hiding bytes the shell consumers do not tolerate.
    return text.removesuffix("\n").split("\n")


def _is_valid_line(line: str) -> bool:
    if _CONTROL_CHAR.search(line):
        return False
    return _is_ignorable(line) or _LINE_RULE.fullmatch(line) is not None


def violations(raw: bytes) -> list[str]:
    """Return every way ``raw`` breaks the testdb.env format; empty when valid."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        return [f"non-ASCII byte {raw[exc.start : exc.start + 1]!r} at offset {exc.start}"]
    bad_lines = [line for line in _lines(text) if not _is_valid_line(line)]
    # A line-by-line shell reader drops an unterminated last line without error.
    if raw and not raw.endswith(b"\n"):
        return [*bad_lines, "missing final newline"]
    return bad_lines


def parse_testdb_env(raw: bytes) -> dict[str, str]:
    """Return the KEY=value pairs in file order; raise ValueError on any violation."""
    bad = violations(raw)
    if bad:
        raise ValueError(f"testdb.env breaks the KEY=value format: {bad!r}")
    pairs = [line.split("=", 1) for line in _lines(raw.decode("ascii")) if not _is_ignorable(line)]
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"testdb.env repeats keys: {duplicates!r}")
    return dict(pairs)


@dataclass(frozen=True)
class Pins:
    """Validated testdb.env values plus the sha256 of the bytes they came from."""

    values: Mapping[str, str]
    digest: str

    def image(self, key: str) -> str:
        """Return the image pinned under ``key``; raise ConfigError when absent or malformed."""
        if key not in self.values:
            raise ConfigError(f"testdb.env does not declare {key}")
        image = self.values[key]
        if not _IMAGE_RULE.fullmatch(image):
            raise ConfigError(f"testdb.env {key}={image} must match {_IMAGE_RULE.pattern}")
        return image


def load_pins(path: Path) -> Pins:
    """Read and validate testdb.env; raise ConfigError naming the problem."""
    try:
        raw = read_testdb_env(path)
        values = parse_testdb_env(raw)
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return Pins(values=MappingProxyType(values), digest=hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """Exit status and decoded output of one command."""

    returncode: int
    stdout: str
    stderr: str


def run_command(
    argv: Sequence[str],
    timeout: float | None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    """Run ``argv`` without a shell; a missing binary or a timeout is a failed result.

    ``env`` None inherits this process's environment; otherwise it is the whole
    environment, and its PATH is the one ``argv[0]`` is looked up on.
    """
    try:
        done = subprocess.run(  # noqa: S603 - argument array, never a shell
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=None if env is None else dict(env),
            cwd=cwd,
        )
    except FileNotFoundError:
        return CommandResult(127, "", f"{argv[0]}: command not found")
    except subprocess.TimeoutExpired:
        return CommandResult(EXIT_TIMED_OUT, "", f"{argv[0]}: timed out after {timeout}s")
    return CommandResult(done.returncode, done.stdout, done.stderr)


# ---------------------------------------------------------------------------
# Profile, checkout and naming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """Which containers and databases a consumer repo's stack holds."""

    repo: str
    roles: tuple[str, ...]
    dbs: tuple[str, ...]


@dataclass(frozen=True)
class Checkout:
    """The git checkout a stack belongs to."""

    toplevel: Path
    digest: str

    @property
    def short(self) -> str:
        """First 8 hex digits of the checkout digest, used in container names."""
        return self.digest[:8]


@dataclass(frozen=True)
class Context:
    """Everything a subcommand needs from the outside world."""

    run: Callable[[Sequence[str], float | None], CommandResult]
    sleep: Callable[[float], None]
    uid: int
    checkout: Checkout
    env_file: Path
    # Runs psql and migration commands: (argv, timeout, env, cwd).
    run_env: Callable[[Sequence[str], float | None, Mapping[str, str], Path | None], CommandResult]
    environ: Mapping[str, str]


def make_profile(repo: str, roles: Sequence[str], dbs: Sequence[str]) -> Profile:
    """Validate the consumer's profile arguments; raise ConfigError on any bad value."""
    if not _REPO_RULE.fullmatch(repo):
        raise ConfigError(f"--repo {repo!r} must match {_REPO_RULE.pattern}")
    if not roles:
        raise ConfigError("at least one --role is required")
    for role in roles:
        if not _ROLE_RULE.fullmatch(role):
            raise ConfigError(f"--role {role!r} must match {_ROLE_RULE.pattern}")
    for db in dbs:
        if not _DB_RULE.fullmatch(db):
            raise ConfigError(f"--db {db!r} must match {_DB_RULE.pattern}")
    for flag, values in (("--role", roles), ("--db", dbs)):
        if len(set(values)) != len(values):
            raise ConfigError(f"{flag} values must be unique")
    if dbs and PRIMARY_PG_ROLE not in roles:
        raise ConfigError(f"--db needs --role {PRIMARY_PG_ROLE}")
    return Profile(repo=repo, roles=tuple(roles), dbs=tuple(dbs))


def checkout_from_toplevel(toplevel: Path) -> Checkout:
    """Build a Checkout keyed by the sha256 of the toplevel path."""
    return Checkout(toplevel, hashlib.sha256(str(toplevel).encode("utf-8")).hexdigest())


def find_checkout(run: Callable[[Sequence[str], float | None], CommandResult]) -> Checkout:
    """Return the git checkout of the working directory."""
    result = run(["git", "rev-parse", "--show-toplevel"], DOCKER_TIMEOUT_S)
    toplevel = result.stdout.strip()
    if result.returncode != 0 or not toplevel:
        raise ConfigError(f"not inside a git checkout: {result.stderr.strip()}")
    return checkout_from_toplevel(Path(toplevel))


def role_kind(role: str) -> str:
    """Return ``pg`` or ``redis`` for a validated role name."""
    return role.split("-", 1)[0]


def container_name(profile: Profile, checkout: Checkout, role: str) -> str:
    """Return the container name for one role of this checkout's stack."""
    name = f"testdb-{profile.repo}-{checkout.short}-{role}"
    for part in _FORBIDDEN_NAME_PARTS:
        if part in name:
            raise ConfigError(f"container name {name!r} contains {part!r}")
    return name


def identity_labels(ctx: Context, profile: Profile, role: str) -> dict[str, str]:
    """Return the labels that tie a container to this repo, role, user and checkout."""
    return {
        REPO_LABEL: profile.repo,
        ROLE_LABEL: role,
        UID_LABEL: str(ctx.uid),
        CHECKOUT_LABEL: ctx.checkout.digest,
    }


def label_problem(
    labels: Mapping[str, str], identity: Mapping[str, str], env_digest: str | None
) -> str | None:
    """Return why ``labels`` do not belong to ``identity``, or None when they do.

    Only labels under the controller's prefix are compared, and their key set
    must be exact, except that the auth label may be absent. ``env_digest``
    None skips the freshness comparison of both env-sha256 and the auth label,
    so ``down`` still removes a container created before the auth label existed.
    """
    own = {key: value for key, value in labels.items() if key.startswith(LABEL_PREFIX)}
    required_keys = {*identity, ENV_DIGEST_LABEL}
    if not required_keys <= own.keys() <= {*required_keys, AUTH_LABEL}:
        return f"label keys {sorted(own)} differ from {sorted({*required_keys, AUTH_LABEL})}"
    differing = sorted(key for key in identity if own[key] != identity[key])
    if differing:
        return f"labels {differing} belong to another stack"
    if not _SHA256_HEX.fullmatch(own[ENV_DIGEST_LABEL]):
        return f"label {ENV_DIGEST_LABEL} is not a sha256 digest"
    if env_digest is None:
        return None
    if own[ENV_DIGEST_LABEL] != env_digest:
        return "started from a different testdb.env (stale env-sha256)"
    if own.get(AUTH_LABEL) != _expected_auth(identity):
        return "started with another authentication setup (stale auth label)"
    return None


def _expected_auth(identity: Mapping[str, str]) -> str | None:
    return PG_AUTH_METHOD if role_kind(identity[ROLE_LABEL]) == "pg" else None


# ---------------------------------------------------------------------------
# docker inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortBinding:
    """One host binding docker reports for a published container port."""

    host_ip: str
    host_port: str


@dataclass(frozen=True)
class ContainerState:
    """The parts of ``docker container inspect`` the controller relies on."""

    id: str
    name: str
    labels: Mapping[str, str]
    running: bool
    ports: Mapping[str, tuple[PortBinding, ...]]
    addresses: tuple[str, ...] = ()


def _docker(
    ctx: Context, args: Sequence[str], timeout: float | None = DOCKER_TIMEOUT_S
) -> CommandResult:
    return ctx.run(["docker", *args], timeout)


def _require_ok(result: CommandResult, what: str) -> CommandResult:
    if result.returncode != 0:
        raise StackError(f"{what} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result


def _as_dict(value: Any, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StackError(f"docker inspect: {what} is not an object")
    return value


def _parse_bindings(spec: str, value: Any) -> tuple[PortBinding, ...]:
    # docker reports an exposed but unpublished port as null.
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise StackError(f"docker inspect: Ports[{spec!r}] is not a list of objects")
    bindings = tuple((item.get("HostIp"), item.get("HostPort")) for item in value)
    if not all(isinstance(ip, str) and isinstance(port, str) for ip, port in bindings):
        raise StackError(f"docker inspect: Ports[{spec!r}] has a non-string field")
    return tuple(PortBinding(ip, port) for ip, port in bindings)


def _parse_ports(entry: dict[str, Any]) -> Mapping[str, tuple[PortBinding, ...]]:
    raw = _as_dict(_as_dict(entry.get("NetworkSettings"), "NetworkSettings").get("Ports"), "Ports")
    return MappingProxyType({str(spec): _parse_bindings(spec, v) for spec, v in raw.items()})


def _parse_addresses(entry: dict[str, Any]) -> tuple[str, ...]:
    settings = _as_dict(entry.get("NetworkSettings"), "NetworkSettings")
    networks = _as_dict(settings.get("Networks"), "Networks")
    addresses = [_as_dict(net, "Networks entry").get("IPAddress") for net in networks.values()]
    if not all(isinstance(address, str) for address in addresses):
        raise StackError("docker inspect: Networks has a non-string IPAddress")
    return tuple(str(address) for address in addresses if address)


def _parse_labels(entry: dict[str, Any]) -> Mapping[str, str]:
    labels = _as_dict(_as_dict(entry.get("Config"), "Config").get("Labels"), "Labels")
    return MappingProxyType({str(k): str(v) for k, v in labels.items()})


def parse_inspect(name: str, stdout: str) -> ContainerState:
    """Parse ``docker container inspect <name>`` output; fail closed on any surprise."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise StackError(f"docker inspect {name}: unparseable output") from exc
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise StackError(f"docker inspect {name}: expected exactly one container")
    entry = data[0]
    if entry.get("Name") != f"/{name}":
        raise StackError(f"docker inspect {name}: returned {entry.get('Name')!r}")
    container_id = entry.get("Id")
    if not isinstance(container_id, str) or not _SHA256_HEX.fullmatch(container_id):
        raise StackError(f"docker inspect {name}: Id is not a full container ID")
    running = _as_dict(entry.get("State"), "State").get("Running")
    if not isinstance(running, bool):
        raise StackError(f"docker inspect {name}: State.Running is not a boolean")
    return ContainerState(
        container_id,
        name,
        _parse_labels(entry),
        running,
        _parse_ports(entry),
        _parse_addresses(entry),
    )


def inspect_container(
    ctx: Context, name: str, timeout: float = DOCKER_TIMEOUT_S
) -> ContainerState | None:
    """Return the container's state, or None when no container has that name."""
    result = _docker(ctx, ["container", "inspect", name], timeout)
    if result.returncode != 0:
        if re.search(r"no such container", result.stderr, re.IGNORECASE):
            return None
        _require_ok(result, f"docker container inspect {name}")
    return parse_inspect(name, result.stdout)


def owned_containers(
    ctx: Context, profile: Profile, env_digest: str | None, refusal: str
) -> dict[str, ContainerState | None]:
    """Inspect each role's container and verify it belongs to this stack.

    Returns role -> state (None when no container has the name). Any present
    container whose labels do not match raises StackError naming ``refusal``,
    before anything is changed. Callers address the returned ``state.id``,
    never the name, in every later docker call.
    """
    states = {role: inspect_container(ctx, name) for role, name in _names(ctx, profile).items()}
    refusals = [
        f"{state.name}: {problem}; {refusal}"
        for role, state in states.items()
        if state is not None
        and (
            problem := label_problem(state.labels, identity_labels(ctx, profile, role), env_digest)
        )
    ]
    if refusals:
        raise StackError("; ".join(refusals))
    return states


def container_port(role: str) -> str:
    """Return the container-side port spec for a role."""
    return "5432/tcp" if role_kind(role) == "pg" else "6379/tcp"


def host_port(state: ContainerState, role: str) -> int:
    """Return the loopback host port docker assigned; fail closed on anything else."""
    spec = container_port(role)
    bindings = state.ports.get(spec, ())
    if len(bindings) != 1:
        raise StackError(f"{state.name}: expected one host binding for {spec}, got {bindings!r}")
    host_ip, port = bindings[0].host_ip, bindings[0].host_port
    if host_ip != LOOPBACK:
        raise StackError(f"{state.name}: {spec} bound on {host_ip!r}, not {LOOPBACK}")
    if not _HOST_PORT.fullmatch(port) or int(port) > 65535:
        raise StackError(f"{state.name}: {spec} has invalid host port {port!r}")
    return int(port)


# ---------------------------------------------------------------------------
# Lifecycle steps
# ---------------------------------------------------------------------------


def run_argv(ctx: Context, profile: Profile, role: str, pins: Pins) -> list[str]:
    """Return the ``docker run`` argv that starts one role's container."""
    labels = {**identity_labels(ctx, profile, role), ENV_DIGEST_LABEL: pins.digest}
    if role_kind(role) == "pg":
        labels[AUTH_LABEL] = PG_AUTH_METHOD
    argv = ["docker", "run", "--detach", "--name", container_name(profile, ctx.checkout, role)]
    for key, value in labels.items():
        argv += ["--label", f"{key}={value}"]
    argv += ["--publish", f"{LOOPBACK}::{container_port(role).split('/')[0]}"]
    if role_kind(role) == "pg":
        argv += ["--env", f"POSTGRES_USER={PG_USER}", "--env", f"POSTGRES_DB={PG_MAINTENANCE_DB}"]
        argv += ["--env", f"POSTGRES_PASSWORD={PG_PASSWORD}"]
        argv += ["--env", f"POSTGRES_HOST_AUTH_METHOD={PG_AUTH_METHOD}"]
        argv += ["--env", f"POSTGRES_INITDB_ARGS={_PG_INITDB_AUTH}"]
        argv.append(pins.image("PGVECTOR_IMAGE"))
    else:
        argv.append(pins.image("REDIS_IMAGE"))
    return argv


def _is_docker_failure(result: CommandResult) -> bool:
    return result.returncode in _DOCKER_EXEC_FAILURES or bool(
        _DOCKER_FAILURE_STDERR.search(result.stderr)
    )


def _probe_failure(argv: Sequence[str], result: CommandResult) -> StackError:
    return StackError(
        f"{' '.join(['docker', *argv])} failed (exit {result.returncode}): {result.stderr.strip()}"
    )


def probe_ready(ctx: Context, container_id: str, role: str) -> bool:
    """Return whether the service accepts connections right now.

    The service still starting is False; a docker-level failure (the exec
    could not run, or the daemon reported an error) raises StackError.
    """
    if role_kind(role) == "pg":
        # TCP, so the socket-only server of the image's init phase cannot pass.
        argv = ["exec", container_id, "pg_isready", "-q", "-h", LOOPBACK, "-U", PG_USER]
        argv += ["-d", PG_MAINTENANCE_DB]
    else:
        argv = ["exec", container_id, "redis-cli", "ping"]
    result = _docker(ctx, argv, PROBE_TIMEOUT_S)
    if _is_docker_failure(result):
        raise _probe_failure(argv, result)
    if role_kind(role) == "redis":
        return result.returncode == 0 and result.stdout.strip() == "PONG"
    if result.returncode not in {0, EXIT_TIMED_OUT, *_PG_NOT_READY_EXITS}:
        raise _probe_failure(argv, result)
    return result.returncode == 0


def wait_ready(ctx: Context, state: ContainerState, role: str) -> None:
    """Probe up to READY_ATTEMPTS times; raise StackError if never ready."""
    for attempt in range(READY_ATTEMPTS):
        if probe_ready(ctx, state.id, role):
            return
        if attempt < READY_ATTEMPTS - 1:
            ctx.sleep(READY_INTERVAL_S)
    raise StackError(f"{state.name}: not ready after {READY_ATTEMPTS} attempts")


def _psql_argv(container_id: str, sql: str) -> list[str]:
    argv = ["exec", "--env", f"PGPASSWORD={PG_PASSWORD}", container_id, "psql", "-X", "-q"]
    argv += ["-U", PG_USER, "-d", PG_MAINTENANCE_DB]
    return [*argv, "-v", "ON_ERROR_STOP=1", "-tAc", sql]


def ensure_databases(ctx: Context, state: ContainerState, dbs: Sequence[str]) -> None:
    """Create each database in ``dbs`` that does not exist yet."""
    name, cid = state.name, state.id
    for db in dbs:
        # db matched _DB_RULE in make_profile, so it is safe to interpolate.
        lookup = f"SELECT 1 FROM pg_database WHERE datname = '{db}'"  # noqa: S608
        exists = _require_ok(_docker(ctx, _psql_argv(cid, lookup)), f"{name}: database lookup")
        if exists.stdout.strip() != "1":
            _require_ok(
                _docker(ctx, _psql_argv(cid, f'CREATE DATABASE "{db}"')), f"{name}: create {db}"
            )


def _env_key(role: str) -> str:
    return role.upper().replace("-", "_")


def stack_env(
    profile: Profile, names: Mapping[str, str], ports: Mapping[str, int]
) -> list[tuple[str, str]]:
    """Return the ``.testdb.env`` pairs for a running stack."""
    pairs: list[tuple[str, str]] = []
    for role in profile.roles:
        key, port = f"TESTDB_{_env_key(role)}", ports[role]
        pairs += [
            (f"{key}_CONTAINER", names[role]),
            (f"{key}_HOST", LOOPBACK),
            (f"{key}_PORT", str(port)),
        ]
        if role_kind(role) == "pg":
            base = f"postgresql://{PG_USER}:{PG_PASSWORD}@{LOOPBACK}:{port}"
            pairs.append((f"{key}_URL", f"{base}/{PG_MAINTENANCE_DB}"))
            if role == PRIMARY_PG_ROLE:
                pairs += [(f"{key}_URL_{db.upper()}", f"{base}/{db}") for db in profile.dbs]
        else:
            pairs.append((f"{key}_URL", f"redis://{LOOPBACK}:{port}/0"))
    return pairs


def render_stack_env(pairs: Sequence[tuple[str, str]]) -> str:
    """Render pairs as bash-sourceable ``KEY='value'`` lines."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ConfigError(f"profile produces duplicate env keys: {keys!r}")
    for key, value in pairs:
        if not _SHELL_SAFE_VALUE.fullmatch(value):
            raise StackError(f"{key}: value is not shell-safe: {value!r}")
    lines = [f"{key}='{value}'" for key, value in pairs]
    return (
        "# Written by tools/testdb/testdb.py up; regenerated on every up.\n"
        + "\n".join(lines)
        + "\n"
    )


def parse_stack_env(text: str) -> dict[str, str]:
    """Parse a ``.testdb.env`` this tool wrote; raise ValueError on any other line."""
    pairs: dict[str, str] = {}
    for line in _lines(text):
        if line.startswith("#"):
            continue
        match = _STACK_ENV_LINE.fullmatch(line)
        if match is None or match[1] in pairs:
            raise ValueError(f"{STACK_ENV_NAME}: unexpected line {line!r}")
        pairs[match[1]] = match[2]
    return pairs


def stack_env_matches(ctx: Context, profile: Profile, ports: Mapping[str, int]) -> bool:
    """Return whether ``.testdb.env`` names exactly the live containers and ports."""
    try:
        text = (ctx.checkout.toplevel / STACK_ENV_NAME).read_text(encoding="ascii")
        recorded = parse_stack_env(text)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return recorded == dict(stack_env(profile, _names(ctx, profile), ports))


def write_stack_env(toplevel: Path, content: str) -> Path:
    """Atomically replace ``.testdb.env`` at the checkout root."""
    target = toplevel / STACK_ENV_NAME
    fd, tmp = tempfile.mkstemp(dir=toplevel, prefix=f"{STACK_ENV_NAME}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(content)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _names(ctx: Context, profile: Profile) -> dict[str, str]:
    return {role: container_name(profile, ctx.checkout, role) for role in profile.roles}


def _state_problem(
    ctx: Context, profile: Profile, role: str, state: ContainerState | None, env_digest: str
) -> str | None:
    """Return why a role's container is not a running member of this stack, or None."""
    if state is None:
        return "missing"
    problem = label_problem(state.labels, identity_labels(ctx, profile, role), env_digest)
    if problem is None and not state.running:
        return "not running"
    return problem


def _running_stack(ctx: Context, profile: Profile, pins: Pins) -> dict[str, ContainerState]:
    """Return every role's verified-owned running container; raise StackError on any miss."""
    states = owned_containers(ctx, profile, pins.digest, "not a container of this stack")
    problems = [
        f"{container_name(profile, ctx.checkout, role)}: missing"
        if state is None
        else f"{state.name}: not running"
        for role, state in states.items()
        if state is None or not state.running
    ]
    if problems:
        raise StackError("; ".join(problems))
    return {role: state for role, state in states.items() if state is not None}


def _start_role(
    ctx: Context, profile: Profile, pins: Pins, role: str, state: ContainerState | None
) -> str:
    """Create or start one role's container; return its full container ID."""
    if state is not None:
        if not state.running:
            _require_ok(_docker(ctx, ["start", state.id]), f"docker start {state.name}")
        return state.id
    name = container_name(profile, ctx.checkout, role)
    created = _require_ok(ctx.run(run_argv(ctx, profile, role, pins), None), f"docker run {name}")
    container_id = created.stdout.strip()
    if not _SHA256_HEX.fullmatch(container_id):
        raise StackError(f"docker run {name}: printed {container_id!r}, not a container ID")
    return container_id


def cmd_up(ctx: Context, profile: Profile) -> int:
    """Start (or reuse) this checkout's stack and write ``.testdb.env``."""
    pins = load_pins(ctx.env_file)
    refusal = "refusing to reuse it (run down, then up)"
    states = owned_containers(ctx, profile, pins.digest, refusal)
    ids = {role: _start_role(ctx, profile, pins, role, state) for role, state in states.items()}
    live = _running_stack(ctx, profile, pins)
    replaced = sorted(live[role].name for role in profile.roles if live[role].id != ids[role])
    if replaced:
        raise StackError(f"{replaced} were replaced by another container during up")
    for role, state in live.items():
        wait_ready(ctx, state, role)
    if profile.dbs:
        ensure_databases(ctx, live[PRIMARY_PG_ROLE], profile.dbs)
    ports = {role: host_port(state, role) for role, state in live.items()}
    names = _names(ctx, profile)
    target = write_stack_env(
        ctx.checkout.toplevel, render_stack_env(stack_env(profile, names, ports))
    )
    for role in profile.roles:
        sys.stdout.write(f"{role} {names[role]} {LOOPBACK}:{ports[role]}\n")
    sys.stdout.write(f"wrote {target}\n")
    return EXIT_OK


def cmd_down(ctx: Context, profile: Profile) -> int:
    """Remove this checkout's containers; refuse, removing nothing, on any foreign one."""
    states = owned_containers(ctx, profile, None, "refusing to remove it")
    for state in states.values():
        if state is None:
            continue
        rm_argv = ["rm", "--force", "--volumes", state.id]
        _require_ok(_docker(ctx, rm_argv), f"docker rm {state.name}")
        sys.stdout.write(f"removed {state.name}\n")
    (ctx.checkout.toplevel / STACK_ENV_NAME).unlink(missing_ok=True)
    return EXIT_OK


def stack_problems(
    ctx: Context, profile: Profile, pins: Pins
) -> tuple[dict[str, str | None], dict[str, int]]:
    """Return each role's problem (None when ready) and the ready roles' host ports."""
    problems: dict[str, str | None] = {}
    ports: dict[str, int] = {}
    for role, name in _names(ctx, profile).items():
        state = inspect_container(ctx, name, PROBE_TIMEOUT_S)
        problem = _state_problem(ctx, profile, role, state, pins.digest)
        if problem is None and state is not None:
            ports[role] = host_port(state, role)
            problem = None if probe_ready(ctx, state.id, role) else "not ready"
        problems[role] = problem
    return problems, ports


def cmd_status(ctx: Context, profile: Profile, *, require_ready: bool) -> int:
    """Report the stack; with ``require_ready`` print only the fail-fast line on a miss."""
    pins = load_pins(ctx.env_file)
    try:
        problems, ports = stack_problems(ctx, profile, pins)
    except StackError:
        if require_ready:
            sys.stderr.write(f"{NOT_RUNNING}\n")
            return EXIT_STACK
        raise
    is_ready = all(problem is None for problem in problems.values())
    if require_ready:
        if not (is_ready and stack_env_matches(ctx, profile, ports)):
            sys.stderr.write(f"{NOT_RUNNING}\n")
            return EXIT_STACK
        return EXIT_OK
    names = _names(ctx, profile)
    for role, problem in problems.items():
        sys.stdout.write(f"{role} {names[role]} {problem or 'ready'}\n")
    return EXIT_OK if is_ready else EXIT_STACK


def cmd_env(ctx: Context, profile: Profile) -> int:
    """Print the ``.testdb.env`` content for the running stack."""
    pins = load_pins(ctx.env_file)
    live = _running_stack(ctx, profile, pins)
    ports = {role: host_port(state, role) for role, state in live.items()}
    sys.stdout.write(render_stack_env(stack_env(profile, _names(ctx, profile), ports)))
    return EXIT_OK


def cmd_check_env(env_file: Path, github_output: Path | None) -> int:
    """Validate testdb.env and emit its KEY=value lines to stdout or ``github_output``."""
    pins = load_pins(env_file)
    lines = "".join(f"{key}={value}\n" for key, value in pins.values.items())
    if github_output is None:
        sys.stdout.write(lines)
        return EXIT_OK
    try:
        with github_output.open("a", encoding="ascii") as handle:
            handle.write(lines)
    except OSError as exc:
        raise ConfigError(f"{github_output}: cannot append: {exc}") from exc
    return EXIT_OK


# ---------------------------------------------------------------------------
# psql on PATH
# ---------------------------------------------------------------------------


def _host_psql(path: str) -> str | None:
    """Return the first ``psql`` on ``path`` outside the wrapper dir, or None."""
    entries = [
        entry
        for entry in path.split(os.pathsep)
        if entry and Path(entry).resolve() != PSQL_WRAPPER_DIR
    ]
    return shutil.which("psql", path=os.pathsep.join(entries)) if entries else None


def psql_major(
    run: Callable[[Sequence[str], float | None], CommandResult], psql: str
) -> str | None:
    """Return the major version ``psql --version`` reports, or None if it does not."""
    result = run([psql, "--version"], PROBE_TIMEOUT_S)
    match = _PSQL_VERSION.match(result.stdout) if result.returncode == 0 else None
    return match[1] if match else None


def psql_path_prefix(
    run: Callable[[Sequence[str], float | None], CommandResult], path: str, pg_major: str
) -> Path | None:
    """Return the wrapper dir to prepend to ``path``, or None when a host psql fits."""
    host = _host_psql(path)
    if host is not None and psql_major(run, host) == pg_major:
        return None
    return PSQL_WRAPPER_DIR


def cmd_psql_path(env_file: Path) -> int:
    """Print the wrapper dir when PATH lacks a host psql of PG_MAJOR."""
    pins = load_pins(env_file)
    prefix = psql_path_prefix(run_command, os.environ.get("PATH", ""), _pg_major(pins, env_file))
    if prefix is not None:
        sys.stdout.write(f"{prefix}\n")
    return EXIT_OK


def _pg_major(pins: Pins, env_file: Path) -> str:
    pg_major = pins.values.get("PG_MAJOR", "")
    if not _PG_MAJOR_RULE.fullmatch(pg_major):
        raise ConfigError(f"{env_file}: PG_MAJOR={pg_major} must match {_PG_MAJOR_RULE.pattern}")
    return pg_major


# ---------------------------------------------------------------------------
# psql-plan
# ---------------------------------------------------------------------------

# psql options bin/psql can carry into a stack container unchanged, keyed by
# every spelling; the value is the short form re-emitted (``--csv`` has none).
_PSQL_FLAGS = {
    **{f"-{c}": f"-{c}" for c in "XqtAabeEnsSxz0wW1H"},
    "--no-psqlrc": "-X",
    "--quiet": "-q",
    "--tuples-only": "-t",
    "--no-align": "-A",
    "--echo-all": "-a",
    "--echo-errors": "-b",
    "--echo-queries": "-e",
    "--echo-hidden": "-E",
    "--no-readline": "-n",
    "--single-step": "-s",
    "--single-line": "-S",
    "--expanded": "-x",
    "--field-separator-zero": "-z",
    "--record-separator-zero": "-0",
    "--no-password": "-w",
    "--password": "-W",
    "--single-transaction": "-1",
    "--html": "-H",
    "--csv": "--csv",
}
_PSQL_VALUED = {
    **{f"-{c}": f"-{c}" for c in "cdfvhpUFRPTLo"},
    "--command": "-c",
    "--dbname": "-d",
    "--file": "-f",
    "--set": "-v",
    "--variable": "-v",
    "--host": "-h",
    "--port": "-p",
    "--username": "-U",
    "--field-separator": "-F",
    "--record-separator": "-R",
    "--pset": "-P",
    "--table-attr": "-T",
    "--log-file": "-L",
    "--output": "-o",
}
# They write host files, which inside a container would land in the container.
_PSQL_HOST_OUTPUTS = frozenset({"-o", "-L"})
# Each can send libpq to a server other than the one -h/-p or the URL names.
_PSQL_REDIRECTING_ENV = ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE")
_PSQL_STDIN_TOKENS = frozenset({"--", "-"})
_PSQL_URL_SCHEMES = frozenset({"postgresql", "postgres"})
_CONNINFO = re.compile(r"\s*([a-z_]+)\s*=\s*('(?:[^'\\]|\\.)*'|[^\s']\S*)\s*", re.ASCII)
_CONNINFO_KEYS = frozenset({"dbname", "user"})
PG_CONTAINER_PORT = "5432"
PLAN_EXEC = "exec"
PLAN_RUN = "run"
PLAN_END = "end"


class _NotExecutable(Exception):
    """The command line is outside what the exec path can carry; run it instead."""


@dataclass(frozen=True)
class PsqlCommand:
    """A psql command line split into connection settings and everything else."""

    options: tuple[tuple[str, str | None], ...]
    host: str | None
    port: str | None
    user: str | None
    dbname: str | None


def _option_spelling(token: str) -> tuple[str, str | None]:
    """Split one option token into (spelling, attached value or None)."""
    if token.startswith("--"):
        name, sep, value = token.partition("=")
        return name, value if sep else None
    return token[:2], token[2:] or None


def _short_bundle(token: str, rest: list[str]) -> list[tuple[str, str | None]]:
    """Expand ``-tAc SQL``-style bundles the way psql's getopt reads them."""
    parsed: list[tuple[str, str | None]] = []
    for index, char in enumerate(token[1:], start=1):
        spelling = f"-{char}"
        if spelling in _PSQL_FLAGS:
            parsed.append((_PSQL_FLAGS[spelling], None))
            continue
        if spelling not in _PSQL_VALUED:
            raise _NotExecutable(spelling)
        attached = token[index + 1 :]
        if not attached and not rest:
            raise _NotExecutable(f"{spelling} without a value")
        parsed.append((_PSQL_VALUED[spelling], attached or rest.pop(0)))
        return parsed
    return parsed


def _parse_options(argv: Sequence[str]) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Return psql's options in order and its positional arguments."""
    rest = list(argv)
    options: list[tuple[str, str | None]] = []
    positionals: list[str] = []
    while rest:
        token = rest.pop(0)
        if token in _PSQL_STDIN_TOKENS:
            raise _NotExecutable(token)
        if not token.startswith("-"):
            positionals.append(token)
        elif token.startswith("--"):
            spelling, value = _option_spelling(token)
            if spelling in _PSQL_FLAGS and value is None:
                options.append((_PSQL_FLAGS[spelling], None))
            elif spelling in _PSQL_VALUED:
                if value is None and not rest:
                    raise _NotExecutable(f"{spelling} without a value")
                options.append((_PSQL_VALUED[spelling], rest.pop(0) if value is None else value))
            else:
                raise _NotExecutable(spelling)
        else:
            options += _short_bundle(token, rest)
    return options, positionals


def parse_psql_command(argv: Sequence[str]) -> PsqlCommand:
    """Parse a psql command line; raise _NotExecutable for anything unrecognized."""
    parsed, positionals = _parse_options(argv)
    connection: dict[str, str | None] = {"-h": None, "-p": None, "-U": None, "-d": None}
    options: list[tuple[str, str | None]] = []
    for name, value in parsed:
        if name in connection:
            if connection[name] is not None:
                raise _NotExecutable(f"{name} given twice")
            connection[name] = value
        elif name in _PSQL_HOST_OUTPUTS:
            raise _NotExecutable(name)
        else:
            options.append((name, value))
    if len(positionals) > 2 or (positionals and connection["-d"] is not None):
        raise _NotExecutable("positional connection arguments")
    if len(positionals) == 2 and connection["-U"] is not None:
        raise _NotExecutable("user given twice")
    dbname = connection["-d"] if connection["-d"] is not None else next(iter(positionals), None)
    user = connection["-U"] if len(positionals) < 2 else positionals[1]
    return PsqlCommand(tuple(options), connection["-h"], connection["-p"], user, dbname)


def _loopback_port(host: str | None, port: str | None) -> str:
    if host not in CI_HOSTS or port is None or not _HOST_PORT.fullmatch(port):
        raise _NotExecutable("target is not a loopback host on an explicit port")
    return port


def _is_url(dbname: str | None) -> bool:
    if dbname is None:
        return False
    scheme, separator, _ = dbname.partition("://")
    return bool(separator) and scheme in _PSQL_URL_SCHEMES


def _split_url(url: str) -> tuple[urllib.parse.SplitResult, str, str]:
    """Return the URL's parts, ``userinfo@`` prefix and loopback port."""
    parts = urllib.parse.urlsplit(url)
    if parts.query or parts.fragment or "," in parts.netloc:
        raise _NotExecutable("URL carries settings beyond user, host, port and dbname")
    userinfo, at, hostport = parts.netloc.rpartition("@")
    host, _, port = hostport.rpartition(":")
    return parts, f"{userinfo}{at}", _loopback_port(host, port)


def _rewrite_url(url: str, address: str) -> str:
    """Point a ``postgresql://`` URL at ``address`` on the container port."""
    parts, userinfo, _ = _split_url(url)
    netloc = f"{userinfo}{address}:{PG_CONTAINER_PORT}"
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def _check_conninfo(dbname: str) -> None:
    """Allow a ``key=value`` dbname only when it names nothing but dbname and user."""
    if "=" not in dbname:
        return
    end, keys = 0, []
    while end < len(dbname):
        match = _CONNINFO.match(dbname, end)
        if match is None or match.end() == end:
            raise _NotExecutable("unparseable conninfo")
        keys.append(match[1])
        end = match.end()
    if not set(keys) <= _CONNINFO_KEYS:
        raise _NotExecutable("conninfo names connection settings")


def target_port(command: PsqlCommand) -> str:
    """Return the loopback host port the command connects to."""
    if _is_url(command.dbname):
        if command.host is not None or command.port is not None or command.user is not None:
            raise _NotExecutable("both a URL and -h, -p or -U")
        return _split_url(command.dbname or "")[2]
    _check_conninfo(command.dbname or "")
    return _loopback_port(command.host, command.port)


def exec_argv(command: PsqlCommand, address: str, stdin_file: str | None) -> list[str]:
    """Return the psql argv for the container: target rewritten, ``-f PATH`` as ``-f -``."""
    argv: list[str] = []
    for name, value in command.options:
        if name == "-f" and stdin_file is not None:
            argv += ["-f", "-"]
        else:
            argv += [name] if value is None else [name, value]
    dbname = command.dbname
    if dbname is not None and _is_url(dbname):
        return [*argv, "-d", _rewrite_url(dbname, address)]
    argv += ["-h", address, "-p", PG_CONTAINER_PORT]
    if command.user is not None:
        argv += ["-U", command.user]
    return argv if dbname is None else [*argv, "-d", dbname]


def sql_input_file(command: PsqlCommand, cwd: Path) -> str | None:
    """Return the resolved ``-f`` file, or None when psql reads no file.

    A file outside the repository's SQL directories is a ConfigError.
    """
    files = [value for name, value in command.options if name == "-f" and value is not None]
    if len(files) > 1:
        raise _NotExecutable("more than one -f")
    if not files or files[0] == "-":
        return None
    try:
        path = (cwd / files[0]).resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"-f {files[0]}: {exc.strerror or exc}") from exc
    allowed = [d.resolve() for d in PSQL_SQL_DIRS]
    if not path.is_file() or not any(path.is_relative_to(d) for d in allowed):
        dirs = ", ".join(str(d) for d in allowed)
        raise ConfigError(f"-f {files[0]}: only regular files under {dirs} can be passed")
    return str(path)


def _owned_pg_container(
    run: Callable[[Sequence[str], float | None], CommandResult],
    uid: int,
    checkout: Checkout,
    env_digest: str,
    port: str,
) -> ContainerState | None:
    """Return this checkout's running pg container published on ``port``, or None."""
    listed = run(
        [
            "docker", "ps", "--filter", f"label={CHECKOUT_LABEL}={checkout.digest}",
            "--filter", f"label={UID_LABEL}={uid}", "--filter", f"publish={port}",
            "--format", "{{.Names}}",
        ],
        PROBE_TIMEOUT_S,
    )  # fmt: skip
    names = listed.stdout.split() if listed.returncode == 0 else []
    if len(names) != 1:
        return None
    inspected = run(["docker", "container", "inspect", names[0]], PROBE_TIMEOUT_S)
    if inspected.returncode != 0:
        return None
    try:
        state = parse_inspect(names[0], inspected.stdout)
        return state if _is_exec_target(state, uid, checkout, env_digest, port) else None
    except (StackError, ConfigError):
        return None


def _is_exec_target(
    state: ContainerState, uid: int, checkout: Checkout, env_digest: str, port: str
) -> bool:
    """Apply the ``env`` ownership check to one container, plus port and address."""
    repo, role = state.labels.get(REPO_LABEL, ""), state.labels.get(ROLE_LABEL, "")
    if not _REPO_RULE.fullmatch(repo) or not _ROLE_RULE.fullmatch(role) or role_kind(role) != "pg":
        return False
    identity = {REPO_LABEL: repo, ROLE_LABEL: role, UID_LABEL: str(uid)}
    identity[CHECKOUT_LABEL] = checkout.digest
    profile = Profile(repo=repo, roles=(role,), dbs=())
    return (
        label_problem(state.labels, identity, env_digest) is None
        and state.name == container_name(profile, checkout, role)
        and state.running
        and host_port(state, role) == int(port)
        and container_address(state) is not None
    )


def container_address(state: ContainerState) -> str | None:
    """Return the container's one IPv4 network address, or None."""
    if len(state.addresses) != 1:
        return None
    try:
        return str(ipaddress.IPv4Address(state.addresses[0]))
    except ValueError:
        return None


@dataclass(frozen=True)
class PsqlPlan:
    """How bin/psql runs one command line; see ``psql-plan`` in the module docstring."""

    mode: str
    target: str
    stdin_file: str | None = None
    argv: tuple[str, ...] = ()

    def fields(self) -> list[str]:
        """Return the NUL-separated fields bin/psql reads."""
        if self.mode == PLAN_RUN:
            return [PLAN_RUN, self.target, PLAN_END]
        return [PLAN_EXEC, self.target, self.stdin_file or "", *self.argv, PLAN_END]


def plan_psql(
    argv: Sequence[str],
    *,
    run: Callable[[Sequence[str], float | None], CommandResult],
    uid: int,
    environ: Mapping[str, str],
    cwd: Path,
    env_file: Path,
) -> PsqlPlan:
    """Decide between ``docker exec`` into this checkout's stack and ``docker run``."""
    pins = load_pins(env_file)
    fallback = PsqlPlan(PLAN_RUN, pins.image("PGVECTOR_IMAGE"))
    try:
        command = parse_psql_command(argv)
        port = target_port(command)
    except _NotExecutable:
        return fallback
    if any(name in environ for name in _PSQL_REDIRECTING_ENV):
        return fallback
    try:
        checkout = find_checkout(run)
    except ConfigError:
        return fallback
    state = _owned_pg_container(run, uid, checkout, pins.digest, port)
    address = None if state is None else container_address(state)
    if state is None or address is None:
        return fallback
    try:
        stdin_file = sql_input_file(command, cwd)
    except _NotExecutable:
        return fallback
    return PsqlPlan(PLAN_EXEC, state.id, stdin_file, tuple(exec_argv(command, address, stdin_file)))


def cmd_psql_plan(argv: Sequence[str]) -> int:
    """Print the plan for one psql command line as NUL-terminated fields."""
    plan = plan_psql(
        argv,
        run=run_command,
        uid=os.getuid(),
        environ=os.environ,
        cwd=Path.cwd(),
        env_file=TESTDB_ENV,
    )
    sys.stdout.buffer.write(b"".join(os.fsencode(field) + b"\0" for field in plan.fields()))
    return EXIT_OK


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Migration:
    """One migration command: its working directory and argument array."""

    cwd: Path
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ResetOptions:
    """Everything ``reset`` takes beyond the profile."""

    ci: bool
    bootstrap_vars: tuple[tuple[str, str], ...]
    dsn_envs: tuple[str, ...]
    migrations: tuple[Migration, ...]


@dataclass(frozen=True)
class PgTarget:
    """One cluster ``reset`` works on, and the profile databases it holds."""

    role: str
    host: str
    port: int
    user: str
    password: str | None
    dbs: tuple[str, ...]

    def dsn(self, db: str) -> str:
        """Return a libpq URL for ``db``; it carries the password, so never print it."""
        user = urllib.parse.quote(self.user, safe="")
        if self.password is None:
            return f"postgresql://{user}@{self.host}:{self.port}/{db}"
        auth = f"{user}:{urllib.parse.quote(self.password, safe='')}"
        return f"postgresql://{auth}@{self.host}:{self.port}/{db}"


@dataclass(frozen=True)
class ResetRuntime:
    """The resolved psql binary, the child environment and the strings to redact."""

    psql: str
    env: Mapping[str, str]
    secrets: tuple[str, ...]


def split_migrations(argv: Sequence[str]) -> tuple[list[str], tuple[Migration, ...]]:
    """Split ``reset`` argv at each ``--migrate`` into the head and the migration commands."""
    args = list(argv)
    if not args or args[0] != "reset" or MIGRATE_FLAG not in args:
        return args, ()
    first = args.index(MIGRATE_FLAG)
    segments: list[list[str]] = []
    for token in args[first:]:
        if token == MIGRATE_FLAG:
            segments.append([])
        else:
            segments[-1].append(token)
    for segment in segments:
        if len(segment) < 2:
            raise ConfigError(f"{MIGRATE_FLAG} needs a working directory and a command")
    return args[:first], tuple(Migration(Path(s[0]), tuple(s[1:])) for s in segments)


def make_reset_options(
    args: argparse.Namespace, migrations: Sequence[Migration], toplevel: Path
) -> ResetOptions:
    """Validate the reset options; raise ConfigError before anything runs."""
    pairs: list[tuple[str, str]] = []
    for item in args.bootstrap_vars:
        name, _, value = item.partition("=")
        if name not in BOOTSTRAP_BOOL_VARS or not _BOOL_VALUE.fullmatch(value):
            raise ConfigError(
                f"--bootstrap-var {item!r} must be NAME=VALUE with NAME in "
                f"{sorted(BOOTSTRAP_BOOL_VARS)} and VALUE a boolean"
            )
        pairs.append((name, value))
    for name in args.dsn_envs:
        if not _ENV_NAME_RULE.fullmatch(name):
            raise ConfigError(f"--migrate-dsn-env {name!r} must match {_ENV_NAME_RULE.pattern}")
    if migrations and not args.dsn_envs:
        raise ConfigError(f"{MIGRATE_FLAG} needs at least one --migrate-dsn-env")
    resolved = tuple(Migration(toplevel / m.cwd, m.argv) for m in migrations)
    for migration in resolved:
        if not migration.cwd.is_dir():
            raise ConfigError(f"{MIGRATE_FLAG} directory {migration.cwd} is not a directory")
    return ResetOptions(args.ci, tuple(pairs), tuple(args.dsn_envs), resolved)


def _check_reset_dbs(profile: Profile) -> None:
    protected = sorted(set(profile.dbs) & PROTECTED_DBS)
    if protected:
        raise ConfigError(f"reset never drops {protected}; remove them from --db")


def _pg_roles(profile: Profile) -> list[str]:
    return [role for role in profile.roles if role_kind(role) == "pg"]


def _role_dbs(profile: Profile, role: str) -> tuple[str, ...]:
    return profile.dbs if role == PRIMARY_PG_ROLE else ()


def local_targets(ctx: Context, profile: Profile) -> list[PgTarget]:
    """Return this checkout's verified running clusters; never starts a container."""
    live = _running_stack(ctx, profile, load_pins(ctx.env_file))
    return [
        PgTarget(role, LOOPBACK, host_port(live[role], role), PG_USER, PG_PASSWORD, dbs)
        for role in _pg_roles(profile)
        for dbs in (_role_dbs(profile, role),)
    ]


def ci_url_env(role: str) -> str:
    """Return the env var a CI job sets to the cluster DSN of ``role``."""
    return f"TESTDB_{_env_key(role)}_URL"


def parse_ci_target(role: str, url: str, dbs: tuple[str, ...]) -> PgTarget:
    """Parse a CI cluster DSN; refuse anything but a loopback postgres URL."""
    name = ci_url_env(role)
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port or 5432
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid URL") from exc
    if parts.scheme not in {"postgres", "postgresql"}:
        raise ConfigError(f"{name} must be a postgresql:// URL")
    if parts.hostname not in CI_HOSTS:
        raise StackError(f"{name} host {parts.hostname!r} is not one of {sorted(CI_HOSTS)}")
    if not parts.username:
        raise ConfigError(f"{name} must name a user")
    user = urllib.parse.unquote(parts.username)
    password = None if parts.password is None else urllib.parse.unquote(parts.password)
    return PgTarget(role, parts.hostname, port, user, password, dbs)


def ci_targets(ctx: Context, profile: Profile) -> list[PgTarget]:
    """Return the CI service clusters named by environment variables."""
    if ctx.environ.get("GITHUB_ACTIONS") != "true":
        raise ConfigError("--ci is refused unless GITHUB_ACTIONS=true")
    targets = []
    for role in _pg_roles(profile):
        url = ctx.environ.get(ci_url_env(role))
        if not url:
            raise ConfigError(f"--ci needs {ci_url_env(role)} set")
        targets.append(parse_ci_target(role, url, _role_dbs(profile, role)))
    return targets


def redact(text: str, secrets: Sequence[str]) -> str:
    """Mask every secret and every URL password in ``text``."""
    for secret in secrets:
        text = text.replace(secret, "***")
    return _URL_PASSWORD.sub(r"\1:***@", text)


def reset_runtime(ctx: Context, targets: Sequence[PgTarget]) -> ResetRuntime:
    """Resolve psql (host psql of PG_MAJOR, else the wrapper) and the child environment."""
    pg_major = _pg_major(load_pins(ctx.env_file), ctx.env_file)
    path = ctx.environ.get("PATH", "")

    def version_run(argv: Sequence[str], timeout: float | None) -> CommandResult:
        return ctx.run_env(argv, timeout, ctx.environ, None)

    prefix = psql_path_prefix(version_run, path, pg_major)
    if prefix is not None:
        path = os.pathsep.join(p for p in (str(prefix), path) if p)
    psql = shutil.which("psql", path=path) if prefix is None else str(prefix / "psql")
    if psql is None:
        raise ConfigError("no psql on PATH")
    return ResetRuntime(psql, child_env(ctx.environ, path), reset_secrets(ctx, targets))


def child_env(environ: Mapping[str, str], path: str) -> Mapping[str, str]:
    """Return the environment for psql and migrations, minus anything naming a database.

    libpq settings (PGHOST, PGSERVICE, ...) could point psql at another server, and
    an inherited postgres URL (say JOURNAL_DB_MIGRATION_URL) could win over the one
    reset sets for a migration. Migrations see only the DSNs reset passes them.
    """
    env = {
        key: value
        for key, value in environ.items()
        if (key in BOOTSTRAP_PASSWORD_ENVS or not _LIBPQ_ENV.fullmatch(key))
        and not _PG_DSN_VALUE.match(value)
    }
    return MappingProxyType({**env, "PATH": path})


def reset_secrets(ctx: Context, targets: Sequence[PgTarget]) -> tuple[str, ...]:
    """Return every password reset may see, longest first so redaction masks it whole."""
    found = {t.password for t in targets if t.password}
    found |= {ctx.environ[k] for k in BOOTSTRAP_PASSWORD_ENVS if ctx.environ.get(k)}
    return tuple(sorted(found, key=len, reverse=True))


def _run_checked(
    ctx: Context,
    rt: ResetRuntime,
    argv: Sequence[str],
    what: str,
    env: Mapping[str, str],
    cwd: Path | None = None,
    timeout: float = PSQL_TIMEOUT_S,
) -> CommandResult:
    result = ctx.run_env(argv, timeout, env, cwd)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-20:])
        raise StackError(f"{what} failed (exit {result.returncode}): {redact(tail, rt.secrets)}")
    return result


def _psql_env(rt: ResetRuntime, target: PgTarget) -> dict[str, str]:
    env = dict(rt.env)
    if target.password is not None:
        env["PGPASSWORD"] = target.password
    return env


def conninfo_dbname(db: str) -> str:
    """Return a libpq conninfo naming only ``db``.

    psql expands a bare ``-d`` holding ``=`` or a URL prefix into connection
    settings, so a database name could otherwise pick the host, port or user.
    """
    escaped = db.replace("\\", "\\\\").replace("'", "\\'")
    return f"dbname='{escaped}'"


def _psql_conn(rt: ResetRuntime, target: PgTarget, db: str) -> list[str]:
    conn = ["-h", target.host, "-p", str(target.port), "-U", target.user]
    return [rt.psql, "-X", "-q", "-v", "ON_ERROR_STOP=1", *conn, "-d", conninfo_dbname(db)]


def psql_sql(ctx: Context, rt: ResetRuntime, target: PgTarget, db: str, sql: str) -> str:
    """Run one SQL string in ``db`` and return its unaligned, tuples-only output."""
    argv = [*_psql_conn(rt, target, db), "-tAc", sql]
    what = f"{target.role}: psql in {db}"
    return _run_checked(ctx, rt, argv, what, _psql_env(rt, target)).stdout


def psql_statements(
    ctx: Context, rt: ResetRuntime, target: PgTarget, db: str, statements: Sequence[str]
) -> None:
    """Run ``statements`` in ``db`` in one psql session, each in its own transaction."""
    commands = [arg for sql in statements for arg in ("-c", sql)]
    argv = [*_psql_conn(rt, target, db), "-tA", *commands]
    _run_checked(ctx, rt, argv, f"{target.role}: psql in {db}", _psql_env(rt, target))


def _json_rows(output: str, what: str) -> list[Any]:
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StackError(f"{what}: unparseable psql output") from exc
    if not isinstance(rows, list):
        raise StackError(f"{what}: expected a JSON array")
    return rows


def quote_ident(name: str) -> str:
    """Return ``name`` as a SQL identifier, doubling embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def droppable_roles(rows: Sequence[Any], bootstrap_user: str) -> list[str]:
    """Return the roles reset drops: not a superuser, not ``pg_*``, not the bootstrap user."""
    names = []
    for row in rows:
        if not (isinstance(row, dict) and isinstance(row.get("name"), str)):
            raise StackError("role list: malformed row")
        if not isinstance(row.get("super"), bool):
            raise StackError(f"role list: rolsuper of {row['name']!r} is not a boolean")
        name = row["name"]
        if row["super"] or name.startswith("pg_") or name == bootstrap_user:
            continue
        names.append(name)
    return names


def drop_databases(ctx: Context, rt: ResetRuntime, target: PgTarget) -> None:
    """Drop the target's profile databases, disconnecting any session in them."""
    for db in target.dbs:
        sql = f"DROP DATABASE IF EXISTS {quote_ident(db)} WITH (FORCE)"
        psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, sql)


def surviving_databases(ctx: Context, rt: ResetRuntime, target: PgTarget) -> list[tuple[str, bool]]:
    """Return ``(name, allows connections)`` for every database but template0."""
    out = psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, DB_LIST_SQL)
    found = []
    for row in _json_rows(out, f"{target.role}: database list"):
        if not (
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and isinstance(row.get("allowconn"), bool)
        ):
            raise StackError(f"{target.role}: database list: malformed row")
        found.append((row["name"], row["allowconn"]))
    return found


def clear_owned(
    ctx: Context, rt: ResetRuntime, target: PgTarget, db: str, roles: Sequence[str]
) -> None:
    """Hand the roles' objects in ``db`` to the bootstrap user, then drop their grants.

    One DROP OWNED naming several roles fails when one of them holds a default
    privilege granted to another, so each role gets its own.
    """
    idents = ", ".join(quote_ident(role) for role in roles)
    reassign = f"REASSIGN OWNED BY {idents} TO {quote_ident(target.user)}"
    drops = [f"DROP OWNED BY {quote_ident(role)}" for role in roles]
    psql_statements(ctx, rt, target, db, [reassign, *drops])


def with_connections_allowed(
    ctx: Context, rt: ResetRuntime, target: PgTarget, db: str, step: Callable[[], None]
) -> None:
    """Run ``step`` with connections to ``db`` allowed, then disallow them again."""
    alter = f"ALTER DATABASE {quote_ident(db)} ALLOW_CONNECTIONS"
    psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, f"{alter} true")
    try:
        step()
    finally:
        psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, f"{alter} false")


def drop_roles(ctx: Context, rt: ResetRuntime, target: PgTarget) -> None:
    """Drop every droppable role, first clearing what it owns in every surviving database."""
    out = psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, ROLE_LIST_SQL)
    roles = droppable_roles(_json_rows(out, f"{target.role}: role list"), target.user)
    if not roles:
        return
    for db, allows_connections in surviving_databases(ctx, rt, target):
        clear = functools.partial(clear_owned, ctx, rt, target, db, roles)
        if allows_connections:
            clear()
        else:
            with_connections_allowed(ctx, rt, target, db, clear)
    idents = ", ".join(quote_ident(role) for role in roles)
    psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, f"DROP ROLE {idents}")


def create_databases(ctx: Context, rt: ResetRuntime, target: PgTarget) -> None:
    """Create the target's profile databases."""
    for db in target.dbs:
        psql_sql(ctx, rt, target, PG_MAINTENANCE_DB, f"CREATE DATABASE {quote_ident(db)}")


def bootstrap_db(
    ctx: Context, rt: ResetRuntime, target: PgTarget, db: str, options: ResetOptions
) -> None:
    """Run bootstrap.sql in ``db``; the vector extension is per database."""
    variables = [arg for name, value in options.bootstrap_vars for arg in ("-v", f"{name}={value}")]
    argv = [*_psql_conn(rt, target, db), *variables, "-f", str(BOOTSTRAP_SQL)]
    _run_checked(ctx, rt, argv, f"{target.role}: bootstrap {db}", _psql_env(rt, target))


def migrate_db(
    ctx: Context,
    rt: ResetRuntime,
    target: PgTarget,
    db: str,
    migration: Migration,
    dsn_envs: Sequence[str],
) -> None:
    """Run one migration command against ``db``, its DSN in each of ``dsn_envs``."""
    env = {**rt.env, **{name: target.dsn(db) for name in dsn_envs}}
    what = f"{target.role}: migrate {db} ({redact(' '.join(migration.argv), rt.secrets)})"
    _run_checked(ctx, rt, migration.argv, what, env, migration.cwd, MIGRATE_TIMEOUT_S)


def _timed(label: str, step: Callable[[], None]) -> None:
    started = time.monotonic()
    step()
    sys.stdout.write(f"{label} {time.monotonic() - started:.2f}s\n")
    sys.stdout.flush()


def reset_target(ctx: Context, rt: ResetRuntime, target: PgTarget, options: ResetOptions) -> None:
    """Drop, recreate, bootstrap and migrate one cluster, timing each step."""
    name = target.role
    _timed(f"{name}: drop databases", lambda: drop_databases(ctx, rt, target))
    _timed(f"{name}: drop roles", lambda: drop_roles(ctx, rt, target))
    _timed(f"{name}: create databases", lambda: create_databases(ctx, rt, target))
    for db in target.dbs:
        bootstrap = functools.partial(bootstrap_db, ctx, rt, target, db, options)
        _timed(f"{name}: bootstrap {db}", bootstrap)
        for index, migration in enumerate(options.migrations, start=1):
            step = functools.partial(migrate_db, ctx, rt, target, db, migration, options.dsn_envs)
            command = redact(migration.argv[0], rt.secrets)
            _timed(f"{name}: migrate {db} [{index}] {command}", step)


def cmd_reset(ctx: Context, profile: Profile, options: ResetOptions) -> int:
    """Reset this stack's clusters (or, with --ci, the CI services) to a fresh state."""
    _check_reset_dbs(profile)
    targets = ci_targets(ctx, profile) if options.ci else local_targets(ctx, profile)
    if not targets:
        raise ConfigError("reset needs at least one pg --role")
    rt = reset_runtime(ctx, targets)
    started = time.monotonic()
    for target in targets:
        reset_target(ctx, rt, target, options)
    sys.stdout.write(f"reset {time.monotonic() - started:.2f}s\n")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser; the module docstring documents every option."""
    parser = argparse.ArgumentParser(prog="testdb.py", description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("up", "down", "status", "env"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--repo", required=True)
        cmd.add_argument("--role", action="append", required=True, dest="roles")
        cmd.add_argument("--db", action="append", default=[], dest="dbs")
        if command == "status":
            cmd.add_argument("--require-ready", action="store_true")
    check = sub.add_parser("check-env")
    check.add_argument("--file", type=Path, default=TESTDB_ENV)
    check.add_argument("--github-output", type=Path, default=None)
    psql_path = sub.add_parser("psql-path")
    psql_path.add_argument("--file", type=Path, default=TESTDB_ENV)
    _add_reset_parser(sub)
    return parser


def _add_reset_parser(sub: Any) -> None:
    reset = sub.add_parser("reset")
    reset.add_argument("--repo", required=True)
    reset.add_argument("--role", action="append", required=True, dest="roles")
    reset.add_argument("--db", action="append", default=[], dest="dbs")
    reset.add_argument("--ci", action="store_true")
    reset.add_argument("--bootstrap-var", action="append", default=[], dest="bootstrap_vars")
    reset.add_argument("--migrate-dsn-env", action="append", default=[], dest="dsn_envs")


def _dispatch(
    args: argparse.Namespace, ctx_factory: Callable[[], Context], migrations: Sequence[Migration]
) -> int:
    if args.command == "check-env":
        return cmd_check_env(args.file, args.github_output)
    if args.command == "psql-path":
        return cmd_psql_path(args.file)
    profile = make_profile(args.repo, args.roles, args.dbs)
    ctx = ctx_factory()
    if args.command == "up":
        return cmd_up(ctx, profile)
    if args.command == "down":
        return cmd_down(ctx, profile)
    if args.command == "status":
        return cmd_status(ctx, profile, require_ready=args.require_ready)
    if args.command == "reset":
        options = make_reset_options(args, migrations, ctx.checkout.toplevel)
        return cmd_reset(ctx, profile, options)
    return cmd_env(ctx, profile)


def default_context() -> Context:
    """Build the real-world context: subprocess, wall-clock sleep, this checkout."""
    return Context(
        run=run_command,
        sleep=time.sleep,
        uid=os.getuid(),
        checkout=find_checkout(run_command),
        env_file=TESTDB_ENV,
        run_env=run_command,
        environ=MappingProxyType(dict(os.environ)),
    )


def main(
    argv: Sequence[str] | None = None, ctx_factory: Callable[[], Context] = default_context
) -> int:
    """Parse ``argv`` and run one subcommand; see the module docstring for exit codes."""
    args_in = sys.argv[1:] if argv is None else list(argv)
    if args_in[:1] == [PSQL_PLAN]:
        try:
            return cmd_psql_plan(args_in[1:])
        except ConfigError as exc:
            sys.stderr.write(f"testdb: {exc}\n")
            return EXIT_INVALID
    try:
        head, migrations = split_migrations(args_in)
    except ConfigError as exc:
        sys.stderr.write(f"testdb: {exc}\n")
        return EXIT_INVALID
    args = build_parser().parse_args(head)
    try:
        return _dispatch(args, ctx_factory, migrations)
    except ConfigError as exc:
        sys.stderr.write(f"testdb: {exc}\n")
        return EXIT_INVALID
    except StackError as exc:
        sys.stderr.write(f"testdb: {exc}\n")
        return EXIT_STACK


if __name__ == "__main__":
    sys.exit(main())
