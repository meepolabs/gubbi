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
testdb.env bytes it was started from. Ports are published on 127.0.0.1 only,
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
            included; run ``down`` then ``up`` after testdb.env changes.
``down``    removes this checkout's containers and ``.testdb.env``. It only
            removes a container whose repo, role, owner-uid and checkout labels
            all match; any other container under the name is a refusal and
            nothing is removed. A stale env-sha256 does not block ``down``, and
            ``down`` never reads testdb.env, so a broken pin file cannot block
            teardown.
``status``  prints ``<role> <container> <state>`` per role. With
            ``--require-ready`` it prints nothing on success, and on any miss
            (container missing, stopped, not ready, foreign labels, stale
            env-sha256, docker unavailable, ``.testdb.env`` missing or not
            naming exactly the live containers and ports, as after a container
            restart moved its port) prints exactly this line to stderr:

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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


def run_command(argv: Sequence[str], timeout: float | None) -> CommandResult:
    """Run ``argv`` without a shell; a missing binary or a timeout is a failed result."""
    try:
        done = subprocess.run(  # noqa: S603 - argument array, never a shell
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
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
    must be exact. ``env_digest`` None skips the freshness comparison.
    """
    own = {key: value for key, value in labels.items() if key.startswith(LABEL_PREFIX)}
    expected_keys = {*identity, ENV_DIGEST_LABEL}
    if own.keys() != expected_keys:
        return f"label keys {sorted(own)} differ from {sorted(expected_keys)}"
    differing = sorted(key for key in identity if own[key] != identity[key])
    if differing:
        return f"labels {differing} belong to another stack"
    if not _SHA256_HEX.fullmatch(own[ENV_DIGEST_LABEL]):
        return f"label {ENV_DIGEST_LABEL} is not a sha256 digest"
    if env_digest is not None and own[ENV_DIGEST_LABEL] != env_digest:
        return "started from a different testdb.env (stale env-sha256)"
    return None


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
    return ContainerState(container_id, name, _parse_labels(entry), running, _parse_ports(entry))


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
    argv = ["docker", "run", "--detach", "--name", container_name(profile, ctx.checkout, role)]
    for key, value in labels.items():
        argv += ["--label", f"{key}={value}"]
    argv += ["--publish", f"{LOOPBACK}::{container_port(role).split('/')[0]}"]
    if role_kind(role) == "pg":
        argv += ["--env", f"POSTGRES_USER={PG_USER}", "--env", f"POSTGRES_DB={PG_MAINTENANCE_DB}"]
        argv += ["--env", f"POSTGRES_PASSWORD={PG_PASSWORD}", pins.image("PGVECTOR_IMAGE")]
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
    argv = ["exec", container_id, "psql", "-X", "-q", "-U", PG_USER, "-d", PG_MAINTENANCE_DB]
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
    pg_major = pins.values.get("PG_MAJOR", "")
    if not _PG_MAJOR_RULE.fullmatch(pg_major):
        raise ConfigError(f"{env_file}: PG_MAJOR={pg_major} must match {_PG_MAJOR_RULE.pattern}")
    prefix = psql_path_prefix(run_command, os.environ.get("PATH", ""), pg_major)
    if prefix is not None:
        sys.stdout.write(f"{prefix}\n")
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
    return parser


def _dispatch(args: argparse.Namespace, ctx_factory: Callable[[], Context]) -> int:
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
    return cmd_env(ctx, profile)


def default_context() -> Context:
    """Build the real-world context: subprocess, wall-clock sleep, this checkout."""
    return Context(
        run=run_command,
        sleep=time.sleep,
        uid=os.getuid(),
        checkout=find_checkout(run_command),
        env_file=TESTDB_ENV,
    )


def main(
    argv: Sequence[str] | None = None, ctx_factory: Callable[[], Context] = default_context
) -> int:
    """Parse ``argv`` and run one subcommand; see the module docstring for exit codes."""
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args, ctx_factory)
    except ConfigError as exc:
        sys.stderr.write(f"testdb: {exc}\n")
        return EXIT_INVALID
    except StackError as exc:
        sys.stderr.write(f"testdb: {exc}\n")
        return EXIT_STACK


if __name__ == "__main__":
    sys.exit(main())
