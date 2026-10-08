"""Run every DB-backed pytest stage of gubbi against a freshly reset test stack.

Stages, in order, each run even after an earlier one fails:

``reset``        ``tools/testdb/testdb.py reset`` with gubbi's profile (working
                 cluster ``pg`` with journal_test and journal_rls_test, plus the
                 ``pg-disposable`` cluster), journal_admin keeping CREATEROLE,
                 and ``alembic upgrade head`` as the migration.
``integration``  the integration, e2e, api, extraction, security and storage
                 suites, ``-m "not hosted_live"``. The only stage that sees
                 ``TEST_DISPOSABLE_CLUSTER_URL``: the tests it gates drop roles.
``coverage``     the crypto + RLS coverage gate (``--cov-fail-under=80``).

Usage::

    python3 tools/run_db_suites.py [--ci]

DSNs. Locally they come from ``.testdb.env`` at the checkout root (written by
``testdb.py up``). With ``--ci`` they come from the environment, under the names
``testdb.py reset --ci`` reads: ``TESTDB_PG_URL`` and
``TESTDB_PG_DISPOSABLE_URL``, and ``--ci`` is passed on to reset. Either way the
working cluster URL's database is replaced to give ``TEST_DATABASE_URL``
(journal_test) and ``TEST_DATABASE_URL_RLS`` (journal_rls_test), and the
disposable cluster URL is ``TEST_DISPOSABLE_CLUSTER_URL``.

The caller supplies ``JOURNAL_OPERATOR_EMAIL`` and
``JOURNAL_ENCRYPTION_MASTER_KEY_V1``; the runner refuses to start without them.
Every pytest child gets ``testdb.py psql-path`` prepended to PATH (its stderr
hint, if any, is relayed), and runs without ``PYTEST_ADDOPTS``,
``PYTEST_PLUGINS``, any ``TESTDB_*`` variable and any inherited suite DSN, so
neither the caller's environment nor a stale DSN can change a stage's
selection or target.

A pytest stage is green only when pytest exits 0 and its JUnit report parses,
counts at least one test, records no failures or errors and at least one
passed test. A command that cannot start counts as exit 127. Every child's
output, and every summary reason, is relayed with URL passwords masked, so no
DSN password reaches the output. The summary prints each stage's
seconds and collected/passed/failed/errors/skipped counts.

Exit 0 when every stage is green, 1 when any stage failed, 2 on a usage or
configuration error (nothing runs).

Standard-library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTDB_PY = REPO_ROOT / "tools" / "testdb" / "testdb.py"
STACK_ENV_PATH = REPO_ROOT / ".testdb.env"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
PYTEST_NO_TESTS = 5
EXIT_NOT_STARTED = 127

PROFILE = (
    "--repo",
    "gubbi",
    "--role",
    "pg",
    "--role",
    "pg-disposable",
    "--db",
    "journal_test",
    "--db",
    "journal_rls_test",
)
RESET_OPTIONS = (
    "--bootstrap-var",
    "admin_createrole=true",
    "--bootstrap-var",
    "with_otel_ro=true",
)
MIGRATION = (
    "--migrate-dsn-env",
    "JOURNAL_DB_MIGRATION_URL",
    "--migrate",
    ".",
    "poetry",
    "run",
    "alembic",
    "upgrade",
    "head",
)

WORKING_CLUSTER_SOURCE = "TESTDB_PG_URL"
DISPOSABLE_CLUSTER_SOURCE = "TESTDB_PG_DISPOSABLE_URL"
WORKING_DSNS = {"TEST_DATABASE_URL": "journal_test", "TEST_DATABASE_URL_RLS": "journal_rls_test"}
DISPOSABLE_DSN = "TEST_DISPOSABLE_CLUSTER_URL"
SUITE_DSN_NAMES = (*WORKING_DSNS, DISPOSABLE_DSN)
REQUIRED_ENV = ("JOURNAL_OPERATOR_EMAIL", "JOURNAL_ENCRYPTION_MASTER_KEY_V1")
DROPPED_PYTEST_VARS = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
DROPPED_PREFIX = "TESTDB_"

PYTEST = ("poetry", "run", "pytest")
SUITE_DIRS = (
    "tests/integration",
    "tests/e2e",
    "tests/api",
    "tests/extraction",
    "tests/security",
    "tests/storage",
)
COVERAGE_FILES = (
    "tests/unit",
    "tests/integration/test_rls_isolation.py",
    "tests/integration/test_entries_encryption.py",
    "tests/integration/test_encryption_contract.py",
    "tests/integration/test_encryption_correctness.py",
    "tests/integration/test_repo_inserts_under_rls.py",
)
COVERAGE_MODULES = ("gubbi.core.crypto", "gubbi.core.cipher_guard", "gubbi.core.db_context")
COVERAGE_THRESHOLD = 80
JUNIT_COUNT_KEYS = ("tests", "failures", "errors", "skipped")


@dataclass(frozen=True)
class Stage:
    """One pytest invocation and the step-scoped DSN names only it receives."""

    name: str
    argv: tuple[str, ...]
    scoped_dsns: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage(
        "integration",
        (*PYTEST, *SUITE_DIRS, "-m", "not hosted_live", "--tb=short"),
        (DISPOSABLE_DSN,),
    ),
    Stage(
        "coverage",
        (
            *PYTEST,
            *COVERAGE_FILES,
            *(f"--cov={module}" for module in COVERAGE_MODULES),
            "--cov-report=term-missing",
            f"--cov-fail-under={COVERAGE_THRESHOLD}",
        ),
    ),
)
RESET_STAGE = "reset"


class ConfigError(Exception):
    """The runner cannot start: a DSN or required variable is missing or malformed."""


class ReportError(Exception):
    """A JUnit report is missing or does not carry usable counts."""


class Counts(NamedTuple):
    """Test outcome counts from one pytest JUnit report."""

    passed: int
    failed: int
    errors: int
    skipped: int

    @property
    def collected(self) -> int:
        """Every test the report counted, whatever its outcome."""
        return self.passed + self.failed + self.errors + self.skipped


class StageResult(NamedTuple):
    """What one stage did; ``reason`` is empty only when the stage is green."""

    name: str
    exit_code: int
    seconds: float
    counts: Counts | None
    reason: str

    @property
    def is_green(self) -> bool:
        """True only for a stage with no failure reason."""
        return not self.reason


def _load_testdb() -> Any:
    name = "testdb_tool"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TESTDB_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {TESTDB_PY}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolve their module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


testdb = _load_testdb()


def say(line: str) -> None:
    """Write one line now, so it lands in order with the children's output."""
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# DSNs and environments
# ---------------------------------------------------------------------------


def read_stack_env(path: Path) -> dict[str, str]:
    """Parse the ``.testdb.env`` that ``testdb.py up`` wrote."""
    try:
        text = path.read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise ConfigError(f"{path.name} not found: start the stack with testdb.py up") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path.name} is unreadable: {type(exc).__name__}") from exc
    try:
        pairs: dict[str, str] = testdb.parse_stack_env(text)
    except ValueError as exc:
        raise ConfigError(f"{path.name} is malformed; rerun testdb.py up") from exc
    return pairs


def with_database(url: str, db: str) -> str:
    """Return ``url`` with its path replaced by ``/db``."""
    return urllib.parse.urlsplit(url)._replace(path=f"/{db}").geturl()


def _cluster_url(source: Mapping[str, str], name: str) -> str:
    url = source.get(name, "")
    if not url:
        raise ConfigError(f"{name} is not set")
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid URL") from exc
    if parts.scheme not in {"postgres", "postgresql"} or not parts.hostname:
        raise ConfigError(f"{name} must be a postgresql:// URL with a host")
    return url


def suite_dsns(source: Mapping[str, str]) -> dict[str, str]:
    """Map the cluster URLs in ``source`` onto the DSN names the suites read."""
    working = _cluster_url(source, WORKING_CLUSTER_SOURCE)
    disposable = _cluster_url(source, DISPOSABLE_CLUSTER_SOURCE)
    dsns = {name: with_database(working, db) for name, db in WORKING_DSNS.items()}
    return {**dsns, DISPOSABLE_DSN: disposable}


def dsn_source(*, ci: bool, environ: Mapping[str, str], stack_env: Path) -> Mapping[str, str]:
    """Return where the cluster URLs come from: the environment in CI, else ``.testdb.env``."""
    return environ if ci else read_stack_env(stack_env)


def check_required_env(environ: Mapping[str, str]) -> None:
    """Refuse to start without the job-level variables the suites expect."""
    missing = [name for name in REQUIRED_ENV if not environ.get(name)]
    if missing:
        raise ConfigError(f"set {', '.join(missing)} (security-tests.yml job env)")


def dsn_passwords(dsns: Mapping[str, str]) -> tuple[str, ...]:
    """Return every password the DSNs carry, longest first so redaction masks it whole."""
    found = {urllib.parse.urlsplit(url).password for url in dsns.values()}
    return tuple(sorted((p for p in found if p), key=len, reverse=True))


def stage_env(
    stage: Stage, environ: Mapping[str, str], dsns: Mapping[str, str], path: str
) -> dict[str, str]:
    """Return a pytest child's environment: the inherited one cleaned, plus this stage's DSNs."""
    dropped = {*DROPPED_PYTEST_VARS, *SUITE_DSN_NAMES}
    env = {
        key: value
        for key, value in environ.items()
        if key not in dropped and not key.startswith(DROPPED_PREFIX)
    }
    env |= {name: dsns[name] for name in WORKING_DSNS}
    env |= {name: dsns[name] for name in stage.scoped_dsns}
    return {**env, "PATH": path}


def psql_path(environ: Mapping[str, str]) -> str:
    """Return PATH with the psql wrapper dir prepended when no host psql of PG_MAJOR exists."""
    done = subprocess.run(  # noqa: S603 - argument array, this interpreter
        [sys.executable, str(TESTDB_PY), "psql-path"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(environ),
    )
    if done.returncode != 0:
        raise ConfigError(f"testdb.py psql-path failed: {done.stderr.strip()}")
    sys.stderr.write(done.stderr)
    prefix = done.stdout.strip()
    path = environ.get("PATH", "")
    return os.pathsep.join(p for p in (prefix, path) if p)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def reset_command(*, ci: bool) -> list[str]:
    """Build the ``testdb.py reset`` argv for gubbi's profile."""
    mode = ["--ci"] if ci else []
    return [sys.executable, str(TESTDB_PY), "reset", *PROFILE, *mode, *RESET_OPTIONS, *MIGRATION]


def streaming_runner(secrets: Sequence[str]) -> Callable[[Sequence[str], Mapping[str, str]], int]:
    """Return a runner that relays a child's output line by line with passwords masked."""

    def run(argv: Sequence[str], env: Mapping[str, str]) -> int:
        with subprocess.Popen(  # noqa: S603 - argument array built here
            list(argv),
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as proc:
            assert proc.stdout is not None  # noqa: S101 - stdout=PIPE guarantees it
            for line in proc.stdout:
                sys.stdout.write(testdb.redact(line, secrets))
                sys.stdout.flush()
        return proc.returncode

    return run


# ---------------------------------------------------------------------------
# JUnit counts and verdicts
# ---------------------------------------------------------------------------


def _suite_count(suite: ET.Element, key: str) -> int:
    # Reasons name the attribute only: its value is unvetted report content.
    value = suite.get(key)
    if value is None:
        raise ReportError(f"report count {key} missing")
    try:
        return int(value)
    except ValueError as exc:
        raise ReportError(f"report count {key} malformed") from exc


def _suite_counts(suite: ET.Element) -> tuple[int, int, int, int]:
    tests, failures, errors, skipped = (_suite_count(suite, key) for key in JUNIT_COUNT_KEYS)
    if min(tests, failures, errors, skipped) < 0 or failures + errors + skipped > tests:
        raise ReportError("inconsistent report counts")
    return tests, failures, errors, skipped


def read_counts(junit: Path) -> Counts:
    """Sum the counts of every testsuite in a JUnit report; raise ReportError if unusable."""
    try:
        root = ET.parse(junit).getroot()  # noqa: S314 - report pytest just wrote
    except OSError as exc:
        raise ReportError("no report") from exc
    except ET.ParseError as exc:
        raise ReportError("unparseable report") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise ReportError("report has no testsuite")
    passed = failed = errors = skipped = 0
    for suite in suites:
        tests, s_failed, s_errors, s_skipped = _suite_counts(suite)
        passed += tests - s_failed - s_errors - s_skipped
        failed, errors, skipped = failed + s_failed, errors + s_errors, skipped + s_skipped
    return Counts(passed, failed, errors, skipped)


def judge_stage(exit_code: int, junit: Path) -> tuple[Counts | None, str]:
    """Return the report counts and the failure reason ("" only when the stage is green)."""
    try:
        counts: Counts | None = read_counts(junit)
        report_reason = ""
    except ReportError as exc:
        counts, report_reason = None, str(exc)
    if exit_code == PYTEST_NO_TESTS:
        return counts, "no tests collected"
    if exit_code != 0:
        return counts, f"pytest exit {exit_code}"
    if counts is None:
        return None, report_reason
    if counts.collected == 0:
        return counts, "no tests collected"
    if counts.failed or counts.errors:
        return counts, "report records failures"
    if counts.passed == 0:
        return counts, "no tests executed"
    return counts, ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Everything the stages need, resolved before any of them runs."""

    ci: bool
    environ: Mapping[str, str]
    dsns: Mapping[str, str]
    path: str


def start_command(
    name: str,
    plan: Plan,
    run: Callable[[Sequence[str], Mapping[str, str]], int],
    argv: Sequence[str],
    env: Mapping[str, str],
) -> int:
    """Run one stage's command; a command that cannot start is exit 127, not a crash."""
    try:
        return run(argv, env)
    except OSError as exc:
        line = f"db-suites: stage {name} could not start {argv[0]}: {exc.strerror}"
        say(testdb.redact(line, dsn_passwords(plan.dsns)))
        return EXIT_NOT_STARTED


def run_reset(
    plan: Plan,
    run: Callable[[Sequence[str], Mapping[str, str]], int],
    clock: Callable[[], float],
) -> StageResult:
    """Reset the stack's clusters; the stage is green when reset exits 0."""
    say(f"db-suites: stage {RESET_STAGE}")
    start = clock()
    exit_code = start_command(RESET_STAGE, plan, run, reset_command(ci=plan.ci), plan.environ)
    reason = "" if exit_code == 0 else f"reset exit {exit_code}"
    return StageResult(RESET_STAGE, exit_code, clock() - start, None, reason)


def run_stage(
    stage: Stage,
    plan: Plan,
    run: Callable[[Sequence[str], Mapping[str, str]], int],
    clock: Callable[[], float],
    workdir: Path,
) -> StageResult:
    """Run one pytest stage with its own environment and JUnit report."""
    say(f"db-suites: stage {stage.name}")
    junit = workdir / f"{stage.name}.xml"
    env = stage_env(stage, plan.environ, plan.dsns, plan.path)
    start = clock()
    exit_code = start_command(stage.name, plan, run, [*stage.argv, f"--junitxml={junit}"], env)
    seconds = clock() - start
    counts, reason = judge_stage(exit_code, junit)
    return StageResult(stage.name, exit_code, seconds, counts, reason)


def run_all(
    plan: Plan,
    run: Callable[[Sequence[str], Mapping[str, str]], int],
    clock: Callable[[], float] = time.monotonic,
) -> list[StageResult]:
    """Run the reset and every pytest stage, never stopping at a failure."""
    ignored = [name for name in DROPPED_PYTEST_VARS if name in plan.environ]
    if ignored:
        say(f"db-suites: ignoring {' '.join(ignored)} for every pytest stage")
    results = [run_reset(plan, run, clock)]
    with tempfile.TemporaryDirectory(prefix="db-suites-") as workdir:
        results += [run_stage(stage, plan, run, clock, Path(workdir)) for stage in STAGES]
    return results


def format_result(result: StageResult, secrets: Sequence[str] = ()) -> str:
    """Render one summary line: stage, verdict, seconds and counts, with secrets masked."""
    verdict = "PASS" if result.is_green else f"FAIL ({testdb.redact(result.reason, secrets)})"
    line = f"  {result.name:<12} {verdict:<32} {result.seconds:8.1f}s"
    if result.counts is None:
        return line
    c = result.counts
    return (
        f"{line}  collected {c.collected}: {c.passed} passed, {c.failed} failed, "
        f"{c.errors} errors, {c.skipped} skipped"
    )


def summarize(
    results: Sequence[StageResult], *, seconds: float, secrets: Sequence[str] = ()
) -> int:
    """Print the per-stage summary and return the overall exit code."""
    say("db-suites: summary")
    for result in results:
        say(format_result(result, secrets))
    say(f"  {'total':<12} {'':<32} {seconds:8.1f}s")
    is_complete = len(results) == len(STAGES) + 1
    return EXIT_OK if is_complete and all(r.is_green for r in results) else EXIT_FAILED


def make_plan(*, ci: bool, environ: Mapping[str, str]) -> Plan:
    """Resolve DSNs, required variables and PATH; raise ConfigError before anything runs."""
    check_required_env(environ)
    dsns = suite_dsns(dsn_source(ci=ci, environ=environ, stack_env=STACK_ENV_PATH))
    return Plan(ci, environ, dsns, psql_path(environ))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--ci", action="store_true", help="read DSNs from the environment")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every stage; see the module docstring for the exit code."""
    args = parse_args(argv)
    try:
        plan = make_plan(ci=args.ci, environ=dict(os.environ))
    except ConfigError as exc:
        sys.stderr.write(f"db-suites: {exc}\n")
        return EXIT_USAGE
    start = time.monotonic()
    secrets = dsn_passwords(plan.dsns)
    results = run_all(plan, streaming_runner(secrets))
    return summarize(results, seconds=time.monotonic() - start, secrets=secrets)


if __name__ == "__main__":
    sys.exit(main())
