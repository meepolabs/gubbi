"""Unit tests for the DB suite runner in tools/run_db_suites.py.

No database, docker or pytest child is run except where a test needs a real
child process to prove output redaction; every other stage goes through a fake
runner that records argv and environment and writes a scripted JUnit report.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tools" / "run_db_suites.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "security-tests.yml"
_JOB = "security-tests"
_PASSWORD = "testpass"


def _load_runner() -> ModuleType:
    name = "run_db_suites_tool"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
testdb = runner.testdb


def _stack_env_text() -> str:
    """A ``.testdb.env`` rendered by the controller itself for gubbi's profile."""
    profile = testdb.make_profile(
        "gubbi", ["pg", "pg-disposable"], ["journal_test", "journal_rls_test"]
    )
    names = {"pg": "testdb-gubbi-0000aaaa-pg", "pg-disposable": "testdb-gubbi-0000aaaa-pg-disp"}
    ports = {"pg": 41001, "pg-disposable": 41002}
    return str(testdb.render_stack_env(testdb.stack_env(profile, names, ports)))


def _stack_dsns(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / ".testdb.env"
    path.write_text(_stack_env_text(), encoding="ascii")
    return dict(runner.suite_dsns(runner.read_stack_env(path)))


_CALLER_ENV = {
    "PATH": "/usr/bin",
    "JOURNAL_OPERATOR_EMAIL": "operator@test.local",
    "JOURNAL_ENCRYPTION_MASTER_KEY_V1": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
}


def _plan(tmp_path: Path, environ: Mapping[str, str] | None = None, *, ci: bool = False) -> Any:
    env = dict(_CALLER_ENV if environ is None else environ)
    return runner.Plan(ci, env, _stack_dsns(tmp_path), "/wrapper/bin:/usr/bin")


# ---------------------------------------------------------------------------
# Stage table vs today's workflow
# ---------------------------------------------------------------------------


def _pytest_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    return [s for s in steps if isinstance(s, dict) and "pytest" in str(s.get("run", ""))]


def _step_argv(step: dict[str, Any]) -> tuple[str, ...]:
    return tuple(shlex.split(str(step["run"]).replace("\\\n", " ")))


def test_stage_commands_equal_the_workflow_pytest_invocations() -> None:
    expected = [_step_argv(step) for step in _pytest_steps()]

    actual = [stage.argv for stage in runner.STAGES]

    assert expected, f"no pytest step found in job {_JOB!r}"
    assert actual == expected


def test_stage_scoped_dsns_match_the_workflow_step_env() -> None:
    steps = _pytest_steps()

    step_scoped = [
        sorted(set(step.get("env") or {}) & set(runner.SUITE_DSN_NAMES)) for step in steps
    ]

    assert [sorted(stage.scoped_dsns) for stage in runner.STAGES] == step_scoped


@pytest.mark.parametrize(
    ("needle", "stage_name"),
    [
        ("not hosted_live", "integration"),
        ("--cov-fail-under=80", "coverage"),
        ("--cov=gubbi.core.crypto", "coverage"),
        ("--cov=gubbi.core.cipher_guard", "coverage"),
        ("--cov=gubbi.core.db_context", "coverage"),
        ("tests/storage", "integration"),
    ],
)
def test_each_gate_parameter_belongs_to_its_stage(needle: str, stage_name: str) -> None:
    stages = {stage.name: stage.argv for stage in runner.STAGES}

    holders = [name for name, argv in stages.items() if needle in argv]

    assert holders == [stage_name]


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def test_disposable_cluster_url_reaches_only_the_integration_stage(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    holders = [
        stage.name
        for stage in runner.STAGES
        if runner.DISPOSABLE_DSN in runner.stage_env(stage, plan.environ, plan.dsns, plan.path)
    ]

    assert holders == ["integration"]


def test_an_inherited_disposable_url_is_not_passed_to_other_stages(tmp_path: Path) -> None:
    caller = {**_CALLER_ENV, runner.DISPOSABLE_DSN: "postgresql://x:y@127.0.0.1:1/postgres"}
    plan = _plan(tmp_path, caller)
    coverage = next(stage for stage in runner.STAGES if stage.name == "coverage")

    env = runner.stage_env(coverage, plan.environ, plan.dsns, plan.path)

    assert runner.DISPOSABLE_DSN not in env


def test_stage_env_maps_stack_urls_onto_the_suite_dsn_names(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    integration = runner.STAGES[0]

    env = runner.stage_env(integration, plan.environ, plan.dsns, plan.path)

    base = f"postgresql://journal:{_PASSWORD}@127.0.0.1"
    assert env["TEST_DATABASE_URL"] == f"{base}:41001/journal_test"
    assert env["TEST_DATABASE_URL_RLS"] == f"{base}:41001/journal_rls_test"
    assert env["TEST_DISPOSABLE_CLUSTER_URL"] == f"{base}:41002/postgres"
    assert env["PATH"] == "/wrapper/bin:/usr/bin"
    assert env["JOURNAL_OPERATOR_EMAIL"] == "operator@test.local"


@pytest.mark.parametrize(
    "name", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "TESTDB_PG_URL", "TESTDB_PG_PORT"]
)
def test_stage_env_drops_selection_and_stack_variables(tmp_path: Path, name: str) -> None:
    plan = _plan(tmp_path, {**_CALLER_ENV, name: "-k nothing"})

    envs = [runner.stage_env(s, plan.environ, plan.dsns, plan.path) for s in runner.STAGES]

    assert all(name not in env for env in envs)


def test_ci_dsns_come_from_the_environment() -> None:
    environ = {
        "TESTDB_PG_URL": "postgresql://journal:pw@localhost:5433/postgres",
        "TESTDB_PG_DISPOSABLE_URL": "postgresql://journal:pw@localhost:5434/postgres",
    }

    dsns = runner.suite_dsns(runner.dsn_source(ci=True, environ=environ, stack_env=Path("/no")))

    assert dsns == {
        "TEST_DATABASE_URL": "postgresql://journal:pw@localhost:5433/journal_test",
        "TEST_DATABASE_URL_RLS": "postgresql://journal:pw@localhost:5433/journal_rls_test",
        "TEST_DISPOSABLE_CLUSTER_URL": "postgresql://journal:pw@localhost:5434/postgres",
    }


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({"TESTDB_PG_DISPOSABLE_URL": "postgresql://a@h/p"}, id="working-missing"),
        pytest.param({"TESTDB_PG_URL": "postgresql://a@h/p"}, id="disposable-missing"),
        pytest.param(
            {"TESTDB_PG_URL": "redis://h/0", "TESTDB_PG_DISPOSABLE_URL": "postgresql://a@h/p"},
            id="not-postgres",
        ),
    ],
)
def test_missing_or_foreign_cluster_urls_are_refused(environ: dict[str, str]) -> None:
    with pytest.raises(runner.ConfigError):
        runner.suite_dsns(environ)


def test_a_missing_stack_env_file_names_the_up_command(tmp_path: Path) -> None:
    with pytest.raises(runner.ConfigError, match=r"testdb\.py up"):
        runner.read_stack_env(tmp_path / ".testdb.env")


@pytest.mark.parametrize("name", ["JOURNAL_OPERATOR_EMAIL", "JOURNAL_ENCRYPTION_MASTER_KEY_V1"])
def test_required_job_variables_are_enforced(name: str) -> None:
    environ = {key: value for key, value in _CALLER_ENV.items() if key != name}

    with pytest.raises(runner.ConfigError, match=name):
        runner.check_required_env(environ)


# ---------------------------------------------------------------------------
# Reset command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("ci", "has_ci_flag"), [(False, False), (True, True)])
def test_reset_command_uses_gubbis_profile(ci: bool, has_ci_flag: bool) -> None:
    argv = runner.reset_command(ci=ci)
    migrate = argv.index("--migrate")

    assert argv[1:3] == [str(runner.TESTDB_PY), "reset"]
    assert ("--ci" in argv) is has_ci_flag
    assert argv[migrate:] == ["--migrate", ".", "poetry", "run", "alembic", "upgrade", "head"]
    assert "admin_createrole=true" in argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--role"] == ["pg", "pg-disposable"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--db"] == [
        "journal_test",
        "journal_rls_test",
    ]


def test_reset_command_is_accepted_by_the_controller_parser() -> None:
    head, migrations = testdb.split_migrations(runner.reset_command(ci=True)[2:])

    args = testdb.build_parser().parse_args(head)

    assert args.ci is True
    assert args.dsn_envs == ["JOURNAL_DB_MIGRATION_URL"]
    assert [m.argv for m in migrations] == [("poetry", "run", "alembic", "upgrade", "head")]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _junit(path: Path, tests: int, failures: int, errors: int, skipped: int) -> Path:
    path.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest" '
        f'tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/>'
        "</testsuites>",
        encoding="ascii",
    )
    return path


@pytest.mark.parametrize(
    ("exit_code", "report", "reason"),
    [
        pytest.param(0, (10, 0, 0, 2), "", id="green"),
        pytest.param(1, (10, 1, 0, 0), "pytest exit 1", id="pytest-failed"),
        pytest.param(5, (0, 0, 0, 0), "no tests collected", id="pytest-no-tests"),
        pytest.param(0, (0, 0, 0, 0), "no tests collected", id="empty-report"),
        pytest.param(0, (3, 0, 0, 3), "no tests executed", id="all-skipped"),
        pytest.param(0, (3, 1, 0, 0), "report records failures", id="report-failure"),
        pytest.param(0, (3, 0, 1, 0), "report records failures", id="report-error"),
        pytest.param(0, None, "no report", id="no-report"),
    ],
)
def test_judge_stage(
    tmp_path: Path, exit_code: int, report: tuple[int, int, int, int] | None, reason: str
) -> None:
    junit = tmp_path / "r.xml"
    if report is not None:
        _junit(junit, *report)

    _counts, actual = runner.judge_stage(exit_code, junit)

    assert actual == reason


def test_read_counts_sums_every_testsuite(tmp_path: Path) -> None:
    junit = tmp_path / "r.xml"
    junit.write_text(
        "<testsuites>"
        '<testsuite tests="5" failures="1" errors="0" skipped="1"/>'
        '<testsuite tests="4" failures="0" errors="1" skipped="2"/>'
        "</testsuites>",
        encoding="ascii",
    )

    counts = runner.read_counts(junit)

    assert counts == runner.Counts(passed=4, failed=1, errors=1, skipped=3)
    assert counts.collected == 9


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class FakeRun:
    """Records every child and writes a scripted JUnit report for pytest stages."""

    def __init__(self, exits: Mapping[str, int]) -> None:
        self.exits = exits
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> int:
        self.calls.append((list(argv), dict(env)))
        if "reset" in argv:
            return self.exits.get("reset", 0)
        stage = next(s.name for s in runner.STAGES if list(s.argv) == list(argv[:-1]))
        junit = Path(argv[-1].removeprefix("--junitxml="))
        code = self.exits.get(stage, 0)
        _junit(junit, 4, 1 if code else 0, 0, 1)
        return code


def _ticks() -> Any:
    clock = iter(range(1000))
    return lambda: float(next(clock))


def test_every_stage_runs_after_a_failed_reset(tmp_path: Path) -> None:
    fake = FakeRun({"reset": 1})

    results = runner.run_all(_plan(tmp_path), fake, _ticks())

    assert [r.name for r in results] == ["reset", "integration", "coverage"]
    assert [r.is_green for r in results] == [False, True, True]
    assert runner.summarize(results, seconds=1.0) == runner.EXIT_FAILED


def test_every_stage_runs_after_a_failed_integration_stage(tmp_path: Path) -> None:
    fake = FakeRun({"integration": 1})

    results = runner.run_all(_plan(tmp_path), fake, _ticks())

    assert [r.is_green for r in results] == [True, False, True]
    assert runner.summarize(results, seconds=1.0) == runner.EXIT_FAILED


def test_green_run_exits_zero_and_prints_seconds_and_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = runner.run_all(_plan(tmp_path), FakeRun({}), _ticks())

    code = runner.summarize(results, seconds=3.0)

    out = capsys.readouterr().out
    assert code == runner.EXIT_OK
    assert "integration  PASS" in out
    assert "collected 4: 3 passed, 0 failed, 0 errors, 1 skipped" in out
    assert "1.0s" in out


def test_ci_plan_passes_ci_to_reset(tmp_path: Path) -> None:
    fake = FakeRun({})

    runner.run_all(_plan(tmp_path, ci=True), fake, _ticks())

    reset_argv = fake.calls[0][0]
    assert "reset" in reset_argv
    assert "--ci" in reset_argv


def test_streaming_runner_masks_dsn_passwords(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dsns = _stack_dsns(tmp_path)
    child = "import os; print(os.environ['TEST_DATABASE_URL']); print('pw=' + os.environ['PW'])"
    env = {**dsns, "PW": _PASSWORD, "PATH": "/usr/bin"}

    code = runner.streaming_runner(runner.dsn_passwords(dsns))([sys.executable, "-c", child], env)

    out = capsys.readouterr().out
    assert code == 0
    assert "postgresql://journal:***@127.0.0.1:41001/journal_test" in out
    assert "pw=***" in out
    assert _PASSWORD not in out
