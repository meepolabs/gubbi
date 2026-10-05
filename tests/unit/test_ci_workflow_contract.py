"""Contract tests for the database-backed CI job in security-tests.yml.

The job hands every database-backed suite to one runner call,
``python3 tools/run_db_suites.py --ci``, which owns the stage order and every
stage's environment (its stage table is pinned by
``tests/unit/test_run_db_suites.py``). The service images and the PostgreSQL
client major are not written in the workflow: the ``testdb-config`` job reads
them from ``tools/testdb/testdb.env`` through that file's one validator and
exports them as job outputs. What remains static workflow text -- service env,
port mappings, health options, the runner's endpoint env -- is pinned here.

``tests/integration/test_audit_log_dedup_read_paths.py`` and
``test_audit_log_app_dedup_read.py`` shell out to ``psql`` and to
``deployment/scripts/verify-db-invariants.sh``, and FAIL rather than skip when
``psql`` is absent. A runner image ships whatever client major it happens to
ship, so the job installs the client from PGDG at the pinned major, verifies it,
and only then calls the runner.

Parsed with PyYAML rather than grepped, so a commented-out step or a renamed
job does not pass on a substring match. Comparisons are equality, and a missing
key fails rather than passing as absent.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest
import yaml

from tests.fixtures.testdb_tool import load_testdb_tool

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "security-tests.yml"
_TESTDB_ENV = _ROOT / "tools" / "testdb" / "testdb.env"
_RUNNER = _ROOT / "tools" / "run_db_suites.py"

_CONFIG_JOB = "testdb-config"
_JOB = "security-tests"
_PINS_STEP_ID = "pins"
_PINS_COMMAND = 'python3 tools/testdb/testdb.py check-env --github-output "$GITHUB_OUTPUT"'
_PIN_KEYS = ("PGVECTOR_IMAGE", "PG_MAJOR", "REDIS_IMAGE")
_RUNNER_SCRIPT = "tools/run_db_suites.py"
_RUNNER_COMMAND = f"python3 {_RUNNER_SCRIPT} --ci"
_INSTALL_STEP = "Install PostgreSQL client"
_VERIFY_STEP = "Verify psql major matches the pin"

# The server major the pinned pgvector image provides. testdb.env declares it
# next to the image, since a digest carries no readable version; this constant
# is the test's independent copy, so a bump of one without the other fails.
_EXPECTED_MAJOR = "17"

# The immutable pgvector build every DB contract runs against. Verified out of
# band: it resolves and runs PostgreSQL 17.11 with pgvector 0.8.6. Bumping the
# pin means re-verifying that and updating this constant in the same change.
# Compared as repository + digest, so a ``repo:tag@sha256:...`` value passes.
_PGVECTOR_REPOSITORY = "pgvector/pgvector"
_PGVECTOR_DIGEST = "sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f"
_IMAGE_REFERENCE = re.compile(
    r"(?P<repository>[a-z0-9][a-z0-9._/-]*)(?::[A-Za-z0-9._-]+)?@(?P<digest>sha256:[0-9a-f]{64})"
)

# The service superuser every cluster DSN authenticates as; reset --ci drops
# and recreates roles and databases, so it needs the role the container creates.
_SUPERUSER = "journal"
_SUPERUSER_PASSWORD = "testpass"
_WORKING_PORT = "5433"
_DISPOSABLE_PORT = "5434"


def _needs_output(key: str) -> str:
    return f"${{{{ needs.{_CONFIG_JOB}.outputs.{key} }}}}"


def _health_options(database: str) -> str:
    return (
        f'--health-cmd "pg_isready -U {_SUPERUSER} -d {database}" '
        "--health-interval 5s --health-timeout 3s --health-retries 10"
    )


def _cluster_service(database: str, host_port: str) -> dict[str, Any]:
    return {
        "image": _needs_output("PGVECTOR_IMAGE"),
        "env": {
            "POSTGRES_DB": database,
            "POSTGRES_USER": _SUPERUSER,
            "POSTGRES_PASSWORD": _SUPERUSER_PASSWORD,
        },
        "ports": [f"{host_port}:5432"],
        "options": _health_options(database),
    }


# Two clusters on one image. The missing-required-role contracts need a role
# to be genuinely ABSENT, and roles are cluster-global, so absence gets its own
# throwaway cluster on its own port; the runner hands its DSN to one stage only.
_SERVICES = {
    "postgres": _cluster_service("journal_test", _WORKING_PORT),
    "postgres_disposable": _cluster_service("postgres", _DISPOSABLE_PORT),
}
_HEALTH_FLAGS = ["--health-cmd", "--health-interval", "--health-timeout", "--health-retries"]


def _cluster_url(port: str) -> str:
    return f"postgresql://{_SUPERUSER}:{_SUPERUSER_PASSWORD}@localhost:{port}/postgres"


# The cluster DSNs testdb.py reset --ci reads, and the runner maps onto the
# suite DSN names. Nothing else: suite DSNs set here would bypass the runner's
# per-stage scoping.
_RUNNER_ENV = {
    "TESTDB_PG_URL": _cluster_url(_WORKING_PORT),
    "TESTDB_PG_DISPOSABLE_URL": _cluster_url(_DISPOSABLE_PORT),
}


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


def _workflow() -> dict[str, Any]:
    parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{_WORKFLOW} did not parse as a mapping"
    return parsed


def _job(name: str = _JOB) -> dict[str, Any]:
    jobs = _workflow()["jobs"]
    assert name in jobs, f"workflow no longer defines the {name!r} job: {sorted(jobs)}"
    job = jobs[name]
    assert isinstance(job, dict)
    return job


def _steps(job: str = _JOB) -> list[dict[str, Any]]:
    steps = _job(job)["steps"]
    assert all(isinstance(step, dict) for step in steps)
    return list(steps)


def _index_of(predicate: Callable[[dict[str, Any]], bool], job: str = _JOB) -> int:
    matches = [i for i, step in enumerate(_steps(job)) if predicate(step)]
    assert len(matches) == 1, f"expected exactly one matching step in {job}; found {matches}"
    return matches[0]


def _named(name: str) -> Callable[[dict[str, Any]], bool]:
    return lambda step: step.get("name") == name


def _runs(text: str) -> Callable[[dict[str, Any]], bool]:
    return lambda step: text in str(step.get("run", ""))


def _step(name: str) -> dict[str, Any]:
    return _steps()[_index_of(_named(name))]


def _runner_step() -> dict[str, Any]:
    return _steps()[_index_of(_runs(_RUNNER_SCRIPT))]


def _pins() -> dict[str, str]:
    """testdb.env as its one validator reads it."""
    testdb = load_testdb_tool()
    return dict(testdb.load_pins(_TESTDB_ENV).values)


def _resolved(expression: str) -> str:
    """The value a ``needs.testdb-config.outputs.<KEY>`` expression resolves to in CI.

    Only an exact output expression of a key the config job exports from the
    validator is resolved; anything else is returned unchanged, so a variant
    fails the equality assertions instead of being normalised into the answer.
    """
    for key in _PIN_KEYS:
        if expression == _needs_output(key):
            outputs = _job(_CONFIG_JOB).get("outputs") or {}
            if outputs.get(key) == f"${{{{ steps.{_PINS_STEP_ID}.outputs.{key} }}}}":
                return _pins()[key]
    return expression


# ---------------------------------------------------------------------------
# testdb-config: the pins come from the one validator
# ---------------------------------------------------------------------------


def test_the_config_job_exports_exactly_the_validated_pins() -> None:
    outputs = _job(_CONFIG_JOB)["outputs"]

    assert outputs == {key: f"${{{{ steps.{_PINS_STEP_ID}.outputs.{key} }}}}" for key in _PIN_KEYS}


def test_the_pins_come_from_check_env_and_nothing_else() -> None:
    """check-env is the one validator; a shell regex here would be a second, weaker copy."""
    runs = [step for step in _steps(_CONFIG_JOB) if "run" in step]

    assert len(runs) == 1, f"{_CONFIG_JOB} must run exactly one command; found {runs}"
    assert runs[0].get("id") == _PINS_STEP_ID
    assert runs[0]["run"].strip() == _PINS_COMMAND
    assert "working-directory" not in runs[0]


def test_the_config_job_validates_this_repositorys_own_checkout() -> None:
    def is_checkout(step: dict[str, Any]) -> bool:
        return str(step.get("uses", "")).startswith("actions/checkout@")

    checkout_at = _index_of(is_checkout, _CONFIG_JOB)
    checkout = _steps(_CONFIG_JOB)[checkout_at]

    assert set(checkout.get("with") or {}) <= {"persist-credentials"}
    assert checkout_at < _index_of(lambda step: step.get("id") == _PINS_STEP_ID, _CONFIG_JOB)


def test_the_db_job_needs_the_config_job() -> None:
    assert _job()["needs"] == _CONFIG_JOB


def test_testdb_env_holds_the_verified_pgvector_build_and_major() -> None:
    pins = _pins()
    reference = _IMAGE_REFERENCE.fullmatch(pins["PGVECTOR_IMAGE"])

    assert reference is not None, f"not a digest-pinned reference: {pins['PGVECTOR_IMAGE']}"
    assert (reference["repository"], reference["digest"]) == (
        _PGVECTOR_REPOSITORY,
        _PGVECTOR_DIGEST,
    ), (
        "tools/testdb/testdb.env pgvector pin changed: verify the new image build out of band, "
        "then update _PGVECTOR_DIGEST (and the tag/major) in this test in the same change"
    )
    assert pins["PG_MAJOR"] == _EXPECTED_MAJOR


@pytest.mark.parametrize(
    ("image", "accepted"),
    [
        pytest.param(f"{_PGVECTOR_REPOSITORY}@{_PGVECTOR_DIGEST}", True, id="digest-only"),
        pytest.param(f"{_PGVECTOR_REPOSITORY}:0.8.6-pg17@{_PGVECTOR_DIGEST}", True, id="tagged"),
        pytest.param(f"{_PGVECTOR_REPOSITORY}:pg17", False, id="tag-only"),
        pytest.param(f"{_PGVECTOR_REPOSITORY}@sha256:{'0' * 64}", False, id="other-digest"),
        pytest.param(f"{_PGVECTOR_REPOSITORY}@{_PGVECTOR_DIGEST[:-1]}", False, id="truncated"),
        pytest.param(f"other/pgvector@{_PGVECTOR_DIGEST}", False, id="other-repository"),
    ],
)
def test_the_pgvector_pin_comparison_is_repository_and_digest(image: str, accepted: bool) -> None:
    """Control for the comparison above: a tag may come and go, the build may not change."""
    reference = _IMAGE_REFERENCE.fullmatch(image)
    pair = None if reference is None else (reference["repository"], reference["digest"])

    assert (pair == (_PGVECTOR_REPOSITORY, _PGVECTOR_DIGEST)) is accepted


# ---------------------------------------------------------------------------
# Services: image from the pins, everything else static and exact
# ---------------------------------------------------------------------------


def test_the_services_are_exactly_the_two_pgvector_clusters() -> None:
    assert _job()["services"] == _SERVICES


@pytest.mark.parametrize("service", sorted(_SERVICES))
def test_each_service_image_is_the_validated_pin(service: str) -> None:
    """A digest written here would be a second copy that drifts from testdb.env."""
    image = _job()["services"][service]["image"]

    assert image == _needs_output("PGVECTOR_IMAGE")
    assert _resolved(image) == _pins()["PGVECTOR_IMAGE"]


@pytest.mark.parametrize("service", sorted(_SERVICES))
def test_each_service_waits_on_its_healthcheck(service: str) -> None:
    """Control for the folded-string equality: the options are four flag/value pairs."""
    tokens = shlex.split(str(_job()["services"][service]["options"]))
    options = dict(zip(tokens[::2], tokens[1::2], strict=True))

    assert list(options) == _HEALTH_FLAGS


def test_the_two_clusters_publish_distinct_host_ports() -> None:
    """A shared host port would make the role-dropping cluster the working one."""
    host_ports = [
        str(port).split(":")[0] for service in _SERVICES.values() for port in service["ports"]
    ]

    assert host_ports == [_WORKING_PORT, _DISPOSABLE_PORT]


# ---------------------------------------------------------------------------
# The PostgreSQL client pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_name", [_INSTALL_STEP, _VERIFY_STEP])
def test_client_pin_steps_take_exactly_the_validated_major(step_name: str) -> None:
    """``PG_MAJOR`` is the config output verbatim -- no literal, no variant expression."""
    env = _step(step_name)["env"]

    assert env == {"PG_MAJOR": _needs_output("PG_MAJOR")}
    assert _resolved(env["PG_MAJOR"]) == _EXPECTED_MAJOR


@pytest.mark.parametrize(
    "rejected",
    [
        "17",
        "${{ needs.testdb-config.outputs.PG_MAJOR }}-beta",
        "x${{ needs.testdb-config.outputs.PG_MAJOR }}",
        "${{ needs.testdb-config.outputs.PG_MAJOR_OLD }}",
        "${{ needs.other.outputs.PG_MAJOR }}",
        "${{ env.PGVECTOR_PG_MAJOR }}",
        "$PG_MAJOR",
        "",
    ],
    ids=[
        "bare_literal",
        "suffixed",
        "prefixed",
        "similar_output_name",
        "other_job",
        "job_env_reference",
        "shell_style_reference",
        "empty",
    ],
)
def test_the_major_resolver_rejects_every_variant(rejected: str) -> None:
    """Control for the resolver: each near-miss stays unresolved, so equality fails on it."""
    assert _resolved(rejected) == rejected


def test_postgres_client_install_step_installs_the_versioned_package() -> None:
    run = _step(_INSTALL_STEP)["run"]

    assert '"postgresql-client-${PG_MAJOR}"' in run, (
        f"{_INSTALL_STEP!r} must install the versioned client package from the pin, "
        "not an unversioned postgresql-client that follows the runner image"
    )


def test_postgres_client_install_step_puts_the_pinned_binary_first_on_path() -> None:
    """PGDG installs into ``/usr/lib/postgresql/<major>/bin``, which is not on PATH."""
    run = _step(_INSTALL_STEP)["run"]

    assert re.search(r'/usr/lib/postgresql/\$\{PG_MAJOR\}/bin"?\s*>>\s*"?\$GITHUB_PATH', run), (
        f"{_INSTALL_STEP!r} must append the versioned PGDG bin dir to GITHUB_PATH "
        f"so later steps resolve psql to the pin:\n{run}"
    )


def test_a_psql_major_verification_step_guards_the_pin_at_runtime() -> None:
    """The install can succeed while PATH still resolves another client major."""
    run = _step(_VERIFY_STEP)["run"]

    assert "psql --version" in run
    assert '"${actual}" != "${PG_MAJOR}"' in run
    assert "exit 1" in run, f"{_VERIFY_STEP!r} must fail the job on a mismatch:\n{run}"


# ---------------------------------------------------------------------------
# The runner call
# ---------------------------------------------------------------------------


def test_the_db_job_calls_the_runner_once_with_ci() -> None:
    step = _runner_step()

    assert step["run"].strip() == _RUNNER_COMMAND
    assert _load_runner().parse_args(shlex.split(step["run"])[2:]).ci is True
    assert "working-directory" not in step


@pytest.mark.parametrize(
    "tool",
    ["pytest", "alembic", "CREATE ROLE", "bootstrap.sql", "psql -v", "psql -f", "psql -h"],
)
def test_no_suite_or_bootstrap_runs_inline_in_the_db_job(tool: str) -> None:
    """Every suite, migration and role bootstrap belongs to the runner and its reset."""
    inline = [step.get("name") for step in _steps() if tool in str(step.get("run", ""))]

    assert inline == [], f"{tool!r} runs outside the runner in: {inline}"


def test_the_runner_runs_after_the_client_install_and_its_check() -> None:
    at = _index_of(_runs(_RUNNER_SCRIPT))
    prerequisites = {
        "checkout": _index_of(
            lambda step: str(step.get("uses", "")).startswith("actions/checkout@")
        ),
        "poetry install": _index_of(_runs("poetry install")),
        _INSTALL_STEP: _index_of(_named(_INSTALL_STEP)),
        _VERIFY_STEP: _index_of(_named(_VERIFY_STEP)),
    }

    assert all(index < at for index in prerequisites.values()), (prerequisites, at)


def test_the_runner_env_names_the_jobs_own_clusters() -> None:
    assert _runner_step()["env"] == _RUNNER_ENV


@pytest.mark.parametrize(
    ("variable", "service"),
    [("TESTDB_PG_URL", "postgres"), ("TESTDB_PG_DISPOSABLE_URL", "postgres_disposable")],
)
def test_each_cluster_dsn_is_its_services_superuser(variable: str, service: str) -> None:
    """reset --ci drops roles and databases; it needs the role the container creates."""
    url = urlsplit(_runner_step()["env"][variable])
    spec = _job()["services"][service]
    host_port = str(spec["ports"][0]).split(":")[0]

    assert (url.username, url.password) == (
        spec["env"]["POSTGRES_USER"],
        spec["env"]["POSTGRES_PASSWORD"],
    )
    assert (url.hostname, str(url.port)) == ("localhost", host_port)


def test_the_runner_finds_the_variables_it_requires_at_job_level() -> None:
    runner = _load_runner()
    job_env = _job().get("env") or {}

    assert all(job_env.get(name) for name in runner.REQUIRED_ENV), sorted(job_env)


def test_no_scope_of_the_workflow_sets_a_suite_dsn() -> None:
    """The runner scopes the suite DSNs per stage; set here, they reach every stage."""
    runner = _load_runner()
    workflow = _workflow()
    jobs = list(workflow["jobs"].values())
    scopes = [
        workflow.get("env") or {},
        *((job.get("env") or {}) for job in jobs),
        *((step.get("env") or {}) for job in jobs for step in job.get("steps", [])),
    ]

    for env in scopes:
        assert not set(runner.SUITE_DSN_NAMES) & set(env), f"suite DSN set in the workflow: {env}"
