"""Unit tests for ``testdb.py reset``.

Neither docker nor psql is run: the controller tests' fake docker answers
``docker container inspect``, and a fake psql runner answers the role and
database list queries and records every psql and migration argv with its
environment and working directory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.testdb_tool import load_testdb_tool
from tests.unit.test_testdb_controller import (
    _ENV_BYTES,
    _TOPLEVEL,
    _UID,
    FakeDocker,
    _add_stack,
    _labels,
    _name,
    _port,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

pytestmark = pytest.mark.unit

testdb = load_testdb_tool()

_PROFILE = ["--repo", "gubbi", "--role", "pg", "--role", "pg-disposable"]
_DBS = ["--db", "journal_test", "--db", "journal_rls_test"]
_ROLES = [
    {"name": "journal", "super": True},
    {"name": "journal_admin", "super": False},
    {"name": "journal_app", "super": False},
    {"name": "other_super", "super": True},
    {"name": "pg_monitor", "super": False},
    {"name": "zz_leftover", "super": False},
]
_SURVIVING_DBS = ["postgres", "template1"]


@dataclass
class Call:
    """One psql or migration invocation."""

    argv: list[str]
    env: dict[str, str]
    cwd: Path | None

    @property
    def sql(self) -> str | None:
        return self.argv[self.argv.index("-tAc") + 1] if "-tAc" in self.argv else None

    @property
    def db(self) -> str:
        return self.argv[self.argv.index("-d") + 1]


@dataclass
class FakePsql:
    """Answers psql and migration commands; fails the call whose argv holds ``fail_on``."""

    psql: str
    calls: list[Call] = field(default_factory=list)
    roles: list[dict[str, Any]] = field(default_factory=lambda: list(_ROLES))
    fail_on: str | None = None
    fail_stderr: str = ""

    def __call__(
        self,
        argv: Sequence[str],
        _timeout: float | None,
        env: Mapping[str, str],
        cwd: Path | None,
    ) -> Any:
        if list(argv[1:]) == ["--version"]:
            return testdb.CommandResult(0, "psql (PostgreSQL) 17.11\n", "")
        call = Call(list(argv), dict(env), cwd)
        self.calls.append(call)
        if self.fail_on is not None and any(self.fail_on in arg for arg in argv):
            return testdb.CommandResult(1, "", self.fail_stderr)
        if call.sql == testdb.ROLE_LIST_SQL:
            return testdb.CommandResult(0, json.dumps(self.roles) + "\n", "")
        if call.sql == testdb.DB_LIST_SQL:
            return testdb.CommandResult(0, json.dumps(_SURVIVING_DBS) + "\n", "")
        return testdb.CommandResult(0, "", "")

    def psql_calls(self) -> list[Call]:
        return [call for call in self.calls if call.argv[0] == self.psql]

    def sql_calls(self, port: str | None = None) -> list[Call]:
        return [
            call
            for call in self.psql_calls()
            if call.sql is not None and (port is None or _arg(call, "-p") == port)
        ]

    def bootstraps(self) -> list[Call]:
        return [call for call in self.psql_calls() if "-f" in call.argv]


def _arg(call: Call, flag: str) -> str:
    return call.argv[call.argv.index(flag) + 1]


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "testdb.env"
    path.write_bytes(_ENV_BYTES)
    return path


@pytest.fixture
def docker() -> FakeDocker:
    return FakeDocker()


@pytest.fixture
def psql_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "hostbin"
    bin_dir.mkdir()
    psql = bin_dir / "psql"
    psql.write_text("#!/bin/sh\nexit 1\n")
    psql.chmod(0o755)
    return bin_dir


@pytest.fixture
def fake_psql(psql_bin: Path) -> FakePsql:
    return FakePsql(psql=str(psql_bin / "psql"))


@pytest.fixture
def toplevel(tmp_path: Path) -> Path:
    path = tmp_path / "checkout"
    (path / "sibling").mkdir(parents=True)
    return path


def _reset(
    docker: FakeDocker,
    psql: FakePsql,
    env_file: Path,
    toplevel: Path,
    argv: list[str],
    environ: dict[str, str] | None = None,
) -> int:
    path = os.path.dirname(psql.psql)
    ctx = testdb.Context(
        run=docker,
        sleep=lambda _seconds: None,
        uid=_UID,
        checkout=testdb.checkout_from_toplevel(toplevel),
        env_file=env_file,
        run_env=psql,
        environ={"PATH": path, **(environ or {})},
    )
    return int(testdb.main(["reset", *argv], ctx_factory=lambda: ctx))


@pytest.fixture
def stack(docker: FakeDocker, toplevel: Path) -> FakeDocker:
    _add_stack(docker, toplevel)
    return docker


# ---------------------------------------------------------------------------
# Refusals: nothing runs
# ---------------------------------------------------------------------------


def test_reset_refuses_a_container_with_foreign_labels(
    docker: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    docker.add(_name("pg", toplevel), _labels("pg", _TOPLEVEL))
    docker.add(_name("pg-disposable", toplevel), _labels("pg-disposable", toplevel))

    rc = _reset(docker, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    assert rc == 1
    assert fake_psql.calls == []


def test_reset_refuses_a_missing_container_and_never_starts_one(
    docker: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    docker.add(_name("pg", toplevel), _labels("pg", toplevel), running=False)
    docker.add(_name("pg-disposable", toplevel), _labels("pg-disposable", toplevel))

    rc = _reset(docker, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    assert rc == 1
    assert fake_psql.calls == []
    assert [call[1] for call in docker.calls] == ["container", "container"]


@pytest.mark.parametrize("db", ["postgres", "template0", "template1"])
def test_reset_refuses_to_drop_a_protected_database(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path, db: str
) -> None:
    rc = _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, "--db", db])

    assert rc == 2
    assert fake_psql.calls == []


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({}, id="unset"),
        pytest.param({"GITHUB_ACTIONS": "false"}, id="false"),
        pytest.param({"GITHUB_ACTIONS": "1"}, id="not-the-literal-true"),
    ],
)
def test_reset_ci_refuses_without_github_actions_true(
    docker: FakeDocker,
    fake_psql: FakePsql,
    env_file: Path,
    toplevel: Path,
    environ: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    urls = {"TESTDB_PG_URL": "postgresql://journal:pw@localhost:5433/postgres"}

    rc = _reset(docker, fake_psql, env_file, toplevel, ["--ci", *_PROFILE[:4]], urls | environ)

    assert rc == 2
    assert "GITHUB_ACTIONS=true" in capsys.readouterr().err
    assert fake_psql.calls == []
    assert docker.calls == []


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("postgresql://journal:pw@db.example.com:5432/postgres", id="remote-host"),
        pytest.param("postgresql://journal:pw@10.0.0.5:5432/postgres", id="private-ip"),
        pytest.param("postgresql://journal:pw@127.0.0.1.example.com/postgres", id="lookalike"),
    ],
)
def test_reset_ci_refuses_a_non_loopback_host(
    docker: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path, url: str
) -> None:
    environ = {"GITHUB_ACTIONS": "true", "TESTDB_PG_URL": url}

    rc = _reset(docker, fake_psql, env_file, toplevel, ["--ci", *_PROFILE[:4]], environ)

    assert rc == 1
    assert fake_psql.calls == []


def test_reset_ci_refuses_a_missing_dsn(
    docker: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    environ = {"GITHUB_ACTIONS": "true", "TESTDB_PG_URL": "postgresql://journal@localhost/x"}

    rc = _reset(docker, fake_psql, env_file, toplevel, ["--ci", *_PROFILE], environ)

    assert rc == 2
    assert fake_psql.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--bootstrap-var", "admin_password=x"], id="password-var"),
        pytest.param(["--bootstrap-var", "admin_createrole=maybe"], id="non-boolean"),
        pytest.param(["--migrate", "."], id="migrate-without-command"),
        pytest.param(["--migrate", ".", "alembic"], id="migrate-without-dsn-env"),
        pytest.param(
            ["--migrate-dsn-env", "X", "--migrate", "missing", "alembic"], id="missing-dir"
        ),
        pytest.param(["--migrate-dsn-env", "lower", "--migrate", ".", "a"], id="bad-env-name"),
    ],
)
def test_reset_rejects_bad_options_before_running_anything(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path, extra: list[str]
) -> None:
    rc = _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS, *extra])

    assert rc == 2
    assert fake_psql.calls == []


# ---------------------------------------------------------------------------
# What is dropped, recreated and bootstrapped
# ---------------------------------------------------------------------------


def test_reset_drops_only_non_superuser_non_pg_roles_other_than_the_bootstrap_user(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    rc = _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    drops = [c.sql for c in fake_psql.sql_calls() if c.sql and c.sql.startswith("DROP ROLE")]
    assert rc == 0
    assert drops == ['DROP ROLE "journal_admin", "journal_app", "zz_leftover"'] * 2


def test_reset_clears_owned_objects_in_every_surviving_database_before_dropping_roles(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])
    port = _port(stack, _name("pg", toplevel))

    calls = fake_psql.sql_calls(port)
    owned = [i for i, c in enumerate(calls) if c.sql and c.sql.startswith("REASSIGN OWNED")]
    drop = next(i for i, c in enumerate(calls) if c.sql and c.sql.startswith("DROP ROLE"))
    assert [calls[i].db for i in owned] == _SURVIVING_DBS
    assert calls[owned[0]].sql == (
        'REASSIGN OWNED BY "journal_admin", "journal_app", "zz_leftover" TO "journal"; '
        'DROP OWNED BY "journal_admin", "journal_app", "zz_leftover"'
    )
    assert max(owned) < drop


def test_reset_skips_the_role_drop_when_no_role_is_droppable(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    fake_psql.roles = [{"name": "journal", "super": True}, {"name": "pg_read", "super": False}]

    rc = _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    assert rc == 0
    assert not [c for c in fake_psql.sql_calls() if "OWNED" in (c.sql or "")]


def test_reset_runs_the_steps_in_order_on_the_inspected_port(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    port = _port(stack, _name("pg", toplevel))
    environ = {"TESTDB_PG_PORT": "1", "TESTDB_PG_HOST": "10.0.0.9", "PGHOST": "10.0.0.9"}

    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS], environ)

    steps = [
        (c.sql or "bootstrap").split(" ")[0] + " " + c.db
        for c in fake_psql.psql_calls()
        if _arg(c, "-p") == port and not (c.sql or "").startswith(("SELECT", "REASSIGN"))
    ]
    assert steps == [
        "DROP postgres",
        "DROP postgres",
        "DROP postgres",
        "CREATE postgres",
        "CREATE postgres",
        "bootstrap journal_test",
        "bootstrap journal_rls_test",
    ]
    assert {_arg(c, "-h") for c in fake_psql.psql_calls()} == {"127.0.0.1"}
    assert all("PGHOST" not in c.env for c in fake_psql.calls)


def test_reset_drops_and_recreates_exactly_the_profile_databases(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    sqls = [c.sql or "" for c in fake_psql.sql_calls()]
    assert [s for s in sqls if "DATABASE" in s] == [
        'DROP DATABASE IF EXISTS "journal_test" WITH (FORCE)',
        'DROP DATABASE IF EXISTS "journal_rls_test" WITH (FORCE)',
        'CREATE DATABASE "journal_test"',
        'CREATE DATABASE "journal_rls_test"',
    ]


def test_reset_bootstraps_each_profile_database_exactly_once(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    boots = fake_psql.bootstraps()
    assert sorted(c.db for c in boots) == ["journal_rls_test", "journal_test"]
    assert {_arg(c, "-p") for c in boots} == {_port(stack, _name("pg", toplevel))}
    assert all(_arg(c, "-f") == str(testdb.BOOTSTRAP_SQL) for c in boots)
    assert all(_arg(c, "-v") == "ON_ERROR_STOP=1" for c in boots)


def test_reset_passes_bootstrap_variables_to_every_bootstrap(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    extra = ["--bootstrap-var", "admin_createrole=false", "--bootstrap-var", "with_otel_ro=on"]

    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS, *extra])

    for call in fake_psql.bootstraps():
        values = [call.argv[i + 1] for i, arg in enumerate(call.argv) if arg == "-v"]
        assert values == ["ON_ERROR_STOP=1", "admin_createrole=false", "with_otel_ro=on"]


def test_reset_passes_the_superuser_password_only_in_the_environment(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    _reset(stack, fake_psql, env_file, toplevel, [*_PROFILE, *_DBS])

    assert all(c.env["PGPASSWORD"] == testdb.PG_PASSWORD for c in fake_psql.psql_calls())
    assert not [c for c in fake_psql.psql_calls() if any(testdb.PG_PASSWORD in a for a in c.argv)]


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_reset_runs_migration_chains_in_order_per_database_without_a_shell(
    stack: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    chains = ["--migrate", "sibling", "alembic", "-c", "a b.ini", "upgrade", "head"]
    chains += ["--migrate", ".", "poetry", "run", "alembic", "upgrade", "head"]
    environ = {"JOURNAL_DB_MIGRATION_URL": "postgresql://x:y@db.example.com/prod", "KEEP": "1"}
    argv = [*_PROFILE, *_DBS, "--migrate-dsn-env", "JOURNAL_DB_ADMIN_URL", *chains]

    rc = _reset(stack, fake_psql, env_file, toplevel, argv, environ)

    port = _port(stack, _name("pg", toplevel))
    runs = [c for c in fake_psql.calls if c.argv[0] != fake_psql.psql]
    assert rc == 0
    assert [(c.argv, c.cwd) for c in runs] == [
        (["alembic", "-c", "a b.ini", "upgrade", "head"], toplevel / "sibling"),
        (["poetry", "run", "alembic", "upgrade", "head"], toplevel / "."),
    ] * 2
    dsn = f"postgresql://journal:testpass@127.0.0.1:{port}/"
    assert [c.env["JOURNAL_DB_ADMIN_URL"] for c in runs] == [
        f"{dsn}journal_test",
        f"{dsn}journal_test",
        f"{dsn}journal_rls_test",
        f"{dsn}journal_rls_test",
    ]
    assert all("JOURNAL_DB_MIGRATION_URL" not in c.env and c.env["KEEP"] == "1" for c in runs)


def test_split_migrations_leaves_argv_without_migrate_untouched() -> None:
    argv = ["reset", "--repo", "gubbi", "--role", "pg"]

    assert testdb.split_migrations(argv) == (argv, ())


def test_split_migrations_only_applies_to_reset() -> None:
    argv = ["up", "--repo", "gubbi", "--role", "pg", "--migrate", ".", "x"]

    assert testdb.split_migrations(argv) == (argv, ())


# ---------------------------------------------------------------------------
# CI mode
# ---------------------------------------------------------------------------


def test_reset_ci_targets_the_environment_dsns_and_never_calls_docker(
    docker: FakeDocker, fake_psql: FakePsql, env_file: Path, toplevel: Path
) -> None:
    environ = {
        "GITHUB_ACTIONS": "true",
        "TESTDB_PG_URL": "postgresql://journal:testpass@localhost:5433/postgres",
        "TESTDB_PG_DISPOSABLE_URL": "postgresql://journal:testpass@127.0.0.1:5434/postgres",
    }

    rc = _reset(docker, fake_psql, env_file, toplevel, ["--ci", *_PROFILE, *_DBS], environ)

    assert rc == 0
    assert docker.calls == []
    assert {(_arg(c, "-h"), _arg(c, "-p")) for c in fake_psql.psql_calls()} == {
        ("localhost", "5433"),
        ("127.0.0.1", "5434"),
    }
    assert sorted(c.db for c in fake_psql.bootstraps()) == ["journal_rls_test", "journal_test"]


# ---------------------------------------------------------------------------
# Output never carries a password
# ---------------------------------------------------------------------------


def test_reset_prints_step_names_and_durations_without_a_password(
    stack: FakeDocker,
    fake_psql: FakePsql,
    env_file: Path,
    toplevel: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [*_PROFILE, *_DBS, "--migrate-dsn-env", "X", "--migrate", ".", "alembic"]

    _reset(stack, fake_psql, env_file, toplevel, argv)

    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert lines[0].startswith("pg: drop databases ")
    assert "pg: migrate journal_rls_test [1] alembic " in out.out
    assert re.fullmatch(r"reset [0-9]+\.[0-9]{2}s", lines[-1])
    assert testdb.PG_PASSWORD not in out.out + out.err


def test_reset_redacts_passwords_from_a_failing_commands_output(
    stack: FakeDocker,
    fake_psql: FakePsql,
    env_file: Path,
    toplevel: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_psql.fail_on = "alembic"
    fake_psql.fail_stderr = (
        "connect postgresql://journal:testpass@127.0.0.1:1/x failed\n"
        "role secret-app-pw and postgresql://u:other@h/d\n"
    )
    argv = [*_PROFILE, *_DBS, "--migrate-dsn-env", "X", "--migrate", ".", "alembic"]

    rc = _reset(
        stack, fake_psql, env_file, toplevel, argv, {"JOURNAL_DB_APP_PASSWORD": "secret-app-pw"}
    )

    err = capsys.readouterr().err
    assert rc == 1
    assert "migrate journal_test (alembic) failed" in err
    for secret in ("testpass", "secret-app-pw", "other"):
        assert secret not in err
