"""Unit tests for ``testdb.py psql-plan``: when bin/psql may exec into the stack.

Neither docker nor git is run: the controller tests' fake docker answers
``docker container inspect``, and this module answers ``docker ps`` from the
same container table (honouring its label and publish filters) and ``git
rev-parse`` with a fixed checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.testdb_tool import load_testdb_tool
from tests.unit.test_testdb_controller import (
    _ENV_BYTES,
    _PG_IMAGE,
    _TOPLEVEL,
    _UID,
    FakeDocker,
    _add_stack,
    _labels,
    _name,
    _port,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit

testdb = load_testdb_tool()

_SAMPLE_ADDRESS = "192.0.2.10"
_SQL_DIR = testdb.PSQL_SQL_DIRS[1]
_BOOTSTRAP = _SQL_DIR / "bootstrap.sql"


def _matches(entry: dict[str, Any], flt: str) -> bool:
    key, _, value = flt.partition("=")
    if key == "label":
        name, _, wanted = value.partition("=")
        return bool(entry["Config"]["Labels"].get(name) == wanted)
    assert key == "publish", flt
    bindings = [b for spec in entry["NetworkSettings"]["Ports"].values() for b in spec or ()]
    return any(binding["HostPort"] == value for binding in bindings)


class Runner:
    """``run`` for the planner: git and ``docker ps`` here, the rest to FakeDocker."""

    def __init__(self, docker: FakeDocker, toplevel: Path | None = _TOPLEVEL) -> None:
        self.docker = docker
        self.toplevel = toplevel

    def __call__(self, argv: Sequence[str], timeout: float | None) -> Any:
        argv = list(argv)
        if argv[0] == "git":
            if self.toplevel is None:
                return testdb.CommandResult(128, "", "fatal: not a git repository")
            return testdb.CommandResult(0, f"{self.toplevel}\n", "")
        if argv[:2] == ["docker", "ps"]:
            filters = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--filter"]
            names = [
                name
                for name, entry in self.docker.containers.items()
                if entry["State"]["Running"] and all(_matches(entry, f) for f in filters)
            ]
            return testdb.CommandResult(0, "".join(f"{n}\n" for n in names), "")
        return self.docker(argv, timeout)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "testdb.env"
    path.write_bytes(_ENV_BYTES)
    return path


@pytest.fixture
def stack() -> FakeDocker:
    docker = FakeDocker()
    _add_stack(docker)
    return docker


def _plan(
    argv: list[str],
    docker: FakeDocker,
    env_file: Path,
    *,
    toplevel: Path | None = _TOPLEVEL,
    environ: dict[str, str] | None = None,
    cwd: Path = Path("/"),
) -> Any:
    return testdb.plan_psql(
        argv,
        run=Runner(docker, toplevel),
        uid=_UID,
        environ=environ or {},
        cwd=cwd,
        env_file=env_file,
    )


def _url(port: str, db: str = "journal_test") -> str:
    return f"postgresql://journal:testpass@127.0.0.1:{port}/{db}"


# ---------------------------------------------------------------------------
# exec: the target is this checkout's own pg container
# ---------------------------------------------------------------------------


def test_url_target_execs_into_the_container_publishing_its_port(
    stack: FakeDocker, env_file: Path
) -> None:
    port = _port(stack, _name("pg"))

    plan = _plan(["-v", "ON_ERROR_STOP=1", "-tAc", "select 1", _url(port)], stack, env_file)

    assert plan.mode == "exec"
    assert plan.target == stack.id_of(_name("pg"))
    assert plan.stdin_file is None
    assert plan.argv == (
        "-v",
        "ON_ERROR_STOP=1",
        "-t",
        "-A",
        "-c",
        "select 1",
        "-d",
        f"postgresql://journal:testpass@{_SAMPLE_ADDRESS}:5432/journal_test",
    )


def test_host_and_port_flags_pick_the_matching_one_of_two_clusters(
    stack: FakeDocker, env_file: Path
) -> None:
    port = _port(stack, _name("pg-disposable"))
    argv = ["-X", "-h", "127.0.0.1", "-p", port, "-U", "journal", "-d", "dbname='postgres'"]

    plan = _plan([*argv, "-c", "select 1"], stack, env_file)

    assert plan.mode == "exec"
    assert plan.target == stack.id_of(_name("pg-disposable"))
    assert plan.argv == (
        "-X",
        "-c",
        "select 1",
        "-h",
        _SAMPLE_ADDRESS,
        "-p",
        "5432",
        "-U",
        "journal",
        "-d",
        "dbname='postgres'",
    )


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(["--command=select 1"], id="long-attached"),
        pytest.param(["--command", "select 1"], id="long-separate"),
        pytest.param(["-cselect 1"], id="short-attached"),
        pytest.param(["-tAcselect 1"], id="bundle-attached"),
    ],
)
def test_option_spellings_normalise_to_the_short_form(
    stack: FakeDocker, env_file: Path, spelling: list[str]
) -> None:
    plan = _plan([*spelling, _url(_port(stack, _name("pg")))], stack, env_file)

    assert plan.mode == "exec"
    assert plan.argv[plan.argv.index("-c") + 1] == "select 1"


def test_positional_user_is_carried_as_dash_u(stack: FakeDocker, env_file: Path) -> None:
    port = _port(stack, _name("pg"))

    plan = _plan(
        ["-h", "localhost", "-p", port, "-c", "x", "journal_test", "journal"], stack, env_file
    )

    assert plan.mode == "exec"
    assert plan.argv[-4:] == ("-U", "journal", "-d", "journal_test")


@pytest.mark.parametrize(
    "file_arg",
    [
        pytest.param(str(_BOOTSTRAP), id="absolute"),
        pytest.param("bootstrap.sql", id="relative-to-cwd"),
    ],
)
def test_a_file_under_the_sql_dirs_is_fed_on_stdin(
    stack: FakeDocker, env_file: Path, file_arg: str
) -> None:
    port = _port(stack, _name("pg"))

    plan = _plan(
        ["-v", "ON_ERROR_STOP=1", "-f", file_arg, _url(port)], stack, env_file, cwd=_SQL_DIR
    )

    assert plan.mode == "exec"
    assert plan.stdin_file == str(_BOOTSTRAP.resolve())
    assert plan.argv[:4] == ("-v", "ON_ERROR_STOP=1", "-f", "-")


@pytest.mark.parametrize(
    "file_arg",
    [
        pytest.param("/etc/passwd", id="outside-the-repo"),
        pytest.param(str(_SQL_DIR / ".." / ".." / "pyproject.toml"), id="dot-dot-escape"),
        pytest.param(str(_SQL_DIR), id="a-directory"),
        pytest.param(str(_SQL_DIR / "no-such.sql"), id="missing"),
    ],
)
def test_a_file_outside_the_sql_dirs_is_refused(
    stack: FakeDocker, env_file: Path, file_arg: str
) -> None:
    port = _port(stack, _name("pg"))

    with pytest.raises(testdb.ConfigError, match="-f "):
        _plan(["-f", file_arg, _url(port)], stack, env_file)


def test_a_symlink_into_the_sql_dirs_from_outside_resolves_before_the_check(
    stack: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    link = tmp_path / "grants.sql"
    link.symlink_to(_BOOTSTRAP)
    escape = tmp_path / "escape.sql"
    escape.symlink_to(Path("/etc/passwd"))
    port = _port(stack, _name("pg"))

    plan = _plan(["-f", str(link), _url(port)], stack, env_file)

    assert plan.stdin_file == str(_BOOTSTRAP.resolve())
    with pytest.raises(testdb.ConfigError):
        _plan(["-f", str(escape), _url(port)], stack, env_file)


# ---------------------------------------------------------------------------
# run: everything the exec path cannot or must not carry
# ---------------------------------------------------------------------------


def _assert_run(plan: Any) -> None:
    assert plan.mode == "run"
    assert plan.target == _PG_IMAGE
    assert plan.fields() == ["run", _PG_IMAGE, "end"]


def test_no_stack_container_on_the_port_runs(stack: FakeDocker, env_file: Path) -> None:
    _assert_run(_plan(["-c", "x", _url("1")], stack, env_file))


def test_no_stack_at_all_runs(env_file: Path) -> None:
    _assert_run(_plan(["-c", "x", _url("40001")], FakeDocker(), env_file))


def test_outside_a_git_checkout_runs(stack: FakeDocker, env_file: Path) -> None:
    port = _port(stack, _name("pg"))

    _assert_run(_plan(["-c", "x", _url(port)], stack, env_file, toplevel=None))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"ai.gubbi.testdb.repo": "Bad Repo"}, id="invalid-repo"),
        pytest.param({"ai.gubbi.testdb.role": "redis"}, id="not-a-pg-role"),
        pytest.param({"ai.gubbi.testdb.env-sha256": "0" * 64}, id="stale-env-sha256"),
        pytest.param({"ai.gubbi.testdb.auth": "trust"}, id="stale-auth-label"),
        pytest.param({"ai.gubbi.testdb.extra": "x"}, id="extra-label"),
    ],
)
def test_a_container_failing_the_label_check_runs(
    env_file: Path, overrides: dict[str, str]
) -> None:
    docker = FakeDocker()
    name = _name("pg")
    docker.add(name, _labels("pg", **overrides))

    _assert_run(_plan(["-c", "x", _url(_port(docker, name))], docker, env_file))


def test_another_checkouts_container_on_the_port_runs(env_file: Path) -> None:
    docker = FakeDocker()
    other = Path("/work/checkout-b")
    name = _name("pg", other)
    docker.add(name, _labels("pg", other))

    _assert_run(_plan(["-c", "x", _url(_port(docker, name))], docker, env_file))


def test_a_container_renamed_away_from_its_labels_runs(env_file: Path) -> None:
    docker = FakeDocker()
    docker.add("testdb-gubbi-ffffffff-pg", _labels("pg"))

    _assert_run(
        _plan(["-c", "x", _url(_port(docker, "testdb-gubbi-ffffffff-pg"))], docker, env_file)
    )


def test_another_users_container_runs(env_file: Path) -> None:
    docker = FakeDocker()
    name = _name("pg")
    docker.add(name, _labels("pg", **{"ai.gubbi.testdb.owner-uid": str(_UID + 1)}))

    _assert_run(_plan(["-c", "x", _url(_port(docker, name))], docker, env_file))


@pytest.mark.parametrize(
    "networks",
    [
        pytest.param({}, id="no-network"),
        pytest.param(
            {"a": {"IPAddress": "192.0.2.10"}, "b": {"IPAddress": "192.0.2.11"}}, id="two-networks"
        ),
        pytest.param({"a": {"IPAddress": "not-an-ip"}}, id="unparseable"),
    ],
)
def test_a_container_without_exactly_one_ipv4_address_runs(
    stack: FakeDocker, env_file: Path, networks: dict[str, Any]
) -> None:
    name = _name("pg")
    stack.containers[name]["NetworkSettings"]["Networks"] = networks

    _assert_run(_plan(["-c", "x", _url(_port(stack, name))], stack, env_file))


def _pg_port() -> str:
    docker = FakeDocker()
    _add_stack(docker)
    return _port(docker, _name("pg"))


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["-c", "x", "postgresql://journal@db.example:{port}/x"], id="remote-host"),
        pytest.param(["-c", "x", "postgresql://journal@127.0.0.1/x"], id="url-without-port"),
        pytest.param(["-c", "x", "postgresql://j@127.0.0.1:{port}/x?sslmode=disable"], id="query"),
        pytest.param(["-c", "x", "postgresql://j@127.0.0.1:{port},h:1/x"], id="multi-host"),
        pytest.param(
            ["-c", "x", "-h", "127.0.0.1", "postgresql://j@127.0.0.1:{port}/x"], id="url+h"
        ),
        pytest.param(
            ["-c", "x", "-h", "127.0.0.1", "-p", "{port}", "-d", "host=h dbname=x"], id="ci-host"
        ),
        pytest.param(["-c", "x", "-h", "127.0.0.1", "-p", "{port}", "-d", "port=1"], id="ci-port"),
        pytest.param(["-c", "x", "-h", "127.0.0.1"], id="no-port"),
        pytest.param(["-c", "x", "-p", "{port}"], id="no-host"),
        pytest.param(["-c", "x", "-h", "/var/run/postgresql", "-p", "{port}"], id="socket-dir"),
        pytest.param(["-c", "x", "-h", "127.0.0.1", "-h", "127.0.0.1", "-p", "{port}"], id="two-h"),
        pytest.param(["-o", "out.txt", "-c", "x", _url("{port}")], id="output-file"),
        pytest.param(["-L", "log.txt", "-c", "x", _url("{port}")], id="log-file"),
        pytest.param(["--unknown", "-c", "x", _url("{port}")], id="unknown-option"),
        pytest.param(["-l", _url("{port}")], id="unknown-short"),
        pytest.param(["-c", "x", "--", _url("{port}")], id="double-dash"),
        pytest.param(["-f", "a.sql", "-f", "b.sql", _url("{port}")], id="two-files"),
        pytest.param(["-c", "x", _url("{port}"), "u", "extra"], id="three-positionals"),
        pytest.param(["-c"], id="missing-value"),
        pytest.param(["--version"], id="version"),
    ],
)
def test_command_lines_outside_the_exec_shape_run(
    stack: FakeDocker, env_file: Path, argv: list[str]
) -> None:
    port = _port(stack, _name("pg"))

    _assert_run(_plan([a.replace("{port}", port) for a in argv], stack, env_file))


@pytest.mark.parametrize("name", ["PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"])
def test_a_redirecting_libpq_variable_runs(stack: FakeDocker, env_file: Path, name: str) -> None:
    port = _port(stack, _name("pg"))

    _assert_run(_plan(["-c", "x", _url(port)], stack, env_file, environ={name: "x"}))


@pytest.mark.parametrize("dbname", ["postgres", "postgresql"])
def test_a_bare_database_named_like_a_url_scheme_is_not_a_url(
    stack: FakeDocker, env_file: Path, dbname: str
) -> None:
    port = _port(stack, _name("pg"))

    plan = _plan(["-h", "127.0.0.1", "-p", port, "-c", "x", "-d", dbname], stack, env_file)

    assert plan.mode == "exec"
    assert plan.argv[-6:] == ("-h", _SAMPLE_ADDRESS, "-p", "5432", "-d", dbname)


def test_the_url_rewrite_keeps_userinfo_and_database() -> None:
    rewritten = testdb.exec_argv(
        testdb.parse_psql_command(["postgresql://us%40er:p%3Aw@localhost:40001/my_db"]),
        "192.0.2.10",
        None,
    )

    assert rewritten == ["-d", "postgresql://us%40er:p%3Aw@192.0.2.10:5432/my_db"]


def test_psql_plan_main_prints_nul_terminated_fields(
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    status = testdb.main(["psql-plan", "--version"])

    out = capfdbinary.readouterr().out
    assert status == 0
    assert out.endswith(b"end\0")
    assert out.split(b"\0")[0] == b"run"


def test_psql_plan_main_exits_2_on_a_refused_file(
    capfdbinary: pytest.CaptureFixture[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise testdb.ConfigError("-f x: refused")

    monkeypatch.setattr(testdb, "plan_psql", refuse)

    status = testdb.main(["psql-plan", "-f", "x"])

    captured = capfdbinary.readouterr()
    assert status == 2
    assert captured.out == b""
    assert b"-f x: refused" in captured.err
