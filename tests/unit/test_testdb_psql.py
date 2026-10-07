"""Tests for the docker psql wrapper in tools/testdb/bin and ``testdb.py psql-path``.

docker is never run: ``psql-path`` is exercised against fake ``psql`` scripts on
PATH, and the wrapper against a fake ``docker`` that records its argv (and, for
the exec path, answers ``ps`` and ``container inspect`` with one pg container of
the test's checkout), so the image pin, mounts, passed-through environment,
psql arguments and the exec-or-run choice are all pinned without a daemon.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.fixtures.testdb_tool import TESTDB_TOOL, load_testdb_tool

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

testdb = load_testdb_tool()

_WRAPPER_DIR = TESTDB_TOOL.parent / "bin"
_WRAPPER = _WRAPPER_DIR / "psql"
_PG_IMAGE = "pgvector/pgvector@sha256:" + "a" * 64
_ENV_BYTES = f"PGVECTOR_IMAGE={_PG_IMAGE}\nPG_MAJOR=17\n".encode()
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _fake_psql(directory: Path, version_line: str) -> Path:
    return _write_script(directory / "psql", f"echo '{version_line}'")


@dataclass(frozen=True)
class _Outcome:
    returncode: int
    stdout: str
    stderr: str


@pytest.fixture
def pgdg_root(tmp_path: Path) -> Path:
    """The PGDG bin root ``psql-path`` probes; empty unless a test installs a psql."""
    return tmp_path / "pgdg"


@pytest.fixture
def psql_path_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], pgdg_root: Path
) -> Callable[..., _Outcome]:
    """Run ``testdb.py psql-path`` with PATH set and the PGDG root under tmp_path.

    In-process, so a real ``/usr/lib/postgresql`` on the machine never decides a test.
    """
    monkeypatch.setattr(testdb, "PGDG_BIN_ROOT", pgdg_root)

    def run(path: str, *extra: str) -> _Outcome:
        monkeypatch.setenv("PATH", path)
        capsys.readouterr()
        code = testdb.main(["psql-path", *extra])
        out, err = capsys.readouterr()
        return _Outcome(code, out, err)

    return run


def _pgdg_psql(pgdg_root: Path, version_line: str) -> Path:
    return _fake_psql(pgdg_root / "17" / "bin", version_line).parent


def _wrapper_hint(pg_major: str) -> str:
    return (
        f"testdb: no host psql {pg_major}; using the docker wrapper (slow pre-push);"
        f" install postgresql-client-{pg_major}\n"
    )


# ---------------------------------------------------------------------------
# psql-path
# ---------------------------------------------------------------------------


def test_psql_path_prints_nothing_when_host_psql_matches_pg_major(
    tmp_path: Path, psql_path_cli: Callable[..., _Outcome]
) -> None:
    host = _fake_psql(tmp_path / "host", "psql (PostgreSQL) 17.4 (Debian 17.4-1)").parent

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "version_line",
    [
        pytest.param("psql (PostgreSQL) 16.8", id="older-major"),
        pytest.param("psql (PostgreSQL) 170.1", id="major-prefix-only"),
        pytest.param("not a psql", id="unparseable"),
    ],
)
def test_psql_path_prints_wrapper_dir_when_host_psql_is_another_major(
    tmp_path: Path, version_line: str, psql_path_cli: Callable[..., _Outcome]
) -> None:
    host = _fake_psql(tmp_path / "host", version_line).parent

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_prints_wrapper_dir_when_no_psql_on_path(
    tmp_path: Path, psql_path_cli: Callable[..., _Outcome]
) -> None:
    (tmp_path / "empty").mkdir()

    result = psql_path_cli(str(tmp_path / "empty"))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_ignores_the_wrapper_itself_on_path(
    tmp_path: Path, psql_path_cli: Callable[..., _Outcome]
) -> None:
    (tmp_path / "empty").mkdir()

    result = psql_path_cli(os.pathsep.join([str(_WRAPPER_DIR), str(tmp_path / "empty")]))

    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_uses_the_first_psql_on_path(
    tmp_path: Path, psql_path_cli: Callable[..., _Outcome]
) -> None:
    first = _fake_psql(tmp_path / "first", "psql (PostgreSQL) 16.8").parent
    second = _fake_psql(tmp_path / "second", "psql (PostgreSQL) 17.4").parent

    result = psql_path_cli(os.pathsep.join([str(first), str(second)]))

    assert result.stdout == f"{_WRAPPER_DIR}\n"


@pytest.mark.parametrize(
    ("raw", "named"),
    [
        pytest.param(f"PGVECTOR_IMAGE={_PG_IMAGE}\n".encode(), "PG_MAJOR=", id="missing"),
        pytest.param(
            f"PGVECTOR_IMAGE={_PG_IMAGE}\nPG_MAJOR=1x\n".encode(), "PG_MAJOR=1x", id="bad"
        ),
        pytest.param(b"PG_MAJOR=17", "missing final newline", id="invalid-file"),
    ],
)
def test_psql_path_exits_2_on_a_bad_pg_major(
    tmp_path: Path, raw: bytes, named: str, psql_path_cli: Callable[..., _Outcome]
) -> None:
    env_file = tmp_path / "testdb.env"
    env_file.write_bytes(raw)
    host = _fake_psql(tmp_path / "host", "psql (PostgreSQL) 17.4").parent

    result = psql_path_cli(str(host), "--file", str(env_file))

    assert result.returncode == 2
    assert named in result.stderr
    assert "testdb.env" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "on_path",
    [
        pytest.param(None, id="nothing-on-path"),
        pytest.param("psql (PostgreSQL) 17.4", id="matching-psql-on-path"),
    ],
)
def test_psql_path_prefers_the_pgdg_bin_dir_of_pg_major(
    tmp_path: Path, pgdg_root: Path, on_path: str | None, psql_path_cli: Callable[..., _Outcome]
) -> None:
    pgdg = _pgdg_psql(pgdg_root, "psql (PostgreSQL) 17.6 (Ubuntu 17.6-1.pgdg24.04+1)")
    host = tmp_path / "host"
    host.mkdir()
    if on_path is not None:
        _fake_psql(host, on_path)

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{pgdg}\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("pgdg_version", "on_path", "expected"),
    [
        pytest.param(
            "psql (PostgreSQL) 16.8", "psql (PostgreSQL) 17.4", "", id="other-major-host-fits"
        ),
        pytest.param("psql (PostgreSQL) 16.8", None, "wrapper", id="other-major-no-host"),
        pytest.param(None, "psql (PostgreSQL) 17.4", "", id="missing-host-fits"),
        pytest.param(None, None, "wrapper", id="missing-no-host"),
    ],
)
def test_psql_path_falls_through_when_the_pgdg_psql_does_not_fit(
    tmp_path: Path,
    pgdg_root: Path,
    pgdg_version: str | None,
    on_path: str | None,
    expected: str,
    psql_path_cli: Callable[..., _Outcome],
) -> None:
    if pgdg_version is not None:
        _pgdg_psql(pgdg_root, pgdg_version)
    host = tmp_path / "host"
    host.mkdir()
    if on_path is not None:
        _fake_psql(host, on_path)

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == (f"{_WRAPPER_DIR}\n" if expected == "wrapper" else "")


@pytest.mark.parametrize(
    ("on_path", "expected_stdout", "expected_stderr"),
    [
        pytest.param("psql (PostgreSQL) 17.4", "", "", id="host-fits"),
        pytest.param(None, f"{_WRAPPER_DIR}\n", _wrapper_hint("17"), id="no-host"),
    ],
)
def test_psql_path_falls_through_when_the_pgdg_psql_cannot_be_launched(
    tmp_path: Path,
    pgdg_root: Path,
    on_path: str | None,
    expected_stdout: str,
    expected_stderr: str,
    psql_path_cli: Callable[..., _Outcome],
) -> None:
    (_pgdg_psql(pgdg_root, "psql (PostgreSQL) 17.4") / "psql").chmod(0o644)
    host = tmp_path / "host"
    host.mkdir()
    if on_path is not None:
        _fake_psql(host, on_path)

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr


def test_psql_path_hints_at_the_client_package_when_it_picks_the_wrapper(
    tmp_path: Path, psql_path_cli: Callable[..., _Outcome]
) -> None:
    env_file = tmp_path / "testdb.env"
    env_file.write_bytes(f"PGVECTOR_IMAGE={_PG_IMAGE}\nPG_MAJOR=18\n".encode())
    host = _fake_psql(tmp_path / "host", "psql (PostgreSQL) 17.4").parent

    result = psql_path_cli(str(host), "--file", str(env_file))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{_WRAPPER_DIR}\n"
    assert result.stderr == _wrapper_hint("18")


@pytest.mark.parametrize(
    ("install", "expected_stdout"),
    [
        pytest.param("pgdg", "pgdg", id="pgdg-psql"),
        pytest.param("path", "", id="psql-on-path"),
    ],
)
def test_psql_path_prints_no_hint_when_a_host_psql_is_chosen(
    tmp_path: Path,
    pgdg_root: Path,
    install: str,
    expected_stdout: str,
    psql_path_cli: Callable[..., _Outcome],
) -> None:
    host = tmp_path / "host"
    host.mkdir()
    pgdg = pgdg_root / "17" / "bin"
    if install == "pgdg":
        _pgdg_psql(pgdg_root, "psql (PostgreSQL) 17.4")
    else:
        _fake_psql(host, "psql (PostgreSQL) 17.4")

    result = psql_path_cli(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == (f"{pgdg}\n" if expected_stdout == "pgdg" else "")
    assert result.stderr == ""


# ---------------------------------------------------------------------------
# bin/psql wrapper
# ---------------------------------------------------------------------------


@pytest.fixture
def wrapper_copy(tmp_path: Path) -> Path:
    """Return a copy of the wrapper whose sibling testdb.env the test controls."""
    tool_dir = tmp_path / "tools" / "testdb"
    (tool_dir / "bin").mkdir(parents=True)
    shutil.copy2(TESTDB_TOOL, tool_dir / "testdb.py")
    shutil.copy2(_WRAPPER, tool_dir / "bin" / "psql")
    (tool_dir / "testdb.env").write_bytes(_ENV_BYTES)
    (tmp_path / "deployment" / "scripts").mkdir(parents=True)
    (tmp_path / ".env").write_text("SECRET=x\n", encoding="ascii")
    return tool_dir / "bin" / "psql"


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    """Return a PATH dir whose ``docker`` writes one argv entry per line to docker.argv."""
    log = tmp_path / "docker.argv"
    _write_script(tmp_path / "fakebin" / "docker", f"printf '%s\\n' \"$@\" > '{log}'")
    return tmp_path / "fakebin"


def _run_wrapper(
    wrapper: Path, fake_docker: Path, cwd: Path, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    path = os.pathsep.join([str(fake_docker), os.environ.get("PATH", "")])
    return subprocess.run(  # noqa: S603 -- the wrapper under test with literal args
        [str(wrapper), *args],
        cwd=cwd,
        env={"PATH": path, "TMPDIR": str(cwd), **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def _docker_argv(fake_docker: Path) -> list[str]:
    return (fake_docker.parent / "docker.argv").read_text(encoding="utf-8").splitlines()


def test_wrapper_runs_the_pinned_image_with_psql_args_last(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    result = _run_wrapper(
        wrapper_copy, fake_docker, tmp_path, "-f", "x.sql", "-c", "select 1; -- a b"
    )

    assert result.returncode == 0, result.stderr
    argv = _docker_argv(fake_docker)
    assert argv[:5] == ["run", "--rm", "--interactive", "--quiet", "--network"]
    assert argv[argv.index("--") :] == [
        "--",
        _PG_IMAGE,
        "psql",
        "-f",
        "x.sql",
        "-c",
        "select 1; -- a b",
    ]
    assert argv[argv.index("--workdir") + 1] == str(tmp_path.resolve())


_MOUNT_FLAGS = ("--volume", "-v", "--mount")


def _mount_args(argv: list[str]) -> list[str]:
    head = argv[: argv.index("--")]
    return [head[i + 1] for i, arg in enumerate(head) if arg in _MOUNT_FLAGS]


def test_wrapper_mounts_only_the_sql_dirs_read_only(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version")

    root = tmp_path.resolve()
    sql_dirs = [root / "deployment" / "scripts", root / "tools" / "testdb"]
    assert _mount_args(_docker_argv(fake_docker)) == [f"{d}:{d}:ro" for d in sql_dirs]


def test_wrapper_does_not_mount_the_checkout_root_or_temp_dir(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # noqa: S603, S607
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()

    _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version", TMPDIR=str(temp_dir))

    sources = [Path(mount.split(":")[0]) for mount in _mount_args(_docker_argv(fake_docker))]
    assert sources
    for exposed in (tmp_path.resolve(), temp_dir.resolve()):
        assert not any(exposed.is_relative_to(source) for source in sources), exposed


def test_wrapper_skips_a_missing_sql_dir(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    (tmp_path / "deployment" / "scripts").rmdir()

    _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version")

    tool_dir = (tmp_path / "tools" / "testdb").resolve()
    assert _mount_args(_docker_argv(fake_docker)) == [f"{tool_dir}:{tool_dir}:ro"]


def test_wrapper_keeps_a_subdirectory_working_dir(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    _run_wrapper(wrapper_copy, fake_docker, tmp_path / "deployment", "--version")

    argv = _docker_argv(fake_docker)
    assert argv[argv.index("--workdir") + 1] == str((tmp_path / "deployment").resolve())


def test_wrapper_passes_pg_and_journal_db_env_by_name_only(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    _run_wrapper(
        wrapper_copy,
        fake_docker,
        tmp_path,
        "--version",
        PGPASSWORD="secret",
        JOURNAL_DB_APP_PASSWORD="app",
        UNRELATED="x",
    )

    argv = _docker_argv(fake_docker)
    passed = {argv[i + 1] for i, arg in enumerate(argv) if arg == "--env"}
    assert {"PGPASSWORD", "JOURNAL_DB_APP_PASSWORD"} <= passed
    assert "UNRELATED" not in passed
    assert not any("secret" in arg for arg in argv)


@pytest.mark.parametrize(
    "bad_line",
    [
        pytest.param(b"PGVECTOR_IMAGE=pgvector/pgvector:pg17 trailing\n", id="space"),
        pytest.param(b"PGVECTOR_IMAGE=\n", id="empty"),
        pytest.param(b"PGVECTOR_IMAGE=pgvector/pgvector:pg17\r\n", id="crlf"),
    ],
)
def test_wrapper_exits_nonzero_naming_testdb_env_on_an_invalid_pin(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path, bad_line: bytes
) -> None:
    (wrapper_copy.parent.parent / "testdb.env").write_bytes(b"PG_MAJOR=17\n" + bad_line)

    result = _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version")

    assert result.returncode != 0
    assert "testdb.env" in result.stderr
    assert not (fake_docker.parent / "docker.argv").exists()


def test_wrapper_exits_nonzero_when_pgvector_image_is_absent(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    (wrapper_copy.parent.parent / "testdb.env").write_bytes(b"PG_MAJOR=17\n")

    result = _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version")

    assert result.returncode != 0
    assert "testdb.env does not declare PGVECTOR_IMAGE" in result.stderr
    assert not (fake_docker.parent / "docker.argv").exists()


def test_wrapper_is_executable_in_git() -> None:
    result = subprocess.run(  # noqa: S603 -- literal git argv
        ["git", "ls-files", "--stage", "--", str(_WRAPPER)],  # noqa: S607 -- git from PATH
        cwd=_WRAPPER_DIR,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.startswith("100755 ")


# ---------------------------------------------------------------------------
# bin/psql wrapper: docker exec into this checkout's stack
# ---------------------------------------------------------------------------

_FAKE_DOCKER_PY = """
import json, sys
state = json.load(open(sys.argv[1]))
args = sys.argv[2:]
if args[0] == "ps":
    sys.stdin.read()
    print(state["ps"], end="")
elif args[:2] == ["container", "inspect"]:
    print(json.dumps(state["inspect"]))
else:
    with open(state["log"], "w") as log:
        log.write("\\n".join(args) + "\\n")
    with open(state["stdin_log"], "w") as log:
        log.write(sys.stdin.read())
"""
_STACK_PORT = "40001"


@pytest.fixture
def stack_docker(tmp_path: Path, wrapper_copy: Path) -> Path:
    """Return a PATH dir whose ``docker`` reports one pg container of this checkout.

    ``ps`` and ``container inspect`` answer from the recorded inspect sample,
    relabelled for the wrapper copy's checkout; ``ps`` also drains its stdin, so
    any stdin the planner inherits is lost to psql. ``exec`` and ``run`` record
    their argv in docker.argv and their stdin in docker.stdin.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # noqa: S603, S607
    toplevel = str(tmp_path.resolve())
    digest = hashlib.sha256(toplevel.encode()).hexdigest()
    name = f"testdb-gubbi-{digest[:8]}-pg"
    entry = json.loads((_FIXTURES / "testdb_docker_inspect.json").read_text())["container_inspect"][
        0
    ]
    entry["Name"] = f"/{name}"
    entry["Config"]["Labels"] = {
        "ai.gubbi.testdb.repo": "gubbi",
        "ai.gubbi.testdb.role": "pg",
        "ai.gubbi.testdb.owner-uid": str(os.getuid()),
        "ai.gubbi.testdb.checkout": digest,
        "ai.gubbi.testdb.env-sha256": hashlib.sha256(_ENV_BYTES).hexdigest(),
        "ai.gubbi.testdb.auth": "scram-sha-256",
    }
    state = {
        "ps": f"{name}\n",
        "inspect": [entry],
        "log": str(tmp_path / "docker.argv"),
        "stdin_log": str(tmp_path / "docker.stdin"),
    }
    (tmp_path / "docker.json").write_text(json.dumps(state))
    script = tmp_path / "fake_docker.py"
    script.write_text(_FAKE_DOCKER_PY)
    _write_script(
        tmp_path / "stackbin" / "docker",
        f"exec '{sys.executable}' '{script}' '{tmp_path / 'docker.json'}' \"$@\"",
    )
    return tmp_path / "stackbin"


def _stack_id(tmp_path: Path) -> str:
    state = json.loads((tmp_path / "docker.json").read_text())
    container_id: str = state["inspect"][0]["Id"]
    return container_id


def _no_stack(tmp_path: Path) -> None:
    state = json.loads((tmp_path / "docker.json").read_text())
    (tmp_path / "docker.json").write_text(json.dumps({**state, "ps": ""}))


def test_wrapper_execs_into_the_owned_stack_container(
    wrapper_copy: Path, stack_docker: Path, tmp_path: Path
) -> None:
    url = f"postgresql://journal:pw@127.0.0.1:{_STACK_PORT}/journal_test"

    result = _run_wrapper(
        wrapper_copy, stack_docker, tmp_path, "-tAc", "select 1", url, PGPASSWORD="pw"
    )

    assert result.returncode == 0, result.stderr
    argv = _docker_argv(stack_docker)
    assert argv[:4] == ["exec", "--interactive", "--user", "65534:65534"]
    assert _mount_args(argv) == []
    assert argv[argv.index("--") :] == [
        "--",
        _stack_id(tmp_path),
        "psql",
        "-t",
        "-A",
        "-c",
        "select 1",
        "-d",
        "postgresql://journal:pw@192.0.2.10:5432/journal_test",
    ]
    assert argv[argv.index("--env") + 1] == "PGPASSWORD"


def test_wrapper_runs_a_throwaway_container_when_no_stack_container_is_found(
    wrapper_copy: Path, stack_docker: Path, tmp_path: Path
) -> None:
    _no_stack(tmp_path)
    url = f"postgresql://journal:pw@127.0.0.1:{_STACK_PORT}/journal_test"

    result = _run_wrapper(wrapper_copy, stack_docker, tmp_path, "-tAc", "select 1", url)

    assert result.returncode == 0, result.stderr
    argv = _docker_argv(stack_docker)
    assert argv[0] == "run"
    assert argv[argv.index("--") :] == ["--", _PG_IMAGE, "psql", "-tAc", "select 1", url]


def test_wrapper_feeds_an_sql_file_on_stdin_under_exec(
    wrapper_copy: Path, stack_docker: Path, tmp_path: Path
) -> None:
    sql = tmp_path / "deployment" / "scripts" / "grants.sql"
    sql.write_text("SELECT 'from the host file';\n", encoding="ascii")
    url = f"postgresql://journal:pw@127.0.0.1:{_STACK_PORT}/journal_test"

    result = _run_wrapper(
        wrapper_copy, stack_docker, tmp_path, "-v", "ON_ERROR_STOP=1", "-f", str(sql), url
    )

    assert result.returncode == 0, result.stderr
    argv = _docker_argv(stack_docker)
    assert argv[0] == "exec"
    assert argv[argv.index("psql") + 1 : argv.index("psql") + 5] == [
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        "-",
    ]
    assert (tmp_path / "docker.stdin").read_text() == "SELECT 'from the host file';\n"


def test_wrapper_refuses_a_file_outside_the_sql_dirs_under_exec(
    wrapper_copy: Path, stack_docker: Path, tmp_path: Path
) -> None:
    url = f"postgresql://journal:pw@127.0.0.1:{_STACK_PORT}/journal_test"

    result = _run_wrapper(wrapper_copy, stack_docker, tmp_path, "-f", str(tmp_path / ".env"), url)

    assert result.returncode == 2
    assert "only regular files under" in result.stderr
    assert not (tmp_path / "docker.argv").exists()


def test_wrapper_passes_its_stdin_to_psql_under_exec(
    wrapper_copy: Path, stack_docker: Path, tmp_path: Path
) -> None:
    url = f"postgresql://journal:pw@127.0.0.1:{_STACK_PORT}/journal_test"
    path = os.pathsep.join([str(stack_docker), os.environ.get("PATH", "")])

    result = subprocess.run(  # noqa: S603 -- the wrapper under test with literal args
        [str(wrapper_copy), "-tA", url],
        cwd=tmp_path,
        env={"PATH": path},
        input="select 42;\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _docker_argv(stack_docker)[0] == "exec"
    assert (tmp_path / "docker.stdin").read_text() == "select 42;\n"
