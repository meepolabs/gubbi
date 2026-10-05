"""Tests for the docker psql wrapper in tools/testdb/bin and ``testdb.py psql-path``.

docker is never run: ``psql-path`` is exercised against fake ``psql`` scripts on
PATH, and the wrapper against a fake ``docker`` that records its argv, so the
image pin, mounts, passed-through environment and psql arguments are all pinned
without a daemon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.testdb_tool import TESTDB_TOOL, load_testdb_tool

pytestmark = pytest.mark.unit

testdb = load_testdb_tool()

_WRAPPER_DIR = TESTDB_TOOL.parent / "bin"
_WRAPPER = _WRAPPER_DIR / "psql"
_PG_IMAGE = "pgvector/pgvector@sha256:" + "a" * 64
_ENV_BYTES = f"PGVECTOR_IMAGE={_PG_IMAGE}\nPG_MAJOR=17\n".encode()


def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _fake_psql(directory: Path, version_line: str) -> Path:
    return _write_script(directory / "psql", f"echo '{version_line}'")


def _psql_path(path: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- this interpreter running the controller
        [sys.executable, str(TESTDB_TOOL), "psql-path", *extra],
        env={"PATH": path},
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# psql-path
# ---------------------------------------------------------------------------


def test_psql_path_prints_nothing_when_host_psql_matches_pg_major(tmp_path: Path) -> None:
    host = _fake_psql(tmp_path / "host", "psql (PostgreSQL) 17.4 (Debian 17.4-1)").parent

    result = _psql_path(str(host))

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
    tmp_path: Path, version_line: str
) -> None:
    host = _fake_psql(tmp_path / "host", version_line).parent

    result = _psql_path(str(host))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_prints_wrapper_dir_when_no_psql_on_path(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    result = _psql_path(str(tmp_path / "empty"))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_ignores_the_wrapper_itself_on_path(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    result = _psql_path(os.pathsep.join([str(_WRAPPER_DIR), str(tmp_path / "empty")]))

    assert result.stdout == f"{_WRAPPER_DIR}\n"


def test_psql_path_uses_the_first_psql_on_path(tmp_path: Path) -> None:
    first = _fake_psql(tmp_path / "first", "psql (PostgreSQL) 16.8").parent
    second = _fake_psql(tmp_path / "second", "psql (PostgreSQL) 17.4").parent

    result = _psql_path(os.pathsep.join([str(first), str(second)]))

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
def test_psql_path_exits_2_on_a_bad_pg_major(tmp_path: Path, raw: bytes, named: str) -> None:
    env_file = tmp_path / "testdb.env"
    env_file.write_bytes(raw)
    host = _fake_psql(tmp_path / "host", "psql (PostgreSQL) 17.4").parent

    result = _psql_path(str(host), "--file", str(env_file))

    assert result.returncode == 2
    assert named in result.stderr
    assert "testdb.env" in result.stderr
    assert result.stdout == ""


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


def test_wrapper_mounts_the_temp_dir_read_only(
    wrapper_copy: Path, fake_docker: Path, tmp_path: Path
) -> None:
    _run_wrapper(wrapper_copy, fake_docker, tmp_path, "--version")

    argv = _docker_argv(fake_docker)
    volumes = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--volume"]
    real = tmp_path.resolve()
    assert f"{real}:{real}:ro" in volumes
    assert all(volume.endswith(":ro") for volume in volumes)


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
