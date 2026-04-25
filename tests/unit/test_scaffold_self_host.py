"""Unit tests for deployment.scaffold_self_host."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_MODULE = "deployment.scaffold_self_host"

_FAKE_UUID = "550e8400-e29b-41d4-a716-446655440000"
_EMAIL = "op@test.local"
_DB_URL = "postgresql://fake@localhost/test"


def _make_conn(insert_tag: str = "INSERT 0 1", user_id: str | None = _FAKE_UUID) -> MagicMock:
    """Return a mock asyncpg connection with pre-configured execute/fetchval."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=insert_tag)
    conn.fetchval = AsyncMock(return_value=user_id)
    conn.close = AsyncMock(return_value=None)
    return conn


def _env(**overrides: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "JOURNAL_OPERATOR_EMAIL": _EMAIL,
        "JOURNAL_DB_ADMIN_URL": _DB_URL,
    }
    base.update(overrides)
    return base


async def _run_main(
    conn: MagicMock,
    env: dict[str, str | None],
    extra_argv: list[str] | None = None,
) -> None:
    """Import and run main() with patched asyncpg.connect and env."""
    argv = ["provision_operator"] + (extra_argv or [])
    saved = list(sys.argv)
    backup = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.argv[:] = argv
        # Force reimport so env is re-read
        sys.modules.pop(_MODULE, None)
        import importlib

        mod = importlib.import_module(_MODULE)
        with patch("asyncpg.connect", AsyncMock(return_value=conn)):
            await mod.main()
    finally:
        sys.argv[:] = saved
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop(_MODULE, None)


async def test_idempotent_provisioning(capsys: pytest.CaptureFixture[str]) -> None:
    """Running the provisioner twice yields the same UUID."""
    conn = _make_conn(insert_tag="INSERT 0 1", user_id=_FAKE_UUID)
    await _run_main(conn, _env())
    out1 = capsys.readouterr().out

    # Second call: INSERT returns conflict tag, fetchval still returns same UUID
    conn2 = _make_conn(insert_tag="INSERT 0 0", user_id=_FAKE_UUID)
    await _run_main(conn2, _env())
    out2 = capsys.readouterr().out

    assert _FAKE_UUID in out1
    assert _FAKE_UUID in out2


async def test_missing_email_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """No email set should exit with SystemExit(1)."""
    conn = _make_conn()
    with pytest.raises(SystemExit) as exc_info:
        await _run_main(
            conn,
            _env(**{"JOURNAL_OPERATOR_EMAIL": None}),  # type: ignore[arg-type]
        )
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Email required" in err or "email" in err.lower()


async def test_existing_row_graceful(capsys: pytest.CaptureFixture[str]) -> None:
    """When INSERT is a no-op (conflict), script prints 'already exists' and exits 0."""
    conn = _make_conn(insert_tag="INSERT 0 0", user_id=_FAKE_UUID)
    await _run_main(conn, _env())
    out = capsys.readouterr().out
    assert "already exists" in out
    assert _FAKE_UUID in out
