"""Worker startup wires ``probe_pg_log_settings`` after the PG pool opens.

The Arq worker hits the same encrypted INSERT path as the HTTP API
(via ``extract_conversation``), so the probe runs in both surfaces.
This test stubs every heavy dependency the ``startup`` hook touches and
asserts the probe is invoked exactly once with the worker's pool and
the env-resolved ``mode`` keyword.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from gubbi_common.bootstrap.pg_log_probe import PgLogProbeError

from gubbi.extraction import worker as worker_module


def _patch_worker_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Stub out everything the worker startup hook would normally hit.

    Critically, the ``threading.Thread`` constructor used by the worker
    is patched on the **module's** ``threading`` reference -- NOT on the
    stdlib ``threading`` module -- so global Thread creation (asyncio
    executors, etc.) stays untouched. The health-server thread the
    worker creates is replaced with a no-op stand-in whose ``start`` is
    a sync ``MagicMock``.
    """
    pool = MagicMock()
    pool.close = AsyncMock()

    monkeypatch.setattr(worker_module, "init_pool", AsyncMock(return_value=pool))

    redis_pool_stub = MagicMock()
    redis_pool_stub.aclose = AsyncMock()
    redis_client_stub = MagicMock()
    redis_client_stub.aclose = AsyncMock()
    redis_client_stub.register_script = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(
        "redis.asyncio.ConnectionPool.from_url",
        MagicMock(return_value=redis_pool_stub),
    )
    monkeypatch.setattr(
        "redis.asyncio.Redis",
        MagicMock(return_value=redis_client_stub),
    )

    # Replace the worker's ``threading`` reference so the health-server
    # thread is a no-op. Don't patch stdlib ``threading`` -- that would
    # break asyncio's internal Thread usage.
    fake_threading = MagicMock()
    fake_thread = MagicMock()
    fake_thread.start = MagicMock()
    fake_threading.Thread = MagicMock(return_value=fake_thread)
    monkeypatch.setattr(worker_module, "threading", fake_threading)

    monkeypatch.setattr(worker_module, "initialize_logger", MagicMock())
    monkeypatch.setattr(worker_module, "_build_content_cipher", MagicMock(return_value=None))

    # The Anthropic provider is constructed inside ``startup`` -- patch
    # it to a no-op so no real API client is built.
    monkeypatch.setattr(worker_module, "AnthropicProvider", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(worker_module, "ExtractionService", MagicMock(return_value=MagicMock()))

    return {
        "pool": pool,
        "redis_client": redis_client_stub,
        "redis_pool": redis_pool_stub,
    }


@pytest.mark.unit
async def test_worker_startup_calls_pg_log_probe_with_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``startup`` invokes ``probe_pg_log_settings`` with the worker pool + strict mode."""
    monkeypatch.delenv("JOURNAL_PG_LOG_PROBE_MODE", raising=False)
    handles = _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    probe_mock.assert_awaited_once()
    args, kwargs = probe_mock.call_args
    assert args == (handles["pool"],)
    assert kwargs.get("mode") == "strict"


@pytest.mark.unit
async def test_worker_startup_forwards_warn_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JOURNAL_PG_LOG_PROBE_MODE=warn`` is forwarded to the probe."""
    monkeypatch.setenv("JOURNAL_PG_LOG_PROBE_MODE", "warn")
    _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    _, kwargs = probe_mock.call_args
    assert kwargs.get("mode") == "warn"


@pytest.mark.unit
async def test_worker_startup_aborts_when_pg_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the probe raises, the worker pool is closed and the error propagates."""
    handles = _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(side_effect=PgLogProbeError("unsafe log_statement=all"))
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    with pytest.raises(PgLogProbeError):
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
def test_worker_module_exposes_probe_pg_log_settings_symbol() -> None:
    """``probe_pg_log_settings`` is bound on ``gubbi.extraction.worker`` for monkeypatch."""
    assert hasattr(worker_module, "probe_pg_log_settings"), (
        "gubbi.extraction.worker must import probe_pg_log_settings at module "
        "scope so test monkeypatch can intercept the call."
    )
