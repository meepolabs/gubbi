"""Supervision for long-running background loops owned by a process lifespan.

A loop runs one tick per interval. A tick that raises is logged and counted
and the loop carries on; a loop that stops on its own, by returning or
raising, is logged and counted once. Shutdown cancels every loop and waits a
bounded time for them to unwind; a loop still running at the deadline is
logged and counted, and is given one more bounded wait before the caller
tears down the resources it uses.

Every count goes to the ``background_task.died_total`` counter with the
loop's name and a ``stage`` of ``tick``, ``loop``, ``not_started`` or
``stop_timeout``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

import structlog

from gubbi.telemetry.metrics import record_background_task_died

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable
    from typing import Any

__all__: list[str] = [
    "SHUTDOWN_TIMEOUT_SECS",
    "TEARDOWN_GRACE_SECS",
    "report_not_started",
    "run_every",
    "start_supervised",
    "stop_supervised",
    "stop_supervised_before_teardown",
]

logger = structlog.get_logger(__name__)

SHUTDOWN_TIMEOUT_SECS: Final = 5.0
TEARDOWN_GRACE_SECS: Final = 1.0

_STAGE_TICK: Final = "tick"
_STAGE_LOOP: Final = "loop"
_STAGE_NOT_STARTED: Final = "not_started"
_STAGE_STOP_TIMEOUT: Final = "stop_timeout"


async def run_every(
    name: str,
    tick: Callable[[], Awaitable[object]],
    *,
    interval_secs: float,
) -> None:
    """Sleep ``interval_secs``, run ``tick``, forever; a raising tick does not end the loop."""
    while True:
        await asyncio.sleep(interval_secs)
        try:
            await tick()
        except Exception:
            await logger.error("background_tick_failed", task=name, exc_info=True)
            record_background_task_died(task=name, stage=_STAGE_TICK)


def start_supervised(name: str, loop: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    """Run ``loop`` as a task that logs and counts its own unexpected end."""
    return asyncio.create_task(_supervise(name, loop), name=name)


async def _supervise(name: str, loop: Coroutine[Any, Any, None]) -> None:
    try:
        await loop
    except Exception:
        await logger.error("background_task_died", task=name, exc_info=True)
    else:
        await logger.error("background_task_died", task=name, reason="returned")
    record_background_task_died(task=name, stage=_STAGE_LOOP)


async def stop_supervised(
    tasks: Iterable[asyncio.Task[None]],
    *,
    timeout_secs: float = SHUTDOWN_TIMEOUT_SECS,
) -> set[asyncio.Task[None]]:
    """Cancel ``tasks``, wait up to ``timeout_secs``, and return those still running.

    A task still running at the deadline is logged and counted, then left
    behind, so shutdown never hangs on a loop that swallows its cancellation.
    """
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if not pending:
        return set()
    _, stuck = await asyncio.wait(pending, timeout=timeout_secs)
    for task in stuck:
        await logger.error("background_task_stop_timeout", task=task.get_name())
        record_background_task_died(task=task.get_name(), stage=_STAGE_STOP_TIMEOUT)
    return stuck


async def stop_supervised_before_teardown(
    tasks: Iterable[asyncio.Task[None]],
    *,
    timeout_secs: float = SHUTDOWN_TIMEOUT_SECS,
    grace_secs: float = TEARDOWN_GRACE_SECS,
) -> None:
    """Stop ``tasks`` ahead of closing the resources they use, bounded in time.

    A task that outlives the stop deadline gets one more ``grace_secs`` wait;
    one still running after that is logged and the caller's teardown goes
    ahead under it.
    """
    stuck = await stop_supervised(tasks, timeout_secs=timeout_secs)
    if not stuck:
        return
    _, alive = await asyncio.wait(stuck, timeout=grace_secs)
    for task in alive:
        await logger.error("background_task_alive_at_teardown", task=task.get_name())


async def report_not_started(name: str, reason: str) -> None:
    """Log and count a loop that should run but cannot be started."""
    await logger.error("background_task_not_started", task=name, reason=reason)
    record_background_task_died(task=name, stage=_STAGE_NOT_STARTED)
