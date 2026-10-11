"""The extraction worker's heartbeat: a counter incremented on a fixed interval.

``extraction.worker.heartbeat`` grows by one per interval while the worker's
event loop runs. A counter, not a timestamp gauge, so a stalled loop in a live
process still reads as absent: the exporter keeps sending the unchanged total
and its rate drops to zero. Alert on the rate, which separates a dead or stuck
worker from an empty queue.
"""

from __future__ import annotations

from typing import Final

from gubbi.extraction.supervised_loop import run_every
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_counter

__all__: list[str] = ["WORKER_HEARTBEAT_TASK_NAME", "run_worker_heartbeat"]

WORKER_HEARTBEAT_TASK_NAME: Final = "extraction_worker_heartbeat"


async def _beat() -> None:
    record_counter(MetricNames.EXTRACTION_WORKER_HEARTBEAT)


async def run_worker_heartbeat(*, interval_secs: float) -> None:
    """Beat once now, then once every ``interval_secs`` until cancelled."""
    await _beat()
    await run_every(WORKER_HEARTBEAT_TASK_NAME, _beat, interval_secs=interval_secs)
