"""Drift-guard: app pool max must stay above worker max_jobs + 2 headroom.

Pins the H-5 invariant: ``APP_POOL_SIZE_MAX`` must remain large enough to
accommodate every Arq worker concurrently holding a connection plus a small
headroom for idempotency probes / audit writes during the connection-split
extraction job. Marked as ``unit`` because it does not touch a DB; it lives
under ``tests/integration/`` only because that path is the canonical home
for sizing/topology assertions in the repository.
"""

from __future__ import annotations

import pytest

from gubbi.constants import APP_POOL_SIZE_MAX
from gubbi.extraction.worker import WorkerSettings


@pytest.mark.unit
def test_app_pool_max_exceeds_worker_max_jobs() -> None:
    """``APP_POOL_SIZE_MAX`` must be at least ``WorkerSettings.max_jobs + 2``.

    The +2 headroom covers idempotency probes and audit writes that run
    alongside the worker's per-job connection during the connection-split
    extraction path (see H-5 closure for the rationale).
    """
    headroom = 2
    minimum_required = WorkerSettings.max_jobs + headroom
    assert minimum_required <= APP_POOL_SIZE_MAX, (
        f"APP_POOL_SIZE_MAX ({APP_POOL_SIZE_MAX}) must be >= "
        f"WorkerSettings.max_jobs ({WorkerSettings.max_jobs}) + {headroom} headroom; "
        "see gubbi/constants.py:18 for the H-5/H-6 rationale."
    )
