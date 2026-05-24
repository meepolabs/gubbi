"""Lifespan canonical probe-order pin (T1).

Uses an aggregating recording runner to capture the (name, required)
tuples the lifespan would have run, without invoking probe.run().
Round 4 fix-up (lifespan-probe streamline, 2026-05-24): the lifespan
now drives ``StartupRunner.run`` TWICE -- Phase 1 (config-only) before
wiring, Phase 2 (resources) after wiring -- so the recorder must
preserve both call lists rather than overwriting on the second call.
Adding a probe -- or reordering -- forces this test to update, which is
the structural guard the spec calls for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from gubbi_common.bootstrap import RecordedProbe

import gubbi.main
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import structlog
    from gubbi_common.bootstrap import ProbeOutcome, StartupProbe


class _AggregatingRecordingRunner:
    """Stand-in for ``StartupRunner`` that records each ``run()`` call.

    The lifespan body now invokes ``runner.run`` twice (Phase 1 = config
    probes, Phase 2 = resource probes); the upstream
    ``RecordingProbeRunner`` from ``gubbi_common.bootstrap.testing``
    overwrites ``self.recorded`` on each call, which would erase the
    Phase-1 list before the test could read it.  This local subclass
    appends one ``list[RecordedProbe]`` per ``run()`` invocation so the
    test can assert against both phases.
    """

    def __init__(self) -> None:
        self.calls: list[list[RecordedProbe]] = []

    async def run(
        self,
        probes: Sequence[StartupProbe],
        *,
        logger: structlog.stdlib.AsyncBoundLogger | None = None,
    ) -> tuple[ProbeOutcome, ...]:
        del logger  # accepted for signature parity with StartupRunner.run
        self.calls.append([RecordedProbe(p.name, p.required) for p in probes])
        return ()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_canonical_probe_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan composes the canonical two-phase probe sequence in order.

    Pinned via :class:`_AggregatingRecordingRunner`: every probe instance
    handed to ``run()`` is captured as a ``(name, required)`` tuple and
    grouped by call.  Phase 1 fires the config-only probes BEFORE the
    pool / OAuth / Hydra / Redis wiring; Phase 2 fires the resource
    probes AFTER wiring.  Adding / reordering / re-phasing a probe forces
    this test to update, which is the structural guard the spec calls
    for.

    The lifespan's other dependencies (pool init, OAuth, Hydra, Redis
    connection objects) are still mocked via the shared
    ``_patch_lifespan_dependencies`` helper because the lifespan body
    still constructs them BETWEEN the two probe phases.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)

    recording = _AggregatingRecordingRunner()

    with patch("gubbi.main.StartupRunner", return_value=recording):
        app = FastAPI(lifespan=gubbi.main.lifespan)
        async with LifespanManager(app):
            pass

    expected_phase_1 = [
        RecordedProbe(name="bind_address", required=True),
    ]
    expected_phase_2 = [
        RecordedProbe(name="pg_log", required=True),
        RecordedProbe(name="redis_ping", required=True),
        RecordedProbe(name="replica_count", required=False),
    ]
    assert (
        len(recording.calls) == 2
    ), f"expected runner.run to be called twice (config + resource); got {len(recording.calls)}"
    assert (
        recording.calls[0] == expected_phase_1
    ), f"phase 1 (config) probe order broken: expected {expected_phase_1}, got {recording.calls[0]}"
    assert recording.calls[1] == expected_phase_2, (
        f"phase 2 (resource) probe order broken: expected {expected_phase_2}, "
        f"got {recording.calls[1]}"
    )
