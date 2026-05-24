"""Startup probes for the gubbi HTTP service lifespan.

Each probe is a small dataclass meeting the ``StartupProbe`` Protocol
shipped in ``gubbi_common.bootstrap``: a ``name``, ``required`` flag,
``timeout_s``, plus an async ``run() -> ProbeResult``. The runner
orchestrates the sequence and emits the structured outcome.

Probes accept their dependencies via constructor injection -- they
must NOT read ``app.state`` inside ``run()``. Settings is the source
of truth for invariants; the lifespan body extracts the relevant value
and passes it in at probe construction time.
"""

from __future__ import annotations

from gubbi.bootstrap.probes.bind_address import BindAddressProbe
from gubbi.bootstrap.probes.pg_log import PgLogProbe
from gubbi.bootstrap.probes.redis_ping import RedisPingProbe
from gubbi.bootstrap.probes.replica_count import ReplicaCountWarnProbe
from gubbi.bootstrap.probes.worker_replica_count import WorkerReplicaCountWarnProbe

__all__ = [
    "BindAddressProbe",
    "PgLogProbe",
    "RedisPingProbe",
    "ReplicaCountWarnProbe",
    "WorkerReplicaCountWarnProbe",
]
