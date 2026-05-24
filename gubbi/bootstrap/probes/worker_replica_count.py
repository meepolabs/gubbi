"""Worker replica-count policy-violation probe.

M4 #138 (worker variant): the extraction worker is fixed at a SINGLE
replica by deploy convention -- a single extraction worker is the
locked shape regardless of gubbi-web replica count (Anthropic's
per-account rate limit is the bottleneck before the DB pool at this
scale). So a worker ``JOURNAL_REPLICA_COUNT > 1`` is a deploy-policy
violation, not just an over-provisioning footgun.

Same shape as :class:`ReplicaCountWarnProbe` -- structured WARN +
``gateway.replica_count_warning`` counter -- but the diagnostic
flags the deploy-policy violation explicitly so HyperDX rules can
distinguish the two emitters via the WARN event semantic (the OTel
counter NAME is shared cross-service; emitter distinction is via
the ``service.name`` RESOURCE attribute).

Used by the extraction worker's startup hook (T3 owns the wiring
in ``gubbi/extraction/worker.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from gubbi_common.bootstrap import ProbeResult, ProbeStatus

from gubbi.telemetry.metrics import record_replica_count_warning


@dataclass
class WorkerReplicaCountWarnProbe:
    """Probe that flags worker replica-count > 1 as a policy violation."""

    replica_count: int
    pool_max_per_pod: int
    name: str = "worker_replica_count"
    required: bool = False
    timeout_s: float = 1.0

    async def run(self) -> ProbeResult:
        """Return WARN with policy-violation diagnostic when replicas > 1.

        The structured WARN replaces the worker's previous direct
        ``logger.warning("extraction_worker_replica_policy_violation",
        ...)`` emit. The counter fires via
        ``record_replica_count_warning`` -- the alertable counterpart
        whose NAME (gateway.replica_count_warning) is shared across
        gubbi + worker + cloud-api so a single HyperDX rule rolls up
        across all three; emitters split via service.name.
        """
        if self.replica_count <= 1:
            return ProbeResult(
                ProbeStatus.OK,
                diagnostic={"replica_count": self.replica_count},
            )

        record_replica_count_warning(replica_count=self.replica_count)
        return ProbeResult(
            ProbeStatus.WARN,
            diagnostic={
                "replica_count": self.replica_count,
                "pool_max_per_pod": self.pool_max_per_pod,
                "pool_max_total": self.pool_max_per_pod * self.replica_count,
                "note": (
                    "policy violation: extraction worker is fixed at a "
                    "single replica by deploy convention; "
                    "JOURNAL_REPLICA_COUNT > 1 multiplies the per-pod "
                    "extraction-budget logic AND the DB pool N-fold and "
                    "violates the single-worker policy"
                ),
            },
        )
