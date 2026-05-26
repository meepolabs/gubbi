"""Replica-count over-provisioning warn probe.

gubbi's per-pod DB connection pool runs in-process, so under
a multi-replica deploy the cluster carries
``pool_max_per_pod * REPLICA_COUNT`` connections from gubbi alone --
and the cloud-api startup connection-budget guard does NOT see them
(it only counts cloud-api pools).

This probe surfaces the over-provisioning gap as a structured WARN +
the alertable ``gateway.replica_count_warning`` counter so the gap is
visible in HyperDX before a connection-refused storm at first peak.
Default replica count is 1, so single-instance dev does not warn.

Soft probe (``required=False``): a WARN status records but does not
abort boot.
"""

from __future__ import annotations

from dataclasses import dataclass

from gubbi_common.bootstrap import ProbeResult, ProbeStatus

from gubbi.telemetry.metrics import record_replica_count_warning


@dataclass
class ReplicaCountWarnProbe:
    """Probe that warns on multi-replica DB pool over-provisioning."""

    replica_count: int
    pool_max_per_pod: int
    name: str = "replica_count"
    required: bool = False
    timeout_s: float = 1.0

    async def run(self) -> ProbeResult:
        """Return WARN with effective-connection diagnostic when replicas > 1.

        The structured WARN replaces the previous direct
        ``logger.warning("db_pool_over_provisioned", ...)`` emit;
        ``record_replica_count_warning`` continues to fire the alertable
        counter (the log-sampling-proof counterpart) so the alert rule
        contract is unchanged.
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
                    "DB connection pool is per-pod; effective gubbi DB "
                    "connections = POOL_MAX_PER_POD * REPLICA_COUNT. The "
                    "cloud-api startup budget guard does not account for "
                    "these -- watch the cluster max_connections headroom."
                ),
            },
        )
