"""Replica-count warn probe (parameterized by role).

gubbi's per-pod DB connection pool runs in-process, so under
a multi-replica deploy the cluster carries
``pool_max_per_pod * REPLICA_COUNT`` connections from gubbi alone --
and the cloud-api startup connection-budget guard does NOT see them
(it only counts cloud-api pools).

For role="web" (default) the probe surfaces the over-provisioning gap.
For role="worker" it flags a deploy-policy violation -- the extraction
worker is fixed at a single replica by convention.

Both roles emit a structured WARN + the alertable
``gateway.replica_count_warning`` counter so the gap is visible in
HyperDX before a connection-refused storm at first peak.

Soft probe (``required=False``): a WARN status records but does not
abort boot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gubbi_common.bootstrap import ProbeResult, ProbeStatus

from gubbi.telemetry.metrics import record_replica_count_warning

_VALID_ROLES = frozenset({"web", "worker"})

_NOTES: dict[str, str] = {
    "web": (
        "DB connection pool is per-pod; effective gubbi DB "
        "connections = POOL_MAX_PER_POD * REPLICA_COUNT. The "
        "cloud-api startup budget guard does not account for "
        "these -- watch the cluster max_connections headroom."
    ),
    "worker": (
        "policy violation: extraction worker is fixed at a "
        "single replica by deploy convention; "
        "JOURNAL_REPLICA_COUNT > 1 multiplies the per-pod "
        "extraction-budget logic AND the DB pool N-fold and "
        "violates the single-worker policy"
    ),
}


def _name_for_role(role: str) -> str:
    if role == "web":
        return "replica_count"
    return f"{role}_replica_count"


@dataclass
class ReplicaCountWarnProbe:
    """Probe that warns on multi-replica over-provisioning or policy violation.

    Parameterized by ``role``:
    - "web" (default): DB pool over-provisioning warning.
    - "worker": single-replica deploy-policy violation.
    """

    replica_count: int
    pool_max_per_pod: int
    role: str = "web"
    name: str = field(init=False)
    required: bool = False
    timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"ReplicaCountWarnProbe: role must be one of "
                f"{sorted(_VALID_ROLES)}; got {self.role!r}"
            )
        self.name = _name_for_role(self.role)

    async def run(self) -> ProbeResult:
        """Return WARN with diagnostic when replicas > 1.

        The structured WARN replaces the previous direct logger.warning
        emits; ``record_replica_count_warning`` continues to fire the
        alertable counter (the log-sampling-proof counterpart) so the
        alert rule contract is unchanged.
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
                "note": _NOTES[self.role],
            },
        )
