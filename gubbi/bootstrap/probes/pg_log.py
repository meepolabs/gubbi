"""Postgres log-settings probe.

Wraps :func:`gubbi_common.bootstrap.probe_pg_log_settings` in a
``StartupProbe`` shape. Refuses to start when the cluster is configured
to capture statement text or bound parameters in its log -- which would
silently turn the DB into a plaintext sink for journal content.

Mode is consumed from ``settings.pg_log_probe_mode`` (one of STRICT /
WARN / OFF). STRICT raises on any unsafe GUC; WARN logs and returns;
OFF skips the probe entirely.

Required: the underlying contract is a security guard that a
misconfigured operator must not silently bypass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gubbi_common.bootstrap import (
    PgLogProbeError,
    PgLogProbeMode,
    ProbeResult,
    ProbeStatus,
    probe_pg_log_settings,
)

if TYPE_CHECKING:
    import asyncpg


@dataclass
class PgLogProbe:
    """Probe that fails when Postgres log GUCs would capture plaintext."""

    pool: asyncpg.Pool
    mode: PgLogProbeMode
    name: str = "pg_log"
    required: bool = True
    timeout_s: float = 5.0

    async def run(self) -> ProbeResult:
        """Check Postgres log GUCs; fail or warn per ``mode``.

        ``probe_pg_log_settings`` raises ``PgLogProbeError`` only in
        STRICT mode -- WARN logs in-place and returns; OFF skips the
        DB query. So the except branch fires only when STRICT detected
        unsafe GUCs; ``required=True`` makes the FAIL escalate to
        ``ProbeFailure`` and abort boot.
        """
        try:
            await probe_pg_log_settings(self.pool, mode=self.mode)
        except PgLogProbeError as exc:
            return ProbeResult(
                ProbeStatus.FAIL,
                diagnostic={"mode": self.mode.value, "findings": str(exc)},
            )
        return ProbeResult(ProbeStatus.OK, diagnostic={"mode": self.mode.value})
