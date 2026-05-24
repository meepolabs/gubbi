"""Redis fail-fast PING probe.

Issues a single PING against a fully-constructed ``aioredis.Redis``
client. Aborts the lifespan when Redis is unreachable rather than
yielding a half-open service whose first SSE / arq / budget call
would surface ConnectionError at request time.

Required: Redis is a hard dependency of the SSE pub/sub channel and
DEC-098's audit fail-open path; a silent boot against a dead Redis
erases that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gubbi_common.bootstrap import ProbeResult, ProbeStatus

if TYPE_CHECKING:
    import redis.asyncio as aioredis


@dataclass
class RedisPingProbe:
    """Probe that aborts boot when ``client.ping()`` fails."""

    client: aioredis.Redis
    name: str = "redis_ping"
    required: bool = True
    timeout_s: float = 5.0

    async def run(self) -> ProbeResult:
        """Issue a PING; fail with a credential-free diagnostic on error.

        The probe diagnostic deliberately does NOT include the Redis
        URL: the runner's ``error_message`` field is auto-scrubbed for
        URL credentials, but a probe-author-supplied diagnostic is not.
        Operators correlating a failing PING against the deploy can use
        ``service.name`` + ``error_type`` instead.
        """
        try:
            await self.client.ping()
        except Exception as exc:
            return ProbeResult(
                ProbeStatus.FAIL,
                diagnostic={"error_type": type(exc).__name__},
            )
        return ProbeResult(ProbeStatus.OK)
