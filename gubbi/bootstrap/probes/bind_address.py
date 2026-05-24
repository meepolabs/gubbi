"""Bind-address safety probe.

Wraps :func:`gubbi.bootstrap._gateway.check_trust_gateway_bind_address`
in a ``StartupProbe`` shape so the runner can sequence + observe it
alongside the other lifespan probes. The legacy helper continues to
own the actual classification logic; this probe's only job is to
adapt the call to ``ProbeResult`` and report a failure diagnostic
that does NOT echo the host value.

Required: trust-gateway misconfiguration is a deploy-time security
contract violation; failing here aborts boot rather than yielding a
half-open service.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from gubbi_common.bootstrap import ProbeResult, ProbeStatus

from gubbi.bootstrap._gateway import check_trust_gateway_bind_address

# Module-level structlog logger reused on every probe invocation.  The
# bind-address helper takes a structlog logger for its warning-emit path
# on the unspecified-bind case; using a module-level logger avoids the
# per-call ``structlog.get_logger("gubbi")`` re-resolution that the
# round-3 review flagged.
_logger = structlog.get_logger("gubbi")


@dataclass
class BindAddressProbe:
    """Probe that fails fast on trust_gateway + public-routable bind."""

    host: str
    trust_gateway: bool
    name: str = "bind_address"
    required: bool = True
    timeout_s: float = 5.0

    async def run(self) -> ProbeResult:
        """Invoke the bind-address classifier; fail on a public-routable bind.

        The probe diagnostic intentionally OMITS the host value -- the
        host is required for the classification call but operationally
        sensitive in structured-log sinks. The wrapped helper logs its
        own warning for the unspecified-bind case via the consumer's
        bound logger; that path is unchanged.
        """
        # The bind-address helper takes a structlog logger for its
        # warning-emit path on the unspecified-bind case. Use the
        # module-level logger here -- the probe must not log the host
        # via ``self._probe_logger`` (the runner's log_tail capture is
        # not the right channel for an operational warning).
        await check_trust_gateway_bind_address(self.host, self.trust_gateway, _logger)
        return ProbeResult(ProbeStatus.OK, diagnostic={"trust_gateway": self.trust_gateway})
