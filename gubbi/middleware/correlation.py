"""Correlation ID middleware for gubbi.

Re-exports the canonical implementation from gubbi_common.middleware,
which extracts ``X-Correlation-ID`` from inbound HTTP requests,
generates a UUID4 if absent, stores it in a ContextVar for structured
logging and span attributes, and echoes it back in the response header.

Also re-exports the canonical envelope type and helpers from
:mod:`gubbi_common.correlation` (gubbi-common 0.13.0) so gubbi
code that needs the ``CorrelationContext`` shape, the ContextVar
handles, or the shared ``cid_from_scope`` helper has a single import
point on the gubbi side too.
"""

from gubbi_common.correlation import (
    CorrelationContext,
    cid_from_scope,
    get_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)
from gubbi_common.middleware import CorrelationIDMiddleware

__all__ = [
    "CorrelationContext",
    "CorrelationIDMiddleware",
    "cid_from_scope",
    "get_correlation_id",
    "reset_correlation_id",
    "set_correlation_id",
]
