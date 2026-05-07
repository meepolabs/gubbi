"""Correlation ID middleware for gubbi.

Re-exports the canonical implementation from gubbi_common.middleware,
which extracts ``X-Correlation-ID`` from inbound HTTP requests,
generates a UUID4 if absent, stores it in a ContextVar for structured
logging and span attributes, and echoes it back in the response header.
"""

from gubbi_common.middleware import CorrelationIDMiddleware

__all__ = ["CorrelationIDMiddleware"]
