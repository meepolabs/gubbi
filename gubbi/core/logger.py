"""Re-export of initialize_logger from gubbi-common.

This module re-exports the canonical ``initialize_logger`` function
provided by gubbi_common.telemetry.logging so that callers (e.g. main.py)
continue to import from the traditional ``gubbi.core.logger`` path.
"""

from gubbi_common.telemetry.logging import initialize_logger

__all__ = ["initialize_logger"]
