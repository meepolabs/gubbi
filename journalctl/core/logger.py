"""Structured logging setup.

Configures both structlog (for async code in main.py) and stdlib
logging (for sync code in storage/oauth) to produce consistent
JSON output through a shared ProcessorFormatter.
"""

import logging
import os
from logging import handlers

import structlog
from structlog.types import EventDict, WrappedLogger

LOG_ROTATE_WHEN = os.getenv(key="LOG_ROTATE_WHEN", default="W6")
LOG_ROTATE_BACKUP = int(os.getenv(key="LOG_ROTATE_BACKUP", default="4"))


def _safe_add_logger_name(
    logger: WrappedLogger,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """Like add_logger_name but handles None logger.

    The MCP SDK's internal loggers pass records through the
    ProcessorFormatter where the logger reference can be None,
    causing the standard add_logger_name to crash with
    AttributeError: 'NoneType' object has no attribute 'name'.
    """
    record = event_dict.get("_record")
    if record is not None:
        event_dict["logger"] = record.name
    elif logger is not None:
        event_dict["logger"] = getattr(logger, "name", "unknown")
    else:
        event_dict["logger"] = "unknown"
    return event_dict


# Shared processors used by both structlog and ProcessorFormatter
_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    _safe_add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
    structlog.processors.JSONRenderer(),
]


def initialize_logger(logger_name: str, log_dir: str = "logs") -> None:
    """Initialize structured logging for the application.

    Sets up a ProcessorFormatter on the file handler so that ALL
    loggers (both structlog and plain stdlib) produce consistent
    JSON output.  This means modules can safely use either:

        # Async context (main.py, lifespan):
        logger = structlog.get_logger("journalctl")
        logger.info("event", key=value)

        # Sync (storage, oauth):
        logger = logging.getLogger("journalctl.oauth.login")
        logger.info("event", extra={"key": value})

    Args:
        logger_name: Name of the logger and log file.
        log_dir: Directory for log files.
    """
    os.makedirs(log_dir, exist_ok=True)

    # File handler with rotation
    log_file_path = f"{log_dir}/{logger_name}.log"
    file_handler = handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=LOG_ROTATE_WHEN,
        backupCount=LOG_ROTATE_BACKUP,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    # ProcessorFormatter: renders ALL stdlib log records through
    # structlog processors, producing consistent JSON output
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                *_SHARED_PROCESSORS,
            ],
        ),
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler],
    )

    # Configure structlog for async callers (main.py)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.AsyncBoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
