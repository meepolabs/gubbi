"""Capture every log surface a structlog emit on this codebase can write to.

Two surfaces are needed. Structured fields land on the ``LogRecord`` and
are rendered by the ProcessorFormatter. The exception itself does NOT:
the configured processor chain ends in ``ExceptionPrettyPrinter``, which
pops ``exc_info`` and prints the formatted traceback to its own file
object, so a caller passing an exception (``logger.exception(...)``, or
``exc_info=<exc>``) leaks through a surface that inspecting
``LogRecord.exc_text`` cannot see.

:func:`log_capture_handler` is wired as the ``log_capture`` fixture in
``tests/conftest.py``; test modules take that fixture by name.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__: list[str] = ["LogCapture", "log_capture_handler"]


class LogCapture(logging.Handler):
    """Collect structured log lines plus pretty-printed tracebacks."""

    def __init__(self, exception_sink: io.StringIO) -> None:
        super().__init__(level=logging.DEBUG)
        self._exception_sink = exception_sink
        self.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    @property
    def text(self) -> str:
        """Every captured structured line plus every pretty-printed traceback."""
        return "\n".join([*self.lines, self._exception_sink.getvalue()])


def log_capture_handler() -> Iterator[LogCapture]:
    """Attach a root handler and redirect the pretty-printer's own output."""
    printers = [
        processor
        for processor in structlog.get_config()["processors"]
        if isinstance(processor, structlog.processors.ExceptionPrettyPrinter)
    ]
    assert printers, (
        "the configured structlog chain has no ExceptionPrettyPrinter; "
        "the traceback surface this fixture captures has moved -- re-derive it "
        "from gubbi_common.telemetry.logging.initialize_logger before trusting "
        "any assertion built on this fixture"
    )
    sink = io.StringIO()
    original_files = [printer._file for printer in printers]
    for printer in printers:
        printer._file = sink

    handler = LogCapture(sink)
    root = logging.getLogger()
    original_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(original_level)
        for printer, original_file in zip(printers, original_files, strict=True):
            printer._file = original_file
