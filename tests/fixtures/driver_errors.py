"""The failing-statement exception a PostgreSQL error would raise, with markers.

Telemetry tests assert that no driver-supplied text reaches a span, a log
line or a response body. They need an exception that genuinely carries
such text, so :func:`planted_driver_error` embeds a distinct marker in
each class of identity-bearing value a real server message or ``DETAIL``
block can quote back, and :func:`assert_no_markers` scans a flattened
surface for all of them.

One source deliberately: a surface-specific copy of these markers would
let one test's notion of "leaking text" drift from another's.
"""

from __future__ import annotations

import json

import asyncpg

__all__: list[str] = [
    "MARKERS",
    "MARKER_DETAIL",
    "MARKER_IP",
    "MARKER_SID",
    "MARKER_TOKEN",
    "MARKER_UNICODE",
    "MARKER_USER_AGENT",
    "SQLSTATE",
    "assert_no_markers",
    "markers_present",
    "planted_driver_error",
]

# Synthetic values planted in the exception the driver would raise. Each
# stands for one class of identity-bearing text a real PostgreSQL error
# for this statement can quote back: the session id, the originating IP,
# the User-Agent, a bearer credential and the server's DETAIL block.
# Deliberately low-entropy, hyphenated words rather than key-shaped
# strings: a realistic-looking credential here trips the repo's secret
# scanner, and the assertions only need each marker to be unique.
MARKER_SID = "planted-session-marker"
MARKER_IP = "203.0.113.77"
MARKER_USER_AGENT = "PlantedAgent/9.9"
MARKER_TOKEN = "planted-bearer-marker"
MARKER_DETAIL = "planted-detail-marker"

# A journal holds arbitrary user text, so a rejected row's values reach a
# driver message as any codepoint the user typed. Written as escapes to
# keep this source ASCII-only; the VALUE is non-ASCII, which is the point
# -- ``JSONRenderer`` emits ``ensure_ascii=True``, so this marker appears
# on a captured log line only in its ``\uXXXX`` form and a raw substring
# scan cannot see it.
MARKER_UNICODE = "planted-caf\u00e9-\u00f1-marker"

MARKERS: tuple[str, ...] = (
    MARKER_SID,
    MARKER_IP,
    MARKER_USER_AGENT,
    MARKER_TOKEN,
    MARKER_DETAIL,
    MARKER_UNICODE,
)

SQLSTATE = "42501"


def planted_driver_error() -> asyncpg.PostgresError:
    """Build a privilege-rejection exception with every marker embedded.

    Both halves matter: ``str(exc)`` concatenates the message AND the
    ``DETAIL`` block, so a surface that stringifies the exception leaks
    both, while one that reads only ``exc.detail`` leaks only the
    second.
    """
    exc = asyncpg.exceptions.InsufficientPrivilegeError(
        "permission denied for table audit_log while inserting "
        f"sid={MARKER_SID} ip={MARKER_IP} ua={MARKER_USER_AGENT} "
        f"token={MARKER_TOKEN} note={MARKER_UNICODE}"
    )
    exc.detail = f"DETAIL: rejected row carried {MARKER_DETAIL}"
    return exc


def _escaped_forms(marker: str) -> tuple[str, ...]:
    """Return every spelling of *marker* a captured log surface can hold.

    The raw string covers a pretty-printed traceback and flattened span
    attributes, both written verbatim. ``json.dumps`` covers the
    structured line: the configured renderer escapes non-ASCII to
    ``\\uXXXX`` and backslashes to ``\\\\``, so a marker containing either
    is present only in escaped form. The quotes ``json.dumps`` adds are
    stripped -- the marker is a fragment of a longer field, not the whole
    value.
    """
    encoded = json.dumps(marker)[1:-1]
    return (marker,) if encoded == marker else (marker, encoded)


def markers_present(text: str) -> list[str]:
    """Return the planted markers *text* carries, raw or JSON-escaped.

    The positive counterpart of :func:`assert_no_markers`, sharing its
    escape rule so a control cannot certify a surface the real scan would
    read differently.
    """
    return [marker for marker in MARKERS if any(form in text for form in _escaped_forms(marker))]


def assert_no_markers(text: str, surface: str) -> None:
    """Fail when *text* carries any planted marker, raw or JSON-escaped."""
    leaked = markers_present(text)
    assert not leaked, f"{surface} exported planted driver text {leaked}: {text}"
