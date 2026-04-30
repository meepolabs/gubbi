"""Response-size safety net for MCP tool outputs (TASK-03.23).

Guards against pathological single-record sizes (e.g. a 200KB entry)
slipping through pagination. Errors-out with a structured response
rather than silently truncating -- the error is an actionable signal
for the model to retry with tighter parameters.
"""

from __future__ import annotations

import json
from typing import Any

_ERROR_THRESHOLD_CHARS: int = 80_000
# When the size check fires, suggested_max_limit = threshold / (size_per_record) * factor.
# 0.5 gives a 2x headroom margin over the per-record average.
_SUGGESTED_LIMIT_HEADROOM_FACTOR: float = 0.5


def _assert_response_ok(
    payload: Any,  # noqa: ANN401
    *,
    tool_name: str | None = None,
    error_threshold_chars: int = _ERROR_THRESHOLD_CHARS,
) -> dict[str, Any] | None:
    """Check payload size. Returns None if OK, returns error dict if too large.

    Caller pattern::

        err = _assert_response_ok(payload, tool_name="journal_timeline")
        if err is not None:
            await _report_oversized(tool_name, err)
            return err
        return result
    """
    try:
        size = len(json.dumps(payload))
    except (TypeError, ValueError):
        return None  # un-serializable payloads pass through; serialization will fail later

    # Always record the histogram so we have telemetry coverage on the
    # happy path as well.
    if tool_name is not None:
        from journalctl.telemetry.metrics import record_tool_response_size

        record_tool_response_size(size, tool_name)
    if size > error_threshold_chars:
        record_count = max(1, _estimate_record_count(payload))
        size_per_record = size / record_count
        suggested = max(
            1,
            int(error_threshold_chars / size_per_record * _SUGGESTED_LIMIT_HEADROOM_FACTOR),
        )
        return {
            "error": "response_too_large",
            "message": "Reduce limit or use more specific filters",
            "current_size_chars": size,
            "suggested_max_limit": suggested,
        }
    return None


async def _report_oversized(tool_name: str, err: dict[str, Any]) -> None:
    """Emit the OTel size-check span for an oversized tool response.

    Centralises the fire-and-forget OTel call that all read tools make
    when _assert_response_ok fires. Best-effort -- failures are swallowed
    so they never block the error response.
    """
    try:
        from journalctl.telemetry.spans import record_mcp_tool_response_size_check  # noqa: PLC0415

        async with record_mcp_tool_response_size_check(
            tool_name,
            size_chars=err["current_size_chars"],
            error_threshold_hit=True,
        ):
            pass
    except Exception:  # noqa: BLE001, S110
        pass


def _estimate_record_count(payload: Any) -> int:  # noqa: ANN401
    """Best-effort count of records in the payload for suggested_max_limit heuristic."""
    if isinstance(payload, dict):
        for key in ("entries", "results", "conversations", "messages", "timeline"):
            val = payload.get(key)
            if isinstance(val, list) and val:
                return len(val)
        return 1
    if isinstance(payload, list):
        return max(1, len(payload))
    return 1
