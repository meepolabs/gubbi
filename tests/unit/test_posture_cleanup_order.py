"""Error precedence of the cleanup that wraps the bootstrap posture scratch database."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from tests.integration.test_bootstrap_role_posture import cleanup_after


class BodyError(Exception):
    pass


class DropError(Exception):
    pass


class RestoreError(Exception):
    pass


def _step(calls: list[str], label: str, error: Exception | None) -> Callable[[], Awaitable[None]]:
    async def run() -> None:
        calls.append(label)
        if error is not None:
            raise error

    return run


async def _drive(
    calls: list[str],
    *,
    body_error: Exception | None = None,
    drop_error: Exception | None = None,
    restore_error: Exception | None = None,
) -> None:
    async with cleanup_after(
        ("drop", _step(calls, "drop", drop_error)),
        ("restore", _step(calls, "restore", restore_error)),
    ):
        calls.append("body")
        if body_error is not None:
            raise body_error


async def test_failing_body_survives_both_cleanup_failures_as_notes() -> None:
    calls: list[str] = []

    with pytest.raises(BodyError, match="body broke") as caught:
        await _drive(
            calls,
            body_error=BodyError("body broke"),
            drop_error=DropError("drop broke"),
            restore_error=RestoreError("restore broke"),
        )

    assert calls == ["body", "drop", "restore"]
    assert caught.value.__notes__ == [
        "cleanup also failed: DropError: drop broke",
        "cleanup also failed: RestoreError: restore broke",
    ]


@pytest.mark.parametrize(
    ("drop_error", "restore_error", "expected"),
    [
        pytest.param(DropError("drop broke"), None, DropError, id="drop-fails"),
        pytest.param(None, RestoreError("restore broke"), RestoreError, id="restore-fails"),
    ],
)
async def test_clean_body_raises_the_failing_cleanup_step(
    drop_error: Exception | None,
    restore_error: Exception | None,
    expected: type[Exception],
) -> None:
    calls: list[str] = []

    with pytest.raises(expected):
        await _drive(calls, drop_error=drop_error, restore_error=restore_error)

    assert calls == ["body", "drop", "restore"]


async def test_clean_body_raises_the_first_cleanup_failure_with_later_ones_noted() -> None:
    calls: list[str] = []

    with pytest.raises(DropError) as caught:
        await _drive(
            calls, drop_error=DropError("drop broke"), restore_error=RestoreError("restore broke")
        )

    assert "cleanup also failed: RestoreError: restore broke" in caught.value.__notes__


async def test_clean_body_and_cleanup_raise_nothing() -> None:
    calls: list[str] = []

    await _drive(calls)

    assert calls == ["body", "drop", "restore"]
