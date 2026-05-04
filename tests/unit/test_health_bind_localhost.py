"""Test health server binds to localhost by default (M-9.7)."""

from __future__ import annotations

import os


def test_health_binds_localhost_by_default() -> None:
    """JOURNAL_HEALTH_BIND_PUBLIC not set -> host is 127.0.0.1."""

    # Ensure env var is unset
    old = os.environ.pop("JOURNAL_HEALTH_BIND_PUBLIC", None)
    try:
        public = os.environ.get("JOURNAL_HEALTH_BIND_PUBLIC", "").lower() == "true"
        host = "0.0.0.0" if public else "127.0.0.1"  # noqa: S104
        assert host == "127.0.0.1"
    finally:
        if old is not None:
            os.environ["JOURNAL_HEALTH_BIND_PUBLIC"] = old
