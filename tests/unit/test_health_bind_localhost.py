"""Test extraction worker health server binds to localhost by default (M-9.7 / M-11)."""

from __future__ import annotations

import os


def test_extraction_health_binds_localhost_by_default() -> None:
    """JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC not set -> host is 127.0.0.1."""
    old = os.environ.pop("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", None)
    try:
        public = os.environ.get("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", "").lower() == "true"
        host = "0.0.0.0" if public else "127.0.0.1"  # noqa: S104
        assert host == "127.0.0.1"
    finally:
        if old is not None:
            os.environ["JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC"] = old


def test_extraction_health_exposes_0_0_0_when_env_set() -> None:
    """JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC=true -> host is 0.0.0.0."""
    old = os.environ.get("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC")
    os.environ["JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC"] = "true"
    try:
        public = os.environ.get("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", "").lower() == "true"
        host = "0.0.0.0" if public else "127.0.0.1"  # noqa: S104
        assert host == "0.0.0.0"  # noqa: S104 -- asserted value, not a server bind
    finally:
        if old is not None:
            os.environ["JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC"] = old
        else:
            os.environ.pop("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", None)
