"""Test client_ip honours JOURNAL_TRUST_FORWARDED_HEADERS flag (M-9.3)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from gubbi.config import get_settings


class TestClientIpTrustFlag:
    """M-9.3: client_ip default is NOT to trust X-Forwarded-For."""

    async def test_client_ip_trust_flag_default_off(self) -> None:
        """Default: request.client.host wins; XFF header is ignored."""
        from gubbi.oauth.forms import client_ip

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "10.0.0.5, 172.16.0.1"
        mock_client = MagicMock()
        mock_client.host = "192.168.1.10"
        mock_request.client = mock_client

        # Ensure trust_forwarded_headers is False (default)
        old_val = os.environ.get("JOURNAL_TRUST_FORWARDED_HEADERS")
        if "JOURNAL_TRUST_FORWARDED_HEADERS" in os.environ:
            del os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"]
        get_settings.cache_clear()

        try:
            ip = client_ip(mock_request)
            assert ip == "192.168.1.10", "Without trust flag, XFF must be ignored"
        finally:
            if old_val is not None:
                os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"] = old_val

    async def test_client_ip_trust_flag_enabled(self) -> None:
        """When trust flag set to 'true', RIGHTMOST XFF wins (DEC-086)."""
        from gubbi.oauth.forms import client_ip

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "10.0.0.5, 172.16.0.1"
        mock_client = MagicMock()
        mock_client.host = "192.168.1.10"
        mock_request.client = mock_client

        os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"] = "true"
        get_settings.cache_clear()

        try:
            ip = client_ip(mock_request)
            assert ip == "172.16.0.1", (
                "With trust flag, rightmost XFF is the trusted-proxy stamp " "(DEC-086 rule 4)"
            )
        finally:
            del os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"]

    async def test_login_rate_limit_uses_rightmost_xff(self) -> None:
        """DEC-086 rule 4: rightmost XFF entry is the trusted-proxy stamp.

        Supplies a 3-hop XFF chain "1.2.3.4, 5.6.7.8, 9.10.11.12" and
        asserts client_ip() returns the rightmost ("9.10.11.12"), not
        the leftmost client-controllable value. Regression-locks the
        H-A4 fix (gubbi-common 0.10.0 client_ip helper).
        """
        from gubbi.oauth.forms import client_ip

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "1.2.3.4, 5.6.7.8, 9.10.11.12"
        mock_client = MagicMock()
        mock_client.host = "192.168.1.10"
        mock_request.client = mock_client

        os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"] = "true"
        get_settings.cache_clear()

        try:
            ip = client_ip(mock_request)
            assert ip == "9.10.11.12", "Rightmost XFF entry must win per DEC-086 rule 4"
        finally:
            del os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"]

    async def test_client_ip_no_xff_falls_back_to_host(self) -> None:
        """When no XFF header, falls back to request.client.host."""
        from gubbi.oauth.forms import client_ip

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""
        mock_client = MagicMock()
        mock_client.host = "192.168.1.10"
        mock_request.client = mock_client

        os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"] = "true"
        get_settings.cache_clear()

        try:
            ip = client_ip(mock_request)
            assert ip == "192.168.1.10"
        finally:
            del os.environ["JOURNAL_TRUST_FORWARDED_HEADERS"]
