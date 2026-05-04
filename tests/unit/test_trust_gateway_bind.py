"""Unit tests for _check_trust_gateway_bind_address startup assertion."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, patch

import pytest

from gubbi.main import _check_trust_gateway_bind_address


@pytest.fixture
def mock_logger() -> AsyncMock:
    """Return an AsyncMock whose child attributes are themselves awaitable."""
    return AsyncMock()


class TestTrustGatewayBind:
    """Every branch of _check_trust_gateway_bind_address."""

    # -- trust_gateway=False: no constraint, no log ---------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",  # noqa: S104
            "127.0.0.1",
            "::1",
            "8.8.8.8",
            "2001:db8::1",
            "10.0.0.1",
            "192.168.1.1",
            "public.example.com",
        ],
    )
    async def test_trust_gateway_false_never_blocks(
        self, mock_logger: AsyncMock, host: str
    ) -> None:
        """When trust_gateway=False the check is a no-op for every address."""
        await _check_trust_gateway_bind_address(host, False, mock_logger)
        mock_logger.warning.assert_not_awaited()

    # -- Loopback -------------------------------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.0.0.0",
            "127.255.255.255",
            "::1",
        ],
    )
    async def test_loopback_passes(self, mock_logger: AsyncMock, host: str) -> None:
        await _check_trust_gateway_bind_address(host, True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    async def test_localhost_hostname_passes(self, mock_logger: AsyncMock) -> None:
        """'localhost' resolves to 127.0.0.1 and should pass."""
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            ],
        ):
            await _check_trust_gateway_bind_address("localhost", True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    # -- RFC 1918 private -----------------------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ],
    )
    async def test_private_rfc1918_passes(self, mock_logger: AsyncMock, host: str) -> None:
        await _check_trust_gateway_bind_address(host, True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    # -- IPv6 unique-local (fc00::/7) -----------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "fc00::1",
            "fd00::1",
            "fd12:3456:789a::1",
        ],
    )
    async def test_unique_local_v6_passes(self, mock_logger: AsyncMock, host: str) -> None:
        await _check_trust_gateway_bind_address(host, True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    # -- Link-local -----------------------------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "169.254.1.1",
            "169.254.255.255",
            "fe80::1",
            "fe80::abcd",
        ],
    )
    async def test_link_local_passes(self, mock_logger: AsyncMock, host: str) -> None:
        await _check_trust_gateway_bind_address(host, True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    # -- Unspecified (warning, not error) -------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",  # noqa: S104
            "::",
        ],
    )
    async def test_unspecified_warns(self, mock_logger: AsyncMock, host: str) -> None:
        """Unspecified bind addresses emit a warning but do not block startup."""
        await _check_trust_gateway_bind_address(host, True, mock_logger)
        mock_logger.warning.assert_awaited_once()

    # -- Public-routable ------------------------------------------------------

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "1.1.1.1",
            "2600::1",
        ],
    )
    async def test_public_ip_raises(self, mock_logger: AsyncMock, host: str) -> None:
        with pytest.raises(RuntimeError) as exc:
            await _check_trust_gateway_bind_address(host, True, mock_logger)
        msg = str(exc.value)
        assert "JOURNAL_TRUST_GATEWAY=true" in msg
        assert host in msg

    async def test_hostname_resolves_to_public_raises(self, mock_logger: AsyncMock) -> None:
        with (
            patch.object(
                socket,
                "getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
                ],
            ),
            pytest.raises(RuntimeError) as exc,
        ):
            await _check_trust_gateway_bind_address("public.example.com", True, mock_logger)
        msg = str(exc.value)
        assert "JOURNAL_TRUST_GATEWAY=true" in msg
        assert "public.example.com" in msg

    async def test_hostname_with_public_and_unspecified_raises(
        self, mock_logger: AsyncMock
    ) -> None:
        with (
            patch.object(
                socket,
                "getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", 0)),  # noqa: S104
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
                ],
            ),
            pytest.raises(RuntimeError),
        ):
            await _check_trust_gateway_bind_address("mixed.example.com", True, mock_logger)
        mock_logger.warning.assert_not_awaited()

    async def test_hostname_all_unspecified_warns(self, mock_logger: AsyncMock) -> None:
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", 0)),  # noqa: S104
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::", 0, 0, 0)),
            ],
        ):
            await _check_trust_gateway_bind_address("wildcard.example.com", True, mock_logger)
        mock_logger.warning.assert_awaited_once()

    # -- Hostname resolution failure ------------------------------------------

    async def test_unresolvable_hostname_raises(self, mock_logger: AsyncMock) -> None:
        with (
            patch.object(socket, "getaddrinfo", side_effect=socket.gaierror),
            pytest.raises(RuntimeError) as exc,
        ):
            await _check_trust_gateway_bind_address("nonexistent.invalid", True, mock_logger)
        msg = str(exc.value)
        assert "JOURNAL_TRUST_GATEWAY=true" in msg
        assert "nonexistent.invalid" in msg

    # -- Hostname resolves to private/loopback --------------------------------

    @pytest.mark.parametrize(
        ("hostname", "resolved"),
        [
            ("internal.box", "10.0.0.5"),
            ("app.local", "192.168.1.50"),
            ("proxy.local", "172.20.0.1"),
        ],
    )
    async def test_hostname_resolves_to_private_passes(
        self, mock_logger: AsyncMock, hostname: str, resolved: str
    ) -> None:
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", (resolved, 0)),
            ],
        ):
            await _check_trust_gateway_bind_address(hostname, True, mock_logger)
        mock_logger.warning.assert_not_awaited()
