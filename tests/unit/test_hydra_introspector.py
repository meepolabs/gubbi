"""Unit tests for Hydra token introspector."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest
import structlog

from journalctl.auth.hydra import (
    HydraError,
    HydraIntrospector,
    HydraInvalidToken,
    HydraUnreachable,
    InMemoryHydraCache,
    TokenClaims,
    _cache_key,
    _log_fp,
    _token_fp,
)

FAKE_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
FAKE_TOKEN = "test-access-token-xyz"
FAKE_TOKEN_FINGERPRINT = _token_fp(FAKE_TOKEN)


@pytest.fixture
def mock_logger() -> MagicMock:
    logger = MagicMock(spec=structlog.stdlib.AsyncBoundLogger)
    logger.error = AsyncMock()
    logger.info = AsyncMock()
    return logger


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def introspector(mock_httpx_client: MagicMock, mock_logger: MagicMock) -> HydraIntrospector:
    return HydraIntrospector(
        admin_url="http://hydra:4445",
        http_client=mock_httpx_client,
        logger=mock_logger,
        timeout_seconds=3.0,
    )


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


class TestTokenFingerprint:
    def test_produces_12_char_hex(self) -> None:
        fp = _log_fp("any-token")
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_same_token_same_fp(self) -> None:
        assert _log_fp("same") == _log_fp("same")

    def test_different_token_different_fp(self) -> None:
        assert _log_fp("aaa") != _log_fp("bbb")

    def test_token_fp_alias_matches_log_fp(self) -> None:
        """Back-compat alias: _token_fp should equal _log_fp."""
        assert _token_fp("abc") == _log_fp("abc")


class TestCacheKey:
    def test_cache_key_is_full_sha256(self) -> None:
        """Cache key must be 64 hex chars (256 bits) to prevent collision-based cross-tenant leaks."""
        key = _cache_key("any-token")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_differs_from_log_fp(self) -> None:
        """Log fingerprint truncation must not leak into cache lookup."""
        token = "some-access-token"
        assert _cache_key(token) != _log_fp(token)
        assert _cache_key(token).startswith(_log_fp(token))


class TestInMemoryHydraCache:
    def test_set_and_get(self) -> None:
        cache = InMemoryHydraCache()
        claims = TokenClaims(sub=FAKE_UUID, scope="journal", exp=int(time.time()) + 3600)
        cache.set("abc", claims)
        assert cache.get("abc") == claims

    def test_get_missing_key_returns_none(self) -> None:
        cache = InMemoryHydraCache()
        assert cache.get("nonexistent") is None

    def test_different_keys_stored_separately(self) -> None:
        cache = InMemoryHydraCache()
        claims1 = TokenClaims(sub=FAKE_UUID, scope="journal", exp=9999999999)
        claims2 = TokenClaims(sub=FAKE_UUID, scope="openid", exp=9999999999)
        cache.set("key1", claims1)
        cache.set("key2", claims2)
        assert cache.get("key1") == claims1
        assert cache.get("key2") == claims2

    def test_overwriting_key_updates_value(self) -> None:
        cache = InMemoryHydraCache()
        claims1 = TokenClaims(sub=FAKE_UUID, scope="journal", exp=9999999999)
        claims2 = TokenClaims(sub=FAKE_UUID, scope="openid", exp=9999999999)
        cache.set("key", claims1)
        cache.set("key", claims2)
        assert cache.get("key") == claims2


class TestIntrospectBasic:
    async def test_active_true_valid_uuid_returns_claims(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock, mock_logger: MagicMock
    ) -> None:
        exp_ts = int(time.time()) + 3600
        resp = _make_response(
            200,
            {
                "active": True,
                "sub": str(FAKE_UUID),
                "scope": "openid journal profile",
                "exp": exp_ts,
            },
        )
        mock_httpx_client.post.return_value = resp

        result = await introspector.introspect(FAKE_TOKEN)

        assert result.sub == FAKE_UUID
        assert result.scope == "openid journal profile"
        assert result.exp == exp_ts
        mock_logger.info.assert_called_once()
        log_kwargs = mock_logger.info.call_args[1]
        assert log_kwargs["token_fp"] == FAKE_TOKEN_FINGERPRINT
        assert "sub" in log_kwargs
        assert "scopes" in log_kwargs
        mock_logger.error.assert_not_called()

    async def test_returns_cached_claims_on_second_call(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        cache = InMemoryHydraCache()
        introspector.cache = cache
        exp_ts = int(time.time()) + 3600
        resp = _make_response(
            200,
            {
                "active": True,
                "sub": str(FAKE_UUID),
                "scope": "journal",
                "exp": exp_ts,
            },
        )
        mock_httpx_client.post.return_value = resp

        claims1 = await introspector.introspect(FAKE_TOKEN)
        claims2 = await introspector.introspect(FAKE_TOKEN)

        assert claims1 == claims2
        mock_httpx_client.post.assert_called_once()

    async def test_explicit_cache_hit_skips_http_call(
        self, mock_httpx_client: MagicMock, mock_logger: MagicMock
    ) -> None:
        cache = InMemoryHydraCache()
        exp_ts = int(time.time()) + 3600
        cached_claims = TokenClaims(sub=FAKE_UUID, scope="journal", exp=exp_ts)
        cache.set(_cache_key(FAKE_TOKEN), cached_claims)

        introspector = HydraIntrospector(
            admin_url="http://hydra:4445",
            http_client=mock_httpx_client,
            logger=mock_logger,
            cache=cache,
        )

        result = await introspector.introspect(FAKE_TOKEN)
        assert result == cached_claims
        mock_httpx_client.post.assert_not_called()

    async def test_cache_only_stores_success_no_negative_results(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        cache = InMemoryHydraCache()
        introspector.cache = cache
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": False,
                "sub": str(FAKE_UUID),
                "scope": "journal",
                "exp": int(time.time()) + 3600,
            },
        )

        with pytest.raises(HydraInvalidToken):
            await introspector.introspect(FAKE_TOKEN)

        # Cache should not have stored the negative result
        assert introspector.cache.get(_cache_key(FAKE_TOKEN)) is None

    async def test_raw_token_never_logged(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock, mock_logger: MagicMock
    ) -> None:
        exp_ts = int(time.time()) + 3600
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": True,
                "sub": str(FAKE_UUID),
                "scope": "journal",
                "exp": exp_ts,
            },
        )

        await introspector.introspect(FAKE_TOKEN)
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[1]
        # Verify raw token does not appear in any log argument
        for _k, value in log_call.items():
            assert FAKE_TOKEN not in str(value)
        assert log_call["token_fp"] == FAKE_TOKEN_FINGERPRINT


class TestIntrospectErrors:
    async def test_active_false_raises_invalid_token(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": False,
            },
        )
        with pytest.raises(HydraInvalidToken):
            await introspector.introspect(FAKE_TOKEN)

    async def test_missing_active_field_raises_invalid_token(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "scope": "journal",
                "exp": int(time.time()) + 3600,
            },
        )
        with pytest.raises(HydraInvalidToken):
            await introspector.introspect(FAKE_TOKEN)

    async def test_malformed_sub_raises_invalid_token(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": True,
                "sub": "not-a-uuid",
                "scope": "journal",
                "exp": int(time.time()) + 3600,
            },
        )
        with pytest.raises(HydraInvalidToken, match="malformed sub"):
            await introspector.introspect(FAKE_TOKEN)

    async def test_missing_sub_raises_invalid_token(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": True,
                "scope": "journal",
            },
        )
        with pytest.raises(HydraInvalidToken):
            await introspector.introspect(FAKE_TOKEN)

    async def test_exp_in_past_raises_expired(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": True,
                "sub": str(FAKE_UUID),
                "scope": "journal",
                "exp": int(time.time()) - 100,
            },
        )
        with pytest.raises(HydraInvalidToken, match="expired"):
            await introspector.introspect(FAKE_TOKEN)

    async def test_missing_exp_raises_invalid(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        """Hydra response lacking exp must be rejected (defense in depth)."""
        mock_httpx_client.post.return_value = _make_response(
            200,
            {
                "active": True,
                "sub": str(FAKE_UUID),
                "scope": "journal",
            },
        )
        with pytest.raises(HydraInvalidToken, match="missing exp"):
            await introspector.introspect(FAKE_TOKEN)

    async def test_non_json_body_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        """Non-JSON 200 response → HydraUnreachable, not uncaught ValueError."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        mock_httpx_client.post.return_value = resp
        with pytest.raises(HydraUnreachable, match="non-JSON"):
            await introspector.introspect(FAKE_TOKEN)

    async def test_non_object_json_body_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        """JSON array or scalar body → HydraUnreachable, not uncaught AttributeError."""
        mock_httpx_client.post.return_value = _make_response(200, [{"active": True}])  # type: ignore[arg-type]
        with pytest.raises(HydraUnreachable, match="non-object"):
            await introspector.introspect(FAKE_TOKEN)

    async def test_httpx_timeout_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)

    async def test_httpx_connect_error_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.side_effect = httpx.ConnectError("connection refused")
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)

    async def test_500_response_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(500, {"error": "internal"})
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)

    async def test_503_response_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(503, {"error": "service unavailable"})
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)

    async def test_400_response_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(400, {"error": "bad request"})
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)

    async def test_401_response_raises_unreachable(
        self, introspector: HydraIntrospector, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(401, {"error": "unauthorized"})
        with pytest.raises(HydraUnreachable):
            await introspector.introspect(FAKE_TOKEN)


class TestHydraErrorHierarchy:
    def test_unreachable_is_hydra_error(self) -> None:
        assert issubclass(HydraUnreachable, HydraError)

    def test_invalid_token_is_hydra_error(self) -> None:
        assert issubclass(HydraInvalidToken, HydraError)

    def test_unreachable_not_invalid_token(self) -> None:
        assert not issubclass(HydraInvalidToken, HydraUnreachable)
