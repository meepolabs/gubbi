"""Tests for the OAuth provider."""

import sqlite3
import time

import pytest
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
)
from mcp.shared.auth import OAuthClientInformationFull

from gubbi.config import OAUTH_ACCESS_TOKEN_TTL_SECS
from gubbi.oauth.provider import JournalOAuthProvider
from gubbi.oauth.storage import OAuthStorage


def _make_client(client_id: str = "test-client") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="test-secret",
        redirect_uris=["http://localhost/callback"],
    )


def _make_provider(oauth_storage: OAuthStorage) -> JournalOAuthProvider:
    return JournalOAuthProvider(
        storage=oauth_storage,
        server_url="http://localhost:8100",
    )


class TestClientRegistration:
    async def test_register_and_get_client(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        client = _make_client()
        await provider.register_client(client)

        result = await provider.get_client("test-client")
        assert result is not None
        assert result.client_id == "test-client"

    async def test_get_nonexistent_client(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        assert await provider.get_client("nonexistent") is None


class TestAuthorize:
    async def test_authorize_returns_login_url(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        client = _make_client()

        params = AuthorizationParams(
            state="test-state",
            scopes=["read"],
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )

        url = await provider.authorize(client, params)
        assert "/login" in url
        assert "client_id=test-client" in url
        assert "state=test-state" in url
        assert "code_challenge=test-challenge" in url


class TestCodeExchange:
    async def test_exchange_authorization_code(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        client = _make_client()

        # Save an auth code
        auth_code = AuthorizationCode(
            code="test-code",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("test-code", auth_code)

        # Load it
        loaded = await provider.load_authorization_code(client, "test-code")
        assert loaded is not None

        # Exchange it
        token = await provider.exchange_authorization_code(client, loaded)
        assert token.access_token
        assert token.refresh_token
        assert token.expires_in == OAUTH_ACCESS_TOKEN_TTL_SECS

        # Code is deleted after exchange (one-time use)
        assert oauth_storage.get_auth_code("test-code") is None

    async def test_exchange_code_wrong_client_burns_code(self, oauth_storage: OAuthStorage) -> None:
        """Fix #2: auth code is deleted when wrong client attempts to use it."""
        provider = _make_provider(oauth_storage)
        other_client = _make_client("other-client")

        auth_code = AuthorizationCode(
            code="test-code",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("test-code", auth_code)

        # Wrong client can't load the code
        loaded = await provider.load_authorization_code(other_client, "test-code")
        assert loaded is None

        # Code is now burned — correct client can't use it either
        correct_client = _make_client("test-client")
        loaded2 = await provider.load_authorization_code(correct_client, "test-code")
        assert loaded2 is None

    async def test_expired_code_rejected(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        client = _make_client()

        auth_code = AuthorizationCode(
            code="expired-code",
            scopes=["read"],
            expires_at=time.time() - 10,  # already expired
            client_id="test-client",
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("expired-code", auth_code)

        loaded = await provider.load_authorization_code(client, "expired-code")
        assert loaded is None


class TestTokenRefresh:
    async def test_refresh_token_rotation(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        client = _make_client()

        # Create initial tokens via code exchange
        auth_code = AuthorizationCode(
            code="code-1",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("code-1", auth_code)
        loaded = await provider.load_authorization_code(client, "code-1")
        initial_token = await provider.exchange_authorization_code(client, loaded)

        # Load the refresh token
        refresh = await provider.load_refresh_token(client, initial_token.refresh_token)
        assert refresh is not None

        # Exchange it
        new_token = await provider.exchange_refresh_token(client, refresh, ["read"])
        assert new_token.access_token != initial_token.access_token
        assert new_token.refresh_token != initial_token.refresh_token

        # Old refresh token is gone
        old_refresh = await provider.load_refresh_token(client, initial_token.refresh_token)
        assert old_refresh is None

    async def test_refresh_deletes_old_access_token(self, oauth_storage: OAuthStorage) -> None:
        """Fix #1: old access token is deleted when refresh token is exchanged."""
        provider = _make_provider(oauth_storage)
        client = _make_client()

        # Create initial tokens
        auth_code = AuthorizationCode(
            code="code-2",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="test-challenge",
            redirect_uri="http://localhost/callback",
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("code-2", auth_code)
        loaded = await provider.load_authorization_code(client, "code-2")
        initial_token = await provider.exchange_authorization_code(client, loaded)
        old_access = initial_token.access_token

        # Verify old access token exists
        assert oauth_storage.get_access_token(old_access) is not None

        # Refresh
        refresh = await provider.load_refresh_token(client, initial_token.refresh_token)
        new_token = await provider.exchange_refresh_token(client, refresh, ["read"])

        # Old access token is gone
        assert oauth_storage.get_access_token(old_access) is None
        # New access token exists
        assert oauth_storage.get_access_token(new_token.access_token) is not None


class TestAccessTokenValidation:
    async def test_load_valid_access_token(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)

        token = AccessToken(
            token="valid-token",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) + 3600,
        )
        oauth_storage.save_access_token("valid-token", token)

        loaded = await provider.load_access_token("valid-token")
        assert loaded is not None
        assert loaded.token == "valid-token"

    async def test_load_expired_access_token(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)

        token = AccessToken(
            token="expired-token",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) - 10,
        )
        oauth_storage.save_access_token("expired-token", token)

        loaded = await provider.load_access_token("expired-token")
        assert loaded is None

    async def test_load_nonexistent_token(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)
        assert await provider.load_access_token("nonexistent") is None


class TestRevocation:
    async def test_revoke_access_token(self, oauth_storage: OAuthStorage) -> None:
        provider = _make_provider(oauth_storage)

        token = AccessToken(
            token="revoke-me",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) + 3600,
        )
        oauth_storage.save_access_token("revoke-me", token)

        await provider.revoke_token(token)
        assert await provider.load_access_token("revoke-me") is None


class TestRotationAtomicity:
    async def test_rotation_leaves_no_partial_state(self, oauth_storage: OAuthStorage) -> None:
        """HIGH-1: if rotate_refresh_token fails mid-way, old state must be intact."""
        provider = _make_provider(oauth_storage)
        client = _make_client()

        # Seed tokens via normal flow
        auth_code = AuthorizationCode(
            code="code-atomic",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="c",
            redirect_uri="http://localhost/callback",  # type: ignore[arg-type]
            redirect_uri_provided_explicitly=True,
        )
        oauth_storage.save_auth_code("code-atomic", auth_code)
        loaded = await provider.load_authorization_code(client, "code-atomic")
        assert loaded is not None
        initial = await provider.exchange_authorization_code(client, loaded)

        # Monkey-patch storage to raise mid-rotation
        original = oauth_storage.rotate_refresh_token

        def boom(**kwargs: object) -> None:
            raise sqlite3.OperationalError("simulated mid-rotation failure")

        oauth_storage.rotate_refresh_token = boom  # type: ignore[assignment]
        try:
            refresh = await provider.load_refresh_token(client, initial.refresh_token)  # type: ignore[arg-type]
            assert refresh is not None
            with pytest.raises(sqlite3.OperationalError):
                await provider.exchange_refresh_token(client, refresh, ["read"])
        finally:
            oauth_storage.rotate_refresh_token = original  # type: ignore[assignment]

        # Old access + refresh must still be valid
        assert oauth_storage.get_access_token(initial.access_token) is not None
        assert oauth_storage.get_refresh_token(initial.refresh_token) is not None  # type: ignore[arg-type]
