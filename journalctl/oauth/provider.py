"""OAuth 2.0 Authorization Server Provider for the journal MCP server.

Implements the MCP SDK's OAuthAuthorizationServerProvider protocol.
Single-user: only the journal owner (verified by bcrypt password) can authorize.
Dual-mode: accepts both legacy API keys and OAuth access tokens.
"""

from __future__ import annotations

import logging
import secrets
import time

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from journalctl.config import Settings
from journalctl.oauth.storage import OAuthStorage

logger = logging.getLogger("journalctl.oauth.provider")


class JournalOAuthProvider(  # type: ignore[type-arg]
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken],
):
    """Single-user OAuth provider for the journal MCP server."""

    def __init__(
        self,
        storage: OAuthStorage,
        server_url: str,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.server_url = server_url.rstrip("/")
        self.settings = settings

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.storage.get_client(client_id)

    async def register_client(
        self,
        client_info: OAuthClientInformationFull,
    ) -> None:
        self.storage.save_client(client_info)

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Redirect to our login page with all OAuth params forwarded."""
        if not client.client_id:
            raise AuthorizeError(
                error="invalid_request",
                error_description="Missing client_id",
            )

        redirect: str = construct_redirect_uri(
            f"{self.server_url}/login",
            client_id=client.client_id or "",
            redirect_uri=str(params.redirect_uri),
            state=params.state,
            code_challenge=params.code_challenge,
            scope=" ".join(params.scopes) if params.scopes else None,
        )
        return redirect

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        code = self.storage.get_auth_code(authorization_code)
        if code is None:
            return None
        # Verify it belongs to this client (timing-safe)
        if not secrets.compare_digest(code.client_id, client.client_id or ""):
            self.storage.delete_auth_code(authorization_code)
            logger.warning("Auth code client_id mismatch for client %s", client.client_id)
            return None
        # Check expiry
        if code.expires_at < time.time():
            self.storage.delete_auth_code(authorization_code)
            return None
        return code

    def _issue_token_pair(
        self,
        client_id: str,
        scopes: list[str],
        resource: str | None = None,
    ) -> OAuthToken:
        """Generate, store, and link an access+refresh token pair."""
        now = int(time.time())

        access_token_str = secrets.token_urlsafe(32)
        access_token = AccessToken(
            token=access_token_str,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + self.settings.oauth_access_token_ttl,
            resource=resource,
        )
        refresh_token_str = secrets.token_urlsafe(32)
        refresh_token = RefreshToken(
            token=refresh_token_str,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + self.settings.oauth_refresh_token_ttl,
        )
        self.storage.save_issued_token_pair(
            access_token_str, access_token, refresh_token_str, refresh_token
        )

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",  # noqa: S106 — protocol constant
            expires_in=self.settings.oauth_access_token_ttl,
            refresh_token=refresh_token_str,
            scope=" ".join(scopes) if scopes else None,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        """Exchange auth code for access + refresh tokens."""
        self.storage.delete_auth_code(authorization_code.code)
        return self._issue_token_pair(
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        token = self.storage.get_refresh_token(refresh_token)
        if token is None:
            return None
        if not secrets.compare_digest(token.client_id, client.client_id or ""):
            return None
        # Check expiry
        if token.expires_at is not None and token.expires_at < int(time.time()):
            self.storage.delete_refresh_token(refresh_token)
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Rotate both access and refresh tokens."""
        # Revoke old access tokens before issuing new pair
        for at in self.storage.get_paired_access_tokens(refresh_token.token):
            self.storage.delete_access_token(at)
        self.storage.delete_token_pair_by_refresh(refresh_token.token)
        self.storage.delete_refresh_token(refresh_token.token)

        logger.info("Refresh token rotated for client_id=%s", client.client_id)
        effective_scopes = scopes if scopes else refresh_token.scopes
        return self._issue_token_pair(
            client_id=client.client_id or "",
            scopes=effective_scopes,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Load and validate an OAuth access token."""
        access_token = self.storage.get_access_token(token)
        if access_token is None:
            return None
        # Check expiry
        if access_token.expires_at is not None and access_token.expires_at < int(time.time()):
            self.storage.delete_access_token(token)
            return None
        return access_token

    async def revoke_token(
        self,
        token: AccessToken | RefreshToken,
    ) -> None:
        """Revoke a token and its paired counterpart."""
        if isinstance(token, AccessToken):
            paired_refresh = self.storage.get_paired_refresh_token(token.token)
            self.storage.delete_access_token(token.token)
            self.storage.delete_token_pair_by_access(token.token)
            if paired_refresh:
                self.storage.delete_refresh_token(paired_refresh)
            logger.info("Access token revoked for client_id=%s", token.client_id)
        elif isinstance(token, RefreshToken):
            paired_access = self.storage.get_paired_access_tokens(token.token)
            self.storage.delete_refresh_token(token.token)
            self.storage.delete_token_pair_by_refresh(token.token)
            for at in paired_access:
                self.storage.delete_access_token(at)
            logger.info("Refresh token revoked for client_id=%s", token.client_id)
