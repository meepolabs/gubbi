"""Integration tests for the full OAuth flow."""

from __future__ import annotations

import re
from pathlib import Path

import bcrypt
from fastapi import FastAPI
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from journalctl.config import get_settings
from journalctl.oauth.forms import create_login_handler
from journalctl.oauth.provider import JournalOAuthProvider
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.oauth.templates import CSRF_COOKIE_NAME

SERVER_URL = "http://localhost:8100"
TEST_PASSWORD = "test-password"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt()).decode()
TEST_CLIENT_ID = "test-client"
TEST_REDIRECT_URI = "http://localhost/callback"


def _register_test_client(storage: OAuthStorage) -> None:
    """Pre-register the test client so redirect_uri validation passes."""
    client = OAuthClientInformationFull(
        client_id=TEST_CLIENT_ID,
        client_secret="test-secret",  # noqa: S106
        redirect_uris=[TEST_REDIRECT_URI],  # type: ignore[arg-type]
        client_name="test-app",
    )
    storage.save_client(client)


def _create_test_app(oauth_storage: OAuthStorage) -> Starlette:
    """Create a minimal Starlette app with OAuth routes for testing."""
    provider = JournalOAuthProvider(
        storage=oauth_storage,
        server_url=SERVER_URL,
    )

    issuer_url = AnyHttpUrl(SERVER_URL)

    routes = create_auth_routes(
        provider=provider,
        issuer_url=issuer_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
    )

    pr_routes = create_protected_resource_routes(
        resource_url=AnyHttpUrl(f"{SERVER_URL}/mcp"),
        authorization_servers=[issuer_url],
    )

    login_handler = create_login_handler(
        storage=oauth_storage,
        password_hash=TEST_PASSWORD_HASH,
        secure_cookies=False,
    )

    all_routes = [
        *routes,
        *pr_routes,
        Route("/login", endpoint=login_handler, methods=["GET", "POST"]),
    ]

    return Starlette(routes=all_routes)


def _get_csrf_token(client: TestClient) -> str:
    """GET /login to extract CSRF token and set cookie on client."""
    resp = client.get(
        "/login",
        params={
            "client_id": "x",
            "redirect_uri": "http://localhost/cb",
            "state": "s",
            "code_challenge": "c",
        },
    )
    # Extract token from hidden field
    match = re.search(r'name="csrf_token" value="([^"]+)"', resp.text)
    assert match, "CSRF token not found in form"
    # Cookie is automatically set on the client instance
    return match.group(1)


class TestMetadataEndpoints:
    def test_authorization_server_metadata(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.get("/.well-known/oauth-authorization-server")
        assert response.status_code == 200
        data = response.json()
        assert data["issuer"] == SERVER_URL + "/"
        assert data["authorization_endpoint"] == f"{SERVER_URL}/authorize"
        assert data["token_endpoint"] == f"{SERVER_URL}/token"
        assert data["registration_endpoint"] == f"{SERVER_URL}/register"
        assert "S256" in data["code_challenge_methods_supported"]

    def test_protected_resource_metadata(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.get("/.well-known/oauth-protected-resource/mcp")
        assert response.status_code == 200
        data = response.json()
        assert data["resource"] == f"{SERVER_URL}/mcp"


class TestClientRegistration:
    def test_register_client(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.post(
            "/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": "test-app",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "client_id" in data
        assert "client_secret" in data


class TestLoginPage:
    def test_login_page_renders_with_csrf(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.get(
            "/login",
            params={
                "client_id": "test-client",
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
            },
        )
        assert response.status_code == 200
        assert "Authorize Access" in response.text
        assert "test-client" in response.text
        assert 'name="csrf_token"' in response.text
        assert CSRF_COOKIE_NAME in response.cookies

    def test_login_rejects_missing_csrf(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        # POST without CSRF token or cookie
        response = client.post(
            "/login",
            data={
                "client_id": "test-client",
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
                "scope": "",
                "password": TEST_PASSWORD,
            },
        )
        assert response.status_code == 403
        assert "CSRF" in response.text

    def test_login_rejects_wrong_csrf(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        # GET to set valid cookie on client
        _get_csrf_token(client)

        # POST with mismatched form token
        response = client.post(
            "/login",
            data={
                "csrf_token": "wrong-token",
                "client_id": "test-client",
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
                "scope": "",
                "password": TEST_PASSWORD,
            },
        )
        assert response.status_code == 403

    def test_login_wrong_password(self, oauth_storage: OAuthStorage) -> None:
        _register_test_client(oauth_storage)
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        csrf_token = _get_csrf_token(client)

        response = client.post(
            "/login",
            data={
                "csrf_token": csrf_token,
                "client_id": "test-client",
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
                "scope": "",
                "password": "wrong-password",
            },
        )
        assert response.status_code == 200
        assert "Invalid password" in response.text

    def test_login_correct_password_redirects(self, oauth_storage: OAuthStorage) -> None:
        _register_test_client(oauth_storage)
        app = _create_test_app(oauth_storage)
        client = TestClient(app, follow_redirects=False)

        csrf_token = _get_csrf_token(client)

        response = client.post(
            "/login",
            data={
                "csrf_token": csrf_token,
                "client_id": "test-client",
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
                "scope": "read",
                "password": TEST_PASSWORD,
            },
        )
        assert response.status_code == 302
        location = response.headers["location"]
        assert "http://localhost/callback" in location
        assert "code=" in location
        assert "state=test-state" in location


class TestCSPNonce:
    def test_login_page_has_csp_with_nonce(self, oauth_storage: OAuthStorage) -> None:
        """Fix #7: CSP should use nonce, not unsafe-inline."""
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.get(
            "/login",
            params={
                "client_id": "x",
                "redirect_uri": "http://localhost/cb",
                "state": "s",
                "code_challenge": "c",
            },
        )
        assert response.status_code == 200
        csp = response.headers.get("content-security-policy", "")
        assert "unsafe-inline" not in csp
        assert "nonce-" in csp
        # Verify the nonce in CSP matches the nonce in the style tag
        nonce_match = re.search(r"nonce-([A-Za-z0-9_-]+)", csp)
        assert nonce_match
        nonce = nonce_match.group(1)
        assert f'nonce="{nonce}"' in response.text


class TestXSSPrevention:
    def test_xss_in_client_id_escaped(self, oauth_storage: OAuthStorage) -> None:
        app = _create_test_app(oauth_storage)
        client = TestClient(app)

        response = client.get(
            "/login",
            params={
                "client_id": '<script>alert("xss")</script>',
                "redirect_uri": "http://localhost/callback",
                "state": "test-state",
                "code_challenge": "test-challenge",
            },
        )
        assert response.status_code == 200
        # Script tag must be escaped, not rendered raw
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text


class TestTokenLengthValidation:
    def test_oversized_token_rejected(self, oauth_storage: OAuthStorage) -> None:
        """Fix #9: tokens longer than 256 chars should be rejected."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from journalctl.middleware import BearerAuthMiddleware

        async def echo(request: object) -> JSONResponse:  # noqa: ARG001
            return JSONResponse({"ok": True})

        inner = Starlette(routes=[Route("/", echo)])
        app = BearerAuthMiddleware(inner, api_key="unused-for-oversized-check")
        client = TestClient(app)

        oversized = "x" * 300
        response = client.get("/", headers={"Authorization": f"Bearer {oversized}"})
        assert response.status_code == 401


class TestFullOAuthFlow:
    def test_register_authorize_token(self, oauth_storage: OAuthStorage) -> None:
        """Test the complete OAuth flow: register -> authorize -> login -> token."""
        app = _create_test_app(oauth_storage)
        client = TestClient(app, follow_redirects=False)

        # Step 1: Register client
        reg_response = client.post(
            "/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": "flow-test",
            },
        )
        assert reg_response.status_code == 201
        reg_data = reg_response.json()
        client_id = reg_data["client_id"]
        client_secret = reg_data["client_secret"]

        # Step 2: Authorize (this redirects to /login)
        auth_response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
                "code_challenge_method": "S256",
                "state": "flow-state",
            },
        )
        assert auth_response.status_code == 302
        assert "/login" in auth_response.headers["location"]

        # Step 2.5: GET the login page to get CSRF token (cookie set on client)
        login_page = client.get(
            "/login",
            params={
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "state": "flow-state",
                "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            },
        )
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
        assert csrf_match
        csrf_token = csrf_match.group(1)

        # Step 3: Login with password + CSRF
        login_response = client.post(
            "/login",
            data={
                "csrf_token": csrf_token,
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "state": "flow-state",
                "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
                "scope": "",
                "password": TEST_PASSWORD,
            },
        )
        assert login_response.status_code == 302
        location = login_response.headers["location"]
        assert "code=" in location

        # Extract the code from the redirect
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(location)
        code = parse_qs(parsed.query)["code"][0]

        # Step 4: Exchange code for token
        # PKCE verifier "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk" hashes to
        # challenge "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM" (RFC 7636)
        token_response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "client_secret": client_secret,
                "code_verifier": "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
            },
        )
        assert token_response.status_code == 200
        token_data = token_response.json()
        assert "access_token" in token_data
        assert "refresh_token" in token_data
        assert token_data["token_type"].lower() == "bearer"

        # Verify the access token is in storage
        at = oauth_storage.get_access_token(token_data["access_token"])
        assert at is not None
        assert at.client_id == client_id


class TestRegisterRateLimit:
    def test_register_rate_limited_per_ip(self, oauth_storage: OAuthStorage) -> None:
        """HIGH-4: /register returns 429 after REGISTER_MAX_ATTEMPTS in window."""
        from journalctl.oauth.constants import REGISTER_MAX_ATTEMPTS

        # Use the real register_oauth_routes so the wrap is applied
        settings = get_settings()
        app = FastAPI()
        register_oauth_routes(app, oauth_storage, settings)

        client = TestClient(app)
        for i in range(REGISTER_MAX_ATTEMPTS):
            resp = client.post(
                "/register",
                json={
                    "redirect_uris": [f"http://localhost/{i}"],
                    "client_name": f"c{i}",
                },
            )
            assert resp.status_code in (201, 200), f"attempt {i} got {resp.status_code}"

        resp = client.post(
            "/register",
            json={
                "redirect_uris": ["http://localhost/last"],
                "client_name": "last",
            },
        )
        assert resp.status_code == 429


class TestLoginRateLimitSharedAcrossInstances:
    def test_lockout_persists_when_storage_reopened(self, tmp_path: Path) -> None:
        """CRITICAL-2: simulate two workers by opening storage twice on same DB."""
        from journalctl.oauth.constants import LOGIN_MAX_FAILURES

        db_path = tmp_path / "oauth.db"

        storage_a = OAuthStorage(db_path)
        _ = storage_a.conn
        for _ in range(LOGIN_MAX_FAILURES):
            storage_a.record_rate_limit_event("login_failure:1.2.3.4")

        # "Worker B" opens the same DB
        storage_b = OAuthStorage(db_path)
        count_b = storage_b.count_rate_limit_events("login_failure:1.2.3.4", 600)
        assert count_b == LOGIN_MAX_FAILURES

        storage_a.close()
        storage_b.close()
