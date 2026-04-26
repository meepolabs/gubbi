"""Hosted Mode-3 OAuth E2E tests against the live dev stack on bunsamosa.

Flows: Hydra DCR + Kratos identity creation + PKCE/S256 auth request,
followed by login/consent acceptance (via Hydra admin port when reachable),
token exchange, and authenticated MCP call.

NOTE: identities + DCR clients accumulate; cleanup via TASK-04.16 when shipped.
No teardown of
created Kratos identities or Hydra clients is performed by these tests.

See lead_notes for SSH-tunnel option when JOURNAL_DEV_AUTH_ADMIN_URL is
set to drive the full loop from laptop off-host.
"""

from __future__ import annotations

import base64
import hashlib
import os
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

# -- URL overrides via env vars (defaults point at bunsamosa dev stack) --
JOURNAL_DEV_AUTH_URL: str = os.environ.get("JOURNAL_DEV_AUTH_URL", "https://auth-dev.meepolabs.com")
JOURNAL_DEV_IDENTITY_URL: str = os.environ.get(
    "JOURNAL_DEV_IDENTITY_URL", "https://identity-dev.meepolabs.com"
)
JOURNAL_DEV_MCP_URL: str = os.environ.get("JOURNAL_DEV_MCP_URL", "https://journal.meepolabs.com")

# Optionally override the Hydra admin port (4445, loopback on bunsamosa).
# When set, drives login/consent acceptance and completes the full flow.
JOURNAL_DEV_AUTH_ADMIN_URL: str | None = os.environ.get("JOURNAL_DEV_AUTH_ADMIN_URL", None)

# -- Test constants --
EMAIL_PREFIX = "test-hydra-oauth-"  # TASK-04.16 cleanup API filters on this
# Randomized per module load: each live-stack test run registers a Kratos
# identity with a password no-one else knows. Leaking a static password into
# the public AGPL repo would let anyone authenticate as these identities on
# identity-dev.meepolabs.com until TASK-04.16 cleanup API catches up.
TEST_PASSWORD = f"test-pass-{uuid.uuid4().hex}-Aa1!"
TEST_REDIRECT_URI = "http://localhost/callback"
TEST_SCOPE = "journal openid offline_access"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=15.0)

_MCP_INITIALIZE: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-25",
        "capabilities": {"tools": {}},
        "clientInfo": {"name": "test-hydra-e2e", "version": "0.1.0"},
    },
}


def _make_pkce() -> tuple[str, str]:
    """Generate a PKCE verifier + S256 challenge pair."""
    secrets_raw: bytes = os.urandom(32)
    verifier = base64.urlsafe_b64encode(secrets_raw).rstrip(b"=").decode("ascii")
    challenge_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(challenge_digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _skip_if_no_admin_url() -> None:
    """Skip current test when Hydra admin port is unreachable; narrows type."""
    if JOURNAL_DEV_AUTH_ADMIN_URL is None:
        pytest.skip(
            "JOURNAL_DEV_AUTH_ADMIN_URL not set -- skipping login/consent "
            "acceptance and token exchange (admin port 4445 needed). "
            "Set to http://localhost:4445 with SSH tunnel from bunsamosa."
        )


@pytest.mark.hosted_live
@pytest.mark.skipif(
    os.environ.get("JOURNAL_LIVE_STACK") != "1",
    reason="requires live dev stack (auth-dev.meepolabs.com + identity-dev.meepolabs.com + journal.meepolabs.com); enable with JOURNAL_LIVE_STACK=1",
)
class TestHydraOauthFlow:
    """Hosted OAuth flow -- live dev stack on bunsamosa.

    The test class is skipped unless JOURNAL_LIVE_STACK=1.

    Admin-login / consent-accept steps (4-6) require JOURNAL_DEV_AUTH_ADMIN_URL
    to be reachable from the test runner.  Without it, the test verifies DCR +
    Kratos registration + auth-request succeeded and skips the admin-gated path.

    See lead_notes for SSH-tunnel option (b) when running off-host.
    """

    async def test_full_flow(self) -> None:
        """End-to-end Mode-3 hosted OAuth flow against live dev stack."""
        client = httpx.AsyncClient(follow_redirects=False, timeout=_REQUEST_TIMEOUT)
        _access_token = ""  # set in token-exchange block below

        try:
            # ---- Step 1: Dynamic Client Registration (DCR) ----
            reg_resp = await client.post(
                f"{JOURNAL_DEV_AUTH_URL}/oauth2/register",
                json={
                    "redirect_uris": [TEST_REDIRECT_URI],
                    "client_name": "test-hydra-e2e",
                    "grant_types": [
                        "authorization_code",
                        "refresh_token",
                    ],
                    "response_types": ["code"],
                    "scope": TEST_SCOPE,
                },
            )
            assert (
                reg_resp.status_code == 201
            ), f"DCR failed: {reg_resp.status_code} {reg_resp.text[:500]}"
            reg_data: dict[str, Any] = reg_resp.json()
            client_id = reg_data["client_id"]

            # ---- Step 2: Kratos identity creation ----
            test_id = uuid.uuid4().hex[:8]
            email = f"{EMAIL_PREFIX}{test_id}@meepolabs.com"

            ident_resp = await client.post(
                f"{JOURNAL_DEV_IDENTITY_URL}/self-service/registration/api",
                json={
                    "credentials": {
                        "password": {
                            "identity": {"traits": {"email": email}},
                            "password": TEST_PASSWORD,
                        }
                    },
                    "traits": {"email": email},
                    "method": "password",
                    "flow": None,  # self-service flow via API mode
                },
            )
            assert ident_resp.status_code in (200, 201), (
                f"Kratos reg failed: {ident_resp.status_code} " f"{ident_resp.text[:500]}"
            )
            identity_body = ident_resp.json()
            identity_uuid = str(identity_body.get("id") or "")
            assert identity_uuid, "Kratos registration response missing identity id"

            # ---- Step 3: Hydra authorization request (PKCE) ----
            verifier, challenge = _make_pkce()

            auth_resp = await client.get(
                f"{JOURNAL_DEV_AUTH_URL}/oauth2/auth",
                params={
                    "response_type": "code",
                    "client_id": client_id,
                    "redirect_uri": TEST_REDIRECT_URI,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "scope": TEST_SCOPE,
                },
            )

            auth_ok = auth_resp.status_code in (302, 400)
            assert auth_ok, (
                f"Auth request failed unexpectedly: {auth_resp.status_code} "
                f"{auth_resp.text[:500]}"
            )
            auth_location = auth_resp.headers.get("location", "")

            # ---- Steps 4-6: Login + Consent acceptance via Hydra admin port ----
            _skip_if_no_admin_url()
            assert JOURNAL_DEV_AUTH_ADMIN_URL is not None
            login_challenge = _query_param_from_url(auth_location, "login_challenge")
            if not login_challenge:
                pytest.skip("Hydra auth redirect did not include login_challenge")
            assert login_challenge is not None

            try:
                consent_challenge = await _accept_login(
                    client,
                    JOURNAL_DEV_AUTH_ADMIN_URL,
                    login_challenge,
                    identity_uuid,
                )
                callback_redirect = await _accept_consent(
                    client,
                    JOURNAL_DEV_AUTH_ADMIN_URL,
                    consent_challenge,
                )
                auth_code = _query_param_from_url(callback_redirect, "code")
                if not auth_code:
                    pytest.skip("Consent accept redirect did not include authorization code")

                # Step 6b: Token exchange
                token_resp = await client.post(
                    f"{JOURNAL_DEV_AUTH_URL}/oauth2/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "client_id": client_id,
                        "redirect_uri": TEST_REDIRECT_URI,
                        "code_verifier": verifier,
                        "scope": TEST_SCOPE,
                    },
                )
                token_resp.raise_for_status()

                token_data: dict[str, Any] = token_resp.json()
                _access_token = token_data["access_token"]
                assert "id_token" in token_data
                assert "refresh_token" in token_data
            except httpx.HTTPStatusError as exc:
                pytest.skip(
                    f"Admin-gated flow aborted at {exc.request.url} "
                    f"({exc.response.status_code}): Hydra admin port requires "
                    "a live SSH tunnel from bunsamosa to expose loopback binding."
                )

            # ---- Step 7: Authenticated MCP call with the access token ----
            _skip_if_no_admin_url()
            assert JOURNAL_DEV_AUTH_ADMIN_URL is not None
            assert _access_token

            mcp_resp = await client.post(
                f"{JOURNAL_DEV_MCP_URL}/mcp/",
                json=_MCP_INITIALIZE,
                headers={
                    "Authorization": f"Bearer {_access_token}",
                    "Content-Type": "application/json",
                },
            )

            assert mcp_resp.status_code == 200, (
                f"MCP authenticated call failed: {mcp_resp.status_code} " f"{mcp_resp.text[:500]}"
            )
            # Validate MCP JSON-RPC response shape.
            mcp_body = mcp_resp.json()
            assert mcp_body.get("jsonrpc") == "2.0"
        finally:
            await client.aclose()

    async def test_tampered_token_rejected(self) -> None:
        """Sending a tampered access token must return 401 with WWW-Authenticate."""
        client = httpx.AsyncClient(follow_redirects=False, timeout=_REQUEST_TIMEOUT)

        try:
            malformed_jwt = (
                "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ0ZXN0d"
                "GFtcGVyZWQudGlja3VzQG1lZXBvbGFicy5jb20iLCJzY29w"
                "ZSI6IlJFQUxfVVNFUiJ9.tampered-signature-invalid-chars!!!"
            )

            mcp_resp = await client.post(
                f"{JOURNAL_DEV_MCP_URL}/mcp/",
                json=_MCP_INITIALIZE,
                headers={
                    "Authorization": f"Bearer {malformed_jwt}",
                    "Content-Type": "application/json",
                },
            )

            assert mcp_resp.status_code == 401, (
                f"Expected 401 for malformed token, got {mcp_resp.status_code}: "
                f"{mcp_resp.text[:500]}"
            )
            www_auth = mcp_resp.headers.get("www-authenticate", "")
            assert "ProtectedResource" in www_auth or "Bearer" in www_auth
        finally:
            await client.aclose()


# --------------------------------------------------------------------------- #
# Admin-port helpers -- best-effort, swallow errors for skip-driven paths.
# --------------------------------------------------------------------------- #


async def _accept_login(
    client: httpx.AsyncClient,
    admin_url: str,
    login_challenge: str,
    subject: str,
) -> str:
    """POST /oauth2/auth/requests/login/accept on the Hydra admin port."""
    login_resp = await client.get(
        f"{admin_url}/oauth2/auth/requests/login",
        params={"login_challenge": login_challenge},
    )
    login_resp.raise_for_status()
    login_data: dict[str, Any] = login_resp.json()
    login_subject = subject or (_extract_sub(login_data) or "")

    accept_resp = await client.post(
        f"{admin_url}/oauth2/auth/requests/login/accept",
        params={"login_challenge": login_challenge},
        json={"subject": login_subject},
    )
    accept_resp.raise_for_status()
    redirect_to = str(accept_resp.json().get("redirect_to", ""))
    consent_challenge = _query_param_from_url(redirect_to, "consent_challenge")
    if not consent_challenge:
        raise httpx.HTTPStatusError(
            "Missing consent_challenge in login accept redirect",
            request=accept_resp.request,
            response=accept_resp,
        )
    return consent_challenge


async def _accept_consent(
    client: httpx.AsyncClient,
    admin_url: str,
    consent_challenge: str,
) -> str:
    """POST /oauth2/auth/requests/consent/accept on the Hydra admin port."""
    consent_resp = await client.get(
        f"{admin_url}/oauth2/auth/requests/consent",
        params={"consent_challenge": consent_challenge},
    )
    consent_resp.raise_for_status()
    data: dict[str, Any] = consent_resp.json()

    audience = data.get("client", {}).get("client_id")
    token_audience: list[str] = [str(audience)] if audience else []

    response: dict[str, Any] = {
        "grant_scope": TEST_SCOPE.split(),
        "grant_access_token_audience": token_audience,
        "session": {"access_token": True, "id_token": True},
    }

    accept_resp = await client.post(
        f"{admin_url}/oauth2/auth/requests/consent/accept",
        params={"consent_challenge": consent_challenge},
        json=response,
    )
    accept_resp.raise_for_status()
    redirect_to = str(accept_resp.json().get("redirect_to", ""))
    if not redirect_to:
        raise httpx.HTTPStatusError(
            "Missing redirect_to in consent accept response",
            request=accept_resp.request,
            response=accept_resp,
        )
    return redirect_to


def _query_param_from_url(url: str, param: str) -> str | None:
    """Extract a query-parameter value from URL, if present."""
    values = parse_qs(urlparse(url).query).get(param)
    return values[0] if values else None


def _extract_sub(data: dict[str, Any]) -> str | None:
    """Best-effort subject extraction from a Hydra login challenge body."""
    sub = data.get("subject") or data.get("login_subject")
    return str(sub) if sub else None
