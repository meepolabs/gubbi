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
import secrets
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

# -- URL overrides via env vars (defaults point at bunsamosa dev stack) --
JOURNAL_DEV_AUTH_URL: str = os.environ.get("JOURNAL_DEV_AUTH_URL", "https://auth-dev.gubbi.ai")
JOURNAL_DEV_IDENTITY_URL: str = os.environ.get(
    "JOURNAL_DEV_IDENTITY_URL", "https://identity-dev.gubbi.ai"
)
JOURNAL_DEV_MCP_URL: str = os.environ.get("JOURNAL_DEV_MCP_URL", "https://mcp-dev.gubbi.ai")

# Optionally override the Hydra admin port (4445, loopback on bunsamosa).
# When set, drives login/consent acceptance and completes the full flow.
JOURNAL_DEV_AUTH_ADMIN_URL: str | None = os.environ.get("JOURNAL_DEV_AUTH_ADMIN_URL", None)

# -- Test constants --
EMAIL_PREFIX = "test-hydra-oauth-"  # TASK-04.16 cleanup API filters on this
# Randomized per module load: each live-stack test run registers a Kratos
# identity with a password no-one else knows. Leaking a static password into
# the public AGPL repo would let anyone authenticate as these identities on
# identity-dev.gubbi.ai until TASK-04.16 cleanup API catches up.
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
    reason="requires live dev stack (auth-dev.gubbi.ai + identity-dev.gubbi.ai + mcp-dev.gubbi.ai); enable with JOURNAL_LIVE_STACK=1",
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
                    # Public PKCE client -- no client_secret on token exchange.
                    # Matches how MCP clients (claude.ai, etc.) will actually
                    # register in production.
                    "token_endpoint_auth_method": "none",
                },
            )
            assert (
                reg_resp.status_code == 201
            ), f"DCR failed: {reg_resp.status_code} {reg_resp.text[:500]}"
            reg_data: dict[str, Any] = reg_resp.json()
            client_id = reg_data["client_id"]

            # ---- Step 2: Kratos identity creation (two-step flow: init + submit) ----
            test_id = uuid.uuid4().hex[:8]
            email = f"{EMAIL_PREFIX}{test_id}@gubbi.ai"

            flow_resp = await client.get(
                f"{JOURNAL_DEV_IDENTITY_URL}/self-service/registration/api",
                headers={"Accept": "application/json"},
            )
            assert flow_resp.status_code == 200, (
                f"Kratos flow init failed: {flow_resp.status_code} " f"{flow_resp.text[:500]}"
            )
            flow_action = flow_resp.json().get("ui", {}).get("action")
            assert flow_action, "Kratos flow response missing ui.action"

            ident_resp = await client.post(
                flow_action,
                headers={"Accept": "application/json"},
                json={
                    "method": "password",
                    "password": TEST_PASSWORD,
                    "traits": {"email": email, "timezone": "UTC"},
                },
            )
            assert ident_resp.status_code in (200, 201), (
                f"Kratos reg failed: {ident_resp.status_code} " f"{ident_resp.text[:500]}"
            )
            identity_body = ident_resp.json()
            # API-mode submit returns {session_token, session, identity, continue_with};
            # the identity id lives at body.identity.id (and session.identity.id).
            identity_uuid = str(identity_body.get("identity", {}).get("id") or "")
            assert identity_uuid, "Kratos registration response missing identity.id"

            # ---- Step 3: Hydra authorization request (PKCE) ----
            verifier, challenge = _make_pkce()
            # `state` must be >= 8 chars (Hydra enforces minimum entropy) --
            # without it Hydra 303s straight back to redirect_uri with
            # error=invalid_state instead of 303'ing to the login UI.
            state = secrets.token_urlsafe(24)

            auth_resp = await client.get(
                f"{JOURNAL_DEV_AUTH_URL}/oauth2/auth",
                params={
                    "response_type": "code",
                    "client_id": client_id,
                    "redirect_uri": TEST_REDIRECT_URI,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "scope": TEST_SCOPE,
                    "state": state,
                },
            )

            # Hydra responds 302 or 303 with Location pointing at the login UI
            # (302 on some versions, 303 See Other on v2.2+); 400 surfaces PKCE
            # or param-shape errors for debugging.
            auth_ok = auth_resp.status_code in (302, 303, 400)
            assert auth_ok, (
                f"Auth request failed unexpectedly: {auth_resp.status_code} "
                f"{auth_resp.text[:500]}"
            )
            auth_location = auth_resp.headers.get("location", "")
            # Fail loudly if Hydra redirected back to redirect_uri with an
            # error rather than forward to the login UI -- this surfaces
            # malformed auth requests (bad state, bad PKCE) instead of
            # masquerading as "admin port unreachable" further down.
            if "error=" in auth_location:
                raise AssertionError(
                    f"Hydra /oauth2/auth returned error in redirect: {auth_location[:500]}"
                )

            # ---- Steps 4-6: Login + Consent acceptance via Hydra admin port ----
            _skip_if_no_admin_url()
            assert JOURNAL_DEV_AUTH_ADMIN_URL is not None
            login_challenge = _query_param_from_url(auth_location, "login_challenge")
            assert (
                login_challenge
            ), f"Hydra auth redirect missing login_challenge: {auth_location[:500]}"

            # Admin-gated flow: once past _skip_if_no_admin_url the tunnel is
            # assumed live; let HTTP errors surface as real test failures so
            # future regressions (e.g. Hydra API path changes) are caught loudly
            # instead of silently skipped.
            #
            # Hydra v2 flow:
            #  login/accept -> redirect_to /oauth2/auth?login_verifier=...
            #  follow that ->   redirect_to consent UI (with consent_challenge)
            #                   OR directly to callback (if consent was skipped)
            #  consent/accept -> redirect_to /oauth2/auth?consent_verifier=...
            #  follow that ->   redirect_to callback with ?code=...
            login_accept_redirect = await _accept_login(
                client,
                JOURNAL_DEV_AUTH_ADMIN_URL,
                login_challenge,
                identity_uuid,
            )
            auth_code = await _drive_to_callback(
                client,
                JOURNAL_DEV_AUTH_ADMIN_URL,
                login_accept_redirect,
            )

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
                    # MCP streamable-HTTP requires clients advertise support
                    # for both JSON and SSE on every request.
                    "Accept": "application/json, text/event-stream",
                },
            )

            assert mcp_resp.status_code == 200, (
                f"MCP authenticated call failed: {mcp_resp.status_code} " f"{mcp_resp.text[:500]}"
            )
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
    """POST /admin/oauth2/auth/requests/login/accept on the Hydra admin port.

    Hydra v2 moved admin routes under `/admin`; the legacy unprefixed paths
    return 307 redirects that httpx (follow_redirects=False) will not follow
    for POST, so we target the final path directly.
    """
    login_resp = await client.get(
        f"{admin_url}/admin/oauth2/auth/requests/login",
        params={"login_challenge": login_challenge},
    )
    login_resp.raise_for_status()
    login_data: dict[str, Any] = login_resp.json()
    login_subject = subject or (_extract_sub(login_data) or "")

    accept_resp = await client.put(
        f"{admin_url}/admin/oauth2/auth/requests/login/accept",
        params={"login_challenge": login_challenge},
        json={"subject": login_subject},
    )
    accept_resp.raise_for_status()
    redirect_to = str(accept_resp.json().get("redirect_to", ""))
    if not redirect_to:
        raise AssertionError("login/accept response missing redirect_to")
    return redirect_to


async def _accept_consent(
    client: httpx.AsyncClient,
    admin_url: str,
    consent_challenge: str,
) -> str:
    """PUT /admin/oauth2/auth/requests/consent/accept on the Hydra admin port."""
    consent_resp = await client.get(
        f"{admin_url}/admin/oauth2/auth/requests/consent",
        params={"consent_challenge": consent_challenge},
    )
    consent_resp.raise_for_status()
    data: dict[str, Any] = consent_resp.json()

    audience = data.get("client", {}).get("client_id")
    token_audience: list[str] = [str(audience)] if audience else []

    # `session.access_token` and `session.id_token` must be map[string]interface{}
    # (extra claims to embed in the respective token), NOT booleans. Empty
    # dicts mean "grant with no extra claims" -- Hydra still issues both
    # tokens because they come from grant_scope/openid/audience, not from
    # this field. Sending `true` here yields a 400 unmarshal error.
    response: dict[str, Any] = {
        "grant_scope": TEST_SCOPE.split(),
        "grant_access_token_audience": token_audience,
        "session": {"access_token": {}, "id_token": {}},
    }

    accept_resp = await client.put(
        f"{admin_url}/admin/oauth2/auth/requests/consent/accept",
        params={"consent_challenge": consent_challenge},
        json=response,
    )
    if accept_resp.status_code >= 400:
        raise AssertionError(
            f"consent/accept failed {accept_resp.status_code}: " f"{accept_resp.text[:500]}"
        )
    redirect_to = str(accept_resp.json().get("redirect_to", ""))
    if not redirect_to:
        raise AssertionError("consent/accept response missing redirect_to")
    return redirect_to


async def _drive_to_callback(
    client: httpx.AsyncClient,
    admin_url: str,
    login_accept_redirect: str,
) -> str:
    """Follow the Hydra verifier dance until we land on the callback auth code.

    Hydra v2 splits the post-admin-accept chain: the redirect we get from
    login/accept (or consent/accept) points back at /oauth2/auth with a
    *_verifier query param; following that redirect advances the state
    machine to either the consent UI or the final callback. We cap hops to
    prevent infinite loops on misconfiguration.
    """
    location = login_accept_redirect
    for _ in range(10):
        # Terminal state: callback with authorization code.
        code = _query_param_from_url(location, "code")
        if code:
            return code

        # Intermediate state: Hydra wants consent. Accept it via admin API.
        consent_challenge = _query_param_from_url(location, "consent_challenge")
        if consent_challenge:
            location = await _accept_consent(client, admin_url, consent_challenge)
            continue

        # Otherwise this is a verifier URL pointing back at /oauth2/auth;
        # follow it with GET (same hostname, so tests against remote Hydra
        # will issue a real HTTPS request to the public issuer).
        resp = await client.get(location)
        if resp.status_code not in (302, 303):
            raise AssertionError(
                f"Expected redirect while driving OAuth flow, got "
                f"{resp.status_code} at {location[:200]}"
            )
        next_location = resp.headers.get("location", "")
        if not next_location:
            raise AssertionError(f"Missing Location header on redirect at {location[:200]}")
        location = next_location

    raise AssertionError("Exceeded 10 hops driving OAuth flow to callback")


def _query_param_from_url(url: str, param: str) -> str | None:
    """Extract a query-parameter value from URL, if present."""
    values = parse_qs(urlparse(url).query).get(param)
    return values[0] if values else None


def _extract_sub(data: dict[str, Any]) -> str | None:
    """Best-effort subject extraction from a Hydra login challenge body."""
    sub = data.get("subject") or data.get("login_subject")
    return str(sub) if sub else None
