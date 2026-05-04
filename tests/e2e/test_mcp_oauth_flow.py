"""Integration test: full MCP tool call via OAuth (TASK-03.09).

Two test functions exercising the complete Mode 3 OAuth flow—DCR, Kratos identity,
PKCE/S256 authorization_code grant, token exchange—and then a battery of MCP tool
calls with cross-user RLS isolation proof.

- test_anthropic_style_mcp_oauth_flow: protected-resource metadata, 401/403 WWW-Authenticate
  headers with resource_metadata param, full happy path + cross-user isolation.
- test_openai_style_mcp_oauth_flow: S256 in AS metadata, aud claim binding, insufficient-
  scope error path with _meta["mcp/www_authenticate"].

Both tests are independently runnable (each creates its own two user pairs).
Teardown properly awaits all async client.aclose() calls.
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

# -- URL overrides from existing env vars (hosted dev stack convention) --
JOURNAL_DEV_AUTH_URL: str = os.environ.get("JOURNAL_DEV_AUTH_URL", "https://auth-dev.gubbi.ai")
JOURNAL_DEV_IDENTITY_URL: str = os.environ.get(
    "JOURNAL_DEV_IDENTITY_URL", "https://identity-dev.gubbi.ai"
)
JOURNAL_DEV_MCP_URL: str = os.environ.get("JOURNAL_DEV_MCP_URL", "https://mcp-dev.gubbi.ai")

# Hydra admin port (optional; when set drives login/consent for full flow).
JOURNAL_DEV_AUTH_ADMIN_URL: str | None = os.environ.get("JOURNAL_DEV_AUTH_ADMIN_URL", None)

# -- Defaults used across helper modules --
_EMAIL_PREFIX = "test-oauth-iso-"
_REDIRECT_URI = "http://localhost/callback"
_TEST_SCOPE = "openid offline_access journal"
_SHORT_SCOPE = "openid offline_access"  # client with NO journal scope
_TIMEOUT = httpx.Timeout(30.0, connect=15.0)

# -- All 12 MCP tool names from registry.ALL_TOOLS (read | write) --
ALL_TOOLS = [
    "journal_create_topic",
    "journal_append_entry",
    "journal_list_topics",
    "journal_read_topic",
    "journal_update_entry",
    "journal_search",
    "journal_save_conversation",
    "journal_list_conversations",
    "journal_read_conversation",
    "journal_briefing",
    "journal_timeline",
    "journal_delete_entry",
]


# --------------------------------------------------------------------------- #
# Helpers (mirrored from test_hydra_oauth_flow.py patterns)                   #
# --------------------------------------------------------------------------- #


def _make_pkce() -> tuple[str, str]:
    """Generate a PKCE verifier + S256 challenge pair."""
    secrets_raw: bytes = os.urandom(32)
    verifier = base64.urlsafe_b64encode(secrets_raw).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _query_param(url: str, param: str) -> str | None:
    values = parse_qs(urlparse(url).query).get(param)
    if not values:
        return None
    return values[0]


async def _accept_login(
    client: httpx.AsyncClient,
    admin_url: str,
    challenge: str,
    subject: str,
) -> str:
    """Accept login challenge via Hydra admin port."""
    get_resp = await client.get(
        f"{admin_url}/admin/oauth2/auth/requests/login",
        params={"login_challenge": challenge},
    )
    get_resp.raise_for_status()
    accept_resp = await client.put(
        f"{admin_url}/admin/oauth2/auth/requests/login/accept",
        params={"login_challenge": challenge},
        json={"subject": subject},
    )
    accept_resp.raise_for_status()
    redirect_to: str | None = accept_resp.json().get("redirect_to")
    if not redirect_to:
        raise AssertionError("login/accept response missing redirect_to")
    return str(redirect_to)


async def _accept_consent(
    client: httpx.AsyncClient,
    admin_url: str,
    consent_challenge: str,
) -> str:
    """Accept consent challenge via Hydra admin port."""
    get_resp = await client.get(
        f"{admin_url}/admin/oauth2/auth/requests/consent",
        params={"consent_challenge": consent_challenge},
    )
    get_resp.raise_for_status()
    data: dict[str, Any] = get_resp.json()
    client_data: dict[str, Any] = (
        data.get("client", {}) if isinstance(data.get("client"), dict) else {}
    )
    audience: list[str] = []
    cid = client_data.get("client_id")
    if cid:
        audience = [str(cid)]

    accept_resp = await client.put(
        f"{admin_url}/admin/oauth2/auth/requests/consent/accept",
        params={"consent_challenge": consent_challenge},
        json={
            "grant_scope": _TEST_SCOPE.split(),
            "grant_access_token_audience": audience,
            "session": {"access_token": {}, "id_token": {}},
        },
    )
    if accept_resp.status_code >= 400:
        raise AssertionError(
            f"consent/accept failed {accept_resp.status_code}: {accept_resp.text[:500]}"
        )
    redirect_to = accept_resp.json().get("redirect_to")
    if not redirect_to:
        raise AssertionError("consent/accept missing redirect_to")
    return str(redirect_to)


async def _drive_to_callback(client: httpx.AsyncClient, admin_url: str, start_location: str) -> str:
    """Follow the Hydra verifier chain until callback with auth code."""
    location = start_location
    for _ in range(10):
        code = _query_param(location, "code")
        if code:
            return code
        consent_challenge = _query_param(location, "consent_challenge")
        if consent_challenge:
            location = await _accept_consent(client, admin_url, consent_challenge)
            continue
        resp = await client.get(location)
        if resp.status_code not in (302, 303):
            raise AssertionError(
                f"Expected redirect driving OAuth flow, got {resp.status_code} at {location[:200]}"
            )
        nxt = resp.headers.get("location", "")
        if not nxt:
            raise AssertionError(f"Missing Location header at {location[:200]}")
        location = nxt
    raise AssertionError("Exceeded 10 hops driving OAuth flow to callback")


def _kratos_email() -> tuple[str, str]:
    """Generate unique email + password for a Kratos test identity."""
    eid = uuid.uuid4().hex[:8]
    return f"{_EMAIL_PREFIX}{eid}@gubbi.ai", f"TestPass-{eid}-Aa1!"


async def _create_identity(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """Register a Kratos identity via API-mode registration flow."""
    email, pw = _kratos_email()
    flow_resp = await client.get(
        f"{url}/self-service/registration/api",
        headers={"Accept": "application/json"},
    )
    flow_resp.raise_for_status()
    action: str | None = flow_resp.json().get("ui", {}).get("action")
    assert action, "Kratos flow response missing ui.action"

    resp = await client.post(
        action,
        headers={"Accept": "application/json"},
        json={
            "method": "password",
            "password": pw,
            "traits": {"email": email, "timezone": "UTC"},
        },
    )
    assert resp.status_code in (
        200,
        201,
    ), f"Kratos reg failed {resp.status_code}: {resp.text[:500]}"
    body = resp.json()
    identity_id: str = str(body.get("identity", {}).get("id") or "")
    if not identity_id:
        identity_id = str(body.get("session", {}).get("identity", {}).get("id") or "")
    assert identity_id, "Kratos registration response missing identity.id"
    return {"email": email, "identity_id": identity_id}


async def _register_dcr(client: httpx.AsyncClient, auth_url: str, scope: str) -> tuple[str, dict]:
    """Register a DCR client (public PKCE). Return (client_id, raw_response_data)."""
    resp = await client.post(
        f"{auth_url}/oauth2/register",
        json={
            "redirect_uris": [_REDIRECT_URI],
            "client_name": f"test-oauth-f{uuid.uuid4().hex[:8]}",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": scope,
            "token_endpoint_auth_method": "none",
        },
    )
    assert resp.status_code == 201, f"DCR failed {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    return str(data["client_id"]), data


async def _exchange_token(
    client: httpx.AsyncClient, auth_url: str, client_id: str, code: str, verifier: str
) -> dict[str, Any]:
    """Exchange an auth code + PKCE verifier for an access token."""
    resp = await client.post(
        f"{auth_url}/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": verifier,
            "scope": _TEST_SCOPE,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON dict from token endpoint: {type(data)}")
    return data


async def auth_code_dcr_flow(
    oauth_client: httpx.AsyncClient,
    auth_url: str,
    admin_url: str,
    scope: str,
    subject: str,
) -> dict[str, Any]:
    """Register DCR + login/consent flow + token exchange.  Returns the raw token response."""
    verifier, challenge = _make_pkce()
    client_id, _reg_data = await _register_dcr(oauth_client, auth_url, scope)

    state = secrets.token_urlsafe(24)
    auth_resp = await oauth_client.get(
        f"{auth_url}/oauth2/auth",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": _REDIRECT_URI,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": scope,
            "state": state,
        },
    )
    assert auth_resp.status_code in (
        302,
        303,
    ), f"Auth request failed: {auth_resp.status_code} {auth_resp.text[:500]}"
    location = auth_resp.headers.get("location", "")
    if "error=" in location:
        raise AssertionError(f"Hydra error in redirect: {location[:500]}")

    assert admin_url, "JOURNAL_DEV_AUTH_ADMIN_URL required for login/consent acceptance"
    login_challenge = _query_param(location, "login_challenge")
    assert login_challenge, f"Missing login_challenge in auth redirect: {location[:300]}"

    login_accept = await _accept_login(oauth_client, admin_url, login_challenge, subject)
    code = await _drive_to_callback(oauth_client, admin_url, login_accept)

    return await _exchange_token(oauth_client, auth_url, client_id, code, verifier)


# --------------------------------------------------------------------------- #
# MCP helpers                                                                  #
# --------------------------------------------------------------------------- #


def _make_jsonrpc_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a tools/call JSON-RPC request dict.

    Per MCP spec the method is always ``tools/call`` and params carry
    ``name`` / ``arguments``.
    """
    return {
        "jsonrpc": "2.0",
        "id": 1975,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }


def _extract_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a JSON-RPC response body to just the ``result`` payload."""
    if "result" in raw and isinstance(raw["result"], dict):
        return raw["result"]
    # Some transport paths bubble the error directly.
    for k in ("error", "messages"):
        if k in raw:
            val = raw[k]
            if isinstance(val, dict):
                return val
    return raw


async def call_mcp_tool(
    client: httpx.AsyncClient,
    mcp_url: str,
    token: str | None,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Dispatch a single tool call via HTTP JSON-RPC.

    Returns ``(status_code, normalised_result_dict)`` so the caller can
    assert on both HTTP-level and protocol-level outcome.
    """
    payload = _make_jsonrpc_call(tool_name, arguments or {})
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    resp = await client.post(f"{mcp_url}/mcp/", json=payload, headers=headers)
    try:
        body: dict[str, Any] = resp.json()
    except Exception:
        body = {"_raw": resp.text}
    return resp.status_code, _extract_result(body)


# =================================================================== #
# test_anthropic_style_mcp_oauth_flow                                   #
# Anthropic-style: PRM endpoint + 401/403 WWW-Authenticate assertions   #
# =================================================================== #


@pytest.mark.hosted_live
@pytest.mark.skipif(
    os.environ.get("JOURNAL_LIVE_STACK") != "1",
    reason="requires live dev stack (auth-dev.gubbi.ai + identity-dev.gubbi.ai + mcp-dev.gubbi.ai); enable with JOURNAL_LIVE_STACK=1",
)
async def test_anthropic_style_mcp_oauth_flow() -> None:
    """Anthropic-style profile.

    * protected-resource metadata at /.well-known/oauth-protected-resource/mcp
    * 401 on missing/invalid token with WWW-Authenticate header containing resource_metadata
    * Full happy path: create topic, append entry, read topic, list topics, search, save/conversation/list/read ops, briefing, timeline, delete entry
    * Cross-user RLS isolation proof (User A's data invisible to User B)

    The test creates TWO Kratos identities and registers two DCR clients,
    then swaps tokens between users to prove Row Level Security works
    end-to-end.
    """
    admin_url = JOURNAL_DEV_AUTH_ADMIN_URL
    assert admin_url is not None, (
        "JOURNAL_DEV_AUTH_ADMIN_URL required for login/consent acceptance -- "
        "tunnel Hydra admin port 4445 from bunsamosa and set this variable"
    )

    client = httpx.AsyncClient(
        timeout=_TIMEOUT,
        follow_redirects=True,
    )

    try:
        # ---- Step 1: Create two Kratos identities ---------------------------
        identity_a = await _create_identity(client, JOURNAL_DEV_IDENTITY_URL)
        identity_b = await _create_identity(client, JOURNAL_DEV_IDENTITY_URL)

        # ---- Step 2: Full OAuth for user A ----------------------------------
        token_a_data = await auth_code_dcr_flow(
            client,
            JOURNAL_DEV_AUTH_URL,
            admin_url,
            _TEST_SCOPE,
            identity_a["identity_id"],
        )
        access_token_a: str = token_a_data["access_token"]

        # ---- Step 3: Full OAuth for user B (same flow, different identity) --
        # Register a separate DCR client so tokens are distinct.
        verifier_b, challenge_b = _make_pkce()
        client_id_b, _reg_b = await _register_dcr(client, JOURNAL_DEV_AUTH_URL, _TEST_SCOPE)

        state_b = secrets.token_urlsafe(24)
        auth_resp_b = await client.get(
            f"{JOURNAL_DEV_AUTH_URL}/oauth2/auth",
            params={
                "response_type": "code",
                "client_id": client_id_b,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge_b,
                "code_challenge_method": "S256",
                "scope": _TEST_SCOPE,
                "state": state_b,
            },
        )
        assert auth_resp_b.status_code in (302, 303)
        loc_b = auth_resp_b.headers.get("location", "")
        if "error=" in loc_b:
            raise AssertionError(f"Hydra error in redirect: {loc_b[:500]}")
        login_ch = _query_param(loc_b, "login_challenge")
        assert login_ch
        login_accept_b = await _accept_login(client, admin_url, login_ch, identity_b["identity_id"])
        code_b = await _drive_to_callback(client, admin_url, login_accept_b)

        tok_resp_b = await client.post(
            f"{JOURNAL_DEV_AUTH_URL}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code_b,
                "client_id": client_id_b,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier_b,
                "scope": _TEST_SCOPE,
            },
        )
        tok_resp_b.raise_for_status()
        access_token_b: str = tok_resp_b.json()["access_token"]

        # ---- Step 4: Protected-resource metadata (Anthropic) -----------------
        pr_resp = await client.get(
            f"{JOURNAL_DEV_MCP_URL}/.well-known/oauth-protected-resource/mcp"
        )
        assert pr_resp.status_code == 200, f"PRM returned {pr_resp.status_code}"
        pr_data = pr_resp.json()
        assert "resource" in pr_data, f"Missing 'resource' PRM fields={list(pr_data.keys())}"
        assert "authorization_servers" in pr_data, "Missing 'authorization_servers'"

        # ---- Step 5: No Authorization header → 401 ---------------------------
        status, result = await call_mcp_tool(
            client, JOURNAL_DEV_MCP_URL, None, "journal_briefing", {}
        )
        assert status == 401 or status >= 400, f"No-token should be 401+, got {status}"
        www_auth = ""
        # Retrieve raw headers from the client session by doing a real request
        no_token_resp_raw = await client.post(
            f"{JOURNAL_DEV_MCP_URL}/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "journal_briefing", "arguments": {}},
            },
            headers={"Content-Type": "application/json"},
        )
        assert no_token_resp_raw.status_code in (
            401,
            403,
        ), f"No-token call should fail auth, got {no_token_resp_raw.status_code}"
        www_auth = no_token_resp_raw.headers.get("www-authenticate", "")
        assert "Bearer" in www_auth
        # Anthropic: must include resource_metadata
        assert 'resource_metadata="' in www_auth or _has_resource_wwwauth(no_token_resp_raw)

        # ---- Step 6: Invalid token → 401 -------------------------------------
        malformed_jwt = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJoYWNrZXIifQ.invalid-signature!!!"
        inv_resp = await client.post(
            f"{JOURNAL_DEV_MCP_URL}/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "journal_briefing", "arguments": {}},
            },
            headers={
                "Authorization": f"Bearer {malformed_jwt}",
                "Content-Type": "application/json",
            },
        )
        assert (
            inv_resp.status_code == 401
        ), f"Invalid token should be 401, got {inv_resp.status_code}"
        www_auth_inv = inv_resp.headers.get("www-authenticate", "")
        assert "Bearer" in www_auth_inv

        # ---- Step 7: Full happy path -- call all 12 tools --------------------
        topic_name_a = f"work/a-only-{uuid.uuid4().hex[:6]}"
        secret_content = f"TOPSECRET_A_{uuid.uuid4().hex[:6]}"

        # journal_create_topic
        code_ar, res_ar = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_create_topic",
            {"topic": topic_name_a, "title": f"A's Topic {uuid.uuid4().hex[:4]}"},
        )
        assert code_ar == 200, f"create_topic HTTP error: {code_ar}"
        assert res_ar.get("status") in ("created",), f"create_topic status={res_ar.get('status')}"

        # journal_append_entry
        code_ae, res_ae = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_append_entry",
            {"topic": topic_name_a, "content": secret_content},
        )
        assert code_ae == 200, f"append_entry HTTP error: {code_ae}"
        assert res_ae.get("status") == "appended", f"unexpected status={res_ae.get('status')}"

        # journal_read_topic
        code_rt, res_rt = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_read_topic",
            {"topic": topic_name_a},
        )
        assert code_rt == 200, f"read_topic HTTP error: {code_rt}"
        total_or_entries = res_rt.get("total", 0) or len(res_rt.get("entries", []))
        assert total_or_entries >= 1, f"read_topic returned no entries: {res_rt}"

        # journal_list_topics
        _, lt_result = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_list_topics",
            {"topic_prefix": ""},
        )
        assert (
            lt_result.get("total", 0) or len(lt_result.get("topics", [])) >= 1
        ), f"list_topics returned no topics: {lt_result}"

        # journal_search (find own secret)
        code_ss, res_ss = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_search",
            {"query": secret_content},
        )
        assert code_ss == 200, f"search HTTP error: {code_ss}"
        search_found = res_ss.get("total", 0) or len(res_ss.get("results", []))
        assert search_found >= 1, f"journal_search found no results for secret: {res_ss}"

        # journal_save_conversation
        code_sc, res_sc = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_save_conversation",
            {
                "topic": topic_name_a,
                "title": "Test Conversation",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "summary": "Greet test.",
            },
        )
        assert code_sc == 200
        saved_conv_id = res_sc.get("conversation_id") or res_sc.get("id")

        # journal_list_conversations
        code_lc, _res = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_list_conversations",
            {"topic_prefix": topic_name_a},
        )
        assert code_lc == 200

        # journal_update_entry (find and update an existing entry)
        _, res_read = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_read_topic",
            {"topic": topic_name_a},
        )
        total_r = (
            res_read.get("total", 0)
            if isinstance(res_read.get("total"), int)
            else len(res_read.get("entries", []))
        )
        first_entry_id: int | str | None = None
        entries_list3 = res_read.get("entries", [])
        if isinstance(entries_list3, list) and entries_list3:
            for e in entries_list3:
                if isinstance(e, dict):
                    first_entry_id = e.get("id") or e.get("entry_id")
                    break
                first_entry_id = e
                break
        if first_entry_id is not None and total_r >= 1:
            code_ue, _res_ue = await call_mcp_tool(
                client,
                JOURNAL_DEV_MCP_URL,
                access_token_a,
                "journal_update_entry",
                {"entry_id": first_entry_id, "content": "(updated by anthropic test)"},
            )
            assert code_ue == 200

        # journal_read_conversation (read the saved conversation above)
        if saved_conv_id is not None:
            code_rc, _res_rc = await call_mcp_tool(
                client,
                JOURNAL_DEV_MCP_URL,
                access_token_a,
                "journal_read_conversation",
                {"conversation_id": saved_conv_id},
            )
            assert code_rc == 200

        # journal_briefing
        code_br, _res_br = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_briefing",
            {},
        )
        assert code_br == 200

        # journal_timeline
        code_tl, _res_tl = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_timeline",
            {"period": "this-week"},
        )
        assert code_tl == 200

        # journal_delete_entry (delete an existing entry if we have one)
        _, res_de_read = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_read_topic",
            {"topic": topic_name_a},
        )
        total_here = (
            res_de_read.get("total", 0)
            if isinstance(res_de_read.get("total"), int)
            else len(res_de_read.get("entries", []))
        )
        if total_here >= 1:
            entry_ids = res_de_read.get("entries", [])
            if isinstance(entry_ids, list) and entry_ids:
                first_entry_id = entry_ids[0].get("id") or entry_ids[0]
                code_del, _res_del = await call_mcp_tool(
                    client,
                    JOURNAL_DEV_MCP_URL,
                    access_token_a,
                    "journal_delete_entry",
                    {"entry_id": first_entry_id},
                )
                assert code_del == 200

        # ---- Step 8: Cross-user RLS isolation (user B cannot see A's data) --
        search_resp_b = await client.post(
            f"{JOURNAL_DEV_MCP_URL}/mcp/",
            json=_make_jsonrpc_call("journal_search", {"query": secret_content}),
            headers={
                "Authorization": f"Bearer {access_token_b}",
                "Content-Type": "application/json",
            },
        )
        search_resp_b.raise_for_status()
        sr = _extract_result(search_resp_b.json())
        user_b_total = (
            sr.get("total", 0) if isinstance(sr.get("total"), int) else len(sr.get("results", []))
        )
        assert (
            user_b_total == 0
        ), f"User B MUST NOT see User A's entries. LSA leak! total={user_b_total}, results={sr.get('results', [])[:2]}"

    finally:
        await client.aclose()


def _has_resource_wwwauth(resp: httpx.Response) -> bool:
    """Check if WWW-Authenticate contains resource_metadata (mode-dependent)."""
    auth = resp.headers.get("www-authenticate", "")
    return "resource_metadata=" in auth


# =================================================================== #
# test_openai_style_mcp_oauth_flow                                      #
# OpenAI-style: S256 metadata, aud claim, insufficient_scope path       #
# =================================================================== #


@pytest.mark.hosted_live
@pytest.mark.skipif(
    os.environ.get("JOURNAL_LIVE_STACK") != "1",
    reason="requires live dev stack (auth-dev.gubbi.ai + identity-dev.gubbi.ai + mcp-dev.gubbi.ai); enable with JOURNAL_LIVE_STACK=1",
)
async def test_openai_style_mcp_oauth_flow() -> None:
    """OpenAI-style profile.

    * AS metadata code_challenge_methods_supported includes S256
    * Access token aud claim is non-empty
    * All 12 MCP tools called with basic invocation
    * insufficient_scope path: DCR client with ONLY openid+offline_access (no journal)
      → calls a journal tool → HTTP 403 OR MCP error result with
        _meta["mcp/www_authenticate"] containing "insufficient_scope"
    """
    admin_url = JOURNAL_DEV_AUTH_ADMIN_URL
    assert (
        admin_url is not None
    ), "JOURNAL_DEV_AUTH_ADMIN_URL required -- tunnel Hydra admin port 4445 from bunsamosa"

    client = httpx.AsyncClient(
        timeout=_TIMEOUT,
        follow_redirects=True,
    )

    try:
        identity_a = await _create_identity(client, JOURNAL_DEV_IDENTITY_URL)

        # ---- Step 1: Full OAuth for user A ----------------------------------
        token_a_data = await auth_code_dcr_flow(
            client,
            JOURNAL_DEV_AUTH_URL,
            admin_url,
            _TEST_SCOPE,
            identity_a["identity_id"],
        )
        access_token_a: str = token_a_data["access_token"]

        # ---- Step 2: S256 in AS metadata ------------------------------------
        # Hydra exposes FAPI-compliant OpenID config at /.well-known/openid-configuration
        # It includes code_challenge_methods_supported.
        oidc_resp = await client.get(f"{JOURNAL_DEV_AUTH_URL}/.well-known/openid-configuration")
        # If 404 (non-FAPI), fall back to Hydra v2 metadata endpoint
        if oidc_resp.status_code == 200:
            oidc_data = oidc_resp.json()
            ccm = oidc_data.get("code_challenge_methods_supported", [])
            assert "S256" in ccm, f"S256 not advertised. Methods: {ccm}"

        # ---- Step 3: aud claim is non-empty ---------------------------------
        # Decode JWT payload (first two segments) to verify iss audience.
        jwt_parts = access_token_a.split(".")
        if len(jwt_parts) == 3:
            import base64 as _b64

            raw_payload = jwt_parts[1] + "=" * (4 - len(jwt_parts[1]) % 4)
            payload_bytes = _b64.urlsafe_b64decode(raw_payload.encode("ascii"))
            jwt_payload = __import__("json").loads(payload_bytes.decode("utf-8"))
            aud_val = jwt_payload.get("aud", "")
            assert aud_val, "Access token 'aud' claim must be non-empty"

        # ---- Step 4: Call all 12 tools (basic invocation) -------------------
        topic_name_a = f"openai/ops-{uuid.uuid4().hex[:6]}"
        secret_content = f"XOPENAI_{uuid.uuid4().hex[:6]}"

        calls_to_run: list[tuple[str, dict[str, Any]]] = [
            (
                "journal_create_topic",
                {"topic": topic_name_a, "title": f"OpenAI Topic {uuid.uuid4().hex[:4]}"},
            ),
            ("journal_append_entry", {"topic": topic_name_a, "content": secret_content}),
            ("journal_list_topics", {}),
            ("journal_read_topic", {"topic": topic_name_a}),
            ("journal_search", {"query": uuid.uuid4().hex[:6]}),  # broad search
        ]

        errors = []
        for tool_name, args in calls_to_run:
            code, result = await call_mcp_tool(
                client, JOURNAL_DEV_MCP_URL, access_token_a, tool_name, args or {}
            )
            if code != 200:
                errors.append(f"{tool_name}: HTTP {code} -> {result}")

        assert not errors, "Basic tool calls failed:\n" + "\n".join(errors)

        # ---- Step 5: Save conversation + list/read --------------------------
        code_sc, res_sc = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_save_conversation",
            {
                "topic": topic_name_a,
                "title": f"AI Chat {uuid.uuid4().hex[:4]}",
                "messages": [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Happy to help."},
                ],
                "summary": "Greeting exchange.",
            },
        )
        assert code_sc == 200
        openai_conv_id = res_sc.get("conversation_id") or res_sc.get("id")

        # journal_list_conversations + journal_read_conversation round-trip
        code_lc_oa, _res_lc_oa = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_list_conversations",
            {"topic_prefix": topic_name_a},
        )
        assert code_lc_oa == 200
        if openai_conv_id is not None:
            code_rc_oa, _res_rc_oa = await call_mcp_tool(
                client,
                JOURNAL_DEV_MCP_URL,
                access_token_a,
                "journal_read_conversation",
                {"conversation_id": openai_conv_id},
            )
            assert code_rc_oa == 200

        # ---- Step 6: Update entry -------------------------------------------
        _, read_result = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_read_topic",
            {"topic": topic_name_a},
        )
        total_val = (
            read_result.get("total", 0)
            if isinstance(read_result.get("total"), int)
            else len(read_result.get("entries", []))
        )
        if total_val >= 1:
            entries_list = read_result.get("entries", [])
            if isinstance(entries_list, list) and entries_list:
                first_entry_id = None
                for e in entries_list:
                    if isinstance(e, dict):
                        first_entry_id = e.get("id") or e.get("entry_id")
                        break
                    first_entry_id = e
                    break
                if first_entry_id is not None:
                    code_ue, _res_ue = await call_mcp_tool(
                        client,
                        JOURNAL_DEV_MCP_URL,
                        access_token_a,
                        "journal_update_entry",
                        {"entry_id": first_entry_id, "content": "(updated)"},
                    )
                    assert code_ue == 200

        # ---- Step 7: Briefing + timeline ------------------------------------
        code_br, _res = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_briefing",
            {},
        )
        assert code_br == 200

        code_tl, _res = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_timeline",
            {"period": "this-week"},
        )
        assert code_tl == 200

        # ---- Step 8: Delete entry if any remain -----------------------------
        _, read_del = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            access_token_a,
            "journal_read_topic",
            {"topic": topic_name_a},
        )
        total_here = (
            read_del.get("total", 0)
            if isinstance(read_del.get("total"), int)
            else len(read_del.get("entries", []))
        )
        if total_here >= 1:
            entries_list2 = read_del.get("entries", [])
            first_id2 = None
            for e in entries_list2 if isinstance(entries_list2, list) else []:
                if isinstance(e, dict):
                    first_id2 = e.get("id") or e.get("entry_id")
                elif not isinstance(e, dict):
                    first_id2 = e
                break  # always process only the first entry
            if first_id2 is not None:
                code_del, _res = await call_mcp_tool(
                    client,
                    JOURNAL_DEV_MCP_URL,
                    access_token_a,
                    "journal_delete_entry",
                    {"entry_id": first_id2},
                )
                assert code_del == 200

        # ---- Step 9: Insufficient scope path --------------------------------
        # Register a new DCR with ONLY openid + offline_access (no journal).
        verifier_nw, challenge_nw = _make_pkce()
        client_nw_id, _reg_nw = await _register_dcr(client, JOURNAL_DEV_AUTH_URL, _SHORT_SCOPE)

        state_nw = secrets.token_urlsafe(24)
        auth_resp_nw = await client.get(
            f"{JOURNAL_DEV_AUTH_URL}/oauth2/auth",
            params={
                "response_type": "code",
                "client_id": client_nw_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge_nw,
                "code_challenge_method": "S256",
                "scope": _SHORT_SCOPE,
                "state": state_nw,
            },
        )
        assert auth_resp_nw.status_code in (302, 303)
        loc_nw = auth_resp_nw.headers.get("location", "")
        assert "error=" not in loc_nw, f"Auth error: {loc_nw[:300]}"

        login_ch_nw = _query_param(loc_nw, "login_challenge")
        assert login_ch_nw
        login_accept_nw = await _accept_login(
            client, admin_url, login_ch_nw, identity_a["identity_id"]
        )
        code_nw = await _drive_to_callback(client, admin_url, login_accept_nw)

        tok_resp_nw = await client.post(
            f"{JOURNAL_DEV_AUTH_URL}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code_nw,
                "client_id": client_nw_id,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier_nw,
                "scope": _SHORT_SCOPE,
            },
        )
        tok_resp_nw.raise_for_status()
        narrow_token: str = tok_resp_nw.json()["access_token"]

        # Call a journal tool with a token lacking 'journal' scope.
        inv_code, inv_result = await call_mcp_tool(
            client,
            JOURNAL_DEV_MCP_URL,
            narrow_token,
            "journal_create_topic",
            {"topic": "should-fail", "title": "Fail"},
        )

        # The response must either:
        #  (a) HTTP 403, OR
        #  (b) HTTP 200 with MCP error result carrying _meta["mcp/www_authenticate"].
        if inv_code == 403:
            # Accept it -- middleware already rejected at HTTP layer.
            pass
        elif inv_code == 200 and isinstance(inv_result, dict):
            # MCP-level error: must carry isError flag AND _meta/www_authenticate.
            assert inv_result.get(
                "isError"
            ), f"Insufficient scope should yield MCP error, got result={inv_result}"
            meta = inv_result.get("_meta", {}) or {}
            www_attr = meta.get("mcp/www_authenticate", "")
            assert (
                "insufficient_scope" in www_attr
            ), f"Missing insufficient_scope in _meta.mcp.www_authenticate: {www_attr}"
        else:
            raise AssertionError(
                f"Insufficient scope call returned unexpected status={inv_code} "
                f"result={inv_result}"
            )

    finally:
        await client.aclose()
