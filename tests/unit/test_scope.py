"""Tests for scope checking infrastructure (TASK-03.06).

Covers:
- check_scope() with the data-driven SCOPE_GRANTS mapping
- @require_scope decorator on tool handlers
- insufficient_scope_response builder
- BearerAuthMiddleware stores scopes in context var
- OriginValidationMiddleware allowlist enforcement (incl. DNS-rebinding guard)
- READ_TOOLS / WRITE_TOOLS categorization
- filter_tools_by_scope hook (default-deny)
- Per-tool annotation matrix verification (AST-parsed against spec table)
- Consent UI renders scope descriptions (XSS-escaped)
"""

from __future__ import annotations

import ast
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ListToolsRequest, ListToolsResult, ServerResult

from journalctl.auth.hydra import HydraIntrospector, TokenClaims
from journalctl.core.auth_context import current_token_scopes
from journalctl.core.scope import (
    SCOPE_DESCRIPTIONS,
    SCOPE_GRANTS,
    check_scope,
    insufficient_scope_response,
    require_scope,
)
from journalctl.middleware.auth import BearerAuthMiddleware
from journalctl.middleware.origin import OriginValidationMiddleware
from journalctl.tools.registry import (
    ALL_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    _wire_scope_filter,
    filter_tools_by_scope,
)

TEST_API_KEY = "a" * 64
TEST_TOKEN = "ory_at_" + "x" * 80
TEST_SUB = UUID("550e8400-e29b-41d4-a716-446655440000")
TEST_OP_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


# ---------------------------------------------------------------------------
# check_scope
# ---------------------------------------------------------------------------


class TestCheckScope:
    def test_journal_grants_read(self) -> None:
        assert check_scope({"journal"}, "journal:read") is True

    def test_journal_grants_write(self) -> None:
        assert check_scope({"journal"}, "journal:write") is True

    def test_journal_read_only_does_not_grant_write(self) -> None:
        assert check_scope({"journal:read"}, "journal:write") is False

    def test_journal_write_only_does_not_grant_read(self) -> None:
        assert check_scope({"journal:write"}, "journal:read") is False

    def test_journal_read_grants_read(self) -> None:
        assert check_scope({"journal:read"}, "journal:read") is True

    def test_journal_write_grants_write(self) -> None:
        assert check_scope({"journal:write"}, "journal:write") is True

    def test_unrelated_scope_rejected(self) -> None:
        assert check_scope({"openid", "email"}, "journal:read") is False

    def test_empty_scope_rejected(self) -> None:
        assert check_scope(set(), "journal:read") is False

    def test_mixed_scopes(self) -> None:
        # journal + openid -> journal grants read+write
        assert check_scope({"journal", "openid"}, "journal:read") is True
        assert check_scope({"journal", "openid"}, "journal:write") is True

    def test_substring_scope_rejected(self) -> None:
        # "journaling" should NOT satisfy "journal:read"
        assert check_scope({"journaling"}, "journal:read") is False

    def test_frozenset_input(self) -> None:
        assert check_scope(frozenset({"journal"}), "journal:read") is True


# ---------------------------------------------------------------------------
# SCOPE_GRANTS mapping
# ---------------------------------------------------------------------------


class TestScopeGrantsMapping:
    def test_journal_maps_to_read_and_write(self) -> None:
        assert "journal:read" in SCOPE_GRANTS["journal"]
        assert "journal:write" in SCOPE_GRANTS["journal"]

    def test_journal_read_self_grant(self) -> None:
        assert SCOPE_GRANTS["journal:read"] == frozenset({"journal:read"})

    def test_journal_write_self_grant(self) -> None:
        assert SCOPE_GRANTS["journal:write"] == frozenset({"journal:write"})


# ---------------------------------------------------------------------------
# SCOPE_DESCRIPTIONS
# ---------------------------------------------------------------------------


class TestScopeDescriptions:
    def test_journal_description(self) -> None:
        assert "journal" in SCOPE_DESCRIPTIONS
        assert "read" in SCOPE_DESCRIPTIONS["journal"].lower()
        assert "write" in SCOPE_DESCRIPTIONS["journal"].lower()
        assert "search" in SCOPE_DESCRIPTIONS["journal"].lower()

    def test_all_required_scopes_have_descriptions(self) -> None:
        for scope in ("journal", "openid", "email", "offline_access"):
            assert scope in SCOPE_DESCRIPTIONS, f"Missing description for {scope}"


# ---------------------------------------------------------------------------
# insufficient_scope_response
# ---------------------------------------------------------------------------


class TestInsufficientScopeResponse:
    def test_error_shape(self) -> None:
        result = insufficient_scope_response("journal:read")
        assert result.isError is True
        assert result.content is not None
        assert len(result.content) >= 1

    def test_meta_contains_www_authenticate(self) -> None:
        result = insufficient_scope_response("journal:write")
        assert result.meta is not None
        assert "mcp/www_authenticate" in result.meta
        assert "insufficient_scope" in result.meta["mcp/www_authenticate"]

    def test_custom_detail(self) -> None:
        result = insufficient_scope_response("journal:read", "custom detail")
        assert result.meta is not None
        assert "custom detail" in result.meta["mcp/www_authenticate"]


# ---------------------------------------------------------------------------
# @require_scope decorator
# ---------------------------------------------------------------------------


class TestRequireScopeDecorator:
    async def test_decorator_passes_with_valid_scope(self) -> None:
        token_scopes_reset = current_token_scopes.set(frozenset({"journal"}))
        try:

            @require_scope("journal:read")
            async def my_tool() -> dict:
                return {"ok": True}

            result = await my_tool()
            assert result == {"ok": True}
        finally:
            current_token_scopes.reset(token_scopes_reset)

    async def test_decorator_returns_error_on_missing_scope(self) -> None:
        token_scopes_reset = current_token_scopes.set(frozenset({"openid"}))
        try:

            @require_scope("journal:read")
            async def my_tool() -> dict:
                return {"ok": True}

            result = await my_tool()
            assert result.isError is True
            assert result.meta is not None
            assert "mcp/www_authenticate" in result.meta
        finally:
            current_token_scopes.reset(token_scopes_reset)

    async def test_decorator_with_no_token_scopes(self) -> None:
        token_scopes_reset = current_token_scopes.set(None)
        try:

            @require_scope("journal:read")
            async def my_tool() -> dict:
                return {"ok": True}

            result = await my_tool()
            assert result.isError is True
        finally:
            current_token_scopes.reset(token_scopes_reset)

    async def test_decorator_journal_grants_write(self) -> None:
        """v1: 'journal' scope grants both journal:read and journal:write."""
        token_scopes_reset = current_token_scopes.set(frozenset({"journal"}))
        try:

            @require_scope("journal:write")
            async def my_tool() -> dict:
                return {"status": "created"}

            result = await my_tool()
            assert result == {"status": "created"}
        finally:
            current_token_scopes.reset(token_scopes_reset)

    async def test_decorator_preserves_function_name(self) -> None:
        @require_scope("journal:read")
        async def journal_search() -> dict:
            return {}

        assert journal_search.__name__ == "journal_search"


# ---------------------------------------------------------------------------
# BearerAuthMiddleware stores scopes in context var
# ---------------------------------------------------------------------------


def _asgi_app(
    *,
    response_status: int = 200,
    response_body: bytes = b"ok",
) -> Callable[..., Awaitable[None]]:
    async def _app(scope: dict, receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": response_status,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": response_body})

    return _app


class TestMiddlewareScopeStorage:
    async def test_hydra_token_scopes_stored_in_context_var(self) -> None:
        """After successful Hydra introspection, token_scopes should be set."""
        captured_scopes: list[frozenset[str] | None] = []

        async def capture_app(scope, receive, send):
            captured_scopes.append(current_token_scopes.get())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        claims = TokenClaims(sub=TEST_SUB, scope="openid journal email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)

        mw = BearerAuthMiddleware(
            capture_app,
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})

        assert len(captured_scopes) == 1
        assert captured_scopes[0] is not None
        assert "journal" in captured_scopes[0]

    async def test_context_var_reset_after_hydra_request(self) -> None:
        """ContextVar should be reset to None after the request."""
        claims = TokenClaims(sub=TEST_SUB, scope="journal", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)

        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})

        assert current_token_scopes.get() is None

    async def test_api_key_path_sets_journal_scope(self) -> None:
        """Static API key auth should set 'journal' scope (full access)."""
        captured_scopes: list[frozenset[str] | None] = []

        async def capture_app(scope, receive, send):
            captured_scopes.append(current_token_scopes.get())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = BearerAuthMiddleware(
            capture_app,
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})

        assert len(captured_scopes) == 1
        assert captured_scopes[0] == frozenset({"journal:read", "journal:write"})


# ---------------------------------------------------------------------------
# OriginValidationMiddleware
# ---------------------------------------------------------------------------


class TestOriginValidation:
    async def test_allowed_origin_passes(self) -> None:
        allowed = frozenset({"https://claude.ai", "https://chatgpt.com"})
        mw = OriginValidationMiddleware(_asgi_app(), allowed)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "https://claude.ai"})
        assert resp.status_code == 200

    async def test_disallowed_origin_returns_403(self) -> None:
        allowed = frozenset({"https://claude.ai"})
        mw = OriginValidationMiddleware(_asgi_app(), allowed)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "https://evil.com"})
        assert resp.status_code == 403

    async def test_no_origin_header_passes(self) -> None:
        """Clients without Origin header (curl, SDK) pass through."""
        mw = OriginValidationMiddleware(_asgi_app(), frozenset({"https://claude.ai"}))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 200

    async def test_loopback_origin_allowed(self) -> None:
        """Loopback origins are always allowed for dev."""
        mw = OriginValidationMiddleware(_asgi_app(), frozenset())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200

    async def test_loopback_127_allowed(self) -> None:
        mw = OriginValidationMiddleware(_asgi_app(), frozenset())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://127.0.0.1:8100"})
        assert resp.status_code == 200

    async def test_journal_meepolabs_allowed(self) -> None:
        allowed = frozenset({"https://journal.meepolabs.com"})
        mw = OriginValidationMiddleware(_asgi_app(), allowed)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "https://journal.meepolabs.com"})
        assert resp.status_code == 200

    async def test_non_http_scope_bypass(self) -> None:
        """WebSocket and other non-HTTP scopes bypass origin check."""
        got_scope: list[dict] = []

        async def capture(scope, receive, send):
            got_scope.append(scope)

        mw = OriginValidationMiddleware(capture, frozenset())
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "headers": [(b"origin", b"https://evil.com")],
        }
        await mw(scope, MagicMock(), MagicMock())
        assert len(got_scope) == 1

    async def test_dns_rebinding_localhost_evil_rejected(self) -> None:
        """Regression: http://localhost.evil.com must NOT match http://localhost loopback prefix."""
        mw = OriginValidationMiddleware(_asgi_app(), frozenset({"https://claude.ai"}))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://localhost.evil.com"})
        assert resp.status_code == 403

    async def test_dns_rebinding_127_0_0_1_evil_rejected(self) -> None:
        mw = OriginValidationMiddleware(_asgi_app(), frozenset({"https://claude.ai"}))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://127.0.0.1.evil.com"})
        assert resp.status_code == 403

    async def test_loopback_exact_match_allowed(self) -> None:
        """Bare loopback origin (no port) is allowed."""
        mw = OriginValidationMiddleware(_asgi_app(), frozenset())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://localhost"})
        assert resp.status_code == 200

    async def test_loopback_ipv6_allowed(self) -> None:
        """IPv6 loopback origin allowed (with port)."""
        mw = OriginValidationMiddleware(_asgi_app(), frozenset())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Origin": "http://[::1]:8100"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# READ_TOOLS / WRITE_TOOLS categorization
# ---------------------------------------------------------------------------


class TestToolCategorization:
    def test_read_tools_count(self) -> None:
        assert len(READ_TOOLS) == 7

    def test_write_tools_count(self) -> None:
        assert len(WRITE_TOOLS) == 5

    def test_all_tools_covered(self) -> None:
        assert len(ALL_TOOLS) == 12
        assert ALL_TOOLS == READ_TOOLS | WRITE_TOOLS

    def test_no_overlap(self) -> None:
        assert set() == READ_TOOLS & WRITE_TOOLS

    def test_read_tools_membership(self) -> None:
        expected = {
            "journal_search",
            "journal_list_topics",
            "journal_list_conversations",
            "journal_read_topic",
            "journal_read_conversation",
            "journal_briefing",
            "journal_timeline",
        }
        assert expected == READ_TOOLS

    def test_write_tools_membership(self) -> None:
        expected = {
            "journal_append_entry",
            "journal_update_entry",
            "journal_create_topic",
            "journal_save_conversation",
            "journal_delete_entry",
        }
        assert expected == WRITE_TOOLS


class TestFilterToolsByScope:
    def test_journal_scope_returns_all_tools(self) -> None:
        all_names = list(ALL_TOOLS)
        result = filter_tools_by_scope(all_names, {"journal"})
        assert set(result) == ALL_TOOLS

    def test_no_scope_returns_empty_list(self) -> None:
        """Default-deny: middleware rejects no-scope tokens at the HTTP
        layer; this filter must not fail open if that ever changes."""
        all_names = list(ALL_TOOLS)
        result = filter_tools_by_scope(all_names, set())
        assert result == []

    def test_unknown_scope_returns_empty_list(self) -> None:
        """Default-deny: unknown scopes get no tools."""
        all_names = list(ALL_TOOLS)
        result = filter_tools_by_scope(all_names, {"some-unknown-scope"})
        assert result == []

    def test_journal_plus_unknown_returns_all(self) -> None:
        """Mixed: journal grants access regardless of other scopes."""
        all_names = list(ALL_TOOLS)
        result = filter_tools_by_scope(all_names, {"journal", "some-unknown"})
        assert set(result) == ALL_TOOLS


# ---------------------------------------------------------------------------
# Tool annotation matrix verification
# ---------------------------------------------------------------------------


def _extract_tool_annotations() -> dict[str, dict[str, bool]]:
    """Parse tool source files via AST and extract per-tool annotation kwargs.

    Walks each ``journalctl/tools/*.py`` (excluding registry / constants /
    __init__) and pulls every ``async def journal_*`` function decorated
    with ``@mcp.tool(annotations=ToolAnnotations(...))``.  Returns a map
    from tool name (function name) to its annotation kwargs (constants
    only -- non-constant kwargs are skipped).

    Static parsing keeps the test deterministic: no FastMCP runtime
    setup, no AppContext stub, no closure traversal.
    """
    out: dict[str, dict[str, bool]] = {}
    tools_dir = Path(__file__).resolve().parents[2] / "journalctl" / "tools"
    skip = {"__init__.py", "registry.py", "constants.py"}
    for path in sorted(tools_dir.glob("*.py")):
        if path.name in skip:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("journal_"):
                continue
            annotations: dict[str, bool] = {}
            for dec in node.decorator_list:
                if not (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "tool"
                ):
                    continue
                for kw in dec.keywords:
                    if kw.arg != "annotations" or not isinstance(kw.value, ast.Call):
                        continue
                    for ann_kw in kw.value.keywords:
                        if ann_kw.arg and isinstance(ann_kw.value, ast.Constant):
                            annotations[ann_kw.arg] = ann_kw.value.value
            if annotations:
                out[node.name] = annotations
    return out


# Locked spec table from milestone-03-mcp-hosted.md TASK-03.06 (lines 268-282).
# Columns omitted from the spec ("--") are intentionally absent from the
# expectation; the test only enforces what the spec calls out explicitly.
_EXPECTED_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "journal_search": {"readOnlyHint": True},
    "journal_list_topics": {"readOnlyHint": True},
    "journal_list_conversations": {"readOnlyHint": True},
    "journal_read_topic": {"readOnlyHint": True},
    "journal_read_conversation": {"readOnlyHint": True},
    "journal_briefing": {"readOnlyHint": True},
    "journal_timeline": {"readOnlyHint": True},
    "journal_append_entry": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": False,
    },
    "journal_update_entry": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    "journal_create_topic": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": False,
    },
    "journal_save_conversation": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    "journal_delete_entry": {
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
}


class TestToolAnnotations:
    """Verify the locked annotation table from the spec is reflected in code."""

    def test_all_12_tools_registered(self) -> None:
        assert len(ALL_TOOLS) == 12

    def test_read_tools_membership_matches_spec(self) -> None:
        expected_read = {
            "journal_search",
            "journal_list_topics",
            "journal_list_conversations",
            "journal_read_topic",
            "journal_read_conversation",
            "journal_briefing",
            "journal_timeline",
        }
        assert expected_read == READ_TOOLS

    def test_write_tools_membership_matches_spec(self) -> None:
        expected_write = {
            "journal_append_entry",
            "journal_update_entry",
            "journal_create_topic",
            "journal_save_conversation",
            "journal_delete_entry",
        }
        assert expected_write == WRITE_TOOLS

    def test_every_tool_has_annotations(self) -> None:
        actual = _extract_tool_annotations()
        for tool in ALL_TOOLS:
            assert tool in actual, f"{tool} missing @mcp.tool(annotations=ToolAnnotations(...))"

    def test_read_tools_have_readonly_hint_true(self) -> None:
        actual = _extract_tool_annotations()
        for tool in READ_TOOLS:
            assert (
                actual[tool].get("readOnlyHint") is True
            ), f"{tool}: expected readOnlyHint=True, got {actual[tool]}"

    def test_write_tools_have_readonly_hint_false(self) -> None:
        actual = _extract_tool_annotations()
        for tool in WRITE_TOOLS:
            assert (
                actual[tool].get("readOnlyHint") is False
            ), f"{tool}: expected readOnlyHint=False, got {actual[tool]}"

    def test_only_delete_entry_is_destructive(self) -> None:
        actual = _extract_tool_annotations()
        destructive_tools = {
            name for name, ann in actual.items() if ann.get("destructiveHint") is True
        }
        assert destructive_tools == {"journal_delete_entry"}, (
            f"destructiveHint=True must be reserved for delete_entry only, "
            f"got {destructive_tools}"
        )

    def test_idempotent_hints_match_spec(self) -> None:
        actual = _extract_tool_annotations()
        # Per spec: idempotent = update_entry, save_conversation, delete_entry.
        # NOT idempotent = append_entry, create_topic.
        expected_idempotent_true = {
            "journal_update_entry",
            "journal_save_conversation",
            "journal_delete_entry",
        }
        expected_idempotent_false = {
            "journal_append_entry",
            "journal_create_topic",
        }
        for tool in expected_idempotent_true:
            assert (
                actual[tool].get("idempotentHint") is True
            ), f"{tool}: expected idempotentHint=True"
        for tool in expected_idempotent_false:
            assert (
                actual[tool].get("idempotentHint") is False
            ), f"{tool}: expected idempotentHint=False"

    def test_all_annotations_match_spec_table(self) -> None:
        """Full per-tool, per-field assertion against the locked spec table."""
        actual = _extract_tool_annotations()
        for tool, expected_fields in _EXPECTED_ANNOTATIONS.items():
            assert tool in actual, f"{tool} missing annotations"
            for field, expected_val in expected_fields.items():
                got = actual[tool].get(field)
                assert got == expected_val, f"{tool}.{field}: expected {expected_val}, got {got}"


# ---------------------------------------------------------------------------
# _render_scopes_html (consent UI)
# ---------------------------------------------------------------------------


class TestRenderScopesHtml:
    """Verify the consent UI scope rendering helper escapes input and loops over scopes."""

    def test_empty_string_returns_empty(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        assert _render_scopes_html("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        assert _render_scopes_html("   ") == ""

    def test_single_known_scope_renders_description(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        out = _render_scopes_html("journal")
        assert 'class="scopes"' in out
        assert "journal" in out
        assert "read, write, search" in out

    def test_multiple_scopes_each_rendered(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        out = _render_scopes_html("journal openid email")
        assert out.count('class="scope-item"') == 3
        assert "journal" in out
        assert "openid" in out
        assert "email" in out

    def test_unknown_scope_falls_back_to_placeholder(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        out = _render_scopes_html("unknown_scope_xyz")
        assert "unknown_scope_xyz" in out
        assert "No description available." in out

    def test_scope_name_html_escaped(self) -> None:
        """A malicious scope name in the request must be HTML-escaped, not rendered as markup."""
        from journalctl.oauth.templates import _render_scopes_html

        out = _render_scopes_html("<script>alert(1)</script>")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_loops_over_all_scopes_in_input_order(self) -> None:
        from journalctl.oauth.templates import _render_scopes_html

        out = _render_scopes_html("journal email")
        # journal item should appear before email item
        assert out.index("journal") < out.index("email")


# ---------------------------------------------------------------------------
# _wire_scope_filter integration with FastMCP
# ---------------------------------------------------------------------------


class TestScopeFilterWired:
    """_wire_scope_filter restricts tools/list output by token scopes.

    Constructs a FastMCP instance, registers dummy tools, wires the scope
    filter, then exercises the lowlevel handler with various
    current_token_scopes values.
    """

    @pytest.fixture
    def mcp(self) -> FastMCP:
        return FastMCP("test-filter")

    @pytest.fixture
    def wired_mcp(self, mcp: FastMCP) -> FastMCP:
        """FastMCP with dummy tools + scope filter wired."""
        _register_dummy_tools(mcp)
        _wire_scope_filter(mcp)
        return mcp

    async def _list_tool_names(self, mcp: FastMCP) -> set[str]:
        """Call the lowlevel tools/list handler and return tool names."""
        handler = mcp._mcp_server.request_handlers[ListToolsRequest]
        server_result = await handler(None)
        assert isinstance(server_result, ServerResult)
        list_result = server_result.root
        assert isinstance(list_result, ListToolsResult)
        return {t.name for t in list_result.tools}

    # -- empty scopes -> zero tools --

    async def test_empty_scopes_returns_no_tools(self, wired_mcp: FastMCP) -> None:
        token_reset = current_token_scopes.set(frozenset())
        try:
            names = await self._list_tool_names(wired_mcp)
            assert names == set()
        finally:
            current_token_scopes.reset(token_reset)

    # -- journal:read -> only read tools --

    async def test_read_scope_returns_only_read_tools(self, wired_mcp: FastMCP) -> None:
        token_reset = current_token_scopes.set(frozenset({"journal:read"}))
        try:
            names = await self._list_tool_names(wired_mcp)
            assert names == READ_TOOLS
        finally:
            current_token_scopes.reset(token_reset)

    # -- journal:write -> only write tools --

    async def test_write_scope_returns_only_write_tools(self, wired_mcp: FastMCP) -> None:
        token_reset = current_token_scopes.set(frozenset({"journal:write"}))
        try:
            names = await self._list_tool_names(wired_mcp)
            assert names == WRITE_TOOLS
        finally:
            current_token_scopes.reset(token_reset)

    # -- journal (legacy) -> all tools --

    async def test_legacy_journal_scope_returns_all_tools(self, wired_mcp: FastMCP) -> None:
        token_reset = current_token_scopes.set(frozenset({"journal"}))
        try:
            names = await self._list_tool_names(wired_mcp)
            assert names == ALL_TOOLS
        finally:
            current_token_scopes.reset(token_reset)


def _register_dummy_tools(mcp: FastMCP) -> None:
    """Register one dummy async function per tool name so the tool_manager
    has entries for _wire_scope_filter to filter against."""

    async def _dummy(**kwargs: Any) -> dict[str, Any]:
        return {}

    for name in ALL_TOOLS:
        mcp._tool_manager.add_tool(_dummy, name=name, description=f"Dummy {name}")
