"""Contract test pinning the field shape of the protected-resource metadata document.

Anthropic and OpenAI MCP clients GET ``/.well-known/oauth-protected-resource/mcp``
at consent time to discover the authorization server. RFC 9728 + the MCP spec
define the required and recommended fields for that JSON response. ChatGPT in
particular renders its consent dialog from ``scopes_supported``: if a refactor
silently drops that key, the client-side consent UX breaks with no server-side
error.

These tests pin the *currently emitted* field set so any future drop surfaces
here. They do not assert exact values (e.g. they don't lock in the specific
scope strings or doc URL), only that each field is present and well-formed.

Splits:
- Required (RFC 9728): ``resource``, ``authorization_servers``
- Recommended (MCP spec / client expectations): ``scopes_supported``,
  ``bearer_methods_supported``, ``resource_documentation``
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import AnyHttpUrl

from gubbi.config import AuthConfig, DbConfig, ServerConfig, Settings
from gubbi.oauth.wellknown import TLS_OR_LOOPBACK_PREFIXES, register

WELLKNOWN_PATH = "/.well-known/oauth-protected-resource/mcp"
TEST_SERVER_URL = "http://localhost:8100"
TEST_AUTH_SERVER = "https://auth.example.com"

# URL schemes accepted for credential-bearing endpoints (authorization_servers,
# resource_documentation). https-only is the security target; localhost loopbacks
# are admitted for dev environments. Non-TLS off-host URLs are downgrade vectors
# and would let credentials transit cleartext.
#
# Source of truth lives in gubbi.oauth.wellknown alongside the runtime validator
# that enforces the same rule on deployed Mode-3 hosted setups; the local alias
# below preserves the existing assertion sites without spelling the import name
# at every call.
_TLS_OR_LOOPBACK_PREFIXES: tuple[str, ...] = TLS_OR_LOOPBACK_PREFIXES

# Source of truth: oauth/wellknown.py:register's scopes_supported list.
# Keep in sync. Hard-pinned so a refactor that adds (e.g. ``admin``,
# ``write:all``) or drops a scope requires an explicit test edit -- silent
# expansion of the published consent surface is the security concern.
# Tuple (not list) so the module-level constant is immutable.
_EXPECTED_SCOPES: tuple[str, ...] = tuple(sorted(["journal", "offline_access", "openid", "email"]))


def _make_settings(server_url: str = TEST_SERVER_URL) -> Settings:
    """Build a minimal Settings instance that drives the wellknown route.

    Mirrors the construction pattern in ``test_oauth_wellknown.py`` and
    ``test_oauth_router.py`` -- ``model_construct`` skips validators so the
    test stays focused on the metadata document and never touches the
    deploy-shape rules.
    """
    return Settings.model_construct(
        db=DbConfig.model_construct(app_url="sqlite:///memory:", admin_url=""),
        auth=AuthConfig.model_construct(
            api_key="testkey123",
            password_hash="",
            hydra_admin_url="",
            hydra_public_issuer_url="",
            hydra_public_url=None,
            operator_email="admin@example.com",
            trust_gateway=False,
        ),
        server=ServerConfig.model_construct(
            url=server_url,
            host="0.0.0.0",  # noqa: S104
            port=8100,
            transport="streamable-http",
        ),
    )  # type: ignore[call-arg]


def _build_app() -> FastAPI:
    """Register the wellknown route on a fresh FastAPI app and return it."""
    app = FastAPI()
    register(
        app,
        _make_settings(),
        authorization_servers=[AnyHttpUrl(TEST_AUTH_SERVER)],
    )
    return app


def _fetch_metadata() -> dict:
    """GET the wellknown route and return the parsed JSON body."""
    client = TestClient(_build_app())
    response = client.get(WELLKNOWN_PATH)
    assert response.status_code == 200, (
        f"Expected 200 from {WELLKNOWN_PATH}; got {response.status_code} "
        f"with body: {response.text}"
    )
    assert response.headers["content-type"].startswith(
        "application/json"
    ), f"Expected JSON content-type; got {response.headers['content-type']}"
    return response.json()


class TestProtectedResourceMetadataRequiredFields:
    """RFC 9728 section 2 required fields. Dropping any of these is a CRITICAL break."""

    def test_resource_field_present_and_is_url_string(self) -> None:
        """``resource`` must be present and a non-empty URL string."""
        body = _fetch_metadata()
        assert "resource" in body, f"Missing required 'resource' field. Body: {body}"
        resource = body["resource"]
        assert isinstance(
            resource, str
        ), f"'resource' must be a string; got {type(resource).__name__}: {resource!r}"
        assert resource, "'resource' must be a non-empty string"
        assert resource.startswith(
            ("http://", "https://")
        ), f"'resource' must be an http(s) URL; got {resource!r}"

    def test_authorization_servers_present_nonempty_list_of_url_strings(self) -> None:
        """``authorization_servers`` must be a list of >=1 https URL strings.

        RFC 9728 marks this REQUIRED with min length 1; the MCP SDK enforces
        ``min_length=1`` on the Pydantic model. A future change that emits an
        empty list would still surface here via the per-element checks.

        Each entry MUST be ``https://`` (or a localhost loopback for dev) --
        a non-TLS authorization server would be a downgrade vector for
        the credential endpoints (token, registration) the client follows
        from this metadata document.
        """
        body = _fetch_metadata()
        assert (
            "authorization_servers" in body
        ), f"Missing required 'authorization_servers' field. Body: {body}"
        servers = body["authorization_servers"]
        assert isinstance(
            servers, list
        ), f"'authorization_servers' must be a list; got {type(servers).__name__}"
        assert (
            len(servers) >= 1
        ), "'authorization_servers' must contain at least one entry; got empty list"
        for idx, server in enumerate(servers):
            assert isinstance(server, str), (
                f"authorization_servers[{idx}] must be a string; "
                f"got {type(server).__name__}: {server!r}"
            )
            assert server.startswith(_TLS_OR_LOOPBACK_PREFIXES), (
                f"authorization_servers[{idx}] must be https:// or a localhost "
                f"loopback; got {server!r}. A non-TLS authorization server is a "
                "downgrade vector -- credential endpoints would transit cleartext."
            )


class TestProtectedResourceMetadataRecommendedFields:
    """RFC 9728 / MCP spec recommended fields. Gubbi emits these today.

    A drop should surface here so we notice before the consent UX breaks
    on the client side (especially ChatGPT, which renders its consent
    dialog from ``scopes_supported``).
    """

    def test_scopes_supported_present_as_list_of_strings(self) -> None:
        """``scopes_supported`` -- ChatGPT renders consent dialog from this.

        Hard-pin the exact set so a refactor that adds (e.g. ``admin``,
        ``write:all``) or drops a scope requires an explicit test edit.
        Silent expansion of the published consent surface is the security
        concern this assertion guards.
        """
        body = _fetch_metadata()
        assert "scopes_supported" in body, (
            "Missing 'scopes_supported'. ChatGPT renders its consent dialog "
            f"from this field; dropping it breaks the consent UX. Body: {body}"
        )
        scopes = body["scopes_supported"]
        assert isinstance(
            scopes, list
        ), f"'scopes_supported' must be a list; got {type(scopes).__name__}"
        assert (
            len(scopes) >= 1
        ), "'scopes_supported' must contain at least one scope; got empty list"
        for idx, scope in enumerate(scopes):
            assert isinstance(scope, str), (
                f"scopes_supported[{idx}] must be a string; "
                f"got {type(scope).__name__}: {scope!r}"
            )
            assert scope, f"scopes_supported[{idx}] must be a non-empty string"

        # Hard-pin: any addition or removal must be a deliberate test edit.
        # See _EXPECTED_SCOPES (module top) for the source-of-truth pointer.
        assert tuple(sorted(scopes)) == _EXPECTED_SCOPES, (
            f"scopes_supported drift detected. Expected {list(_EXPECTED_SCOPES)}; "
            f"got {sorted(scopes)}. If this change is intentional, update "
            "_EXPECTED_SCOPES in this test (scopes published on the consent "
            "dialog widen the credential surface; review the addition/removal "
            "at PR time)."
        )

    def test_bearer_methods_supported_present_as_list_of_strings(self) -> None:
        """``bearer_methods_supported`` -- the SDK defaults to ``["header"]``."""
        body = _fetch_metadata()
        assert (
            "bearer_methods_supported" in body
        ), f"Missing 'bearer_methods_supported'. Body: {body}"
        methods = body["bearer_methods_supported"]
        assert isinstance(
            methods, list
        ), f"'bearer_methods_supported' must be a list; got {type(methods).__name__}"
        assert (
            len(methods) >= 1
        ), "'bearer_methods_supported' must contain at least one method; got empty list"
        for idx, method in enumerate(methods):
            assert isinstance(method, str), (
                f"bearer_methods_supported[{idx}] must be a string; "
                f"got {type(method).__name__}: {method!r}"
            )
            assert method, f"bearer_methods_supported[{idx}] must be a non-empty string"

    def test_resource_documentation_when_present_is_url_string(self) -> None:
        """``resource_documentation`` is recommended; gubbi currently emits it.

        Allowed states:
        - Present and a valid https URL string (or localhost loopback for dev).
        - Absent (RFC 9728 marks it OPTIONAL).

        If present but not a string / not a valid URL scheme, fail. Same
        scheme tightening as ``authorization_servers``: non-TLS doc URLs
        could host malicious content under a phishing-friendly hostname.
        """
        body = _fetch_metadata()
        if "resource_documentation" in body:
            doc = body["resource_documentation"]
            assert isinstance(doc, str), (
                f"'resource_documentation' must be a string when present; "
                f"got {type(doc).__name__}: {doc!r}"
            )
            assert doc.startswith(_TLS_OR_LOOPBACK_PREFIXES), (
                f"'resource_documentation' must be https:// or a localhost "
                f"loopback when present; got {doc!r}"
            )

    def test_resource_documentation_currently_emitted(self) -> None:
        """Pin the *current* behaviour: gubbi emits ``resource_documentation``.

        Kept as a separate test from the structural check above so a deliberate
        future decision to drop the field (RFC 9728 makes it OPTIONAL) yields a
        single clear failure here, not a cascade across the document.
        """
        body = _fetch_metadata()
        assert "resource_documentation" in body, (
            "Gubbi currently emits 'resource_documentation' (see oauth/wellknown.py). "
            "If this is being dropped on purpose, also remove this assertion. "
            f"Body: {body}"
        )
