"""Mode-gated TLS validator for /.well-known/oauth-protected-resource/mcp.

The validator in ``gubbi.oauth.wellknown.register`` audits the
credential-bearing URLs (``authorization_servers`` + ``resource_documentation``)
at lifespan setup. It runs ONLY when:

- ``settings.is_deployed`` is True (``staging``/``production``), AND
- ``settings.auth.hydra_admin_url`` is non-empty (Mode-3 hosted multi-tenant)

Inside the gate the check is **https-only** -- loopback prefixes
(``http://localhost``, ``http://127.0.0.1``, ``http://[::1]``) are
intentionally rejected. Hosted Mode-3 never legitimately runs Hydra on
loopback, so the loopback allowlist would be pure attack surface there.

Self-host Modes 1 (API-key only) + 2 (full self-host with password) and
non-deployed envs (``dev``, ``ci``, testbench) bypass the check
unconditionally -- those operators may legitimately run on a LAN URL
without TLS, and breaking their deploy on startup would be a regression.

These tests pin both the rejection path (Mode-3 deployed, non-https URL ->
ValueError) and the bypass paths (dev / Mode-1 / Mode-2 -> registers
normally). The latter group is the one that catches a future tightening
that would accidentally break self-host or testbench.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pydantic import AnyHttpUrl

from gubbi.config import AuthConfig, DbConfig, Environment, ServerConfig, Settings
from gubbi.oauth import wellknown
from gubbi.oauth.wellknown import register

WELLKNOWN_PATH = "/.well-known/oauth-protected-resource/mcp"


def _make_settings(
    *,
    app_env: Environment = "production",
    hydra_admin_url: str = "https://hydra-admin.example.com",
    api_key: str = "",
    password_hash: str = "",
    operator_email: str = "",
    server_url: str = "https://mcp.example.com",
) -> Settings:
    """Build a Settings instance via model_construct (skips validators).

    Mirrors ``_make_settings`` in ``test_oauth_protected_resource_metadata.py``
    + ``test_oauth_wellknown.py``. ``model_construct`` lets us drive the
    deploy-shape coordinates (``app_env``, ``hydra_admin_url``, etc.)
    without satisfying the full Mode-3 contract (``hydra_public_issuer_url``,
    ``hydra_public_url``) that the deploy-shape validator would otherwise
    require -- the wellknown validator only reads the two coordinates above.
    """
    return Settings.model_construct(
        app_env=app_env,
        db=DbConfig.model_construct(app_url="sqlite:///memory:", admin_url=""),
        auth=AuthConfig.model_construct(
            api_key=api_key,
            password_hash=password_hash,
            hydra_admin_url=hydra_admin_url,
            hydra_public_issuer_url="",
            hydra_public_url=None,
            operator_email=operator_email,
            trust_gateway=False,
        ),
        server=ServerConfig.model_construct(
            url=server_url,
            host="0.0.0.0",  # noqa: S104
            port=8100,
            transport="streamable-http",
        ),
    )  # type: ignore[call-arg]


def _wellknown_route_present(app: FastAPI) -> bool:
    """True iff register() inserted the wellknown route on this app."""
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    return any(WELLKNOWN_PATH in p for p in paths)


class TestModeThreeDeployedRejectsNonHttps:
    """Deployed Mode-3 hosted: validator runs, non-https URLs raise ValueError."""

    def test_non_https_authorization_server_raises(self) -> None:
        """Mode 3 + production + http authorization server -> ValueError.

        This is the headline case: a misconfigured Mode-3 deploy that
        published a non-TLS authorization server would let MCP clients
        carry credentials to it cleartext. The validator must reject at
        lifespan setup so the misconfig never reaches a real client.
        """
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="non-https"):
            register(app, settings, [AnyHttpUrl("http://evil.example.com")])
        assert not _wellknown_route_present(app), (
            "Validator failed but route was inserted -- registration "
            "must abort before insertion when validation raises."
        )

    def test_loopback_authorization_server_raises_inside_gate(self) -> None:
        """Mode 3 + production + http://localhost authorization server -> ValueError.

        Pins the security-driven decision to reject loopback inside the
        deployed Mode-3 gate (even though it kernel-loopback safe).
        Legitimate hosted Mode-3 never runs Hydra on loopback, so a
        loopback URL here means a deploy mistake or compromised config
        routing through a local interceptor.
        """
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="non-https"):
            register(app, settings, [AnyHttpUrl("http://localhost:4444")])
        assert not _wellknown_route_present(app)

    def test_https_loopback_authorization_server_raises(self) -> None:
        """Mode 3 + production + https://localhost -> ValueError.

        Closes the gap where the validator's error message advertised
        loopback rejection but ``_is_https`` alone admitted any
        ``https://localhost*`` URL. A hosted Mode-3 deploy that publishes
        an https-wrapped loopback authorization server is still routing
        credential traffic through the local interface -- exactly the
        downgrade vector the gate exists to block.
        """
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="loopback"):
            register(app, settings, [AnyHttpUrl("https://localhost:4444")])
        assert not _wellknown_route_present(app)

    def test_https_loopback_127_0_0_1_raises(self) -> None:
        """Mode 3 + production + https://127.0.0.1 -> ValueError.

        IPv4 loopback companion to ``test_https_loopback_authorization_server_raises``;
        pins that the loopback rejection covers the dotted-quad form as
        well as the ``localhost`` hostname.
        """
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="loopback"):
            register(app, settings, [AnyHttpUrl("https://127.0.0.1:4444")])
        assert not _wellknown_route_present(app)

    def test_https_loopback_ipv6_raises(self) -> None:
        """Mode 3 + production + https://[::1] -> ValueError.

        IPv6 loopback companion; pins that ``_is_loopback_host``
        recognises the bracketed-IPv6 form that ``urlparse`` exposes via
        ``parsed.hostname == "::1"``.
        """
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="loopback"):
            register(app, settings, [AnyHttpUrl("https://[::1]:4444")])
        assert not _wellknown_route_present(app)

    def test_https_authorization_server_passes(self) -> None:
        """Mode 3 + production + https authorization server -> registers."""
        # Arrange
        settings = _make_settings()
        app = FastAPI()

        # Act
        register(app, settings, [AnyHttpUrl("https://auth.example.com")])

        # Assert
        assert _wellknown_route_present(app)

    def test_non_https_resource_documentation_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mode 3 + production + http resource_documentation -> ValueError.

        ``RESOURCE_DOCUMENTATION_URL`` is a module-level constant in
        ``gubbi.oauth.wellknown`` so the validator audits it alongside
        ``authorization_servers``. Patch the constant for this test to
        simulate a misconfigured documentation URL without making the
        URL itself injectable through the call signature.
        """
        # Arrange
        monkeypatch.setattr(
            wellknown,
            "RESOURCE_DOCUMENTATION_URL",
            "http://docs.example.com",
        )
        settings = _make_settings()
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="non-https"):
            register(app, settings, [AnyHttpUrl("https://auth.example.com")])
        assert not _wellknown_route_present(app)

    def test_non_https_resource_url_raises(self) -> None:
        """Mode 3 + production + http server_url -> ValueError.

        ``resource_url`` is constructed at lifespan setup from
        ``settings.server.url`` (the ``JOURNAL_SERVER_URL`` env var) and
        is the third credential-bearing URL the metadata document
        advertises -- it is the audience MCP clients present bearer
        tokens to. The validator must reject a non-TLS server_url in
        deployed Mode-3 hosted so a misconfigured ``JOURNAL_SERVER_URL``
        is caught loudly at startup rather than producing a silent
        downgrade vector at the bearer-presentation step.
        """
        # Arrange
        settings = _make_settings(server_url="http://mcp.example.com")
        app = FastAPI()

        # Act + Assert
        with pytest.raises(ValueError, match="non-https"):
            register(app, settings, [AnyHttpUrl("https://auth.example.com")])
        assert not _wellknown_route_present(app)


class TestValidatorBypass:
    """Self-host + non-deployed envs bypass the validator and register normally.

    These tests pin the no-op behaviour for the three flows where a
    non-TLS URL is legitimate:
    - Dev / testbench (``app_env`` is ``dev`` -> ``is_deployed`` is False)
    - Self-host Mode 2 (deployed but no Hydra; LAN URL is the operator's choice)
    - Self-host Mode 1 (API-key only; same LAN reasoning)

    A regression that tightens the validator to fire on any of these
    surfaces here as a hard failure rather than silently breaking real
    deploys.
    """

    def test_dev_mode_non_tls_passes(self) -> None:
        """Testbench scenario: app_env=dev, localhost Hydra -> validator skipped.

        Testbench drives Hydra at ``http://localhost:4444``; if the
        validator fired here the local OAuth E2E suite would refuse
        to start. ``is_deployed`` is False for ``dev`` per DEC-094.
        """
        # Arrange
        settings = _make_settings(
            app_env="dev",
            hydra_admin_url="http://localhost:4444",
        )
        app = FastAPI()

        # Act
        register(app, settings, [AnyHttpUrl("http://localhost:4444")])

        # Assert
        assert _wellknown_route_present(app)

    def test_self_host_mode_2_non_tls_passes(self) -> None:
        """Self-host Mode 2 (production + password, no Hydra) on a LAN URL.

        Operator deploys gubbi on their home server at e.g.
        ``http://192.168.1.10:8100``. Without ``hydra_admin_url`` the
        validator must skip even though ``is_deployed`` is True.
        """
        # Arrange
        settings = _make_settings(
            app_env="production",
            hydra_admin_url="",
            password_hash="$argon2id$dummy",
            api_key="x" * 32,
            operator_email="op@example.com",
        )
        app = FastAPI()

        # Act
        register(app, settings, [AnyHttpUrl("http://192.168.1.10:8100")])

        # Assert
        assert _wellknown_route_present(app)

    def test_self_host_mode_1_non_tls_passes(self) -> None:
        """Self-host Mode 1 (production + API-key only, no Hydra, no password).

        The minimal CLI-only deploy. Same LAN reasoning as Mode 2.
        """
        # Arrange
        settings = _make_settings(
            app_env="production",
            hydra_admin_url="",
            password_hash="",
            api_key="x" * 32,
            operator_email="op@example.com",
        )
        app = FastAPI()

        # Act
        register(app, settings, [AnyHttpUrl("http://192.168.1.10:8100")])

        # Assert
        assert _wellknown_route_present(app)
