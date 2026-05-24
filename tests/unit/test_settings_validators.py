"""Unit tests for Settings model validators (AAA-pattern, pytest.parametrize).

Pins behaviour of the three model-validators on ``Settings``:

* ``_validate_deploy_shape`` -- the 3-shape matrix
  (Mode 1 = API-key only; Mode 2 = full self-host; Mode 3 = multi-tenant
  hosted via Hydra).
* ``_warn_on_require_signature_without_secret`` -- gateway-secret presence
  warning on AuthConfig.
* ``_validate_trust_gateway_signature`` -- the env-gated trust-gateway
  signature requirement (deployed envs only).

Also pins the structural-guard sentinel pattern (Q1) that closes the
JSON-decode bypass attack class on the parent Settings sub-config fields.
"""

from __future__ import annotations

import os
import re
import typing

import pytest
from pydantic import ValidationError

from gubbi.config import Settings, get_settings


def _unwrap_annotation(ann: object) -> type | None:
    """Unwrap Annotated/Optional/Union wrappers; return the bare class or None.

    Used by ``test_nested_anchor_sentinels_are_distinct_per_subconfig`` so a
    future 5th sub-config field annotated as ``Optional[CacheConfig]`` or
    ``Annotated[CacheConfig, ...]`` does not silently slip past the
    isinstance/issubclass filter.
    """
    origin = typing.get_origin(ann)
    if origin is not None:
        # Annotated[T, ...] -- get_args returns (T, *metadata); take T.
        if hasattr(typing, "Annotated") and origin is typing.Annotated:
            ann = typing.get_args(ann)[0]
            origin = typing.get_origin(ann)
        # Union/Optional -- pick the first non-None arg if it's a single type.
        if origin is typing.Union:
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(args) == 1:
                ann = args[0]
    return ann if isinstance(ann, type) else None


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum env that lets Settings construct without raising.

    Also clears any ``JOURNAL_*`` env vars from the developer's local
    environment so test runs are isolated from a stray Doppler / .env
    export that might otherwise flip a test result.
    """
    for name in list(os.environ):
        if name.startswith("JOURNAL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://app@host/db")
    monkeypatch.setenv("JOURNAL_API_KEY", "x" * 32)
    monkeypatch.setenv("JOURNAL_OPERATOR_EMAIL", "op@example.com")
    get_settings.cache_clear()


# -------- _validate_deploy_shape --------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "match"),
    [
        # Both Hydra + password -> reject (mutually exclusive).
        (
            {
                "JOURNAL_HYDRA_ADMIN_URL": "http://h:4445",
                "JOURNAL_HYDRA_PUBLIC_URL": "https://auth/",
                "JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "https://auth/",
                "JOURNAL_PASSWORD_HASH": "$2b$12$abc",
            },
            "mutually exclusive",
        ),
        # Hydra admin alone (no public_issuer) -> reject.
        (
            {
                "JOURNAL_HYDRA_ADMIN_URL": "http://h:4445",
                "JOURNAL_API_KEY": "",
            },
            "JOURNAL_HYDRA_PUBLIC_ISSUER_URL",
        ),
        # Issuer alone (no admin URL) -> reject.
        (
            {"JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "https://auth/"},
            "JOURNAL_HYDRA_ADMIN_URL",
        ),
        # Public URL alone (no admin URL) -> reject.
        (
            {"JOURNAL_HYDRA_PUBLIC_URL": "https://auth/"},
            "JOURNAL_HYDRA_ADMIN_URL",
        ),
        # ADMIN + ISSUER set, PUBLIC_URL missing -> reject.
        (
            {
                "JOURNAL_HYDRA_ADMIN_URL": "http://h:4445",
                "JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "https://auth/",
                "JOURNAL_API_KEY": "",
            },
            "JOURNAL_HYDRA_PUBLIC_URL is required",
        ),
        # Mode 3 (Hydra triplet) with operator_email set -> reject.
        (
            {
                "JOURNAL_HYDRA_ADMIN_URL": "http://h:4445",
                "JOURNAL_HYDRA_PUBLIC_URL": "https://auth/",
                "JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "https://auth/",
                "JOURNAL_OPERATOR_EMAIL": "op@example.com",
                "JOURNAL_API_KEY": "",
            },
            "JOURNAL_OPERATOR_EMAIL",
        ),
        # Mode 1 with empty API_KEY (no Hydra) -> reject.
        (
            {"JOURNAL_API_KEY": ""},
            "JOURNAL_API_KEY",
        ),
        # Mode 1 with empty OPERATOR_EMAIL -> reject.
        (
            {"JOURNAL_OPERATOR_EMAIL": ""},
            "JOURNAL_OPERATOR_EMAIL",
        ),
    ],
)
def test_validate_deploy_shape_rejects(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    match: str,
) -> None:
    """Each invalid env combination raises ValidationError with the expected hint."""
    # Arrange
    _base_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # Act + Assert
    with pytest.raises(ValidationError, match=match):
        Settings()


@pytest.mark.unit
@pytest.mark.parametrize(
    "env",
    [
        # Mode 1 happy path -- only the base env vars.
        {},
        # Mode 2 happy path -- adds password_hash, no Hydra.
        {"JOURNAL_PASSWORD_HASH": "$2b$12$abc"},
        # Mode 3 happy path -- Hydra triplet, no operator_email/password/api_key.
        {
            "JOURNAL_HYDRA_ADMIN_URL": "http://h:4445",
            "JOURNAL_HYDRA_PUBLIC_URL": "https://auth/",
            "JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "https://auth/",
            "JOURNAL_API_KEY": "",
            "JOURNAL_OPERATOR_EMAIL": "",
        },
    ],
)
def test_validate_deploy_shape_accepts(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """Each valid deploy shape constructs without raising."""
    # Arrange
    _base_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # Act + Assert (must not raise)
    Settings()


# -------- _warn_on_require_signature_without_secret --------


@pytest.mark.unit
def test_warn_emitted_when_signature_required_but_secret_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Signature-required + empty secret -> warning is logged."""
    # Arrange
    _base_env(monkeypatch)
    monkeypatch.setenv("JOURNAL_GATEWAY_REQUIRE_SIGNATURE", "true")
    monkeypatch.setenv("JOURNAL_GUBBI_GATEWAY_SECRET", "")

    # Act
    Settings()

    # Assert
    assert any("GATEWAY_REQUIRE_SIGNATURE" in str(r.getMessage()) for r in caplog.records)


@pytest.mark.unit
def test_warn_silent_when_secret_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Signature-required + non-empty secret -> no warning is logged."""
    # Arrange
    _base_env(monkeypatch)
    monkeypatch.setenv("JOURNAL_GATEWAY_REQUIRE_SIGNATURE", "true")
    monkeypatch.setenv("JOURNAL_GUBBI_GATEWAY_SECRET", "ab" * 32)

    # Act
    Settings()

    # Assert
    assert not any("GATEWAY_REQUIRE_SIGNATURE" in str(r.getMessage()) for r in caplog.records)


# -------- _validate_trust_gateway_signature (env-gated) --------


@pytest.mark.unit
def test_trust_gateway_signature_only_enforced_in_deployed_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trust_gateway=True without signature is rejected only in staging/production."""
    # Arrange
    _base_env(monkeypatch)
    monkeypatch.setenv("JOURNAL_TRUST_GATEWAY", "true")
    monkeypatch.setenv("JOURNAL_GATEWAY_REQUIRE_SIGNATURE", "false")
    monkeypatch.setenv("JOURNAL_GUBBI_GATEWAY_SECRET", "ab" * 32)

    # Act + Assert: dev permits the unsafe combination.
    monkeypatch.setenv("JOURNAL_APP_ENV", "dev")
    Settings()

    # Act + Assert: ci permits the unsafe combination.
    monkeypatch.setenv("JOURNAL_APP_ENV", "ci")
    Settings()

    # Act + Assert: production rejects.
    monkeypatch.setenv("JOURNAL_APP_ENV", "production")
    with pytest.raises(ValidationError, match="trust_gateway"):
        Settings()

    # Act + Assert: staging rejects.
    monkeypatch.setenv("JOURNAL_APP_ENV", "staging")
    with pytest.raises(ValidationError, match="trust_gateway"):
        Settings()


# -------- Q1 sentinel structural-guard regression test --------


@pytest.mark.unit
def test_nested_form_env_vars_not_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON-decode bypass attack class is closed -- AUTH={...} is ignored.

    Pins the structural-guard sentinel pattern. Pre-fix, a bare ``DB``,
    ``AUTH`` etc. env var holding a JSON object would have been parsed
    into the corresponding sub-config, overriding per-field flat env vars.
    The validation_alias sentinels on the parent Settings make pydantic-
    settings look up names like ``JOURNAL_NESTED_ANCHOR_DB`` which are
    intentionally never set, so the JSON injection cannot land.
    """
    # Arrange
    _base_env(monkeypatch)
    # Defender's flat env vars.
    monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://defender@host/db")
    # Attacker's JSON-decode injection attempts -- both bare and
    # ``JOURNAL_``-prefixed variants. None of these names match the
    # structural-guard sentinels, so they must be ignored.
    monkeypatch.setenv("AUTH", '{"api_key":"injected","operator_email":"a@b.com"}')
    monkeypatch.setenv("DB", '{"app_url":"postgresql://attacker"}')
    monkeypatch.setenv("SERVER", '{"port":1}')
    monkeypatch.setenv("LLM", '{"api_key":"sneaky"}')
    monkeypatch.setenv("JOURNAL_AUTH", '{"api_key":"injected"}')
    monkeypatch.setenv("JOURNAL_DB", '{"app_url":"postgresql://attacker"}')
    monkeypatch.setenv("JOURNAL_SERVER", '{"port":1}')
    monkeypatch.setenv("JOURNAL_LLM", '{"api_key":"sneaky"}')

    # Act
    s = Settings()

    # Assert: per-field flat env vars win; JSON injection is ignored.
    assert s.auth.api_key == "x" * 32
    assert "injected" not in s.auth.api_key
    assert "attacker" not in s.db.app_url
    assert "defender" in s.db.app_url
    assert "sneaky" not in s.llm.api_key
    # server.port keeps its in-code default (8100); the JSON object
    # never reaches the ServerConfig constructor.
    assert s.server.port == 8100


@pytest.mark.unit
def test_sentinel_set_with_alias_keyed_json_does_not_inject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-braces: setting the sentinel itself drops the dict-shaped input.

    Closes the residual case where a misconfigured environment sets the
    sentinel env var (``JOURNAL_NESTED_ANCHOR_AUTH``) with alias-keyed
    JSON. The mode="before" model validator on ``Settings`` drops dict-
    shaped inputs for the sub-config anchors so ``default_factory`` runs
    unconditionally and per-field flat env vars retain control.
    """
    # Arrange
    _base_env(monkeypatch)
    monkeypatch.setenv(
        "JOURNAL_NESTED_ANCHOR_AUTH",
        '{"JOURNAL_API_KEY":"injected_long_enough_value_xxxxxxxxxxxx"}',
    )
    monkeypatch.setenv(
        "JOURNAL_NESTED_ANCHOR_DB",
        '{"JOURNAL_DB_APP_URL":"postgresql://attacker"}',
    )

    # Act
    s = Settings()

    # Assert: defender's flat values from _base_env() win; sentinel-set
    # JSON dropped by _force_subconfig_default_factory.
    assert s.auth.api_key == "x" * 32
    assert "injected" not in s.auth.api_key
    assert "attacker" not in s.db.app_url


@pytest.mark.unit
def test_field_name_dict_via_kwargs_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings(auth={...}) drops the dict and constructs auth via env-only default_factory.

    Pins the field-name keyed branch of ``_force_subconfig_default_factory``
    (the alias-keyed branch is exercised by
    ``test_sentinel_set_with_alias_keyed_json_does_not_inject``). A
    misconfigured caller passing ``Settings(auth={...})`` must not inject
    via the parent-keyed dict path; the validator pops the field-name key
    so ``default_factory`` runs and reads per-field flat env vars only.
    """
    # Arrange
    _base_env(monkeypatch)

    # Act: call site passes a dict-shaped kwarg keyed by field name.
    s = Settings(auth={"JOURNAL_API_KEY": "injected_" + "x" * 32})

    # Assert: the dict-shaped kwarg was popped by
    # _force_subconfig_default_factory; default_factory ran and read
    # JOURNAL_API_KEY=x*32 from env (set by _base_env).
    assert s.auth.api_key == "x" * 32
    assert "injected" not in s.auth.api_key


@pytest.mark.unit
def test_nested_anchor_sentinels_are_distinct_per_subconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: each sub-config field has a unique sentinel alias.

    Reuse of one sentinel name across two fields would risk one field's
    JSON decode being applied to the other. Discovers BaseSettings sub-
    config fields programmatically so a future 5th sub-config added
    without a sentinel surfaces immediately rather than silently passing
    a hardcoded check.

    Uses ``_unwrap_annotation`` so a future field annotated as
    ``Optional[BaseSettings]`` or ``Annotated[BaseSettings, ...]`` is also
    discovered -- the bare ``isinstance(fi.annotation, type)`` filter
    returns False for those wrappers and would let the field slip past.
    """
    from pydantic_settings import BaseSettings

    # Arrange
    _base_env(monkeypatch)
    subconfig_fields = {}
    for name, fi in Settings.model_fields.items():
        bare = _unwrap_annotation(fi.annotation)
        if bare is not None and issubclass(bare, BaseSettings):
            subconfig_fields[name] = fi

    # Act
    s = Settings()

    # Assert: each BaseSettings sub-config has a sentinel-shaped string alias.
    for name, fi in subconfig_fields.items():
        assert isinstance(fi.validation_alias, str), name
        assert fi.validation_alias.startswith("JOURNAL_NESTED_ANCHOR_"), name

    # All distinct -- reuse would risk one field's JSON decode being
    # applied to a sibling sub-config.
    aliases = [fi.validation_alias for fi in subconfig_fields.values()]
    assert len(set(aliases)) == len(aliases)

    # At least the current 4 sub-configs are covered; a regression that
    # drops one (or a 5th added without a sentinel) trips here.
    assert len(subconfig_fields) >= 4

    # Settings constructed without the sentinels being set -> default
    # factories ran and per-field flat envs populated the sub-configs.
    assert s.db.app_url == "postgresql://app@host/db"
    assert s.auth.api_key == "x" * 32

    # Match the alias regex shape so a future rename catches missing
    # field updates.
    pattern = re.compile(r"^JOURNAL_NESTED_ANCHOR_[A-Z]+$")
    assert all(pattern.match(a) for a in aliases if isinstance(a, str))
