"""Unit tests for ``gubbi.app_state`` typed accessors.

Each ``app.state`` field exposed by the lifespan has a paired
``require_*`` / ``get_optional_*`` accessor. These tests verify the
contract:

* ``require_*`` raises ``RuntimeError`` with a clear message when the
  field is missing from ``request.app.state``.
* ``get_optional_*`` returns ``None`` when the field is missing.

The "field present but value is None" case is also exercised for
required fields. ``gateway_secret`` is intentionally optional-only --
``None`` is a valid configured-disabled state for HMAC signing -- and is
covered separately below (with a locking test that asserts no
``require_gateway_secret`` exists).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.datastructures import State

from gubbi import app_state

# (require_fn, get_optional_fn, field_name, error_message_fragment)
_RAISE_WHEN_NONE_FIELDS: list[tuple[Any, Any, str, str]] = [
    (
        app_state.require_app_ctx,
        app_state.get_optional_app_ctx,
        "app_ctx",
        "app_ctx not initialised",
    ),
    (
        app_state.require_auth_strategies,
        app_state.get_optional_auth_strategies,
        "auth_strategies",
        "auth_strategies not initialised",
    ),
    (
        app_state.require_hydra_introspector,
        app_state.get_optional_hydra_introspector,
        "hydra_introspector",
        "hydra_introspector not initialised",
    ),
    (
        app_state.require_selfhost_token_validator,
        app_state.get_optional_selfhost_token_validator,
        "selfhost_token_validator",
        "selfhost_token_validator not initialised",
    ),
    (
        app_state.require_operator_user_id,
        app_state.get_optional_operator_user_id,
        "operator_user_id",
        "operator_user_id not initialised",
    ),
    (
        app_state.require_redis_client,
        app_state.get_optional_redis_client,
        "redis_client",
        "redis_client not initialised",
    ),
]


def _make_request(state: State | None = None) -> Request:
    """Build a Request whose ``request.app.state`` is the given State.

    A fresh ``State()`` (empty) is used by default so attribute lookups
    raise ``AttributeError`` and the accessors trigger their fallback
    paths. The ``Request`` itself is mocked because the accessors only
    ever touch ``request.app.state``.
    """
    request = MagicMock(spec=Request)
    request.app = MagicMock()
    request.app.state = State() if state is None else state
    return request


@pytest.mark.unit
@pytest.mark.parametrize(
    ("require_fn", "get_optional_fn", "field", "error_fragment"),
    _RAISE_WHEN_NONE_FIELDS,
)
def test_require_raises_when_state_is_empty(
    require_fn: Any,
    get_optional_fn: Any,
    field: str,
    error_fragment: str,
) -> None:
    """``require_*`` raises RuntimeError when the field is absent."""
    request = _make_request()
    with pytest.raises(RuntimeError, match=error_fragment):
        require_fn(request)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("require_fn", "get_optional_fn", "field", "error_fragment"),
    _RAISE_WHEN_NONE_FIELDS,
)
def test_get_optional_returns_none_when_state_is_empty(
    require_fn: Any,
    get_optional_fn: Any,
    field: str,
    error_fragment: str,
) -> None:
    """``get_optional_*`` returns None when the field is absent."""
    request = _make_request()
    assert get_optional_fn(request) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("require_fn", "get_optional_fn", "field", "error_fragment"),
    _RAISE_WHEN_NONE_FIELDS,
)
def test_require_raises_when_value_is_none(
    require_fn: Any,
    get_optional_fn: Any,
    field: str,
    error_fragment: str,
) -> None:
    """For non-gateway-secret fields, present-but-None is also "not initialised"."""
    state = State()
    setattr(state, field, None)
    request = _make_request(state)
    with pytest.raises(RuntimeError, match=error_fragment):
        require_fn(request)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("require_fn", "get_optional_fn", "field", "error_fragment"),
    _RAISE_WHEN_NONE_FIELDS,
)
def test_accessors_return_value_when_present(
    require_fn: Any,
    get_optional_fn: Any,
    field: str,
    error_fragment: str,
) -> None:
    """Both accessors return the stored value when the field is set."""
    sentinel = object()
    state = State()
    setattr(state, field, sentinel)
    request = _make_request(state)
    assert require_fn(request) is sentinel
    assert get_optional_fn(request) is sentinel


# ---------------------------------------------------------------------------
# gateway_secret has no require_* form: ``None`` is a valid configured-
# disabled state (HMAC signing turned off), not a lifespan failure. The
# lifespan reads ``app.state.gubbi_gateway_secret`` directly; no
# production code consumes the accessor via a Request object. Mirrors
# gubbi-cloud/app_state.py.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_require_for_gateway_secret() -> None:
    """Locks asymmetric design: gubbi_gateway_secret has only get_optional_*, no require_*.

    Mirrors gubbi-cloud. None is a valid configured-disabled state.
    """
    import gubbi.app_state as app_state

    assert not hasattr(app_state, "require_gateway_secret")


@pytest.mark.unit
def test_get_optional_gateway_secret_returns_none_when_missing() -> None:
    """``get_optional_gateway_secret`` returns None when absent."""
    request = _make_request()
    assert app_state.get_optional_gateway_secret(request) is None


@pytest.mark.unit
def test_get_optional_gateway_secret_returns_bytes_when_set() -> None:
    """``get_optional_gateway_secret`` returns the bytes value when set."""
    secret = b"\x01\x02\x03"
    state = State()
    state.gubbi_gateway_secret = secret
    request = _make_request(state)
    assert app_state.get_optional_gateway_secret(request) == secret
