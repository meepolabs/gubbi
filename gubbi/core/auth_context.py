"""Compatibility shim -- auth_context moved to gubbi.auth_context."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "current_user_id":
        warnings.warn(
            "gubbi.core.auth_context is deprecated; " "import from gubbi.auth_context instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.auth_context import current_user_id  # noqa: PLC0415

        return current_user_id
    if name == "current_token_scopes":
        warnings.warn(
            "gubbi.core.auth_context is deprecated; " "import from gubbi.auth_context instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.auth_context import current_token_scopes  # noqa: PLC0415

        return current_token_scopes
    if name == "get_current_user_id":
        warnings.warn(
            "gubbi.core.auth_context is deprecated; " "import from gubbi.auth_context instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.auth_context import get_current_user_id  # noqa: PLC0415

        return get_current_user_id
    if name == "AuthenticationError":
        warnings.warn(
            "gubbi.core.auth_context is deprecated; " "import from gubbi.auth_context instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.auth_context import AuthenticationError  # noqa: PLC0415

        return AuthenticationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
