"""Compatibility shim -- audit_decorator moved to gubbi.audit."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "audited":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit import audited

        return audited
    if name == "ACTION_ENTRY_CREATED":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import ACTION_ENTRY_CREATED

        return ACTION_ENTRY_CREATED
    if name == "ACTION_ENTRY_UPDATED":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import ACTION_ENTRY_UPDATED

        return ACTION_ENTRY_UPDATED
    if name == "ACTION_ENTRY_DELETED":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import ACTION_ENTRY_DELETED

        return ACTION_ENTRY_DELETED
    if name == "ACTION_TOPIC_CREATED":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import ACTION_TOPIC_CREATED

        return ACTION_TOPIC_CREATED
    if name == "ACTION_CONVERSATION_SAVED":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import ACTION_CONVERSATION_SAVED

        return ACTION_CONVERSATION_SAVED
    if name == "_extract_target_id":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import _extract_target_id

        return _extract_target_id
    if name == "_result_is_success":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import _result_is_success

        return _result_is_success
    if name == "_TARGET_KEYS":
        warnings.warn(
            "gubbi.core.audit_decorator is deprecated; import from gubbi.audit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.audit.decorator import _TARGET_KEYS

        return _TARGET_KEYS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
