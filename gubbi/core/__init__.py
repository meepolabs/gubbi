"""Deprecation shim for ``gubbi.core`` -- CO.38 package split.

Every attribute formerly importable from :mod:`gubbi.core`,
:mod:`gubbi.core.audit_decorator`, etc. is re-exported here with a
:class:`~warnings.DeprecationWarning`.  All *production* code should use
the canonical paths documented below; the shim exists purely for backward
compatibility during the transition period.

Deprecated import paths and their replacements::

    gubbi.core.audit.Arch                  (no longer exists)
      -> gubbi.audit.Action
    gubbi.core.audit_decorator.audited     -> gubbi.audit.audited
    gubbi.core.audit_decorator._extract_target_id  -> gubbi.audit._extract_target_id
    gubbi.core.audit_decorator._result_is_success  -> gubbi.audit._result_is_success
    gubbi.core.context.AppContext          -> gubbi.app_context.AppContext
    gubbi.core.crypto.ContentCipher        -> gubbi.crypto.ContentCipher
    gubbi.core.crypto.DecryptionError      -> gubbi.crypto.DecryptionError
    gubbi.core.crypto.decrypt_or_raise     -> gubbi.crypto.decrypt_or_raise
    gubbi.crypto.crypto.load_master_keys_from_env -> gubbi.crypto.load_master_keys_from_env
    gubbi.core.cipher_guard.require_cipher -> gubbi.crypto.require_cipher
    gubbi.core.auth_context.current_user_id  -> gubbi.auth_context.current_user_id
    gubbi.core.auth_context.current_token_scopes -> gubbi.auth_context.current_token_scopes
    gubbi.core.auth_context.get_current_user_id   -> gubbi.auth_context.get_current_user_id
    gubbi.core.scope.*                     -> gubbi.auth.scope (any name)
    gubbi.core.validation.*                -> gubbi.validation (any name)
    gubbi.core.logger.initialize_logger    -> gubbi.telemetry.logger.initialize_logger

PEP 562: ``__getattr__`` fires on per-attribute access, NOT at import time.
This ensures that ``pytest --collect-only`` and similar non-executing
operations do not trigger DeprecationWarning spew.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

# All symbols formerly reachable via gubbi.core.* submodule paths.
_MAP: Mapping[str, str] = {
    # -- audit (was top-level gubbi/audit.py + core/audit_decorator.py) --
    "Action": "gubbi.audit",
    "record_audit": "gubbi.audit",
    "audited": "gubbi.audit",
    "_extract_target_id": "gubbi.audit",
    "_result_is_success": "gubbi.audit",
    # -- context (was core/context.py -> app_context.py) --
    "AppContext": "gubbi.app_context",
    # -- crypto (was core/crypto.py, cipher_guard.py -> crypto/ package) --
    "ContentCipher": "gubbi.crypto",
    "load_master_keys_from_env": "gubbi.crypto",
    "DecryptionError": "gubbi.crypto",
    "decrypt_or_raise": "gubbi.crypto",
    # -- cipher_guard (was core/cipher_guard.py -> crypto/guard.py) --
    "require_cipher": "gubbi.crypto",
    # -- auth_context (was core/auth_context.py -> gubbi/auth_context.py) --
    "current_user_id": "gubbi.auth_context",
    "current_token_scopes": "gubbi.auth_context",
    "get_current_user_id": "gubbi.auth_context",
    "AuthenticationError": "gubbi.auth_context",
    # -- scope (was core/scope.py -> gubbi/auth/scope.py) --
    "check_scope": "gubbi.auth.scope",
    "require_scope": "gubbi.auth.scope",
    "insufficient_scope_response": "gubbi.auth.scope",
    "SCOPE_DESCRIPTIONS": "gubbi.auth.scope",
    "SCOPE_GRANTS": "gubbi.auth.scope",
    # -- validation (was core/validation.py -> gubbi/validation.py) --
    # Individual validation functions
    "sanitize_label": "gubbi.validation",
    "sanitize_freetext": "gubbi.validation",
    "reject_tool_call_syntax": "gubbi.validation",
    "validate_topic": "gubbi.validation",
    "harden_llm_topic_path": "gubbi.validation",
    "validate_title": "gubbi.validation",
    "validate_date": "gubbi.validation",
    "local_today": "gubbi.validation",
    "is_future_date": "gubbi.validation",
    "slugify": "gubbi.validation",
    # -- logger (was core/logger.py -> telemetry/logger.py) --
    "initialize_logger": "gubbi.telemetry.logger",
    # -- auth_decorator action constants --
    "ACTION_ENTRY_CREATED": "gubbi.audit",
    "ACTION_ENTRY_UPDATED": "gubbi.audit",
    "ACTION_ENTRY_DELETED": "gubbi.audit",
    "ACTION_TOPIC_CREATED": "gubbi.audit",
    "ACTION_CONVERSATION_SAVED": "gubbi.audit",
    # -- audit_decorator internal --
    "_TARGET_KEYS": "gubbi.audit",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export with a ``DeprecationWarning``."""
    target_module = _MAP.get(name)
    if target_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"gubbi.core.{name} is deprecated; import from {target_module}.{name} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    import importlib

    mod = importlib.import_module(target_module)
    return getattr(mod, name)


def __dir__() -> list[str]:
    """Expose deprecated symbols so ``dir(gubbi.core)`` is useful."""
    return sorted(_MAP.keys())
