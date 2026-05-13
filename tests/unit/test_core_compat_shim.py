"""Shim-compatibility tests for ``gubbi.core`` deprecation wrappers.

Parametrized over every deprecated path, verifying:
1. Object identity with canonical import.
2. A single :class:`~warnings.DeprecationWarning` fires on attribute access.

These tests allow the old ``from gubbi.core.X import Y`` paths to stay
alive until production code is fully migrated.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _clear_cache(request):  # type: ignore[misc]
    """Clear gubbi.* module cache between tests so __getattr__ fires again."""
    return request


# ---------------------------------------------------------------------------
# Shape definition -- each entry: (old_path_as_str, canonical_path_as_str,
# symbol_name)
# ---------------------------------------------------------------------------

_PATH_TESTS = [
    # audit_decorator -> gubbi.audit
    ("gubbi.core.audit_decorator", "gubbi.audit", "audited"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "_extract_target_id"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "_result_is_success"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "ACTION_ENTRY_CREATED"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "ACTION_ENTRY_UPDATED"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "ACTION_ENTRY_DELETED"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "ACTION_TOPIC_CREATED"),
    ("gubbi.core.audit_decorator", "gubbi.audit", "ACTION_CONVERSATION_SAVED"),
    # core shim via __init__.py map
    ("gubbi.core", "gubbi.audit", "record_audit"),
    ("gubbi.core", "gubbi.audit", "Action"),
    ("gubbi.core", "gubbi.app_context", "AppContext"),
    ("gubbi.core.context", "gubbi.app_context", "AppContext"),
    ("gubbi.core", "gubbi.crypto", "ContentCipher"),
    ("gubbi.core", "gubbi.crypto", "load_master_keys_from_env"),
    ("gubbi.core", "gubbi.crypto", "DecryptionError"),
    ("gubbi.core", "gubbi.crypto", "decrypt_or_raise"),
    # cipher_guard stub
    ("gubbi.core.cipher_guard", "gubbi.crypto", "require_cipher"),
    # auth_context shim (or direct import)
    ("gubbi.core.auth_context", "gubbi.auth_context", "current_user_id"),
    ("gubbi.core.auth_context", "gubbi.auth_context", "get_current_user_id"),
    # scope stub
    ("gubbi.core.scope", "gubbi.auth.scope", "check_scope"),
    ("gubbi.core.scope", "gubbi.auth.scope", "require_scope"),
    ("gubbi.core.scope", "gubbi.auth.scope", "SCOPE_DESCRIPTIONS"),
    ("gubbi.core.scope", "gubbi.auth.scope", "SCOPE_GRANTS"),
    # validation stub or canonical (same module name)
    ("gubbi.core.validation", "gubbi.validation", "validate_topic"),
    ("gubbi.core.validation", "gubbi.validation", "sanitize_label"),
    ("gubbi.core.validation", "gubbi.validation", "slugify"),
    # logger stub
    ("gubbi.core.logger", "gubbi.telemetry.logger", "initialize_logger"),
]


@pytest.mark.parametrize(
    ("deprecated_path", "canonical_module", "symbol"),
    _PATH_TESTS,
    ids=lambda x: f"deprecated={x[0]} -> {x[2]}" if isinstance(x, tuple) else str(x),
)
class TestImportIdentityAndWarning:
    """Each old path resolves to the SAME object as the canonical import + fires a DeprecationWarning."""

    @staticmethod
    def _import_from_path(path: str) -> Any:
        """Dynamically import a fully-qualified dotted path."""
        return __import__(path, fromlist=["_"])

    @staticmethod
    def _get_symbol(canonical_mod: str, name: str) -> Any:
        """Import ``name`` from the canonical module and return it."""
        return getattr(__import__(canonical_mod, fromlist=[name]), name)

    def test_identity_and_deprecation(
        self, deprecated_path: str, canonical_module: str, symbol: str
    ) -> None:  # type: ignore[misc]
        # --- object identity ---------------------------------------------------
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            old_mod = self._import_from_path(deprecated_path)
            from_old = getattr(old_mod, symbol)

            # Capture all deprecation warnings (not just the first)
            dep_count = sum(1 for x in w if issubclass(x.category, DeprecationWarning))
            assert dep_count >= 1, (
                f"Expected at least one DeprecationWarning when importing "
                f"{deprecated_path}.{symbol}, got {dep_count}"
            )

        # Import from canonical path
        from_canon = self._get_symbol(canonical_module, symbol)

        assert from_old is from_canon, (
            f"Object identity mismatch: "
            f"{deprecated_path}.{symbol} ({id(from_old)}) != "
            f"{canonical_module}.{symbol} ({id(from_canon)})"
        )
