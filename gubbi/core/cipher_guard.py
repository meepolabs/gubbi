"""Cipher-presence guard for tool-layer entry points (TASK-02.13).

``AppContext.cipher`` is ``ContentCipher | None`` because the lifespan
accepts a dev config with no master key. Any MCP tool handler that writes
or reads encrypted content must call :func:`require_cipher` at the top of
its body, BEFORE any DB work, so a misconfigured deploy fails fast with a
clear error instead of silently writing plaintext or blowing up mid-query.

This replaces a would-be ``JOURNAL_REQUIRE_ENCRYPTION`` feature flag: the
presence of an app-level cipher IS the flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gubbi.core.context import AppContext
    from gubbi.core.crypto import ContentCipher


def require_cipher(app_ctx: AppContext) -> ContentCipher:
    """Return ``app_ctx.cipher`` or raise if it is ``None``.

    Raises
    ------
    RuntimeError
        When ``app_ctx.cipher is None``. The message is intentionally
        short and points at the env var so operators fix the deploy
        rather than debug the tool layer.
    """
    cipher = app_ctx.cipher
    if cipher is None:
        raise RuntimeError(
            "encryption cipher required but not configured -- "
            "set JOURNAL_ENCRYPTION_MASTER_KEY_V1 in the runtime environment"
        )
    return cipher
