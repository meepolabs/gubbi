"""Boundary checks: admin functions reject non-admin pools (BYPASSRLS contract).

The ``_run_reindex`` and ``_reset_all_indexed_at`` admin actions issue
cross-tenant UPDATEs (``mark_indexed_batch``, ``reset_indexed_at_for_ids``,
``reset_indexed_at``). A user-scoped (RLS-enforced) pool would silently
restrict the rowcount to the current ``app.current_user_id`` GUC and
strand rows in claimed-but-never-processed states.

The contract used to be enforced with ``assert``, which Python strips
under ``-O``. Replaced with explicit ``raise ValueError`` so the guard
runs unconditionally. These tests pin the new behaviour.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gubbi.app_context import AppContext
from gubbi.tools.admin import _reset_all_indexed_at, _run_reindex


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_reindex_raises_value_error_when_admin_pool_is_none() -> None:
    """``admin_pool=None`` must raise ``ValueError`` (not pass silently).

    Pre-fix: ``assert admin_pool is app_ctx.admin_pool`` evaluated to
    ``None is None == True`` in single-tenant dev mode (where
    ``app_ctx.admin_pool`` is also ``None``), masking the contract.
    """
    app_ctx = MagicMock(spec=AppContext)
    app_ctx.admin_pool = None
    cipher = MagicMock()

    with pytest.raises(ValueError, match="BYPASSRLS"):
        await _run_reindex(app_ctx, None, cipher)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_reindex_raises_value_error_when_pool_mismatch() -> None:
    """A pool that is not ``app_ctx.admin_pool`` must raise ``ValueError``."""
    app_ctx = MagicMock(spec=AppContext)
    real_pool = MagicMock()
    other_pool = MagicMock()
    app_ctx.admin_pool = real_pool
    cipher = MagicMock()

    with pytest.raises(ValueError, match="BYPASSRLS"):
        await _run_reindex(app_ctx, other_pool, cipher)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_all_indexed_at_requires_app_ctx_argument() -> None:
    """``_reset_all_indexed_at`` makes ``app_ctx`` mandatory (no default).

    Pre-fix the signature was ``app_ctx: AppContext | None = None``, and the
    guard ``app_ctx is not None and admin_pool is not app_ctx.admin_pool``
    short-circuited to ``False`` whenever a caller passed an RLS-scoped pool
    with ``app_ctx=None``, silently bypassing the BYPASSRLS contract. Making
    ``app_ctx`` a required positional argument removes the escape hatch.
    """
    pool = MagicMock()
    with pytest.raises(TypeError):
        # Missing the now-required ``app_ctx`` -- must raise at call time.
        await _reset_all_indexed_at(pool)  # type: ignore[call-arg]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_all_indexed_at_raises_value_error_when_pool_is_none() -> None:
    """``_reset_all_indexed_at`` rejects ``None`` admin pool with ``ValueError``."""
    app_ctx = MagicMock(spec=AppContext)
    app_ctx.admin_pool = None

    with pytest.raises(ValueError, match="BYPASSRLS"):
        await _reset_all_indexed_at(None, app_ctx)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_all_indexed_at_raises_value_error_on_pool_mismatch() -> None:
    """A pool that is not ``app_ctx.admin_pool`` must raise ``ValueError``."""
    app_ctx = MagicMock(spec=AppContext)
    real_pool = MagicMock()
    other_pool = MagicMock()
    app_ctx.admin_pool = real_pool

    with pytest.raises(ValueError, match="BYPASSRLS"):
        await _reset_all_indexed_at(other_pool, app_ctx)
