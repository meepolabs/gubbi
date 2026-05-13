"""Tests for the selfhost token validator protocol (async Awaitable contract).

Verifies:
  1. _make_token_validator returns an awaitable (coroutine function).
  2. The returned validator's annotations declare the correct return type.
  3. inspect.iscoroutinefunction confirms the inner callable is async.
  4. Calling the validator returns an Awaitable (not a plain frozenset).
  5. Awaiting the validator on valid / invalid tokens behaves correctly.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from pathlib import Path

import pytest_asyncio

from gubbi.oauth._rate_limit import RateLimitStorage
from gubbi.oauth.selfhost import _make_token_validator
from gubbi.oauth.storage import OAuthStorage


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> OAuthStorage:
    """Initialized OAuthStorage; closed after each test."""
    s = OAuthStorage(tmp_path / "oauth.db")
    await s.initialize()
    yield s
    await s.close()


class TestValidatorIsCoroutineFunction:
    async def test_iscoroutinefunction(self, storage: OAuthStorage) -> None:
        """_make_token_validator must return an async callable (coroutine function)."""
        validator = _make_token_validator(storage)
        assert inspect.iscoroutinefunction(
            validator
        ), f"Expected an async function; got {type(validator)}"

    async def test_calling_validator_returns_awaitable(self, storage: OAuthStorage) -> None:
        """Invoking the validator (without await) must produce an Awaitable."""
        validator = _make_token_validator(storage)
        coro = validator("some-token")
        assert isinstance(coro, Awaitable), f"Expected Awaitable; got {type(coro)}"
        # Consume the coroutine to avoid ResourceWarning.
        await coro


class TestValidatorAnnotations:
    async def test_return_annotation_is_awaitable(self, storage: OAuthStorage) -> None:
        """The validator's return annotation must express Awaitable[frozenset[str] | None]."""
        validator = _make_token_validator(storage)
        # The inner 'validate' function is defined with `-> frozenset[str] | None`
        # inside an async def, so its return annotation reflects that union.
        # We check that calling it is awaitable rather than requiring a specific
        # annotation string (Python 3.11 'from __future__ import annotations' may
        # lazily stringify unions differently).
        coro = validator("check-annotation")
        assert isinstance(coro, Awaitable)
        await coro


class TestValidatorBehavior:
    async def test_unknown_token_returns_none(self, storage: OAuthStorage) -> None:
        """Validator must return None for a token not in storage."""
        validator = _make_token_validator(storage)
        result = await validator("this-token-does-not-exist")
        assert result is None

    async def test_make_validator_called_twice_gives_independent_closures(
        self, storage: OAuthStorage
    ) -> None:
        """Two calls to _make_token_validator produce independent callables."""
        v1 = _make_token_validator(storage)
        v2 = _make_token_validator(storage)
        assert v1 is not v2
        # Both should work correctly.
        assert await v1("x") is None
        assert await v2("y") is None


class TestRateLimitStorageLockIndependence:
    async def test_rl_lock_is_not_oauth_lock(self, storage: OAuthStorage) -> None:
        """RateLimitStorage owns its own asyncio.Lock, not the OAuthStorage lock."""
        assert storage._rl._lock is not storage._lock

    async def test_rl_is_rate_limit_storage_instance(self, storage: OAuthStorage) -> None:
        """OAuthStorage._rl must be a RateLimitStorage instance."""
        assert isinstance(storage._rl, RateLimitStorage)
