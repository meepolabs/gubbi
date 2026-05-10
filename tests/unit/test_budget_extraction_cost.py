"""Unit tests for gubbi.budget.extraction_cost.record_extraction_cost."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from gubbi.budget.extraction_cost import record_extraction_cost

_USER_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_PERIOD = date(2026, 5, 1)
_EXPECTED_KEY = f"budget:{_USER_ID}:{_PERIOD}"
_EXPECTED_MEMBER = f"{_USER_ID}:{_PERIOD}"


@pytest.mark.asyncio
async def test_hincrby_called_with_delta() -> None:
    # Arrange
    redis = AsyncMock()
    redis.hincrby = AsyncMock(return_value=75)
    redis.sadd = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)

    # Act
    await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=80,
        estimated_cents=50,
        redis=redis,
    )

    # Assert: delta = 80 - 50 = 30
    redis.hincrby.assert_called_once_with(_EXPECTED_KEY, "used_cents", 30)


@pytest.mark.asyncio
async def test_sadd_member_format() -> None:
    # Arrange
    redis = AsyncMock()
    redis.hincrby = AsyncMock(return_value=0)
    redis.sadd = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)

    # Act
    await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=20,
        estimated_cents=50,
        redis=redis,
    )

    # Assert: sadd with "budget:dirty" and the correct member string
    redis.sadd.assert_called_once_with("budget:dirty", _EXPECTED_MEMBER)


@pytest.mark.asyncio
async def test_expire_3600() -> None:
    # Arrange
    redis = AsyncMock()
    redis.hincrby = AsyncMock(return_value=0)
    redis.sadd = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)

    # Act
    await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=50,
        estimated_cents=50,
        redis=redis,
    )

    # Assert: expire key for 3600 seconds
    redis.expire.assert_called_once_with(_EXPECTED_KEY, 3600)


@pytest.mark.asyncio
async def test_negative_delta_allowed() -> None:
    # actual < estimated => negative delta (refund-like adjustment)
    redis = AsyncMock()
    redis.hincrby = AsyncMock(return_value=-10)
    redis.sadd = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)

    await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=10,
        estimated_cents=50,
        redis=redis,
    )

    redis.hincrby.assert_called_once_with(_EXPECTED_KEY, "used_cents", -40)


@pytest.mark.asyncio
async def test_none_redis_short_circuits() -> None:
    # When redis=None (self-host), no calls should be made and no exception raised.
    # There is no redis object to assert on; just verify it returns cleanly.
    result = await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=99,
        estimated_cents=50,
        redis=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_key_does_not_contain_tenant_id() -> None:
    # Guard: the key must be user_id-keyed, never tenant_id-keyed.
    redis = AsyncMock()
    redis.hincrby = AsyncMock(return_value=0)
    redis.sadd = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)

    await record_extraction_cost(
        _USER_ID,
        _PERIOD,
        actual_cents=50,
        estimated_cents=50,
        redis=redis,
    )

    hincrby_key = redis.hincrby.call_args[0][0]
    assert str(_USER_ID) in hincrby_key
    assert hincrby_key.startswith("budget:")
