"""Unit tests for the alembic autocommit helper.

The helper wraps the psycopg autocommit dance used by ``CREATE INDEX
CONCURRENTLY`` migrations. The contract under test:

    1. On entry, the underlying psycopg connection has ``autocommit = True``.
    2. The yielded value IS the underlying psycopg connection.
    3. On normal exit, ``autocommit`` is restored to its prior value.
    4. On exception inside the block, ``autocommit`` is still restored
       (the exception is allowed to propagate).
    5. Restoration uses the captured prior value -- not a hard-coded ``False``
       -- so a future psycopg version starting in a different mode does not
       silently flip behaviour.
    6. The helper resolves the underlying driver connection through the
       SQLAlchemy ``_ConnectionFairy`` (PoolProxiedConnection): writes to
       ``autocommit`` must land on the DRIVER, NOT on the proxy. This is
       the critical fix -- the fairy does not forward ``__setattr__``,
       so writing through the proxy silently never reaches psycopg.
"""

from typing import Any

import pytest

from gubbi.alembic._helpers import autocommit_block

pytestmark = pytest.mark.unit


class _FakeRawConn:
    """Stand-in for a psycopg.Connection with the autocommit attribute."""

    def __init__(self, initial: bool = False) -> None:
        self.autocommit: bool = initial
        self.history: list[bool] = [initial]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "autocommit":
            super().__setattr__(name, value)
            history = getattr(self, "history", None)
            if history is not None:
                history.append(value)
        else:
            super().__setattr__(name, value)


class _FakeBind:
    """Stand-in for the SQLAlchemy connection returned by ``op.get_bind()``.

    This shape passes the raw psycopg connection directly via ``.connection``
    -- it does NOT model the real ``_ConnectionFairy`` proxy. The helper's
    ``or proxy`` fallback path is what makes these tests work, and they
    remain useful as a regression net for that fallback.
    """

    def __init__(self, raw: _FakeRawConn) -> None:
        self.connection = raw


# ---------------------------------------------------------------------------
# Original five cases -- exercise the ``or proxy`` fallback path explicitly.
# Kept because they lock the helper's behaviour when a caller passes a raw
# psycopg-shaped object (e.g. tests, or a non-pooled SQLAlchemy build).
# ---------------------------------------------------------------------------


def test_autocommit_block_sets_true_on_entry_and_restores_on_exit() -> None:
    raw = _FakeRawConn(initial=False)
    bind = _FakeBind(raw)

    with autocommit_block(bind) as yielded:
        assert yielded is raw
        assert raw.autocommit is True

    assert raw.autocommit is False


def test_autocommit_block_restores_prior_value_not_false() -> None:
    # A future psycopg version (or an unusual caller) might start the
    # connection in autocommit=True. The helper must restore THAT value,
    # not blindly reset to False.
    raw = _FakeRawConn(initial=True)
    bind = _FakeBind(raw)

    with autocommit_block(bind) as yielded:
        assert yielded is raw
        assert raw.autocommit is True

    assert raw.autocommit is True


def test_autocommit_block_restores_on_exception() -> None:
    raw = _FakeRawConn(initial=False)
    bind = _FakeBind(raw)

    def _raises() -> None:
        with autocommit_block(bind):
            assert raw.autocommit is True
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _raises()

    # Exception propagated, autocommit still restored.
    assert raw.autocommit is False


def test_autocommit_block_restores_prior_value_on_exception() -> None:
    raw = _FakeRawConn(initial=True)
    bind = _FakeBind(raw)

    def _raises() -> None:
        with autocommit_block(bind):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _raises()

    assert raw.autocommit is True


def test_autocommit_block_history_shape_on_normal_exit() -> None:
    # Sanity check: exactly two transitions -- entry flips False -> True,
    # exit flips True -> False (the prior value).
    raw = _FakeRawConn(initial=False)
    bind = _FakeBind(raw)

    with autocommit_block(bind):
        pass

    # history starts with initial value, then records each setattr.
    assert raw.history == [False, True, False]


# ---------------------------------------------------------------------------
# Proxy-shaped cases (critical fix).
#
# These mirror SQLAlchemy's ``_ConnectionFairy`` (PoolProxiedConnection):
# the proxy exposes ``driver_connection`` / ``dbapi_connection`` accessors
# but does NOT define ``__setattr__`` to forward writes. So any code that
# writes ``proxy.autocommit = True`` lands on the proxy itself and never
# reaches psycopg. The helper must resolve the driver before writing.
# ---------------------------------------------------------------------------


class _FakeProxyFairy:
    """Mirrors SQLAlchemy ``_ConnectionFairy``: no ``__setattr__`` forwarding.

    Reads of unknown attributes are forwarded to the driver (matching the
    real fairy's ``__getattr__``), but writes land on the proxy instance.
    The whole point of this fake is to make the bug observable: if the
    helper wrote to ``proxy.autocommit`` directly, the driver's
    ``autocommit`` would never change and the fairy would silently grow
    a new attribute.
    """

    def __init__(self, driver: _FakeRawConn) -> None:
        # Use the base ``object`` setattr to bypass any descriptor games;
        # we want this attribute set on the instance directly.
        object.__setattr__(self, "_driver", driver)

    @property
    def driver_connection(self) -> _FakeRawConn:
        return self._driver  # type: ignore[no-any-return]

    @property
    def dbapi_connection(self) -> _FakeRawConn:
        return self._driver  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        # Forward UNKNOWN reads to driver, like the real fairy. Writes are
        # NOT forwarded -- there's no ``__setattr__`` override here, so
        # ``proxy.autocommit = X`` would land on the proxy instance.
        return getattr(self._driver, name)


class _FakeBindWithProxy:
    """``op.get_bind()`` shape that holds a fairy proxy on ``.connection``."""

    def __init__(self, driver: _FakeRawConn) -> None:
        self.connection = _FakeProxyFairy(driver)


def test_proxy_shape_autocommit_true_lands_on_driver_not_proxy() -> None:
    # Critical case: writing to the proxy directly would silently fail
    # to flip the driver. The helper must resolve ``driver_connection``.
    driver = _FakeRawConn(initial=False)
    bind = _FakeBindWithProxy(driver)
    proxy = bind.connection

    with autocommit_block(bind) as yielded:
        assert yielded is driver
        # The DRIVER flipped to True.
        assert driver.autocommit is True
        # The PROXY's own attribute namespace was NEVER touched -- it
        # only reads ``autocommit`` through __getattr__ forwarding to the
        # driver, never as an own attribute.
        assert "autocommit" not in proxy.__dict__

    assert driver.autocommit is False


def test_proxy_shape_restores_driver_autocommit_on_exit() -> None:
    # The DRIVER's prior value (True here) is restored, not the proxy's.
    driver = _FakeRawConn(initial=True)
    bind = _FakeBindWithProxy(driver)

    with autocommit_block(bind) as yielded:
        assert yielded is driver
        assert driver.autocommit is True

    assert driver.autocommit is True
    # And the driver actually saw both writes recorded -- if the helper
    # had written to the proxy by mistake, the driver's history would
    # still be just [True].
    assert driver.history == [True, True, True]


def test_proxy_shape_restores_driver_autocommit_on_exception() -> None:
    driver = _FakeRawConn(initial=False)
    bind = _FakeBindWithProxy(driver)
    proxy = bind.connection

    def _raises() -> None:
        with autocommit_block(bind):
            assert driver.autocommit is True
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _raises()

    # DRIVER restored, exception propagated.
    assert driver.autocommit is False
    # Proxy's own dict still untouched.
    assert "autocommit" not in proxy.__dict__


def test_proxy_shape_proxy_attribute_never_written() -> None:
    # Defensive lock: if a future change reverts to writing on the proxy,
    # this assertion fires loudly.
    driver = _FakeRawConn(initial=False)
    bind = _FakeBindWithProxy(driver)
    proxy = bind.connection

    with autocommit_block(bind):
        pass

    # No own ``autocommit`` attribute on the proxy at any point.
    assert "autocommit" not in proxy.__dict__
    # Driver's history records the two transitions: False -> True -> False.
    assert driver.history == [False, True, False]
