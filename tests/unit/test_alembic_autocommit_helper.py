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
    """Stand-in for the SQLAlchemy connection returned by ``op.get_bind()``."""

    def __init__(self, raw: _FakeRawConn) -> None:
        self.connection = raw


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
