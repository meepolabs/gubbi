"""Unit tests for journalctl.audit -- record_audit() helper.

Tests use a mock asyncpg connection so no database is required.
All 13 documented action strings are exercised.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from journalctl.audit import Action, record_audit

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> AsyncMock:
    """Return an async mock that records conn.execute() calls."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    return conn


def _executed_args(conn: AsyncMock) -> tuple[object, ...]:
    """Return the positional args from the first conn.execute() call."""
    result: tuple[object, ...] = conn.execute.call_args[0]
    return result


# ---------------------------------------------------------------------------
# Happy-path: all 13 action strings
# ---------------------------------------------------------------------------

ALL_ACTIONS = [
    Action.IDENTITY_CREATED,
    Action.IDENTITY_DELETED,
    Action.IDENTITY_RESTORED,
    Action.TENANT_PROVISIONED,
    Action.TENANT_SUSPENDED,
    Action.TENANT_REACTIVATED,
    Action.LOGIN_FAILED,
    Action.SUBSCRIPTION_CREATED,
    Action.SUBSCRIPTION_CANCELED,
    Action.SUBSCRIPTION_OVERRIDE,
    Action.SECRET_ROTATED,
    Action.ADMIN_QUERY_EXECUTED,
    Action.ENCRYPTION_KEY_ROTATED,
]


@pytest.mark.parametrize("action_str", ALL_ACTIONS)
async def test_record_audit_inserts_for_each_action(action_str: str) -> None:
    # Arrange
    conn = _make_conn()

    # Act
    await record_audit(
        conn,
        actor_type="admin",
        actor_id="admin@example.com",
        action=action_str,
    )

    # Assert
    conn.execute.assert_called_once()
    args = _executed_args(conn)
    # args: (sql, actor_type, actor_id, action, target_type, target_id,
    #        reason, metadata_json, ip_address, user_agent)
    assert args[1] == "admin"
    assert args[2] == "admin@example.com"
    assert args[3] == action_str


# ---------------------------------------------------------------------------
# actor_type validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("actor_type", ["user", "admin", "system", "hydra_subject"])
async def test_valid_actor_types_accepted(actor_type: str) -> None:
    conn = _make_conn()
    await record_audit(conn, actor_type=actor_type, actor_id="x", action="user.created")
    conn.execute.assert_called_once()


async def test_invalid_actor_type_raises_value_error() -> None:
    conn = _make_conn()
    with pytest.raises(ValueError, match="Invalid actor_type"):
        await record_audit(conn, actor_type="hacker", actor_id="x", action="user.created")
    conn.execute.assert_not_called()


@pytest.mark.parametrize("bad", ["", "root", "ADMIN", "User", "superuser"])
async def test_various_invalid_actor_types(bad: str) -> None:
    conn = _make_conn()
    with pytest.raises(ValueError):
        await record_audit(conn, actor_type=bad, actor_id="x", action="user.created")
    conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# metadata defaults
# ---------------------------------------------------------------------------


async def test_metadata_defaults_to_empty_dict() -> None:
    # Arrange
    conn = _make_conn()

    # Act -- no metadata kwarg passed
    await record_audit(conn, actor_type="system", actor_id="worker-1", action="secret.rotated")

    # Assert
    args = _executed_args(conn)
    # metadata_json is the 7th positional arg after the SQL string (index 7)
    assert json.loads(str(args[7])) == {}


async def test_metadata_none_treated_as_empty_dict() -> None:
    conn = _make_conn()
    await record_audit(
        conn, actor_type="system", actor_id="w", action="secret.rotated", metadata=None
    )
    args = _executed_args(conn)
    assert json.loads(str(args[7])) == {}


async def test_metadata_dict_is_serialized() -> None:
    conn = _make_conn()
    meta = {"secret": "encryption_master_key", "version": "v2"}
    await record_audit(
        conn, actor_type="admin", actor_id="ops", action="encryption.key_rotated", metadata=meta
    )
    args = _executed_args(conn)
    assert json.loads(str(args[7])) == meta


# ---------------------------------------------------------------------------
# Optional fields are nullable
# ---------------------------------------------------------------------------


async def test_all_optional_fields_none_by_default() -> None:
    # Arrange
    conn = _make_conn()

    # Act -- only required args
    await record_audit(conn, actor_type="user", actor_id="uuid-123", action="user.created")

    # Assert
    args = _executed_args(conn)
    # target_type=args[5], target_id=args[6], reason=args[8 - wait, recount]
    # SQL positional: $1=actor_type $2=actor_id $3=action $4=target_type
    #                 $5=target_id $6=reason $7=metadata $8=ip_address $9=user_agent
    # args[0]=sql, [1]=actor_type, [2]=actor_id, [3]=action, [4]=target_type,
    #         [5]=target_id, [6]=reason, [7]=metadata_json, [8]=ip_address, [9]=user_agent
    assert args[4] is None  # target_type
    assert args[5] is None  # target_id
    assert args[6] is None  # reason
    assert args[8] is None  # ip_address
    assert args[9] is None  # user_agent


async def test_optional_fields_passed_through() -> None:
    conn = _make_conn()
    await record_audit(
        conn,
        actor_type="hydra_subject",
        actor_id="11111111-2222-3333-4444-555555555555",
        action="auth.email_collision",
        target_type="user",
        target_id="uuid-abc",
        reason="support investigation",
        metadata={"ticket": "CS-9001"},
        ip_address="203.0.113.42",
        user_agent="Mozilla/5.0",
    )
    args = _executed_args(conn)
    assert args[1] == "hydra_subject"
    assert args[2] == "11111111-2222-3333-4444-555555555555"
    assert args[3] == "auth.email_collision"
    assert args[4] == "user"
    assert args[5] == "uuid-abc"
    assert args[6] == "support investigation"
    assert json.loads(str(args[7])) == {"ticket": "CS-9001"}
    assert args[8] == "203.0.113.42"
    assert args[9] == "Mozilla/5.0"


# ---------------------------------------------------------------------------
# Action constants presence check
# ---------------------------------------------------------------------------


def test_action_constants_count() -> None:
    """All 13 documented actions are present in ALL_ACTIONS.

    Restored to the 0012 count of 13; had dropped to 12 when
    ``auth.founder_impersonation`` was removed along with the ``founder``
    actor_type (single-tenant-era, zero call sites). LOGIN_FAILED added in
    TASK-03.22a brings it back to 13.
    """
    assert len(ALL_ACTIONS) == 13


def test_action_constants_are_strings() -> None:
    for action in ALL_ACTIONS:
        assert isinstance(action, str), f"Expected str, got {type(action)} for {action!r}"


# ---------------------------------------------------------------------------
# Action constant value checks
# ---------------------------------------------------------------------------


def test_tenant_provisioned_value() -> None:
    assert Action.TENANT_PROVISIONED == "tenant.provisioned"


def test_login_failed_value() -> None:
    assert Action.LOGIN_FAILED == "login_failed"
