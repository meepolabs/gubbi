"""Unit tests for ``gubbi.audit`` -- the re-export shim.

As of gubbi-common 0.11.0 (A3 consolidation) ``record_audit`` is a
re-export of ``gubbi_common.audit.sql.record_audit_async``. The deep
validation behaviour (actor_type / actor_id / target_id shape, banned-key
metadata redaction, IP normalisation, metadata size cap, ``audit.write``
OTel span) is exercised in
``gubbi-common/tests/audit/test_sql.py``. These tests guard the
re-export contract -- if a future change reintroduces a local ``record_audit``
shim, the assertions here will surface the divergence.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from gubbi.audit import Action, record_audit
from tests.conftest import InMemoryExporter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> AsyncMock:
    """Return an async mock that records conn.execute() calls."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    return conn


# ---------------------------------------------------------------------------
# Re-export shape
# ---------------------------------------------------------------------------


def test_record_audit_is_gubbi_common_record_audit_async() -> None:
    """``record_audit`` MUST be the canonical gubbi-common helper.

    Regression guard: a future change that reintroduces a local
    ``record_audit`` (with its own validation surface or its own INSERT
    SQL) would diverge silently from the canonical writer. The S2
    HIGH-1 / S2 MEDIUM findings were exactly that kind of drift.
    """
    from gubbi_common.audit.sql import record_audit_async

    assert record_audit is record_audit_async


def test_record_audit_is_keyword_only_after_conn() -> None:
    sig = inspect.signature(record_audit)
    params = list(sig.parameters.values())
    assert params[0].name == "conn"
    for p in params[1:]:
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"param {p.name!r} should be keyword-only -- callers in gubbi/audit "
            "pass everything after conn by keyword"
        )


def test_action_enum_re_exported() -> None:
    """``Action`` is re-exported from gubbi-common."""
    from gubbi_common.audit.actions import Action as _UpstreamAction

    assert Action is _UpstreamAction


# ---------------------------------------------------------------------------
# Smoke -- exercise the canonical INSERT through the re-export
# ---------------------------------------------------------------------------


async def test_record_audit_writes_target_kind_via_canonical_insert() -> None:
    """End-to-end smoke: writing through the gubbi re-export persists target_kind.

    Closes the historical S2 MEDIUM gap where gubbi's local 10-column
    INSERT was the only path that captured target_kind. With the
    re-export, ``record_audit`` and ``record_audit_async`` produce the
    same INSERT shape.
    """
    conn = _make_conn()
    await record_audit(
        conn,
        actor_type="user",
        actor_id="00000000-0000-0000-0000-000000000001",
        action=Action.IDENTITY_CREATED,
        target_type="user",
        target_id="00000000-0000-0000-0000-000000000042",
        target_kind="user",
        metadata={"via": "test"},
    )
    conn.execute.assert_called_once()
    sql, *args = conn.execute.call_args[0]
    assert "target_kind" in sql
    # Layout: $1=actor_type $2=actor_id $3=action $4=target_type $5=target_id
    #         $6=target_kind $7=reason $8=metadata $9=ip $10=user_agent
    assert args[5] == "user"


async def test_record_audit_propagates_on_db_error() -> None:
    """Re-exported helper propagates DB errors (best-effort is the decorator's job)."""
    conn = _make_conn()
    conn.execute.side_effect = asyncpg.PostgresError("insert failed")
    with pytest.raises(asyncpg.PostgresError):
        await record_audit(
            conn,
            actor_type="user",
            actor_id="00000000-0000-0000-0000-000000000001",
            action="entry.created",
        )


# ---------------------------------------------------------------------------
# A3 Q1: audit.write OTel span emitted by the canonical writer
# ---------------------------------------------------------------------------


async def test_record_audit_emits_audit_write_span_with_gubbi_attrs(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    """Regression: the canonical writer emits ``audit.write`` carrying the gubbi attr shape.

    The OTel span lives inside ``record_audit_async`` (gubbi-common
    0.13.x). gubbi-common's ``_AUDIT_WRITE_ALLOWLIST`` deliberately
    excludes ``actor_id``, ``target_id``, and ``target_kind`` -- those
    remain durable in the audit_log row's columns; the span attribute
    projection drops them so external IDs (e.g. payment-provider
    subscription identifiers in target_id) do not reach the OTel
    trace pipeline. This test asserts the surviving attrs land on the
    span AND that the dropped keys do NOT.

    The ``in_memory_tracer`` fixture (tests/conftest.py) swaps the global
    TracerProvider with an in-memory exporter and resets the OTel
    ``_TRACER_PROVIDER_SET_ONCE`` guard so this test passes both in
    isolation and in the full suite. ``record_audit_async`` pulls a
    tracer named "gubbi_common.audit" off the global provider on each
    call, so the swap must be global, not local.
    """
    _, exporter = in_memory_tracer

    conn = _make_conn()
    await record_audit(
        conn,
        actor_type="user",
        actor_id="00000000-0000-0000-0000-000000000001",
        action="entry.created",
        target_type="entry",
        target_id="00000000-0000-0000-0000-000000000099",
        target_kind="entry",
    )

    spans = exporter.spans
    assert spans, "expected at least one finished span from record_audit_async"
    audit_spans = [s for s in spans if s.name == "audit.write"]
    assert audit_spans, f"expected an 'audit.write' span; got names: {[s.name for s in spans]}"
    attrs = dict(audit_spans[0].attributes or {})
    assert attrs.get("event_type") == "entry.created"
    assert attrs.get("actor_type") == "user"
    # latency_ms + success are set in the finally block; they MUST be present.
    assert "latency_ms" in attrs
    assert attrs.get("success") is True
    # actor_id / target_id / target_kind are intentionally excluded from
    # the span allowlist (they remain durable in the audit_log columns).
    assert "actor_id" not in attrs
    assert "target_id" not in attrs
    assert "target_kind" not in attrs
