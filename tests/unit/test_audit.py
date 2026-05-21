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
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from gubbi.audit import Action, record_audit

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


async def test_record_audit_emits_audit_write_span_with_gubbi_attrs() -> None:
    """Regression: the canonical writer emits ``audit.write`` carrying the gubbi attr shape.

    Per A3 Q1 the OTel span moved INTO ``record_audit_async`` in
    gubbi-common 0.11.0. The gubbi-side allowlist for ``audit.write``
    keys on ``{"event_type", "target_id", "actor_type", "success",
    "latency_ms"}``; this test asserts the canonical writer sets at
    least the three pre-execute attrs (``event_type``, ``actor_type``,
    ``target_id``) on the started span.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Swap the global tracer provider for the duration of this test;
    # record_audit_async pulls a tracer named "gubbi_common.audit" off
    # the global provider on each call.
    original = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = None
    trace.set_tracer_provider(provider)
    try:
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

        spans = exporter.get_finished_spans()
        assert spans, "expected at least one finished span from record_audit_async"
        audit_spans = [s for s in spans if s.name == "audit.write"]
        assert audit_spans, f"expected an 'audit.write' span; got names: {[s.name for s in spans]}"
        attrs = dict(audit_spans[0].attributes or {})
        assert attrs.get("event_type") == "entry.created"
        assert attrs.get("actor_type") == "user"
        assert attrs.get("target_id") == "00000000-0000-0000-0000-000000000099"
        # latency_ms + success are set in the finally block; they MUST be present.
        assert "latency_ms" in attrs
        assert attrs.get("success") is True
    finally:
        trace._TRACER_PROVIDER = None
        trace.set_tracer_provider(original)
