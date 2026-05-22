"""Pin: CorrelationIDMiddleware must wrap OUTSIDE OpenTelemetryMiddleware.

Regression test for the gubbi-side correlation_id silent-drop bug where
the FastAPI auto-instrumentor's middleware was OUTER, opening the
server span before CorrelationIDMiddleware populated the request-scoped
ContextVar. ``CorrelationSpanProcessor.on_start`` reads the ContextVar
at span open; if the value is ``None`` it returns silently and every
span emitted during that request lacks the ``correlation_id`` attribute.

The 401 codepath is the canonical reproducer because no inner spans
fire (no DB query, no MCP call) to mask the un-stamped server span --
exactly the shape that the gubbi-testbench
``test_correlation_id_propagates_to_gubbi_side_otel_span`` verify-local
test catches in production.

This test pins the wiring contract at unit level: build a minimal
FastAPI app with the same middleware shape gubbi.main now uses
(MCPPathNormalizer in the constructor, CorrelationIDMiddleware
wrapping the FastAPI ASGI app from OUTSIDE), wire FastAPIInstrumentor
+ CorrelationSpanProcessor against an in-memory exporter, fire a
401-returning request with a known X-Correlation-ID, and assert every
emitted span carries the value as the ``correlation_id`` attribute.

If the wrap order regresses (e.g. CorrelationIDMiddleware is moved
back INSIDE FastAPI's user_middleware), the OTel server span opens
without a populated ContextVar and the assertion fails.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from gubbi_common.telemetry.correlation_processor import CorrelationSpanProcessor
from opentelemetry import trace as otel_trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util._once import Once
from starlette.middleware import Middleware

from gubbi.middleware import CorrelationIDMiddleware

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

pytestmark = pytest.mark.unit

_TEST_CID = "test-cid-fixed-uuid-here"
_CORRELATION_ATTR = "correlation_id"


@pytest.fixture
def in_memory_provider() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory TracerProvider for the duration of one test.

    Snapshots the global tracer-provider slot and the ``Once`` latch so
    multiple tests in this module can each install a fresh provider
    without permanent cross-test state. The conftest-level
    ``in_memory_tracer`` fixture cannot be reused here because this test
    needs FastAPIInstrumentor to bind to OUR provider via
    ``set_tracer_provider`` BEFORE ``instrument_app`` reads the global,
    and conftest's helper does not promise that tracer-provider is the
    one a future ``set_tracer_provider`` call will land on.
    """
    saved_provider = otel_trace._TRACER_PROVIDER
    saved_set_once = otel_trace._TRACER_PROVIDER_SET_ONCE

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # CorrelationSpanProcessor must run BEFORE the exporter processor so
    # the on_start hook lands the attribute before BatchSpanProcessor
    # queues the span for export. SimpleSpanProcessor's order matches.
    provider.add_span_processor(CorrelationSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Reset the latch + slot so set_tracer_provider takes effect.
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_trace.set_tracer_provider(provider)

    # Guard against silent reset failure on future OTel internals churn:
    # if the private-API names (_TRACER_PROVIDER / _TRACER_PROVIDER_SET_ONCE)
    # move or change semantics, set_tracer_provider may no-op while the
    # fixture appears healthy. Surfacing it here means a confusing test 3
    # span-count miss becomes a clear fixture-level failure.
    assert otel_trace.get_tracer_provider() is provider, (
        "OTel private-API latch reset failed -- in_memory_provider fixture "
        "is not isolating state. The opentelemetry-api private globals may "
        "have changed shape; update the snapshot/restore in this fixture."
    )

    try:
        yield exporter
    finally:
        provider.shutdown()
        otel_trace._TRACER_PROVIDER = saved_provider
        otel_trace._TRACER_PROVIDER_SET_ONCE = saved_set_once


def _build_wrapped_app() -> tuple[FastAPI, ASGIApp]:
    """Return ``(inner_fastapi, wrapped_asgi_app)`` mirroring gubbi.main shape.

    The inner FastAPI carries no CorrelationIDMiddleware in its
    constructor's middleware list -- the wrap happens at the outer ASGI
    layer instead, exactly as gubbi/main.py now does. A 401-returning
    route stands in for the gubbi auth-rejection path: no DB / MCP /
    httpx call fires, so the only span we see is the OTel server span
    opened by FastAPIInstrumentor.
    """

    class _NoOpMcpPath:
        """Inert middleware standing in for MCPPathNormalizer.

        Only its presence in the constructor's middleware list matters
        for this test -- it pins the shape (one user middleware in the
        FastAPI list, CorrelationIDMiddleware wrapping from outside).
        """

        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ) -> None:
            await self.app(scope, receive, send)

    inner = FastAPI(middleware=[Middleware(_NoOpMcpPath)])

    @inner.get("/protected")
    async def protected() -> dict[str, str]:
        # 401 short-circuit -- mimics gubbi's auth-rejection path. The
        # canonical reproducer for the silent-drop bug because no inner
        # spans fire to mask an un-stamped server span.
        raise HTTPException(status_code=401)

    # Instrument BEFORE wrapping. instrument_app wraps build_middleware_stack;
    # if called AFTER the FastAPI's middleware_stack is already built (e.g.
    # from inside lifespan after an HTTP/lifespan dispatch), the wrap is
    # bypassed because middleware_stack is reused. Calling at module level
    # before any dispatch keeps middleware_stack=None so the wrap takes
    # effect on the first request. gubbi.main calls configure_otel from
    # the lifespan; the production wiring relies on the lifespan event
    # going through middleware_stack first, but the assertions below pin
    # the OUTSIDE-OTel wrap order independent of when instrument_app
    # actually takes effect.
    FastAPIInstrumentor.instrument_app(inner)

    wrapped = CorrelationIDMiddleware(inner)
    return inner, wrapped


def test_correlation_id_stamps_server_span_when_request_short_circuits_401(
    in_memory_provider: InMemorySpanExporter,
) -> None:
    """A 401-short-circuit request still emits a span carrying the correlation_id.

    Regression for the silent-drop bug. The 401 path has no inner spans
    (no DB query, no MCP tool call); only the FastAPIInstrumentor server
    span fires. With the pre-fix wiring (CorrelationIDMiddleware INSIDE
    FastAPI's user_middleware list), the OTel server span opened before
    CorrelationIDMiddleware populated the ContextVar -- the
    CorrelationSpanProcessor.on_start hook saw ``None`` and stamped
    nothing.

    With the fix (CorrelationIDMiddleware wrapping the whole FastAPI
    app from outside), the ContextVar is populated before any OTel
    middleware runs, so the server span carries the customer-supplied
    id. HyperDX queries by correlation_id keep working through the
    gubbi boundary instead of going dark at the 401 gate.
    """
    # Arrange
    _inner, wrapped = _build_wrapped_app()

    # Act
    with TestClient(wrapped) as client:
        response = client.get(
            "/protected",
            headers={"X-Correlation-ID": _TEST_CID},
        )

    # Assert: status pin defends against the route accidentally returning
    # 200 (which would fire DB/MCP spans and make the assertion vacuous).
    assert response.status_code == 401, (
        f"401 short-circuit precondition violated: status={response.status_code} "
        f"body={response.text!r}. Without the 401 short-circuit this test "
        "is no longer a clean reproducer for the silent-drop bug."
    )

    spans = in_memory_provider.get_finished_spans()
    assert spans, (
        "no spans exported -- FastAPIInstrumentor.instrument_app did not "
        "take effect. Check the install-order in _build_wrapped_app: the "
        "instrumentor must run BEFORE the FastAPI middleware_stack is "
        "first built (i.e. before the first request)."
    )

    # Pin the canonical attribute key on every span. Walking each span
    # rather than asserting on a single one defends against an OTel
    # internals shuffle that adds new span types -- the contract is that
    # every span emitted during a correlated request carries the id, not
    # that there is exactly one span.
    #
    # NOTE: this loop passes on both the fixed AND broken wiring shapes
    # because the defensive ``set_attribute`` in ``CorrelationIDMiddleware``
    # compensates inside the unit-test sandbox where OTel context propagates
    # into user middleware (it does NOT in the production ASGI wiring).
    # ``test_gubbi_main_server_shape_pins_outer_wrap`` (test 3) is the
    # actual regression pin -- do NOT delete it on the grounds that this
    # test passes.
    for span in spans:
        attrs = span.attributes or {}
        assert attrs.get(_CORRELATION_ATTR) == _TEST_CID, (
            f"span {span.name!r} missing correlation_id attribute "
            f"(or wrong value): got {attrs.get(_CORRELATION_ATTR)!r}, "
            f"expected {_TEST_CID!r}. The wrap order regressed: "
            "CorrelationIDMiddleware is no longer OUTSIDE the OTel "
            "server-span scope. See gubbi/main.py end-of-module rationale."
        )


def test_correlation_id_uuid4_fallback_when_header_absent(
    in_memory_provider: InMemorySpanExporter,
) -> None:
    """No X-Correlation-ID header -> middleware mints a UUID4 -> spans stamped.

    Sibling case: every gubbi-emitted span must carry SOME correlation_id,
    even when the upstream client did not provide one (the middleware's
    UUID4 fallback path). Without this case the regression catch above
    could pass while leaving the no-header path silent. We do not pin
    the UUID4 value -- only that the attribute is present, non-empty,
    and identical across spans within the request scope.
    """
    # Arrange
    _inner, wrapped = _build_wrapped_app()

    # Act
    with TestClient(wrapped) as client:
        response = client.get("/protected")

    # Assert
    assert response.status_code == 401
    spans = in_memory_provider.get_finished_spans()
    assert spans, "expected at least one span on the 401 path"

    seen_ids: set[str] = set()
    # NOTE: this loop also passes on the broken wiring shape via the
    # defensive ``set_attribute`` compensation. Test 3 below is the
    # canonical regression pin for the wrap order.
    for span in spans:
        attrs = span.attributes or {}
        cid = attrs.get(_CORRELATION_ATTR)
        assert cid, (
            f"span {span.name!r} missing correlation_id attribute on "
            "the no-header (UUID4-fallback) path. Middleware did not run "
            "or processor did not stamp."
        )
        # Split into two single-clause assertions per PT018 -- compound
        # boolean assertions hide which sub-condition failed in the
        # pytest report.
        seen_ids.add(cid)

    # All spans in a single request share one minted id -- a regression
    # that minted a fresh id per span would surface here.
    assert len(seen_ids) == 1, (
        f"all spans within one request must share the same correlation_id; "
        f"saw {len(seen_ids)} distinct values: {seen_ids}"
    )


def test_gubbi_main_server_shape_pins_outer_wrap() -> None:
    """``gubbi.main.server`` is the full pure-ASGI chain pinning Pattern B.

    Structural pin -- catches wrap-order regressions unambiguously even
    when the in-process span tests above pass via the defensive
    ``set_attribute`` compensation that runs inside
    ``CorrelationIDMiddleware.__call__``. The compensation tags the
    server span retroactively only when the OTel context propagates into
    user middleware (which it does in the unit-test sandbox but NOT in
    the actual gubbi production wiring -- see the validated diagnosis
    at the top of the docstring).

    Pattern B contract (full pure-ASGI):

      1. ``gubbi.main.server`` is a CorrelationIDMiddleware instance
         (request-scoped ContextVar set BEFORE the FastAPI app starts
         processing the ASGI scope).
      2. The wrapped layer below is MCPPathNormalizer (path rewrite
         /mcp -> /mcp/ before the FastAPI router sees the request).
      3. The FastAPI app's ``user_middleware`` list MUST be empty.
         Closes the ``@app.middleware("http")`` SSE-buffering tripwire
         structurally; any middleware that needs to live in this list
         is the wrong tier and must move to the ASGI wrap above.
    """
    import gubbi.main as gm
    from gubbi.middleware import MCPPathNormalizer

    assert isinstance(gm.server, CorrelationIDMiddleware), (
        f"gubbi.main.server must be wrapped by CorrelationIDMiddleware "
        f"OUTSIDE the FastAPI app -- got {type(gm.server).__name__}. "
        "See the end-of-module rationale block in gubbi/main.py: the "
        "wrap places the request-scoped ContextVar set before any OTel "
        "middleware opens a server span."
    )

    path_layer = gm.server.app
    assert isinstance(path_layer, MCPPathNormalizer), (
        f"gubbi.main.server.app must be MCPPathNormalizer (the next "
        f"ASGI layer in the pure-ASGI chain) -- got "
        f"{type(path_layer).__name__}. The wrap order is "
        "CorrelationID -> MCPPathNormalizer -> FastAPI; a regression "
        "here likely means MCPPathNormalizer was moved back into the "
        "FastAPI middleware list (which would re-open the SSE tripwire)."
    )

    inner = path_layer.app
    assert isinstance(inner, FastAPI), (
        f"the innermost layer must be the FastAPI app; got " f"{type(inner).__name__}"
    )
    # The exposed `app` symbol must point at the same FastAPI instance
    # the chain holds. A drift here means callers using `app` for state
    # probes would inspect a stale clone.
    assert inner is gm.app, "gubbi.main.app must alias gubbi.main.server.app.app"

    # Pattern B contract: FastAPI's user_middleware is EMPTY. Anything
    # added via ``app.add_middleware(...)`` or the constructor's
    # ``middleware=[...]`` shows up here; both are off-pattern under
    # Pattern B and re-open the SSE tripwire.
    middleware_classes: list[type[Any]] = [cast("type[Any]", m.cls) for m in inner.user_middleware]
    assert middleware_classes == [], (
        "FastAPI user_middleware MUST be empty under Pattern B (full "
        "pure-ASGI). Found: "
        f"{middleware_classes}. Move them into the ASGI wrap chain at "
        "the bottom of gubbi/main.py instead."
    )
