"""Read-path failure logs must not export driver-supplied text.

The hybrid-search service and the briefing tool catch database errors
raised by a SELECT and log them. An asyncpg exception's ``str()`` carries
the server's own message plus its ``DETAIL`` block, and both can quote
the failing statement's values back -- here the predicate values of a
user's own read, in a log destination that gets shipped and archived.

Each test plants distinct markers in the exception the driver would raise
and asserts that no marker survives onto a log line, while the safe shape
(exception class name, SQLSTATE) does, and that the read behavior the
caller depends on -- surfaced-as-decryption-failed hits, degraded key
facts, a re-raised unexpected error -- is unchanged.

:class:`TestHarnessCanFail` is the positive control: it makes the call
these sites used to make and asserts the harness sees the markers, so a
clean assertion elsewhere is evidence rather than an empty scan.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import asyncpg
import pytest

from gubbi.models.search import SearchResult
from gubbi.services import search as search_svc
from gubbi.storage.exceptions import DatabaseUnavailable
from gubbi.tools import context as context_tool
from tests.fixtures.driver_errors import (
    MARKER_SID,
    MARKER_UNICODE,
    SQLSTATE,
    assert_no_markers,
    planted_driver_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.fixtures.log_capture import LogCapture

pytestmark = pytest.mark.unit

_USER_ID = UUID("11111111-2222-3333-4444-555555555555")


def _wrapped_driver_error() -> DatabaseUnavailable:
    """Return the storage layer's translation of the planted driver error.

    ``DatabaseUnavailable(str(exc))`` copies the driver message into its
    own ``str()`` while failing an ``isinstance(exc,
    asyncpg.PostgresError)`` check, so it reaches the "unexpected"
    except-branch of a read path with the driver text intact.
    """
    driver = planted_driver_error()
    try:
        raise DatabaseUnavailable(str(driver)) from driver
    except DatabaseUnavailable as exc:
        return exc


def _merged_results() -> list[SearchResult]:
    return [
        SearchResult(
            source_key="entry:1",
            doc_type="entry",
            topic="work",
            rank=1.0,
            date="2026-05-01",
            entry_id=1,
        ),
        SearchResult(
            source_key="conversation:9",
            doc_type="conversation",
            topic="chat",
            rank=2.0,
            date="2026-05-03",
            conversation_id=9,
        ),
    ]


class _FakeEmbeddingService:
    """Embedding service whose vector search raises what the test supplies."""

    def __init__(self, *, raises: Callable[[], BaseException]) -> None:
        self._raises = raises

    def encode(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def search_by_vector(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise self._raises()


# ---------------------------------------------------------------------------
# services/search.py -- hydration batch failures and the unexpected branch
# ---------------------------------------------------------------------------


class TestSearchHydrationBatchFailureLog:
    """A failed batch decrypt is logged by type and SQLSTATE only."""

    async def test_entry_batch_failure_log_omits_driver_text(
        self,
        log_capture: LogCapture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        async def _raising_get_texts(*_args: Any, **_kwargs: Any) -> dict[int, Any]:
            raise planted_driver_error()

        async def _ok_titles(
            _conn: Any, _cipher: Any, ids: list[int]
        ) -> dict[int, tuple[str, str]]:
            return dict.fromkeys(ids, ("a title", "a summary"))

        monkeypatch.setattr(search_svc.entry_repo, "get_texts", _raising_get_texts)
        monkeypatch.setattr(search_svc.conv_repo, "get_titles_summaries", _ok_titles)

        # Act
        hydrated = await search_svc._hydrate_results(None, object(), _merged_results())  # type: ignore[arg-type]

        # Assert
        assert_no_markers(log_capture.text, "entry batch failure log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert SQLSTATE in log_capture.text
        entries = [hit for hit in hydrated if hit.doc_type == "entry"]
        assert len(entries) == 1, "a batch failure must not drop the hit"
        assert entries[0].decryption_failed is True
        assert entries[0].content == search_svc.DECRYPTION_FAILED_SENTINEL

    async def test_conversation_batch_failure_log_omits_driver_text(
        self,
        log_capture: LogCapture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        async def _ok_texts(
            _conn: Any, _cipher: Any, ids: list[int]
        ) -> dict[int, tuple[str, str | None]]:
            return dict.fromkeys(ids, ("ok content", None))

        async def _raising_titles(*_args: Any, **_kwargs: Any) -> dict[int, Any]:
            raise planted_driver_error()

        monkeypatch.setattr(search_svc.entry_repo, "get_texts", _ok_texts)
        monkeypatch.setattr(search_svc.conv_repo, "get_titles_summaries", _raising_titles)

        # Act
        hydrated = await search_svc._hydrate_results(None, object(), _merged_results())  # type: ignore[arg-type]

        # Assert
        assert_no_markers(log_capture.text, "conversation batch failure log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert SQLSTATE in log_capture.text
        convs = [hit for hit in hydrated if hit.doc_type == "conversation"]
        assert len(convs) == 1
        assert convs[0].decryption_failed is True
        assert convs[0].title == search_svc.DECRYPTION_FAILED_SENTINEL


class TestSemanticSearchUnexpectedFailureLog:
    """The unexpected-failure branch logs safely and still re-raises."""

    async def test_wrapped_driver_error_is_logged_without_its_text_and_propagates(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        app_ctx = MagicMock()
        app_ctx.embedding_service = _FakeEmbeddingService(raises=_wrapped_driver_error)
        wrapper = _wrapped_driver_error()
        assert MARKER_SID in str(wrapper), (
            "the wrapper must genuinely carry the driver text, or this proves nothing"
        )

        # Act
        with (
            patch.object(search_svc.search_repo, "fts_search", AsyncMock(return_value=[])),
            pytest.raises(DatabaseUnavailable),
        ):
            await search_svc._run_dual_search(
                None,  # type: ignore[arg-type]
                app_ctx,
                "a query",
                [0.1, 0.2, 0.3],
                None,
                None,
                None,
                None,
                None,
                10,
            )

        # Assert
        assert_no_markers(log_capture.text, "semantic search unexpected failure log")
        assert "DatabaseUnavailable" in log_capture.text
        assert SQLSTATE in log_capture.text, (
            "the SQLSTATE is found through the cause chain, not on the wrapper"
        )

    async def test_driver_error_degrades_to_fts_without_exporting_driver_text(
        self,
        log_capture: LogCapture,
    ) -> None:
        """A PostgresError from the semantic backend degrades, never raises."""
        # Arrange
        app_ctx = MagicMock()
        app_ctx.embedding_service = _FakeEmbeddingService(raises=planted_driver_error)
        fts_hit = _merged_results()[0]

        # Act
        with patch.object(search_svc.search_repo, "fts_search", AsyncMock(return_value=[fts_hit])):
            merged = await search_svc._run_dual_search(
                None,  # type: ignore[arg-type]
                app_ctx,
                "a query",
                [0.1, 0.2, 0.3],
                None,
                None,
                None,
                None,
                None,
                10,
            )

        # Assert
        assert [hit.source_key for hit in merged] == ["entry:1"], (
            "a semantic-backend failure must leave the FTS results intact"
        )
        assert_no_markers(log_capture.text, "semantic search degradation log")
        assert "InsufficientPrivilegeError" in log_capture.text, (
            "the degradation log must name the exception class for triage"
        )
        assert SQLSTATE in log_capture.text, (
            "the degradation log must carry the SQLSTATE for triage"
        )


# ---------------------------------------------------------------------------
# tools/context.py -- the three key-facts sites in journal_briefing
# ---------------------------------------------------------------------------


def _briefing_tool(app_ctx: Any) -> Callable[..., Any]:
    """Register the context tools and return the briefing handler.

    Registration is what stacks ``@require_scope`` over the handler, so
    the callable returned here is the one production dispatches.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    context_tool.register(mcp, app_ctx)
    return mcp._tool_manager._tools["journal_briefing"].fn


def _briefing_app_ctx(*, raises: Callable[[], BaseException]) -> MagicMock:
    app_ctx = MagicMock()
    app_ctx.settings.timezone = "UTC"
    app_ctx.cipher = MagicMock()
    app_ctx.embedding_service = _FakeEmbeddingService(raises=raises)
    app_ctx.pool = MagicMock()
    return app_ctx


def _briefing_patches(*, get_texts: Any = None) -> list[Any]:
    """Patch every read the briefing makes apart from the key-facts path."""
    conn = AsyncMock()
    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(return_value=conn)
    connection_cm.__aexit__ = AsyncMock(return_value=False)

    patches = [
        patch("gubbi.tools.context.user_scoped_connection", return_value=connection_cm),
        patch.object(context_tool.entry_repo, "get_by_date_range", AsyncMock(return_value=[])),
        patch.object(context_tool.topic_repo, "list_all", AsyncMock(return_value=([], 0))),
        patch.object(
            context_tool.entry_repo,
            "get_stats",
            AsyncMock(return_value={"total_documents": 1, "topics": 1}),
        ),
        patch.object(context_tool.search_repo, "has_embeddings", AsyncMock(return_value=True)),
    ]
    if get_texts is not None:
        patches.append(patch.object(context_tool.entry_repo, "get_texts", get_texts))
    return patches


async def _call_briefing(app_ctx: Any, patches: list[Any]) -> Any:
    from contextlib import ExitStack

    from gubbi.auth_context import current_token_scopes, current_user_id

    briefing = _briefing_tool(app_ctx)
    user_token = current_user_id.set(_USER_ID)
    scope_token = current_token_scopes.set(frozenset({"journal"}))
    try:
        with ExitStack() as stack:
            for one in patches:
                stack.enter_context(one)
            return await briefing()
    finally:
        current_token_scopes.reset(scope_token)
        current_user_id.reset(user_token)


class TestBriefingKeyFactsFailureLogs:
    """Every key-facts failure path logs by type and SQLSTATE only."""

    async def test_key_facts_query_failure_log_omits_driver_text(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        app_ctx = _briefing_app_ctx(raises=planted_driver_error)

        # Act
        payload = await _call_briefing(app_ctx, _briefing_patches())

        # Assert
        assert_no_markers(log_capture.text, "key facts batch query failure log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert SQLSTATE in log_capture.text
        assert payload["key_facts"] == [], (
            "a key-facts query failure degrades the briefing, it does not fail it"
        )
        assert payload["key_facts_status"] == "empty"

    async def test_key_facts_unexpected_failure_log_omits_driver_text(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        app_ctx = _briefing_app_ctx(raises=_wrapped_driver_error)

        # Act
        with pytest.raises(DatabaseUnavailable):
            await _call_briefing(app_ctx, _briefing_patches())

        # Assert
        assert_no_markers(log_capture.text, "key facts unexpected failure log")
        assert "DatabaseUnavailable" in log_capture.text
        assert SQLSTATE in log_capture.text

    async def test_key_facts_entry_batch_failure_log_omits_driver_text(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        async def _facts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return [{"entry_id": 7}]

        async def _raising_get_texts(*_args: Any, **_kwargs: Any) -> dict[int, Any]:
            raise planted_driver_error()

        app_ctx = _briefing_app_ctx(raises=planted_driver_error)
        app_ctx.embedding_service.search_by_vector = _facts

        # Act
        payload = await _call_briefing(app_ctx, _briefing_patches(get_texts=_raising_get_texts))

        # Assert
        assert_no_markers(log_capture.text, "key facts entry batch failure log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert SQLSTATE in log_capture.text
        assert payload["key_facts"] == [], (
            "an undecryptable key-facts batch yields no facts, not a failure"
        )
        assert payload["key_facts_status"] == "empty"


class TestNonDriverErrorsKeepTheirDiagnostics:
    """Sanitization is scoped to driver text, not to all error logging.

    An exception this codebase raises itself owns its message, and
    withholding it would blind the operator for no gain -- the same split
    ``general_exception_handler`` already makes. These tests fail if a
    catch-all branch replaces the traceback with bounded fields
    unconditionally.
    """

    async def test_semantic_search_non_driver_failure_keeps_its_message(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        def _our_own_error() -> ValueError:
            return ValueError("dimension mismatch in our own encoder")

        app_ctx = MagicMock()
        app_ctx.embedding_service = _FakeEmbeddingService(raises=_our_own_error)

        # Act
        with (
            patch.object(search_svc.search_repo, "fts_search", AsyncMock(return_value=[])),
            pytest.raises(ValueError, match="dimension mismatch"),
        ):
            await search_svc._run_dual_search(
                None,  # type: ignore[arg-type]
                app_ctx,
                "a query",
                [0.1, 0.2, 0.3],
                None,
                None,
                None,
                None,
                None,
                10,
            )

        # Assert
        assert "dimension mismatch in our own encoder" in log_capture.text, (
            "a non-driver exception must keep its own message in the log"
        )

    async def test_key_facts_non_driver_failure_keeps_its_message(
        self,
        log_capture: LogCapture,
    ) -> None:
        # Arrange
        def _our_own_error() -> ValueError:
            return ValueError("dimension mismatch in our own encoder")

        app_ctx = _briefing_app_ctx(raises=_our_own_error)

        # Act
        with pytest.raises(ValueError, match="dimension mismatch"):
            await _call_briefing(app_ctx, _briefing_patches())

        # Assert
        assert "dimension mismatch in our own encoder" in log_capture.text, (
            "a non-driver exception must keep its own message in the log"
        )


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------


class TestHarnessCanFail:
    """The harness must see the driver text the previous shape exported.

    Both read-path modules log through an ``AsyncBoundLogger``. Its
    ``exception()`` reads ``sys.exc_info()`` on the calling task -- where
    the exception is still current -- so the traceback and the
    exception's own ``str()`` do reach the log, unlike an
    ``exc_info=True`` emit whose resolution happens on the worker thread.
    That asymmetry is why these sites are audited by call shape rather
    than by the presence of ``exc_info``.
    """

    @pytest.mark.parametrize(
        "module",
        [
            pytest.param(search_svc, id="services.search"),
            pytest.param(context_tool, id="tools.context"),
        ],
    )
    async def test_default_exception_shape_leaks_driver_text_into_the_capture(
        self,
        module: Any,
        log_capture: LogCapture,
    ) -> None:
        # Arrange / Act -- the call these sites made before sanitization
        try:
            raise planted_driver_error()
        except asyncpg.PostgresError:
            await module.logger.exception("control: default exception shape")

        # Assert
        assert MARKER_SID in log_capture.text, (
            "the capture cannot see the module logger's exception output, so every "
            "clean assertion in this file would be vacuous"
        )

    async def test_marker_scan_catches_a_non_ascii_marker_only_json_escaped(
        self,
        log_capture: LogCapture,
    ) -> None:
        """``assert_no_markers`` must scan the escaped form, not only the raw one.

        ``JSONRenderer`` renders with ``ensure_ascii=True``, so a
        non-ASCII value reaches the structured line as ``\\uXXXX`` and a
        raw substring scan reports "clean" on a line that does carry it.
        This control plants such a value in a structured FIELD (not the
        traceback, which is written verbatim) and asserts both that the
        raw form is genuinely absent and that the scan still trips.
        """
        # Arrange / Act
        await search_svc.logger.error("control: escaped marker", note=MARKER_UNICODE)

        # Assert
        captured = log_capture.text
        assert MARKER_UNICODE not in captured, (
            "if the raw form were present, this control would pass under a "
            "raw-only scan and prove nothing"
        )
        assert json.dumps(MARKER_UNICODE)[1:-1] in captured, (
            "the escaped form must be on the line, or there is nothing to catch"
        )
        with pytest.raises(AssertionError, match="exported planted driver text"):
            assert_no_markers(captured, "escaped-marker control")
