"""User onboarding-signal endpoint.

``GET /api/v1/user/setup-signals`` -- cheap journal-derived booleans used by
the cloud-api setup-state composer. Keeping this read on the journal data plane
means cloud-api never queries journal tables directly.

Distinct from the journal-data ``web`` routers: this surface returns onboarding
signals, not journal content. The two booleans are EXISTS probes (no decryption,
no row hydration), but the response still carries ``Cache-Control: private,
no-store`` because the signals reflect per-user state.
"""

from __future__ import annotations

# UUID is a FastAPI Depends() Annotated type, resolved at route-registration
# time, so it stays a runtime import despite ``from __future__ import
# annotations`` -- ruff TC would otherwise push it under TYPE_CHECKING.
from typing import Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel

from gubbi.api.v1.auth import require_scope
from gubbi.app_state import require_app_ctx
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import setup_signals as setup_signals_repo

__all__: list[str] = [
    "SetupSignalsResponse",
    "get_setup_signals",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


class SetupSignalsResponse(BaseModel):
    """``GET /api/v1/user/setup-signals`` body.

    has_entries               : the user has at least one non-deleted entry.
    has_synced_conversations  : the user has at least one conversation that
                                arrived via the extension/zip ingest path.
    """

    has_entries: bool
    has_synced_conversations: bool


@router.get(
    "/setup-signals",
    response_model=SetupSignalsResponse,
    responses={403: {"description": "missing scope"}},
)
async def get_setup_signals(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> Response:
    """GET /api/v1/user/setup-signals.

    Returns two onboarding booleans derived from the authenticated user's
    journal data. Both are cheap EXISTS probes under the RLS-scoped connection.

    Cache-Control is ``private, no-store``: the signals are per-user state.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        signals = await setup_signals_repo.get_setup_signals(conn)

    await log.info(
        "user_setup_signals",
        has_entries=signals.has_entries,
        has_synced_conversations=signals.has_synced_conversations,
    )
    body = SetupSignalsResponse(
        has_entries=signals.has_entries,
        has_synced_conversations=signals.has_synced_conversations,
    )
    response = JSONResponse(body.model_dump(mode="json"))
    response.headers["Cache-Control"] = "private, no-store"
    return response
