"""Request-scoped authenticated user ID, read by user_scoped_connection (TASK-02.06)."""

from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID

current_user_id: ContextVar[UUID | None] = ContextVar("current_user_id", default=None)
