"""Integration tests for Mode 3 fresh-deploy (Hydra-backed hosted).

Asserts the JIT-provisioning invariant on users table when
JOURNAL_OPERATOR_EMAIL is unset -- no operator row at boot, users get
created by BearerAuthMiddleware on first authenticated request from
/userinfo claims.

Note: Steps 7 and 8 exercise the email-collision / cache short-circuit
policy that lands under the sibling TASK-AUTH-01. If those symbols
are absent from HEAD, steps are skipif'd so this PR merges independently.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest
from starlette.applications import Starlette
from starlette.routing import Route


def _check_task_auth_01_symbols() -> bool:
    """Return True if TASK-AUTH-01 collision-handling symbols are present."""
    try:
        from journalctl.middleware.auth import (  # noqa: PLC0415
            BearerAuthMiddleware,
        )

        if hasattr(BearerAuthMiddleware, "_pre_context_jwt_provision"):
            return True

        jit_fn = getattr(BearerAuthMiddleware, "_jit_provision", None)
        if jit_fn is None or not hasattr(jit_fn, "__code__"):
            return False

        varnames = jit_fn.__code__.co_varnames
        return "collision" in varnames or "email_conflict" in varnames
    except (ImportError, AttributeError):
        return False


TASK_AUTH_01_PRESENT = _check_task_auth_01_symbols()


_TEST_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
_TEST_UUID_2 = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _make_asgi_app() -> Any:
    """Return a minimal ASGI app that responds 200."""

    async def _app(scope: dict, receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    return _app


def _build_mock_hydra(claims_sub: UUID) -> tuple[AsyncMock, MagicMock]:
    """Build a mocked HydraIntrospector with valid claims and /userinfo http mock."""
    claims = MagicMock()
    claims.sub = claims_sub
    claims.scope = "openid journal email"
    claims.exp = 9999999999

    mock_iv = AsyncMock(spec=True)
    mock_iv.introspect = AsyncMock(return_value=claims)

    mock_http = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"email": "newuser@example.com"}
    mock_http.get = AsyncMock(return_value=resp)
    mock_iv.http_client = mock_http

    return mock_iv, mock_http


def _build_mw(
    admin_pool: asyncpg.Pool,
    mock_iv: MagicMock,
    hydra_public_url: str | None = "https://auth.example.com",
) -> Any:
    """Build a BearerAuthMiddleware wrapping a test app, backed by real admin_pool."""

    def _pool_acquire_side_effect(*args: Any, **kwargs: Any) -> Any:
        return admin_pool.acquire(*args, **kwargs)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(side_effect=_pool_acquire_side_effect)

    from journalctl.middleware import BearerAuthMiddleware  # noqa: PLC0415

    app = Starlette(routes=[Route("/", _make_asgi_app())])
    return BearerAuthMiddleware(
        app,
        api_key="",
        introspector=mock_iv,
        required_scope="journal",
        admin_pool=mock_pool,
        jit_pool=mock_pool,
        hydra_public_url=hydra_public_url,
    )


class TestMode3FreshDeploy:
    """Verify JIT provisioning on a freshly deployed Mode 3 instance."""

    def test_settings_boot_with_hydra_empty_operator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step 2: Settings accepts Mode 3 env -- no operator email needed."""
        from journalctl.config import get_settings  # noqa: PLC0415

        get_settings.cache_clear()

        from journalctl.config import Settings  # noqa: PLC0415

        db_app_url = os.environ.get("JOURNAL_DB_APP_URL", "postgresql://test")
        s = Settings(
            hydra_admin_url="http://hydra-internal:4444",
            hydra_public_issuer_url="https://auth.example.com",
            hydra_public_url="https://auth.example.com",
            api_key="",
            operator_email="",
            password_hash="",
            db_app_url=db_app_url,
        )
        assert s.hydra_admin_url == "http://hydra-internal:4444"
        assert s.api_key == ""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_users_table_empty_before_first_auth(
        self,
        admin_pool: asyncpg.Pool,
        clean_rls_db: asyncpg.Pool,
    ) -> None:
        """Step 3: Alembic upgrade head leaves users table empty (COUNT = 0)."""
        _ = clean_rls_db
        async with admin_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM users")
        assert count == 0, "Fresh deploy must have zero users rows"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_jit_provisions_user_on_first_authenticated_request(
        self,
        admin_pool: asyncpg.Pool,
        clean_rls_db: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Step 5-6: First authenticated request provisions a users row.

        Asserts COUNT = 1 after the request, with matching sub and email.
        """
        monkeypatch.setenv("JOURNAL_HYDRA_ADMIN_URL", "http://hydra-internal:4444")
        monkeypatch.setenv("JOURNAL_API_KEY", "")
        monkeypatch.delenv("JOURNAL_OPERATOR_EMAIL", raising=False)
        monkeypatch.delenv("JOURNAL_API_KEY", raising=False)
        _ = clean_rls_db

        mock_iv, _ = _build_mock_hydra(_TEST_UUID)

        import httpx  # noqa: PLC0415

        from journalctl.middleware import BearerAuthMiddleware  # noqa: PLC0415

        mw = Starlette(routes=[Route("/", _make_asgi_app())])
        wrapped = BearerAuthMiddleware(
            mw,
            api_key="",
            introspector=mock_iv,
            required_scope="journal",
            jit_pool=_build_pool_proxy(admin_pool),
            hydra_public_url="https://auth.example.com",
        )

        test_token = "ory_at_first_request"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=wrapped), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {test_token}"})

        assert resp.status_code == 200
        mock_iv.introspect.assert_called_once()

        async with admin_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM users")
        assert count == 1, "First authenticated request must provision a user"

        async with admin_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id, email FROM users")
        assert row["id"] == _TEST_UUID
        assert row["email"] == "newuser@example.com"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_jit_idempotent_on_second_same_token(
        self,
        admin_pool: asyncpg.Pool,
        clean_rls_db: asyncpg.Pool,
    ) -> None:
        """Step 7: Second authenticated request with same sub does not duplicate user."""
        _ = clean_rls_db
        mock_iv, _ = _build_mock_hydra(_TEST_UUID)

        wrapped = _build_mw(admin_pool, mock_iv)

        import httpx  # noqa: PLC0415

        test_token = "ory_at_second_request"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=wrapped), base_url="http://test"
        ) as client:
            resp_1 = await client.get("/", headers={"Authorization": f"Bearer {test_token}"})
            resp_2 = await client.get("/", headers={"Authorization": f"Bearer {test_token}"})

        assert resp_1.status_code == 200
        assert resp_2.status_code == 200

        async with admin_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE id = $1", _TEST_UUID)
        assert count == 1, "JIT UPSERT must be idempotent -- second request does not duplicate user"

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.skipif(
        not TASK_AUTH_01_PRESENT,
        reason=(
            "TASK-AUTH-01 (email-collision + JIT cache) has not merged into HEAD. "
            "This test exercises the collision path inside BearerAuthMiddleware._jit_provision "
            "which does not exist yet; when TASK-AUTH-01 lands, remove this skipif."
        ),
    )
    async def test_different_sub_same_email_hits_collision_path(
        self,
        admin_pool: asyncpg.Pool,
        clean_rls_db: asyncpg.Pool,
    ) -> None:
        """Step 8: Request with different sub but same email returns 401.

        Requires TASK-AUTH-01 email-collision policy to be present in HEAD.
        """
        _ = clean_rls_db
        mock_iv, _ = _build_mock_hydra(_TEST_UUID_2)

        wrapped = _build_mw(admin_pool, mock_iv)

        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=wrapped), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer different_sub_token"})

        # Collision detected -- middleware rejects sharing identity.
        assert resp.status_code == 401, (
            "Different sub with same email should trigger" " email-collision rejection (401)"
        )


def _build_pool_proxy(pool: asyncpg.Pool) -> MagicMock:
    """Build a MagicMock pool whose acquire delegates to the real pool."""

    def _side_effect(*args: Any, **kwargs: Any) -> Any:
        return pool.acquire(*args, **kwargs)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(side_effect=_side_effect)
    return mock_pool
