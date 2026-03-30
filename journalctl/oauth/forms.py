"""Single-user login page for OAuth authorization.

Renders an HTML form where the journal owner enters their password.
On success, generates an authorization code and redirects back to
the OAuth client's redirect_uri.

CSRF protection uses the double-submit cookie pattern:
GET  sets a random csrf_token in an HttpOnly cookie and embeds it in a hidden field.
POST compares the cookie value to the form value (timing-safe).
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable, Coroutine
from typing import Any

import bcrypt
from mcp.server.auth.provider import AuthorizationCode, construct_redirect_uri
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from journalctl.oauth.constants import CSRF_COOKIE_NAME
from journalctl.oauth.storage import OAuthStorage
from journalctl.oauth.templates import render_login_page

LoginHandler = Callable[[Request], Coroutine[Any, Any, Response]]

logger = logging.getLogger("journalctl.oauth.forms")


def create_login_handler(
    storage: OAuthStorage,
    owner_password_hash: str,
    auth_code_ttl: int = 300,
    *,
    secure_cookies: bool = True,
) -> LoginHandler:
    """Create a Starlette endpoint handler for /login."""

    async def login_handler(request: Request) -> Response:
        if request.method == "GET":
            params = request.query_params
            csrf_token = secrets.token_urlsafe(32)
            return render_login_page(
                client_id=params.get("client_id", ""),
                redirect_uri=params.get("redirect_uri", ""),
                state=params.get("state", ""),
                code_challenge=params.get("code_challenge", ""),
                scope=params.get("scope", ""),
                csrf_token=csrf_token,
                cookie_max_age=auth_code_ttl,
                secure_cookies=secure_cookies,
            )

        # POST: verify CSRF token first
        form = await request.form()
        form_csrf = str(form.get("csrf_token", ""))
        cookie_csrf = request.cookies.get(CSRF_COOKIE_NAME, "")

        if not form_csrf or not cookie_csrf or not secrets.compare_digest(form_csrf, cookie_csrf):
            logger.warning("CSRF validation failed")
            return HTMLResponse("CSRF validation failed", status_code=403)

        client_id = str(form.get("client_id", ""))
        redirect_uri = str(form.get("redirect_uri", ""))
        state = str(form.get("state", ""))
        code_challenge = str(form.get("code_challenge", ""))
        scope = str(form.get("scope", ""))
        password = str(form.get("password", ""))

        # Verify password
        if not bcrypt.checkpw(
            password.encode("utf-8"),
            owner_password_hash.encode("utf-8"),
        ):
            client_host = request.client.host if request.client else "unknown"
            logger.warning("Failed login attempt from %s", client_host)
            csrf_token = secrets.token_urlsafe(32)
            return render_login_page(
                client_id=client_id,
                redirect_uri=redirect_uri,
                state=state,
                code_challenge=code_challenge,
                scope=scope,
                csrf_token=csrf_token,
                cookie_max_age=auth_code_ttl,
                error="Invalid password",
                secure_cookies=secure_cookies,
            )

        # Generate authorization code
        code = secrets.token_urlsafe(32)
        scopes = scope.split() if scope else []
        auth_code = AuthorizationCode(
            code=code,
            scopes=scopes,
            expires_at=time.time() + auth_code_ttl,
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,  # type: ignore[arg-type]
            redirect_uri_provided_explicitly=True,
        )
        storage.save_auth_code(code, auth_code)

        client_host = request.client.host if request.client else "unknown"
        logger.info("Authorization code issued from %s", client_host)

        # Redirect back to client, clear CSRF cookie
        callback = construct_redirect_uri(
            redirect_uri,
            code=code,
            state=state,
        )
        response = RedirectResponse(url=callback, status_code=302)
        response.delete_cookie(CSRF_COOKIE_NAME)
        return response

    return login_handler
