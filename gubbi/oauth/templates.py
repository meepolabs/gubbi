"""HTML templates for OAuth pages.

Separated from login.py to keep request-handling logic readable.
"""

from __future__ import annotations

import html as html_mod
import secrets

from starlette.responses import HTMLResponse

from gubbi.core.scope import SCOPE_DESCRIPTIONS
from gubbi.oauth.constants import CSRF_COOKIE_NAME

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Journal - Authorize</title>
    <style nonce="{style_nonce}">
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 2rem;
            width: 100%;
            max-width: 480px;
            margin: 1rem;
        }}
        h1 {{
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }}
        .client-info {{
            background: #0f172a;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
            color: #94a3b8;
            word-break: break-all;
        }}
        .scopes {{
            background: #0f172a;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
        }}
        .scopes h2 {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .scope-item {{
            margin-bottom: 0.5rem;
        }}
        .scope-name {{
            font-family: monospace;
            font-size: 0.8rem;
            color: #38bdf8;
        }}
        .scope-desc {{
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.125rem;
        }}
        label {{
            display: block;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            color: #cbd5e1;
        }}
        input[type="password"] {{
            width: 100%;
            padding: 0.625rem 0.75rem;
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 1rem;
            outline: none;
        }}
        input[type="password"]:focus {{
            border-color: #3b82f6;
        }}
        button {{
            width: 100%;
            padding: 0.625rem;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
        }}
        button:hover {{
            background: #2563eb;
        }}
        .error {{
            background: #7f1d1d;
            color: #fca5a5;
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Authorize Access</h1>
        <p class="subtitle">A client is requesting access to your journal.</p>
        <div class="client-info">Client: {client_id}</div>
        {scopes_html}
        {error_html}
        <form method="POST">
            <input type="hidden" name="csrf_token" value="{csrf_token}">
            <input type="hidden" name="client_id" value="{client_id}">
            <input type="hidden" name="redirect_uri" value="{redirect_uri}">
            <input type="hidden" name="state" value="{state}">
            <input type="hidden" name="code_challenge" value="{code_challenge}">
            <input type="hidden" name="scope" value="{scope}">
            <label for="password">Owner Password</label>
            <input type="password" id="password" name="password"
                   placeholder="Enter your password" autofocus required>
            <button type="submit">Authorize</button>
        </form>
    </div>
</body>
</html>"""


def _render_scopes_html(scope_str: str) -> str:
    """Build the scopes section for the consent UI.

    Loops over the requested scope string (space-delimited) and renders
    each scope with its human-readable description from SCOPE_DESCRIPTIONS.
    """
    if not scope_str or not scope_str.strip():
        return ""
    esc = html_mod.escape
    scopes = scope_str.split()
    items = []
    for s in scopes:
        desc = SCOPE_DESCRIPTIONS.get(s, "")
        items.append(
            f'<div class="scope-item">'
            f'<div class="scope-name">{esc(s)}</div>'
            f'<div class="scope-desc">{esc(desc) if desc else "No description available."}</div>'
            f"</div>"
        )
    return '<div class="scopes">' "<h2>Permissions requested</h2>" f"{''.join(items)}" "</div>"


def render_login_page(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    scope: str,
    csrf_token: str,
    cookie_max_age: int,
    error: str = "",
    *,
    secure_cookies: bool = True,
) -> HTMLResponse:
    """Render the login page with CSRF token and XSS-escaped params."""
    esc = html_mod.escape
    error_html = f'<div class="error">{esc(error)}</div>' if error else ""
    scopes_html = _render_scopes_html(scope)
    style_nonce = secrets.token_urlsafe(16)
    page = LOGIN_HTML.format(
        client_id=esc(client_id),
        redirect_uri=esc(redirect_uri),
        state=esc(state),
        code_challenge=esc(code_challenge),
        scope=esc(scope),
        csrf_token=esc(csrf_token),
        error_html=error_html,
        scopes_html=scopes_html,
        style_nonce=style_nonce,
    )
    response = HTMLResponse(page)
    response.headers["Content-Security-Policy"] = (
        f"default-src 'none'; style-src 'nonce-{style_nonce}'; form-action 'self'"
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        csrf_token,
        httponly=True,
        samesite="lax",
        secure=secure_cookies,
        max_age=cookie_max_age,
        path="/",
    )
    return response
