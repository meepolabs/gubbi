"""Bearer token authentication for the MCP endpoint.

Validates three types of tokens, reflecting the three supported deploy
shapes (see docs/deployment.md for the full matrix):

1. Static API key (Claude Code, Desktop, Cursor, other CLI clients).
   Enabled in Mode 1 (API-key-only) and Mode 2 (full self-host). Disabled
   in Mode 3 (multi-tenant hosted) -- lifespan passes api_key="" so the
   timing-safe compare never matches.
2. Hydra OAuth 2.1 access tokens (Mode 3 -- multi-tenant hosted,
   activated when JOURNAL_HYDRA_ADMIN_URL is set).
3. Self-host OAuth access tokens via external token_validator callback
   (Mode 2 -- single-user self-host via the MCP SDK's DCR-capable OAuth
   routes, activated when JOURNAL_PASSWORD_HASH is set).

Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses -- BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from uuid import UUID

import asyncpg
import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.audit import Action, record_audit
from journalctl.auth.hydra import (
    HydraIntrospector,
    HydraInvalidToken,
    HydraUnreachable,
    TokenClaims,
)
from journalctl.core.auth_context import current_token_scopes, current_user_id
from journalctl.core.scope import check_scope
from journalctl.oauth.constants import MAX_BEARER_TOKEN_LEN


def _build_bearer_challenge(
    error: str,
    resource_metadata_url: str | None,
    *,
    required_scope: str | None = None,
) -> str:
    """Build an RFC 6750 WWW-Authenticate Bearer challenge.

    Per MCP spec 2025-11-25, a protected MCP resource must include a
    ``resource_metadata`` parameter pointing to its OAuth protected-resource
    metadata document so clients can discover the authorization server.
    """
    parts = [f'error="{error}"']
    if required_scope is not None:
        parts.append(f'required_scope="{required_scope}"')
    if resource_metadata_url is not None:
        parts.append(f'resource_metadata="{resource_metadata_url}"')
    return "Bearer " + ", ".join(parts)


def _unauthorized(detail: str, resource_metadata_url: str | None = None) -> JSONResponse:
    """Return a 401 JSONResponse with RFC 6750 Bearer challenge.

    The ``detail`` param is the human-readable error in the JSON body; the
    WWW-Authenticate challenge always uses error="invalid_token" per RFC 6750.
    """
    return JSONResponse(
        {"error": detail},
        status_code=401,
        headers={
            "WWW-Authenticate": _build_bearer_challenge("invalid_token", resource_metadata_url),
        },
    )


def _forbidden(required_scope: str, resource_metadata_url: str | None = None) -> JSONResponse:
    """Return a 403 JSONResponse with Bearer challenge for scope denial."""
    return JSONResponse(
        {"error": "insufficient_scope"},
        status_code=403,
        headers={
            "WWW-Authenticate": _build_bearer_challenge(
                "insufficient_scope",
                resource_metadata_url,
                required_scope=required_scope,
            ),
        },
    )


def _service_unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "auth service unavailable"},
        status_code=503,
        headers={"Retry-After": "5"},
    )


class _EmailCollision(Exception):
    """Raised when a Hydra subject's email collides with an active user."""

    def __init__(self, colliding_sub: str, email: str) -> None:  # noqa: D107
        self.colliding_sub = colliding_sub
        self.email = email


class _PreExistingSub(Exception):
    """Raised when a sub is already provisioned (by id or by email collision)."""

    pass


# ---------------------------------------------------------------------------
# In-process LRU cache for known-provisioned sub UUIDs.
#
# Short-circuits both the /userinfo HTTP call and the DB existence check for
# subjects that have already been provisioned, which is the common case on
# every subsequent request after the first login.  Does NOT replace the
# introspection cache (that caches TokenClaims keyed on token fingerprint);
# this one tracks *subjects* that exist in the database regardless of token.
class _ProvisionedCache:
    """In-memory LRU cache for provisioned Hydra subjects."""

    max_size = 1000
    ttl_seconds = 60.0

    def __init__(self) -> None:  # noqa: D107
        self._cache: OrderedDict[UUID, float] = OrderedDict()

    def find(self, sub: UUID) -> bool:
        """Return True if ``sub`` is provisioned and cache entry is fresh."""
        ts = self._cache.get(sub)
        if ts is None:
            return False
        if time.monotonic() - ts > self.ttl_seconds:
            del self._cache[sub]  # expired
            return False
        self._cache.move_to_end(sub)
        return True

    def put(self, sub: UUID) -> None:
        """Insert or refresh the entry for ``sub``. Evicts oldest entries on capacity."""
        if sub in self._cache:
            self._cache.move_to_end(sub)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[sub] = time.monotonic()


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Validates tokens in three modes:
    1. Direct match against the static API key
    2. Hydra introspection for Ory access tokens (multi-tenant hosted)
    3. Delegated token validation via selfhost_token_validator callback
       (single-user self-host)

    This is NOT BaseHTTPMiddleware -- it passes through the raw ASGI
    interface without buffering, so SSE streaming works.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key: str,
        introspector: HydraIntrospector | None = None,
        required_scope: str = "journal",
        selfhost_token_validator: Callable[[str], bool] | None = None,
        operator_user_id: UUID | None = None,
        jit_pool: asyncpg.Pool | None = None,
        admin_pool: asyncpg.Pool | None = None,
        hydra_public_url: str | None = None,
        protected_resource_metadata_url: str | None = None,
        trust_gateway: bool = False,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.introspector = introspector
        self.required_scope = required_scope
        self.selfhost_token_validator = selfhost_token_validator
        self.trust_gateway = trust_gateway
        # Static API key + self-host OAuth paths both authenticate as a single
        # operator. Binding their requests to this UUID lets
        # user_scoped_connection set app.current_user_id uniformly across all
        # auth modes. When None, operator-bound requests reach DB code without
        # a user binding and MissingUserIdError surfaces as a 500.
        self.operator_user_id = operator_user_id
        # admin pool (BYPASSRLS) used for the pre-context JWT /userinfo path:
        # email-collision detection + INSERT during first-login JIT. This path
        # runs on the admin pool so it is RLS-policy-independent. When None,
        # the pre-context path is skipped; only background JIT (self.jit_pool)
        # can attempt provisioning.
        self.admin_pool = admin_pool
        # admin pool (alias for backwards-compatible background JIT).
        # DEPRECATED: use admin_pool for new DB work. Kept for the legacy
        # _jit_provision path which still uses jit_pool for UPSERT.
        self.jit_pool = jit_pool
        # Hydra public URL base for /userinfo endpoint. Fetched only on the
        # JWT-path or JIT path to source an email for the users row. Null
        # disables email fetching; provisioning skips without blocking.
        self.hydra_public_url = hydra_public_url
        # URL of the OAuth protected-resource metadata document. Surfaced in
        # WWW-Authenticate on 401/403 per MCP spec 2025-11-25 so clients can
        # discover the authorization server. None disables the parameter
        #         (appropriate for Mode 1 API-key-only deployments with no OAuth
        # routes to discover).
        self.protected_resource_metadata_url = protected_resource_metadata_url
        # In-process LRU cache for known-provisioned sub UUIDs. Avoids
        # redundant /userinfo HTTP calls and DB lookups after first login.
        self._provisioned_cache = _ProvisionedCache()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Trust gateway mode: skip all auth logic and trust the upstream
        # X-Auth-User-Id header set by cloud-api. This is used by hosted
        # Mode 3 deployments behind cloud-api.
        # DEPLOYMENT INVARIANT: this path trusts upstream cloud-api;
        # journalctl must not be directly internet-reachable when
        # trust_gateway=True.
        if self.trust_gateway:
            request = Request(scope)
            user_id_header = request.headers.get("x-auth-user-id", "")
            if not user_id_header:
                response = JSONResponse(
                    {"error": "Missing X-Auth-User-Id header"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            try:
                user_uuid = UUID(user_id_header)
            except (ValueError, AttributeError):
                response = JSONResponse(
                    {"error": "Invalid X-Auth-User-Id header"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            token_reset = current_user_id.set(user_uuid)
            scope_reset = current_token_scopes.set(frozenset({"journal"}))
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
                current_token_scopes.reset(scope_reset)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")

        if not auth_header:
            await _unauthorized(
                "Missing or invalid Authorization header", self.protected_resource_metadata_url
            )(scope, receive, send)
            return

        if not auth_header.lower().startswith("bearer "):
            await _unauthorized(
                "Missing or invalid Authorization header", self.protected_resource_metadata_url
            )(scope, receive, send)
            return

        token = auth_header[7:]

        if len(token) > MAX_BEARER_TOKEN_LEN:
            await _unauthorized("Invalid token", self.protected_resource_metadata_url)(
                scope, receive, send
            )
            return

        # Mode 1: Static API key (timing-safe comparison).
        # Empty api_key disables the path entirely (Mode 3 passes ""); the
        # explicit truthiness check prevents an empty-vs-empty match.
        if self.api_key and secrets.compare_digest(token, self.api_key):
            await self._call_with_operator(scope, receive, send)
            return

        # Mode 2: Hydra introspection (Ory access tokens)
        if token.startswith("ory_at_") and self.introspector is not None:
            try:
                claims = await self.introspector.introspect(token)
            except HydraUnreachable:
                await _service_unavailable()(scope, receive, send)
                return
            except HydraInvalidToken:
                await _unauthorized(
                    "Invalid or expired token", self.protected_resource_metadata_url
                )(scope, receive, send)
                return

            scopes = claims.scope.split()
            if not check_scope(set(scopes), self.required_scope):
                await _forbidden(self.required_scope, self.protected_resource_metadata_url)(
                    scope, receive, send
                )
                return

            # Store scopes for @require_scope decorator (per-tool checks).
            token_scope_reset = current_token_scopes.set(frozenset(scopes))

            # Defense-in-depth: TokenClaims.sub is typed UUID and parsed via UUID(sub_raw)
            # in HydraIntrospector, but mirror db_context's isinstance guard so a bypass
            # path (test mock, future cache deserializer) cannot stash a str into the ctxvar.
            if not isinstance(claims.sub, UUID):
                await _unauthorized(
                    "Invalid or expired token", self.protected_resource_metadata_url
                )(scope, receive, send)
                return

            # ---------- pre-context JWT provision ---------------
            # Runs on admin_pool (BYPASSRLS) so this path is
            # RLS-policy-independent.  Email collision causes an early
            # 401 before any contextvar is set; the request never
            # reaches downstream code.

            sub = claims.sub  # noqa: SH102

            if self.admin_pool is not None:
                try:
                    await self._pre_context_jwt_provision(sub, token)
                except _EmailCollision as exc:  # noqa: PERF203
                    jlog = structlog.get_logger("auth.jwt")
                    jlog.error(
                        "JWT email collision -- rejecting request",
                        sub=str(sub),
                        colliding_sub=exc.colliding_sub,
                        email=exc.email,
                    )
                    # Record audit + respond 401 with WWW-Authenticate
                    try:
                        async with self.admin_pool.acquire() as conn:
                            await record_audit(
                                conn,
                                actor_type="hydra_subject",
                                actor_id=str(sub),
                                action="auth.email_collision",
                                target_type="user",
                                target_id=exc.colliding_sub,
                                metadata={"email": exc.email},
                            )
                    except Exception:
                        jlog.exception("failed to write email_collision audit row")

                    await _unauthorized(
                        "Account email collision -- contact support",
                        self.protected_resource_metadata_url,
                    )(scope, receive, send)
                    return
                except _PreExistingSub:
                    # Sub already exists in the DB; nothing more to do.
                    pass
                except Exception:
                    jlog = structlog.get_logger("auth.jwt")
                    jlog.exception(
                        "JWT provision: unexpected exception during pre-context provisioning",
                        sub=str(sub),
                    )
                    await _service_unavailable()(scope, receive, send)
                    return
            else:
                # Backwards-compatible path for single-tenant deploys.
                # No pre-context checks run; legacy JIT provision will
                # UPSERT in the background (ON CONFLICT DO NOTHING).
                await self._jit_provision(claims, token)

            token_reset = current_user_id.set(sub)
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
                current_token_scopes.reset(token_scope_reset)
            return

        # Mode 3: Self-host OAuth callback
        if self.selfhost_token_validator is not None and self.selfhost_token_validator(token):
            await self._call_with_operator(scope, receive, send)
            return

        # None of the modes accepted the token
        await _unauthorized("Invalid or expired token", self.protected_resource_metadata_url)(
            scope, receive, send
        )

    async def _pre_context_jwt_provision(self, sub: UUID, token: str) -> None:
        """Check existence + optional /userinfo for a new Hydra subject.

        Runs as part of the pre-context JWT validation path so that email
        collisions are detected before the request is authenticated. This must
        be fast (single round-trip check, then one INSERT) and never swallow
        real errors -- only ``_PreExistingSub`` for cache-hit / DB-found subjects
        which simply means "no new provisioning needed, skip /userinfo".

        The admin_pool connection is released between the id existence check and
        the /userinfo HTTP round-trip to prevent pool exhaustion under concurrent
        first-logins.

        User-row write paths (M2 review #6):

        * **This path (Mode 3 self-heal)** -- lazy. Fires on first
          authenticated MCP call from a Hydra subject that has no
          ``users`` row, typically because the Kratos webhook missed
          (cloud-api blip during signup). Idempotent on ``users.id``;
          on race, the loser raises ``_PreExistingSub`` and skips the
          /userinfo + audit writes.
        * **Kratos webhook (Mode 3 fast path)** --
          ``journalctl-cloud/journalctl_cloud/webhooks/kratos.py``
          ``_upsert_user``; eager + synchronous. The authoritative
          path for a normal signup; this JIT only fires when the
          webhook didn't.
        * **scaffold_operator (Mode 1/2 self-host)** --
          ``journalctl/users/bootstrap.py``; runs at startup. Disjoint
          from Mode 3.

        M3 refactor (m2-review #1): this JIT path moves into cloud-api
        alongside the Kratos webhook; the two converge in one service.

        Parameters
        ----------
        sub :
            Subject UUID from introspected Hydra token.
        token :
            Raw bearer token used to call Hydra /userinfo on miss.

        Raises
        ------
        _PreExistingSub
            When a users row already exists for this ``sub`` (by id OR by
            email -- the latter is also treated as pre-existing since it means
            the subject is *known* via another sub).
        _EmailCollision
            An active user row has the same email but a different sub.
            This is a hard rejection: call-site returns HTTP 401.
        """
        # Step 0: cache hit -> skip /userinfo + DB entirely. Fastest path.
        if self._provisioned_cache.find(sub):
            return

        logger = structlog.get_logger("auth.jwt")

        if self.admin_pool is None:
            raise _PreExistingSub()  # no pool available; treat as existing

        # First acquire block: SELECT-by-id existence check.
        async with self.admin_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM users WHERE id = $1 AND deleted_at IS NULL",
                sub,
            )
            if row is not None:
                # User exists by id -> no provisioning needed.
                logger.info("JWT provision: sub already found by id", sub=str(sub))
                self._provisioned_cache.put(sub)
                raise _PreExistingSub()
        # Connection released before /userinfo HTTP call.

        # Step 2: user does NOT exist -- attempt /userinfo (no DB conn held).
        email = None
        userinfo_resp = None
        if (
            self.hydra_public_url is not None
            and self.introspector is not None
            and self.introspector.http_client is not None
        ):
            try:
                userinfo_resp = await self.introspector.http_client.get(
                    f"{self.hydra_public_url}/userinfo",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if userinfo_resp.status_code == 200:
                    userinfo = userinfo_resp.json()
                    email = userinfo.get("email") or None
            except Exception:
                logger.warning("JWT provision: /userinfo failed, continuing", sub=str(sub))

        # Second acquire block: SELECT-by-email collision + INSERT.
        async with self.admin_pool.acquire() as conn:
            if email is not None:
                colliding = await conn.fetchrow(
                    "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
                    email,
                )
                if colliding is not None and str(colliding["id"]) != str(sub):
                    logger.error(
                        "JWT provision: email collision",
                        sub=str(sub),
                        colliding_sub=str(colliding["id"]),
                        email=email,
                    )
                    raise _EmailCollision(colliding_sub=str(colliding["id"]), email=email)

                # safe to INSERT.
                try:
                    await conn.execute(
                        "INSERT INTO users (id, email) VALUES ($1, $2)",
                        sub,
                        email,
                    )
                except asyncpg.exceptions.UniqueViolationError:
                    # TOCTOU: another request inserted between our SELECT and INSERT.
                    colliding = await conn.fetchrow(
                        "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
                        email,
                    )
                    if colliding is not None:
                        if str(colliding["id"]) != str(sub):
                            logger.error(
                                "JWT provision: email collision (race)",
                                sub=str(sub),
                                colliding_sub=str(colliding["id"]),
                                email=email,
                            )
                            raise _EmailCollision(
                                colliding_sub=str(colliding["id"]), email=email
                            ) from None
                        logger.info(
                            "JWT provision: new user row inserted (race, same sub)",
                            sub=str(sub),
                        )
                        # The other concurrent request that won the race owns
                        # the user.created audit row; we observe the outcome
                        # but do not double-audit.
                        self._provisioned_cache.put(sub)
                        return
                    raise

                logger.info("JWT provision: new user row inserted", sub=str(sub))
                # Audit the creation. Best-effort: a failure to write the
                # audit row must not break the auth path.
                try:
                    await record_audit(
                        conn,
                        actor_type="hydra_subject",
                        actor_id=str(sub),
                        action=Action.USER_CREATED,
                        target_type="user",
                        target_id=str(sub),
                        metadata={"provision_path": "jit"},
                    )
                except Exception:
                    logger.exception("JWT provision: user.created audit write failed")
                self._provisioned_cache.put(sub)
            elif email is None and userinfo_resp is not None and userinfo_resp.status_code == 200:
                # /userinfo returned 200 but has no email field; user authenticated
                # but no users row created -- emit warning for observability.
                logger.warning(
                    "JWT provision: /userinfo returned no email; "  # noqa: E501
                    "user authenticated but no users row created",
                    sub=str(sub),
                )

    async def _jit_provision(self, claims: TokenClaims, token: str) -> None:
        """Background UPSERT for users rows missing due to failed Kratos webhook.

        Runs AFTER the request contextvar is bound; used for backwards-compatible
        provisioning when ``admin_pool`` is not configured (single-tenant deploys).
        This method never blocks or rejects requests -- all errors are logged as warnings.

        Calls Hydra /userinfo to fetch the email mapped to this token's identity,
        then inserts or ignores (ON CONFLICT DO NOTHING) a users row keyed on
        claims.sub.  If jit_pool is None, hydra_public_url is None, or any step
        errors, the method logs a warning and returns without blocking the request.

        Kratos webhook remains the fast-path for normal logins; this JIT path
        handles orphaned identities that slipped through.
        """
        if self.jit_pool is None:
            return

        if self.hydra_public_url is None:
            structlog.get_logger("jit").warning(
                "JIT provisioning disabled -- no hydra_public_url configured"
            )
            return

        try:
            if self.introspector is None or self.introspector.http_client is None:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: no introspector HTTP client available"
                )
                return

            userinfo_resp = await self.introspector.http_client.get(
                f"{self.hydra_public_url}/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
            if userinfo_resp.status_code != 200:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: /userinfo returned non-200",
                    status_code=userinfo_resp.status_code,
                )
                return
            userinfo = userinfo_resp.json()
            email = userinfo.get("email")
            if not email:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: no email in /userinfo response"
                )
                return

            async with self.jit_pool.acquire() as conn:
                # UUID binds to $1; asyncpg resolves Python UUID → pg UUID type automatically
                await conn.execute(
                    """
                    INSERT INTO users (id, email)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    claims.sub,
                    email,
                )
        except Exception:
            structlog.get_logger("jit").warning(
                "JIT provisioning failed (non-blocking)",
                sub=str(claims.sub),
                exc_info=True,
            )

    async def _call_with_operator(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Invoke the wrapped app with current_user_id bound to the operator UUID.

        Used by operator-identity auth modes (static API key, self-host OAuth).
        When no operator UUID is configured the request returns a 503. This is
        not a silent security bypass.
        """
        if self.operator_user_id is None:
            response = JSONResponse(
                {
                    "error": (
                        "operator not provisioned; app should auto-scaffold on Mode 1/2 startup"
                    ),
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        token_reset = current_user_id.set(self.operator_user_id)
        scope_reset = current_token_scopes.set(frozenset({"journal"}))
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token_reset)
            current_token_scopes.reset(scope_reset)
