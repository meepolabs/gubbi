"""Hydra OAuth 2.0 token introspection with TTL cache."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

import httpx
import structlog
from cachetools import TTLCache  # type: ignore[import-untyped]


@dataclass(frozen=True)
class TokenClaims:
    sub: UUID
    scope: str  # raw space-delimited scope string from Hydra
    exp: int  # unix timestamp


class HydraError(Exception):
    pass


class HydraUnreachable(HydraError):
    pass


class HydraInvalidToken(HydraError):
    pass


class HydraCache(Protocol):
    def get(self, token_fp: str) -> TokenClaims | None: ...

    def set(self, token_fp: str, claims: TokenClaims) -> None: ...


class InMemoryHydraCache:
    def __init__(self, ttl_seconds: int = 30, maxsize: int = 10_000) -> None:
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def get(self, token_fp: str) -> TokenClaims | None:
        val = self._cache.get(token_fp)
        if val is None:
            return None
        return cast("TokenClaims", val)

    def set(self, token_fp: str, claims: TokenClaims) -> None:
        self._cache[token_fp] = claims


def _token_digest(token: str) -> tuple[str, str]:
    """Return (full_sha256_hex, 12_char_log_fp) — single hash, two views."""
    digest = hashlib.sha256(token.encode()).hexdigest()
    return digest, digest[:12]


def _cache_key(token: str) -> str:
    """Full sha256 hex — collision-resistant cache key (256 bits)."""
    return _token_digest(token)[0]


def _log_fp(token: str) -> str:
    """Truncated sha256 for log correlation only — not a cache key."""
    return _token_digest(token)[1]


_token_fp = _log_fp  # back-compat alias


class HydraIntrospector:
    def __init__(
        self,
        admin_url: str,
        http_client: httpx.AsyncClient,
        logger: structlog.stdlib.AsyncBoundLogger,
        cache: HydraCache | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.admin_url = admin_url.rstrip("/")
        self.http_client = http_client
        self.logger = logger
        self.cache = cache
        self.timeout_seconds = timeout_seconds

    async def introspect(self, token: str) -> TokenClaims:
        cache_key, fp = _token_digest(token)

        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            # httpx auto-sets Content-Type: application/x-www-form-urlencoded when data is a dict
            resp = await self.http_client.post(
                f"{self.admin_url}/oauth2/introspect",
                data={"token": token},
            )

            if resp.status_code >= 500:
                await self.logger.error(
                    "Hydra admin returned server error",
                    token_fp=fp,
                    status_code=resp.status_code,
                )
                raise HydraUnreachable(f"Hydra admin returned {resp.status_code}")

            if resp.status_code >= 400:
                await self.logger.error(
                    "Hydra admin returned client error",
                    token_fp=fp,
                    status_code=resp.status_code,
                )
                raise HydraUnreachable(f"Hydra admin returned {resp.status_code}")

            try:
                body = resp.json()
            except ValueError as exc:
                await self.logger.error(
                    "Hydra returned non-JSON body",
                    token_fp=fp,
                    status_code=resp.status_code,
                )
                raise HydraUnreachable("Hydra returned non-JSON body") from exc

            if not isinstance(body, dict):
                await self.logger.error(
                    "Hydra returned non-object JSON body",
                    token_fp=fp,
                    body_type=type(body).__name__,
                )
                raise HydraUnreachable("Hydra returned non-object JSON body")

        except httpx.TimeoutException:
            await self.logger.error("Hydra introspect timeout", token_fp=fp)
            raise HydraUnreachable("introspect timed out") from None
        except (httpx.ConnectError, httpx.NetworkError):
            # ConnectTimeout is a subclass of TimeoutException and is already handled above
            await self.logger.error("Hydra introspect connection failed", token_fp=fp)
            raise HydraUnreachable("Hydra unreachable") from None
        except httpx.HTTPStatusError as exc:
            await self.logger.error(
                "Hydra introspect HTTP error", token_fp=fp, status_code=exc.response.status_code
            )
            raise HydraUnreachable(f"Hydra HTTP error: {exc.response.status_code}") from None

        if body.get("active") is not True:
            await self.logger.info(
                "Token not active",
                token_fp=fp,
            )
            raise HydraInvalidToken("inactive token")

        sub_raw = body.get("sub")
        if not sub_raw:
            raise HydraInvalidToken("malformed sub")
        try:
            sub = UUID(sub_raw)
        except (ValueError, TypeError):
            raise HydraInvalidToken("malformed sub") from None

        scope = body.get("scope", "")
        exp = body.get("exp")
        if exp is None:
            await self.logger.error("Hydra response missing exp field", token_fp=fp)
            raise HydraInvalidToken("missing exp")
        if exp < time.time():
            await self.logger.info("Token expired (defense in depth)", token_fp=fp, exp=exp)
            raise HydraInvalidToken("expired")

        claims = TokenClaims(sub=sub, scope=scope, exp=exp)

        await self.logger.info(
            "Token introspected successfully",
            token_fp=fp,
            sub=str(sub),
            scopes=scope,
        )

        if self.cache is not None:
            self.cache.set(cache_key, claims)

        return claims
