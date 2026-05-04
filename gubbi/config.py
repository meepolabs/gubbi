import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Hydra admin-introspect HTTP timeout, seconds. 3s is comfortable on a local
# docker network; it is not an operator-tunable.
HYDRA_INTROSPECT_TIMEOUT_SECS: Final[float] = 3.0

# OAuth scope every MCP token must carry. Product constant, not a knob.
REQUIRED_OAUTH_SCOPE: Final[str] = "journal"

# Allowed origins for the MCP streamable HTTP endpoint.  Used by
# OriginValidationMiddleware to prevent DNS-rebinding attacks.
# Loopback origins are always allowed; this allowlist is for production
# MCP clients (claude.ai, chatgpt.com, journal.gubbi.ai, mcp.gubbi.ai).
ALLOWED_ORIGINS: Final[frozenset[str]] = frozenset(
    {
        "https://claude.ai",
        "https://chatgpt.com",
        "https://journal.gubbi.ai",
        "https://mcp.gubbi.ai",
        "https://journal-dev.gubbi.ai",
        "https://mcp-dev.gubbi.ai",
    }
)

# OAuth token lifetimes. Protocol-level defaults; operators do not tune them.
OAUTH_ACCESS_TOKEN_TTL_SECS: Final[int] = 3600  # 1 hour
OAUTH_REFRESH_TOKEN_TTL_SECS: Final[int] = 2592000  # 30 days
OAUTH_AUTH_CODE_TTL_SECS: Final[int] = 300  # 5 minutes

_config_logger = logging.getLogger("gubbi.config")

# Maps old flat env var names (without JOURNAL_ prefix) to new double-underscore
# nested names that pydantic-settings v2 understands with env_nested_delimiter="__".
# JOURNAL_DB_APP_URL -> JOURNAL_DB__APP_URL -> settings.db.app_url
_FLAT_TO_NESTED_ENV: dict[str, str] = {
    "JOURNAL_DB_APP_URL": "JOURNAL_DB__APP_URL",
    "JOURNAL_DB_ADMIN_URL": "JOURNAL_DB__ADMIN_URL",
    "JOURNAL_HYDRA_ADMIN_URL": "JOURNAL_AUTH__HYDRA_ADMIN_URL",
    "JOURNAL_HYDRA_PUBLIC_ISSUER_URL": "JOURNAL_AUTH__HYDRA_PUBLIC_ISSUER_URL",
    "JOURNAL_HYDRA_PUBLIC_URL": "JOURNAL_AUTH__HYDRA_PUBLIC_URL",
    "JOURNAL_PASSWORD_HASH": "JOURNAL_AUTH__PASSWORD_HASH",
    "JOURNAL_API_KEY": "JOURNAL_AUTH__API_KEY",
    "JOURNAL_OPERATOR_EMAIL": "JOURNAL_AUTH__OPERATOR_EMAIL",
    "JOURNAL_TRUST_GATEWAY": "JOURNAL_AUTH__TRUST_GATEWAY",
    "JOURNAL_GUBBI_GATEWAY_SECRET": "JOURNAL_AUTH__GATEWAY_SECRET",
    "JOURNAL_GATEWAY_REQUIRE_SIGNATURE": "JOURNAL_AUTH__GATEWAY_REQUIRE_SIGNATURE",
    "JOURNAL_API_KEY_SCOPES": "JOURNAL_AUTH__API_KEY_SCOPES",
    "JOURNAL_SERVER_URL": "JOURNAL_SERVER__URL",
    "JOURNAL_HOST": "JOURNAL_SERVER__HOST",
    "JOURNAL_PORT": "JOURNAL_SERVER__PORT",
    "JOURNAL_TRANSPORT": "JOURNAL_SERVER__TRANSPORT",
    "JOURNAL_LLM_API_KEY": "JOURNAL_LLM__API_KEY",
    "JOURNAL_LLM_MODEL": "JOURNAL_LLM__MODEL",
}


class _FlatCompatEnvSource(EnvSettingsSource):
    """Env source that remaps legacy flat env vars to nested double-underscore form.

    pydantic-settings v2 caches os.environ at source __init__ time, so the
    injection must happen before super().__init__() loads env_vars. The
    injected keys are cleaned up in __del__ to avoid polluting the process env.

    This allows JOURNAL_DB_APP_URL (flat, legacy) to coexist with
    JOURNAL_DB__APP_URL (nested, new-style). When both are set, the nested
    form takes precedence (injection is skipped).
    """

    def __init__(self, settings_cls: type, **kwargs: Any) -> None:
        self._injected: list[str] = []
        for flat, nested in _FLAT_TO_NESTED_ENV.items():
            if flat in os.environ and nested not in os.environ:
                os.environ[nested] = os.environ[flat]
                self._injected.append(nested)
        super().__init__(settings_cls, **kwargs)

    def __del__(self) -> None:
        for k in self._injected:
            os.environ.pop(k, None)


class DbConfig(BaseModel):
    app_url: str
    admin_url: str = ""


class AuthConfig(BaseModel):
    api_key: str = ""
    hydra_admin_url: str = ""
    hydra_public_issuer_url: str = ""
    hydra_public_url: str | None = None
    password_hash: str = ""
    operator_email: str = ""
    trust_gateway: bool = False
    gateway_secret: str = ""
    gateway_require_signature: bool = False
    api_key_scopes: list[str] = ["journal:read", "journal:write"]
    # When True, client_ip() honours the leftmost X-Forwarded-For header as
    # the original client IP.  Requires a trusted reverse-proxy in front of
    # gubbi; default False for direct-to-container deploys (M-9.3).
    trust_forwarded_headers: bool = False

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        # Non-empty keys must still be strong. Length enforcement for the
        # "required vs optional" contract lives in the model validator below,
        # so Mode 3 can leave this empty without tripping the length check.
        if v and len(v) < 32:
            raise ValueError("JOURNAL_API_KEY must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def _warn_on_require_signature_without_secret(self) -> Self:
        if self.gateway_require_signature and not self.gateway_secret:
            _config_logger.warning(
                "JOURNAL_GATEWAY_REQUIRE_SIGNATURE=true but "
                "JOURNAL_GUBBI_GATEWAY_SECRET is empty -- set a hex-encoded "
                "shared secret (>= 64 hex chars) before enabling this "
                "feature in production"
            )
        return self


class ServerConfig(BaseModel):
    url: str = "http://localhost:8100"
    host: str = "0.0.0.0"  # noqa: S104 -- bind all interfaces for Docker
    port: int = 8100
    transport: str = "streamable-http"  # or "stdio"


class LLMConfig(BaseModel):
    """Optional LLM configuration for extraction services.

    All fields are optional (empty-string defaults) so self-hosters who
    do not use extraction can ignore them entirely.
    """

    api_key: str = ""
    model: str = ""


class Settings(BaseSettings):
    """Application settings, loaded from environment variables.

    All variables are prefixed with JOURNAL_ in the environment.
    Managed by Doppler in production, .env in local dev.

    The server supports three mutually-exclusive deploy shapes, selected by
    which of JOURNAL_HYDRA_ADMIN_URL, JOURNAL_PASSWORD_HASH, and
    JOURNAL_HYDRA_PUBLIC_ISSUER_URL are set:

    1. API-key-only self-host -- all three empty. JOURNAL_API_KEY is the only
       accepted credential. Useful for CLI-only deploys.
    2. Full self-host -- PASSWORD_HASH set, HYDRA fields empty. API key
       works AND self-host OAuth (the MCP SDK's DCR routes) works. One
       operator identity.
    3. Multi-tenant hosted -- HYDRA_ADMIN_URL + HYDRA_PUBLIC_ISSUER_URL set,
       PASSWORD_HASH empty. Hydra OAuth introspection handles every request;
       the static API key path is disabled. Operators use OAuth like any
       real user.

    Setting both HYDRA_ADMIN_URL and PASSWORD_HASH is a configuration
    error and fails startup.

    Legacy flat env var names (JOURNAL_DB_APP_URL, JOURNAL_API_KEY, etc.) are
    supported via _FlatCompatEnvSource. New-style double-underscore names
    (JOURNAL_DB__APP_URL, JOURNAL_AUTH__API_KEY) also work and take precedence
    when both are set.

    Additional hardening flags (M-9 cluster):
    - JOURNAL_AUTH__TRUST_FORWARDED_HEADERS: When True, client_ip() honours
      the leftmost X-Forwarded-For header (default False; M-9.3).
    - JOURNAL_HEALTH_BIND_PUBLIC: When set and "true", the extraction
      health server listens on 0.0.0.0 instead of 127.0.0.1 (default
      localhost-only; M-9.7).
    """

    db: DbConfig
    auth: AuthConfig = AuthConfig()
    server: ServerConfig = ServerConfig()
    llm: LLMConfig = LLMConfig()

    # Paths
    data_dir: Path = Path("./journal")

    # Redis -- used by extraction pub/sub SSE endpoint and worker queue.
    # Read from JOURNAL_REDIS_URL env var; falls back to localhost.
    redis_url: str = "redis://localhost:6379"

    # Timezone -- controls the "today" default for journal_append_entry and
    # journal_save_conversation when no explicit date is provided.
    timezone: str = "UTC"

    # Logging
    log_level: str = "info"
    log_dir: Path = Path("./logs")

    @model_validator(mode="after")
    def _validate_deploy_shape(self) -> Self:
        """Enforce the 3-shape matrix (HYDRA_ADMIN_URL and PASSWORD_HASH mutually exclusive).

        Running with both JOURNAL_HYDRA_ADMIN_URL and JOURNAL_PASSWORD_HASH
        set is never a valid configuration -- it would stack two different
        operator-identity bindings on top of each other. Fail loudly at
        startup so operators can't land a misconfigured deploy.

        Also enforce that JOURNAL_API_KEY is present unless Hydra is on.
        """
        hydra_on = bool(self.auth.hydra_admin_url)
        password_on = bool(self.auth.password_hash)
        hydra_issuer_on = bool(self.auth.hydra_public_issuer_url)
        hydra_puburl_on = bool(self.auth.hydra_public_url)

        # Existing: HYDRA_ADMIN_URL and PASSWORD_HASH are mutually exclusive.
        if hydra_on and password_on:
            raise ValueError(
                "JOURNAL_HYDRA_ADMIN_URL and JOURNAL_PASSWORD_HASH are "
                "mutually exclusive -- pick one deploy shape. See "
                "docs/deployment.md for the 3-shape matrix."
            )

        # Mode 3 requires HYDRA_ADMIN_URL + PUBLIC_ISSUER_URL together.
        if hydra_on and not hydra_issuer_on:
            raise ValueError(
                "JOURNAL_HYDRA_PUBLIC_ISSUER_URL is required when "
                "JOURNAL_HYDRA_ADMIN_URL is set -- both must be non-empty "
                "together for Mode 3 (multi-tenant hosted)."
            )
        if hydra_issuer_on and not hydra_on:
            raise ValueError(
                "JOURNAL_HYDRA_ADMIN_URL is required when "
                "JOURNAL_HYDRA_PUBLIC_ISSUER_URL is set -- both must be "
                "non-empty together for Mode 3 (multi-tenant hosted)."
            )

        # HYDRA_ADMIN_URL + PUBLIC_URL both-or-neither.
        # PUBLIC_URL is used for JIT /userinfo calls; without it the JIT
        # path can only no-op, which silently masks provisioning failures.
        if hydra_on and not hydra_puburl_on:
            raise ValueError(
                "JOURNAL_HYDRA_PUBLIC_URL is required when "
                "JOURNAL_HYDRA_ADMIN_URL is set -- both must be non-empty "
                "together for Mode 3 (multi-tenant hosted)."
            )
        if hydra_puburl_on and not hydra_on:
            raise ValueError(
                "JOURNAL_HYDRA_ADMIN_URL is required when "
                "JOURNAL_HYDRA_PUBLIC_URL is set -- both must be "
                "non-empty together for Mode 3 (multi-tenant hosted)."
            )

        # Mode 3: operator_email is irrelevant when Hydra handles
        # authentication; mixing them yields opaque failures, so reject.
        if hydra_on and self.auth.operator_email:
            raise ValueError(
                "JOURNAL_OPERATOR_EMAIL must not be set when JOURNAL_HYDRA_ADMIN_URL "
                "is set -- mode 3 (multi-tenant hosted) has no operator concept; "
                "remove the variable or unset Hydra to switch to mode 1/2."
            )

        if not hydra_on and not self.auth.api_key:
            raise ValueError(
                "JOURNAL_API_KEY is required unless JOURNAL_HYDRA_ADMIN_URL "
                "is set (hosted mode disables the static API key path)."
            )
        if not hydra_on and not self.auth.operator_email:
            raise ValueError(
                "JOURNAL_OPERATOR_EMAIL is required unless JOURNAL_HYDRA_ADMIN_URL "
                "is set -- Modes 1/2 bind every authenticated request to the "
                "operator UUID resolved from this email."
            )
        return self

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def conversations_json_dir(self) -> Path:
        return self.data_dir / "conversations_json"

    @property
    def oauth_db_path(self) -> Path:
        """SQLite file backing the self-host OAuth server (Mode 2)."""
        return self.data_dir / "oauth.db"

    model_config = SettingsConfigDict(env_prefix="JOURNAL_", env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _FlatCompatEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Create and cache settings instance from environment variables."""
    # pydantic-settings reads required fields from env; no Python-level kwargs needed.
    return Settings()
