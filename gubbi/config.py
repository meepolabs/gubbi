import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from gubbi_common.bootstrap import PgLogProbeMode
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__: list[str] = [
    "ALLOWED_ORIGINS",
    "HYDRA_INTROSPECT_TIMEOUT_SECS",
    "OAUTH_ACCESS_TOKEN_TTL_SECS",
    "OAUTH_AUTH_CODE_TTL_SECS",
    "OAUTH_REFRESH_TOKEN_TTL_SECS",
    "REQUIRED_OAUTH_SCOPE",
    "AuthConfig",
    "DbConfig",
    "Environment",
    "LLMConfig",
    "ServerConfig",
    "Settings",
    "get_settings",
]

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

# Stays on stdlib ``logging`` because the only emit is in a Pydantic
# ``@model_validator(mode="after")`` which runs sync. ``structlog.AsyncBoundLogger``
# emits return coroutines that must be awaited; sync callers cannot use it.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs.
#
# Each sub-config is a ``BaseSettings`` whose fields declare a
# ``validation_alias`` carrying the FULL flat env-var name (e.g.
# ``JOURNAL_API_KEY``). The parent ``Settings`` constructs each sub-config
# via ``default_factory``; pydantic-settings on the child reads env vars
# independently using the per-field aliases. The double-underscore nested
# form (``JOURNAL_AUTH__API_KEY``) is INTENTIONALLY not honoured -- the
# external contract is flat-only. The internal class hierarchy
# (``settings.auth.api_key`` etc.) is preserved purely for code organization.
# ---------------------------------------------------------------------------


class DbConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    app_url: str = Field(..., validation_alias="JOURNAL_DB_APP_URL")
    admin_url: str = Field(default="", validation_alias="JOURNAL_DB_ADMIN_URL")


class AuthConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = Field(default="", validation_alias="JOURNAL_API_KEY")
    hydra_admin_url: str = Field(default="", validation_alias="JOURNAL_HYDRA_ADMIN_URL")
    hydra_public_issuer_url: str = Field(
        default="", validation_alias="JOURNAL_HYDRA_PUBLIC_ISSUER_URL"
    )
    hydra_public_url: str | None = Field(default=None, validation_alias="JOURNAL_HYDRA_PUBLIC_URL")
    password_hash: str = Field(default="", validation_alias="JOURNAL_PASSWORD_HASH")
    operator_email: str = Field(default="", validation_alias="JOURNAL_OPERATOR_EMAIL")
    trust_gateway: bool = Field(default=False, validation_alias="JOURNAL_TRUST_GATEWAY")
    gateway_secret: str = Field(default="", validation_alias="JOURNAL_GUBBI_GATEWAY_SECRET")
    gateway_require_signature: bool = Field(
        default=True, validation_alias="JOURNAL_GATEWAY_REQUIRE_SIGNATURE"
    )
    # ``NoDecode`` here suppresses pydantic-settings' default JSON
    # decoding of complex types so the env source delivers the raw
    # string to ``_split_csv_scopes``. The annotation is honored by
    # pydantic-settings 2.x; if a future release stops respecting NoDecode,
    # the validator below will see a pre-decoded list and pass it through,
    # but a CSV operator value would fail at the env source's JSON-decode
    # step with a confusing message. Verify the CSV path still works after
    # each pydantic-settings major bump.
    api_key_scopes: Annotated[list[str], NoDecode] = Field(
        default=["journal:read", "journal:write"],
        validation_alias="JOURNAL_API_KEY_SCOPES",
    )
    # When True, client_ip() honours the rightmost X-Forwarded-For entry --
    # the IP added by the trusted edge proxy -- as the original client IP
    # (DEC-086 rule 4). Requires a trusted reverse-proxy in front of
    # gubbi; default False for direct-to-container deploys (M-9.3).
    trust_forwarded_headers: bool = Field(
        default=False, validation_alias="JOURNAL_TRUST_FORWARDED_HEADERS"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Reject keys shorter than 32 chars (empty allowed for Mode 3)."""
        # Non-empty keys must still be strong. Length enforcement for the
        # "required vs optional" contract lives in the model validator on
        # ``Settings``, so Mode 3 can leave this empty without tripping the
        # length check.
        if v and len(v) < 32:
            raise ValueError("JOURNAL_API_KEY must be at least 32 characters")
        return v

    @field_validator("operator_email", mode="before")
    @classmethod
    def _strip_operator_email(cls, v: object) -> object:
        """Normalize whitespace-only inputs to empty string.

        Pre-launch hardening: a single-space OPERATOR_EMAIL would otherwise
        pass the not-empty deploy-shape check (whitespace is truthy) and
        fail later at operator-UUID lookup. Strip at field-input boundary
        so the deploy-shape validator catches it.
        """
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("api_key_scopes", mode="before")
    @classmethod
    def _split_csv_scopes(cls, v: object) -> Any:
        """Split a comma- or newline-separated string into a list.

        ``NoDecode`` on the field suppresses pydantic-settings' JSON
        decoding of complex types so the env source delivers the raw
        string here. CSV is the only accepted env shape -- matches
        ``cors_allowed_origins`` and the convention everywhere else for
        env-driven scope/origin lists.

        Detection of JSON-array shape uses ``startswith("[")`` only --
        every JSON-array string starts with ``[``, so the leading-bracket
        check catches all malicious inputs without false-positiving on
        any legit scope value. Splitting accepts ``,`` and ``\\n`` /
        ``\\r\\n`` as separators so a Doppler import that introduces CRLF
        does not degrade into a single malformed scope.

        Example: ``JOURNAL_API_KEY_SCOPES=journal:read,journal:write``
        """
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                raise ValueError(
                    "api_key_scopes expects a comma-separated string "
                    "(e.g. 'journal:read,journal:write'), not a JSON "
                    "array. Drop the brackets and quotes."
                )
            return [s for item in re.split(r"[,\n\r]+", stripped) if (s := item.strip())]
        return v

    @model_validator(mode="after")
    def _warn_on_require_signature_without_secret(self) -> Self:
        if self.gateway_require_signature and not self.gateway_secret:
            logger.warning(
                "JOURNAL_GATEWAY_REQUIRE_SIGNATURE=true but "
                "JOURNAL_GUBBI_GATEWAY_SECRET is empty -- set a hex-encoded "
                "shared secret (>= 64 hex chars) before enabling this "
                "feature in production"
            )
        return self


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    url: str = Field(default="http://localhost:8100", validation_alias="JOURNAL_SERVER_URL")
    host: str = Field(
        default="0.0.0.0",  # noqa: S104 -- bind all interfaces for Docker
        validation_alias="JOURNAL_HOST",
    )
    port: int = Field(default=8100, validation_alias="JOURNAL_PORT")
    transport: str = Field(default="streamable-http", validation_alias="JOURNAL_TRANSPORT")


class LLMConfig(BaseSettings):
    """Optional LLM configuration for extraction services.

    All fields are optional (empty-string defaults) so self-hosters who
    do not use extraction can ignore them entirely.
    """

    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = Field(default="", validation_alias="JOURNAL_LLM_API_KEY")
    provider: str = Field(default="anthropic", validation_alias="JOURNAL_LLM_PROVIDER")
    model: str = Field(default="", validation_alias="JOURNAL_LLM_MODEL")
    # Gates budget delta writes from worker (hosted: True; self-host: False).
    llm_budget_enabled: bool = Field(default=False, validation_alias="JOURNAL_LLM_BUDGET_ENABLED")
    # Stale pending-job sweep window.
    orphan_cleanup_threshold_minutes: int = Field(
        default=30, validation_alias="JOURNAL_ORPHAN_CLEANUP_THRESHOLD_MINUTES"
    )


# Canonical Environment Literal (mirrored byte-for-byte across gubbi + gubbi-cloud).
# See DEC-094 (canonical app_env literal); env-contract-lint enforces parity.
Environment = Literal["dev", "ci", "staging", "production"]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables.

    Env vars are flat-named only (e.g. ``JOURNAL_API_KEY``,
    ``JOURNAL_DB_APP_URL``). The internal class structure
    (``settings.auth.api_key`` etc.) is a code-organization convenience and
    does not affect the env-var contract.

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

    Additional hardening flags (M-9 cluster):
    - JOURNAL_TRUST_FORWARDED_HEADERS: When True, client_ip() honours
      the rightmost X-Forwarded-For entry -- the trusted-proxy stamp per
      DEC-086 rule 4 -- (default False; M-9.3).
    - JOURNAL_HEALTH_BIND_PUBLIC: When set and "true", the extraction
      health server listens on 0.0.0.0 instead of 127.0.0.1 (default
      localhost-only; M-9.7).
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="JOURNAL_",
        # ``env_prefix_target`` defaults to ``"variable"`` in pydantic-settings
        # (sources/base.py). Pinned here for reviewer clarity: the env_prefix
        # above is NOT prepended to per-field ``validation_alias`` values, so
        # each sub-config keeps reading its own per-field flat env-vars
        # unchanged.
        env_prefix_target="variable",
    )

    # Deploy environment marker. Controls safe-by-default gates that should
    # only relax in local development. Env var: JOURNAL_APP_ENV.
    # See DEC-094 (canonical app_env literal); default stays "dev" to honour
    # the self-host first principle.
    app_env: Environment = Field(default="dev", validation_alias="JOURNAL_APP_ENV")

    pg_log_probe_mode: PgLogProbeMode = Field(
        default=PgLogProbeMode.STRICT,
        validation_alias="JOURNAL_PG_LOG_PROBE_MODE",
        description=(
            "PG log probe mode: STRICT raises on unsafe GUC; WARN logs; "
            "OFF skips. STRICT in dev/prod; OFF only when an operator "
            "intentionally bypasses the cluster-log leakage guard."
        ),
    )
    replica_count: int = Field(
        default=1,
        ge=1,
        validation_alias="JOURNAL_REPLICA_COUNT",
        description=(
            "Number of pod replicas behind the gateway, used by the "
            "connection-budget guard to size pool ceilings."
        ),
    )

    # Sub-configs construct themselves from per-field flat env-vars via
    # ``default_factory``. The ``validation_alias`` here is a STRUCTURAL
    # GUARD SENTINEL, not a real env-var name -- pydantic-settings looks
    # up this name as JSON before falling back to ``default_factory``.
    # Sentinel values are intentionally never set in any environment, so
    # the JSON-decode bypass class (where ``AUTH={"api_key":"..."}`` would
    # inject a nested config) is closed.
    db: DbConfig = Field(
        default_factory=DbConfig,
        validation_alias="JOURNAL_NESTED_ANCHOR_DB",
    )
    auth: AuthConfig = Field(
        default_factory=AuthConfig,
        validation_alias="JOURNAL_NESTED_ANCHOR_AUTH",
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        validation_alias="JOURNAL_NESTED_ANCHOR_SERVER",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        validation_alias="JOURNAL_NESTED_ANCHOR_LLM",
    )

    # Paths
    data_dir: Path = Field(default=Path("./journal"), validation_alias="JOURNAL_DATA_DIR")

    # Redis -- used by extraction pub/sub SSE endpoint and worker queue.
    # Read from JOURNAL_REDIS_URL env var; falls back to localhost.
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="JOURNAL_REDIS_URL")

    # Timezone -- controls the "today" default for journal_append_entry and
    # journal_save_conversation when no explicit date is provided.
    timezone: str = Field(default="UTC", validation_alias="JOURNAL_TIMEZONE")

    # Logging
    log_level: str = Field(default="info", validation_alias="JOURNAL_LOG_LEVEL")
    log_dir: Path = Field(default=Path("./logs"), validation_alias="JOURNAL_LOG_DIR")

    @model_validator(mode="before")
    @classmethod
    def _force_subconfig_default_factory(cls, data: Any) -> Any:
        """Defense-in-depth: drop dict-shaped inputs for sub-config anchors.

        The sentinel ``validation_alias`` (JOURNAL_NESTED_ANCHOR_<X>) closes
        the bare-name JSON-decode bypass class (``AUTH={...}``,
        ``JOURNAL_AUTH={...}``). This validator closes the residual case
        where a misconfigured environment sets the sentinel itself with
        alias-keyed JSON.

        Sub-configs MUST construct from their own per-field flat env vars
        via ``default_factory``. This validator drops dict-shaped inputs
        for the four sub-config anchors at parent-validation phase so the
        ``default_factory`` always runs.

        Limitation: this validator is NOT a general-purpose "preserve
        explicit BaseSettings instances" guard. Direct kwargs construction
        like ``Settings(auth=AuthConfig(...))`` is supported only for cases
        where the env still satisfies all required fields of every
        sub-config (because each ``BaseSettings`` sub-config still reads
        its own env at construction time).
        """
        if isinstance(data, dict):
            for key in (
                "JOURNAL_NESTED_ANCHOR_DB",
                "JOURNAL_NESTED_ANCHOR_AUTH",
                "JOURNAL_NESTED_ANCHOR_SERVER",
                "JOURNAL_NESTED_ANCHOR_LLM",
                "db",
                "auth",
                "server",
                "llm",
            ):
                value = data.get(key)
                if value is not None and not isinstance(value, BaseModel):
                    # Drop dict-shaped (or string-shaped) inputs; keep
                    # already-constructed BaseSettings/BaseModel instances
                    # so explicit Settings(auth=AuthSettings(...)) at the
                    # call site (tests) still works.
                    data.pop(key)
        return data

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

    @model_validator(mode="after")
    def _validate_trust_gateway_signature(self) -> Self:
        """Refuse trust_gateway=True without signature enforcement in deployed envs.

        Gated on ``self.is_deployed`` (True for staging+production) per
        DEC-094. The unsafe combination is reachable in non-deployed
        envs (``dev``, ``ci``) for local trust-gateway smoke tests and
        the CI harness, but never in ``staging``/``production``.
        """
        if self.auth.trust_gateway and not self.auth.gateway_require_signature and self.is_deployed:
            raise ValueError(
                "auth.trust_gateway=True requires "
                "auth.gateway_require_signature=True in deployed envs "
                "(staging/production)"
            )
        return self

    @property
    def is_deployed(self) -> bool:
        """Return True when running in a deployed environment.

        Canonical predicate replacing every ``app_env != "dev"`` check across
        both gubbi and gubbi-cloud (DEC-094). dev + ci are non-deployed
        (developer laptop, CI runner); staging + production are deployed.
        """
        return self.app_env in ("staging", "production")

    @property
    def knowledge_dir(self) -> Path:
        """Filesystem location of user-knowledge markdown (profile, key facts)."""
        return self.data_dir / "knowledge"

    @property
    def conversations_json_dir(self) -> Path:
        """Filesystem location of archived conversation JSON blobs."""
        return self.data_dir / "conversations_json"

    @property
    def oauth_db_path(self) -> Path:
        """SQLite file backing the self-host OAuth server (Mode 2)."""
        return self.data_dir / "oauth.db"


@lru_cache
def get_settings() -> Settings:
    """Create and cache settings instance from environment variables."""
    # pydantic-settings reads required fields from env; no Python-level kwargs needed.
    return Settings()
