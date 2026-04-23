from functools import lru_cache
from pathlib import Path
from typing import Final, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

# Hydra admin-introspect HTTP timeout, seconds. 3s is comfortable on a local
# docker network; it is not an operator-tunable.
HYDRA_INTROSPECT_TIMEOUT_SECS: Final[float] = 3.0

# OAuth scope every MCP token must carry. Product constant, not a knob.
REQUIRED_OAUTH_SCOPE: Final[str] = "journal"

# OAuth token lifetimes. Protocol-level defaults; operators do not tune them.
OAUTH_ACCESS_TOKEN_TTL_SECS: Final[int] = 3600  # 1 hour
OAUTH_REFRESH_TOKEN_TTL_SECS: Final[int] = 2592000  # 30 days
OAUTH_AUTH_CODE_TTL_SECS: Final[int] = 300  # 5 minutes


class Settings(BaseSettings):
    """Application settings, loaded from environment variables.

    All variables are prefixed with JOURNAL_ in the environment.
    Managed by Doppler in production, .env in local dev.

    The server supports three mutually-exclusive deploy shapes, selected by
    which of JOURNAL_HYDRA_ADMIN_URL and JOURNAL_PASSWORD_HASH are set:

    1. API-key-only self-host -- both empty. JOURNAL_API_KEY is the only
       accepted credential. Useful for CLI-only deploys.
    2. Full self-host -- PASSWORD_HASH set, HYDRA_ADMIN_URL empty. API key
       works AND self-host OAuth (the MCP SDK's DCR routes) works. One
       operator identity.
    3. Multi-tenant hosted -- HYDRA_ADMIN_URL set, PASSWORD_HASH empty.
       Hydra OAuth introspection handles every request; the static API
       key path is disabled. Operators use OAuth like any real user.

    Setting both HYDRA_ADMIN_URL and PASSWORD_HASH is a configuration
    error and fails startup.
    """

    # Auth -- static Bearer token. Required in Modes 1/2; ignored (and not
    # required) in Mode 3 where Hydra owns every request.
    api_key: str = ""

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        # Non-empty keys must still be strong. Length enforcement for the
        # "required vs optional" contract lives in the model validator below,
        # so Mode 3 can leave this empty without tripping the length check.
        if v and len(v) < 32:
            raise ValueError("JOURNAL_API_KEY must be at least 32 characters")
        return v

    # Hydra OAuth 2.1 introspection -- empty = Mode 3 (hosted) disabled.
    hydra_admin_url: str = ""

    # Self-host OAuth -- empty password_hash = Mode 2 (full self-host) disabled.
    password_hash: str = ""
    server_url: str = "http://localhost:8100"

    # Runtime database DSN (role: journal_app, no BYPASSRLS). Required; pydantic
    # raises a missing-field ValidationError if unset.
    db_app_url: str

    # Admin DSN (role: journal_admin, BYPASSRLS). Used by reindex and cross-
    # tenant ops. Empty = fall back to the runtime pool (OK in single-tenant
    # dev before RLS is live).
    db_admin_url: str = ""

    # Operator identity -- binds the static API key + self-host OAuth paths
    # to a concrete user UUID so user_scoped_connection works uniformly
    # regardless of auth mode. The UUID is resolved at startup by looking up
    # users.email = JOURNAL_OPERATOR_EMAIL. Empty = no operator binding;
    # operator-identity tool calls reach DB code without a user id and
    # MissingUserIdError surfaces as a 500. Required in Modes 1/2 (enforced
    # by _validate_deploy_shape below); ignored in Mode 3 where every request
    # carries its own user UUID in the Hydra token.
    operator_email: str = ""

    # Paths
    data_dir: Path = Path("./journal")

    # Timezone -- controls the "today" default for journal_append_entry and
    # journal_save_conversation when no explicit date is provided.
    timezone: str = "UTC"

    # Server
    host: str = "0.0.0.0"  # noqa: S104 -- bind all interfaces for Docker
    port: int = 8100
    transport: str = "streamable-http"  # or "stdio"

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
        if self.hydra_admin_url and self.password_hash:
            raise ValueError(
                "JOURNAL_HYDRA_ADMIN_URL and JOURNAL_PASSWORD_HASH are "
                "mutually exclusive -- pick one deploy shape. See "
                "docs/deployment.md for the 3-shape matrix."
            )
        if not self.hydra_admin_url and not self.api_key:
            raise ValueError(
                "JOURNAL_API_KEY is required unless JOURNAL_HYDRA_ADMIN_URL "
                "is set (hosted mode disables the static API key path)."
            )
        if not self.hydra_admin_url and not self.operator_email:
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

    model_config = {"env_prefix": "JOURNAL_"}


@lru_cache
def get_settings() -> Settings:
    """Create and cache settings instance from environment variables."""
    # pydantic-settings reads required fields from env; no Python-level kwargs needed.
    return Settings()  # type: ignore[call-arg]
