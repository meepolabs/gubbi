from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables.

    All variables are prefixed with JOURNAL_ in the environment.
    Managed by Doppler in production, .env in local dev.
    """

    # Auth
    api_key: str

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JOURNAL_API_KEY must be at least 32 characters")
        return v

    # OAuth — empty owner_password_hash disables OAuth endpoints
    server_url: str = "http://localhost:8100"
    owner_password_hash: str = ""
    oauth_db_path: Path = Path("./data/oauth.db")
    oauth_access_token_ttl: int = 3600  # 1 hour
    oauth_refresh_token_ttl: int = 2592000  # 30 days
    oauth_auth_code_ttl: int = 300  # 5 minutes

    # Database — PostgreSQL connection string
    database_url: str = "postgresql://journal:journal@localhost:5432/journal"

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if v == "postgresql://journal:journal@localhost:5432/journal":
            raise ValueError(
                "JOURNAL_DATABASE_URL is not set — refusing to start with default credentials"
            )
        return v

    # Paths — override via JOURNAL_JOURNAL_ROOT
    journal_root: Path = Path("./journal")

    # Timezone — controls the "today" default for journal_append_entry and
    # journal_save_conversation when no explicit date is provided.
    # Set via JOURNAL_TIMEZONE (e.g. America/Los_Angeles). Defaults to UTC.
    timezone: str = "UTC"

    # Server
    host: str = "0.0.0.0"  # noqa: S104 — bind all interfaces for Docker
    port: int = 8100
    transport: str = "streamable-http"  # or "stdio"

    # Logging
    log_level: str = "info"
    log_dir: Path = Path("./logs")

    @property
    def knowledge_dir(self) -> Path:
        return self.journal_root / "knowledge"

    @property
    def conversations_json_dir(self) -> Path:
        return self.journal_root / "conversations_json"

    model_config = {"env_prefix": "JOURNAL_"}


@lru_cache
def get_settings() -> Settings:
    """Create and cache settings instance from environment variables."""
    return Settings()  # type: ignore[call-arg]
