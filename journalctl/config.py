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
    oauth_db_path: Path = Path("./oauth.db")
    oauth_access_token_ttl: int = 3600  # 1 hour
    oauth_refresh_token_ttl: int = 2592000  # 30 days
    oauth_auth_code_ttl: int = 300  # 5 minutes

    # Paths — override via JOURNAL_JOURNAL_ROOT / JOURNAL_DB_PATH
    journal_root: Path = Path("./journal")
    db_path: Path = Path("./journal.db")

    # Server
    host: str = "0.0.0.0"  # noqa: S104 — bind all interfaces for Docker
    port: int = 8100
    transport: str = "streamable-http"  # or "stdio"

    # Memory service (semantic memory via mcp-memory-service)
    memory_db_path: Path = Path("./memory.db")

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
