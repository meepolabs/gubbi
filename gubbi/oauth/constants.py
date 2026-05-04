"""OAuth module constants."""

from typing import Final

CSRF_COOKIE_NAME: Final = "_journal_csrf"

# Maximum accepted Bearer token length — rejects oversized inputs before any DB lookup
MAX_BEARER_TOKEN_LEN: Final = 256

# Rate limiting — persisted per-IP in oauth.db rate_limit_events table
LOGIN_MAX_FAILURES: Final = 10
LOGIN_LOCKOUT_WINDOW_SECS: Final = 300  # 5 min

REGISTER_MAX_ATTEMPTS: Final = 10
REGISTER_WINDOW_SECS: Final = 3600  # 1 hour

# Prune rate_limit_events rows older than this on each cleanup_expired() call.
# Must be >= max(LOGIN_LOCKOUT_WINDOW_SECS, REGISTER_WINDOW_SECS).
RATE_LIMIT_EVENT_RETENTION_SECS: Final = 3600
