"""OAuth module constants."""

from typing import Final

CSRF_COOKIE_NAME: Final = "_journal_csrf"

# Maximum accepted Bearer token length — rejects oversized inputs before any DB lookup
MAX_BEARER_TOKEN_LEN: Final = 256
