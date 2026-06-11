"""Response builders for the web REST endpoints.

Centralizes the ``Cache-Control`` header so an endpoint returning decrypted
content cannot forget it. Decrypted journal content is per-user and must never
be cached by a shared proxy or the browser disk cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__: list[str] = [
    "PRIVATE_NO_STORE",
    "private_no_store_response",
]

# Applied to every response carrying decrypted user content. ``private``
# forbids shared-cache storage; ``no-store`` forbids any persistence at all.
PRIVATE_NO_STORE: str = "private, no-store"


def private_no_store_response(model: BaseModel) -> JSONResponse:
    """Serialize ``model`` to JSON with ``Cache-Control: private, no-store``.

    Use for any endpoint whose body contains decrypted content. The model is
    dumped in JSON mode (``by_alias=True``) so per-resource field aliases
    (e.g. ``items`` -> ``topics``) render as the spec requires.
    """
    response = JSONResponse(model.model_dump(mode="json", by_alias=True))
    response.headers["Cache-Control"] = PRIVATE_NO_STORE
    return response
