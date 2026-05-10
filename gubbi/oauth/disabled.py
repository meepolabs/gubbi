"""OAuth disabled mode -- no routes registered."""

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    """No-op: register no OAuth routes (Mode 1/3 -- gateway or API-key auth)."""
