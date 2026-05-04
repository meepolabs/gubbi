"""OAuth disabled mode -- no routes registered."""

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    return None
