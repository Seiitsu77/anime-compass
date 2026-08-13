from __future__ import annotations

from fastapi import Request


def container(request: Request):
    return request.app.state.container
