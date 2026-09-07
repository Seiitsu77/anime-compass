from __future__ import annotations

from fastapi import Request

from app.core.errors import AppError
from app.core.startup import FAILED, READY, ApplicationStartup


def startup(request: Request) -> ApplicationStartup:
    """The initialization job and its state.

    Endpoints that must answer during a cold start -- health and readiness --
    depend on this rather than on the container, because the container does not
    exist yet.
    """
    return request.app.state.startup


async def container(request: Request):
    """The built application, or a deterministic refusal while it is building.

    Initialization moved off the ASGI startup path so the server can listen
    immediately, which means a request can arrive before the models are loaded.
    Rather than blocking indefinitely or starting a second build, a caller waits
    briefly on the one shared task and is then told plainly that the service is
    still warming.

    The grace exists because initialization is often nearly done -- or, in a
    test, already effectively instant -- and failing a request that would have
    been servicable a few milliseconds later is needless. It is bounded, and it
    waits on the existing task, so it can neither hang nor duplicate work.
    """
    state: ApplicationStartup = request.app.state.startup

    if state.state == READY:
        return state.container

    if state.state == FAILED:
        raise AppError(
            "Application initialization failed",
            code="initialization_failed",
            status_code=503,
        )

    resolved = await state.wait_for_ready(state.grace_seconds)

    if resolved == READY:
        return state.container
    if resolved == FAILED:
        raise AppError(
            "Application initialization failed",
            code="initialization_failed",
            status_code=503,
        )
    raise AppError(
        "Application is still loading its models",
        code="service_warming",
        status_code=503,
    )
