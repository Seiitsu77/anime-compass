"""Serving must not wait on model loading.

Vercel gives a container roughly fifteen seconds to accept a TCP connection.
Initialization takes about half a minute on one vCPU, and it used to run inside
the ASGI lifespan -- so uvicorn logged "Waiting for application startup" and the
platform gave up before the socket existed.

The fix is architectural: the lifespan starts one background job and returns.
That trades an unreachable server for a reachable one that is honest about not
being ready yet, and these tests pin both halves of that trade.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import container as container_dependency
from app.core.errors import AppError
from app.core.startup import FAILED, READY, STARTING, ApplicationStartup, PhaseRecorder


class Sentinel:
    """Stands in for the built application container."""

    def __init__(self) -> None:
        self.name = "container"


def slow_build(seconds: float, counter: list[int] | None = None):
    def build(recorder: PhaseRecorder) -> Sentinel:
        if counter is not None:
            counter.append(1)
        with recorder.phase("pretend_model_load"):
            time.sleep(seconds)
        return Sentinel()

    return build


def failing_build(recorder: PhaseRecorder) -> Sentinel:
    with recorder.phase("pretend_model_load"):
        raise RuntimeError("artifact checksum mismatch")


# ------------------------------------------------------- the state machine


@pytest.mark.asyncio
async def test_startup_returns_before_initialization_finishes():
    """The whole point: starting the job must not mean waiting for it."""
    startup = ApplicationStartup(slow_build(0.4))
    began = time.perf_counter()
    startup.start()
    assert (time.perf_counter() - began) < 0.1
    assert startup.state == STARTING

    assert await startup.wait_for_ready(10) == READY
    assert isinstance(startup.container, Sentinel)


@pytest.mark.asyncio
async def test_readiness_is_false_while_initializing_and_true_after():
    startup = ApplicationStartup(slow_build(0.5))
    startup.start()
    assert await startup.wait_for_ready(0.05) == STARTING
    assert startup.container is None

    assert await startup.wait_for_ready(10) == READY
    assert startup.status()["state"] == READY


@pytest.mark.asyncio
async def test_initialization_failure_is_surfaced_not_swallowed():
    startup = ApplicationStartup(failing_build)
    startup.start()
    assert await startup.wait_for_ready(10) == FAILED
    assert startup.container is None
    assert "artifact checksum mismatch" in (startup.error or "")
    assert "artifact checksum mismatch" in startup.status()["error"]


@pytest.mark.asyncio
async def test_only_one_initialization_runs_under_concurrent_waiters():
    calls: list[int] = []
    startup = ApplicationStartup(slow_build(0.3, calls))
    tasks = [startup.start() for _ in range(8)]
    assert len({id(task) for task in tasks}) == 1

    await asyncio.gather(*(startup.wait_for_ready(10) for _ in range(8)))
    assert calls == [1], f"initialization ran {len(calls)} times"


@pytest.mark.asyncio
async def test_a_waiter_giving_up_does_not_cancel_the_shared_job():
    """wait_for_ready shields the task; an impatient caller must not kill it."""
    startup = ApplicationStartup(slow_build(0.5))
    task = startup.start()
    assert await startup.wait_for_ready(0.05) == STARTING
    assert not task.cancelled()
    assert await startup.wait_for_ready(10) == READY


@pytest.mark.asyncio
async def test_initialization_does_not_block_the_event_loop():
    """Synchronous work belongs in a thread, not on the loop.

    If the builder ran as a coroutine the loop would stall for its whole
    duration and the server would be just as unreachable as before.
    """
    startup = ApplicationStartup(slow_build(0.5))
    startup.start()

    ticks = 0
    began = time.perf_counter()
    while startup.state == STARTING and (time.perf_counter() - began) < 5:
        await asyncio.sleep(0.01)
        ticks += 1

    assert startup.state == READY
    assert ticks > 5, f"event loop only got {ticks} turns while initializing"


@pytest.mark.asyncio
async def test_phase_timings_are_recorded_for_each_step():
    startup = ApplicationStartup(slow_build(0.2))
    startup.start()
    await startup.wait_for_ready(10)
    phases = startup.status()["phases"]
    assert [entry["phase"] for entry in phases] == ["pretend_model_load"]
    assert phases[0]["duration_ms"] >= 150
    assert startup.status()["elapsed_ms"] >= 150


@pytest.mark.asyncio
async def test_the_builder_runs_off_the_main_thread():
    seen: list[int] = []

    def build(recorder: PhaseRecorder) -> Sentinel:
        seen.append(threading.get_ident())
        return Sentinel()

    startup = ApplicationStartup(build)
    startup.start()
    await startup.wait_for_ready(10)
    assert seen and seen[0] != threading.get_ident()


# ------------------------------------------------- the dependency contract


def app_with(startup: ApplicationStartup) -> FastAPI:
    app = FastAPI()
    app.state.startup = startup
    app.state.container = None
    return app


@pytest.mark.asyncio
async def test_dependency_serves_the_container_once_ready():
    startup = ApplicationStartup(slow_build(0.05))
    startup.start()
    await startup.wait_for_ready(10)

    request = type("R", (), {"app": app_with(startup)})()
    assert isinstance(await container_dependency(request), Sentinel)


@pytest.mark.asyncio
async def test_dependency_refuses_deterministically_while_warming():
    startup = ApplicationStartup(slow_build(5.0), grace_seconds=0.05)
    startup.start()
    request = type("R", (), {"app": app_with(startup)})()

    for _ in range(3):
        with pytest.raises(AppError) as caught:
            await container_dependency(request)
        assert caught.value.status_code == 503
        assert caught.value.code == "service_warming"


@pytest.mark.asyncio
async def test_dependency_reports_a_failed_initialization():
    startup = ApplicationStartup(failing_build, grace_seconds=0.05)
    startup.start()
    await startup.wait_for_ready(10)
    request = type("R", (), {"app": app_with(startup)})()

    with pytest.raises(AppError) as caught:
        await container_dependency(request)
    assert caught.value.status_code == 503
    assert caught.value.code == "initialization_failed"


@pytest.mark.asyncio
async def test_concurrent_requests_while_warming_do_not_rebuild():
    calls: list[int] = []
    startup = ApplicationStartup(slow_build(0.4, calls), grace_seconds=0.02)
    startup.start()
    request = type("R", (), {"app": app_with(startup)})()

    async def attempt() -> str:
        try:
            await container_dependency(request)
            return "served"
        except AppError as exc:
            return exc.code

    outcomes = await asyncio.gather(*(attempt() for _ in range(12)))
    assert set(outcomes) <= {"service_warming", "served"}
    await startup.wait_for_ready(10)
    assert calls == [1]


# --------------------------------------------------- the running application


@pytest.fixture
def warming_app(monkeypatch, tmp_path, catalog: list[dict[str, Any]]) -> Any:
    """A real application whose initialization takes longer than the platform waits."""
    import app.main as main_module

    original = main_module.load_or_create_catalog

    def slow_catalog(*args: Any, **kwargs: Any) -> Any:
        time.sleep(16)
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "load_or_create_catalog", slow_catalog)
    return main_module.create_app(
        settings=main_module.get_settings().model_copy(
            update={
                "database_url": f"sqlite:///{tmp_path / 'warm.db'}",
                "hf_dataset_repo": "",
                "embedding_provider": "hashing",
                "startup_warm_grace_seconds": 0.1,
            }
        ),
    )


def test_http_answers_immediately_while_models_are_still_loading(warming_app):
    """The acceptance case: >15s of initialization, a server that still binds."""
    began = time.perf_counter()
    with TestClient(warming_app) as client:
        bound = time.perf_counter() - began
        assert bound < 5.0, f"startup blocked for {bound:.1f}s"

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["components"]["api"]["status"] == "healthy"
        assert health.json()["status"] == "degraded"

        ready = client.get("/api/ready")
        assert ready.status_code == 503
        assert ready.json()["error"]["code"] == "service_warming"

        warming = client.post("/api/recommend", json={"include_genres": ["Action"], "top_k": 5})
        assert warming.status_code == 503
        assert warming.json()["error"]["code"] == "service_warming"
