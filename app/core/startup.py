"""Application initialization, moved off the ASGI startup path.

Vercel gives a container roughly fifteen seconds to accept a TCP connection.
Loading the catalog, the embedding matrices, the CountSketch index, the ALS
factors and the LambdaMART booster takes about half a minute on one vCPU, and
all of it used to run inside the ASGI lifespan -- that is, before uvicorn began
listening. The platform gave up before the socket existed:

    Application initialization timed out. Error: could not connect to $PORT=80.
    INFO: Started server process
    INFO: Waiting for application startup.

Uvicorn had started and was waiting for the lifespan to return. Nothing about
that work is wrong or avoidable; it was simply in the way of the listen call.

So the lifespan now starts exactly one background job and returns immediately.
The job runs in a worker thread, because the work is synchronous and CPU-bound
and scheduling it as a coroutine would block the event loop just as effectively
as blocking the lifespan did. Readiness is reported honestly while it runs
rather than by making the server unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from backend.anime_agent.process_memory import current_process_rss_bytes

logger = logging.getLogger("anime_compass.startup")

STARTING = "starting"
READY = "ready"
FAILED = "failed"


class PhaseRecorder:
    """Wall time and resident memory for each step of initialization.

    Cold start is the constraint being managed, so it has to be measurable from
    a deployment's own logs rather than only from a profiling run on someone's
    laptop. Each phase logs as it completes, so a container killed midway still
    says how far it got and what that cost.
    """

    def __init__(self) -> None:
        self.phases: list[dict[str, Any]] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        before = current_process_rss_bytes()
        try:
            yield
        finally:
            after = current_process_rss_bytes()
            entry: dict[str, Any] = {
                "phase": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "rss_bytes": after,
                "rss_delta_bytes": (after - before) if (after is not None and before is not None) else None,
            }
            self.phases.append(entry)
            logger.info("initialization_phase", extra={"context": entry})

    def total_ms(self) -> float:
        return round(sum(entry["duration_ms"] for entry in self.phases), 1)


class ApplicationStartup:
    """The single initialization job, and the state it leaves behind.

    One job, created once. Every waiter awaits that same task through a shield,
    so a caller giving up on its own timeout cannot cancel the initialization
    everyone else is waiting for, and a burst of concurrent requests during a
    cold start cannot start a second one.
    """

    def __init__(
        self,
        build: Callable[[PhaseRecorder], Any],
        *,
        grace_seconds: float = 2.0,
        on_ready: Callable[[Any], None] | None = None,
    ) -> None:
        self._build = build
        self._on_ready = on_ready
        self.grace_seconds = grace_seconds
        self.state: str = STARTING
        self.error: str | None = None
        self.container: Any = None
        self.recorder = PhaseRecorder()
        self._task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._elapsed_ms: float | None = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> asyncio.Task[None]:
        """Create the initialization task, or return the one already running."""
        if self._task is None:
            self._started_at = time.perf_counter()
            self._task = asyncio.create_task(self._run(), name="anime-compass-initialization")
        return self._task

    async def _run(self) -> None:
        # to_thread, not a bare coroutine: the builder is synchronous and holds
        # the GIL for tens of seconds. Awaiting it on the event loop would keep
        # the server from answering anything, which is the problem being fixed.
        try:
            container = await asyncio.to_thread(self._build, self.recorder)
        except Exception as exc:
            self._elapsed_ms = self._since_start()
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = FAILED
            logger.exception(
                "initialization_failed",
                extra={
                    "context": {
                        "error": self.error,
                        "elapsed_ms": self._elapsed_ms,
                        "phases": self.recorder.phases,
                    }
                },
            )
            return

        self._elapsed_ms = self._since_start()
        self.container = container
        self.state = READY
        if self._on_ready is not None:
            self._on_ready(container)
        logger.info(
            "initialization_complete",
            extra={"context": {"elapsed_ms": self._elapsed_ms, "phases": self.recorder.phases}},
        )

    def _since_start(self) -> float | None:
        if self._started_at is None:
            return None
        return round((time.perf_counter() - self._started_at) * 1000, 1)

    async def wait_for_ready(self, timeout: float | None = None) -> str:
        """Wait up to `timeout` for initialization, returning the resulting state.

        Shielded, so a caller that times out abandons its own wait and not the
        shared task.
        """
        task = self._task
        if task is None or task.done():
            return self.state
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            # Both, deliberately. The project supports Python 3.10, where
            # asyncio.TimeoutError is not a subclass of the builtin -- catching
            # only the builtin would let a timeout escape there while passing
            # on 3.12, which is the worst of both.
            pass
        return self.state

    async def close(self, timeout: float = 30.0) -> None:
        """Let initialization finish before shutdown, within reason.

        The work runs in a thread that cancellation cannot interrupt, so waiting
        is the only way to avoid tearing down around a half-built container.
        """
        task = self._task
        if task is None or task.done():
            return
        if await self.wait_for_ready(timeout) is STARTING:
            logger.warning(
                "initialization_still_running_at_shutdown",
                extra={"context": {"waited_seconds": timeout}},
            )

    # --------------------------------------------------------------- reporting

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state,
            "phases": list(self.recorder.phases),
        }
        if self._elapsed_ms is not None:
            payload["elapsed_ms"] = self._elapsed_ms
        if self.error is not None:
            payload["error"] = self.error
        return payload
