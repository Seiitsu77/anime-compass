from __future__ import annotations

from typing import Any

import pytest


def anime(
    anime_id: int,
    title: str,
    *,
    genres: list[str],
    synopsis: str,
    score: float = 8.0,
    episodes: int = 12,
    year: int = 2020,
    studios: list[str] | None = None,
    voice_actor: str | None = None,
) -> dict[str, Any]:
    roles = []
    actors = []
    if voice_actor:
        roles = [
            {
                "voice_actor_id": 42,
                "voice_actor": voice_actor,
                "character_id": anime_id + 1000,
                "character": f"Character {anime_id}",
                "language": "Japanese",
            }
        ]
        actors = [{"id": 42, "name": voice_actor, "language": "Japanese"}]
    return {
        "id": anime_id,
        "title": title,
        "score": score,
        "rank": anime_id,
        "popularity": anime_id * 10,
        "members": 10000 - anime_id,
        "synopsis": synopsis,
        "start_year": year,
        "type": "TV",
        "episodes": episodes,
        "image_url": "",
        "genres": genres,
        "genre_groups": {},
        "metadata_tokens": [f"genre_{genre}" for genre in genres],
        "studios": studios or ["Test Studio"],
        "producers": [],
        "characters": [{"id": anime_id + 1000, "name": f"Character {anime_id}", "role": "Main"}],
        "character_names": [f"Character {anime_id}"],
        "character_relationships": [{"id": anime_id + 1000, "name": f"Character {anime_id}", "role": "Main"}],
        "staff": [{"id": 700, "name": "Test Director", "role": "Director"}],
        "staff_relationships": [{"id": 700, "name": "Test Director", "role": "Director"}],
        "creators": [{"id": 700, "name": "Test Director", "role": "Director"}],
        "voice_actors": actors,
        "voice_actor_roles": roles,
    }


def wait_until_ready(client, timeout: float = 60.0) -> None:
    """Block until the application has finished initializing.

    /api/health is a liveness probe: it answers as soon as the server is up and
    deliberately does not wait for models, because an orchestrator that waits
    on a probe kills the container it is waiting for. A test asserting on the
    fully-loaded component report therefore has to establish that state itself
    rather than assume it, which readiness -- not liveness -- is what reports.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if client.get("/api/ready").status_code == 200:
            return
        _time.sleep(0.05)
    raise AssertionError("application did not become ready in time")


@pytest.fixture(autouse=True)
def wait_out_startup(monkeypatch):
    """Let tests see a fully initialized application.

    Initialization moved off the ASGI startup path so the server can bind
    before the models are loaded, which means a request can now arrive while
    the container is still being built and be answered with 503
    service_warming. That is the intended production behaviour, and
    tests/test_startup_readiness.py covers it directly with a deliberately slow
    build.

    Everywhere else the subject is the built application, not the warm-up. A
    generous grace makes each request wait on the one shared initialization
    task instead of racing it, which is what these tests assumed when startup
    was synchronous. It changes no production default: the grace only bounds
    how long a caller waits before being told the service is warming.

    The wait is real here rather than nominal because .env sets
    EMBEDDING_PROVIDER=sentence_transformers, so the first application built in
    a session loads MiniLM from disk and takes seconds.
    """
    monkeypatch.setenv("STARTUP_WARM_GRACE_SECONDS", "60")


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    values = [
        anime(1, "Death Note", genres=["Supernatural", "Mystery"], synopsis="A dark psychological battle."),
        anime(2, "Death Parade", genres=["Supernatural", "Drama"], synopsis="Souls are judged through games."),
        anime(3, "Ghost Hunt", genres=["Supernatural", "Mystery"], synopsis="A team investigates ghosts and spirits."),
        anime(4, "Quiet Romance", genres=["Romance"], synopsis="Two students build a gentle relationship."),
    ]
    values.extend(
        anime(
            anime_id,
            f"Matsuoka Verified {anime_id}",
            genres=["Action", "Fantasy"],
            synopsis=f"Verified fantasy adventure number {anime_id}.",
            voice_actor="Matsuoka, Yoshitsugu",
        )
        for anime_id in range(10, 18)
    )
    return values


@pytest.fixture(autouse=True)
def isolate_production_als_artifact(monkeypatch, tmp_path_factory):
    """Keep tests off the real production ALS artifact.

    Most tests build an app around a small synthetic catalog. The shipped
    production artifact describes the real 18,064-title catalog, and a catalog
    mismatch is deliberately fatal, so tests must not pick it up by default.
    Tests that want an ALS index build their own fixture artifact and pass it
    explicitly.
    """
    absent = tmp_path_factory.mktemp("no-als") / "absent.npz"
    monkeypatch.setenv("ALS_ARTIFACT_PATH", str(absent))
