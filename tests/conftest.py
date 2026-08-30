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
