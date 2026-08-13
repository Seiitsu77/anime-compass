from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.archive_pipeline import build_archive_catalog
from backend.anime_agent.collaborative import CollaborativeIndex
from scripts.build_collaborative_model import build_collaborative_artifact
from scripts.profile_archive import build_report

ANIME_FIELDS = [
    "MAL_ID",
    "Name",
    "Score",
    "Genres",
    "English name",
    "Japanese name",
    "Type",
    "Episodes",
    "Aired",
    "Premiered",
    "Producers",
    "Licensors",
    "Studios",
    "Source",
    "Duration",
    "Rating",
    "Ranked",
    "Popularity",
    "Members",
    "Favorites",
    "Watching",
    "Completed",
    "On-Hold",
    "Dropped",
    "Plan to Watch",
    *[f"Score-{score}" for score in range(1, 11)],
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _anime_row(anime_id: int, name: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "MAL_ID": anime_id,
        "Name": name,
        "Score": "8.0",
        "Genres": "Drama, Mystery",
        "English name": "Unknown",
        "Japanese name": "Unknown",
        "Type": "TV",
        "Episodes": "12",
        "Aired": "Jan 1, 2020 to Mar 1, 2020",
        "Premiered": "Winter 2020",
        "Producers": "Test Producer",
        "Licensors": "Unknown",
        "Studios": "Madhouse",
        "Source": "Manga",
        "Duration": "24 min. per ep.",
        "Rating": "PG-13",
        "Ranked": "100",
        "Popularity": "200",
        "Members": "10000",
        "Favorites": "100",
        "Watching": "10",
        "Completed": "900",
        "On-Hold": "5",
        "Dropped": "2",
        "Plan to Watch": "50",
    }
    row.update({f"Score-{score}": "0" for score in range(1, 11)})
    row["Score-8"] = "100"
    row.update(overrides)
    return row


def _write_archive(
    root: Path,
    anime_rows: list[dict[str, Any]],
    rating_rows: list[tuple[int, int, int]],
) -> None:
    _write_csv(root / "anime.csv", ANIME_FIELDS, anime_rows)
    _write_csv(
        root / "anime_with_synopsis.csv",
        ["MAL_ID", "Name", "Score", "Genres", "sypnopsis"],
        [
            {
                "MAL_ID": row["MAL_ID"],
                "Name": row["Name"],
                "Score": row["Score"],
                "Genres": row["Genres"],
                "sypnopsis": f"Synopsis for {row['Name']}",
            }
            for row in anime_rows
        ],
    )
    _write_csv(
        root / "rating_complete.csv",
        ["user_id", "anime_id", "rating"],
        [{"user_id": user_id, "anime_id": anime_id, "rating": rating} for user_id, anime_id, rating in rating_rows],
    )


def test_archive_catalog_repairs_text_excludes_adult_and_preserves_enrichment(
    tmp_path: Path,
) -> None:
    _write_archive(
        tmp_path,
        [
            _anime_row(1, "PokÃ©mon"),
            _anime_row(2, "Adult title", Genres="Hentai", Rating="Rx - Hentai"),
        ],
        [(1, 1, 9), (1, 2, 8)],
    )
    legacy = [
        {
            "id": 1,
            "title": "Pokemon legacy",
            "score": 8.5,
            "image_url": "https://example.invalid/poster.jpg",
            "staff": [{"id": 7, "name": "A Director", "role": "Director"}],
            "staff_relationships": [{"id": 7, "name": "A Director", "role": "Director"}],
            "creators": [{"id": 7, "name": "A Director", "role": "Director"}],
        },
        {
            "id": 3,
            "title": "Recent extension",
            "score": 8.9,
            "start_year": 2024,
            "type": "TV",
            "genres": ["Drama"],
            "studios": ["New Studio"],
        },
    ]

    catalog = build_archive_catalog(tmp_path, enrichment_catalog=legacy)
    by_id = {item["id"]: item for item in catalog}

    assert set(by_id) == {1, 3}
    assert by_id[1]["title"] == "Pokémon"
    assert by_id[1]["image_url"].endswith("poster.jpg")
    assert by_id[1]["relationship_enriched"] is True
    assert by_id[1]["data_source"] == "archive_2020+legacy_relationships"
    assert by_id[3]["data_source"] == "legacy_recent_extension"
    assert all(not item.get("adult_content") for item in catalog)


def test_archive_catalog_can_include_adult_and_drop_legacy_only(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        [_anime_row(2, "Adult title", Genres="Hentai", Rating="Rx - Hentai")],
        [(1, 2, 8)],
    )

    catalog = build_archive_catalog(
        tmp_path,
        enrichment_catalog=[{"id": 99, "title": "Legacy only"}],
        include_adult=True,
        include_legacy_only=False,
    )

    assert [item["id"] for item in catalog] == [2]
    assert catalog[0]["adult_content"] is True


def test_archive_quality_report_rejects_duplicate_pairs_and_unsorted_users(
    tmp_path: Path,
) -> None:
    _write_archive(
        tmp_path,
        [_anime_row(1, "One"), _anime_row(2, "Two")],
        [(2, 1, 8), (2, 1, 9), (1, 2, 7)],
    )

    report = build_report(tmp_path)

    assert report["ratings"]["duplicate_user_anime_pairs"] == 1
    assert report["ratings"]["sorted_by_user"] is False
    assert report["quality_gates"]["ratings_unique_user_anime_pairs"] is False
    assert report["passed"] is False


def test_collaborative_training_learns_positive_and_negative_item_relationships(
    tmp_path: Path,
) -> None:
    ratings_path = tmp_path / "rating_complete.csv"
    rows: list[dict[str, int]] = []
    for user_id in range(1, 121):
        if user_id % 2:
            ratings = {1: 10, 2: 9, 3: 1, 4: 5}
        else:
            ratings = {1: 2, 2: 1, 3: 10, 4: 5}
        rows.extend(
            {"user_id": user_id, "anime_id": anime_id, "rating": rating} for anime_id, rating in ratings.items()
        )
    _write_csv(ratings_path, ["user_id", "anime_id", "rating"], rows)
    catalog = [{"id": anime_id, "title": f"Item {anime_id}"} for anime_id in range(1, 5)]
    output = tmp_path / "collaborative.npz"

    metadata = build_collaborative_artifact(
        ratings_path,
        catalog,
        output,
        projections=4,
        width=256,
        progress_every=0,
    )
    index = CollaborativeIndex.load(output, catalog)
    scores = index.profile_scores(positive_ids=[1])

    assert metadata["ratings_used"] == 480
    assert metadata["users_seen"] == 120
    assert scores[2] > 0.75
    assert scores.get(3, 0.0) < 0.05
    assert scores.get(4, 0.0) < scores[2]
    assert index.profile_scores(positive_ids=[999]) == {}
    assert 0.0 <= (index.quality_score(1) or -1) <= 1.0


def test_collaborative_artifact_validation_rejects_non_finite_vectors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        anime_ids=np.asarray([1], dtype=np.int64),
        vectors=np.asarray([[np.nan] * 8], dtype=np.float32),
        rating_count=np.asarray([1], dtype=np.int64),
        rating_mean=np.asarray([8.0], dtype=np.float32),
        bayesian_score=np.asarray([0.8], dtype=np.float32),
        metadata_json=np.asarray('{"artifact_version":1}'),
    )

    with pytest.raises(ValueError, match="non-finite"):
        CollaborativeIndex.load(path, [{"id": 1, "title": "One"}])
