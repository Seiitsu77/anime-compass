from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.evaluation.metrics import (
    build_item_popularity_buckets,
    hit_rate_at_k,
    ndcg_at_k,
    paired_bootstrap_difference,
    recall_at_k,
    reciprocal_rank,
    user_activity_segment,
)
from backend.anime_agent.evaluation.models import (
    CountSketchModel,
    CurrentHybridModel,
    PopularityModel,
    build_countsketch_artifact_from_split,
    compute_train_statistics,
    sanitize_catalog_with_training_statistics,
)
from backend.anime_agent.evaluation.runner import EvaluationRunConfig, run_personalized_evaluation
from backend.anime_agent.evaluation.split import (
    FeedbackConfig,
    SplitConfig,
    SplitStore,
    UserSplit,
    build_split_store,
    holdout_sizes,
    select_evaluation_user_ids,
    split_user_positives,
)
from backend.anime_agent.recommender import AnimeRecommender


def _write_ratings(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["user_id", "anime_id", "rating"])
        writer.writerows(rows)


def _catalog(size: int) -> list[dict[str, object]]:
    return [
        {
            "id": anime_id,
            "title": f"Anime {anime_id}",
            "title_english": None,
            "title_japanese": None,
            "synonyms": [],
            "type": "TV",
            "episodes": 12,
            "status": "Finished Airing",
            "start_year": 2020 + anime_id % 3,
            "score": 7.0,
            "members": 100,
            "popularity": anime_id,
            "rank": anime_id,
            "genres": [f"Genre {anime_id % 4}", "Drama"],
            "themes": [],
            "demographics": [],
            "studios": [f"Studio {anime_id % 3}"],
            "producers": [],
            "licensors": [],
            "staff": [],
            "characters": [],
            "synopsis": f"A distinct story about theme {anime_id % 5}.",
        }
        for anime_id in range(1, size + 1)
    ]


def _build_store(
    tmp_path: Path,
    rows: list[tuple[int, int, int]],
    *,
    catalog_size: int,
    config: SplitConfig | None = None,
    name: str = "split.sqlite",
) -> SplitStore:
    ratings_path = tmp_path / f"{name}.csv"
    split_path = tmp_path / name
    _write_ratings(ratings_path, rows)
    build_split_store(
        ratings_path,
        split_path,
        catalog_ids=set(range(1, catalog_size + 1)),
        config=config or SplitConfig(),
    )
    return SplitStore(split_path)


def test_feedback_classes_are_configurable_and_keep_gap_ratings() -> None:
    default = FeedbackConfig()
    assert [default.classify(value) for value in (5, 6, 7, 8, 10)] == [
        "explicit_negative",
        "neutral",
        "neutral",
        "positive",
        "positive",
    ]
    stricter = FeedbackConfig(positive_threshold=9)
    assert stricter.classify(8) == "ignored"
    assert stricter.classify(9) == "positive"


@pytest.mark.parametrize(
    ("positives", "expected"),
    [(4, (0, 0)), (5, (1, 1)), (9, (1, 1)), (10, (1, 2)), (19, (1, 2)), (20, (2, 2)), (29, (2, 2))],
)
def test_holdout_rules(positives: int, expected: tuple[int, int]) -> None:
    assert holdout_sizes(positives) == expected


def test_user_split_is_order_independent_reproducible_and_leakage_free() -> None:
    positives = [(anime_id, 8 + anime_id % 3) for anime_id in range(1, 31)]
    config = SplitConfig(seed=42)
    first = split_user_positives(17, positives, config)
    second = split_user_positives(17, list(reversed(positives)), config)
    assert first == second
    train, validation, test, eligible = first
    assert eligible
    assert len(validation) == 3
    assert len(test) == 3
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(positives)
    assert split_user_positives(17, positives, SplitConfig(seed=7)) != first


def test_persistent_split_preserves_classes_and_reproduces_users(tmp_path: Path) -> None:
    rows = [
        *((1, anime_id, 9) for anime_id in range(1, 6)),
        (1, 6, 8),
        (1, 7, 7),
        (1, 8, 3),
        *((2, anime_id, 9) for anime_id in range(9, 14)),
    ]
    config = SplitConfig(seed=42, feedback=FeedbackConfig(positive_threshold=9))
    first = _build_store(tmp_path, rows, catalog_size=20, config=config, name="first.sqlite")
    second = _build_store(tmp_path, rows, catalog_size=20, config=config, name="second.sqlite")
    assert list(first.iter_users()) == list(second.iter_users())
    user = first.get_user(1)
    assert user is not None and user.eligible
    assert user.neutral == ((7, 7),)
    assert user.explicit_negative == ((8, 3),)
    assert user.ignored == ((6, 8),)
    assert {6, 7, 8}.issubset({anime_id for anime_id, _rating in user.all_observed_training_ratings})
    observed_ids = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
    assert set(user.validation_positive_ids).isdisjoint(observed_ids)
    assert set(user.test_positive_ids).isdisjoint(observed_ids)
    metadata = first.metadata()
    assert metadata["dataset_sha256"] == second.metadata()["dataset_sha256"]
    assert metadata["dataset_sha256_scope"] == "full_file"
    assert metadata["ignored_ratings"] == 1
    audit = first.audit_counts()
    assert audit["passed"] is True
    assert audit["stored_users"] == 2


def test_ranking_metrics_match_hand_computation_with_multiple_positives() -> None:
    ranking = [2, 1, 4, 9]
    relevant = {2, 4}
    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert recall_at_k(ranking, relevant, 3) == 1.0
    assert hit_rate_at_k(ranking, relevant, 2) == 1.0
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, relevant, 3) == pytest.approx(expected_dcg / expected_idcg)
    assert reciprocal_rank(ranking, relevant, cutoff=20) == 1.0
    assert recall_at_k([1, 3], relevant, 20) == 0.0


def test_popularity_and_countsketch_use_training_rows_only(tmp_path: Path) -> None:
    rows = [(1, anime_id, 9) for anime_id in range(1, 6)]
    store = _build_store(tmp_path, rows, catalog_size=8)
    user = store.get_user(1)
    assert user is not None
    held_out = {*user.validation_positive_ids, *user.test_positive_ids}
    statistics = compute_train_statistics(store, range(1, 9))
    count_by_id = statistics.positive_counts_by_id()
    assert all(count_by_id[anime_id] == 0 for anime_id in held_out)

    artifact = tmp_path / "countsketch.npz"
    build_countsketch_artifact_from_split(store, _catalog(8), artifact, projections=1, width=8)
    with np.load(artifact, allow_pickle=False) as payload:
        ids = payload["anime_ids"].tolist()
        counts = payload["rating_count"].tolist()
    artifact_counts = dict(zip(ids, counts, strict=True))
    assert all(artifact_counts[anime_id] == 0 for anime_id in held_out)

    index = CollaborativeIndex.load(artifact, _catalog(8))
    source_catalog = _catalog(8)
    for item in source_catalog:
        item.update(
            archive_score=9.9,
            favorites=999,
            rating_count=999,
            score_distribution={"10": 999},
            watching_stats={"completed": 999},
        )
    sanitized = sanitize_catalog_with_training_statistics(source_catalog, statistics, index)
    heldout_item = next(item for item in sanitized if int(item["id"]) in held_out)
    assert heldout_item["score"] is None
    assert heldout_item["archive_score"] is None
    assert heldout_item["members"] == 0
    assert heldout_item["rating_count"] == 0
    assert heldout_item["favorites"] == 0
    assert heldout_item["score_distribution"] == {}
    assert heldout_item["watching_stats"] == {}


def test_every_model_excludes_all_known_training_items(tmp_path: Path) -> None:
    user = UserSplit(
        user_id=7,
        eligible=True,
        train_positive=((1, 10),),
        validation_positive=((5, 9),),
        test_positive=((6, 9),),
        explicit_negative=((2, 3),),
        neutral=((3, 7),),
        ignored=((4, 8),),
    )
    anime_ids = np.arange(1, 9, dtype=np.int64)
    statistics = compute_train_statistics(
        _build_store(
            tmp_path,
            [(1, anime_id, 9) for anime_id in range(1, 6)],
            catalog_size=8,
            name="known.sqlite",
        ),
        anime_ids,
    )
    popularity = PopularityModel(statistics, build_duration_seconds=0.0, artifact_path=None)

    vectors = np.eye(8, dtype=np.float32)
    index = CollaborativeIndex(
        anime_ids,
        vectors,
        np.ones(8, dtype=np.int64),
        np.full(8, 8.0, dtype=np.float32),
        np.full(8, 0.8, dtype=np.float32),
        {"method": "test"},
    )
    collaborative = CountSketchModel(
        index,
        anime_ids,
        build_duration_seconds=0.0,
        artifact_path=tmp_path / "cf.npz",
    )
    hybrid = CurrentHybridModel(
        AnimeRecommender(_catalog(8), collaborative_index=index),
        build_duration_seconds=0.0,
        artifact_path=tmp_path / "cf.npz",
    )
    known = {1, 2, 3, 4}
    for model in (popularity, collaborative, hybrid):
        assert known.isdisjoint(model.recommend(user, 8).anime_ids)


@pytest.mark.parametrize(
    ("count", "segment"),
    [(0, "none"), (1, "sparse"), (4, "sparse"), (5, "medium"), (19, "medium"), (20, "heavy")],
)
def test_user_activity_segments_use_training_positives(count: int, segment: str) -> None:
    assert user_activity_segment(count) == segment


def test_item_popularity_buckets_are_train_only_and_deterministic() -> None:
    counts = {anime_id: 11 - anime_id for anime_id in range(1, 6)}
    buckets = build_item_popularity_buckets(range(1, 11), counts)
    assert {anime_id for anime_id, bucket in buckets.items() if bucket == "head"} == {1, 2}
    assert {anime_id for anime_id, bucket in buckets.items() if bucket == "mid_tail"} == {3, 4, 5}
    assert {anime_id for anime_id, bucket in buckets.items() if bucket == "long_tail"} == set(range(6, 11))


def test_paired_bootstrap_is_paired_and_reproducible() -> None:
    left = {1: 1.0, 2: 0.5, 3: 0.0, 99: 0.9}
    right = {1: 0.0, 2: 0.5, 3: 1.0, 100: 0.1}
    first = paired_bootstrap_difference(left, right, iterations=500, seed=42)
    second = paired_bootstrap_difference(left, right, iterations=500, seed=42)
    assert first == second
    assert first["users"] == 3
    assert first["difference"] == pytest.approx(0.0)


def test_selected_user_ids_are_stable_and_sorted(tmp_path: Path) -> None:
    rows = [(user_id, anime_id, 9) for user_id in range(1, 11) for anime_id in range(1, 6)]
    store = _build_store(tmp_path, rows, catalog_size=10)
    first = select_evaluation_user_ids(store, limit=4, seed=42, strategy="uniform")
    second = select_evaluation_user_ids(store, limit=4, seed=42, strategy="uniform")
    assert first == second == sorted(first)
    assert len(first) == 4
    assert [user.user_id for user in store.iter_users_by_ids(first)] == first


def test_stratified_user_sample_represents_all_activity_segments(tmp_path: Path) -> None:
    positive_counts = {1: 5, 2: 6, 3: 10, 4: 12, 5: 25, 6: 30}
    rows = [(user_id, anime_id, 9) for user_id, count in positive_counts.items() for anime_id in range(1, count + 1)]
    store = _build_store(tmp_path, rows, catalog_size=35, name="stratified.sqlite")
    selected = select_evaluation_user_ids(store, limit=3, seed=42, strategy="stratified")
    segments = {user_activity_segment(len(user.train_positive)) for user in store.iter_users_by_ids(selected)}
    assert segments == {"sparse", "medium", "heavy"}


def test_end_to_end_runner_uses_same_users_and_writes_all_artifacts(tmp_path: Path) -> None:
    catalog = _catalog(30)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    rows = [
        (user_id, ((user_id * 3 + offset) % 30) + 1, 8 + offset % 3) for user_id in range(1, 7) for offset in range(8)
    ]
    rows.sort()
    store = _build_store(tmp_path, rows, catalog_size=30, name="integration.sqlite")
    output_dir = tmp_path / "results"
    result = run_personalized_evaluation(
        store,
        catalog,
        catalog_path=catalog_path,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=output_dir,
        config=EvaluationRunConfig(
            max_evaluation_users=4,
            bootstrap_iterations=50,
            countsketch_projections=1,
            countsketch_width=8,
            progress_every=10,
        ),
    )
    assert [model["model"] for model in result["models"]] == [
        "popularity",
        "countsketch_cf",
        "current_hybrid",
    ]
    assert all(model["evaluated_users"] == 4 for model in result["models"])
    assert len(result["paired_bootstrap"]) == 6
    for name in (
        "results.json",
        "summary.csv",
        "user_segments.csv",
        "item_popularity.csv",
        "engineering.csv",
        "paired_bootstrap.csv",
        "per_user_metrics.csv.gz",
        "report.md",
        "manifest.json",
    ):
        assert (output_dir / name).is_file()
    with gzip.open(output_dir / "per_user_metrics.csv.gz", "rt", encoding="utf-8") as file:
        per_model_users: dict[str, set[int]] = {}
        for row in csv.DictReader(file):
            per_model_users.setdefault(row["model"], set()).add(int(row["user_id"]))
    assert len(per_model_users) == 3
    assert len({tuple(sorted(user_ids)) for user_ids in per_model_users.values()}) == 1
