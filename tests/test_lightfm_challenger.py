from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.anime_agent.evaluation.lightfm_training import (
    FORBIDDEN_METADATA_FIELDS,
    LightFMCandidateConfig,
    LightFMSearchConfig,
    _export_lightfm_artifact,
    _top_k_ids,
    build_lightfm_item_features,
    build_lightfm_training_data,
    default_search_candidates,
)
from backend.anime_agent.evaluation.split import (
    SplitConfig,
    SplitStore,
    build_split_store,
    catalog_ids_sha256,
    select_evaluation_sample,
)
from backend.anime_agent.lightfm_serving import LIGHTFM_ARTIFACT_VERSION, LightFMServingIndex


def _write_ratings(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["user_id", "anime_id", "rating"])
        writer.writerows(sorted(rows))


def _build_store(
    tmp_path: Path,
    rows: list[tuple[int, int, int]],
    *,
    catalog_ids: set[int],
    name: str = "split.sqlite",
) -> SplitStore:
    ratings_path = tmp_path / f"{name}.csv"
    split_path = tmp_path / name
    _write_ratings(ratings_path, rows)
    build_split_store(
        ratings_path,
        split_path,
        catalog_ids=catalog_ids,
        config=SplitConfig(seed=42),
    )
    return SplitStore(split_path)


def _catalog_item(anime_id: int, *, genre: str, studio: str) -> dict[str, object]:
    return {
        "id": anime_id,
        "title": f"Anime {anime_id}",
        "genres": [genre, "Drama"],
        "type": "TV",
        "source": "Manga",
        "studios": [studio],
        "start_year": 2011 + anime_id,
        "content_rating": "PG-13",
        "score": 9.99,
        "rank": 1,
        "popularity": 1,
        "members": 9_999_999,
        "rating_count": 9_999_999,
    }


def test_lightfm_candidate_configuration_is_validated_and_deterministic() -> None:
    first = LightFMCandidateConfig(loss="warp", no_components=16, epochs=3)
    second = LightFMCandidateConfig(loss="warp", no_components=16, epochs=3)
    changed = LightFMCandidateConfig(loss="bpr", no_components=16, epochs=3)
    assert first.key == second.key
    assert first.key != changed.key
    assert {candidate.loss for candidate in default_search_candidates("smoke")} == {"warp", "bpr"}
    assert {candidate.loss for candidate in default_search_candidates("standard")} == {"warp", "bpr"}
    assert LightFMSearchConfig(candidates=(first, changed)).candidates == (first, changed)
    with pytest.raises(ValueError, match="warp.*bpr"):
        LightFMCandidateConfig(loss="logistic")
    with pytest.raises(ValueError, match="both WARP and BPR"):
        LightFMSearchConfig(candidates=(first,))
    fixed = LightFMSearchConfig(candidates=(first,), require_both_losses=False)
    assert fixed.candidates == (first,)


def test_lightfm_features_are_static_sparse_normalized_and_order_deterministic() -> None:
    catalog = [
        _catalog_item(3, genre="Sci-Fi", studio="Shared Studio"),
        _catalog_item(1, genre="Action", studio="Shared Studio"),
        _catalog_item(2, genre="Romance", studio="Rare Studio"),
    ]
    first = build_lightfm_item_features(catalog, studio_min_frequency=2)

    changed_outcomes = [dict(item) for item in reversed(catalog)]
    for item in changed_outcomes:
        for field in FORBIDDEN_METADATA_FIELDS:
            item[field] = f"deliberately-changed-{field}"
    second = build_lightfm_item_features(changed_outcomes, studio_min_frequency=2)

    assert first.feature_names == second.feature_names
    np.testing.assert_array_equal(first.matrix.toarray(), second.matrix.toarray())
    np.testing.assert_allclose(np.asarray(first.matrix.sum(axis=1)).ravel(), np.ones(3))
    assert first.matrix.format == "csr"
    assert first.feature_names[:3] == ("identity:1", "identity:2", "identity:3")
    assert "studio:shared-studio" in first.feature_names
    assert "studio:rare-studio" not in first.feature_names
    assert "studio:unknown-or-rare" in first.feature_names
    assert set(first.summary["excluded_outcome_fields"]) == FORBIDDEN_METADATA_FIELDS
    assert all(not name.startswith(("score:", "rank:", "members:")) for name in first.feature_names)


def test_lightfm_training_matrix_has_deterministic_mappings_and_no_holdout_leakage(tmp_path: Path) -> None:
    rows = [
        *((1, anime_id, 9) for anime_id in range(1, 8)),
        (1, 8, 3),
        (1, 9, 7),
        *((2, anime_id, 8) for anime_id in range(10, 16)),
        (2, 16, 5),
    ]
    store = _build_store(tmp_path, rows, catalog_ids=set(range(1, 18)))
    metadata = store.metadata()
    assert metadata["train_positive_density"] == metadata["train_positive_sparsity"]
    assert metadata["train_positive_density"] + metadata["train_positive_matrix_sparsity"] == pytest.approx(1.0)
    first = build_lightfm_training_data(store, list(reversed(range(1, 18))))
    second = build_lightfm_training_data(store, list(range(1, 18)))

    np.testing.assert_array_equal(first.user_ids, np.asarray([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(first.anime_ids, np.arange(1, 18, dtype=np.int64))
    np.testing.assert_array_equal(first.user_ids, second.user_ids)
    np.testing.assert_array_equal(first.anime_ids, second.anime_ids)
    np.testing.assert_array_equal(first.interactions.toarray(), second.interactions.toarray())

    matrix = first.interactions.tocsr()
    for row_index, user_id in enumerate(first.user_ids.tolist()):
        user = store.get_user(user_id)
        assert user is not None
        encoded = {int(first.anime_ids[column]) for column in matrix[row_index].indices.tolist()}
        assert encoded == set(user.train_positive_ids)
        assert encoded.isdisjoint(user.validation_positive_ids)
        assert encoded.isdisjoint(user.test_positive_ids)
        assert encoded.isdisjoint(anime_id for anime_id, _rating in user.explicit_negative)
        assert encoded.isdisjoint(anime_id for anime_id, _rating in user.neutral)


def test_numpy_serving_artifact_round_trip_and_known_item_exclusion(tmp_path: Path) -> None:
    anime_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    user_ids = np.asarray([7, 11], dtype=np.int64)
    item_embeddings = np.asarray([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    user_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    item_biases = np.zeros(4, dtype=np.float32)
    user_biases = np.zeros(2, dtype=np.float32)
    metadata = {
        "artifact_version": LIGHTFM_ARTIFACT_VERSION,
        "trainer": "lightfm",
        "variant": "lightfm_id",
        "catalog_ids_sha256": catalog_ids_sha256(anime_ids.tolist()),
        "selected_config": {"loss": "warp"},
    }
    artifact_path = tmp_path / "lightfm_id.npz"
    np.savez_compressed(
        artifact_path,
        anime_ids=anime_ids,
        user_ids=user_ids,
        item_embeddings=item_embeddings,
        item_biases=item_biases,
        user_embeddings=user_embeddings,
        user_biases=user_biases,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    index = LightFMServingIndex.load(artifact_path, [{"id": value} for value in anime_ids.tolist()])
    np.testing.assert_allclose(index.scores_for_user(7), np.asarray([1.0, 0.5, 0.0, 0.0]))
    np.testing.assert_allclose(index.score_pairs(11, [3, 4]), np.asarray([1.0, 0.5]))
    assert index.recommend(7, known_ids=[1], k=3) == [2, 3, 4]
    assert index.recommend(11, known_ids=[3], k=3) == [4, 1, 2]
    assert index.recommend(7, known_ids=[1, 2, 3], k=20) == [4]
    assert _top_k_ids(
        index.scores_for_user(7),
        anime_ids,
        known_ids=[1, 2, 3],
        k=20,
    ) == [4]
    assert index.model_info()["loss"] == "warp"


def test_activity_balanced_sampling_honors_configurable_per_stratum_quota(tmp_path: Path) -> None:
    total_positives = {
        1: 5,
        2: 6,
        3: 6,
        4: 8,
        5: 10,
        6: 12,
        7: 15,
        8: 19,
        9: 30,
        10: 32,
        11: 35,
        12: 40,
    }
    rows = [
        (user_id, user_id * 100 + offset, 9)
        for user_id, count in total_positives.items()
        for offset in range(1, count + 1)
    ]
    catalog_ids = {anime_id for _user_id, anime_id, _rating in rows}
    store = _build_store(tmp_path, rows, catalog_ids=catalog_ids, name="activity.sqlite")

    first = select_evaluation_sample(
        store,
        limit=None,
        users_per_stratum=3,
        seed=17,
        strategy="activity_stratified",
    )
    second = select_evaluation_sample(
        store,
        limit=None,
        users_per_stratum=3,
        seed=17,
        strategy="activity_stratified",
    )
    assert first == second
    assert first.diagnostic is True
    assert first.stratum_selected_counts == {"sparse": 3, "medium": 3, "heavy": 3}
    assert len(first.user_ids) == 9


def test_popularity_stratified_sampling_is_deterministic_and_balanced(tmp_path: Path) -> None:
    rows = [(user_id, user_id * 10 + offset, 9) for user_id in range(1, 10) for offset in range(1, 6)]
    catalog_ids = {anime_id for _user_id, anime_id, _rating in rows}
    store = _build_store(tmp_path, rows, catalog_ids=catalog_ids, name="popularity.sqlite")
    expected_bucket = {
        user_id: ("head" if user_id <= 3 else "mid_tail" if user_id <= 6 else "long_tail") for user_id in range(1, 10)
    }
    bucket_by_id = {anime_id: "head" for anime_id in catalog_ids}
    for user in store.iter_users(eligible_only=True):
        assert len(user.test_positive_ids) == 1
        bucket_by_id[user.test_positive_ids[0]] = expected_bucket[user.user_id]

    first = select_evaluation_sample(
        store,
        limit=None,
        users_per_stratum=2,
        seed=29,
        strategy="popularity_stratified",
        bucket_by_id=bucket_by_id,
    )
    second = select_evaluation_sample(
        store,
        limit=None,
        users_per_stratum=2,
        seed=29,
        strategy="popularity_stratified",
        bucket_by_id=bucket_by_id,
    )
    assert first == second
    assert first.diagnostic is True
    assert first.stratum_population_counts == {"head": 3, "mid_tail": 3, "long_tail": 3}
    assert first.stratum_selected_counts == {"head": 2, "mid_tail": 2, "long_tail": 2}
    assert len(first.user_ids) == 6
    assert "test-positive" in first.selection_target


def test_native_lightfm_scores_match_exported_numpy_scores(tmp_path: Path) -> None:
    lightfm = pytest.importorskip("lightfm")
    from lightfm import LightFM
    from scipy import sparse

    from backend.anime_agent.evaluation.lightfm_training import LightFMTrainingData

    interactions = sparse.coo_matrix(
        np.asarray(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    training_data = LightFMTrainingData(
        user_ids=np.asarray([10, 20, 30], dtype=np.int64),
        anime_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        interactions=interactions,
    )
    model = LightFM(no_components=4, loss="warp", random_state=42)
    model.fit(interactions, epochs=2, num_threads=1)
    artifact_path = tmp_path / "trained.npz"
    exported = _export_lightfm_artifact(
        model,
        variant="lightfm_id",
        item_features=None,
        training_data=training_data,
        output_path=artifact_path,
        metadata={"selected_config": {"loss": "warp"}},
    )
    index = LightFMServingIndex.load(artifact_path, [{"id": value} for value in range(1, 5)])
    native = model.predict(0, np.arange(4, dtype=np.int32), num_threads=1)
    np.testing.assert_allclose(index.scores_for_user(10), native, rtol=0.0, atol=1e-5)
    assert exported["numpy_score_roundtrip_max_abs_error"] <= 1e-5
    assert lightfm.__version__
