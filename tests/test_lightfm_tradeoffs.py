from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from backend.anime_agent.evaluation.lightfm_training import (
    DEFAULT_INCLUDED_METADATA_FIELDS,
    LightFMTrainingData,
    LightFMVariantConfig,
    _export_lightfm_artifact,
    build_lightfm_item_features,
    build_lightfm_user_features,
    item_metadata_ablation_configs,
)
from backend.anime_agent.evaluation.metrics import (
    gini_coefficient,
    recommendation_popularity_concentration,
)
from backend.anime_agent.evaluation.models import LightFMModel
from backend.anime_agent.evaluation.split import UserSplit
from backend.anime_agent.lightfm_serving import LightFMServingIndex


def _catalog_item(anime_id: int, genre: str) -> dict[str, object]:
    return {
        "id": anime_id,
        "title": f"Anime {anime_id}",
        "genres": [genre, "Action"] if genre != "Action" else ["Action"],
        "type": "TV",
        "source": "Manga",
        "start_year": 2010 + anime_id,
        "content_rating": "PG-13",
        "studios": ["Studio A"],
    }


class _UserStore:
    def __init__(self, users: list[UserSplit]):
        self.users = {user.user_id: user for user in users}

    def iter_users_by_ids(self, user_ids: list[int]):
        for user_id in user_ids:
            yield self.users[user_id]


def test_item_metadata_ablation_is_bounded_forward_selection() -> None:
    configs = item_metadata_ablation_configs()
    assert len(configs) == 7
    assert configs[0].name == "lightfm_id"
    assert configs[0].item_fields == ()
    assert configs[-1].item_fields == DEFAULT_INCLUDED_METADATA_FIELDS
    assert [len(config.item_fields) for config in configs] == list(range(7))
    with pytest.raises(ValueError, match="documented forward-selection order"):
        LightFMVariantConfig(name="lightfm_bad_order", item_fields=("type", "genres"))
    with pytest.raises(ValueError, match="Unsupported"):
        LightFMVariantConfig(name="lightfm_bad_field", item_fields=("score",))


def test_item_feature_config_only_includes_requested_metadata() -> None:
    catalog = [_catalog_item(2, "Drama"), _catalog_item(1, "Action")]
    bundle = build_lightfm_item_features(catalog, included_fields=("genres",))
    assert bundle.summary["included_fields"] == ["genres"]
    assert any(name.startswith("genre:") for name in bundle.feature_names)
    assert all(
        not name.startswith(("type:", "source:", "decade:", "content-rating:", "studio:"))
        for name in bundle.feature_names
    )
    np.testing.assert_allclose(np.asarray(bundle.matrix.sum(axis=1)).ravel(), np.ones(2))


def test_user_preferences_are_train_only_log_scaled_and_deterministic() -> None:
    catalog = [
        _catalog_item(1, "Action"),
        _catalog_item(2, "Drama"),
        _catalog_item(3, "Fantasy"),
        _catalog_item(4, "Romance"),
        _catalog_item(5, "Mystery"),
    ]
    first_user = UserSplit(
        user_id=7,
        eligible=True,
        train_positive=((1, 9), (2, 8)),
        validation_positive=((3, 10),),
        test_positive=((4, 10),),
        explicit_negative=(),
        neutral=(),
    )
    changed_holdout = UserSplit(
        user_id=7,
        eligible=True,
        train_positive=first_user.train_positive,
        validation_positive=((5, 10),),
        test_positive=((3, 10), (4, 10)),
        explicit_negative=(),
        neutral=(),
    )
    training_data = LightFMTrainingData(
        user_ids=np.asarray([7], dtype=np.int64),
        anime_ids=np.arange(1, 6, dtype=np.int64),
        interactions=sparse.coo_matrix((1, 5), dtype=np.float32),
    )
    first = build_lightfm_user_features(
        _UserStore([first_user]),  # type: ignore[arg-type]
        catalog,
        training_data,
        included_fields=("genres",),
        preference_mass=0.5,
    )
    second = build_lightfm_user_features(
        _UserStore([changed_holdout]),  # type: ignore[arg-type]
        list(reversed(catalog)),
        training_data,
        included_fields=("genres",),
        preference_mass=0.5,
    )
    assert first.feature_names == second.feature_names
    np.testing.assert_array_equal(first.matrix.toarray(), second.matrix.toarray())
    row = first.matrix.toarray()[0]
    by_name = dict(zip(first.feature_names, row.tolist(), strict=True))
    denominator = np.log1p(2) + np.log1p(1)
    assert by_name["identity-user:7"] == pytest.approx(0.5)
    assert by_name["genre:action"] == pytest.approx(0.5 * np.log1p(2) / denominator)
    assert by_name["genre:drama"] == pytest.approx(0.5 * np.log1p(1) / denominator)
    assert by_name["genre:fantasy"] == 0.0
    assert by_name["genre:romance"] == 0.0
    assert first.summary["source_interactions"] == "training positive interactions only"
    assert first.summary["heldout_interactions_accessed"] is False
    assert float(row.sum()) == pytest.approx(1.0)


def test_popularity_concentration_uses_full_catalog_and_train_order() -> None:
    catalog_ids = list(range(1, 101))
    train_counts = {anime_id: 101 - anime_id for anime_id in catalog_ids}
    exposure = {1: 50, 100: 50}
    result = recommendation_popularity_concentration(exposure, catalog_ids, train_counts)
    assert result["top_1_percent_share"] == pytest.approx(0.5)
    assert result["top_5_percent_share"] == pytest.approx(0.5)
    assert result["top_20_percent_share"] == pytest.approx(0.5)
    assert result["unique_recommended_items"] == 2
    assert result["catalog_coverage"] == pytest.approx(0.02)
    assert result["recommendation_events"] == 100
    assert result["exposure_gini"] > 0.9
    assert gini_coefficient([1, 1, 1]) == pytest.approx(0.0)
    assert gini_coefficient([0, 0, 3]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="non-negative"):
        gini_coefficient([1, -1])


def test_popularity_penalty_changes_order_and_preserves_known_exclusion(tmp_path: Path) -> None:
    index = LightFMServingIndex(
        anime_ids=np.asarray([1, 2, 3], dtype=np.int64),
        user_ids=np.asarray([7], dtype=np.int64),
        item_embeddings=np.zeros((3, 1), dtype=np.float32),
        item_biases=np.asarray([1.0, 0.9, 0.8], dtype=np.float32),
        user_embeddings=np.zeros((1, 1), dtype=np.float32),
        user_biases=np.zeros(1, dtype=np.float32),
        metadata={"variant": "lightfm_id", "selected_config": {"loss": "warp"}},
    )
    user = UserSplit(
        user_id=7,
        eligible=True,
        train_positive=((2, 9),),
        validation_positive=(),
        test_positive=((3, 9),),
        explicit_negative=(),
        neutral=(),
    )
    raw = LightFMModel(index, name="lightfm_raw", artifact_path=tmp_path / "raw.npz")
    adjusted = LightFMModel(
        index,
        name="lightfm_debiased",
        artifact_path=tmp_path / "raw.npz",
        train_positive_counts={1: 100, 2: 1, 3: 0},
        popularity_penalty_lambda=1.0,
    )
    assert raw.recommend(user, 2).anime_ids == [1, 3]
    assert adjusted.recommend(user, 2).anime_ids == [3, 1]
    assert 2 not in adjusted.recommend(user, 3).anime_ids
    assert adjusted.config["popularity_penalty_lambda"] == 1.0


def test_user_and_item_feature_export_matches_native_lightfm(tmp_path: Path) -> None:
    pytest.importorskip("lightfm")
    from lightfm import LightFM

    interactions = sparse.coo_matrix(np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32))
    user_features = sparse.csr_matrix(np.asarray([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=np.float32))
    item_features = sparse.csr_matrix(
        np.asarray(
            [
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
    )
    training_data = LightFMTrainingData(
        user_ids=np.asarray([10, 20], dtype=np.int64),
        anime_ids=np.asarray([1, 2, 3], dtype=np.int64),
        interactions=interactions,
    )
    model = LightFM(no_components=4, loss="warp", random_state=42)
    model.fit(
        interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=2,
        num_threads=1,
    )
    artifact_path = tmp_path / "user_item.npz"
    _export_lightfm_artifact(
        model,
        variant="lightfm_user_item_hybrid",
        item_features=item_features,
        user_features=user_features,
        training_data=training_data,
        output_path=artifact_path,
        metadata={"selected_config": {"loss": "warp"}},
    )
    index = LightFMServingIndex.load(artifact_path, [{"id": 1}, {"id": 2}, {"id": 3}])
    native = model.predict(
        0,
        np.arange(3, dtype=np.int32),
        user_features=user_features,
        item_features=item_features,
        num_threads=1,
    )
    np.testing.assert_allclose(index.scores_for_user(10), native, rtol=0.0, atol=1e-5)
