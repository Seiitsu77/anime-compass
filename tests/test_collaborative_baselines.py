from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.evaluation.collaborative_baselines import (
    ALSModel,
    ItemItemModel,
    build_als_artifact_from_split,
    build_item_item_artifact_from_split,
)
from backend.anime_agent.evaluation.split import UserSplit

pytest.importorskip("scipy", reason="The reference baselines require SciPy")


class FakeStore:
    """Minimal SplitStore stand-in over hand-built user rows."""

    def __init__(self, users: list[UserSplit], path: Path):
        self._users = users
        self.path = path

    def iter_users(self, *, eligible_only: bool = False):
        for user in self._users:
            if eligible_only and not user.eligible:
                continue
            yield user


def user(user_id: int, train: list[tuple[int, int]], test: list[tuple[int, int]] | None = None) -> UserSplit:
    return UserSplit(
        user_id=user_id,
        eligible=True,
        train_positive=tuple(train),
        validation_positive=(),
        test_positive=tuple(test or ()),
        explicit_negative=(),
        neutral=(),
    )


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    return [{"id": anime_id} for anime_id in (1, 2, 3, 4, 5, 6)]


@pytest.fixture
def store(tmp_path: Path) -> FakeStore:
    """Two clean taste clusters: {1,2,3} and {4,5,6}."""
    split_path = tmp_path / "split.sqlite"
    split_path.write_bytes(b"fixture")
    users = [
        user(1, [(1, 10), (2, 9), (3, 10)]),
        user(2, [(1, 9), (2, 10), (3, 9)]),
        user(3, [(1, 10), (2, 9)]),
        user(4, [(4, 10), (5, 9), (6, 10)]),
        user(5, [(4, 9), (5, 10), (6, 9)]),
        user(6, [(4, 10), (5, 10)]),
    ]
    return FakeStore(users, split_path)


def test_item_item_recovers_the_cluster_structure(store, catalog, tmp_path):
    path = tmp_path / "item_item.npz"
    metadata = build_item_item_artifact_from_split(store, catalog, path, neighbors=3, block_size=2)

    assert metadata["method"] == "exact user-centred adjusted-cosine item similarity"
    assert metadata["users_seen"] == 6
    assert path.exists()

    model = ItemItemModel(path, [item["id"] for item in catalog], build_duration_seconds=0.0)
    # A user who liked 1 and 2 should be steered to 3, not into the other cluster.
    recommendation = model.recommend(user(99, [(1, 10), (2, 10)]), k=1)
    assert recommendation.anime_ids == [3]


def test_item_item_excludes_already_observed_items(store, catalog, tmp_path):
    path = tmp_path / "item_item.npz"
    build_item_item_artifact_from_split(store, catalog, path, neighbors=5, block_size=6)
    model = ItemItemModel(path, [item["id"] for item in catalog], build_duration_seconds=0.0)

    recommendation = model.recommend(user(99, [(1, 10), (2, 10), (3, 10)]), k=6)
    assert not ({1, 2, 3} & set(recommendation.anime_ids))


def test_als_recovers_the_cluster_structure(store, catalog, tmp_path):
    path = tmp_path / "als.npz"
    metadata = build_als_artifact_from_split(
        store, catalog, path, factors=8, iterations=30, regularization=0.01, alpha=40.0
    )

    assert metadata["method"] == "implicit-feedback ALS (conjugate gradient)"
    assert metadata["factors"] == 8

    model = ALSModel(path, [item["id"] for item in catalog], build_duration_seconds=0.0)
    recommendation = model.recommend(user(99, [(1, 10), (2, 10)]), k=1)
    assert recommendation.anime_ids == [3]


def test_als_is_deterministic_for_a_fixed_seed(store, catalog, tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    build_als_artifact_from_split(store, catalog, first, factors=8, iterations=5, seed=7)
    build_als_artifact_from_split(store, catalog, second, factors=8, iterations=5, seed=7)
    with np.load(first) as a, np.load(second) as b:
        np.testing.assert_allclose(a["item_factors"], b["item_factors"])


def test_als_handles_a_user_with_no_known_items(store, catalog, tmp_path):
    path = tmp_path / "als.npz"
    build_als_artifact_from_split(store, catalog, path, factors=4, iterations=3)
    model = ALSModel(path, [item["id"] for item in catalog], build_duration_seconds=0.0)

    recommendation = model.recommend(user(99, [(9999, 10)]), k=3)
    assert len(recommendation.anime_ids) == 3
    assert recommendation.diagnostics["profile_score_count"] == 0


def test_both_models_reject_a_tampered_artifact_version(store, catalog, tmp_path):
    path = tmp_path / "item_item.npz"
    build_item_item_artifact_from_split(store, catalog, path, neighbors=2, block_size=6)
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["artifact_version"] = 999
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="Unsupported item-item artifact version"):
        ItemItemModel(path, [item["id"] for item in catalog], build_duration_seconds=0.0)


def test_builders_reject_invalid_hyperparameters(store, catalog, tmp_path):
    with pytest.raises(ValueError):
        build_item_item_artifact_from_split(store, catalog, tmp_path / "x.npz", neighbors=0)
    with pytest.raises(ValueError):
        build_als_artifact_from_split(store, catalog, tmp_path / "y.npz", factors=0)
    with pytest.raises(ValueError):
        build_als_artifact_from_split(store, catalog, tmp_path / "z.npz", alpha=0.0)


def test_builders_reject_an_empty_catalog(store, tmp_path):
    with pytest.raises(ValueError, match="empty catalog"):
        build_item_item_artifact_from_split(store, [], tmp_path / "x.npz")
    with pytest.raises(ValueError, match="empty catalog"):
        build_als_artifact_from_split(store, [], tmp_path / "y.npz")


def test_confidence_mapping_actually_changes_the_model(store, catalog, tmp_path):
    """A weighting that is silently ignored would produce identical factors.

    The first weighted-ALS implementation passed the confidence values into the
    sparse matrix but the conjugate-gradient solver only read the column
    indices, so every mapping produced byte-identical output. This pins that the
    values reach the solver.
    """
    binary = tmp_path / "binary.npz"
    linear = tmp_path / "linear.npz"
    build_als_artifact_from_split(store, catalog, binary, factors=8, iterations=10, confidence_mapping="binary")
    build_als_artifact_from_split(store, catalog, linear, factors=8, iterations=10, confidence_mapping="linear")

    with np.load(binary) as a, np.load(linear) as b:
        assert not np.allclose(a["item_factors"], b["item_factors"]), (
            "the confidence mapping had no effect on the learned factors"
        )


def test_binary_confidence_is_the_unweighted_reference(store, catalog, tmp_path):
    """`binary` must reproduce the original unweighted training exactly."""
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    build_als_artifact_from_split(store, catalog, first, factors=8, iterations=10, seed=3)
    build_als_artifact_from_split(store, catalog, second, factors=8, iterations=10, seed=3, confidence_mapping="binary")
    with np.load(first) as a, np.load(second) as b:
        np.testing.assert_array_equal(a["item_factors"], b["item_factors"])


def test_confidence_weights_are_recorded_in_metadata(store, catalog, tmp_path):
    path = tmp_path / "weighted.npz"
    metadata = build_als_artifact_from_split(store, catalog, path, factors=4, iterations=3, confidence_mapping="log")
    assert metadata["confidence_mapping"] == "log"
    assert metadata["confidence_weights"] == {"8": 1.0, "9": 1.585, "10": 2.0}


def test_unknown_confidence_mapping_is_rejected(store, catalog, tmp_path):
    with pytest.raises(ValueError, match="Unknown confidence mapping"):
        build_als_artifact_from_split(store, catalog, tmp_path / "x.npz", confidence_mapping="nope")


def test_random_model_is_deterministic_and_excludes_known_items(catalog):
    from backend.anime_agent.evaluation.collaborative_baselines import RandomModel

    ids = [item["id"] for item in catalog]
    model = RandomModel(ids, seed=11)
    subject = user(5, [(1, 10), (2, 9)])
    first = model.recommend(subject, 3).anime_ids
    assert first == model.recommend(subject, 3).anime_ids
    assert not ({1, 2} & set(first))


def test_oracle_places_held_out_positives_first(catalog):
    from backend.anime_agent.evaluation.collaborative_baselines import OracleModel

    ids = [item["id"] for item in catalog]
    model = OracleModel(ids, holdout="test")
    subject = user(5, [(1, 10)], test=[(4, 9), (5, 10)])
    ranking = model.recommend(subject, 4).anime_ids
    assert ranking[:2] == [4, 5]
    assert 1 not in ranking
