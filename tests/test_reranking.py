"""Second-stage reranker: leakage guards and ordering contracts.

The reranker's whole claim rests on its features being computable at
recommendation time. These tests pin that, plus the properties the serving path
would depend on if it were promoted: reranking is a permutation, it is
deterministic, and it never introduces an item the retriever did not supply.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.anime_agent.evaluation.reranking import (
    FEATURE_NAMES,
    N_FEATURES,
    LinearReranker,
    RerankerFeatureSpace,
    StandardScaler,
)


@pytest.fixture
def catalog() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "title": "A",
            "genres": ["Action"],
            "studios": ["S1"],
            "source": "Manga",
            "type": "TV",
            "start_year": 2010,
        },
        {
            "id": 2,
            "title": "B",
            "genres": ["Romance"],
            "studios": ["S2"],
            "source": "Novel",
            "type": "TV",
            "start_year": 2012,
        },
        {
            "id": 3,
            "title": "C",
            "genres": ["Action", "Sci-Fi"],
            "studios": ["S1"],
            "source": "Manga",
            "type": "Movie",
            "start_year": 2014,
        },
        {"id": 4, "title": "D", "genres": [], "studios": [], "source": "", "type": "", "start_year": None},
    ]


@pytest.fixture
def space(catalog) -> RerankerFeatureSpace:
    anime_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    return RerankerFeatureSpace.from_artifacts(catalog, anime_ids)


# ----------------------------------------------------------- leakage guards


def test_features_are_built_only_from_the_supplied_profile(space):
    """The signature is the guard: nothing but profile rows reaches a feature."""
    scores = np.asarray([0.9, 0.5, 0.1], dtype=np.float32)
    with_profile = space.build([0], [1, 2, 3], scores)
    without = space.build([], [1, 2, 3], scores)
    assert not np.allclose(with_profile, without), "profile rows must influence features"
    # A different held-out set cannot change anything, because it is never passed.
    assert np.allclose(with_profile, space.build([0], [1, 2, 3], scores))


def test_every_popularity_feature_comes_from_a_train_only_artifact(space, catalog):
    """Catalog `score` and `members` are all-time aggregates and must not leak in."""
    anime_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    enriched = [{**item, "score": 9.9, "members": 999_999, "favorites": 12345} for item in catalog]
    other = RerankerFeatureSpace.from_artifacts(enriched, anime_ids)
    scores = np.asarray([0.9, 0.5, 0.1], dtype=np.float32)
    assert np.allclose(space.build([0], [1, 2, 3], scores), other.build([0], [1, 2, 3], scores)), (
        "catalog-wide audience aggregates changed a feature; only train-only statistics may be used"
    )


def test_feature_matrix_shape_matches_the_declared_names(space):
    features = space.build([0], [1, 2, 3], np.asarray([0.9, 0.5, 0.1], dtype=np.float32))
    assert features.shape == (3, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES, "feature names must be unique"


def test_features_are_finite_even_for_items_with_no_metadata(space):
    """Item 4 has no genres, studio, source, type, or year."""
    features = space.build([0], [3], np.asarray([0.1], dtype=np.float32))
    assert np.isfinite(features).all()


def test_an_empty_candidate_list_yields_an_empty_matrix(space):
    assert space.build([0], [], np.asarray([], dtype=np.float32)).shape == (0, N_FEATURES)


# ---------------------------------------------------------- ordering contract


def make_training_set(space, users: int = 40, seed: int = 7):
    generator = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _ in range(users):
        scores = np.sort(generator.normal(size=3).astype(np.float32))[::-1]
        block = space.build([0], [1, 2, 3], scores)
        features.append(block)
        labels.append(generator.integers(0, 2, 3).astype(np.float32))
    return np.vstack(features), np.concatenate(labels)


def test_fitting_is_deterministic(space):
    features, labels = make_training_set(space)
    first = LinearReranker.fit(features, labels)
    second = LinearReranker.fit(features, labels)
    assert np.allclose(first.weights, second.weights)
    assert first.bias == pytest.approx(second.bias)


def test_reranking_is_a_permutation_of_the_candidate_set(space):
    """A reranker orders candidates. It may not add or drop one."""
    features, labels = make_training_set(space)
    model = LinearReranker.fit(features, labels)
    rows = [1, 2, 3]
    block = space.build([0], rows, np.asarray([0.9, 0.5, 0.1], dtype=np.float32))
    order = np.argsort(-model.score(block), kind="stable")
    reranked = [rows[index] for index in order]
    assert sorted(reranked) == sorted(rows)
    assert len(reranked) == len(rows)


def test_scoring_is_deterministic_for_identical_input(space):
    features, labels = make_training_set(space)
    model = LinearReranker.fit(features, labels)
    block = space.build([0], [1, 2, 3], np.asarray([0.9, 0.5, 0.1], dtype=np.float32))
    assert np.allclose(model.score(block), model.score(block))


def test_a_serialised_model_round_trips_to_the_same_scores(space):
    features, labels = make_training_set(space)
    model = LinearReranker.fit(features, labels)
    payload = model.as_dict()
    restored = LinearReranker(
        np.asarray(payload["weights"], dtype=np.float32),
        payload["bias"],
        StandardScaler(
            np.asarray(payload["scaler_mean"], dtype=np.float32),
            np.asarray(payload["scaler_scale"], dtype=np.float32),
        ),
    )
    block = space.build([0], [1, 2, 3], np.asarray([0.9, 0.5, 0.1], dtype=np.float32))
    assert np.allclose(model.score(block), restored.score(block))
    assert payload["feature_names"] == list(FEATURE_NAMES)


def test_the_scaler_never_divides_by_a_degenerate_scale():
    constant = np.ones((10, 3), dtype=np.float32)
    scaler = StandardScaler.fit(constant)
    assert (scaler.scale > 0).all()
    assert np.isfinite(scaler.transform(constant)).all()


# ------------------------------------------------------------ serving shape


def test_serving_needs_only_numpy_arrays_and_a_dot_product(space):
    """The linear arm is the NumPy-only option; its state is three arrays."""
    features, labels = make_training_set(space)
    payload = LinearReranker.fit(features, labels).as_dict()
    assert set(payload) == {"feature_names", "weights", "bias", "scaler_mean", "scaler_scale"}
    assert len(payload["weights"]) == N_FEATURES
    assert len(payload["scaler_mean"]) == N_FEATURES
    assert len(payload["scaler_scale"]) == N_FEATURES
