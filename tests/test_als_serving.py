"""Production ALS serving: artifact validation, fold-in, and offline equivalence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.als_serving import (
    ALS_ARTIFACT_VERSION,
    ALSArtifactError,
    ALSCollaborativeIndex,
    sha256_file,
)

pytest.importorskip("scipy", reason="Building a fixture artifact uses the offline trainer")

from backend.anime_agent.evaluation.collaborative_baselines import (  # noqa: E402
    ALSModel,
    build_als_artifact_from_split,
)
from backend.anime_agent.evaluation.split import UserSplit  # noqa: E402


class FakeStore:
    def __init__(self, users: list[UserSplit], path: Path):
        self._users = users
        self.path = path

    def iter_users(self, *, eligible_only: bool = False):
        yield from self._users


def user(user_id: int, train: list[tuple[int, int]]) -> UserSplit:
    return UserSplit(
        user_id=user_id,
        eligible=True,
        train_positive=tuple(train),
        validation_positive=(),
        test_positive=(),
        explicit_negative=(),
        neutral=(),
    )


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    return [{"id": anime_id, "genres": ["Action"]} for anime_id in (1, 2, 3, 4, 5, 6)]


@pytest.fixture
def artifact(tmp_path: Path, catalog) -> Path:
    """Two taste clusters: {1,2,3} and {4,5,6}."""
    split = tmp_path / "split.sqlite"
    split.write_bytes(b"fixture")
    store = FakeStore(
        [
            user(1, [(1, 10), (2, 9), (3, 10)]),
            user(2, [(1, 9), (2, 10), (3, 9)]),
            user(3, [(1, 10), (2, 9)]),
            user(4, [(4, 10), (5, 9), (6, 10)]),
            user(5, [(4, 9), (5, 10), (6, 9)]),
            user(6, [(4, 10), (5, 10)]),
        ],
        split,
    )
    path = tmp_path / "als.npz"
    build_als_artifact_from_split(store, catalog, path, factors=8, iterations=25, regularization=0.01, alpha=5.0)
    return path


def test_artifact_loads_and_reports_provenance(artifact, catalog):
    index = ALSCollaborativeIndex.load(artifact, catalog)
    info = index.model_info()
    assert info["available"] is True
    assert info["items"] == len(catalog)
    assert info["artifact_version"] == ALS_ARTIFACT_VERSION
    # Provenance must be traceable from a served recommendation.
    assert info["artifact_sha256"] == sha256_file(artifact)
    assert info["split_sha256"]


def test_checksum_mismatch_is_refused(artifact, catalog):
    with pytest.raises(ALSArtifactError, match="checksum mismatch"):
        ALSCollaborativeIndex.load(artifact, catalog, expected_artifact_sha256="0" * 64)


def test_correct_checksum_is_accepted(artifact, catalog):
    index = ALSCollaborativeIndex.load(artifact, catalog, expected_artifact_sha256=sha256_file(artifact))
    assert index.model_info()["available"] is True


def test_missing_artifact_is_refused(tmp_path, catalog):
    with pytest.raises(ALSArtifactError, match="not found"):
        ALSCollaborativeIndex.load(tmp_path / "absent.npz", catalog)


def test_unsupported_version_is_refused(artifact, catalog):
    with np.load(artifact, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["artifact_version"] = 999
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(artifact, **arrays)
    with pytest.raises(ALSArtifactError, match="Unsupported ALS artifact version"):
        ALSCollaborativeIndex.load(artifact, catalog)


def test_catalog_mismatch_is_refused(artifact):
    """Serving against the wrong catalog is worse than refusing to start."""
    unrelated = [{"id": anime_id} for anime_id in range(9000, 9010)]
    with pytest.raises(ALSArtifactError, match="does not match the active catalog"):
        ALSCollaborativeIndex.load(artifact, unrelated)


def test_non_finite_factors_are_refused(artifact, catalog):
    with np.load(artifact, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    arrays["item_factors"] = arrays["item_factors"].copy()
    arrays["item_factors"][0, 0] = np.nan
    np.savez_compressed(artifact, **arrays)
    with pytest.raises(ALSArtifactError, match="non-finite"):
        ALSCollaborativeIndex.load(artifact, catalog)


def test_serving_scores_match_the_offline_model(artifact, catalog):
    """Production serving must reproduce offline scoring, or the benchmark
    numbers do not describe what users receive."""
    ids = [item["id"] for item in catalog]
    offline = ALSModel(artifact, ids, build_duration_seconds=0.0)
    online = ALSCollaborativeIndex.load(artifact, catalog)

    liked = [1, 2]
    offline_vector = offline._user_vector(liked)
    online_vector = online.user_vector(liked)
    assert offline_vector is not None and online_vector is not None
    np.testing.assert_allclose(offline_vector, online_vector, rtol=1e-5, atol=1e-6)

    offline_scores = offline.item_factors @ offline_vector
    online_scores = online.item_factors @ online_vector
    np.testing.assert_allclose(offline_scores, online_scores, rtol=1e-5, atol=1e-6)


def test_serving_ranking_matches_offline_ranking(artifact, catalog):
    ids = [item["id"] for item in catalog]
    offline = ALSModel(artifact, ids, build_duration_seconds=0.0)
    online = ALSCollaborativeIndex.load(artifact, catalog)

    subject = user(99, [(1, 10), (2, 10)])
    offline_ranking = offline.recommend(subject, 3).anime_ids
    online_ranking = online.top_candidates([1, 2], 3, excluded_ids=[1, 2])
    assert offline_ranking == online_ranking


def test_fold_in_recovers_the_cluster(artifact, catalog):
    online = ALSCollaborativeIndex.load(artifact, catalog)
    ranking = online.top_candidates([1, 2], 1, excluded_ids=[1, 2])
    assert ranking == [3]


def test_known_items_are_excluded_from_candidates(artifact, catalog):
    online = ALSCollaborativeIndex.load(artifact, catalog)
    ranking = online.top_candidates([1, 2], 6, excluded_ids=[1, 2, 3])
    assert not ({1, 2, 3} & set(ranking))


def test_profile_scores_are_normalized_and_exclude_nothing(artifact, catalog):
    online = ALSCollaborativeIndex.load(artifact, catalog)
    scores = online.profile_scores(positive_ids=[1, 2])
    assert scores
    assert max(scores.values()) == pytest.approx(1.0)
    assert all(0.0 < value <= 1.0 for value in scores.values())


def test_explicit_high_ratings_join_the_profile(artifact, catalog):
    online = ALSCollaborativeIndex.load(artifact, catalog)
    from_ids = online.profile_scores(positive_ids=[4, 5])
    from_ratings = online.profile_scores(explicit_ratings={4: 9.0, 5: 10.0})
    assert from_ids.keys() == from_ratings.keys()
    for key in from_ids:
        assert from_ids[key] == pytest.approx(from_ratings[key], rel=1e-5)


def test_low_explicit_ratings_do_not_join_the_profile(artifact, catalog):
    """Implicit ALS has no principled place for negative confidence."""
    online = ALSCollaborativeIndex.load(artifact, catalog)
    assert online.profile_scores(explicit_ratings={4: 3.0, 5: 2.0}) == {}


def test_unknown_profile_yields_no_scores(artifact, catalog):
    online = ALSCollaborativeIndex.load(artifact, catalog)
    assert online.profile_scores(positive_ids=[999999]) == {}
    assert online.top_candidates([999999], 5) == []


def test_quality_scores_come_from_the_supplied_source(artifact, catalog):
    class Quality:
        def quality_score(self, anime_id: int) -> float | None:
            return 0.5 if anime_id == 1 else None

    plain = ALSCollaborativeIndex.load(artifact, catalog)
    assert plain.quality_score(1) is None

    with_quality = ALSCollaborativeIndex.load(artifact, catalog, quality_source=Quality())
    assert with_quality.quality_score(1) == 0.5
    assert with_quality.quality_score(2) is None


def test_index_is_reusable_across_calls(artifact, catalog):
    """The cached Gram matrix must not change results between requests."""
    online = ALSCollaborativeIndex.load(artifact, catalog)
    first = online.top_candidates([1, 2], 4, excluded_ids=[1, 2])
    second = online.top_candidates([1, 2], 4, excluded_ids=[1, 2])
    assert first == second
    assert online.resident_array_bytes > 0
