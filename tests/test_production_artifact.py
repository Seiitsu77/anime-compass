"""Full-data production artifact: training, roles, loading, and fallback.

The evaluation and production artifacts answer different questions and must
never be substituted for one another. These tests pin that separation, the
full-data training path, and the serving behaviour that depends on both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.als_serving import (
    ARTIFACT_ROLE_EVALUATION,
    ARTIFACT_ROLE_PRODUCTION,
    ALSArtifactRoleError,
    ALSCatalogMismatchError,
    ALSCollaborativeIndex,
    catalog_ids_digest,
    sha256_file,
)

pytest.importorskip("scipy", reason="Training the fixture artifact uses the offline trainer")

from backend.anime_agent.evaluation.collaborative_baselines import (  # noqa: E402
    build_als_artifact_from_split,
    build_production_als_artifact,
)
from backend.anime_agent.evaluation.split import UserSplit  # noqa: E402

RATINGS = """user_id,anime_id,rating
0,1,10
0,2,9
0,3,8
0,4,3
1,1,9
1,2,10
1,3,9
2,1,8
2,2,9
3,4,10
3,5,9
3,6,8
4,4,9
4,5,10
4,6,9
5,4,10
5,5,8
5,9999,10
"""


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    return [{"id": anime_id, "genres": ["Action"]} for anime_id in (1, 2, 3, 4, 5, 6)]


@pytest.fixture
def ratings_file(tmp_path: Path) -> Path:
    path = tmp_path / "rating_complete.csv"
    path.write_text(RATINGS, encoding="utf-8")
    return path


@pytest.fixture
def production_artifact(tmp_path: Path, ratings_file: Path, catalog) -> Path:
    path = tmp_path / "als_production.npz"
    build_production_als_artifact(ratings_file, catalog, path, factors=8, iterations=25, regularization=0.01, alpha=5.0)
    return path


# ---------------------------------------------------------------- training


def test_full_data_training_counts_every_positive(tmp_path, ratings_file, catalog):
    path = tmp_path / "als.npz"
    metadata = build_production_als_artifact(ratings_file, catalog, path, factors=4, iterations=3)

    assert metadata["artifact_role"] == ARTIFACT_ROLE_PRODUCTION
    assert metadata["rows_scanned"] == 18
    # 16 positives at >= 8 inside the catalog (3+3+2+3+3+2 across six users);
    # the 3-rating is below threshold and title 9999 is outside the catalog.
    assert metadata["ratings_used"] == 16
    assert metadata["orphan_positive_rows"] == 1
    assert metadata["users_seen"] == 6
    assert metadata["positive_threshold"] == 8


def test_production_metadata_marks_itself_invalid_for_evaluation(production_artifact):
    with np.load(production_artifact, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata_json"].item()))
    assert metadata["not_valid_for_holdout_evaluation"] is True
    assert "split_sha256" not in metadata
    assert metadata["training_source"] == "all historically available positive ratings"


def test_production_metadata_pins_its_inputs(production_artifact, ratings_file, catalog):
    with np.load(production_artifact, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata_json"].item()))
    assert metadata["ratings_sha256"] == sha256_file(ratings_file)
    assert metadata["catalog_ids_sha256"] == catalog_ids_digest(sorted(i["id"] for i in catalog))


def test_row_limit_stops_early(tmp_path, ratings_file, catalog):
    path = tmp_path / "als.npz"
    metadata = build_production_als_artifact(ratings_file, catalog, path, factors=4, iterations=2, row_limit=5)
    assert metadata["rows_scanned"] == 5
    assert metadata["ratings_used"] < 16


def test_training_is_deterministic_for_a_fixed_seed(tmp_path, ratings_file, catalog):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    build_production_als_artifact(ratings_file, catalog, first, factors=8, iterations=5, seed=11)
    build_production_als_artifact(ratings_file, catalog, second, factors=8, iterations=5, seed=11)
    with np.load(first) as a, np.load(second) as b:
        np.testing.assert_array_equal(a["item_factors"], b["item_factors"])


def test_higher_threshold_keeps_fewer_positives(tmp_path, ratings_file, catalog):
    strict = build_production_als_artifact(
        ratings_file, catalog, tmp_path / "s.npz", factors=4, iterations=2, positive_threshold=10
    )
    assert strict["ratings_used"] < 16


def test_malformed_input_is_rejected(tmp_path, catalog):
    bad_header = tmp_path / "bad.csv"
    bad_header.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected rating file header"):
        build_production_als_artifact(bad_header, catalog, tmp_path / "x.npz", factors=4, iterations=1)

    unsorted = tmp_path / "unsorted.csv"
    unsorted.write_text("user_id,anime_id,rating\n5,1,9\n1,2,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sorted by user_id"):
        build_production_als_artifact(unsorted, catalog, tmp_path / "y.npz", factors=4, iterations=1)

    out_of_range = tmp_path / "range.csv"
    out_of_range.write_text("user_id,anime_id,rating\n1,1,42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Rating outside"):
        build_production_als_artifact(out_of_range, catalog, tmp_path / "z.npz", factors=4, iterations=1)


def test_no_positives_is_rejected(tmp_path, catalog):
    path = tmp_path / "low.csv"
    path.write_text("user_id,anime_id,rating\n1,1,3\n1,2,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No positive interactions"):
        build_production_als_artifact(path, catalog, tmp_path / "x.npz", factors=4, iterations=1)


def test_invalid_hyperparameters_are_rejected(tmp_path, ratings_file, catalog):
    for kwargs in ({"factors": 0}, {"iterations": 0}, {"alpha": 0.0}, {"positive_threshold": 42}):
        with pytest.raises(ValueError):
            build_production_als_artifact(ratings_file, catalog, tmp_path / "x.npz", **kwargs)


# ---------------------------------------------------------------- roles


def test_production_artifact_loads_under_its_own_role(production_artifact, catalog):
    index = ALSCollaborativeIndex.load(production_artifact, catalog, expected_role=ARTIFACT_ROLE_PRODUCTION)
    info = index.model_info()
    assert info["artifact_role"] == ARTIFACT_ROLE_PRODUCTION
    assert info["valid_for_holdout_evaluation"] is False
    assert info["ratings_used"] == 16


def test_evaluation_artifact_is_refused_for_production(tmp_path, catalog):
    """Serving an evaluation build would quietly ship a weaker model."""
    split = tmp_path / "split.sqlite"
    split.write_bytes(b"fixture")

    class Store:
        path = split

        def iter_users(self, *, eligible_only: bool = False):
            yield UserSplit(
                user_id=1,
                eligible=True,
                train_positive=((1, 10), (2, 9)),
                validation_positive=(),
                test_positive=(),
                explicit_negative=(),
                neutral=(),
            )

    evaluation = tmp_path / "eval.npz"
    build_als_artifact_from_split(Store(), catalog, evaluation, factors=4, iterations=3)

    with pytest.raises(ALSArtifactRoleError, match="role mismatch"):
        ALSCollaborativeIndex.load(evaluation, catalog, expected_role=ARTIFACT_ROLE_PRODUCTION)

    # It still loads for the job it is for.
    index = ALSCollaborativeIndex.load(evaluation, catalog, expected_role=ARTIFACT_ROLE_EVALUATION)
    assert index.model_info()["valid_for_holdout_evaluation"] is True


def test_production_artifact_is_refused_for_evaluation(production_artifact, catalog):
    """The reverse guard: a production build must never produce holdout metrics."""
    with pytest.raises(ALSArtifactRoleError):
        ALSCollaborativeIndex.load(production_artifact, catalog, expected_role=ARTIFACT_ROLE_EVALUATION)


def test_role_is_not_checked_when_the_caller_does_not_ask(production_artifact, catalog):
    index = ALSCollaborativeIndex.load(production_artifact, catalog)
    assert index.model_info()["artifact_role"] == ARTIFACT_ROLE_PRODUCTION


# ---------------------------------------------------------------- serving


def test_fold_in_recovers_the_planted_clusters(production_artifact, catalog):
    """Users 0-2 like {1,2,3}; users 3-5 like {4,5,6}."""
    index = ALSCollaborativeIndex.load(production_artifact, catalog)
    assert index.top_candidates([1, 2], 1, excluded_ids=[1, 2]) == [3]
    assert index.top_candidates([4, 5], 1, excluded_ids=[4, 5]) == [6]


def test_known_items_are_excluded(production_artifact, catalog):
    index = ALSCollaborativeIndex.load(production_artifact, catalog)
    ranking = index.top_candidates([1, 2], 6, excluded_ids=[1, 2, 3])
    assert not ({1, 2, 3} & set(ranking))


def test_profile_items_are_never_returned_even_without_exclusions(production_artifact, catalog):
    index = ALSCollaborativeIndex.load(production_artifact, catalog)
    assert not ({1, 2} & set(index.top_candidates([1, 2], 6)))


def test_catalog_alignment_is_verified(production_artifact, catalog):
    index = ALSCollaborativeIndex.load(production_artifact, catalog)
    assert set(index.anime_ids.tolist()) == {int(item["id"]) for item in catalog}
    assert index.model_info()["items"] == len(catalog)


def test_catalog_drift_beyond_tolerance_is_refused(production_artifact):
    unrelated = [{"id": anime_id} for anime_id in range(9000, 9020)]
    with pytest.raises(ALSCatalogMismatchError, match="does not match the active catalog"):
        ALSCollaborativeIndex.load(production_artifact, unrelated)


def test_pinned_catalog_digest_catches_drift_the_ratio_would_miss(production_artifact, catalog):
    """Overlap can stay high while the catalog identity changes."""
    with pytest.raises(ALSCatalogMismatchError, match="catalog pin mismatch"):
        ALSCollaborativeIndex.load(production_artifact, catalog, expected_catalog_ids_sha256="0" * 64)

    exact = catalog_ids_digest(sorted(int(item["id"]) for item in catalog))
    index = ALSCollaborativeIndex.load(production_artifact, catalog, expected_catalog_ids_sha256=exact)
    assert index.model_info()["items"] == len(catalog)


def test_artifact_catalog_digest_is_enforced_without_environment_pin(production_artifact, catalog):
    """The model's own provenance must reject subtle catalog drift by default."""
    drifted = [*catalog, {"id": 7, "genres": ["Action"]}]
    with pytest.raises(ALSCatalogMismatchError, match="Catalog identity mismatch"):
        ALSCollaborativeIndex.load(production_artifact, drifted)


def test_pinned_artifact_hash_is_enforced(production_artifact, catalog):
    good = sha256_file(production_artifact)
    assert ALSCollaborativeIndex.load(production_artifact, catalog, expected_artifact_sha256=good)
    with pytest.raises(Exception, match="checksum mismatch"):
        ALSCollaborativeIndex.load(production_artifact, catalog, expected_artifact_sha256="0" * 64)


# ---------------------------------------------------------------- fallback


def test_missing_production_artifact_degrades(tmp_path, catalog):
    from app.core.config import Settings
    from app.main import _load_als_index

    settings = Settings(_env_file=None, als_artifact_path=tmp_path / "absent.npz")
    assert _load_als_index(settings, catalog) is None


def test_production_artifact_serves_through_the_app_loader(production_artifact, catalog):
    from app.core.config import Settings
    from app.main import _load_als_index

    settings = Settings(
        _env_file=None,
        als_artifact_path=production_artifact,
        als_expected_sha256=sha256_file(production_artifact),
        als_expected_catalog_ids_sha256=catalog_ids_digest(sorted(int(i["id"]) for i in catalog)),
    )
    index = _load_als_index(settings, catalog)
    assert index is not None
    assert index.model_info()["artifact_role"] == ARTIFACT_ROLE_PRODUCTION


def test_evaluation_artifact_remains_untouched_by_production_builds():
    """The published metrics depend on this file being byte-stable."""
    evaluation = Path("data/evaluation/personalized/artifacts/holdout_seed42_pos8/als_train_only.npz")
    if not evaluation.exists():
        pytest.skip("evaluation artifact is not present in this checkout")
    assert sha256_file(evaluation) == "a0be5f3f1dde0a406d2bd14af705467a4b8155e8089a286e578c2f6f0ded354b"
