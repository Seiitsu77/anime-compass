"""Training and serving must build the same feature row.

The reranker is a frozen model whose trees split on absolute feature values, so
a serving-side transform that training never applied is not a small error — it
moves every threshold. This already bit once: `profile_scores` normalises to
[0, 1] while the model was fitted on raw scores, which would have left
`als_score_z` intact and silently shifted `als_score`.

These tests pin the two halves of that guarantee: the offline feature space and
the packed serving artifact produce identical rows, and the raw-score path stays
raw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.evaluation.reranking import FEATURE_NAMES, RerankerFeatureSpace

SPLIT_ARTIFACTS = Path("data/evaluation/personalized/artifacts/holdout_seed42_pos8")
FEATURE_ARTIFACT = Path("data/processed/reranker_features.npz")
CATALOG = Path("data/processed/anime_catalog.json")


def requires_artifacts():
    missing = [
        path
        for path in (
            FEATURE_ARTIFACT,
            CATALOG,
            SPLIT_ARTIFACTS / "countsketch_train_only.npz",
            SPLIT_ARTIFACTS / "popularity_train_only.npz",
            SPLIT_ARTIFACTS / "item_item_train_only.npz",
        )
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing artifacts: {', '.join(p.name for p in missing)}")


@pytest.fixture(scope="module")
def catalog():
    if not CATALOG.exists():
        pytest.skip("processed catalog is not present in this checkout")
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_offline_and_serving_feature_rows_are_identical(catalog):
    """The training feature space and the packed serving artifact must agree."""
    requires_artifacts()
    with np.load(FEATURE_ARTIFACT, allow_pickle=False) as payload:
        anime_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
        serving = RerankerFeatureSpace.from_prepared(catalog, anime_ids, payload)

    offline = RerankerFeatureSpace.from_artifacts(
        catalog,
        anime_ids,
        popularity_path=SPLIT_ARTIFACTS / "popularity_train_only.npz",
        quality_path=SPLIT_ARTIFACTS / "countsketch_train_only.npz",
        item_item_path=SPLIT_ARTIFACTS / "item_item_train_only.npz",
    )

    generator = np.random.default_rng(20260904)
    for _ in range(5):
        profile = sorted(generator.choice(len(anime_ids), size=12, replace=False).tolist())
        candidates = sorted(generator.choice(len(anime_ids), size=200, replace=False).tolist())
        scores = generator.normal(size=len(candidates)).astype(np.float32)
        offline_rows = offline.build(profile, candidates, scores)
        serving_rows = serving.build(profile, candidates, scores)
        assert offline_rows.shape == serving_rows.shape
        differing = [
            FEATURE_NAMES[column]
            for column in range(offline_rows.shape[1])
            if not np.allclose(offline_rows[:, column], serving_rows[:, column], atol=1e-6)
        ]
        assert not differing, f"training/serving feature mismatch in: {differing}"


def test_raw_als_scores_are_not_normalised(catalog):
    """`raw_profile_scores` must stay raw; `profile_scores` divides by the peak."""
    artifact = Path("data/processed/als_production_item_factors.npz")
    if not artifact.exists():
        pytest.skip("production ALS artifact is not present in this checkout")
    index = ALSCollaborativeIndex.load(artifact, catalog)
    profile = [9253, 1535, 13601]

    raw = index.raw_profile_scores(profile)
    assert raw is not None
    normalised = index.profile_scores(profile)
    assert float(max(normalised.values())) == pytest.approx(1.0), "profile_scores should peak at 1.0"

    # The distinguishing property is not the magnitude -- raw scores here happen
    # to sit below 1 -- but that raw scores are *not* rescaled so their peak is
    # exactly 1. That rescaling is what would move als_score off the thresholds
    # the trees learned while leaving als_score_z untouched.
    assert float(raw.max()) != pytest.approx(1.0), "raw scores must not be peak-normalised"

    row_by_id = index.index_by_id
    sample = [anime_id for anime_id in list(normalised)[:50] if anime_id in row_by_id]
    ratios = [normalised[anime_id] / float(raw[row_by_id[anime_id]]) for anime_id in sample]
    assert np.allclose(ratios, ratios[0], rtol=1e-4), "normalisation is not a single positive scaling"
    assert ratios[0] != pytest.approx(1.0), "the two paths would be indistinguishable"


def test_the_serving_artifact_declares_its_feature_schema():
    requires_artifacts()
    with np.load(FEATURE_ARTIFACT, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["artifact_version"] == 1
    assert metadata["items"] == 18064
    assert metadata["neighbors_per_item"] == 200
    assert "global_rating_mean" in metadata


def test_a_feature_schema_change_would_be_caught():
    """The model pins its input width; changing FEATURE_NAMES must not slip through."""
    from backend.anime_agent.reranker_serving import EXPECTED_FEATURE_COUNT

    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT, (
        "FEATURE_NAMES changed without bumping EXPECTED_FEATURE_COUNT and retraining; "
        "the frozen model would be served features it was never fitted on"
    )
