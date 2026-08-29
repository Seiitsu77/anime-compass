from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.anime_agent.evaluation.fusion import (
    CHANNELS,
    PairwiseDataset,
    baseline_pairwise_accuracy,
    fit_pairwise_weights,
    load_fusion_weights,
    save_fusion_artifact,
)
from backend.anime_agent.recommender import DEFAULT_CHANNEL_WEIGHTS


def dataset_favouring(channel: str, rows: int = 400, seed: int = 0) -> PairwiseDataset:
    """Pairs where only `channel` carries a consistent positive margin."""
    generator = np.random.default_rng(seed)
    differences = generator.normal(0.0, 0.05, size=(rows, len(CHANNELS)))
    differences[:, CHANNELS.index(channel)] = generator.uniform(0.2, 1.0, size=rows)
    return PairwiseDataset(differences=differences, users=rows // 4, positives_covered=rows)


def test_channels_match_the_production_weight_vector():
    assert CHANNELS == tuple(DEFAULT_CHANNEL_WEIGHTS)


def test_fit_concentrates_weight_on_the_informative_channel():
    result = fit_pairwise_weights(dataset_favouring("collaborative"), iterations=400)
    weights = result["weights"]
    assert weights["collaborative"] == pytest.approx(max(weights.values()))
    assert weights["collaborative"] > DEFAULT_CHANNEL_WEIGHTS["collaborative"]


def test_fitted_weights_are_non_negative_and_normalised():
    result = fit_pairwise_weights(dataset_favouring("quality"), iterations=300)
    weights = result["weights"]
    assert set(weights) == set(CHANNELS)
    assert all(value >= 0.0 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fit_improves_pairwise_accuracy_over_the_hand_set_blend():
    data = dataset_favouring("semantic_embedding")
    result = fit_pairwise_weights(data, iterations=500)
    hand_set = baseline_pairwise_accuracy(data, DEFAULT_CHANNEL_WEIGHTS)
    learned = baseline_pairwise_accuracy(data, result["weights"])
    assert learned >= hand_set


def test_loss_decreases_during_training():
    result = fit_pairwise_weights(dataset_favouring("metadata"), iterations=300)
    assert result["final_loss"] < result["initial_loss"]


def test_fit_is_deterministic_for_a_fixed_seed():
    data = dataset_favouring("novelty")
    first = fit_pairwise_weights(data, iterations=120, seed=11)
    second = fit_pairwise_weights(data, iterations=120, seed=11)
    assert first["weights"] == second["weights"]


def test_fit_rejects_invalid_optimiser_settings():
    data = dataset_favouring("lsa", rows=40)
    with pytest.raises(ValueError):
        fit_pairwise_weights(data, learning_rate=0.0)
    with pytest.raises(ValueError):
        fit_pairwise_weights(data, iterations=0)


def test_artifact_round_trip_preserves_weights(tmp_path: Path):
    result = fit_pairwise_weights(dataset_favouring("creator"), iterations=150)
    path = tmp_path / "learned_weights.json"
    save_fusion_artifact(result, path, split_sha256="abc123")

    loaded = load_fusion_weights(path)
    assert loaded == pytest.approx(result["weights"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_sha256"] == "abc123"
    assert payload["baseline_weights"] == DEFAULT_CHANNEL_WEIGHTS


def test_loading_rejects_an_unsupported_version(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"artifact_version": 99, "weights": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported fusion artifact version"):
        load_fusion_weights(path)


def test_loading_rejects_a_missing_channel(tmp_path: Path):
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps({"artifact_version": 1, "weights": {"metadata": 1.0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing channels"):
        load_fusion_weights(path)


def test_loading_rejects_negative_weights(tmp_path: Path):
    path = tmp_path / "negative.json"
    weights = dict.fromkeys(CHANNELS, 0.1)
    weights["quality"] = -0.5
    path.write_text(json.dumps({"artifact_version": 1, "weights": weights}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_fusion_weights(path)


class StubRecommender:
    """Emits channel breakdowns in the shape the real recommender produces."""

    def __init__(self, relevant: set[int], catalog_size: int = 40):
        self.relevant = relevant
        self.catalog_size = catalog_size
        self.calls: list[dict] = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("limit", 10)
        results = []
        for anime_id in range(1, min(limit, self.catalog_size) + 1):
            # Relevant items carry a stronger collaborative signal.
            strength = 0.9 if anime_id in self.relevant else 0.1
            results.append(
                {
                    "id": anime_id,
                    "score_breakdown": {
                        "channels": {
                            channel: {"raw_score": strength if channel == "collaborative" else 0.2}
                            for channel in CHANNELS
                        }
                    },
                }
            )
        return results


def split_for(user_id: int, train: list[int], validation: list[int]):
    from backend.anime_agent.evaluation.split import UserSplit

    return UserSplit(
        user_id=user_id,
        eligible=True,
        train_positive=tuple((anime_id, 9) for anime_id in train),
        validation_positive=tuple((anime_id, 9) for anime_id in validation),
        test_positive=(),
        explicit_negative=(),
        neutral=(),
    )


def test_extract_channel_signals_reads_the_recommender_breakdown():
    from backend.anime_agent.evaluation.fusion import extract_channel_signals

    recommender = StubRecommender(relevant={2})
    signals = extract_channel_signals(recommender, [1], [], shortlist=5)

    assert set(signals) == {1, 2, 3, 4, 5}
    assert len(signals[1]) == len(CHANNELS)
    # The breakdown is requested without reason strings: explain() would run
    # over every shortlist item and is pure cost for a feature extractor.
    assert recommender.calls[0]["include_score_breakdown"] is True
    assert recommender.calls[0]["include_explanations"] is False
    assert recommender.calls[0]["limit"] == 5


def test_build_pairwise_dataset_pairs_positives_against_negatives():
    from backend.anime_agent.evaluation.fusion import build_pairwise_dataset

    recommender = StubRecommender(relevant={2, 3})
    users = [split_for(1, [10], [2, 3]), split_for(2, [11], [2])]
    data = build_pairwise_dataset(recommender, users, shortlist=10, negatives_per_positive=2)

    assert data.users == 2
    assert data.positives_covered == 3
    assert len(data) == 3 * 2
    assert data.differences.shape[1] == len(CHANNELS)


def test_build_pairwise_dataset_counts_positives_outside_the_shortlist():
    from backend.anime_agent.evaluation.fusion import build_pairwise_dataset

    recommender = StubRecommender(relevant={2})
    # Item 99 is relevant but never surfaced by a shortlist of 5.
    users = [split_for(1, [10], [2, 99])]
    data = build_pairwise_dataset(recommender, users, shortlist=5, negatives_per_positive=1)

    assert data.positives_covered == 1
    assert data.positives_missed == 1


def test_build_pairwise_dataset_learns_the_planted_signal():
    from backend.anime_agent.evaluation.fusion import build_pairwise_dataset

    recommender = StubRecommender(relevant={2, 3, 4})
    users = [split_for(index, [10 + index], [2, 3, 4]) for index in range(1, 8)]
    data = build_pairwise_dataset(recommender, users, shortlist=20, negatives_per_positive=4)
    result = fit_pairwise_weights(data, iterations=400)

    assert result["weights"]["collaborative"] == pytest.approx(max(result["weights"].values()))


def test_build_pairwise_dataset_rejects_bad_arguments():
    from backend.anime_agent.evaluation.fusion import build_pairwise_dataset

    recommender = StubRecommender(relevant={2})
    users = [split_for(1, [10], [2])]
    with pytest.raises(ValueError):
        build_pairwise_dataset(recommender, users, holdout="train")
    with pytest.raises(ValueError):
        build_pairwise_dataset(recommender, users, shortlist=1)


def test_build_pairwise_dataset_raises_when_no_pairs_exist():
    from backend.anime_agent.evaluation.fusion import build_pairwise_dataset

    recommender = StubRecommender(relevant=set())
    users = [split_for(1, [10], [])]
    with pytest.raises(ValueError, match="No usable"):
        build_pairwise_dataset(recommender, users, shortlist=10)


def test_zero_diversity_ranking_matches_score_order(catalog):
    """The no-diversity fast path must return exactly the score-sorted order."""
    from backend.anime_agent.recommender import AnimeRecommender

    recommender = AnimeRecommender(catalog)
    results = recommender.recommend(
        liked_ids=[1],
        session_profile={},
        diversity_strength=0.0,
        exclude_related_series=False,
        one_per_series=False,
        limit=8,
        include_explanations=False,
        include_score_breakdown=True,
    )
    scores = [
        sum(
            channel["raw_score"] * channel["effective_weight"]
            for channel in item["score_breakdown"]["channels"].values()
        )
        for item in results
    ]
    assert scores == sorted(scores, reverse=True)


def test_diversity_reranking_is_still_applied_when_requested(catalog):
    """The fast path must not silently disable diversity when it was asked for."""
    from backend.anime_agent.recommender import AnimeRecommender

    recommender = AnimeRecommender(catalog)
    calls: list[int] = []
    original = recommender._diversity_penalty

    def counting_penalty(item, selected):
        calls.append(int(item["id"]))
        return original(item, selected)

    recommender._diversity_penalty = counting_penalty  # type: ignore[method-assign]

    recommender.recommend(
        liked_ids=[1],
        session_profile={},
        diversity_strength=0.0,
        exclude_related_series=False,
        limit=5,
        include_explanations=False,
    )
    assert calls == [], "zero diversity must not pay for penalty computation"

    recommender.recommend(
        liked_ids=[1],
        session_profile={},
        diversity_strength=0.9,
        exclude_related_series=False,
        limit=5,
        include_explanations=False,
    )
    assert calls, "a non-zero diversity strength must still rerank"
