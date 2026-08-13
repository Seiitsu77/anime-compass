from __future__ import annotations

from scripts.evaluate_recommender import (
    hard_filter_satisfaction,
    hit_rate_at_k,
    one_channel_weights,
)


def test_ablation_weights_activate_exactly_one_channel() -> None:
    weights = one_channel_weights("metadata")

    assert weights["metadata"] == 1.0
    assert sum(weights.values()) == 1.0
    assert all(value == 0.0 for channel, value in weights.items() if channel != "metadata")


def test_proxy_metrics_measure_explicit_constraints_and_series_hits() -> None:
    results = [
        {
            "id": 1,
            "title": "Mystery Example",
            "genres": ["Mystery"],
            "type": "TV",
            "score": 8.2,
            "start_year": 2022,
            "episodes": 12,
        }
    ]
    case = {
        "include_genres": ["Mystery"],
        "formats": ["TV"],
        "min_score": 8.0,
        "min_year": 2020,
        "max_episodes": 12,
    }

    assert hard_filter_satisfaction(results, case) == 1.0
    assert hit_rate_at_k(results, ["Mystery Example Season 2"]) == 1.0
