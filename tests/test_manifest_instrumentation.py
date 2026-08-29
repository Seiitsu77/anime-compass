"""Experiment manifests must record why a run's numbers are what they are."""

from __future__ import annotations

import numpy as np
import pytest

from backend.anime_agent.evaluation.runner import (
    AlignedPrimaryMetrics,
    _channel_activity,
    _score_variance,
)


def test_channel_activity_separates_weighted_from_zero_weight_channels():
    activity = _channel_activity(
        {
            "weights": {"collaborative": 0.22, "semantic_embedding": 0.0, "novelty": 0.03},
            "weight_source": "hand_set",
            "semantic_embedding_available": True,
        }
    )
    assert activity is not None
    assert activity["weighted_channels"] == {"collaborative": 0.22, "novelty": 0.03}
    assert activity["zero_weight_channels"] == ["semantic_embedding"]
    # Available but unweighted is a distinct state from unavailable.
    assert activity["semantic_artifact_loaded"] is True
    assert activity["weight_source"] == "hand_set"


def test_channel_activity_records_a_loaded_but_retired_channel():
    """The exact state that went unnoticed: wired, loaded, contributing nothing."""
    activity = _channel_activity({"weights": {"semantic_embedding": 0.0}, "semantic_embedding_available": False})
    assert activity is not None
    assert activity["zero_weight_channels"] == ["semantic_embedding"]
    assert activity["semantic_artifact_loaded"] is False


def test_channel_activity_is_absent_for_models_without_a_blend():
    assert _channel_activity({"factors": 64, "iterations": 15}) is None


def test_score_variance_reports_spread_and_zero_fraction():
    metrics = AlignedPrimaryMetrics(
        user_ids=np.asarray([1, 2, 3, 4]),
        ndcg_at_10=np.asarray([0.0, 0.0, 0.5, 0.5]),
        recall_at_10=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    variance = _score_variance(metrics)
    assert variance is not None
    assert variance["users"] == 4
    assert variance["ndcg_at_10"]["mean"] == pytest.approx(0.25)
    assert variance["ndcg_at_10"]["zero_fraction"] == pytest.approx(0.5)
    assert variance["recall_at_10"]["variance"] > 0.0


def test_score_variance_of_a_constant_model_is_zero():
    """Near-zero variance is the signal that a component is inert."""
    metrics = AlignedPrimaryMetrics(
        user_ids=np.asarray([1, 2, 3]),
        ndcg_at_10=np.asarray([0.4, 0.4, 0.4]),
        recall_at_10=np.asarray([0.0, 0.0, 0.0]),
    )
    variance = _score_variance(metrics)
    assert variance is not None
    assert variance["ndcg_at_10"]["variance"] == pytest.approx(0.0)
    assert variance["recall_at_10"]["zero_fraction"] == pytest.approx(1.0)


def test_score_variance_handles_a_single_user():
    metrics = AlignedPrimaryMetrics(
        user_ids=np.asarray([1]),
        ndcg_at_10=np.asarray([0.3]),
        recall_at_10=np.asarray([0.2]),
    )
    variance = _score_variance(metrics)
    assert variance is not None
    assert variance["ndcg_at_10"]["variance"] == 0.0


def test_score_variance_is_absent_without_metrics():
    assert _score_variance(None) is None
