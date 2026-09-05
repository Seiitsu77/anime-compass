"""The fast path's own API contract.

Both bugs these cover reached a running deployment and neither was caught by the
suite, because the shared fixtures disable ALS and so never execute the fast
path at all. They are worth pinning directly rather than hoping some future
fixture happens to exercise them.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.api.schemas import AnimeRecommendation, RecommendRequest
from backend.anime_agent.fast_path import FastPathConfig, recommend_fast


class ProtocolSource:
    """A retrieval source that implements `top_candidates`."""

    def __init__(self, order: list[int]):
        self.order = order

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        blocked = set(excluded_ids)
        return [anime_id for anime_id in self.order if anime_id not in blocked][:limit]

    def quality_score(self, anime_id):
        return None


class LegacySource:
    """A source predating the retrieval protocol, like the CountSketch index."""

    def quality_score(self, anime_id):
        return None


@pytest.fixture
def catalog_by_id() -> dict[int, dict[str, Any]]:
    return {anime_id: {"id": anime_id, "title": f"T{anime_id}", "genres": []} for anime_id in range(1, 9)}


# ------------------------------------------------- response contract


def test_a_fast_path_item_validates_without_hybrid_scoring_fields():
    """The fast path has no channel blend or diversity adjustment to report.

    Requiring those made every fast-path response fail validation with a 500
    the moment a real ALS artifact was present.
    """
    item = AnimeRecommendation.model_validate(
        {"id": 1, "anime_id": 1, "title": "T1", "score": 8.0, "genres": ["Action"]}
    )
    assert item.recommendation_mode == "fast"
    assert item.score_breakdown is None
    assert item.final_score is None


def test_hybrid_items_still_carry_their_scoring_fields():
    item = AnimeRecommendation.model_validate(
        {
            "id": 1,
            "anime_id": 1,
            "title": "T1",
            "recommendation_mode": "hybrid",
            "score_breakdown": {
                "recommendation_mode": "hybrid",
                "channels": {},
                "pre_diversity_score": 0.5,
                "diversity_adjustment": 0.0,
                "final_score": 0.5,
            },
            "pre_diversity_score": 0.5,
            "diversity_adjustment": 0.0,
            "final_score": 0.5,
        }
    )
    assert item.final_score == 0.5
    assert item.score_breakdown is not None


# ------------------------------------------------- retrieval protocol


def test_a_source_without_top_candidates_is_skipped_not_called(catalog_by_id):
    """Routing a cold-start profile to the legacy index raised AttributeError."""
    result = recommend_fast(
        [],
        catalog_by_id=catalog_by_id,
        als_source=None,
        fallback_source=LegacySource(),
        limit=5,
        config=FastPathConfig(),
    )
    assert result.anime_ids == [], "no usable source should yield no candidates, not a crash"
    assert result.diagnostics["candidate_pool_size"] == 0


def test_a_usable_fallback_is_still_used(catalog_by_id):
    result = recommend_fast(
        [1],
        catalog_by_id=catalog_by_id,
        als_source=None,
        fallback_source=ProtocolSource([2, 3, 4]),
        limit=3,
        config=FastPathConfig(),
    )
    assert result.anime_ids == [2, 3, 4]


def test_an_unusable_tail_source_does_not_break_retrieval(catalog_by_id):
    from backend.anime_agent.retrieval import RetrievalConfig

    result = recommend_fast(
        [1],
        catalog_by_id=catalog_by_id,
        als_source=ProtocolSource([2, 3, 4, 5]),
        fallback_source=None,
        tail_source=LegacySource(),
        limit=3,
        config=FastPathConfig(retrieval=RetrievalConfig(item_item_top_m=10)),
    )
    assert result.anime_ids == [2, 3, 4]


# ------------------------------------------------- request contract


def test_one_per_series_is_part_of_the_request_contract():
    """It is a documented flag; the fast path used to ignore it silently."""
    assert RecommendRequest().one_per_series is False
    assert RecommendRequest(one_per_series=True).one_per_series is True
