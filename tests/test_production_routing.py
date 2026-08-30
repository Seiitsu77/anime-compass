"""Segment routing, multi-source retrieval, and the default fast path."""

from __future__ import annotations

from typing import Any

import pytest

from backend.anime_agent.fast_path import FastPathConfig, recommend_fast
from backend.anime_agent.path_policy import (
    PathPolicy,
    RecommendationPath,
    choose_recommendation_path,
    describe_paths,
    required_entity_fields,
)
from backend.anime_agent.retrieval import RetrievalConfig, retrieve_candidates
from backend.anime_agent.routing import (
    CollaborativeRoute,
    RoutingPolicy,
    activity_segment,
    choose_collaborative_route,
)


class StubSource:
    """Returns a fixed ranking, honouring the exclusion list."""

    def __init__(self, ranking: list[int]):
        self.ranking = ranking
        self.calls: list[dict[str, Any]] = []

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        self.calls.append({"positives": list(positive_ids), "limit": limit, "excluded": list(excluded_ids)})
        blocked = {int(v) for v in excluded_ids}
        return [a for a in self.ranking if a not in blocked][:limit]


class LeakySource:
    """Ignores the exclusion list, to prove the fast path re-checks."""

    def __init__(self, ranking: list[int]):
        self.ranking = ranking

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        return self.ranking[:limit]


def catalog_map(ids, genres=None):
    return {
        anime_id: {"id": anime_id, "genres": genres.get(anime_id, ["Action"]) if genres else ["Action"]}
        for anime_id in ids
    }


# ---------------------------------------------------------------- routing


def test_activity_segments_match_the_offline_definitions():
    assert activity_segment(0) == "cold"
    assert activity_segment(1) == "sparse"
    assert activity_segment(4) == "sparse"
    assert activity_segment(5) == "medium"
    assert activity_segment(19) == "medium"
    assert activity_segment(20) == "heavy"


def test_every_user_with_history_routes_to_als_by_default():
    """Global ALS is the default: routing measured worse for sparse users."""
    for count in (1, 5, 37, 500):
        decision = choose_collaborative_route(list(range(count)))
        assert decision.route is CollaborativeRoute.ALS
        assert decision.known_positive_count == count
        assert decision.reason == "segment_aware_routing_disabled"


def test_medium_and_heavy_users_route_to_als_when_segmenting_is_enabled():
    policy = RoutingPolicy(segment_aware=True)
    for count in (5, 37, 500):
        decision = choose_collaborative_route(list(range(count)), policy=policy)
        assert decision.route is CollaborativeRoute.ALS
        assert decision.reason.endswith("_activity_user")


def test_sparse_users_route_to_the_fallback_only_when_segmenting_is_enabled():
    """Retained and testable, but not the default: routing sparse users to
    CountSketch measured worse (NDCG@10 0.1660 vs 0.2003 under global ALS)."""
    decision = choose_collaborative_route([1, 2, 3], policy=RoutingPolicy(segment_aware=True))
    assert decision.route is CollaborativeRoute.SPARSE_FALLBACK
    assert decision.reason == "als_sparse_gain_not_demonstrated"
    assert decision.as_dict()["known_positive_count"] == 3


def test_users_with_no_history_get_no_collaborative_route():
    decision = choose_collaborative_route([])
    assert decision.route is CollaborativeRoute.NO_COLLABORATIVE
    assert decision.reason == "no_known_positives"


def test_missing_als_artifact_degrades_rather_than_failing():
    decision = choose_collaborative_route(list(range(50)), als_available=False)
    assert decision.route is CollaborativeRoute.ALS_UNAVAILABLE_FALLBACK
    assert decision.reason == "als_artifact_unavailable"


def test_global_als_is_the_shipped_default():
    assert RoutingPolicy().segment_aware is False
    decision = choose_collaborative_route([1, 2])
    assert decision.route is CollaborativeRoute.ALS
    assert decision.reason == "segment_aware_routing_disabled"


def test_routing_threshold_is_configurable_and_validated():
    policy = RoutingPolicy(medium_threshold=10, segment_aware=True)
    assert choose_collaborative_route(list(range(9)), policy=policy).route is CollaborativeRoute.SPARSE_FALLBACK
    assert choose_collaborative_route(list(range(10)), policy=policy).route is CollaborativeRoute.ALS
    with pytest.raises(ValueError):
        RoutingPolicy(medium_threshold=0)


def test_duplicate_positives_are_counted_once():
    assert choose_collaborative_route([7, 7, 7]).known_positive_count == 1


# ---------------------------------------------------------------- retrieval


def test_union_deduplicates_and_records_every_source():
    als = StubSource([10, 11, 12])
    item_item = StubSource([12, 20, 21])
    result = retrieve_candidates(
        [1],
        {"als": als, "item_item": item_item},
        config=RetrievalConfig(als_top_n=3, item_item_top_m=3),
    )
    assert len(result.anime_ids) == len(set(result.anime_ids))
    shared = next(c for c in result.candidates if c.anime_id == 12)
    assert set(shared.sources) == {"als", "item_item"}
    assert shared.ranks["als"] == 2
    assert shared.ranks["item_item"] == 0


def test_source_scores_are_rank_normalized_not_raw():
    """Raw ALS dot products and item-item cosines are not comparable."""
    result = retrieve_candidates(
        [1],
        {"als": StubSource([10, 11, 12, 13])},
        config=RetrievalConfig(als_top_n=4),
    )
    scores = [c.sources["als"] for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)
    assert scores[0] == pytest.approx(1.0)


def test_tail_source_presence_is_recorded():
    with_tail = retrieve_candidates(
        [1],
        {"als": StubSource([1, 2]), "item_item": StubSource([3])},
        config=RetrievalConfig(item_item_top_m=100),
    )
    assert with_tail.tail_source_used is True
    assert with_tail.diagnostics()["candidate_sources"] == {"als": 2, "item_item": 1}

    without = retrieve_candidates([1], {"als": StubSource([1, 2])}, config=RetrievalConfig())
    assert without.tail_source_used is False


def test_item_item_is_opt_in_by_default():
    """Tail supplementation costs 9% relative NDCG@10, so it is not default-on."""
    assert RetrievalConfig().item_item_top_m == 0
    result = retrieve_candidates(
        [1], {"als": StubSource([1, 2]), "item_item": StubSource([3])}, config=RetrievalConfig()
    )
    assert result.tail_source_used is False
    assert result.anime_ids == [1, 2]


def test_tail_source_contributes_items_als_never_surfaces():
    """The point of item-item is tail reach, not aggregate recall."""
    result = retrieve_candidates(
        [1],
        {"als": StubSource([10, 11]), "item_item": StubSource([9001, 9002])},
        config=RetrievalConfig(als_top_n=2, item_item_top_m=2),
    )
    assert {9001, 9002}.issubset(set(result.anime_ids))


def test_retrieval_honours_the_exclusion_list():
    als = StubSource([1, 2, 3, 4])
    result = retrieve_candidates([9], {"als": als}, config=RetrievalConfig(als_top_n=4), excluded_ids=[2, 3])
    assert result.anime_ids == [1, 4]
    assert als.calls[0]["excluded"] == [2, 3]


def test_zero_limit_source_is_skipped():
    result = retrieve_candidates(
        [1],
        {"als": StubSource([1]), "item_item": StubSource([2])},
        config=RetrievalConfig(item_item_top_m=0),
    )
    assert result.anime_ids == [1]
    assert "item_item" not in result.source_counts


def test_retrieval_config_is_validated():
    with pytest.raises(ValueError):
        RetrievalConfig(als_top_n=0)
    with pytest.raises(ValueError):
        RetrievalConfig(item_item_top_m=-1)


# ---------------------------------------------------------------- fast path


def test_fast_path_uses_als_for_a_medium_user():
    als = StubSource([10, 11, 12])
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=als,
        fallback_source=StubSource([90]),
        limit=3,
    )
    assert result.routing.route is CollaborativeRoute.ALS
    assert result.anime_ids == [10, 11, 12]
    assert result.diagnostics["candidate_sources"]["als"] == 3


def test_fast_path_uses_the_fallback_for_a_sparse_user_when_segmenting_is_on():
    fallback = StubSource([90, 91])
    result = recommend_fast(
        [1, 2],
        catalog_by_id=catalog_map([10, 90, 91]),
        als_source=StubSource([10]),
        fallback_source=fallback,
        limit=2,
        config=FastPathConfig(routing=RoutingPolicy(segment_aware=True)),
    )
    assert result.routing.route is CollaborativeRoute.SPARSE_FALLBACK
    assert result.anime_ids == [90, 91]
    assert "countsketch" in result.diagnostics["candidate_sources"]


def test_fast_path_falls_back_when_als_is_unavailable():
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([90]),
        als_source=None,
        fallback_source=StubSource([90]),
        limit=1,
    )
    assert result.routing.route is CollaborativeRoute.ALS_UNAVAILABLE_FALLBACK
    assert result.anime_ids == [90]


def test_fast_path_never_returns_excluded_items():
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=StubSource([10, 11, 12]),
        fallback_source=None,
        excluded_ids=[11],
        limit=3,
    )
    assert 11 not in result.anime_ids


def test_fast_path_rechecks_exclusions_a_leaky_source_ignored():
    """A source that ignores excluded_ids must not leak into the response."""
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=LeakySource([10, 11, 12]),
        fallback_source=None,
        excluded_ids=[10, 11],
        limit=3,
    )
    assert result.anime_ids == [12]


def test_fast_path_applies_hard_filters_as_hard():
    """A required constraint is a filter, never a down-weight."""
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=StubSource([10, 11, 12]),
        fallback_source=None,
        allowed_ids={12},
        limit=3,
    )
    assert result.anime_ids == [12]
    assert result.diagnostics["hard_filter_applied"] is True


def test_fast_path_drops_candidates_absent_from_the_catalog():
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10]),
        als_source=StubSource([10, 99999]),
        fallback_source=None,
        limit=5,
    )
    assert result.anime_ids == [10]


def test_diversity_is_disabled_by_default():
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=StubSource([10, 11, 12]),
        fallback_source=None,
        limit=3,
    )
    assert result.diagnostics["diversity_applied"] is False
    assert result.diagnostics["diversity_window"] == 0


def test_diversity_reranks_within_a_bounded_window_when_enabled():
    genres = {10: ["Action"], 11: ["Action"], 12: ["Romance"]}
    config = FastPathConfig(diversity_strength=0.9, diversity_window=10, quality_weight=0.0)
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12], genres),
        als_source=StubSource([10, 11, 12]),
        fallback_source=None,
        limit=2,
        config=config,
    )
    assert result.diagnostics["diversity_applied"] is True
    # The Romance title is promoted over the second Action title.
    assert result.anime_ids == [10, 12]


def test_quality_prior_can_reorder_close_candidates():
    class Quality:
        def quality_score(self, anime_id: int) -> float | None:
            return 1.0 if anime_id == 12 else 0.0

    config = FastPathConfig(quality_weight=0.9)
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11, 12]),
        als_source=StubSource([10, 11, 12]),
        fallback_source=None,
        quality_lookup=Quality(),
        limit=1,
        config=config,
    )
    assert result.anime_ids == [12]


def test_trace_carries_every_observability_field():
    result = recommend_fast(
        list(range(30)),
        catalog_by_id=catalog_map([10, 11]),
        als_source=StubSource([10, 11]),
        fallback_source=None,
        tail_source=StubSource([11, 20]),
        limit=2,
    )
    trace = result.trace()
    for key in (
        "collaborative_route",
        "known_positive_count",
        "reason",
        "policy_version",
        "path",
        "fast_path_version",
        "candidate_pool_size",
        "candidate_sources",
        "tail_source_used",
        "retrieval_config_version",
        "diversity_applied",
        "hard_filter_applied",
        "stage_latency_ms",
        "total_latency_ms",
    ):
        assert key in trace, f"missing trace field: {key}"
    assert set(trace["stage_latency_ms"]) == {"routing", "retrieval", "filtering", "ranking", "reranking"}


def test_fast_path_config_is_validated():
    with pytest.raises(ValueError):
        FastPathConfig(diversity_strength=1.5)
    with pytest.raises(ValueError):
        FastPathConfig(diversity_window=0)
    with pytest.raises(ValueError):
        FastPathConfig(quality_weight=-0.1)


# ---------------------------------------------------------------- path policy


def test_unconstrained_request_takes_the_fast_path():
    decision = choose_recommendation_path({"intent": "recommend", "top_k": 10})
    assert decision.path is RecommendationPath.FAST
    assert decision.signals == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_voice_actors", ["Matsuoka, Yoshitsugu"]),
        ("required_studios", ["Madhouse"]),
        ("include_genres", ["Isekai"]),
        ("min_year", 2015),
        ("max_episodes", 12),
        ("formats", ["TV"]),
        ("reference_titles", ["Death Note"]),
        ("free_text_preferences", "something gentle"),
    ],
)
def test_constrained_requests_take_the_rich_path(field, value):
    decision = choose_recommendation_path({"intent": "recommend", field: value})
    assert decision.path is RecommendationPath.CONSTRAINT_RICH
    assert decision.signals


def test_constraint_shaped_intents_take_the_rich_path():
    for intent in ("rank_catalog", "search", "details"):
        assert choose_recommendation_path({"intent": intent}).path is RecommendationPath.CONSTRAINT_RICH


def test_similarity_and_text_signals_are_configurable():
    policy = PathPolicy(reference_titles_are_constraints=False, free_text_is_constraint=False)
    decision = choose_recommendation_path(
        {"intent": "recommend", "reference_titles": ["X"], "free_text_preferences": "y"}, policy=policy
    )
    assert decision.path is RecommendationPath.FAST


def test_required_entity_fields_are_reported_for_constraint_preservation():
    intent = {
        "intent": "recommend",
        "required_voice_actors": ["Matsuoka, Yoshitsugu"],
        "preferred_studios": ["Madhouse"],
    }
    assert required_entity_fields(intent) == ("required_voice_actors",)


def test_path_descriptions_do_not_call_either_path_better():
    text = " ".join(f"{row['serves']} {row['ranking']}" for row in describe_paths()).lower()
    assert "better" not in text
    assert len(describe_paths()) == 2
