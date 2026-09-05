"""Two defects that made the product feel broken, pinned so they stay fixed.

Both were invisible to the existing suite because both need realistic scale to
show up: a catalog of four synthetic titles cannot demonstrate that a browse
request surfaces the most obscure of 3,905 matches, and a fake LLM client always
returns text the validator happens to accept.
"""

from __future__ import annotations

import pytest

from backend.anime_agent.agent import MINIMUM_GROUNDED_TITLES, AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender


def title(anime_id: int, name: str, *, members: int, score: float, genres: list[str]) -> dict:
    return {
        "id": anime_id,
        "title": name,
        "genres": genres,
        "members": members,
        "score": score,
        "popularity": max(1, 20000 - members // 100),
        "type": "TV",
        "episodes": 12,
        "start_year": 2015,
        "synopsis": f"{name} is a story.",
        "studios": [],
        "metadata_tokens": [f"genre_{g}" for g in genres],
    }


@pytest.fixture
def lopsided_catalog() -> list[dict]:
    """One famous Fantasy title among many obscure ones, as the real catalog is."""
    famous = [title(1, "Famous Fantasy", members=3_000_000, score=9.0, genres=["Fantasy"])]
    obscure = [
        title(anime_id, f"Obscure Fantasy {anime_id}", members=200 + anime_id, score=5.5, genres=["Fantasy"])
        for anime_id in range(2, 40)
    ]
    return famous + obscure


# ---------------------------------------------- browsing is not personalizing


def test_a_genre_filter_alone_ranks_by_quality_not_by_genre_match(lopsided_catalog):
    """After filtering to Fantasy, "matches Fantasy" is constant and cannot rank.

    Treating the filter as a taste signal diluted the quality channel from
    weight 1.0 to 0.35 and surfaced the catalog's most obscure entries.
    """
    recommender = AnimeRecommender(lopsided_catalog)
    results = recommender.recommend(include_genres=["Fantasy"], top_k=5)
    titles = [item["title"] for item in results]
    assert titles[0] == "Famous Fantasy", f"a browse request surfaced obscure titles first: {titles}"


def test_a_filter_only_request_reports_the_quality_fallback_mode(lopsided_catalog):
    recommender = AnimeRecommender(lopsided_catalog)
    results = recommender.recommend(include_genres=["Fantasy"], top_k=3)
    assert results[0]["score_breakdown"]["recommendation_mode"] == "quality_fallback"


def test_a_real_taste_signal_still_personalizes(lopsided_catalog):
    """The fix must not disable personalization -- only filter-derived pseudo-taste."""
    recommender = AnimeRecommender(lopsided_catalog)
    results = recommender.recommend(
        include_genres=["Fantasy"],
        free_text_preferences="obscure quiet stories",
        top_k=3,
    )
    assert results[0]["score_breakdown"]["recommendation_mode"] == "hybrid"


def test_liked_titles_still_personalize(lopsided_catalog):
    recommender = AnimeRecommender(lopsided_catalog)
    results = recommender.recommend(reference_titles=["Obscure Fantasy 5"], top_k=3)
    assert results[0]["score_breakdown"]["recommendation_mode"] == "hybrid"


# ------------------------------------------- grounded does not mean exhaustive


@pytest.fixture
def agent(lopsided_catalog) -> AnimeAgent:
    return AnimeAgent(AnimeRecommender(lopsided_catalog))


def test_an_answer_covering_the_top_few_results_is_grounded(agent):
    """Requiring all ten titles rejected good answers and cost a second provider.

    The result list is rendered from tool output; the prose explains it.
    """
    titles = [f"Title {index}" for index in range(1, 11)]
    answer = "\n".join(
        [
            "Here are a few that fit:",
            "- Title 1 (TV): a strong match.",
            "- Title 2 (TV): also fits.",
            "- Title 3 (TV): worth a look.",
        ]
    )
    assert agent._valid_recommendation_answer(answer, titles, [], 10)


def test_an_answer_naming_too_few_titles_is_still_rejected(agent):
    titles = [f"Title {index}" for index in range(1, 11)]
    answer = "- Title 1 (TV): a strong match."
    assert not agent._valid_recommendation_answer(answer, titles, [], 10)
    assert MINIMUM_GROUNDED_TITLES == 3


def test_an_invented_title_is_still_rejected(agent):
    """Relaxing the count must not relax grounding itself."""
    titles = [f"Title {index}" for index in range(1, 11)]
    answer = "\n".join(
        [
            "- Title 1 (TV): fits.",
            "- Title 2 (TV): fits.",
            "- Totally Made Up Show (TV): does not exist.",
        ]
    )
    assert not agent._valid_recommendation_answer(answer, titles, [], 10)


def test_an_excluded_title_still_rejects_outright(agent):
    titles = [f"Title {index}" for index in range(1, 11)]
    answer = "\n".join(
        [
            "- Title 1 (TV): fits.",
            "- Title 2 (TV): fits.",
            "- Title 3 (TV): fits.",
        ]
    )
    assert not agent._valid_recommendation_answer(answer, titles, ["Title 2"], 10)


def test_a_short_result_list_only_needs_what_exists(agent):
    """Two results cannot yield three catalog lines."""
    titles = ["Title 1", "Title 2"]
    answer = "- Title 1 (TV): fits.\n- Title 2 (TV): fits."
    assert agent._valid_recommendation_answer(answer, titles, [], 2)
