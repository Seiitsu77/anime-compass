"""Feedback has to reach the ranker, and only the feedback the user gave.

These pin the V2.1 audit findings on the personalized fast path:

* The 51st like and every later one never reached the ALS fold-in.
* Title-based negative feedback was honoured by the Hybrid and ignored here.
* Fewer than six 18-title batches of "already shown" fitted in one request.

The shared fixture catalog has twelve titles, which cannot express any of that,
so this module builds a larger synthetic one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas import (
    MAX_EXCLUDED_IDS,
    MAX_LIKED_IDS,
    RECOMMEND_REQUEST_LIMITS,
    RecommendRequest,
)
from app.core.config import Settings
from app.main import create_app
from app.repositories.session_repository import SQLiteSessionRepository
from app.services.recommendation_service import RecommendationService
from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.fast_path import FastPathConfig
from backend.anime_agent.recommender import AnimeRecommender
from tests.conftest import anime, wait_until_ready
from tests.test_api import MockProvider
from tests.test_fast_path_integration import build_artifact

CATALOG_SIZE = 320


@pytest.fixture
def wide_catalog() -> list[dict[str, Any]]:
    """One franchise per title, so series collapsing never hides a result."""
    return [
        anime(
            anime_id,
            f"Saga{anime_id} Chronicle",
            genres=["Action"] if anime_id % 2 else ["Drama"],
            synopsis=f"Synthetic story number {anime_id}.",
        )
        for anime_id in range(1, CATALOG_SIZE + 1)
    ]


def build_service(tmp_path: Path, catalog: list[dict[str, Any]], *, debug: bool = False) -> RecommendationService:
    ids = sorted(int(item["id"]) for item in catalog)
    index = ALSCollaborativeIndex.load(build_artifact(tmp_path / "als.npz", ids), catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}", retention_days=30)
    return RecommendationService(
        AnimeRecommender(catalog),
        sessions,
        als_index=index,
        fast_path_config=FastPathConfig(),
        debug_diagnostics=debug,
    )


def result_ids(response: dict[str, Any]) -> list[int]:
    return [int(item["id"]) for item in response["results"]]


# ------------------------------------------------------------ request limits


def test_a_profile_larger_than_fifty_likes_is_accepted():
    request = RecommendRequest(liked_ids=list(range(1, 301)))
    assert len(request.liked_ids) == 300
    RecommendRequest(liked_ids=list(range(1, MAX_LIKED_IDS + 1)))
    with pytest.raises(ValidationError):
        RecommendRequest(liked_ids=list(range(1, MAX_LIKED_IDS + 2)))


def test_more_than_one_hundred_shown_titles_can_be_excluded():
    RecommendRequest(excluded_ids=list(range(1, MAX_EXCLUDED_IDS + 1)))
    with pytest.raises(ValidationError):
        RecommendRequest(excluded_ids=list(range(1, MAX_EXCLUDED_IDS + 2)))


def test_health_advertises_the_request_limits(tmp_path: Path, catalog: list[dict[str, Any]]):
    """The browser truncates against these rather than a guess."""
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    app = create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")})
    with TestClient(app) as client:
        wait_until_ready(client)
        ranking = client.get("/api/health").json()["ranking"]
    assert ranking["request_limits"] == RECOMMEND_REQUEST_LIMITS
    assert ranking["request_limits"]["liked_ids"] == MAX_LIKED_IDS


# ------------------------------------------------------------- like -> model


@pytest.mark.asyncio
async def test_a_like_changes_the_user_vector_and_the_ranking(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog, debug=True)
    before = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    # An odd title: the fixture factors separate items by parity, so this is a
    # like that genuinely moves the profile rather than restating it.
    after = await service.recommend(RecommendRequest(liked_ids=[2, 4, 7], limit=18))

    assert before["diagnostics"]["recommendation_path"] == "fast"
    assert after["diagnostics"]["known_positive_count"] == 3
    assert (
        before["diagnostics"]["debug"]["als_user_vector_sha256_16"]
        != after["diagnostics"]["debug"]["als_user_vector_sha256_16"]
    )
    assert (
        before["diagnostics"]["debug"]["als_retrieved_top_ids"]
        != after["diagnostics"]["debug"]["als_retrieved_top_ids"]
    )
    assert result_ids(before) != result_ids(after)
    assert 7 not in result_ids(after)


@pytest.mark.asyncio
async def test_the_same_profile_yields_the_same_vector_and_ranking(tmp_path, wide_catalog):
    """What a page reload relies on: identical likes, identical recommendations."""
    service = build_service(tmp_path, wide_catalog, debug=True)
    first = await service.recommend(RecommendRequest(liked_ids=[2, 4, 7], limit=18))
    second = await service.recommend(RecommendRequest(liked_ids=[2, 4, 7], limit=18))
    assert first["diagnostics"]["debug"] == second["diagnostics"]["debug"]
    assert result_ids(first) == result_ids(second)


@pytest.mark.asyncio
async def test_every_like_past_the_fiftieth_reaches_the_fold_in(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog, debug=True)
    sixty = list(range(1, 61))
    base = await service.recommend(RecommendRequest(liked_ids=sixty, limit=18))
    one_more = await service.recommend(RecommendRequest(liked_ids=[*sixty, 201], limit=18))

    assert base["diagnostics"]["recommendation_path"] == "fast"
    assert base["diagnostics"]["profile"]["positives_used"] == 60
    assert one_more["diagnostics"]["profile"]["positives_used"] == 61
    assert one_more["diagnostics"]["profile"]["positives_known_to_model"] == 61
    assert (
        base["diagnostics"]["debug"]["als_user_vector_sha256_16"]
        != one_more["diagnostics"]["debug"]["als_user_vector_sha256_16"]
    )


@pytest.mark.asyncio
async def test_liked_titles_never_come_back_as_recommendations(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    liked = list(range(1, 121))
    response = await service.recommend(RecommendRequest(liked_ids=liked, limit=50))
    assert response["results"]
    assert not set(result_ids(response)) & set(liked)


# --------------------------------------------------------- want to watch


@pytest.mark.asyncio
async def test_want_to_watch_is_an_exclusion_and_not_a_preference(tmp_path, wide_catalog):
    """Bookmarking removes the title; it must not teach the model anything."""
    service = build_service(tmp_path, wide_catalog, debug=True)
    plain = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    bookmarked = result_ids(plain)[0]
    saved = await service.recommend(RecommendRequest(liked_ids=[2, 4], excluded_ids=[bookmarked], limit=18))

    assert bookmarked not in result_ids(saved)
    assert saved["diagnostics"]["known_positive_count"] == plain["diagnostics"]["known_positive_count"] == 2
    assert (
        saved["diagnostics"]["debug"]["als_user_vector_sha256_16"]
        == plain["diagnostics"]["debug"]["als_user_vector_sha256_16"]
    )
    # Everything else keeps its relative order: the bookmark only left a gap.
    remaining = [anime_id for anime_id in result_ids(plain) if anime_id != bookmarked]
    assert result_ids(saved)[: len(remaining)] == remaining


# ------------------------------------------------------- not interested


@pytest.mark.asyncio
async def test_title_based_negative_feedback_is_honoured_on_the_fast_path(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    plain = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    first, second, third = result_ids(plain)[:3]

    response = await service.recommend(
        RecommendRequest(
            liked_ids=[2, 4],
            excluded_titles=[f"saga{first} chronicle"],
            seen_titles=[f"Saga{second} Chronicle"],
            session_profile={"disliked_titles": [f"Saga{third} Chronicle"]},
            limit=18,
        )
    )
    assert response["diagnostics"]["recommendation_path"] == "fast"
    assert not {first, second, third} & set(result_ids(response))
    assert response["diagnostics"]["profile"]["excluded_titles_resolved"] == 3
    assert response["diagnostics"]["profile"]["excluded_titles_unresolved"] == 0


@pytest.mark.asyncio
async def test_a_stored_session_dislike_is_honoured_on_the_fast_path(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    plain = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    rejected = result_ids(plain)[0]
    service.sessions.update("browser-session", {"disliked_titles": [f"Saga{rejected} Chronicle"]})

    response = await service.recommend(RecommendRequest(session_id="browser-session", liked_ids=[2, 4], limit=18))
    assert response["diagnostics"]["recommendation_path"] == "fast"
    assert rejected not in result_ids(response)


@pytest.mark.asyncio
async def test_a_near_miss_title_does_not_hide_a_different_anime(tmp_path, wide_catalog):
    """Exclusions match exactly; a fuzzy hit would hide something never rejected."""
    service = build_service(tmp_path, wide_catalog)
    plain = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    response = await service.recommend(RecommendRequest(liked_ids=[2, 4], excluded_titles=["Saga Chronicle"], limit=18))
    assert result_ids(response) == result_ids(plain)
    assert response["diagnostics"]["profile"]["excluded_titles_unresolved"] == 1


# ------------------------------------------------------------- pagination


@pytest.mark.asyncio
async def test_paging_by_exclusion_never_repeats_past_one_hundred_shown(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    liked = [2, 4, 6]
    shown: list[int] = []
    for _batch in range(9):
        response = await service.recommend(RecommendRequest(liked_ids=liked, excluded_ids=shown, limit=18))
        page = result_ids(response)
        assert len(page) == 18
        assert not set(page) & set(shown), "a title already shown was recommended again"
        assert not set(page) & set(liked)
        shown.extend(page)
    assert len(shown) == len(set(shown)) == 162


# ------------------------------------------------------------ diagnostics


@pytest.mark.asyncio
async def test_the_debug_block_is_absent_unless_enabled(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    response = await service.recommend(RecommendRequest(liked_ids=[2, 4], limit=18))
    assert "debug" not in response["diagnostics"]
    # The always-on block carries counts, never identifiers.
    assert all(isinstance(value, (int, bool)) for value in response["diagnostics"]["profile"].values())


def test_debug_diagnostics_default_to_off():
    assert Settings(_env_file=None).recommendation_debug_diagnostics is False


# ------------------------------------------------------- adjacent regressions


def test_an_unranked_title_does_not_lead_the_popularity_or_rank_sort(catalog: list[dict[str, Any]]):
    """The source writes 0 for "no position"; ascending order put it first."""
    unranked = anime(99, "Unranked Music Video", genres=["Music"], synopsis="A one-member entry.")
    unranked.update({"popularity": 0, "rank": 0, "members": 1, "score": None})
    recommender = AnimeRecommender([unranked, *catalog])

    for field in ("popularity", "rank"):
        page, total = recommender.search_page("", limit=50, sort_by=field, semantic=False)
        assert total == len(catalog) + 1
        assert page[0]["title"] != "Unranked Music Video"
        assert page[-1]["title"] == "Unranked Music Video", f"unranked titles sort last by {field}"
        assert [item[field] for item in page[:-1]] == sorted(item[field] for item in page[:-1])

        ranked, diagnostics = recommender.rank_catalog(sort_by=field, limit=50)
        assert ranked[0]["title"] != "Unranked Music Video"
        assert ranked[-1]["title"] == "Unranked Music Video"
        assert diagnostics["missing_sort_values"] == 1


def test_a_browser_profile_sync_keeps_the_agent_conversation_context(tmp_path: Path, catalog: list[dict[str, Any]]):
    """The React client re-syncs the profile before each agent message."""
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    app = create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")})
    with TestClient(app) as client:
        wait_until_ready(client)
        client.post(
            "/api/session/chat/preferences",
            json={"last_recommendations": ["Death Parade", "Ghost Hunt"], "last_recommendation_intent": {"top_k": 2}},
        )
        # Exactly what frontend-react/src/lib/api.js sends: a full replace with no context fields.
        synced = client.post(
            "/api/session/preferences",
            json={
                "session_id": "chat",
                "replace": True,
                "liked_titles": ["Death Note"],
                "disliked_titles": [],
                "seen_titles": [],
                "excluded_titles": [],
                "preferred_genres": [],
                "excluded_genres": [],
                "previous_reference_titles": [],
                "temporary_ratings": {},
            },
        )
        # A later, smaller update must not clear it either.
        client.post("/api/session/chat/preferences", json={"disliked_titles": ["Quiet Romance"]})
        profile = client.get("/api/session/chat").json()["profile"]

    assert synced.status_code == 200
    assert profile["liked_titles"] == ["Death Note"]
    assert profile["disliked_titles"] == ["Quiet Romance"]
    assert profile["last_recommendations"] == ["Death Parade", "Ghost Hunt"]
    assert profile["last_recommendation_intent"] == {"top_k": 2}


def test_an_unknown_preference_field_is_rejected_which_is_why_the_client_whitelists(
    tmp_path: Path, catalog: list[dict[str, Any]]
):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    app = create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")})
    with TestClient(app) as client:
        wait_until_ready(client)
        response = client.post(
            "/api/session/preferences",
            json={"session_id": "s", "replace": True, "liked_titles": ["Death Note"], "custom_flag": "from-a-backup"},
        )
    assert response.status_code == 422
