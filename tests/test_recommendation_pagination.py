"""Paged recommendations: one deterministic sequence, served a slice at a time.

What these pin, in the order the product depends on them:

* a sequence never repeats a title, however it is paged;
* the first window is ordered exactly as the unpaged fast path orders it;
* depth comes from further ALS windows, each reranked, never from shuffling;
* a page can be served by an instance that has never seen the sequence;
* a cursor from a different profile or deployment is refused, not guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import RECOMMENDATION_BATCH_SIZE, RecommendPageRequest, RecommendRequest
from app.core.config import Settings
from app.core.errors import AppError
from app.main import create_app
from app.repositories.session_repository import SQLiteSessionRepository
from app.services import recommendation_service
from app.services.recommendation_sequence import decode_cursor, encode_cursor
from app.services.recommendation_service import RecommendationService
from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.fast_path import FastPathConfig
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.retrieval import RetrievalConfig
from tests.conftest import anime, wait_until_ready
from tests.test_api import MockProvider
from tests.test_fast_path_integration import build_artifact

CATALOG_SIZE = 320
WINDOW = 40
LIKED = [2, 4, 6]


@pytest.fixture
def wide_catalog() -> list[dict[str, Any]]:
    """One franchise per title, so the franchise filter never hides a result."""
    return [
        anime(
            anime_id,
            f"Saga{anime_id} Chronicle",
            genres=["Action"] if anime_id % 2 else ["Drama"],
            synopsis=f"Synthetic story number {anime_id}.",
        )
        for anime_id in range(1, CATALOG_SIZE + 1)
    ]


@pytest.fixture
def franchise_catalog() -> list[dict[str, Any]]:
    """One hundred franchises of three entries each."""
    titles = []
    for franchise in range(1, 101):
        for entry, suffix in enumerate(("", " Season 2", " Movie")):
            titles.append(
                anime(
                    franchise * 10 + entry,
                    f"Saga{franchise} Chronicle{suffix}",
                    genres=["Action"],
                    synopsis=f"Franchise {franchise}, entry {entry}.",
                )
            )
    return titles


class ReversingReranker:
    """Stands in for LambdaMART: a visibly different order within a candidate set."""

    def __init__(self, index_by_id: dict[int, int]):
        self.index_by_id = dict(index_by_id)
        self.calls = 0

    def rerank(self, profile_rows, candidate_rows, als_scores):
        self.calls += 1
        return list(reversed(candidate_rows))

    def model_info(self) -> dict[str, Any]:
        return {"model_sha256": "reversing", "artifact_sha256": "reversing"}


def build_service(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    *,
    name: str = "a",
    windows: int = 4,
    reranker: bool = False,
    debug: bool = False,
) -> RecommendationService:
    ids = sorted(int(item["id"]) for item in catalog)
    artifact = tmp_path / "als.npz"
    if not artifact.exists():
        build_artifact(artifact, ids)
    index = ALSCollaborativeIndex.load(artifact, catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / f'{name}.db').as_posix()}", retention_days=30)
    return RecommendationService(
        AnimeRecommender(catalog),
        sessions,
        als_index=index,
        fast_path_config=FastPathConfig(
            retrieval=RetrievalConfig(als_top_n=WINDOW),
            reranker=ReversingReranker(index.index_by_id) if reranker else None,
        ),
        sequence_windows=windows,
        debug_diagnostics=debug,
    )


def ids_of(response: dict[str, Any]) -> list[int]:
    return [int(item["id"]) for item in response["results"]]


async def read_all(
    service: RecommendationService, limit: int = RECOMMENDATION_BATCH_SIZE, **body: Any
) -> list[list[int]]:
    """Every page of a sequence, until the service says there is no more."""
    pages: list[list[int]] = []
    cursor = None
    for _guard in range(100):
        response = await service.page(RecommendPageRequest(cursor=cursor, limit=limit, **body))
        pages.append(ids_of(response))
        cursor = response["sequence"]["next_cursor"]
        if not response["sequence"]["has_more"]:
            assert cursor is None
            assert response["sequence"]["exhausted"] is True
            return pages
    raise AssertionError("the sequence never ended")


# ------------------------------------------------------------ no repeats


@pytest.mark.asyncio
async def test_batches_of_eighteen_never_repeat_and_end_cleanly(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    pages = await read_all(service, liked_ids=LIKED)
    shown = [anime_id for page in pages for anime_id in page]

    assert all(len(page) == RECOMMENDATION_BATCH_SIZE for page in pages[:-1])
    assert len(shown) == len(set(shown)), "a title was recommended twice in one sequence"
    assert not set(shown) & set(LIKED)
    # Depth is four windows of forty candidates: bounded, and known in advance.
    assert len(shown) == 4 * WINDOW
    assert len(pages) == 9


@pytest.mark.asyncio
async def test_the_default_page_size_is_eighteen(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    response = await service.page(RecommendPageRequest(liked_ids=LIKED))
    assert RECOMMENDATION_BATCH_SIZE == 18
    assert response["sequence"]["returned"] == 18
    assert response["sequence"]["offset"] == 0
    assert response["sequence"]["has_more"] is True


@pytest.mark.asyncio
async def test_replace_and_next_batch_read_the_same_sequence(tmp_path, wide_catalog):
    """A one-title replacement is just the next title; it cannot fork the order."""
    service = build_service(tmp_path, wide_catalog)
    whole = [anime_id for page in await read_all(service, limit=50, liked_ids=LIKED) for anime_id in page]

    first = await service.page(RecommendPageRequest(liked_ids=LIKED, limit=18))
    one = await service.page(RecommendPageRequest(liked_ids=LIKED, limit=1, cursor=first["sequence"]["next_cursor"]))
    another = await service.page(RecommendPageRequest(liked_ids=LIKED, limit=1, cursor=one["sequence"]["next_cursor"]))
    batch = await service.page(
        RecommendPageRequest(liked_ids=LIKED, limit=18, cursor=another["sequence"]["next_cursor"])
    )

    assert ids_of(first) + ids_of(one) + ids_of(another) + ids_of(batch) == whole[:38]


# ------------------------------------------------------- ranking preserved


@pytest.mark.asyncio
async def test_the_first_window_is_ordered_exactly_like_the_unpaged_fast_path(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog, reranker=True)
    # top_k 10 with one_per_series fetches 40 candidates: exactly one window.
    unpaged = await service.recommend(RecommendRequest(liked_ids=LIKED, limit=10, one_per_series=True))
    paged = await service.page(RecommendPageRequest(liked_ids=LIKED, limit=10))

    assert unpaged["diagnostics"]["recommendation_path"] == "fast"
    assert unpaged["diagnostics"]["learned_reranker_applied"] is True
    assert ids_of(paged) == ids_of(unpaged)
    assert paged["diagnostics"]["learned_reranker_applied"] is True


@pytest.mark.asyncio
async def test_depth_is_further_als_windows_each_reranked_and_never_shuffled(tmp_path, wide_catalog):
    plain = build_service(tmp_path, wide_catalog, name="plain")
    learned = build_service(tmp_path, wide_catalog, name="learned", reranker=True)
    als_order = [anime_id for page in await read_all(plain, limit=50, liked_ids=LIKED) for anime_id in page]
    reranked = [anime_id for page in await read_all(learned, limit=50, liked_ids=LIKED) for anime_id in page]

    assert len(als_order) == len(reranked) == 4 * WINDOW
    for window in range(4):
        expected = als_order[window * WINDOW : (window + 1) * WINDOW]
        actual = reranked[window * WINDOW : (window + 1) * WINDOW]
        # The reranker reorders inside a window and cannot move a title across one.
        assert actual == list(reversed(expected)), f"window {window} was not reranked in place"
    assert learned.fast_path_config.reranker.calls == 4

    # Deterministic: the same request ranks to the same sequence every time.
    again = build_service(tmp_path, wide_catalog, name="again", reranker=True)
    assert [a for page in await read_all(again, limit=50, liked_ids=LIKED) for a in page] == reranked


# --------------------------------------------- restarts and stale cursors


@pytest.mark.asyncio
async def test_a_page_can_be_served_by_an_instance_that_never_saw_the_sequence(tmp_path, wide_catalog):
    """What a Cloud Run restart or a second instance looks like to a client."""
    before = build_service(tmp_path, wide_catalog, name="before")
    expected = await read_all(before, liked_ids=LIKED, excluded_ids=[30, 31])
    first = await before.page(RecommendPageRequest(liked_ids=LIKED, excluded_ids=[30, 31]))
    cursor = first["sequence"]["next_cursor"]

    after = build_service(tmp_path, wide_catalog, name="after")
    resumed = await after.page(RecommendPageRequest(liked_ids=LIKED, excluded_ids=[30, 31], cursor=cursor))
    assert resumed["sequence"]["cache"] == "miss"
    assert resumed["sequence"]["offset"] == 18
    assert ids_of(resumed) == expected[1]

    # Deep in the sequence too: the new instance rebuilds every window it needs.
    deep_cursor = encode_cursor(decode_cursor(cursor)[0], 126)
    deep = await build_service(tmp_path, wide_catalog, name="deep").page(
        RecommendPageRequest(liked_ids=LIKED, excluded_ids=[30, 31], cursor=deep_cursor)
    )
    assert ids_of(deep) == expected[7]
    assert deep["sequence"]["windows_computed"] == 4


@pytest.mark.asyncio
async def test_a_cached_page_does_no_ranking(tmp_path, wide_catalog, monkeypatch):
    service = build_service(tmp_path, wide_catalog)
    calls = 0
    original = recommendation_service.recommend_fast

    def counting(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(recommendation_service, "recommend_fast", counting)
    first = await service.page(RecommendPageRequest(liked_ids=LIKED))
    assert (first["sequence"]["cache"], calls) == ("miss", 1)
    repeat = await service.page(RecommendPageRequest(liked_ids=LIKED))
    second = await service.page(RecommendPageRequest(liked_ids=LIKED, cursor=first["sequence"]["next_cursor"]))
    assert repeat["sequence"]["cache"] == second["sequence"]["cache"] == "hit"
    assert calls == 1, "pages inside an already-ranked window must not rank again"
    assert ids_of(repeat) == ids_of(first)
    third = await service.page(RecommendPageRequest(liked_ids=LIKED, cursor=second["sequence"]["next_cursor"]))
    assert (third["sequence"]["cache"], calls) == ("miss", 2), "crossing into the next window ranks exactly one more"


@pytest.mark.asyncio
async def test_a_cursor_from_another_profile_is_refused(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    first = await service.page(RecommendPageRequest(liked_ids=LIKED))
    cursor = first["sequence"]["next_cursor"]

    for changed in (
        {"liked_ids": [*LIKED, 8]},  # a new Like
        {"liked_ids": LIKED, "excluded_ids": [50]},  # a new exclusion
        {"liked_ids": LIKED, "one_per_series": False},  # a different policy
    ):
        with pytest.raises(AppError) as refused:
            await service.page(RecommendPageRequest(cursor=cursor, **changed))
        assert refused.value.code == "sequence_expired"
        assert refused.value.status_code == 409


@pytest.mark.asyncio
async def test_a_cursor_from_another_deployment_is_refused(tmp_path, wide_catalog):
    """A new model or configuration ranks differently; an old offset is meaningless."""
    old = build_service(tmp_path, wide_catalog, name="old", windows=4)
    cursor = (await old.page(RecommendPageRequest(liked_ids=LIKED)))["sequence"]["next_cursor"]

    for new in (
        build_service(tmp_path, wide_catalog, name="deeper", windows=6),
        build_service(tmp_path, wide_catalog, name="reranked", reranker=True),
    ):
        with pytest.raises(AppError) as refused:
            await new.page(RecommendPageRequest(liked_ids=LIKED, cursor=cursor))
        assert refused.value.code == "sequence_expired"


@pytest.mark.asyncio
async def test_a_malformed_cursor_is_a_client_error(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    for cursor in ("not-a-cursor", "e30", encode_cursor("f" * 32, 10**9), encode_cursor("short", 0)):
        with pytest.raises(AppError) as refused:
            await service.page(RecommendPageRequest(liked_ids=LIKED, cursor=cursor))
        assert refused.value.code == "invalid_cursor"
        assert refused.value.status_code == 422


# ------------------------------------------------- filters across windows


@pytest.mark.asyncio
async def test_exclusions_hold_at_every_depth_and_do_not_shorten_the_sequence(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    plain = [a for page in await read_all(service, limit=50, liked_ids=LIKED) for a in page]
    excluded = plain[5:25] + plain[70:90]
    filtered = [a for page in await read_all(service, limit=50, liked_ids=LIKED, excluded_ids=excluded) for a in page]

    assert not set(filtered) & set(excluded)
    assert not set(filtered) & set(LIKED)
    # Exclusions are applied at retrieval, so each window refills to full size.
    assert len(filtered) == len(plain) == 4 * WINDOW


@pytest.mark.asyncio
async def test_one_per_series_holds_across_windows(tmp_path, franchise_catalog):
    service = build_service(tmp_path, franchise_catalog, windows=6)
    liked = [10, 20]  # the first entries of franchises 1 and 2
    shown = [a for page in await read_all(service, limit=50, liked_ids=liked) for a in page]
    franchises = [anime_id // 10 for anime_id in shown]

    assert len(franchises) == len(set(franchises)), "a franchise appeared twice in one sequence"
    assert not {1, 2} & set(franchises), "the current policy hides franchises the user already Liked"
    assert len(shown) >= 60

    everything = [a for page in await read_all(service, limit=50, liked_ids=liked, one_per_series=False) for a in page]
    assert len(everything) == 6 * WINDOW
    assert {1, 2} & {anime_id // 10 for anime_id in everything}


# ----------------------------------------------------------- cold start


@pytest.mark.asyncio
async def test_a_profile_with_no_likes_gets_a_bounded_starter_sequence(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    pages = await read_all(service)
    shown = [anime_id for page in pages for anime_id in page]
    first = await service.page(RecommendPageRequest())

    assert first["sequence"]["source"] == "hybrid"
    assert len(shown) == len(set(shown)) == service.starter_depth == 54
    assert [len(page) for page in pages] == [18, 18, 18]

    # The starter list is resumable by a fresh instance as well.
    resumed = await build_service(tmp_path, wide_catalog, name="fresh").page(
        RecommendPageRequest(cursor=first["sequence"]["next_cursor"])
    )
    assert ids_of(resumed) == pages[1]


@pytest.mark.asyncio
async def test_the_first_starter_page_ranks_one_batch_not_the_whole_list(tmp_path, wide_catalog, monkeypatch):
    """A new user's first screen must not wait for all 54 titles to be ranked."""
    service = build_service(tmp_path, wide_catalog)
    limits: list[int] = []
    original = service.recommender.recommend

    def counting(**kwargs: Any):
        limits.append(kwargs["limit"])
        return original(**kwargs)

    monkeypatch.setattr(service.recommender, "recommend", counting)
    # The client asks for two batches; a starter sequence answers with the one it has.
    first = await service.page(RecommendPageRequest(limit=36))
    assert limits == [18]
    assert first["sequence"]["returned"] == 18
    assert first["sequence"]["has_more"] is True
    second = await service.page(RecommendPageRequest(limit=36, cursor=first["sequence"]["next_cursor"]))
    third = await service.page(RecommendPageRequest(limit=36, cursor=second["sequence"]["next_cursor"]))
    assert limits == [18, 18, 18]
    assert third["sequence"]["has_more"] is False
    shown = ids_of(first) + ids_of(second) + ids_of(third)
    assert len(shown) == len(set(shown)) == 54

    # Same list whatever the page size, and a later chunk never reorders an earlier one.
    fresh = build_service(tmp_path, wide_catalog, name="chunks")
    assert [a for page in await read_all(fresh, limit=7) for a in page] == shown


@pytest.mark.asyncio
async def test_likes_the_model_does_not_know_fall_back_to_a_starter_sequence(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog)
    response = await service.page(RecommendPageRequest(liked_ids=[999_999]))
    assert response["sequence"]["source"] == "hybrid"
    assert response["results"]


# ----------------------------------------------------------- over HTTP


def app_client(tmp_path: Path, catalog: list[dict[str, Any]]) -> TestClient:
    artifact = build_artifact(tmp_path / "als-app.npz", sorted(int(item["id"]) for item in catalog))
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
        als_artifact_path=artifact,
        retrieval_als_top_n=WINDOW,
        reranker_enabled=False,
        recommendation_sequence_windows=3,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")}))


def test_the_endpoint_pages_without_any_shown_ids_in_the_request(tmp_path: Path, wide_catalog):
    body = {"session_id": "browser", "liked_ids": LIKED, "excluded_ids": [30]}
    shown: list[int] = []
    with app_client(tmp_path, wide_catalog) as client:
        wait_until_ready(client)
        pagination = client.get("/api/health").json()["ranking"]["pagination"]
        cursor = None
        for _page in range(20):
            response = client.post("/api/recommend/page", json={**body, "cursor": cursor})
            assert response.status_code == 200
            payload = response.json()
            shown.extend(item["id"] for item in payload["results"])
            cursor = payload["sequence"]["next_cursor"]
            if not payload["sequence"]["has_more"]:
                break
        refused = client.post("/api/recommend/page", json={"liked_ids": [2, 4], "cursor": encode_cursor("0" * 32, 18)})
        unknown_field = client.post("/api/recommend/page", json={**body, "shown_ids": [1, 2, 3]})

    assert pagination == {
        "endpoint": "/api/recommend/page",
        "batch_size": 18,
        "window_size": WINDOW,
        "max_windows": 3,
        "starter_depth": 54,
        "starter_chunk": 18,
    }
    assert len(shown) == len(set(shown)) == 3 * WINDOW
    assert 30 not in shown and not set(shown) & set(LIKED)
    assert payload["sequence"]["exhausted"] is True
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "sequence_expired"
    # The request has no field for already-shown titles: position is the cursor.
    assert unknown_field.status_code == 422


def test_the_legacy_unpaged_endpoint_still_works(tmp_path: Path, wide_catalog):
    with app_client(tmp_path, wide_catalog) as client:
        wait_until_ready(client)
        legacy = client.post("/api/recommend", json={"liked_ids": LIKED, "limit": 10, "one_per_series": True})
        paged = client.post("/api/recommend/page", json={"liked_ids": LIKED, "limit": 10})
    assert legacy.status_code == 200
    assert legacy.json()["diagnostics"]["recommendation_path"] == "fast"
    assert [item["id"] for item in legacy.json()["results"]] == [item["id"] for item in paged.json()["results"]]


# ------------------------------------------------ exclusions name one title


@pytest.mark.asyncio
async def test_on_the_fast_path_an_excluded_title_does_not_hide_its_franchise(tmp_path, franchise_catalog):
    """Already Watched and Want to Watch are about one title."""
    service = build_service(tmp_path, franchise_catalog, windows=8)

    # The user Likes franchise 50 and has watched one entry of franchise 3.
    plain = [a for page in await read_all(service, limit=50, liked_ids=[500]) for a in page]
    watched = next(anime_id for anime_id in plain if anime_id // 10 == 3)
    filtered = [a for page in await read_all(service, limit=50, liked_ids=[500], excluded_ids=[watched]) for a in page]
    assert watched not in filtered
    assert 3 in {anime_id // 10 for anime_id in filtered}, "another entry of the franchise is still recommended"


@pytest.mark.asyncio
async def test_the_starter_path_keeps_the_hybrid_series_policy_unchanged(tmp_path, franchise_catalog):
    """Recorded, not endorsed: with no Likes, an exclusion hides its whole franchise.

    The hybrid seeds its one-per-series filter with every excluded title. That
    is production behaviour and stays as it is until the policy evaluation in
    experiments/v21_quality/SERIES_POLICY.md is reviewed. This test exists so a
    change to it is a decision rather than an accident.
    """
    service = build_service(tmp_path, franchise_catalog)
    starter = [a for page in await read_all(service) for a in page]
    seen = starter[0]
    after = [a for page in await read_all(service, excluded_ids=[seen]) for a in page]
    assert seen not in after
    assert seen // 10 not in {anime_id // 10 for anime_id in after}
    assert len(after) == len(set(after)) == service.starter_depth


@pytest.mark.asyncio
async def test_a_watched_title_is_excluded_without_becoming_a_preference_signal(tmp_path, wide_catalog):
    service = build_service(tmp_path, wide_catalog, debug=True)
    plain = await service.page(RecommendPageRequest(liked_ids=LIKED))
    watched = ids_of(plain)[0]
    by_id = await service.page(RecommendPageRequest(liked_ids=LIKED, excluded_ids=[watched]))
    by_title = await service.page(RecommendPageRequest(liked_ids=LIKED, seen_titles=[f"saga{watched} chronicle"]))

    for response in (by_id, by_title):
        assert watched not in ids_of(response)
        # Same Likes, same folded-in user vector: nothing was learned from it.
        assert (
            response["diagnostics"]["debug"]["als_user_vector_sha256_16"]
            == plain["diagnostics"]["debug"]["als_user_vector_sha256_16"]
        )
        assert response["diagnostics"]["known_positive_count"] == len(LIKED)
        # Everything else keeps its order: the watched title only left a gap.
        assert ids_of(response)[:17] == ids_of(plain)[1:18]
    assert by_title["diagnostics"]["profile"]["excluded_titles_resolved"] == 1
