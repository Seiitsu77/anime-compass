from __future__ import annotations

from copy import deepcopy
from typing import Any

from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender


class OfflineClient:
    model = "deterministic-test"
    base_url = "local"

    @staticmethod
    def is_available() -> bool:
        return False


def _clone(
    source: dict[str, Any],
    anime_id: int,
    title: str,
    **changes: Any,
) -> dict[str, Any]:
    item = deepcopy(source)
    item.update({"id": anime_id, "title": title, **changes})
    item["rank"] = changes.get("rank", anime_id)
    item["popularity"] = changes.get("popularity", anime_id)
    return item


def _agent(
    catalog: list[dict[str, Any]],
    session: dict[str, Any] | None = None,
) -> tuple[AnimeAgent, dict[str, Any]]:
    mutable_session = session or {}

    def get_session(_session_id: str | None) -> dict[str, Any]:
        return deepcopy(mutable_session)

    def update_session(
        _session_id: str | None,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, list):
                mutable_session[key] = list(dict.fromkeys([*mutable_session.get(key, []), *value]))
            elif isinstance(value, dict) and isinstance(mutable_session.get(key), dict):
                mutable_session[key].update(value)
            else:
                mutable_session[key] = deepcopy(value)
        return deepcopy(mutable_session)

    return (
        AnimeAgent(
            AnimeRecommender(catalog),
            client=OfflineClient(),
            get_session_profile=get_session,
            update_session_preferences=update_session,
        ),
        mutable_session,
    )


def test_madhouse_chat_ranking_matches_exact_manual_filter(
    catalog: list[dict[str, Any]],
) -> None:
    base = catalog[0]
    titles_and_scores = [
        ("Madhouse One", 9.4, 2024),
        ("Madhouse Two", 9.2, 2018),
        ("Madhouse Three", 8.9, 2012),
        ("Madhouse Four", 8.7, 2019),
        ("Madhouse Five", 8.5, 2015),
        ("Madhouse Six", 8.0, 2020),
    ]
    ranking_catalog = [
        _clone(
            base,
            100 + index,
            title,
            score=score,
            start_year=year,
            type="TV",
            studios=["Madhouse"],
        )
        for index, (title, score, year) in enumerate(titles_and_scores)
    ]
    ranking_catalog.append(
        _clone(
            base,
            200,
            "Wrong Studio",
            score=10.0,
            start_year=2024,
            type="TV",
            studios=["Other Studio"],
        )
    )
    agent, _ = _agent(ranking_catalog)

    response = agent.respond("Show me the 5 highest-rated Madhouse TV anime released after 2010.")
    chat_results = response["trace"][0]["result"]["results"]
    manual_results, _ = agent.recommender.search_page(
        "",
        required_studios=["Madhouse"],
        formats=["TV"],
        min_year=2011,
        sort_by="score",
        limit=5,
    )

    expected = [title for title, _score, _year in titles_and_scores[:5]]
    assert [item["title"] for item in chat_results] == expected
    assert [item["title"] for item in manual_results] == expected


def test_catalog_ranking_matches_aliases_and_requires_every_studio(
    catalog: list[dict[str, Any]],
) -> None:
    base = catalog[0]
    ranking_catalog = [
        _clone(
            base,
            250,
            "Localized Title",
            aliases=["Project Alias"],
            score=9.0,
            studios=["Studio A", "Studio B"],
        ),
        _clone(
            base,
            251,
            "Wrong Co-production",
            aliases=["Project Alias"],
            score=10.0,
            studios=["Studio A"],
        ),
    ]
    recommender = AnimeRecommender(ranking_catalog)

    ranked, diagnostics = recommender.rank_catalog(
        query="Project Alias",
        required_studios=["Studio A", "Studio B"],
        limit=5,
    )

    assert [item["title"] for item in ranked] == ["Localized Title"]
    assert diagnostics["candidate_count"] == 1


def test_gundam_tv_query_returns_requested_top_eight(
    catalog: list[dict[str, Any]],
) -> None:
    base = catalog[0]
    ranking_catalog = [
        _clone(
            base,
            300 + index,
            f"Mobile Suit Gundam Entry {index}",
            score=9.0 - index / 10,
            type="TV",
            studios=["Sunrise"],
        )
        for index in range(1, 10)
    ]
    ranking_catalog.append(
        _clone(
            base,
            400,
            "Mobile Suit Gundam Movie",
            score=10.0,
            type="Movie",
            studios=["Sunrise"],
        )
    )
    agent, _ = _agent(ranking_catalog)

    response = agent.respond("Give me the top 8 Gundam anime based on scores. I only want TV series.")
    results = response["trace"][0]["result"]["results"]

    assert len(results) == 8
    assert all(item["type"] == "TV" and "Gundam" in item["title"] for item in results)
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results],
        reverse=True,
    )


def test_explicit_negated_title_never_becomes_a_positive_reference(
    catalog: list[dict[str, Any]],
) -> None:
    rewrite = _clone(
        catalog[0],
        30,
        "Death Note: Rewrite",
        score=8.1,
    )
    agent, _ = _agent([*catalog, rewrite])

    response = agent.respond("Recommend five mystery anime, but not Death Note or related titles.")
    arguments = response["trace"][0]["arguments"]
    result_titles = [item["title"] for item in response["trace"][0]["result"]["results"]]

    assert "Death Note" in arguments["excluded_titles"]
    assert "Death Note" not in arguments["reference_titles"]
    assert all(not title.startswith("Death Note") for title in result_titles)


def test_positive_sequel_request_is_a_title_family_lookup_not_negative_feedback(
    catalog: list[dict[str, Any]],
) -> None:
    rewrite = _clone(catalog[0], 30, "Death Note: Rewrite", score=8.1)
    agent, _ = _agent([*catalog, rewrite])

    response = agent.respond("Show me sequels to Death Note.")

    assert response["mode"] == "catalog_search"
    assert "Death Note: Rewrite" in response["answer"]
    assert "title-family match" in response["answer"]
    assert response["trace"][0]["tool"] == "search_anime"


def test_missing_director_credit_returns_honest_constraint_error(
    catalog: list[dict[str, Any]],
) -> None:
    monster = _clone(
        catalog[0],
        40,
        "Monster",
        staff=[],
        staff_relationships=[],
        creators=[],
    )
    agent, _ = _agent([*catalog, monster])

    response = agent.respond("Recommend 3 works by the director of Monster.")

    assert response["mode"] == "catalog_constraint_error"
    assert "no Director credit" in response["answer"]
    assert all(step["tool"] != "recommend_anime" for step in response["trace"])


def test_result_ordinals_work_for_details_and_standalone_watched_updates(
    catalog: list[dict[str, Any]],
) -> None:
    session = {
        "last_recommendations": ["Death Note", "Death Parade", "Ghost Hunt"],
        "last_recommendation_intent": {},
    }
    agent, mutable_session = _agent(catalog, session)

    details = agent.respond(
        "What score did the second have?",
        session_id="ordinal-session",
    )
    watched = agent.respond(
        "I watched the third.",
        session_id="ordinal-session",
    )

    assert details["mode"] == "catalog_introduction"
    assert "Death Parade" in details["answer"]
    assert "8.00/10" in details["answer"]
    assert watched["mode"] == "session_update"
    assert "Ghost Hunt" in mutable_session["seen_titles"]


def test_season_numbers_are_not_mistaken_for_previous_result_positions(
    catalog: list[dict[str, Any]],
) -> None:
    agent, _ = _agent(catalog, {"last_recommendations": ["Death Note"]})

    references, watched = agent._classified_result_indices("Recommend the second season of a supernatural series.")

    assert references == []
    assert watched == []


def test_fresh_explicit_request_does_not_inherit_previous_hard_genres(
    catalog: list[dict[str, Any]],
) -> None:
    session = {
        "last_recommendations": ["Quiet Romance"],
        "last_recommendation_intent": {
            "include_genres": ["Romance"],
            "free_text_preferences": "gentle romance",
            "top_k": 5,
        },
    }
    agent, _ = _agent(catalog, session)

    response = agent.respond(
        "Recommend 3 supernatural anime.",
        session_id="fresh-request",
    )
    arguments = next(step["arguments"] for step in response["trace"] if step["tool"] == "recommend_anime")

    assert arguments["include_genres"] == ["Supernatural"]
    assert "Romance" not in arguments["include_genres"]


def test_agent_honors_fifty_item_limit_without_silent_clamping(
    catalog: list[dict[str, Any]],
) -> None:
    base = catalog[0]
    large_catalog = [
        _clone(
            base,
            1_000 + index,
            f"Standalone Choice {index:02d}",
            score=8.0 + (index % 10) / 100,
            type="TV",
        )
        for index in range(60)
    ]
    agent, _ = _agent(large_catalog)

    response = agent.respond("Show me 50 TV anime.")
    recommendation = next(step for step in response["trace"] if step["tool"] == "recommend_anime")

    assert recommendation["arguments"]["top_k"] == 50
    assert recommendation["result"]["total_results"] == 50
    assert len(recommendation["result"]["result_titles"]) == 50
    assert sum(line.lstrip().startswith("- ") for line in response["answer"].splitlines()) == 50
