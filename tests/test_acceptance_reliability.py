from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.agents.schemas import AgentIntent, IntentEntityMention
from backend.anime_agent.agent import AnimeAgent, format_anime_introduction
from backend.anime_agent.intent import StructuredIntent
from backend.anime_agent.recommender import AnimeRecommender


class OfflineClient:
    model = "offline-test"
    base_url = "http://offline.test"

    @staticmethod
    def is_available() -> bool:
        return False


def build_item(
    source: dict[str, Any],
    anime_id: int,
    title: str,
    *,
    score: float,
    media_type: str = "TV",
    year: int = 2020,
    episodes: int = 12,
    genres: list[str] | None = None,
    studios: list[str] | None = None,
) -> dict[str, Any]:
    item = deepcopy(source)
    item.update(
        {
            "id": anime_id,
            "anime_id": anime_id,
            "title": title,
            "score": score,
            "type": media_type,
            "start_year": year,
            "episodes": episodes,
            "genres": genres or ["Action"],
            "studios": studios or ["Test Studio"],
        }
    )
    return item


def run_intent(
    catalog: list[dict[str, Any]],
    message: str,
    intent: AgentIntent,
    *,
    history: list[dict[str, str]] | None = None,
    session: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], StructuredIntent]:
    agent = AnimeAgent(AnimeRecommender(catalog), client=OfflineClient())  # type: ignore[arg-type]
    legacy = intent.to_legacy()
    response = agent._respond_from_intent(legacy, message, history or [], None, session or {})
    return response, legacy


def primary_step(response: dict[str, Any]) -> dict[str, Any]:
    return next(step for step in response["trace"] if step["tool"] in {"recommend_anime", "rank_catalog"})


def test_golden_ghibli_tv_request_overrides_hallucinated_character(
    catalog: list[dict[str, Any]],
) -> None:
    values = [
        build_item(
            catalog[0],
            101,
            "Ghibli Television Story",
            score=8.4,
            studios=["Studio Ghibli"],
        ),
        build_item(
            catalog[0],
            102,
            "Ghibli Film",
            score=9.1,
            media_type="Movie",
            studios=["Studio Ghibli"],
        ),
        build_item(catalog[0], 103, "Other Television Story", score=9.5),
    ]
    flawed = AgentIntent(
        intent="recommend",
        required_characters=["With"],
        entity_mentions=[IntentEntityMention(text="With", entity_type="character", relation="direct")],
        top_k=10,
    )

    response, normalized = run_intent(
        values,
        "Recommend 5 Studio Ghibli TV anime.",
        flawed,
    )

    step = primary_step(response)
    assert normalized.formats == ["TV"]
    assert normalized.required_characters == []
    assert normalized.required_studios == ["Studio Ghibli"]
    assert step["result"]["results"]
    assert all(item["type"] == "TV" for item in step["result"]["results"])
    assert all("Studio Ghibli" in item["studios"] for item in step["result"]["results"])


def test_golden_iyashikei_request_recovers_genre_and_format(
    catalog: list[dict[str, Any]],
) -> None:
    values = [
        build_item(catalog[0], 201, "Quiet Morning", score=8.2, genres=["Iyashikei"]),
        build_item(catalog[0], 202, "Calm Evening", score=8.0, genres=["Iyashikei"]),
        build_item(catalog[0], 203, "Loud Battle", score=9.6, genres=["Action"]),
    ]
    flawed = AgentIntent(
        intent="recommend",
        required_characters=["With"],
        entity_mentions=[IntentEntityMention(text="With", entity_type="character", relation="direct")],
        top_k=10,
    )

    response, normalized = run_intent(values, "Find 2 iyashikei TV anime.", flawed)
    step = primary_step(response)

    assert normalized.include_genres == ["Iyashikei"]
    assert normalized.formats == ["TV"]
    assert normalized.required_characters == []
    assert {item["title"] for item in step["result"]["results"]} == {
        "Quiet Morning",
        "Calm Evening",
    }


def test_golden_gundam_ranking_is_recovered_from_wrong_llm_intent(
    catalog: list[dict[str, Any]],
) -> None:
    values = [
        build_item(catalog[0], 301, "Mobile Suit Gundam Alpha", score=7.9),
        build_item(catalog[0], 302, "Mobile Suit Gundam Zenith", score=9.2),
        build_item(catalog[0], 303, "Mobile Suit Gundam Middle", score=8.5),
        build_item(
            catalog[0],
            304,
            "Mobile Suit Gundam Movie",
            score=9.8,
            media_type="Movie",
        ),
        build_item(catalog[0], 305, "Unrelated Masterpiece", score=10.0),
    ]

    response, normalized = run_intent(
        values,
        "give me the top 8 Gundam anime based on scores. I only want TV series.",
        AgentIntent(intent="recommend", exclude_genres=["OVA", "ONA", "Movie"], top_k=10),
    )
    step = primary_step(response)

    assert normalized.intent == "rank_catalog"
    assert normalized.catalog_query == "Gundam"
    assert normalized.formats == ["TV"]
    assert normalized.top_k == 8
    assert [item["title"] for item in step["result"]["results"]] == [
        "Mobile Suit Gundam Zenith",
        "Mobile Suit Gundam Middle",
        "Mobile Suit Gundam Alpha",
    ]


def test_golden_studio_ranking_applies_studio_year_and_format_filters(
    catalog: list[dict[str, Any]],
) -> None:
    values = [
        build_item(
            catalog[0],
            401,
            "Madhouse New High",
            score=9.1,
            year=2021,
            studios=["Madhouse"],
        ),
        build_item(
            catalog[0],
            402,
            "Madhouse New Mid",
            score=8.4,
            year=2016,
            studios=["Madhouse"],
        ),
        build_item(catalog[0], 403, "Madhouse New Three", score=8.1, year=2019, studios=["Madhouse"]),
        build_item(catalog[0], 404, "Madhouse New Four", score=7.9, year=2017, studios=["Madhouse"]),
        build_item(catalog[0], 405, "Madhouse New Five", score=7.7, year=2012, studios=["Madhouse"]),
        build_item(catalog[0], 406, "Madhouse Old", score=9.8, year=2008, studios=["Madhouse"]),
        build_item(catalog[0], 407, "Other Studio High", score=10.0, year=2022),
    ]

    response, normalized = run_intent(
        values,
        "show me the 5 highest-rated Madhouse TV anime released after 2010.",
        AgentIntent(intent="search", catalog_query="after", top_k=10),
    )
    step = primary_step(response)

    assert normalized.intent == "rank_catalog"
    assert normalized.catalog_query == ""
    assert normalized.required_studios == ["Madhouse"]
    assert normalized.min_year == 2011
    assert normalized.top_k == 5
    assert [item["title"] for item in step["result"]["results"]] == [
        "Madhouse New High",
        "Madhouse New Mid",
        "Madhouse New Three",
        "Madhouse New Four",
        "Madhouse New Five",
    ]


def test_golden_numeric_constraints_survive_bad_entity_parse(
    catalog: list[dict[str, Any]],
) -> None:
    values = [
        build_item(catalog[0], 501, "Eligible One", score=8.7, year=2019, episodes=12),
        build_item(catalog[0], 502, "Eligible Two", score=8.2, year=2023, episodes=23),
        build_item(catalog[0], 503, "Exactly Eight", score=8.0, year=2022, episodes=12),
        build_item(catalog[0], 504, "Too Long", score=9.4, year=2022, episodes=24),
        build_item(catalog[0], 505, "Too Old", score=9.5, year=2015, episodes=12),
    ]
    flawed = AgentIntent(
        intent="search",
        entity_mentions=[IntentEntityMention(text="Score", entity_type="character", relation="direct")],
        top_k=10,
    )

    response, normalized = run_intent(
        values,
        "Find 5 TV anime under 24 episodes, after 2015, with score > 8.",
        flawed,
    )
    step = primary_step(response)

    assert normalized.intent == "recommend"
    assert normalized.min_score == 8.01
    assert normalized.min_year == 2016
    assert normalized.max_episodes == 23
    assert normalized.required_characters == []
    assert {item["title"] for item in step["result"]["results"]} == {
        "Eligible One",
        "Eligible Two",
    }


def test_golden_voice_actor_request_recovers_catalog_relationship(
    catalog: list[dict[str, Any]],
) -> None:
    actor_catalog = []
    titles = ["Crimson Journey", "Azure Signal", "Golden Archive", "Silver Horizon"]
    for title, source in zip(titles, catalog[4:8], strict=True):
        item = deepcopy(source)
        item["title"] = title
        actor_catalog.append(item)
    flawed = AgentIntent(
        intent="search",
        required_characters=["Matsuoka, Yoshitsugu"],
        entity_mentions=[
            IntentEntityMention(
                text="Matsuoka, Yoshitsugu",
                entity_type="character",
                relation="direct",
            )
        ],
        top_k=10,
    )

    response, normalized = run_intent(
        actor_catalog,
        "Recommend 3 anime featuring voice actor Matsuoka, Yoshitsugu.",
        flawed,
    )
    step = primary_step(response)

    assert normalized.intent == "recommend"
    assert normalized.required_characters == []
    assert normalized.required_voice_actors == ["Matsuoka, Yoshitsugu"]
    assert len(step["result"]["results"]) == 3
    assert all(item["matched_voice_actors"] == ["Matsuoka, Yoshitsugu"] for item in step["result"]["results"])


def test_fresh_similarity_request_does_not_inherit_previous_filters(
    catalog: list[dict[str, Any]],
) -> None:
    agent = AnimeAgent(AnimeRecommender(catalog), client=OfflineClient())  # type: ignore[arg-type]
    intent = StructuredIntent(
        intent="recommend",
        reference_titles=["Steins;Gate"],
        free_text_preferences="similar to Steins;Gate",
        top_k=4,
    )
    session = {
        "last_recommendation_intent": {
            "required_studios": ["Madhouse"],
            "formats": ["Movie"],
            "min_year": 2020,
            "top_k": 7,
            "free_text_preferences": "Madhouse movies",
        }
    }

    merged = agent._merge_followup_intent(
        intent,
        "Recommend anime similar to Steins;Gate.",
        [],
        session,
    )

    assert merged.required_studios == []
    assert merged.formats == []
    assert merged.min_year is None
    assert merged.top_k == 4
    assert merged.free_text_preferences == "similar to Steins;Gate"


def test_negative_feedback_targets_named_or_ordinal_items_only(
    catalog: list[dict[str, Any]],
) -> None:
    agent = AnimeAgent(AnimeRecommender(catalog), client=OfflineClient())  # type: ignore[arg-type]
    history = [
        {
            "role": "assistant",
            "content": "- Death Note (TV): first\n- Death Parade (TV): second",
        }
    ]

    assert agent._collect_negative_feedback_titles("I dislike horror.", history) == []
    assert agent._collect_negative_feedback_titles("I disliked the second one.", history) == ["Death Parade"]
    assert agent._collect_negative_feedback_titles("None of these work.", history) == [
        "Death Note",
        "Death Parade",
    ]


def test_spoiler_safe_introduction_uses_only_opening_premise(
    catalog: list[dict[str, Any]],
) -> None:
    item = deepcopy(catalog[0])
    item["synopsis"] = (
        "A student discovers a mysterious notebook. He begins testing its dangerous power. "
        "The detective reveals the hidden identity. The final confrontation ends the story. "
        "[Written by MAL Rewrite]"
    )

    introduction = format_anime_introduction(item)

    assert "mysterious notebook" in introduction
    assert "dangerous power" in introduction
    assert "hidden identity" not in introduction
    assert "final confrontation" not in introduction
    assert "MAL Rewrite" not in introduction
