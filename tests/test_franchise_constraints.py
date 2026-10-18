"""A named franchise is a constraint on what results *are*, not what they resemble.

The bug these cover: "I like watching Gundam movies. Is there any movies after
2020?" returned Gintama: The Final and Evangelion: 3.0+1.0 alongside two real
Gundam films. Both are 2021 movies and both rank well, so every scoring stage
was working as designed -- nothing in the pipeline had ever been told that
"Gundam" named the subject rather than a taste. The model was handed them as
verified candidates and dutifully wrote "While not a Gundam title..." next to
one of them.

The distinction being protected here is between two readings of the same word:

    "I like Gundam. Recommend similar anime."   -> similarity, may widen
    "Gundam movies after 2020"                  -> franchise, may not widen

Both readings must keep working, and the second must never be satisfied by
padding the list with plausible neighbours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.replan import PROTECTED_FIELDS, RELAXATION_LADDER
from app.agents.schemas import AgentIntent, ProviderHealth
from app.core.config import Settings
from app.main import create_app
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.title_family import (
    constraints_from_intent,
    enforce_result_eligibility,
    extract_required_title_families,
)

PRODUCTION_QUERY = "I like watching Gundam movies. Is there any movies after 2020?"

# The two titles the production bug surfaced. Both are legitimate similarity
# recommendations and neither is a Gundam entry.
GINTAMA = "Gintama: The Final"
EVANGELION = "Evangelion: 3.0+1.0 Thrice Upon a Time"


def entry(
    anime_id: int,
    title: str,
    *,
    media_type: str,
    year: int,
    genres: list[str],
    score: float = 8.0,
    episodes: int = 1,
    studios: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": anime_id,
        "title": title,
        "aliases": [title],
        "score": score,
        "rank": anime_id,
        "popularity": anime_id * 10,
        "members": 100_000 - anime_id,
        "synopsis": f"{title} is a {' and '.join(genres).lower()} story.",
        "start_year": year,
        "type": media_type,
        "episodes": episodes,
        "image_url": "",
        "genres": genres,
        "genre_groups": {},
        "metadata_tokens": [f"genre_{genre}" for genre in genres],
        "studios": studios or ["Sunrise"],
        "producers": [],
        "characters": [{"id": anime_id + 5000, "name": f"Pilot {anime_id}", "role": "Main"}],
        "character_names": [f"Pilot {anime_id}"],
        "character_relationships": [{"id": anime_id + 5000, "name": f"Pilot {anime_id}", "role": "Main"}],
        "staff": [{"id": 900, "name": "Test Director", "role": "Director"}],
        "staff_relationships": [{"id": 900, "name": "Test Director", "role": "Director"}],
        "creators": [{"id": 900, "name": "Test Director", "role": "Director"}],
        "voice_actors": [],
        "voice_actor_roles": [],
    }


MECHA = ["Action", "Sci-Fi", "Mecha"]


@pytest.fixture
def franchise_catalog() -> list[dict[str, Any]]:
    """Enough of the real shape to reproduce the bug and its fix.

    Deliberately stacked against the constraint: the non-Gundam 2021 movies
    score higher than every Gundam entry, share its genres, and one of them
    even shares its studio. If the franchise constraint is not enforced, hybrid
    scoring will prefer them, which is exactly what happened in production.
    """
    return [
        # Gundam movies after 2020 -- the correct answer to the production query.
        entry(
            1,
            "Gundam: G no Reconguista Movie IV - Gekitou ni Sakebu Ai",
            media_type="Movie",
            year=2022,
            genres=MECHA,
            score=7.6,
        ),
        entry(2, "Kidou Senshi Gundam SEED Freedom", media_type="Movie", year=2024, genres=MECHA, score=8.1),
        entry(3, "Mobile Suit Gundam: Hathaway's Flash", media_type="Movie", year=2021, genres=MECHA, score=7.9),
        # Gundam, but the wrong type or the wrong years.
        entry(
            4, "Kidou Senshi Gundam: Suisei no Majo", media_type="TV", year=2022, genres=MECHA, score=8.3, episodes=12
        ),
        entry(5, "SD Gundam World Heroes", media_type="TV", year=2021, genres=MECHA, score=6.4, episodes=24),
        entry(
            6, "Mobile Suit Gundam Wing: Endless Waltz Movie", media_type="Movie", year=1998, genres=MECHA, score=8.0
        ),
        entry(7, "Turn A Gundam I Movie: Earth Light", media_type="Movie", year=2002, genres=MECHA, score=7.4),
        # Not Gundam. Same year, same type, higher score -- the bug's payload.
        entry(8, GINTAMA, media_type="Movie", year=2021, genres=["Action", "Sci-Fi", "Comedy"], score=9.05),
        entry(9, EVANGELION, media_type="Movie", year=2021, genres=MECHA, score=8.9, studios=["Khara"]),
        entry(
            10,
            "Suzume no Tojimari",
            media_type="Movie",
            year=2022,
            genres=["Adventure", "Drama"],
            score=8.6,
            studios=["CoMix Wave Films"],
        ),
        # Neighbours a similarity request is entitled to reach for.
        entry(11, "Code Geass: Hangyaku no Lelouch", media_type="TV", year=2006, genres=MECHA, score=8.7, episodes=25),
        entry(12, "Tengen Toppa Gurren Lagann", media_type="TV", year=2007, genres=MECHA, score=8.6, episodes=27),
    ]


@pytest.fixture
def agent(franchise_catalog: list[dict[str, Any]]) -> AnimeAgent:
    return AnimeAgent(AnimeRecommender(franchise_catalog))


def answer_titles(response: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for step in response.get("trace") or []:
        result = step.get("result") or {}
        for row in result.get("results") or []:
            if isinstance(row, dict) and row.get("title"):
                titles.append(str(row["title"]))
    return titles


def respond(agent: AnimeAgent, message: str) -> dict[str, Any]:
    """The deterministic path, end to end, with no language model involved."""
    return agent._respond_from_intent(agent._rule_based_intent(message, []), message, [], None, {})


# --------------------------------------------------------------- extraction


@pytest.mark.parametrize(
    "message",
    [
        PRODUCTION_QUERY,
        "Gundam movies after 2020",
        "Gundam movies from 2021 onward",
        "Gundam TV series after 2020",
        "Gundam movies before 2020",
        "Are there any Gundam movies after 2020?",
        "What Gundam titles are in the Gundam franchise?",
    ],
)
def test_franchise_queries_produce_a_required_family(agent: AnimeAgent, message: str) -> None:
    assert extract_required_title_families(message, agent.recommender) == ["gundam"]


@pytest.mark.parametrize(
    "message",
    [
        "I like Gundam. Recommend similar anime.",
        "Recommend anime like Gundam",
        "Something like Gundam, please",
        "Recommend some mecha anime",
        "Show me sci-fi movies from 2021",
        "Recommend anime from Sunrise",
        "Recommend something dark and psychological",
    ],
)
def test_similarity_and_generic_queries_produce_no_family(agent: AnimeAgent, message: str) -> None:
    assert extract_required_title_families(message, agent.recommender) == []


def test_a_studio_name_is_not_franchise_evidence(agent: AnimeAgent) -> None:
    """Sunrise animates Gundam; Sunrise is not Gundam, in either direction."""
    assert extract_required_title_families("Sunrise movies after 2020", agent.recommender) == []


def test_a_genre_is_not_franchise_evidence(agent: AnimeAgent) -> None:
    assert extract_required_title_families("Mecha movies after 2020", agent.recommender) == []


# ------------------------------------------------------- the invariant itself


@pytest.mark.parametrize(
    ("message", "media_type", "min_year", "max_year"),
    [
        (PRODUCTION_QUERY, "Movie", 2021, None),
        ("Gundam movies after 2020", "Movie", 2021, None),
        ("Gundam movies from 2021 onward", "Movie", 2021, None),
        ("Gundam TV series after 2020", "TV", 2021, None),
        ("Gundam movies before 2020", "Movie", None, 2019),
    ],
)
def test_every_returned_entry_satisfies_family_type_and_year(
    agent: AnimeAgent,
    franchise_catalog: list[dict[str, Any]],
    message: str,
    media_type: str,
    min_year: int | None,
    max_year: int | None,
) -> None:
    by_title = {item["title"]: item for item in franchise_catalog}
    titles = answer_titles(respond(agent, message))

    assert titles, f"{message!r} should still find the entries the catalog does hold"
    for title in titles:
        item = by_title[title]
        assert "gundam" in title.casefold(), f"{title} is not a Gundam entry"
        assert item["type"] == media_type, f"{title} is a {item['type']}, not a {media_type}"
        if min_year is not None:
            assert item["start_year"] >= min_year
        if max_year is not None:
            assert item["start_year"] <= max_year


def test_the_production_query_cannot_return_gintama_or_evangelion(agent: AnimeAgent) -> None:
    """The exact two titles the bug produced, named so a regression is obvious."""
    titles = answer_titles(respond(agent, PRODUCTION_QUERY))

    assert GINTAMA not in titles
    assert EVANGELION not in titles
    assert set(titles) == {
        "Gundam: G no Reconguista Movie IV - Gekitou ni Sakebu Ai",
        "Kidou Senshi Gundam SEED Freedom",
        "Mobile Suit Gundam: Hathaway's Flash",
    }


def test_a_franchise_query_returns_fewer_than_top_k_rather_than_padding(agent: AnimeAgent) -> None:
    """Three matches means three results, not ten with seven neighbours."""
    response = respond(agent, "Recommend 10 Gundam movies after 2020")
    titles = answer_titles(response)

    assert len(titles) == 3
    assert all("gundam" in title.casefold() for title in titles)


def test_no_verified_match_is_reported_rather_than_filled(agent: AnimeAgent) -> None:
    """An unsatisfiable franchise constraint returns zero rows and says so."""
    response = respond(agent, "Gundam movies after 2030")

    assert answer_titles(response) == []
    assert "no verified match" in response["answer"].casefold()
    assert GINTAMA not in response["answer"]


def test_a_similarity_query_may_still_return_non_gundam_titles(agent: AnimeAgent) -> None:
    """The other reading of the same word has to keep working.

    The point is that the candidate pool is never narrowed: a similarity
    request ranks the whole catalog, so titles outside the franchise remain
    reachable. Ten slots against a twelve-row catalog makes that observable
    rather than a matter of where the fixture happens to rank things.
    """
    message = "I like Gundam. Recommend 10 similar anime."
    assert extract_required_title_families(message, agent.recommender) == []

    titles = answer_titles(respond(agent, message))

    assert titles
    assert any("gundam" not in title.casefold() for title in titles), (
        "a similarity request must be free to widen beyond the franchise"
    )


def test_a_franchise_query_does_not_collapse_the_series_it_asked_for(agent: AnimeAgent) -> None:
    """One-per-series de-duplication is the opposite of enumerating a franchise."""
    intent = agent._rule_based_intent(PRODUCTION_QUERY, [])
    agent._normalize_explicit_request(intent, PRODUCTION_QUERY)

    assert intent.required_title_families == ["gundam"]
    assert intent.one_per_series is False
    assert intent.exclude_related_series is False


# ----------------------------------------------- the final authoritative gate


def test_the_final_gate_removes_a_row_nothing_upstream_caught(agent: AnimeAgent) -> None:
    """The gate must not depend on the stage that produced the row.

    A row is injected directly into a finished response, as a reranking or
    merging bug would, and the gate is expected to catch it on the catalog
    evidence alone.
    """
    response = respond(agent, PRODUCTION_QUERY)
    step = next(step for step in response["trace"] if step["tool"] == "recommend_anime")
    step["result"]["results"].append({"id": 8, "title": GINTAMA})
    step["result"]["result_titles"].append(GINTAMA)

    intent = agent._rule_based_intent(PRODUCTION_QUERY, [])
    agent._normalize_explicit_request(intent, PRODUCTION_QUERY)
    dropped = agent.enforce_result_eligibility(intent, response)

    assert dropped == [GINTAMA]
    assert GINTAMA not in answer_titles(response)
    assert GINTAMA not in step["result"]["result_titles"]
    assert GINTAMA not in response["answer"]


def test_the_gate_rejects_a_row_whose_id_is_not_in_the_catalog(agent: AnimeAgent) -> None:
    """A row is a claim; the catalog is the evidence."""
    intent = agent._rule_based_intent(PRODUCTION_QUERY, [])
    agent._normalize_explicit_request(intent, PRODUCTION_QUERY)
    response = {
        "trace": [
            {
                "tool": "recommend_anime",
                "arguments": {},
                "result": {"results": [{"id": 99_999, "title": "Mobile Suit Gundam: Invented"}]},
            }
        ]
    }

    dropped = enforce_result_eligibility(agent.recommender, constraints_from_intent(intent), response)

    assert dropped == ["Mobile Suit Gundam: Invented"]
    assert response["trace"][0]["result"]["results"] == []


def test_the_gate_leaves_an_ordinary_similarity_request_alone(agent: AnimeAgent) -> None:
    """Nothing is filtered when the request never pinned a subject."""
    message = "Recommend some mecha anime"
    intent = agent._rule_based_intent(message, [])
    agent._normalize_explicit_request(intent, message)
    response = respond(agent, message)
    before = answer_titles(response)

    assert before
    assert agent.enforce_result_eligibility(intent, response) == []
    assert answer_titles(response) == before


# --------------------------------------------------------------- replanning


def test_a_required_family_is_protected_from_the_relaxation_ladder() -> None:
    assert "required_title_families" in PROTECTED_FIELDS
    assert PROTECTED_FIELDS.isdisjoint({field for field, _description, _transform in RELAXATION_LADDER})


# ----------------------------------------------------------- through the API


class FixedIntentProvider:
    """Parses to a caller-supplied intent and echoes back whatever it is given.

    Modelled on the real failure: the provider returns the intent Gemini
    actually produced for the production query -- franchise in the free text,
    nowhere in the constraints -- so the test proves the backend does not need
    the model to get this right.
    """

    def __init__(self, intent: AgentIntent, answer: str | None = None):
        self.name = "gemini"
        self.model = "fixed-intent"
        self.intent = intent
        self.answer = answer
        self.verified_data: dict[str, Any] | None = None

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        return self.intent

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        self.verified_data = verified_tool_data
        return self.answer if self.answer is not None else str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def client_for(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    provider: FixedIntentProvider,
) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'franchise.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers={"gemini": provider}))


def gemini_style_intent() -> AgentIntent:
    """What the real provider returned for the production query."""
    return AgentIntent(
        intent="recommend",
        catalog_query="Gundam",
        formats=["Movie"],
        min_year=2021,
        top_k=10,
        free_text_preferences="I like watching Gundam movies.",
    )


def test_the_model_only_ever_sees_eligible_candidates(
    tmp_path: Path,
    franchise_catalog: list[dict[str, Any]],
) -> None:
    provider = FixedIntentProvider(gemini_style_intent())
    with client_for(tmp_path, franchise_catalog, provider) as client:
        response = client.post("/api/chat", json={"message": PRODUCTION_QUERY, "debug": True})

    assert response.status_code == 200
    assert provider.verified_data is not None
    handed_to_model = [
        row["title"]
        for step in provider.verified_data["verified_tool_trace"]
        for row in step.get("result", {}).get("results", [])
        if row.get("title")
    ]
    assert handed_to_model, "the model should still receive the real matches"
    assert all("gundam" in title.casefold() for title in handed_to_model)
    assert GINTAMA not in handed_to_model
    assert EVANGELION not in handed_to_model


def test_an_answer_naming_an_ineligible_title_is_not_served(
    tmp_path: Path,
    franchise_catalog: list[dict[str, Any]],
) -> None:
    """Even a model that ignores its evidence cannot put Gintama in the answer."""
    invented = (
        "Here are some movies after 2020:\n"
        f"- {GINTAMA}: while not a Gundam title, it is highly rated.\n"
        f"- {EVANGELION}: a psychological mecha drama.\n"
    )
    provider = FixedIntentProvider(gemini_style_intent(), answer=invented)
    with client_for(tmp_path, franchise_catalog, provider) as client:
        response = client.post("/api/chat", json={"message": PRODUCTION_QUERY, "debug": True})

    body = response.json()
    assert GINTAMA not in body["answer"]
    assert EVANGELION not in body["answer"]
    assert body["mode"] != "gemini_grounded"


def test_a_franchise_request_is_never_replanned_into_a_wider_one(
    tmp_path: Path,
    franchise_catalog: list[dict[str, Any]],
) -> None:
    """Relaxing the year would answer a question nobody asked."""
    provider = FixedIntentProvider(AgentIntent(intent="recommend", formats=["Movie"], min_year=2031, top_k=10))
    with client_for(tmp_path, franchise_catalog, provider) as client:
        response = client.post("/api/chat", json={"message": "Gundam movies after 2030", "debug": True})

    body = response.json()
    assert body.get("relaxations") in (None, [])
    assert body["debug"]["validated_intent"]["required_title_families"] == ["gundam"]
    assert answer_titles(body) == []
    assert "no verified match" in body["answer"].casefold()


def test_an_ordinary_request_still_replans(
    tmp_path: Path,
    franchise_catalog: list[dict[str, Any]],
) -> None:
    """The franchise guard must not switch replanning off for everything else."""
    provider = FixedIntentProvider(AgentIntent(intent="recommend", include_genres=["Mecha"], min_score=9.99, top_k=3))
    with client_for(tmp_path, franchise_catalog, provider) as client:
        response = client.post("/api/chat", json={"message": "Mecha anime rated above 9.99", "debug": True})

    body = response.json()
    assert [step["field"] for step in body.get("relaxations") or []] == ["min_score"]
