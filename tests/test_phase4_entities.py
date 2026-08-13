from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.agents.schemas import AgentIntent, IntentEntityMention, ProviderHealth
from app.core.config import Settings
from app.main import create_app
from backend.anime_agent.entities import EntityResolver
from backend.anime_agent.recommender import AnimeRecommender


class EntityIntentProvider:
    name = "gemini"
    model = "phase4-test-model"

    def __init__(self, intents: list[AgentIntent]):
        self.intents = list(intents)

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        return self.intents.pop(0)

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def relationship_anime(
    anime_id: int,
    title: str,
    *,
    matching: bool,
    producer_only: bool = False,
    score: float = 8.0,
) -> dict[str, Any]:
    producer = "Entity Producer" if matching or producer_only else "Unrelated Producer"
    producer_id = 600 if matching or producer_only else 601
    studio = "Entity Studio" if matching else "Unrelated Studio"
    studio_id = 500 if matching else 501
    staff = (
        [
            {"id": 700, "name": "Sato, Ken", "role": "Director"},
            {"id": 701, "name": "Ito, Emi", "role": "Original Creator"},
        ]
        if matching
        else [{"id": 702, "name": "Other Person", "role": "Director"}]
    )
    character = (
        {"id": 800, "name": "Hero Prime", "role": "Main"}
        if matching
        else {"id": 801, "name": "Other Hero", "role": "Main"}
    )
    voice_actor_roles = (
        [
            {
                "voice_actor_id": 900,
                "voice_actor": "Matsuoka, Yoshitsugu",
                "character_id": character["id"],
                "character": character["name"],
                "language": "Japanese",
            }
        ]
        if matching
        else []
    )
    genres = ["Fantasy", "Isekai", "Shounen"] if matching else ["Action"]
    genre_groups = {"themes": ["Isekai"], "demographics": ["Shounen"]} if matching else {}
    return {
        "id": anime_id,
        "title": title,
        "score": score,
        "rank": anime_id,
        "popularity": anime_id * 10,
        "members": 10000,
        "synopsis": (
            "A quiet fantasy journey through another world."
            if matching
            else "Entity Producer Sato Ken Hero Prime Isekai Shounen, repeated for a strong text match."
        ),
        "start_year": 2022,
        "type": "TV",
        "episodes": 12,
        "image_url": "",
        "genres": genres,
        "genre_groups": genre_groups,
        "metadata_tokens": [f"genre_{value.casefold()}" for value in genres],
        "studios": [studio],
        "studio_relationships": [{"id": studio_id, "name": studio, "role": "Studio"}],
        "producers": [producer],
        "producer_relationships": [{"id": producer_id, "name": producer, "role": "Producer"}],
        "characters": [character],
        "character_names": [character["name"]],
        "character_relationships": [character],
        "staff": staff,
        "staff_relationships": staff,
        "creators": staff,
        "voice_actors": ([{"id": 900, "name": "Matsuoka, Yoshitsugu", "language": "Japanese"}] if matching else []),
        "voice_actor_roles": voice_actor_roles,
    }


def phase4_catalog() -> list[dict[str, Any]]:
    catalog = [
        relationship_anime(1, "Crimson Archive", matching=True, score=8.1),
        relationship_anime(2, "Azure Signal", matching=True, score=7.9),
        relationship_anime(3, "Golden Thread", matching=True, score=7.7),
        relationship_anime(4, "Producer Side Story", matching=False, producer_only=True, score=8.4),
        relationship_anime(99, "Unrelated Perfect Match", matching=False, score=10.0),
    ]
    conflicting_characters = [
        {"id": 802, "name": "Sato, Ken", "role": "Main"},
        {"id": 803, "name": "Matsuoka, Yoshitsugu", "role": "Supporting"},
    ]
    catalog[-1]["characters"] = conflicting_characters
    catalog[-1]["character_names"] = [character["name"] for character in conflicting_characters]
    catalog[-1]["character_relationships"] = conflicting_characters
    return catalog


def phase4_client(tmp_path: Path, provider: EntityIntentProvider) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'phase4.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    return TestClient(create_app(settings=settings, catalog=phase4_catalog(), providers={"gemini": provider}))


def test_entity_resolver_supports_all_phase4_types_and_normalized_names() -> None:
    resolver = EntityResolver(phase4_catalog())
    expected = {
        "anime": "Crimson Archive",
        "character": "Hero Prime",
        "voice_actor": "Matsuoka, Yoshitsugu",
        "staff": "Sato, Ken",
        "director": "Sato, Ken",
        "original_creator": "Ito, Emi",
        "studio": "Entity Studio",
        "producer": "Entity Producer",
        "genre": "Fantasy",
        "theme": "Isekai",
        "demographic": "Shounen",
    }
    queries = {
        "anime": "Crimson Archiv",
        "character": " hero...prime ",
        "voice_actor": "YOSHITSUGU   MATSUOKA!!",
        "staff": "ken sato",
        "director": "  KEN, SATO  ",
        "original_creator": "emi ito",
        "studio": "ENTITY--STUDIO",
        "producer": "entity producer",
        "genre": "fantasy",
        "theme": "isekai",
        "demographic": "shounen",
    }

    for entity_type, query in queries.items():
        match = resolver.resolve(query, entity_type)
        assert match is not None, entity_type
        assert match["matched_name"] == expected[entity_type]
        assert match["related_anime_ids"], entity_type

    character = resolver.resolve("Hero Prime", "character")
    assert character is not None
    assert character["ambiguous"] is False
    assert character["entity_id"] == 800


def test_ambiguous_same_name_entities_preserve_distinct_ids() -> None:
    catalog = phase4_catalog()
    catalog[4]["producers"] = ["Entity Producer"]
    catalog[4]["producer_relationships"] = [{"id": 999, "name": "Entity Producer", "role": "Producer"}]

    match = EntityResolver(catalog).resolve("Entity Producer", "producer")

    assert match is not None
    assert match["ambiguous"] is True
    assert {value["entity_id"] for value in match["alternatives"]} == {600, 999}


def test_generic_relationship_constraints_filter_before_hybrid_reranking() -> None:
    catalog = phase4_catalog()
    resolver = EntityResolver(catalog)
    recommender = AnimeRecommender(catalog)
    constraints = {
        "producer": "Entity Producer",
        "director": "Ken Sato",
        "original_creator": "Emi Ito",
        "theme": "Isekai",
        "demographic": "Shounen",
    }

    for entity_type, query in constraints.items():
        resolved = resolver.resolve(query, entity_type)
        assert resolved is not None
        diagnostics: dict[str, Any] = {}
        results = recommender.recommend(
            required_entity_constraints=[resolved],
            query="Unrelated Perfect Match with strongest possible text similarity",
            one_per_series=False,
            limit=5,
            diagnostics=diagnostics,
        )
        allowed_ids = set(resolved["related_anime_ids"])
        assert results
        assert all(result["id"] in allowed_ids for result in results)
        assert all(
            any(value["entity_type"] == entity_type for value in result["entity_relationships"]) for result in results
        )
        assert diagnostics["verified_entity_matches"] == len(allowed_ids)
        assert 99 not in {result["id"] for result in results}


def test_explicit_producer_request_is_a_hard_agent_constraint_and_ignores_session(
    tmp_path: Path,
) -> None:
    provider = EntityIntentProvider(
        [
            AgentIntent(
                intent="recommend",
                entity_mentions=[
                    IntentEntityMention(text="Entity Producer", entity_type="producer", relation="direct")
                ],
                top_k=3,
            )
        ]
    )
    with phase4_client(tmp_path, provider) as client:
        client.post(
            "/api/session/phase4-producer/preferences",
            json={"liked_titles": ["Unrelated Perfect Match"]},
        )
        body = client.post(
            "/api/chat",
            json={
                "message": "Recommend 3 anime produced by Entity Producer.",
                "session_id": "phase4-producer",
                "debug": True,
            },
        ).json()

    recommendation = next(step for step in body["trace"] if step["tool"] == "recommend_anime")
    results = recommendation["result"]["results"]
    assert len(results) == 3
    assert all("Entity Producer" in result["producers"] for result in results)
    assert all(result["matched_producers"] == ["Entity Producer"] for result in results)
    assert all(
        any(
            evidence["entity_type"] == "producer"
            and evidence["entity_id"] == 600
            and evidence["relationship"] == "production_credit"
            for evidence in result["entity_relationships"]
        )
        for result in results
    )
    assert body["debug"]["required_entity_constraints"][0]["entity_type"] == "producer"
    assert "liked_titles" in body["debug"]["ignored_session_fields"]
    assert "Unrelated Perfect Match" not in {result["title"] for result in results}


def test_normal_order_director_request_returns_only_verified_director_credits(tmp_path: Path) -> None:
    provider = EntityIntentProvider(
        [
            AgentIntent(
                intent="recommend",
                entity_mentions=[IntentEntityMention(text="Ken Sato", entity_type="director", relation="direct")],
                top_k=3,
            )
        ]
    )
    with phase4_client(tmp_path, provider) as client:
        body = client.post(
            "/api/chat",
            json={"message": "Recommend 3 anime by director Ken Sato.", "debug": True},
        ).json()

    recommendation = next(step for step in body["trace"] if step["tool"] == "recommend_anime")
    results = recommendation["result"]["results"]
    assert len(results) == 3
    assert all(result["matched_directors"] == ["Sato, Ken"] for result in results)
    assert all(
        any(
            evidence["entity_type"] == "director" and evidence["role"] == "Director"
            for evidence in result["entity_relationships"]
        )
        for result in results
    )


def test_backend_recovers_explicit_director_when_provider_omits_entity(tmp_path: Path) -> None:
    provider = EntityIntentProvider(
        [
            AgentIntent(
                intent="recommend",
                entity_mentions=[IntentEntityMention(text="Ken Sato", entity_type="character", relation="direct")],
            )
        ]
    )
    with phase4_client(tmp_path, provider) as client:
        body = client.post(
            "/api/chat",
            json={"message": "Recommend 3 anime by director Ken Sato.", "debug": True},
        ).json()

    recommendation = next(step for step in body["trace"] if step["tool"] == "recommend_anime")
    results = recommendation["result"]["results"]
    assert len(results) == 3
    assert body["debug"]["resolved_entity"]["matched_name"] == "Sato, Ken"
    assert body["debug"]["entity_type"] == "director"
    assert all(result["matched_directors"] == ["Sato, Ken"] for result in results)


def test_unknown_required_entity_stops_before_recommendation(tmp_path: Path) -> None:
    provider = EntityIntentProvider(
        [
            AgentIntent(
                intent="recommend",
                entity_mentions=[
                    IntentEntityMention(text="Unknown Company", entity_type="producer", relation="direct")
                ],
                top_k=3,
            )
        ]
    )
    with phase4_client(tmp_path, provider) as client:
        body = client.post(
            "/api/chat",
            json={"message": "Recommend anime produced by Unknown Company.", "debug": True},
        ).json()

    assert body["mode"] == "catalog_constraint_error"
    assert body["debug"]["selected_tool"] == "search_entities"
    assert body["debug"]["errors"][0]["type"] == "unknown_required_entity"
    assert not any(step["tool"] == "recommend_anime" for step in body["trace"])


def test_voice_actor_relationship_contract_remains_verified_end_to_end(tmp_path: Path) -> None:
    provider = EntityIntentProvider(
        [
            AgentIntent(
                intent="recommend",
                required_voice_actors=["Yoshitsugu Matsuoka"],
                entity_mentions=[
                    IntentEntityMention(
                        text="Yoshitsugu Matsuoka",
                        entity_type="voice_actor",
                        relation="direct",
                    ),
                    IntentEntityMention(
                        text="Matsuoka, Yoshitsugu",
                        entity_type="character",
                        relation="direct",
                    ),
                ],
                top_k=3,
            )
        ]
    )
    with phase4_client(tmp_path, provider) as client:
        body = client.post(
            "/api/chat",
            json={
                "message": "Recommend 3 anime that have voice actor Yoshitsugu Matsuoka involved.",
                "debug": True,
            },
        ).json()

    recommendation = next(step for step in body["trace"] if step["tool"] == "recommend_anime")
    results = recommendation["result"]["results"]
    assert len(results) == 3
    assert all("Matsuoka, Yoshitsugu" in result["matched_voice_actors"] for result in results)
    assert all(result["voice_actor_roles"][0]["language"] == "Japanese" for result in results)
