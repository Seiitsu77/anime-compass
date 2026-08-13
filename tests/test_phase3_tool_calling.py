from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.schemas import AgentIntent, IntentEntityMention, IntentPreferenceUpdate, ProviderHealth
from app.agents.tools import CatalogToolRegistry, ToolContractError
from app.core.config import Settings
from app.main import create_app
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender


class IntentSequenceProvider:
    name = "gemini"
    model = "phase3-test-model"

    def __init__(self, intents: list[AgentIntent], *, malicious_response: bool = False):
        self.intents = list(intents)
        self.malicious_response = malicious_response
        self.verified_payloads: list[dict[str, Any]] = []
        self.response_calls = 0

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        return self.intents.pop(0)

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        self.response_calls += 1
        self.verified_payloads.append(verified_tool_data)
        if self.malicious_response:
            results = next(
                step["result"]["results"]
                for step in verified_tool_data["verified_tool_trace"]
                if step["tool"] == "recommend_anime"
            )
            return f"- {results[0]['title']}: verified\n- Invented Anime: not in the tool result"
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def phase3_client(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    provider: IntentSequenceProvider,
) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'phase3.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers={"gemini": provider}))


def test_agent_intent_is_strict_normalized_and_hard_constraints_win() -> None:
    intent = AgentIntent(
        intent="recommend",
        required_voice_actors=[" Matsuoka, Yoshitsugu ", "matsuoka, yoshitsugu"],
        preferred_voice_actors=["Matsuoka, Yoshitsugu", "Hayami, Saori"],
        min_year=2020,
        max_year=2025,
        top_k=7,
    )

    assert intent.required_voice_actors == ["Matsuoka, Yoshitsugu"]
    assert intent.preferred_voice_actors == ["Hayami, Saori"]
    assert AgentIntent(intent="conversational").intent == "conversation"

    with pytest.raises(ValidationError):
        AgentIntent(intent="recommend", min_year=2025, max_year=2020)
    with pytest.raises(ValidationError):
        AgentIntent(
            intent="recommend",
            entity_mentions=[{"text": "Someone", "entity_type": "celebrity", "relation": "direct"}],
        )
    with pytest.raises(ValidationError):
        AgentIntent.model_validate({"intent": "recommend", "unexpected": True})


def test_provider_schema_is_compact_while_runtime_validation_remains_strict() -> None:
    schema = AgentIntent.provider_json_schema()
    serialized = json.dumps(schema)

    assert "inferred_constraints" not in schema["properties"]
    assert "IntentInferredConstraint" not in schema["$defs"]
    assert all("title" not in value for value in _schema_metadata_values(schema, "title"))
    assert '"default"' not in serialized
    assert len(serialized) < 4000
    essential_fields = {
        "intent",
        "catalog_query",
        "rank_by",
        "sort_order",
        "reference_titles",
        "entity_mentions",
        "include_genres",
        "exclude_genres",
        "required_studios",
        "required_staff",
        "required_characters",
        "required_voice_actors",
        "formats",
        "min_score",
        "min_year",
        "max_year",
        "max_episodes",
        "excluded_titles",
        "seen_titles",
        "top_k",
        "free_text_preferences",
        "preference_update",
    }
    assert essential_fields.issubset(schema["properties"])
    with pytest.raises(ValidationError):
        AgentIntent.model_validate({"intent": "recommend", "top_k": 200})


def _schema_metadata_values(value: Any, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        if key in value:
            found.append(str(value[key]))
        for child in value.values():
            found.extend(_schema_metadata_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_schema_metadata_values(child, key))
    return found


def test_tool_registry_plans_all_supported_intents() -> None:
    registry = CatalogToolRegistry()

    recommendation = registry.plan(
        AgentIntent(
            intent="recommend",
            required_voice_actors=["Matsuoka, Yoshitsugu"],
            entity_mentions=[IntentEntityMention(text="Matsuoka, Yoshitsugu", entity_type="voice_actor")],
        )
    )
    assert recommendation.primary_tool == "recommend_anime"
    assert recommendation.prerequisite_tools == ["search_entities", "resolve_entity"]
    assert registry.plan(AgentIntent(intent="rank_catalog")).primary_tool == "rank_catalog"
    assert registry.plan(AgentIntent(intent="search")).primary_tool == "search_entities"
    assert registry.plan(AgentIntent(intent="details")).primary_tool == "get_anime_details"
    assert registry.plan(AgentIntent(intent="update_preferences")).primary_tool == "update_session_preferences"
    assert registry.plan(AgentIntent(intent="conversation")).primary_tool is None


def test_tool_registry_rejects_wrong_tool_and_invalid_arguments() -> None:
    registry = CatalogToolRegistry()

    with pytest.raises(ToolContractError, match="not allowed"):
        registry.validate_trace(
            AgentIntent(intent="search"),
            [{"tool": "recommend_anime", "arguments": {"top_k": 2}}],
            response_mode="catalog_search",
        )
    with pytest.raises(ToolContractError, match="Invalid arguments"):
        registry.validate_trace(
            AgentIntent(intent="recommend"),
            [{"tool": "recommend_anime", "arguments": {"top_k": 99}}],
            response_mode="catalog_fallback",
        )


def test_each_structured_intent_routes_to_the_expected_backend_tool(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    provider = IntentSequenceProvider(
        [
            AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2),
            AgentIntent(
                intent="search",
                entity_mentions=[IntentEntityMention(text="Death Note", entity_type="anime")],
                top_k=3,
            ),
            AgentIntent(intent="details", reference_titles=["Death Note"]),
            AgentIntent(
                intent="update_preferences",
                preference_update=IntentPreferenceUpdate(liked_titles=["Death Note"]),
            ),
            AgentIntent(intent="conversation"),
        ]
    )
    with phase3_client(tmp_path, catalog, provider) as client:
        recommend = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True})
        search = client.post("/api/chat", json={"message": "Search for Death Note", "debug": True})
        details = client.post("/api/chat", json={"message": "Tell me about Death Note", "debug": True})
        update = client.post(
            "/api/chat",
            json={"message": "I liked Death Note", "session_id": "phase3-tools", "debug": True},
        )
        conversation = client.post("/api/chat", json={"message": "Thanks", "debug": True})

    assert recommend.status_code == 200
    assert recommend.json()["debug"]["tool_plan"]["primary_tool"] == "recommend_anime"
    assert "recommend_anime" in recommend.json()["debug"]["tool_calls"]
    assert search.json()["debug"]["tool_calls"] == ["search_entities"]
    assert details.json()["debug"]["tool_calls"] == ["get_anime_details"]
    assert update.json()["debug"]["tool_calls"] == ["update_session_preferences"]
    assert conversation.json()["debug"]["tool_calls"] == []
    assert provider.response_calls == 3


def test_gundam_tv_catalog_ranking_is_deterministic_and_ignores_session(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    ranked_catalog = []
    for anime_id, title, score, media_type in (
        (101, "Mobile Suit Gundam Alpha", 7.8, "TV"),
        (102, "Mobile Suit Gundam Zenith", 9.1, "TV"),
        (103, "Mobile Suit Gundam Middle", 8.4, "TV"),
        (104, "Mobile Suit Gundam Movie", 9.8, "Movie"),
        (105, "Unrelated Masterpiece", 10.0, "TV"),
    ):
        item = deepcopy(catalog[0])
        item.update({"id": anime_id, "anime_id": anime_id, "title": title, "score": score, "type": media_type})
        ranked_catalog.append(item)
    provider = IntentSequenceProvider(
        [
            AgentIntent(
                intent="rank_catalog",
                catalog_query="Gundam",
                formats=["TV"],
                rank_by="score",
                sort_order="desc",
                top_k=3,
            )
        ]
    )
    with phase3_client(tmp_path, ranked_catalog, provider) as client:
        client.post(
            "/api/session/rank-session/preferences",
            json={"liked_titles": ["Unrelated Masterpiece"], "preferred_genres": ["Mystery"]},
        )
        response = client.post(
            "/api/chat",
            json={
                "message": "Show me the 3 highest-scored Gundam TV anime",
                "session_id": "rank-session",
                "debug": True,
            },
        )

    body = response.json()
    step = next(step for step in body["trace"] if step["tool"] == "rank_catalog")
    assert [item["title"] for item in step["result"]["results"]] == [
        "Mobile Suit Gundam Zenith",
        "Mobile Suit Gundam Middle",
        "Mobile Suit Gundam Alpha",
    ]
    assert step["result"]["diagnostics"]["session_preferences_used"] is False
    assert body["debug"]["selected_tool"] == "rank_catalog"


def test_rule_parser_recognizes_gundam_score_ranking(catalog: list[dict[str, Any]]) -> None:
    agent = AnimeAgent(AnimeRecommender(catalog))
    intent = agent._rule_based_intent("Show me the 5 highest-scored Gundam TV anime", [])

    assert intent.intent == "rank_catalog"
    assert intent.catalog_query == "Gundam"
    assert intent.formats == ["TV"]
    assert intent.rank_by == "score"
    assert intent.sort_order == "desc"
    assert intent.top_k == 5


def test_grounding_accepts_short_title_and_explanatory_bullet_but_rejects_new_title(
    catalog: list[dict[str, Any]],
) -> None:
    item_86 = deepcopy(catalog[0])
    item_86.update({"id": 86, "anime_id": 86, "title": "86"})
    agent = AnimeAgent(AnimeRecommender([item_86, catalog[1]]))
    titles = ["86", "Death Parade"]
    valid = "1. 86 (TV): A verified catalog fit.\n2. Death Parade: Another fit.\n- Why these fit: both match."
    invalid = "1. 86 (TV): A verified catalog fit.\n2. Invented Anime: unsupported."

    assert agent._valid_recommendation_answer(valid, titles, [], 2)
    assert not agent._valid_recommendation_answer(invalid, titles, [], 2)


def test_previous_result_indices_are_resolved_by_backend_session_state(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    provider = IntentSequenceProvider(
        [
            AgentIntent(intent="recommend", free_text_preferences="varied anime", top_k=4),
            AgentIntent(
                intent="recommend",
                reference_result_indices=[4],
                watched_result_indices=[2],
                free_text_preferences="more like the selected result",
                top_k=3,
            ),
        ]
    )
    with phase3_client(tmp_path, catalog, provider) as client:
        first = client.post(
            "/api/chat",
            json={"message": "Recommend four varied anime", "session_id": "phase3-indices", "debug": True},
        ).json()
        previous_titles = next(
            step["result"]["results"] for step in first["trace"] if step["tool"] == "recommend_anime"
        )
        second = client.post(
            "/api/chat",
            json={
                "message": "I watched the second result. Give me more like the fourth one.",
                "session_id": "phase3-indices",
                "debug": True,
            },
        ).json()

    arguments = next(step["arguments"] for step in second["trace"] if step["tool"] == "recommend_anime")
    assert previous_titles[3]["title"] in arguments["reference_titles"]
    assert previous_titles[1]["title"] in arguments["seen_titles"]
    assert second["debug"]["validated_intent"]["reference_result_indices"] == [4]
    assert second["debug"]["validated_intent"]["watched_result_indices"] == [2]


def test_unverified_model_title_is_rejected_and_deterministic_answer_is_used(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    provider = IntentSequenceProvider(
        [AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2)],
        malicious_response=True,
    )
    with phase3_client(tmp_path, catalog, provider) as client:
        body = client.post(
            "/api/chat",
            json={"message": "Recommend two supernatural anime", "debug": True},
        ).json()

    assert body["debug"]["response_provider"] is None
    assert body["debug"]["fallback_used"] is True
    assert "Invented Anime" not in body["answer"]
    assert body["debug"]["provider_attempts"][-1]["phase"] == "response"
    assert body["debug"]["provider_attempts"][-1]["outcome"] == "failed"
    assert provider.verified_payloads[0]["validated_tool_calls"][0]["tool"] == "recommend_anime"
