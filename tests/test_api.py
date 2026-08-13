from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.schemas import AgentIntent, ProviderHealth
from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.main import create_app


class MockProvider:
    def __init__(self, name: str, intent: AgentIntent | None = None, failure: Exception | None = None):
        self.name = name
        self.model = f"mock-{name}"
        self.intent = intent or AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2)
        self.failure = failure

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        if self.failure:
            raise self.failure
        return self.intent

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.name,
            model=self.model,
            available=self.failure is None,
            detail=type(self.failure).__name__ if self.failure else None,
        )

    async def close(self) -> None:
        return None


def app_client(tmp_path: Path, catalog: list[dict[str, Any]], **settings_overrides: Any) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
        **settings_overrides,
    )
    providers = {"gemini": MockProvider("gemini"), "ollama": MockProvider("ollama")}
    return TestClient(create_app(settings=settings, catalog=catalog, providers=providers))


def test_valid_recommendation_and_reconstructable_score(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        response = client.post("/api/recommend", json={"include_genres": ["Supernatural"], "top_k": 2})
    assert response.status_code == 200
    result = response.json()["recommendations"][0]
    breakdown = result["score_breakdown"]
    contribution_sum = sum(channel["weighted_contribution"] for channel in breakdown["channels"].values())
    assert contribution_sum == pytest.approx(breakdown["pre_diversity_score"], abs=1e-5)
    assert breakdown["pre_diversity_score"] + breakdown["diversity_adjustment"] == pytest.approx(
        breakdown["final_score"], abs=1e-5
    )


def test_invalid_request_validation_is_structured(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        invalid_limit = client.post("/api/recommend", json={"top_k": 51})
        invalid_years = client.post("/api/recommend", json={"min_year": 2030, "max_year": 2020})
    assert invalid_limit.status_code == 422
    assert invalid_years.status_code == 422
    assert invalid_limit.json()["error"]["code"] == "validation_error"


def test_details_fuzzy_search_and_empty_hard_constraint(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        missing = client.get("/api/anime/999999")
        fuzzy = client.post(
            "/api/entities/search",
            json={"query": "Deat Nte", "entity_types": ["anime"], "top_k": 3},
        )
        empty = client.post("/api/recommend", json={"required_voice_actors": ["Unknown Person"], "top_k": 3})
    assert missing.status_code == 404
    assert fuzzy.json()["results"][0]["matched_name"] == "Death Note"
    assert empty.status_code == 200
    assert empty.json()["recommendations"] == []


def test_search_supports_filters_sorting_and_pagination(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        first = client.post(
            "/api/search",
            json={
                "query": "",
                "include_genres": ["Supernatural"],
                "exclude_genres": ["Drama"],
                "min_year": 2020,
                "sort_by": "score",
                "top_k": 1,
                "offset": 0,
                "semantic": False,
            },
        )
        second = client.post(
            "/api/search",
            json={
                "query": "",
                "include_genres": ["Supernatural"],
                "exclude_genres": ["Drama"],
                "min_year": 2020,
                "sort_by": "score",
                "top_k": 1,
                "offset": 1,
                "semantic": False,
            },
        )

    assert first.status_code == 200
    assert first.json()["total"] == 2
    assert first.json()["has_more"] is True
    assert first.json()["results"][0]["title"] == "Death Note"
    assert second.json()["offset"] == 1
    assert second.json()["has_more"] is False
    assert second.json()["results"][0]["title"] == "Ghost Hunt"


def test_rank_endpoint_is_explicit_and_session_independent(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        response = client.post(
            "/api/rank",
            json={
                "include_genres": ["Supernatural"],
                "sort_by": "title",
                "sort_order": "asc",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert [item["title"] for item in body["results"]] == ["Death Note", "Death Parade", "Ghost Hunt"]
    assert [item["ranking"]["position"] for item in body["results"]] == [1, 2, 3]
    assert body["diagnostics"]["operation"] == "rank_catalog"
    assert body["diagnostics"]["session_preferences_used"] is False


def test_required_voice_actor_is_verified_for_all_seven_results(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    with app_client(tmp_path, catalog) as client:
        response = client.post(
            "/api/recommend",
            json={"required_voice_actors": ["Yoshitsugu Matsuoka"], "top_k": 7},
        )
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 7
    assert all("Matsuoka, Yoshitsugu" in item["matched_voice_actors"] for item in recommendations)
    assert all("Matsuoka, Yoshitsugu" in item["reasons"][0] for item in recommendations)


def test_sessions_are_isolated(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        client.post("/api/session/alpha/preferences", json={"liked_titles": ["Death Note"]})
        alpha = client.get("/api/session/alpha").json()["profile"]
        beta = client.get("/api/session/beta").json()["profile"]
    assert alpha["liked_titles"] == ["Death Note"]
    assert beta["liked_titles"] == []


def test_session_path_rejects_invalid_identifiers(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        invalid_characters = client.get("/api/session/not%20a%20valid%20id")
        oversized = client.get(f"/api/session/{'x' * 129}")

    assert invalid_characters.status_code == 422
    assert oversized.status_code == 422


def test_agent_falls_back_from_gemini_to_ollama(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'fallback.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    providers = {
        "gemini": MockProvider("gemini", failure=ProviderUnavailable("gemini")),
        "ollama": MockProvider("ollama"),
    }
    with TestClient(create_app(settings=settings, catalog=catalog, providers=providers)) as client:
        response = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True})
    body = response.json()
    assert body["debug"]["llm_provider"] == "ollama"
    assert body["debug"]["fallback_used"] is True
    assert body["debug"]["tool_calls"] == ["recommend_anime"]


def test_both_providers_unavailable_use_rule_fallback(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'rules.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    providers = {name: MockProvider(name, failure=ProviderUnavailable(name)) for name in ("gemini", "ollama")}
    with TestClient(create_app(settings=settings, catalog=catalog, providers=providers)) as client:
        response = client.post("/api/chat", json={"message": "Recommend 2 supernatural anime", "debug": True})
    assert response.status_code == 200
    assert response.json()["debug"]["parser_mode"] == "rule_fallback"


def test_health_reports_each_optional_provider(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    with app_client(tmp_path, catalog) as client:
        response = client.get("/api/health")
    components = response.json()["components"]
    assert response.status_code == 200
    assert components["api"]["status"] == "healthy"
    assert components["database"]["status"] == "healthy"
    assert components["gemini"]["status"] == "healthy"
    assert components["ollama"]["status"] == "healthy"


def test_rate_limit_and_body_limit(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    limited_settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'limited.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1,
        max_request_body_bytes=1024,
    )
    providers = {"gemini": MockProvider("gemini"), "ollama": MockProvider("ollama")}
    with TestClient(create_app(settings=limited_settings, catalog=catalog, providers=providers)) as client:
        assert client.get("/api/meta").status_code == 200
        assert client.get("/api/meta").status_code == 429

    with app_client(tmp_path / "body", catalog, max_request_body_bytes=1024) as client:
        oversized = client.post("/api/chat", json={"message": "x" * 1800})
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_too_large"


def test_frontend_and_errors_do_not_expose_secret(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    sentinel = "TEST-PRIVATE-KEY-DO-NOT-LEAK"
    frontend = "\n".join(path.read_text(encoding="utf-8") for path in Path("frontend").glob("*.*"))
    assert sentinel not in frontend
    assert ".env" in Path(".gitignore").read_text(encoding="utf-8")
    with app_client(tmp_path, catalog) as client:
        response = client.post("/api/recommend", json={"top_k": 999, "query": sentinel})
    assert sentinel not in response.text
