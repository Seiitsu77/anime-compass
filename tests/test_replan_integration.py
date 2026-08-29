"""End-to-end checks that bounded replanning fires through the real API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.agents.schemas import AgentIntent, ProviderHealth
from app.core.config import Settings
from app.main import create_app


class FixedIntentProvider:
    """Returns one caller-supplied intent, so the test controls the constraints."""

    def __init__(self, intent: AgentIntent):
        self.name = "gemini"
        self.model = "fixed-intent"
        self.intent = intent

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        return self.intent

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def client_for(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    intent: AgentIntent,
    **overrides: Any,
) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'replan.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
        **overrides,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers={"gemini": FixedIntentProvider(intent)}))


def test_over_constrained_request_recovers_by_relaxing(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    """A score floor no catalog row meets should be relaxed, not returned empty."""
    intent = AgentIntent(intent="recommend", include_genres=["Supernatural"], min_score=9.99, top_k=3)
    with client_for(tmp_path, catalog, intent) as client:
        response = client.post("/api/chat", json={"message": "Supernatural anime rated above 9.99", "debug": True})

    body = response.json()
    assert response.status_code == 200
    relaxations = body.get("relaxations") or []
    assert [step["field"] for step in relaxations] == ["min_score"]
    # The agent normalises "above 9.99" to a 10.0 floor before ranking, so the
    # relaxed value is the effective constraint rather than the parsed one.
    assert relaxations[0]["removed_value"] == 10.0
    assert relaxations[0]["result_count"] > 0


def test_satisfiable_request_does_not_replan(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    intent = AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2)
    with client_for(tmp_path, catalog, intent) as client:
        response = client.post("/api/chat", json={"message": "Recommend supernatural anime", "debug": True})

    body = response.json()
    assert not body.get("relaxations")


def test_replanning_can_be_disabled(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    intent = AgentIntent(intent="recommend", include_genres=["Supernatural"], min_score=9.99, top_k=3)
    with client_for(tmp_path, catalog, intent, max_replan_steps=0) as client:
        response = client.post("/api/chat", json={"message": "Supernatural anime rated above 9.99", "debug": True})

    body = response.json()
    assert not body.get("relaxations")


def test_required_entity_constraints_survive_replanning(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    """Relaxation must never widen a request away from a verified entity."""
    intent = AgentIntent(
        intent="recommend",
        required_voice_actors=["Matsuoka, Yoshitsugu"],
        min_score=9.99,
        top_k=3,
    )
    with client_for(tmp_path, catalog, intent) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Matsuoka anime rated above 9.99", "debug": True},
        )

    body = response.json()
    assert response.status_code == 200
    trace = body.get("debug", {}).get("tool_calls") or []
    # Whatever was relaxed, the voice-actor requirement is still in the plan.
    for step in body.get("relaxations") or []:
        assert step["field"] != "required_voice_actors"
    assert trace or body.get("answer")
