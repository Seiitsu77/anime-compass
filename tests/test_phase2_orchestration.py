from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.agents.ollama_provider import OllamaAgentProvider
from app.agents.schemas import AgentIntent, ProviderHealth
from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.main import create_app


class TrackingProvider:
    def __init__(
        self,
        name: str,
        *,
        parse_failure: Exception | None = None,
        response_failure: Exception | None = None,
    ):
        self.name = name
        self.model = f"tracking-{name}"
        self.parse_failure = parse_failure
        self.response_failure = response_failure
        self.parse_calls = 0
        self.response_calls = 0

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        self.parse_calls += 1
        if self.parse_failure:
            raise self.parse_failure
        return AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2)

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        self.response_calls += 1
        if self.response_failure:
            raise self.response_failure
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def test_settings_accept_comma_separated_cors_origins(monkeypatch: Any) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://first.example,https://second.example")
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["https://first.example", "https://second.example"]


def phase2_client(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    providers: dict[str, TrackingProvider],
    **overrides: Any,
) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'phase2.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
        **overrides,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers=providers))


def test_response_generation_falls_back_to_alternate_provider(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    gemini = TrackingProvider("gemini", response_failure=ProviderUnavailable("gemini"))
    ollama = TrackingProvider("ollama")
    with phase2_client(tmp_path, catalog, {"gemini": gemini, "ollama": ollama}) as client:
        response = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True})

    body = response.json()
    assert response.status_code == 200
    assert body["debug"]["intent_provider"] == "gemini"
    assert body["debug"]["response_provider"] == "ollama"
    assert body["debug"]["fallback_used"] is True
    assert body["agent"]["provider"] == "ollama"
    assert [attempt["outcome"] for attempt in body["debug"]["provider_attempts"]] == [
        "success",
        "failed",
        "success",
    ]


def test_timeout_uses_alternate_provider_for_intent(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    gemini = TrackingProvider("gemini", parse_failure=TimeoutError())
    ollama = TrackingProvider("ollama")
    with phase2_client(tmp_path, catalog, {"gemini": gemini, "ollama": ollama}) as client:
        response = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True})

    body = response.json()
    assert body["debug"]["intent_provider"] == "ollama"
    assert body["debug"]["provider_attempts"][0]["error_type"] == "TimeoutError"
    assert "Recommend two supernatural anime" not in str(body["debug"]["provider_errors"])


def test_circuit_opens_after_repeated_generation_failures(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    gemini = TrackingProvider("gemini", response_failure=ProviderUnavailable("gemini"))
    ollama = TrackingProvider("ollama")
    with phase2_client(
        tmp_path,
        catalog,
        {"gemini": gemini, "ollama": ollama},
        provider_failure_threshold=2,
        provider_cooldown_seconds=300,
    ) as client:
        client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True})
        second = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True}).json()
        third = client.post("/api/chat", json={"message": "Recommend two supernatural anime", "debug": True}).json()

    assert second["debug"]["circuit_breakers"]["gemini"]["open"] is True
    assert third["debug"]["provider_attempts"][0] == {
        "phase": "intent",
        "provider": "gemini",
        "outcome": "circuit_open",
    }
    assert third["debug"]["intent_provider"] == "ollama"
    assert gemini.parse_calls == 2


def test_provider_diagnostics_are_absent_without_debug(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
) -> None:
    providers = {"gemini": TrackingProvider("gemini"), "ollama": TrackingProvider("ollama")}
    with phase2_client(tmp_path, catalog, providers) as client:
        body = client.post("/api/chat", json={"message": "Recommend two supernatural anime"}).json()

    assert "debug" not in body
    assert "parser_mode" not in body
    assert "validated_intent" not in body
    assert "fallback_used" not in body["agent"]


def test_ollama_health_requires_configured_model() -> None:
    def installed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "other:latest"}]})

    async def check() -> tuple[ProviderHealth, ProviderHealth]:
        available_provider = OllamaAgentProvider(
            base_url="http://ollama.invalid",
            model="qwen3:1.7b",
            timeout=2,
            max_output_tokens=500,
            transport=httpx.MockTransport(installed),
        )
        missing_provider = OllamaAgentProvider(
            base_url="http://ollama.invalid",
            model="qwen3:1.7b",
            timeout=2,
            max_output_tokens=500,
            transport=httpx.MockTransport(missing),
        )
        available = await available_provider.health_check()
        unavailable = await missing_provider.health_check()
        await available_provider.close()
        await missing_provider.close()
        return available, unavailable

    available, unavailable = asyncio.run(check())
    assert available.available is True
    assert unavailable.available is False
    assert unavailable.detail == "Configured model is not installed"
