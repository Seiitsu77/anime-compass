"""A provider failure must be legible in production logs.

Cloud Run returned `"mode": "catalog_fallback"` for every agent request. That
says only that no provider produced a grounded answer -- not whether Gemini was
misconfigured, out of quota, pointed at a model that no longer exists, or never
reached at all. The failure was caught in `AgentOrchestrator.respond` and
recorded in `provider_attempts`, which a caller sees only by passing
`debug=true`, so the logs showed nothing.

These tests pin the diagnostics that closed that gap, and the rule that makes
them safe to emit: an API key must never reach a log record, not even when the
upstream error quotes the key back at us -- which is exactly what Google's
"API key not valid" response does.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agents.gemini_provider import GeminiAgentProvider
from app.agents.orchestrator import AgentOrchestrator
from app.agents.schemas import AgentIntent, ProviderHealth, ProviderParseContext
from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.core.logging import JsonFormatter
from app.repositories.session_repository import SQLiteSessionRepository
from backend.anime_agent.recommender import AnimeRecommender

# Shaped like a real Google API key so the redaction pattern is exercised, and
# obviously fake so nobody mistakes it for one.
API_KEY = "AIzaSyTESTONLY0123456789abcdefghijklmn"


class CapturedLogs(logging.Handler):
    """Log records rendered exactly as the application would emit them.

    Formatting through JsonFormatter matters: a test that inspected
    `record.context` directly would pass even if the formatter leaked the value
    it is supposed to redact.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rendered: list[str] = []
        self._formatter = JsonFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        self.rendered.append(self._formatter.format(record))

    def events(self, message: str) -> list[dict[str, Any]]:
        payloads = [json.loads(line) for line in self.rendered]
        return [payload for payload in payloads if payload.get("message") == message]

    @property
    def text(self) -> str:
        return "\n".join(self.rendered)


@pytest.fixture
def captured() -> Iterator[CapturedLogs]:
    handler = CapturedLogs()
    logger = logging.getLogger("anime_compass.agent")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class StubProvider:
    """A provider whose outcome the test chooses."""

    def __init__(self, name: str, *, parse_failure: Exception | None = None) -> None:
        self.name = name
        self.model = f"stub-{name}"
        self.parse_failure = parse_failure

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        if self.parse_failure is not None:
            raise self.parse_failure
        return AgentIntent(intent="recommend", include_genres=["Supernatural"], top_k=2)

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def gemini_against(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = API_KEY,
) -> GeminiAgentProvider:
    return GeminiAgentProvider(
        api_key=api_key,
        model="gemini-3.1-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        timeout=5.0,
        max_output_tokens=256,
        transport=httpx.MockTransport(handler),
    )


def bad_key_response(request: httpx.Request) -> httpx.Response:
    """Google's real shape for a rejected key -- it echoes the key back."""
    return httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": f"API key not valid: {API_KEY}. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )


def quota_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Quota exceeded for quota metric 'Generate Content API requests'",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def parse_context() -> ProviderParseContext:
    return ProviderParseContext(genres=["Supernatural"], formats=["TV"], session={}, history=[])


@pytest.fixture
def make_orchestrator(tmp_path: Path, catalog: list[dict[str, Any]]) -> Callable[[dict[str, Any]], AgentOrchestrator]:
    def build(providers: dict[str, Any]) -> AgentOrchestrator:
        settings = Settings(
            _env_file=None,
            database_url=f"sqlite:///{(tmp_path / 'diagnostics.db').as_posix()}",
            llm_provider="gemini",
        )
        sessions = SQLiteSessionRepository(settings.database_url, retention_days=settings.session_retention_days)
        return AgentOrchestrator(AnimeRecommender(catalog), sessions, providers, settings)

    return build


# ------------------------------------------------ what the provider can see


@pytest.mark.asyncio
async def test_http_failure_carries_the_upstream_status_and_error():
    """ "HTTP 400" alone cannot distinguish a bad key from a stale model name."""
    provider = gemini_against(bad_key_response)
    with pytest.raises(ProviderUnavailable) as caught:
        await provider.parse_intent("recommend something", parse_context())

    diagnostics = caught.value.diagnostics
    assert diagnostics["status_code"] == 400
    assert diagnostics["api_error_code"] == 400
    assert diagnostics["api_error_status"] == "INVALID_ARGUMENT"
    assert "API key not valid" in diagnostics["api_error_message"]
    # The message is unchanged, so nothing that reads it behaves differently.
    assert caught.value.message == "Gemini returned HTTP 400"


@pytest.mark.asyncio
async def test_the_key_is_stripped_from_an_upstream_message_that_quotes_it():
    provider = gemini_against(bad_key_response)
    with pytest.raises(ProviderUnavailable) as caught:
        await provider.parse_intent("recommend something", parse_context())

    message = caught.value.diagnostics["api_error_message"]
    assert API_KEY not in message
    assert "AIza" not in message
    assert "[REDACTED]" in message


@pytest.mark.asyncio
async def test_a_missing_key_is_distinguishable_from_a_rejected_one():
    """The likeliest deployment fault: the secret never reached the container."""
    provider = gemini_against(bad_key_response, api_key=None)
    with pytest.raises(ProviderUnavailable) as caught:
        await provider.parse_intent("recommend something", parse_context())

    assert caught.value.diagnostics == {"reason": "api_key_not_configured"}


# ------------------------------------------------- what the logs then show


@pytest.mark.asyncio
async def test_a_failed_attempt_is_logged_with_the_reason(make_orchestrator, captured):
    orchestrator = make_orchestrator({"gemini": gemini_against(quota_response)})
    response = await orchestrator.respond("Recommend supernatural anime", history=[], session_id=None, debug=False)

    attempt = captured.events("agent_provider_attempt")[0]
    assert attempt["phase"] == "intent"
    assert attempt["provider"] == "gemini"
    assert attempt["model"] == "gemini-3.1-flash-lite"

    failure = captured.events("agent_provider_failure")[0]
    assert failure["provider"] == "gemini"
    assert failure["exception_class"] == "ProviderUnavailable"
    assert failure["status_code"] == 429
    assert failure["api_error_code"] == 429
    assert failure["api_error_status"] == "RESOURCE_EXHAUSTED"
    assert failure["failover_entered"] is False
    assert failure["elapsed_ms"] >= 0

    completed = captured.events("agent_completed")[0]
    assert completed["mode"] == response["mode"]


@pytest.mark.asyncio
async def test_quota_failure_uses_catalog_without_another_llm(make_orchestrator, captured):
    orchestrator = make_orchestrator({"gemini": gemini_against(quota_response)})
    response = await orchestrator.respond("Recommend supernatural anime", history=[], session_id=None, debug=False)

    failure = next(f for f in captured.events("agent_provider_failure") if f["provider"] == "gemini")
    assert failure["failover_entered"] is False
    assert not captured.events("agent_provider_success")
    assert any(step["tool"] == "recommend_anime" and step["result"]["results"] for step in response["trace"])


@pytest.mark.asyncio
async def test_a_successful_attempt_is_logged_with_its_duration(make_orchestrator, captured):
    orchestrator = make_orchestrator({"gemini": StubProvider("gemini")})
    response = await orchestrator.respond("Recommend supernatural anime", history=[], session_id=None, debug=False)

    successes = captured.events("agent_provider_success")
    assert "intent" in [entry["phase"] for entry in successes]
    assert all(entry["elapsed_ms"] >= 0 for entry in successes)
    assert not captured.events("agent_provider_failure")

    completed = captured.events("agent_completed")[0]
    assert completed["mode"] == response["mode"]
    if str(response["mode"]).endswith("_grounded"):
        assert "response" in [entry["phase"] for entry in successes]


# ---------------------------------------------------- secrets stay unlogged


@pytest.mark.asyncio
async def test_no_log_record_contains_the_api_key(make_orchestrator, captured):
    """End to end: a real provider, a real orchestrator, a key-quoting error."""
    orchestrator = make_orchestrator({"gemini": gemini_against(bad_key_response)})
    await orchestrator.respond("Recommend supernatural anime", history=[], session_id=None, debug=False)

    assert captured.rendered, "nothing was logged, so this proves nothing"
    assert API_KEY not in captured.text
    assert "AIza" not in captured.text
    assert "[REDACTED]" in captured.events("agent_provider_failure")[0]["api_error_message"]


@pytest.mark.asyncio
async def test_only_allowlisted_diagnostics_reach_a_log_record(make_orchestrator, captured):
    """A careless provider must not be able to leak through this path.

    The orchestrator copies an allowlist of keys out of the exception, so
    anything a future provider attaches -- headers, prompts, whole responses --
    is dropped rather than logged.
    """
    reckless = ProviderUnavailable(
        "gemini",
        "Gemini returned HTTP 400",
        diagnostics={
            "status_code": 400,
            "request_headers": {"x-goog-api-key": API_KEY},
            "prompt": "the entire user prompt",
        },
    )
    orchestrator = make_orchestrator({"gemini": StubProvider("gemini", parse_failure=reckless)})
    await orchestrator.respond("Recommend supernatural anime", history=[], session_id=None, debug=False)

    failure = captured.events("agent_provider_failure")[0]
    assert failure["status_code"] == 400
    assert "request_headers" not in failure
    assert "prompt" not in failure
    assert API_KEY not in captured.text


def test_the_formatter_redacts_credential_named_fields():
    """Belt to the provider's braces, for anything logged elsewhere."""
    record = logging.LogRecord("anime_compass.agent", logging.INFO, __file__, 1, "agent_test", None, None)
    record.context = {  # type: ignore[attr-defined]
        "provider": "gemini",
        "api_key": API_KEY,
        "gemini_api_key": API_KEY,
        "authorization": f"Bearer {API_KEY}",
    }
    payload = json.loads(JsonFormatter().format(record))

    assert payload["provider"] == "gemini"
    for field in ("api_key", "gemini_api_key", "authorization"):
        assert payload[field] == "[REDACTED]"
    assert API_KEY not in json.dumps(payload)
