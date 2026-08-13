from __future__ import annotations

import json

import httpx
import pytest

from app.agents.gemini_provider import GeminiAgentProvider
from app.agents.schemas import ProviderHealth, ProviderParseContext
from app.core.errors import ProviderUnavailable


def gemini_response(intent: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": json.dumps(intent)}]}}]},
    )


@pytest.mark.asyncio
async def test_gemini_structured_intent_and_backend_only_key_header() -> None:
    observed: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["key"] = request.headers.get("x-goog-api-key", "")
        observed["url"] = str(request.url)
        return gemini_response(
            {
                "intent": "recommend",
                "required_voice_actors": ["Matsuoka, Yoshitsugu"],
                "top_k": 7,
            }
        )

    provider = GeminiAgentProvider(
        api_key="private-test-key",
        model="test-model",
        base_url="https://example.invalid/v1beta",
        timeout=2,
        max_output_tokens=500,
        transport=httpx.MockTransport(handler),
    )
    intent = await provider.parse_intent(
        "recommend seven anime with Matsuoka",
        ProviderParseContext(genres=[], formats=[]),
    )
    await provider.close()
    assert intent.required_voice_actors == ["Matsuoka, Yoshitsugu"]
    assert observed["key"] == "private-test-key"
    assert "private-test-key" not in observed["url"]


@pytest.mark.asyncio
async def test_invalid_gemini_output_raises_provider_unavailable() -> None:
    provider = GeminiAgentProvider(
        api_key="test-key",
        model="test-model",
        base_url="https://example.invalid/v1beta",
        timeout=2,
        max_output_tokens=500,
        transport=httpx.MockTransport(lambda request: gemini_response({"intent": "not-valid"})),
    )
    with pytest.raises(ProviderUnavailable):
        await provider.parse_intent("recommend", ProviderParseContext(genres=[], formats=[]))
    await provider.close()


@pytest.mark.asyncio
async def test_gemini_timeout_is_wrapped_without_key_leak() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = GeminiAgentProvider(
        api_key="never-log-this",
        model="test-model",
        base_url="https://example.invalid/v1beta",
        timeout=0.01,
        max_output_tokens=500,
        transport=httpx.MockTransport(timeout),
    )
    with pytest.raises(ProviderUnavailable) as captured:
        await provider.parse_intent("recommend", ProviderParseContext(genres=[], formats=[]))
    await provider.close()
    assert "never-log-this" not in str(captured.value)


@pytest.mark.asyncio
async def test_gemini_health_requires_generate_content_support() -> None:
    def supported(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"supportedGenerationMethods": ["generateContent"]})

    def unsupported(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"supportedGenerationMethods": ["embedContent"]})

    async def check(transport: httpx.MockTransport) -> ProviderHealth:
        provider = GeminiAgentProvider(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1beta",
            timeout=2,
            max_output_tokens=500,
            transport=transport,
        )
        health = await provider.health_check()
        await provider.close()
        return health

    available = await check(httpx.MockTransport(supported))
    unavailable = await check(httpx.MockTransport(unsupported))

    assert available.available is True
    assert unavailable.available is False
    assert unavailable.detail == "Configured model does not support generateContent"


@pytest.mark.asyncio
async def test_gemini_http_error_is_wrapped_without_response_body_or_key() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "secret response details"}},
            request=request,
        )

    provider = GeminiAgentProvider(
        api_key="never-log-this",
        model="retired-model",
        base_url="https://example.invalid/v1beta",
        timeout=2,
        max_output_tokens=500,
        transport=httpx.MockTransport(unavailable),
    )
    with pytest.raises(ProviderUnavailable) as captured:
        await provider.parse_intent("recommend", ProviderParseContext(genres=[], formats=[]))
    await provider.close()

    message = str(captured.value)
    assert "HTTP 404" in message
    assert "secret response details" not in message
    assert "never-log-this" not in message
