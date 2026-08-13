from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.errors import ProviderUnavailable

from .gemini_provider import GeminiAgentProvider
from .schemas import AgentIntent, ProviderHealth, ProviderParseContext


class OllamaAgentProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        max_output_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def parse_intent(self, message: str, context: ProviderParseContext) -> AgentIntent:
        prompt = GeminiAgentProvider._intent_prompt(message, context)
        content = await self._chat(
            [{"role": "system", "content": prompt}, {"role": "user", "content": message}],
            json_mode=True,
        )
        try:
            return AgentIntent.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderUnavailable(self.name, "Ollama returned invalid structured intent") from exc

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Anime Compass. Use only the verified catalog JSON. List exactly the "
                        "verified result titles in the supplied order and explain them with user-facing catalog evidence. "
                        "Never invent facts or titles. Do not mention JSON, tools, embeddings, TF-IDF, "
                        "ranking channels, or hybrid scores."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{user_message}\n\nVerified JSON:\n{json.dumps(verified_tool_data, ensure_ascii=False)[:18_000]}",
                },
            ]
        )

    async def generate_explanation(
        self,
        user_message: str,
        verified_recommendation: dict[str, Any],
    ) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=2.0)
            response.raise_for_status()
            payload = response.json()
            installed_models = {
                str(model.get("model") or model.get("name") or "")
                for model in payload.get("models", [])
                if isinstance(model, dict)
            }
            model_available = self.model in installed_models
            return ProviderHealth(
                provider=self.name,
                model=self.model,
                available=model_available,
                detail=None if model_available else "Configured model is not installed",
            )
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            return ProviderHealth(provider=self.name, model=self.model, available=False, detail=type(exc).__name__)

    async def close(self) -> None:
        await self.client.aclose()

    async def _chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": self.max_output_tokens,
            },
        }
        if json_mode:
            payload["format"] = AgentIntent.provider_json_schema()
        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = str(response.json().get("message", {}).get("content", "")).strip()
            if not content:
                raise ValueError("empty Ollama response")
            return content
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise ProviderUnavailable(self.name, f"Ollama request failed: {type(exc).__name__}") from exc
