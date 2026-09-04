from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from app.core.errors import ProviderUnavailable

from .prompting import render_intent_prompt, render_response_prompt
from .runtime_state import build_runtime_context, tool_observation_from_verified
from .schemas import AgentIntent, ProviderHealth, ProviderParseContext


class GeminiAgentProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout: float,
        max_output_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def parse_intent(self, message: str, context: ProviderParseContext) -> AgentIntent:
        prompt = self._intent_prompt(message, context)
        content = await self._generate(
            prompt,
            response_schema=AgentIntent.provider_json_schema(),
            response_mime_type="application/json",
        )
        try:
            return AgentIntent.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderUnavailable(self.name, "Gemini returned invalid structured intent") from exc

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return await self._generate(
            render_response_prompt(
                user_message,
                tool_observation_from_verified(verified_tool_data),
                runtime=verified_tool_data.get("runtime_context"),
            )
        )

    async def generate_explanation(
        self,
        user_message: str,
        verified_recommendation: dict[str, Any],
    ) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(
                provider=self.name, model=self.model, available=False, detail="API key is not configured"
            )
        try:
            response = await self.client.get(
                f"{self.base_url}/models/{self.model}",
                headers=self._headers(),
                timeout=3.0,
            )
            if not response.is_success:
                return ProviderHealth(
                    provider=self.name,
                    model=self.model,
                    available=False,
                    detail=f"HTTP {response.status_code}",
                )
            data = response.json()
            methods = data.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                return ProviderHealth(
                    provider=self.name,
                    model=self.model,
                    available=False,
                    detail="Configured model does not support generateContent",
                )
            return ProviderHealth(
                provider=self.name,
                model=self.model,
                available=True,
            )
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            return ProviderHealth(provider=self.name, model=self.model, available=False, detail=type(exc).__name__)

    async def close(self) -> None:
        await self.client.aclose()

    async def _generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        if not self.api_key:
            raise ProviderUnavailable(self.name, "Gemini API key is not configured")
        generation_config: dict[str, Any] = {
            "temperature": 0.15,
            "maxOutputTokens": self.max_output_tokens,
        }
        if response_schema:
            generation_config["responseMimeType"] = response_mime_type or "application/json"
            generation_config["responseJsonSchema"] = response_schema
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            content = "".join(str(part.get("text") or "") for part in parts).strip()
            if not content:
                raise ValueError("empty Gemini response")
            return content
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailable(
                self.name,
                f"Gemini returned HTTP {exc.response.status_code}",
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable(self.name, f"Gemini request failed: {type(exc).__name__}") from exc

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key or ""}

    @staticmethod
    def _intent_prompt(message: str, context: ProviderParseContext) -> str:
        """Static policy plus this request's runtime state. No rules live here."""
        runtime = build_runtime_context(
            catalog_genres=context.genres,
            catalog_formats=context.formats,
            session=context.session,
        )
        return render_intent_prompt(message, runtime, history=context.history)
