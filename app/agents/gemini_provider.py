from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.errors import ProviderUnavailable

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
        prompt = (
            "You are Anime Compass, a warm and concise anime guide. Answer using only the verified "
            "catalog JSON supplied below. The backend has already selected and executed every tool. "
            "Never add, replace, or infer titles, people, scores, episode counts, dates, studios, roles, "
            "or relationships that are absent from that JSON. For recommendations, list exactly the "
            "verified result titles and explain each choice with user-facing catalog evidence such as "
            "premise, themes, cast, staff, format, or constraints. Do not mention embeddings, TF-IDF, "
            "channels, hybrid scores, JSON, tools, or internal implementation details. For anime details, "
            "give a spoiler-light introduction. If the verified result is empty, say that no catalog "
            "matches satisfied all constraints. Lead with the answer and end with one useful next step "
            "only when natural.\n\n"
            f"User request:\n{user_message}\n\nVerified catalog JSON:\n"
            + json.dumps(verified_tool_data, ensure_ascii=False)[:18_000]
        )
        return await self._generate(prompt)

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
        return f"""You are the English intent parser for Anime Compass.
Return only one JSON object matching the supplied schema. Do not answer the user, choose anime,
invent catalog facts, or emit a tool call. The backend maps this validated intent to catalog tools.

Intent rules:
- recommend: the user wants titles to watch or asks for something similar.
- rank_catalog: the user requests an exact catalog ordering such as highest-scored
  Gundam TV anime or most-popular romance movies. Set catalog_query, rank_by, and
  sort_order; this operation ignores session taste.
- search: the user wants to find or identify catalog entities without asking for recommendations.
- details: the user asks for an introduction, premise, cast, staff, or facts about one anime.
- update_preferences: the user only reports likes, dislikes, watched titles, or exclusions.
- conversation: no catalog operation is needed.
- Use reference_result_indices and watched_result_indices for phrases such as "the second result".
- Keep named anime examples in reference_titles. Put watched examples in seen_titles and the
  preference update. Preserve every explicit exclusion and numeric constraint.
- Use canonical catalog genres and formats only. Put mood, premise, atmosphere, and other prose in
  free_text_preferences.

Entity rules:
- anime from/by/featuring a named studio -> required_studios
- anime by/with a named staff member -> required_staff
- anime with a named character -> required_characters
- anime with/voiced by/featuring/"has them involved" for a named voice actor -> required_voice_actors
- anime produced by a named producer -> add a producer entity mention with relation direct
- anime by a named director or original creator -> add the role-specific entity mention with relation direct
- anime with a named theme or demographic -> add a theme or demographic entity mention with relation direct
- General admiration without a membership requirement may use the corresponding preferred field.
- Required entity fields are hard filters and must override unrelated session preferences.
- For explicit recommendation relationships, always add an entity mention with the most specific type;
  the backend resolves its catalog ID and applies its related-anime set as a hard constraint.
- For "the director of Monster", add an anime entity mention for Monster with relation director_of;
  never guess the director's name.
- Do not put the same person or studio in both required and preferred fields.

Selection and feedback rules:
- "I enjoyed X" means X is a reference title and is both liked and watched.
- "I do not want X" puts X in excluded_titles; do not put it in reference_titles.
- "More like the fourth, and I watched the second" uses 1-based result indices.
- Explicit instructions in the latest message override conflicting older session context.

Ranking rules:
- "highest rated" and "highest scored" map to rank_by=score, sort_order=desc.
- "most popular" maps to rank_by=popularity, sort_order=asc because lower catalog
  popularity ranks are more popular.
- Put title-family terms such as Gundam in catalog_query; do not put them in reference_titles.

Catalog genres: {json.dumps(context.genres, ensure_ascii=True)}
Catalog formats: {json.dumps(context.formats, ensure_ascii=True)}
Recent session context: {json.dumps(context.session, ensure_ascii=True)[:4000]}
Recent history: {json.dumps(context.history[-12:], ensure_ascii=True)[:4000]}
User message: {message}
"""
