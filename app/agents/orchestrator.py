from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.repositories.session_repository import SQLiteSessionRepository
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.ollama_client import OllamaUnavailable
from backend.anime_agent.recommender import AnimeRecommender

from .base import AgentProvider
from .replan import RETRIEVAL_INTENTS, replan_until_results, result_count
from .schemas import AgentIntent, ProviderParseContext
from .tools import CatalogToolRegistry, ToolContractError, ValidatedToolCall

logger = logging.getLogger("anime_compass.agent")


class DeterministicClient:
    """Stand-in provider for the legacy agent's deterministic paths.

    `is_available()` is always False, so the agent never reaches `chat`; the
    method exists to satisfy the ChatClient protocol and fails loudly if the
    guard is ever removed.
    """

    model = "deterministic"
    base_url = "local"

    @staticmethod
    def is_available() -> bool:
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise OllamaUnavailable("The deterministic client cannot generate text.")


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None


class AgentOrchestrator:
    def __init__(
        self,
        recommender: AnimeRecommender,
        sessions: SQLiteSessionRepository,
        providers: dict[str, AgentProvider],
        settings: Settings,
    ):
        self.recommender = recommender
        self.sessions = sessions
        self.providers = providers
        self.settings = settings
        self.circuits = {name: CircuitState() for name in providers}
        self.tool_registry = CatalogToolRegistry()
        self.legacy_agent = AnimeAgent(
            recommender,
            client=DeterministicClient(),
            get_session_profile=sessions.get,
            update_session_preferences=sessions.update,
        )

    async def respond(
        self,
        message: str,
        *,
        history: list[dict[str, str]],
        session_id: str | None,
        debug: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        history = history[-self.settings.max_conversation_history :]
        session = await asyncio.to_thread(self.sessions.get, session_id)
        context = ProviderParseContext(
            genres=self.recommender.meta()["genres"],
            formats=self.recommender.meta()["types"],
            session=session,
            history=history,
        )
        selected_provider: AgentProvider | None = None
        response_provider: AgentProvider | None = None
        parser_errors: list[str] = []
        provider_attempts: list[dict[str, Any]] = []
        failed_providers: set[str] = set()
        parser_mode = "rule_fallback"
        agent_intent: AgentIntent | None = None
        intent = None

        parse_started = time.perf_counter()
        for provider_name in self._provider_order():
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            if not self._circuit_allows(provider_name):
                provider_attempts.append({"phase": "intent", "provider": provider_name, "outcome": "circuit_open"})
                continue
            try:
                agent_intent = await asyncio.wait_for(
                    provider.parse_intent(message, context),
                    timeout=self.settings.llm_timeout_seconds,
                )
                intent = agent_intent.to_legacy()
                selected_provider = provider
                parser_mode = "structured_llm"
                provider_attempts.append({"phase": "intent", "provider": provider_name, "outcome": "success"})
                break
            except Exception as exc:
                error_type = type(exc).__name__
                error_detail = self._safe_error_detail(exc)
                parser_errors.append(f"{provider_name} intent: {error_detail}")
                provider_attempts.append(
                    {
                        "phase": "intent",
                        "provider": provider_name,
                        "outcome": "failed",
                        "error_type": error_type,
                        "error_detail": error_detail,
                    }
                )
                failed_providers.add(provider_name)
                self._record_failure(provider_name)

        if intent is None:
            intent = await asyncio.to_thread(self.legacy_agent._rule_based_intent, message, history)
        assert intent is not None
        if agent_intent is None:
            agent_intent = AgentIntent.from_legacy(intent)
        parse_ms = (time.perf_counter() - parse_started) * 1000

        response, agent_intent, validated_calls = await self._execute_catalog_tools(
            agent_intent,
            intent,
            message=message,
            history=history,
            session_id=session_id,
            session=session,
            parser_mode=parser_mode,
            provider_attempts=provider_attempts,
            parser_errors=parser_errors,
        )
        if provider_attempts and provider_attempts[-1].get("phase") == "tool_routing_fallback":
            parser_mode = "rule_fallback"
        generation_ms = 0.0
        if (
            selected_provider is not None
            and response.get("trace")
            and agent_intent.intent in {"recommend", "rank_catalog", "search", "details"}
        ):
            generation_started = time.perf_counter()
            generation_order = [
                selected_provider.name,
                *[
                    name
                    for name in self._provider_order()
                    if name != selected_provider.name and name not in failed_providers
                ],
            ]
            verified_data = {
                "mode": response.get("mode"),
                "validated_intent": agent_intent.model_dump(mode="json"),
                "validated_tool_calls": [call.model_dump(mode="json") for call in validated_calls],
                "verified_tool_trace": response.get("trace", []),
                "deterministic_fallback": response.get("answer", ""),
            }
            for provider_name in generation_order:
                provider = self.providers.get(provider_name)
                if provider is None:
                    continue
                if not self._circuit_allows(provider_name):
                    provider_attempts.append(
                        {"phase": "response", "provider": provider_name, "outcome": "circuit_open"}
                    )
                    continue
                try:
                    candidate = await asyncio.wait_for(
                        provider.generate_tool_response(message, verified_data),
                        timeout=self.settings.llm_timeout_seconds,
                    )
                    if not self._validated_generated_answer(candidate, response):
                        raise ValueError("grounded response validation failed")
                    response["answer"] = candidate
                    response["mode"] = f"{provider.name}_grounded"
                    response_provider = provider
                    provider_attempts.append({"phase": "response", "provider": provider_name, "outcome": "success"})
                    self._record_success(provider_name)
                    break
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_detail = self._safe_error_detail(exc)
                    parser_errors.append(f"{provider_name} response: {error_detail}")
                    provider_attempts.append(
                        {
                            "phase": "response",
                            "provider": provider_name,
                            "outcome": "failed",
                            "error_type": error_type,
                            "error_detail": error_detail,
                        }
                    )
                    failed_providers.add(provider_name)
                    if isinstance(exc, ValueError) and str(exc) == "grounded response validation failed":
                        self._record_success(provider_name)
                    else:
                        self._record_failure(provider_name)
            generation_ms = (time.perf_counter() - generation_started) * 1000
        elif selected_provider is not None:
            self._record_success(selected_provider.name)

        internal_debug = response.pop("_debug", {})
        active_provider = response_provider or selected_provider
        used_deterministic_response = bool(response.get("trace")) and response_provider is None
        fallback_used = (
            active_provider is None or active_provider.name != self.settings.llm_provider or used_deterministic_response
        )
        response["agent"] = {
            "provider": active_provider.name if active_provider else "deterministic",
            "model": active_provider.model if active_provider else None,
            "available": active_provider is not None,
        }
        if debug:
            diagnostics = {
                "llm_provider": active_provider.name if active_provider else None,
                "intent_provider": selected_provider.name if selected_provider else None,
                "response_provider": response_provider.name if response_provider else None,
                "fallback_used": fallback_used,
                "parser_mode": parser_mode,
                "model_name": active_provider.model if active_provider else None,
                "provider_errors": parser_errors,
                "provider_attempts": provider_attempts,
                "circuit_breakers": {
                    name: {
                        "failures": state.failures,
                        "open": state.failures >= self.settings.provider_failure_threshold,
                    }
                    for name, state in self.circuits.items()
                },
                "validated_intent": agent_intent.model_dump(mode="json"),
                "tool_plan": self.tool_registry.plan(agent_intent).model_dump(mode="json"),
                "tool_calls": [call.tool for call in validated_calls],
                "timing_ms": {
                    "intent_parsing": round(parse_ms, 2),
                    "explanation_generation": round(generation_ms, 2),
                    "total": round((time.perf_counter() - started) * 1000, 2),
                },
                **internal_debug,
            }
            response["debug"] = diagnostics
            response["parser_mode"] = parser_mode
            response["validated_intent"] = agent_intent.model_dump(mode="json")
        await asyncio.to_thread(
            self.sessions.log_event,
            session_id,
            "tool",
            "agent_tools_executed",
            {"tools": [call.tool for call in validated_calls]},
        )
        logger.info(
            "agent_completed",
            extra={
                "context": {
                    "provider": selected_provider.name if selected_provider else "deterministic",
                    "parser_mode": parser_mode,
                    "fallback_used": fallback_used,
                    "tool_names": [call.tool for call in validated_calls],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            },
        )
        return response

    async def _execute_catalog_tools(
        self,
        agent_intent: AgentIntent,
        legacy_intent: Any,
        *,
        message: str,
        history: list[dict[str, str]],
        session_id: str | None,
        session: dict[str, Any],
        parser_mode: str,
        provider_attempts: list[dict[str, Any]],
        parser_errors: list[str],
    ) -> tuple[dict[str, Any], AgentIntent, list[ValidatedToolCall]]:
        # The first execution uses the legacy intent unchanged, so the
        # non-replanning path is byte-for-byte what it was before replanning
        # existed. Only relaxed candidates go through the schema round-trip.
        response = await asyncio.to_thread(
            self.legacy_agent._respond_from_intent,
            legacy_intent,
            message,
            history,
            session_id,
            session,
        )
        validated_intent = AgentIntent.from_legacy(legacy_intent)

        def execute(candidate: AgentIntent, relaxed_fields: frozenset[str]) -> dict[str, Any]:
            return self.legacy_agent._respond_from_intent(
                candidate.to_legacy(),
                message,
                history,
                session_id,
                session,
                relaxed_fields,
            )

        relaxations: list[dict[str, Any]] = []
        if (
            self.settings.max_replan_steps > 0
            and validated_intent.intent in RETRIEVAL_INTENTS
            and result_count(response) == 0
        ):
            replanned_intent, replanned_response, steps = await asyncio.to_thread(
                replan_until_results,
                validated_intent,
                execute,
                max_steps=self.settings.max_replan_steps,
                initial_response=response,
            )
            if steps:
                relaxations = [step.as_dict() for step in steps]
                # Only adopt a relaxed plan that actually recovered candidates.
                if result_count(replanned_response) > 0:
                    validated_intent = replanned_intent
                    legacy_intent = replanned_intent.to_legacy()
                    response = replanned_response
                provider_attempts.append(
                    {
                        "phase": "replan",
                        "provider": "backend",
                        "outcome": "recovered" if result_count(response) > 0 else "exhausted",
                        "steps": len(steps),
                    }
                )
        if relaxations:
            response["relaxations"] = relaxations
        try:
            calls = self.tool_registry.validate_trace(
                validated_intent,
                response.get("trace", []),
                response_mode=str(response.get("mode") or ""),
            )
            return response, validated_intent, calls
        except ToolContractError as exc:
            if parser_mode != "structured_llm":
                raise
            parser_errors.append(f"backend tool routing: {exc}")
            provider_attempts.append(
                {
                    "phase": "tool_routing",
                    "provider": "backend",
                    "outcome": "failed",
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc)[:160],
                }
            )

        fallback_intent = await asyncio.to_thread(self.legacy_agent._rule_based_intent, message, history)
        fallback_response = await asyncio.to_thread(
            self.legacy_agent._respond_from_intent,
            fallback_intent,
            message,
            history,
            session_id,
            session,
        )
        validated_fallback = AgentIntent.from_legacy(fallback_intent)
        fallback_calls = self.tool_registry.validate_trace(
            validated_fallback,
            fallback_response.get("trace", []),
            response_mode=str(fallback_response.get("mode") or ""),
        )
        provider_attempts.append(
            {
                "phase": "tool_routing_fallback",
                "provider": "backend",
                "outcome": "success",
            }
        )
        return fallback_response, validated_fallback, fallback_calls

    def _validated_generated_answer(self, content: str, response: dict[str, Any]) -> bool:
        recommendation_step = next(
            (step for step in response.get("trace", []) if step.get("tool") in {"recommend_anime", "rank_catalog"}),
            None,
        )
        if recommendation_step:
            results = recommendation_step.get("result", {}).get("results", [])
            titles = [
                str(value)
                for value in recommendation_step.get("result", {}).get(
                    "result_titles",
                    [],
                )
                if value
            ] or [str(item.get("title")) for item in results if item.get("title")]
            arguments = recommendation_step.get("arguments", {})
            return self.legacy_agent._valid_recommendation_answer(
                content,
                titles,
                [str(value) for value in arguments.get("excluded_titles", [])],
                int(arguments.get("top_k") or len(titles)),
                (
                    [str(value) for value in arguments.get("required_voice_actors", [])]
                    if recommendation_step.get("tool") == "recommend_anime"
                    else []
                ),
            )
        details_step = next(
            (step for step in response.get("trace", []) if step.get("tool") in {"get_anime_details", "anime_details"}),
            None,
        )
        if details_step:
            title = str(details_step.get("result", {}).get("result", {}).get("title") or "")
            return bool(title) and self.legacy_agent._valid_introduction_answer(content, title)
        search_step = next(
            (step for step in response.get("trace", []) if step.get("tool") in {"search_anime", "search_entities"}),
            None,
        )
        if search_step:
            results = search_step.get("result", {}).get("results", [])
            names = [str(item.get("matched_name") or item.get("title") or "") for item in results]
            names = [name for name in names if name]
            if not names:
                return bool(content.strip())
            content_key = content.casefold()
            return any(name.casefold() in content_key for name in names)
        return False

    @staticmethod
    def _safe_error_detail(exc: Exception) -> str:
        if isinstance(exc, ProviderUnavailable):
            return exc.message[:160]
        if isinstance(exc, TimeoutError):
            return "Provider request timed out"
        return type(exc).__name__

    async def health(self) -> dict[str, Any]:
        checks = await asyncio.gather(
            *(provider.health_check() for provider in self.providers.values()),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        for name, check in zip(self.providers, checks, strict=True):
            if isinstance(check, BaseException):
                result[name] = {"provider": name, "available": False, "detail": type(check).__name__}
            else:
                result[name] = check.model_dump()
        return result

    async def close(self) -> None:
        await asyncio.gather(*(provider.close() for provider in self.providers.values()), return_exceptions=True)

    def _provider_order(self) -> list[str]:
        if self.settings.llm_provider == "gemini":
            return ["gemini", "ollama"]
        return ["ollama", "gemini"]

    def _circuit_allows(self, provider_name: str) -> bool:
        state = self.circuits[provider_name]
        if state.failures < self.settings.provider_failure_threshold:
            return True
        if state.opened_at is None:
            return False
        if time.monotonic() - state.opened_at >= self.settings.provider_cooldown_seconds:
            state.failures = 0
            state.opened_at = None
            return True
        return False

    def _record_failure(self, provider_name: str) -> None:
        state = self.circuits[provider_name]
        state.failures += 1
        if state.failures >= self.settings.provider_failure_threshold:
            state.opened_at = time.monotonic()

    def _record_success(self, provider_name: str) -> None:
        self.circuits[provider_name] = CircuitState()
