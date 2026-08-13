from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import AgentIntent, EntityRelation, EntityType, IntentPreferenceUpdate

ToolName = Literal[
    "search_anime",
    "rank_catalog",
    "search_entities",
    "resolve_entity",
    "recommend_anime",
    "get_anime_details",
    "anime_details",
    "update_session_preferences",
]


class ToolContractError(ValueError):
    pass


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchAnimeArguments(ToolArguments):
    query: str = Field(default="", max_length=300)
    limit: int = Field(default=5, ge=1, le=50)


class RankCatalogArguments(ToolArguments):
    query: str = Field(default="", max_length=300)
    include_genres: list[str] = Field(default_factory=list, max_length=30)
    exclude_genres: list[str] = Field(default_factory=list, max_length=30)
    formats: list[str] = Field(default_factory=list, max_length=10)
    required_studios: list[str] = Field(default_factory=list, max_length=20)
    min_score: float | None = Field(default=None, ge=0, le=10)
    min_year: int | None = Field(default=None, ge=1900, le=2100)
    max_year: int | None = Field(default=None, ge=1900, le=2100)
    max_episodes: int | None = Field(default=None, gt=0, le=100000)
    excluded_titles: list[str] = Field(default_factory=list, max_length=100)
    sort_by: Literal["score", "popularity", "rank", "members", "start_year", "title"] = "score"
    sort_order: Literal["asc", "desc"] | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class SearchEntitiesArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=300)
    entity_types: list[EntityType] = Field(default_factory=list, max_length=12)
    limit: int = Field(default=10, ge=1, le=20)


class ResolveEntityArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=300)
    entity_type: EntityType
    relation: EntityRelation | None = None


class RequiredEntityConstraintArguments(ToolArguments):
    entity_type: EntityType
    entity_id: int | None = None
    matched_name: str = Field(min_length=1, max_length=200)
    related_anime_ids: list[int] = Field(default_factory=list, max_length=10000)


class RecommendAnimeArguments(ToolArguments):
    reference_titles: list[str] = Field(default_factory=list, max_length=20)
    liked_ids: list[int] = Field(default_factory=list, max_length=50)
    liked_titles: list[str] = Field(default_factory=list, max_length=50)
    excluded_ids: list[int] = Field(default_factory=list, max_length=100)
    excluded_titles: list[str] = Field(default_factory=list, max_length=100)
    seen_titles: list[str] = Field(default_factory=list, max_length=200)
    genres: list[str] = Field(default_factory=list, max_length=30)
    include_genres: list[str] = Field(default_factory=list, max_length=30)
    exclude_genres: list[str] = Field(default_factory=list, max_length=30)
    formats: list[str] = Field(default_factory=list, max_length=10)
    media_type: str | None = Field(default=None, max_length=40)
    min_score: float | None = Field(default=None, ge=0, le=10)
    min_year: int | None = Field(default=None, ge=1900, le=2100)
    max_year: int | None = Field(default=None, ge=1900, le=2100)
    max_episodes: int | None = Field(default=None, gt=0, le=100000)
    query: str = Field(default="", max_length=2000)
    free_text_preferences: str = Field(default="", max_length=1200)
    required_studios: list[str] = Field(default_factory=list, max_length=20)
    preferred_studios: list[str] = Field(default_factory=list, max_length=20)
    required_staff: list[str] = Field(default_factory=list, max_length=20)
    preferred_staff: list[str] = Field(default_factory=list, max_length=20)
    required_characters: list[str] = Field(default_factory=list, max_length=20)
    preferred_characters: list[str] = Field(default_factory=list, max_length=20)
    required_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    required_voice_actor_ids: list[int] = Field(default_factory=list, max_length=20)
    preferred_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    required_entity_constraints: list[RequiredEntityConstraintArguments] = Field(
        default_factory=list,
        max_length=30,
    )
    novelty_preference: Literal["neutral", "less_famous", "mainstream"] = "neutral"
    exclude_related_series: bool = True
    one_per_series: bool = True
    top_k: int = Field(default=10, ge=1, le=50)
    session_id: str | None = Field(default=None, max_length=128)
    ignore_session_preferences: bool = False
    diversity_strength: float = Field(default=0.12, ge=0, le=1)


class GetAnimeDetailsArguments(ToolArguments):
    anime_id: int = Field(gt=0)
    field: str | None = Field(default=None, max_length=80)


class UpdateSessionPreferencesArguments(IntentPreferenceUpdate):
    session_id: str | None = Field(default=None, max_length=128)
    seen_titles: list[str] = Field(default_factory=list, max_length=1000)
    preferred_studios: list[str] = Field(default_factory=list, max_length=100)
    preferred_staff: list[str] = Field(default_factory=list, max_length=100)
    preferred_characters: list[str] = Field(default_factory=list, max_length=100)
    preferred_voice_actors: list[str] = Field(default_factory=list, max_length=100)
    temporary_ratings: dict[str, float] = Field(default_factory=dict)
    previous_reference_titles: list[str] = Field(default_factory=list, max_length=50)
    last_recommendations: list[str] = Field(default_factory=list, max_length=50)
    last_recommendation_intent: dict[str, Any] = Field(default_factory=dict)
    reset: bool = False


class ValidatedToolCall(BaseModel):
    tool: ToolName
    arguments: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class ToolPlan(BaseModel):
    intent: str
    primary_tool: ToolName | None
    prerequisite_tools: list[ToolName] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class CatalogToolRegistry:
    argument_models: dict[str, type[BaseModel]] = {
        "search_anime": SearchAnimeArguments,
        "rank_catalog": RankCatalogArguments,
        "search_entities": SearchEntitiesArguments,
        "resolve_entity": ResolveEntityArguments,
        "recommend_anime": RecommendAnimeArguments,
        "get_anime_details": GetAnimeDetailsArguments,
        "anime_details": GetAnimeDetailsArguments,
        "update_session_preferences": UpdateSessionPreferencesArguments,
    }
    allowed_tools: dict[str, set[str]] = {
        "recommend": {
            "search_entities",
            "resolve_entity",
            "recommend_anime",
            "get_anime_details",
            "update_session_preferences",
        },
        "rank_catalog": {"rank_catalog"},
        "details": {"search_entities", "resolve_entity", "get_anime_details", "anime_details"},
        "search": {"search_anime", "search_entities", "resolve_entity"},
        "update_preferences": {"update_session_preferences"},
        "conversation": set(),
    }

    def plan(self, intent: AgentIntent) -> ToolPlan:
        primary: ToolName | None
        prerequisites: list[ToolName] = []
        if intent.intent == "recommend":
            primary = "recommend_anime"
            if intent.required_voice_actors:
                prerequisites.append("search_entities")
            if intent.entity_mentions or intent.reference_titles:
                prerequisites.append("resolve_entity")
        elif intent.intent == "rank_catalog":
            primary = "rank_catalog"
        elif intent.intent == "details":
            primary = "get_anime_details"
            prerequisites.append("resolve_entity")
        elif intent.intent == "search":
            primary = "search_entities"
        elif intent.intent == "update_preferences":
            primary = "update_session_preferences"
        else:
            primary = None
        return ToolPlan(
            intent=intent.intent,
            primary_tool=primary,
            prerequisite_tools=list(dict.fromkeys(prerequisites)),
        )

    def validate_trace(
        self,
        intent: AgentIntent,
        trace: list[dict[str, Any]],
        *,
        response_mode: str,
    ) -> list[ValidatedToolCall]:
        allowed = self.allowed_tools[intent.intent]
        validated: list[ValidatedToolCall] = []
        for index, step in enumerate(trace):
            if not isinstance(step, dict):
                raise ToolContractError(f"Tool trace item {index} is not an object")
            tool = str(step.get("tool") or "")
            if tool not in self.argument_models:
                raise ToolContractError(f"Unknown catalog tool: {tool or '<empty>'}")
            if tool not in allowed:
                raise ToolContractError(f"Tool {tool} is not allowed for {intent.intent} intent")
            arguments = step.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ToolContractError(f"Arguments for {tool} must be an object")
            try:
                parsed = self.argument_models[tool].model_validate(arguments)
            except ValueError as exc:
                raise ToolContractError(f"Invalid arguments for {tool}") from exc
            validated.append(ValidatedToolCall(tool=tool, arguments=parsed.model_dump(exclude_none=True)))

        plan = self.plan(intent)
        executed = {call.tool for call in validated}
        primary = plan.primary_tool
        unresolved_constraint = response_mode == "catalog_constraint_error" and bool(
            {"search_entities", "resolve_entity"}.intersection(executed)
        )
        unresolved_details = intent.intent == "details" and "search_entities" in executed
        if primary and primary not in executed and not unresolved_constraint and not unresolved_details:
            raise ToolContractError(f"Expected {primary} was not executed for {intent.intent} intent")
        if intent.intent == "conversation" and validated:
            raise ToolContractError("Conversation intent cannot execute catalog tools")
        return validated
