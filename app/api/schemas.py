from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(ApiModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    debug: bool = False


class ChatResponse(BaseModel):
    mode: str
    answer: str
    trace: list[dict[str, Any]] = Field(default_factory=list)
    agent: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None
    parser_mode: str | None = None
    validated_intent: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class HybridChannelScore(ApiModel):
    raw_score: float
    normalized_score: float
    configured_weight: float
    effective_weight: float
    weighted_contribution: float
    active: bool
    inactive_reason: str | None = None


class HybridScoreBreakdown(ApiModel):
    recommendation_mode: str
    channels: dict[str, HybridChannelScore]
    pre_diversity_score: float
    diversity_adjustment: float
    final_score: float


class VoiceActorRole(ApiModel):
    voice_actor: str
    character: str
    language: str


class AnimeRecommendation(BaseModel):
    id: int
    anime_id: int
    title: str
    score: float | None = None
    start_year: int | None = None
    type: str | None = None
    episodes: int | None = None
    genres: list[str] = Field(default_factory=list)
    studios: list[str] = Field(default_factory=list)
    matched_voice_actors: list[str] = Field(default_factory=list)
    voice_actor_roles: list[VoiceActorRole] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    recommendation_mode: str
    score_breakdown: HybridScoreBreakdown
    pre_diversity_score: float
    diversity_adjustment: float
    final_score: float

    model_config = ConfigDict(extra="allow")


class RecommendRequest(ApiModel):
    session_id: str = Field(default="", max_length=128)
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
    query: str | None = Field(default=None, max_length=1200)
    free_text_preferences: str = Field(default="", max_length=1200)
    required_studios: list[str] = Field(default_factory=list, max_length=20)
    preferred_studios: list[str] = Field(default_factory=list, max_length=20)
    required_staff: list[str] = Field(default_factory=list, max_length=20)
    preferred_staff: list[str] = Field(default_factory=list, max_length=20)
    required_characters: list[str] = Field(default_factory=list, max_length=20)
    preferred_characters: list[str] = Field(default_factory=list, max_length=20)
    required_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    preferred_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    novelty_preference: Literal["neutral", "less_famous", "mainstream"] = "neutral"
    exclude_related_series: bool = True
    one_per_series: bool = False
    top_k: int = Field(default=12, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=50)
    diversity_strength: float = Field(default=0.12, ge=0, le=1)
    weights: dict[str, float] | None = None
    session_profile: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "reference_titles",
        "liked_titles",
        "excluded_titles",
        "seen_titles",
        "genres",
        "include_genres",
        "exclude_genres",
        "formats",
        "required_studios",
        "preferred_studios",
        "required_staff",
        "preferred_staff",
        "required_characters",
        "preferred_characters",
        "required_voice_actors",
        "preferred_voice_actors",
    )
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if text and text.casefold() not in seen:
                result.append(text)
                seen.add(text.casefold())
        return result

    @model_validator(mode="after")
    def validate_ranges(self) -> RecommendRequest:
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")
        if self.limit is not None:
            self.top_k = self.limit
        if self.media_type and self.media_type not in self.formats:
            self.formats.append(self.media_type)
        if not self.include_genres and self.genres:
            self.include_genres = list(self.genres)
        return self


class RecommendResponse(ApiModel):
    query: dict[str, Any]
    resolved_titles: list[dict[str, Any]]
    recommendations: list[AnimeRecommendation]
    results: list[AnimeRecommendation]
    model_info: dict[str, Any]
    diagnostics: dict[str, Any]
    timing_ms: float
    cache_hit: bool = False
    message: str | None = None


class SearchRequest(ApiModel):
    query: str = Field(default="", max_length=300)
    top_k: int = Field(default=24, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=10000)
    genres: list[str] = Field(default_factory=list, max_length=30)
    include_genres: list[str] = Field(default_factory=list, max_length=30)
    exclude_genres: list[str] = Field(default_factory=list, max_length=30)
    formats: list[str] = Field(default_factory=list, max_length=10)
    required_studios: list[str] = Field(default_factory=list, max_length=20)
    media_type: str | None = Field(default=None, max_length=40)
    min_score: float | None = Field(default=None, ge=0, le=10)
    min_year: int | None = Field(default=None, ge=1900, le=2100)
    max_year: int | None = Field(default=None, ge=1900, le=2100)
    max_episodes: int | None = Field(default=None, gt=0, le=100000)
    sort_by: Literal["relevance", "score", "popularity", "rank", "members", "start_year", "title"] = "relevance"
    sort_order: Literal["asc", "desc"] | None = None
    semantic: bool = True

    @model_validator(mode="after")
    def normalize_search_filters(self) -> SearchRequest:
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")
        if not self.include_genres and self.genres:
            self.include_genres = list(self.genres)
        if self.media_type and self.media_type not in self.formats:
            self.formats.append(self.media_type)
        return self


class RankRequest(ApiModel):
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
    top_k: int = Field(default=24, ge=1, le=50)

    @model_validator(mode="after")
    def validate_rank_years(self) -> RankRequest:
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")
        return self


class EntitySearchRequest(ApiModel):
    query: str = Field(min_length=1, max_length=300)
    entity_types: list[str] | None = Field(default=None, max_length=12)
    top_k: int = Field(default=10, ge=1, le=20)


class EntitySearchResponse(ApiModel):
    results: list[dict[str, Any]]


class SessionPreferenceUpdate(ApiModel):
    replace: bool = False
    reset: bool = False
    liked_titles: list[str] = Field(default_factory=list, max_length=500)
    disliked_titles: list[str] = Field(default_factory=list, max_length=500)
    seen_titles: list[str] = Field(default_factory=list, max_length=1000)
    watched_titles: list[str] = Field(default_factory=list, max_length=1000)
    excluded_titles: list[str] = Field(default_factory=list, max_length=1000)
    preferred_genres: list[str] = Field(default_factory=list, max_length=100)
    excluded_genres: list[str] = Field(default_factory=list, max_length=100)
    preferred_studios: list[str] = Field(default_factory=list, max_length=100)
    preferred_staff: list[str] = Field(default_factory=list, max_length=100)
    preferred_characters: list[str] = Field(default_factory=list, max_length=100)
    preferred_voice_actors: list[str] = Field(default_factory=list, max_length=100)
    temporary_ratings: dict[str, float] = Field(default_factory=dict, max_length=500)
    previous_reference_titles: list[str] = Field(default_factory=list, max_length=50)
    last_recommendations: list[str] = Field(default_factory=list, max_length=50)
    last_recommendation_intent: dict[str, Any] = Field(default_factory=dict)

    @field_validator("temporary_ratings")
    @classmethod
    def validate_temporary_ratings(
        cls,
        values: dict[str, float],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for title, rating in values.items():
            clean_title = str(title).strip()
            numeric_rating = float(rating)
            if not clean_title:
                raise ValueError("temporary rating titles cannot be empty")
            if not math.isfinite(numeric_rating) or not 1 <= numeric_rating <= 10:
                raise ValueError("temporary ratings must be finite values from 1 to 10")
            result[clean_title] = numeric_rating
        return result


class LegacySessionPreferenceUpdate(SessionPreferenceUpdate):
    session_id: str = Field(min_length=1, max_length=128)


class ComponentHealth(ApiModel):
    status: Literal["healthy", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(ApiModel):
    ok: bool = True
    status: Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, ComponentHealth]
    catalog: dict[str, Any]
    agent: dict[str, Any]


class ErrorDetail(ApiModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponse(ApiModel):
    ok: Literal[False] = False
    error: ErrorDetail
