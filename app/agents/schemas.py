from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.anime_agent.intent import (
    EntityMention,
    PreferenceUpdate,
    StructuredIntent,
)
from backend.anime_agent.intent import (
    InferredConstraint as LegacyInferredConstraint,
)
from backend.anime_agent.recommender import normalize_label

EntityType = Literal[
    "anime",
    "character",
    "voice_actor",
    "staff",
    "director",
    "original_creator",
    "studio",
    "producer",
    "genre",
    "theme",
    "demographic",
    "result_index",
]
EntityRelation = Literal["direct", "reference", "exclude", "watched", "director_of", "related_to"]


def _is_nameable(value: str) -> bool:
    """True when a string could plausibly name a catalog entity.

    Stopwords tokenize to nothing, so a value like "With" or "The" leaves an
    empty label and cannot match anything.
    """
    return bool(normalize_label(value))


class IntentEntityMention(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    entity_type: EntityType = "anime"
    relation: EntityRelation = "direct"
    index: int | None = Field(default=None, ge=1, le=50)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IntentPreferenceUpdate(BaseModel):
    liked_titles: list[str] = Field(default_factory=list, max_length=50)
    disliked_titles: list[str] = Field(default_factory=list, max_length=50)
    watched_titles: list[str] = Field(default_factory=list, max_length=100)
    excluded_titles: list[str] = Field(default_factory=list, max_length=100)
    preferred_genres: list[str] = Field(default_factory=list, max_length=30)
    excluded_genres: list[str] = Field(default_factory=list, max_length=30)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IntentInferredConstraint(BaseModel):
    field: Literal["max_episodes", "min_score", "min_year", "max_year"]
    value: float
    confidence: float = Field(ge=0, le=1)
    source_text: str = Field(min_length=1, max_length=200)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentIntent(BaseModel):
    intent: Literal["recommend", "rank_catalog", "search", "details", "update_preferences", "conversation"]
    catalog_query: str = Field(default="", max_length=300)
    rank_by: Literal["score", "popularity", "rank", "members", "start_year", "title"] = "score"
    sort_order: Literal["asc", "desc"] | None = None
    reference_titles: list[str] = Field(default_factory=list, max_length=20)
    entity_mentions: list[IntentEntityMention] = Field(default_factory=list, max_length=30)
    include_genres: list[str] = Field(default_factory=list, max_length=30)
    exclude_genres: list[str] = Field(default_factory=list, max_length=30)
    required_studios: list[str] = Field(default_factory=list, max_length=20)
    preferred_studios: list[str] = Field(default_factory=list, max_length=20)
    required_staff: list[str] = Field(default_factory=list, max_length=20)
    preferred_staff: list[str] = Field(default_factory=list, max_length=20)
    required_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    preferred_voice_actors: list[str] = Field(default_factory=list, max_length=20)
    required_characters: list[str] = Field(default_factory=list, max_length=20)
    preferred_characters: list[str] = Field(default_factory=list, max_length=20)
    formats: list[str] = Field(default_factory=list, max_length=10)
    min_score: float | None = Field(default=None, ge=0, le=10)
    min_year: int | None = Field(default=None, ge=1900, le=2100)
    max_year: int | None = Field(default=None, ge=1900, le=2100)
    max_episodes: int | None = Field(default=None, gt=0)
    excluded_titles: list[str] = Field(default_factory=list, max_length=100)
    seen_titles: list[str] = Field(default_factory=list, max_length=100)
    exclude_related_series: bool = True
    one_per_series: bool = True
    top_k: int = Field(default=10, ge=1, le=50)
    free_text_preferences: str = Field(default="", max_length=1200)
    novelty_preference: Literal["neutral", "less_famous", "mainstream"] = "neutral"
    reference_result_indices: list[int] = Field(default_factory=list, max_length=20)
    watched_result_indices: list[int] = Field(default_factory=list, max_length=20)
    inferred_constraints: list[IntentInferredConstraint] = Field(default_factory=list, max_length=8)
    preference_update: IntentPreferenceUpdate = Field(default_factory=IntentPreferenceUpdate)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_conversation_intent(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().casefold() == "conversational":
            return "conversation"
        return value

    @field_validator("reference_result_indices", "watched_result_indices")
    @classmethod
    def validate_result_indices(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 50 for value in values):
            raise ValueError("result indices must be between 1 and 50")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def normalize_and_validate(self) -> AgentIntent:
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")

        list_fields = (
            "reference_titles",
            "include_genres",
            "exclude_genres",
            "required_studios",
            "preferred_studios",
            "required_staff",
            "preferred_staff",
            "required_voice_actors",
            "preferred_voice_actors",
            "required_characters",
            "preferred_characters",
            "formats",
            "excluded_titles",
            "seen_titles",
        )
        for field_name in list_fields:
            setattr(self, field_name, self._dedupe_strings(getattr(self, field_name)))

        # A required entity that carries no content is worse than no constraint:
        # it filters the catalog to nothing. The parser once produced
        # required_characters=["With"] from the preposition in "something with
        # dark psychological mind games", which then resolved against no
        # catalog entity and emptied the result set.
        # Entity mentions are resolved as hard constraints, so a mention that
        # names nothing empties the result set rather than narrowing it. A
        # mention carrying only an index still refers to a previous result and
        # is kept.
        self.entity_mentions = [
            mention for mention in self.entity_mentions if mention.index is not None or _is_nameable(mention.text)
        ]

        for entity_field in (
            "required_studios",
            "required_staff",
            "required_voice_actors",
            "required_characters",
            "preferred_studios",
            "preferred_staff",
            "preferred_voice_actors",
            "preferred_characters",
        ):
            setattr(
                self,
                entity_field,
                [value for value in getattr(self, entity_field) if _is_nameable(value)],
            )

        for required_name, preferred_name in (
            ("required_studios", "preferred_studios"),
            ("required_staff", "preferred_staff"),
            ("required_voice_actors", "preferred_voice_actors"),
            ("required_characters", "preferred_characters"),
        ):
            required_keys = {value.casefold() for value in getattr(self, required_name)}
            setattr(
                self,
                preferred_name,
                [value for value in getattr(self, preferred_name) if value.casefold() not in required_keys],
            )
        return self

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = value.strip()
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    def to_legacy(self) -> StructuredIntent:
        return StructuredIntent(
            intent="conversational" if self.intent == "conversation" else self.intent,
            catalog_query=self.catalog_query,
            rank_by=self.rank_by,
            sort_order=self.sort_order,
            reference_titles=self.reference_titles,
            entity_mentions=[EntityMention(**mention.model_dump()) for mention in self.entity_mentions],
            include_genres=self.include_genres,
            exclude_genres=self.exclude_genres,
            required_studios=self.required_studios,
            preferred_studios=self.preferred_studios,
            required_staff=self.required_staff,
            preferred_staff=self.preferred_staff,
            required_characters=self.required_characters,
            preferred_characters=self.preferred_characters,
            required_voice_actors=self.required_voice_actors,
            preferred_voice_actors=self.preferred_voice_actors,
            formats=self.formats,
            min_score=self.min_score,
            min_year=self.min_year,
            max_year=self.max_year,
            max_episodes=self.max_episodes,
            excluded_titles=self.excluded_titles,
            seen_titles=self.seen_titles,
            exclude_related_series=self.exclude_related_series,
            one_per_series=self.one_per_series,
            top_k=self.top_k,
            free_text_preferences=self.free_text_preferences,
            novelty_preference=self.novelty_preference,
            reference_result_indices=self.reference_result_indices,
            watched_result_indices=self.watched_result_indices,
            inferred_constraints=[
                LegacyInferredConstraint(**constraint.model_dump()) for constraint in self.inferred_constraints
            ],
            preference_update=PreferenceUpdate(**self.preference_update.model_dump()),
        )

    @classmethod
    def from_legacy(cls, intent: StructuredIntent) -> AgentIntent:
        payload = intent.to_dict()
        if payload.get("intent") == "conversational":
            payload["intent"] = "conversation"
        return cls.model_validate(payload)

    @classmethod
    def provider_json_schema(cls) -> dict[str, Any]:
        """Return a compact generation schema; full validation still uses this model."""
        schema = deepcopy(cls.model_json_schema())
        schema.get("properties", {}).pop("inferred_constraints", None)
        definitions = schema.get("$defs", {})
        definitions.pop("IntentInferredConstraint", None)

        def compact(value: Any) -> None:
            if isinstance(value, dict):
                for key in (
                    "default",
                    "title",
                    "minLength",
                    "maxLength",
                    "minItems",
                    "maxItems",
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                ):
                    value.pop(key, None)
                for child in value.values():
                    compact(child)
            elif isinstance(value, list):
                for child in value:
                    compact(child)

        compact(schema)
        entity_properties = definitions.get("IntentEntityMention", {}).get("properties", {})
        for field_name in ("entity_type", "relation"):
            field_schema = entity_properties.get(field_name)
            if isinstance(field_schema, dict):
                field_schema.pop("enum", None)
        return schema


class ProviderParseContext(BaseModel):
    genres: list[str]
    formats: list[str]
    session: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, str]] = Field(default_factory=list)


class ProviderHealth(BaseModel):
    provider: str
    model: str
    available: bool
    detail: str | None = None
