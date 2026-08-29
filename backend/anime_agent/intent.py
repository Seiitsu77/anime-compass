from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

ALLOWED_INTENTS = {"recommend", "rank_catalog", "search", "details", "update_preferences", "conversational"}
ALLOWED_ENTITY_TYPES = {
    "anime",
    "character",
    "staff",
    "director",
    "original_creator",
    "studio",
    "producer",
    "voice_actor",
    "genre",
    "theme",
    "demographic",
    "result_index",
}
ALLOWED_RELATIONS = {"direct", "reference", "exclude", "watched", "director_of", "related_to"}
ALLOWED_NOVELTY_PREFERENCES = {"neutral", "less_famous", "mainstream"}
ALLOWED_CONSTRAINT_FIELDS = {"max_episodes", "min_score", "min_year", "max_year"}
ALLOWED_RANK_FIELDS = {"score", "popularity", "rank", "members", "start_year", "title"}
ALLOWED_SORT_ORDERS = {"asc", "desc"}

INTENT_SCHEMA = {
    "intent": "recommend | rank_catalog | search | details | update_preferences | conversational",
    "catalog_query": "string",
    "rank_by": "score | popularity | rank | members | start_year | title",
    "sort_order": "asc | desc | null",
    "reference_titles": ["string"],
    "entity_mentions": [
        {
            "text": "string",
            "entity_type": "anime | character | staff | director | original_creator | studio | producer | voice_actor | genre | theme | demographic | result_index",
            "relation": "direct | reference | exclude | watched | director_of | related_to",
            "index": "integer or null",
        }
    ],
    "include_genres": ["string"],
    "exclude_genres": ["string"],
    "required_studios": ["string"],
    "preferred_studios": ["string"],
    "required_staff": ["string"],
    "preferred_staff": ["string"],
    "required_characters": ["string"],
    "preferred_characters": ["string"],
    "required_voice_actors": ["string"],
    "preferred_voice_actors": ["string"],
    "formats": ["string"],
    "min_score": "number or null",
    "min_year": "integer or null",
    "max_year": "integer or null",
    "max_episodes": "integer or null",
    "excluded_titles": ["string"],
    "seen_titles": ["string"],
    "exclude_related_series": "boolean",
    "one_per_series": "boolean",
    "top_k": "integer from 1 to 50",
    "free_text_preferences": "string",
    "novelty_preference": "neutral | less_famous | mainstream",
    "reference_result_indices": ["integer"],
    "watched_result_indices": ["integer"],
    "inferred_constraints": [
        {
            "field": "max_episodes | min_score | min_year | max_year",
            "value": "number",
            "confidence": "number from 0 to 1",
            "source_text": "string",
        }
    ],
    "preference_update": {
        "liked_titles": ["string"],
        "disliked_titles": ["string"],
        "watched_titles": ["string"],
        "excluded_titles": ["string"],
        "preferred_genres": ["string"],
        "excluded_genres": ["string"],
    },
}


class IntentValidationError(ValueError):
    pass


@dataclass
class InferredConstraint:
    field: str
    value: float
    confidence: float
    source_text: str


@dataclass
class EntityMention:
    text: str
    entity_type: str = "anime"
    relation: str = "direct"
    index: int | None = None


@dataclass
class PreferenceUpdate:
    liked_titles: list[str] = field(default_factory=list)
    disliked_titles: list[str] = field(default_factory=list)
    watched_titles: list[str] = field(default_factory=list)
    excluded_titles: list[str] = field(default_factory=list)
    preferred_genres: list[str] = field(default_factory=list)
    excluded_genres: list[str] = field(default_factory=list)


@dataclass
class StructuredIntent:
    intent: str = "conversational"
    catalog_query: str = ""
    rank_by: str = "score"
    sort_order: str | None = None
    reference_titles: list[str] = field(default_factory=list)
    entity_mentions: list[EntityMention] = field(default_factory=list)
    include_genres: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    required_studios: list[str] = field(default_factory=list)
    preferred_studios: list[str] = field(default_factory=list)
    required_staff: list[str] = field(default_factory=list)
    preferred_staff: list[str] = field(default_factory=list)
    required_characters: list[str] = field(default_factory=list)
    preferred_characters: list[str] = field(default_factory=list)
    required_voice_actors: list[str] = field(default_factory=list)
    preferred_voice_actors: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    min_score: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    max_episodes: int | None = None
    excluded_titles: list[str] = field(default_factory=list)
    seen_titles: list[str] = field(default_factory=list)
    exclude_related_series: bool = True
    one_per_series: bool = True
    top_k: int = 10
    free_text_preferences: str = ""
    novelty_preference: str = "neutral"
    reference_result_indices: list[int] = field(default_factory=list)
    watched_result_indices: list[int] = field(default_factory=list)
    inferred_constraints: list[InferredConstraint] = field(default_factory=list)
    preference_update: PreferenceUpdate = field(default_factory=PreferenceUpdate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_structured_intent(
    content: str,
    available_genres: list[str] | None = None,
    available_formats: list[str] | None = None,
) -> StructuredIntent:
    payload = extract_json_object(content)
    return validate_structured_intent(payload, available_genres, available_formats)


def extract_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    inline = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if inline:
        candidates.append(inline.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise IntentValidationError("Qwen did not return one valid JSON object")


def validate_structured_intent(
    payload: dict[str, Any],
    available_genres: list[str] | None = None,
    available_formats: list[str] | None = None,
) -> StructuredIntent:
    if not isinstance(payload, dict):
        raise IntentValidationError("Structured intent must be a JSON object")

    allowed_keys = set(INTENT_SCHEMA)
    unknown_keys = set(payload).difference(allowed_keys)
    if unknown_keys:
        raise IntentValidationError("Unknown structured-intent fields: " + ", ".join(sorted(unknown_keys)))

    intent_name = str(payload.get("intent") or "").strip().casefold()
    if intent_name not in ALLOWED_INTENTS:
        raise IntentValidationError(f"Unsupported intent: {intent_name or '<empty>'}")

    genre_lookup = {genre.casefold(): genre for genre in available_genres or []}
    format_lookup = {value.casefold(): value for value in available_formats or []}

    include_genres = _canonical_list(payload.get("include_genres"), genre_lookup, "include_genres")
    exclude_genres = _canonical_list(payload.get("exclude_genres"), genre_lookup, "exclude_genres")
    formats = _canonical_list(payload.get("formats"), format_lookup, "formats")

    min_score = _optional_float(payload.get("min_score"), "min_score", 0.0, 10.0)
    min_year = _optional_int(payload.get("min_year"), "min_year", 1900, 2100)
    max_year = _optional_int(payload.get("max_year"), "max_year", 1900, 2100)
    max_episodes = _optional_int(payload.get("max_episodes"), "max_episodes", 1, 100000)
    if min_year is not None and max_year is not None and min_year > max_year:
        raise IntentValidationError("min_year cannot be greater than max_year")

    novelty = str(payload.get("novelty_preference") or "neutral").strip().casefold()
    if novelty not in ALLOWED_NOVELTY_PREFERENCES:
        raise IntentValidationError(f"Unsupported novelty_preference: {novelty}")

    rank_by = str(payload.get("rank_by") or "score").strip().casefold()
    if rank_by not in ALLOWED_RANK_FIELDS:
        raise IntentValidationError(f"Unsupported rank_by: {rank_by}")
    raw_sort_order = payload.get("sort_order")
    sort_order = str(raw_sort_order).strip().casefold() if raw_sort_order not in (None, "") else None
    if sort_order is not None and sort_order not in ALLOWED_SORT_ORDERS:
        raise IntentValidationError(f"Unsupported sort_order: {sort_order}")

    mentions = _entity_mentions(payload.get("entity_mentions"))
    inferred = _inferred_constraints(payload.get("inferred_constraints"))
    preference_update = _preference_update(payload.get("preference_update"))

    return StructuredIntent(
        intent=intent_name,
        catalog_query=str(payload.get("catalog_query") or "").strip()[:300],
        rank_by=rank_by,
        sort_order=sort_order,
        reference_titles=_string_list(payload.get("reference_titles"), "reference_titles"),
        entity_mentions=mentions,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        required_studios=_string_list(payload.get("required_studios"), "required_studios"),
        preferred_studios=_string_list(payload.get("preferred_studios"), "preferred_studios"),
        required_staff=_string_list(payload.get("required_staff"), "required_staff"),
        preferred_staff=_string_list(payload.get("preferred_staff"), "preferred_staff"),
        required_characters=_string_list(payload.get("required_characters"), "required_characters"),
        preferred_characters=_string_list(payload.get("preferred_characters"), "preferred_characters"),
        required_voice_actors=_string_list(payload.get("required_voice_actors"), "required_voice_actors"),
        preferred_voice_actors=_string_list(payload.get("preferred_voice_actors"), "preferred_voice_actors"),
        formats=formats,
        min_score=min_score,
        min_year=min_year,
        max_year=max_year,
        max_episodes=max_episodes,
        excluded_titles=_string_list(payload.get("excluded_titles"), "excluded_titles"),
        seen_titles=_string_list(payload.get("seen_titles"), "seen_titles"),
        exclude_related_series=_boolean(payload.get("exclude_related_series"), True, "exclude_related_series"),
        one_per_series=_boolean(payload.get("one_per_series"), True, "one_per_series"),
        top_k=_optional_int(payload.get("top_k", 10), "top_k", 1, 50) or 10,
        free_text_preferences=str(payload.get("free_text_preferences") or "").strip()[:1200],
        novelty_preference=novelty,
        reference_result_indices=_index_list(payload.get("reference_result_indices"), "reference_result_indices"),
        watched_result_indices=_index_list(payload.get("watched_result_indices"), "watched_result_indices"),
        inferred_constraints=inferred,
        preference_update=preference_update,
    )


def intent_parser_prompt(
    genres: list[str],
    formats: list[str],
    session_context: dict[str, Any],
    history: list[dict[str, str]],
) -> str:
    return f"""You are the English-language intent parser and tool planner for Anime Compass.

Convert the latest user message into exactly one JSON object. Do not answer the user.
Use only the fields in this schema and do not add fields:
{json.dumps(INTENT_SCHEMA, ensure_ascii=True)}

Catalog genres: {json.dumps(genres, ensure_ascii=True)}
Catalog formats: {json.dumps(formats, ensure_ascii=True)}

Rules:
- Choose rank_catalog only for deterministic sorting requests such as "highest-scored
  Gundam TV anime" or "most popular romance movies". Put the title/family lookup text
  in catalog_query, and set rank_by plus sort_order. Do not use session preferences.
- Choose recommend when the user wants titles to watch, including paraphrases such as
  "something that feels like X" or "another show with the same atmosphere."
- Put descriptive mood, premise, theme, character, and atmosphere language in
  free_text_preferences even when it also maps to a catalog genre.
- Explicit numeric limits are hard fields. Vague phrases such as "short", "not too long",
  or "finish over a weekend" belong in inferred_constraints and free_text_preferences;
  do not silently set a hard max_episodes for vague language.
- Use include_genres and exclude_genres only for canonical catalog genre names above.
- Never invent a genre. Words such as ghosts, spirits, curses, paranormal, and occult
  should remain in free_text_preferences and may map to canonical Supernatural when it
  exists in the catalog.
- "Show me anime involving ..." is a recommendation request, not entity search.
- Entity relation must be exactly one of: direct, reference, exclude, watched,
  director_of, related_to. Genre inclusion belongs in include_genres, never relation.
- Use formats=[] when the user did not request a format. Never fill every format.
- Use novelty_preference="less_famous" for requests such as "less famous" or "hidden gem."
- For "the director of Monster", add an anime entity mention for Monster with
  relation="director_of". Do not invent the director's name.
- For named characters, staff, studios, producers, and voice actors, add entity_mentions
  and the corresponding preferred field when it is already explicit.
- Explicit relationship requests such as "anime from Madhouse", "anime by this
  director", and "anime with Light Yagami" use required_studios, required_staff,
  and required_characters. These fields are hard catalog filters.
- Explicit producer, director, original-creator, theme, and demographic relationships
  must also add a direct entity mention with the most specific entity_type. The backend
  resolves that entity to verified related anime before ranking.
- `required_voice_actors` is a hard catalog relationship constraint. Use it when the
  user asks for anime featuring, voiced by, or having a named voice actor involved.
  Phrases such as "that have him involved" refer back to the named voice actor and
  must populate required_voice_actors, not preferred_voice_actors.
- `preferred_voice_actors` is only a soft ranking preference when cast membership is
  not required. Never substitute it for an explicit cast-membership request.
- For previous-result references, use 1-based reference_result_indices and
  watched_result_indices. Example: "watched the second; more like the fourth" means
  watched_result_indices=[2] and reference_result_indices=[4].
- Preserve explicit exclusions and preference updates. "Watched" maps to watched_titles.
- Treat "I enjoyed [anime]" and "I liked [anime]" as both liked and watched; use the
  same catalog title in liked_titles and watched_titles.
- Set exclude_related_series=true when the user excludes a title and says anything from
  the same series or franchise should also be excluded.
- The active language is English. Do not translate or guess unsupported-language intent.

Valid minimal examples:
User: Show me the 5 highest-scored Gundam TV anime.
JSON: {{"intent":"rank_catalog","catalog_query":"Gundam","rank_by":"score","sort_order":"desc","formats":["TV"],"top_k":5}}

User: Show me anime involving ghosts and spirits.
JSON: {{"intent":"recommend","include_genres":["Supernatural"],"free_text_preferences":"ghosts and spirits","top_k":5}}

User: I enjoyed Death Note and want something with a similar atmosphere.
JSON: {{"intent":"recommend","reference_titles":["Death Note"],"seen_titles":["Death Note"],"free_text_preferences":"similar atmosphere","top_k":5,"preference_update":{{"liked_titles":["Death Note"],"watched_titles":["Death Note"]}}}}

User: I like the director of Monster.
JSON: {{"intent":"recommend","reference_titles":[],"entity_mentions":[{{"text":"Monster","entity_type":"anime","relation":"director_of","index":null}}],"free_text_preferences":"works by the same director","top_k":5}}

User: Give me something short.
JSON: {{"intent":"recommend","free_text_preferences":"something short","inferred_constraints":[{{"field":"max_episodes","value":24,"confidence":0.65,"source_text":"short"}}],"top_k":5}}

User: I really like a voice actor called Matsuoka, Yoshitsugu. Could you recommend 7 anime that have him involved?
JSON: {{"intent":"recommend","entity_mentions":[{{"text":"Matsuoka, Yoshitsugu","entity_type":"voice_actor","relation":"direct","index":null}}],"required_voice_actors":["Matsuoka, Yoshitsugu"],"top_k":7}}

For every array of objects, use [] when there is no complete object. Never emit a
placeholder inferred constraint or entity mention with missing required values.

Recent session context:
{json.dumps(session_context, ensure_ascii=True)[:5000]}

Recent conversation:
{json.dumps(history[-8:], ensure_ascii=True)[:5000]}
"""


def _string_list(value: Any, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise IntentValidationError(f"{field_name} must be an array")
    result: list[Any] = []
    for item in value:
        if not isinstance(item, str):
            raise IntentValidationError(f"{field_name} must contain only strings")
        text = item.strip()
        if text and text.casefold() not in {current.casefold() for current in result}:
            result.append(text[:240])
    return result[:50]


def _canonical_list(value: Any, lookup: dict[str, str], field_name: str) -> list[str]:
    values = _string_list(value, field_name)
    if not lookup:
        return values
    result = []
    for item in values:
        canonical = lookup.get(item.casefold())
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def _entity_mentions(value: Any) -> list[EntityMention]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise IntentValidationError("entity_mentions must be an array")
    mentions = []
    for item in value[:30]:
        if isinstance(item, str):
            mentions.append(EntityMention(text=item.strip()))
            continue
        if not isinstance(item, dict):
            raise IntentValidationError("entity_mentions entries must be strings or objects")
        unknown = set(item).difference({"text", "entity_type", "relation", "index"})
        if unknown:
            raise IntentValidationError("Unknown entity mention fields: " + ", ".join(sorted(unknown)))
        text = str(item.get("text") or "").strip()
        entity_type = str(item.get("entity_type") or "anime").strip().casefold()
        relation = str(item.get("relation") or "direct").strip().casefold()
        if entity_type not in ALLOWED_ENTITY_TYPES:
            raise IntentValidationError(f"Unsupported entity type: {entity_type}")
        if relation not in ALLOWED_RELATIONS:
            raise IntentValidationError(f"Unsupported entity relation: {relation}")
        index = _optional_int(item.get("index"), "entity mention index", 1, 100)
        if not text and index is None:
            raise IntentValidationError("Entity mention requires text or index")
        mentions.append(EntityMention(text=text, entity_type=entity_type, relation=relation, index=index))
    return mentions


def _inferred_constraints(value: Any) -> list[InferredConstraint]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise IntentValidationError("inferred_constraints must be an array")
    result = []
    for item in value[:10]:
        if not isinstance(item, dict):
            raise IntentValidationError("inferred_constraints entries must be objects")
        unknown = set(item).difference({"field", "value", "confidence", "source_text"})
        if unknown:
            raise IntentValidationError("Unknown inferred-constraint fields: " + ", ".join(sorted(unknown)))
        field_name = str(item.get("field") or "").strip()
        if field_name not in ALLOWED_CONSTRAINT_FIELDS:
            raise IntentValidationError(f"Unsupported inferred constraint: {field_name}")
        inferred_value = _optional_float(item.get("value"), "inferred value", 0.0, 100000.0)
        confidence = _optional_float(item.get("confidence"), "inferred confidence", 0.0, 1.0)
        source_text = str(item.get("source_text") or "").strip()
        if inferred_value is None or confidence is None or not source_text:
            raise IntentValidationError("Inferred constraints require value, confidence, and source_text")
        result.append(InferredConstraint(field_name, inferred_value, confidence, source_text[:300]))
    return result


def _preference_update(value: Any) -> PreferenceUpdate:
    if value in (None, ""):
        return PreferenceUpdate()
    if not isinstance(value, dict):
        raise IntentValidationError("preference_update must be an object")
    fields = set(PreferenceUpdate.__dataclass_fields__)
    unknown = set(value).difference(fields)
    if unknown:
        raise IntentValidationError("Unknown preference-update fields: " + ", ".join(sorted(unknown)))
    return PreferenceUpdate(**{name: _string_list(value.get(name), f"preference_update.{name}") for name in fields})


def _index_list(value: Any, field_name: str) -> list[int]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise IntentValidationError(f"{field_name} must be an array")
    result = []
    for item in value:
        index = _optional_int(item, field_name, 1, 50)
        if index is not None and index not in result:
            result.append(index)
    return result


def _optional_int(value: Any, field_name: str, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise IntentValidationError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise IntentValidationError(f"{field_name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise IntentValidationError(f"{field_name} must be between {minimum} and {maximum}")
    return parsed


def _optional_float(value: Any, field_name: str, minimum: float, maximum: float) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise IntentValidationError(f"{field_name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IntentValidationError(f"{field_name} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise IntentValidationError(f"{field_name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _boolean(value: Any, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise IntentValidationError(f"{field_name} must be a boolean")
    return value
