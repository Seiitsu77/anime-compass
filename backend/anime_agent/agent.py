from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .entities import EntityResolver
from .intent import (
    EntityMention,
    InferredConstraint,
    IntentValidationError,
    PreferenceUpdate,
    StructuredIntent,
    intent_parser_prompt,
    parse_structured_intent,
)
from .ollama_client import OllamaClient, OllamaUnavailable
from .recommender import AnimeRecommender, series_key, tokenize

SYSTEM_PROMPT = """You are Anime Compass, a friendly and proactive local anime guide.

You have access to trusted anime catalog tools. Always use them for catalog facts,
recommendations, introductions, characters, staff, and voice actors. Never invent a
title, score, genre, studio, person, or story detail.

When you need a tool, respond with only one JSON object:
{"tool":"search_anime","arguments":{"query":"title or keywords","limit":5}}
{"tool":"rank_catalog","arguments":{"query":"Gundam","formats":["TV"],"sort_by":"score","sort_order":"desc","top_k":5}}
{"tool":"recommend_anime","arguments":{"reference_titles":["Title"],"include_genres":["Action"],"formats":["TV"],"top_k":5}}
{"tool":"get_anime_details","arguments":{"anime_id":123}}
{"tool":"update_session_preferences","arguments":{"liked_titles":["Title"],"excluded_genres":["Mecha"]}}

Response style:
- Lead with the answer instead of a generic greeting.
- Sound warm, natural, and confident, but never overexcited or overly wordy.
- Explain why a recommendation fits the user's request using catalog evidence.
- For an introduction, give a spoiler-light premise, tone and themes, practical details,
  and notable cast or staff only when those fields are present in the tool result.
- End with one concrete next step when useful, such as offering similar titles, cast
  details, or a deeper spoiler-free explanation.
- Keep lists short unless the user asks for more.
"""

INTRODUCTION_PATTERNS = (
    "introduce ",
    "introduction to ",
    "tell me about ",
    "give me an overview of ",
    "what is ",
    "what's ",
    "what is the anime ",
    "describe ",
    "explain ",
    "synopsis of ",
    "plot of ",
    "characters in ",
    "cast of ",
    "voice actors in ",
    "who voices ",
    "who made ",
    "which studio made ",
    "worth watching",
)

RECOMMENDATION_INTENTS = {
    "another",
    "find",
    "give",
    "more",
    "option",
    "options",
    "recommend",
    "recommendations",
    "recommendation",
    "suggest",
    "similar",
    "like",
    "looking",
    "watch",
    "watching",
    "want",
}

RECOMMENDATION_FOLLOWUP_PATTERNS = (
    "more",
    "another",
    "other",
    "options",
    "score",
    "rating",
    "at least",
    "above",
    "same series",
    "same franchise",
    "from the same",
    "new batch",
    "fresh batch",
    "no repeats",
    "do not repeat",
    "don't repeat",
    "not repeat",
    "2020 or later",
    "or later",
    "shorter",
    "short",
    "not too long",
    "weekend",
)

NEGATIVE_FEEDBACK_PATTERNS = (
    "don't like",
    "don't want",
    "do not like",
    "do not want",
    "didn't like",
    "did not like",
    "without",
    "not like",
    "dislike",
    "disliked",
    "hate",
    "hated",
    "not interested",
    "no more",
    "avoid",
    "exclude",
    "not these",
    "none of these",
    "bad recommendations",
    "same recommendations",
    "was not for me",
    "wasn't for me",
    "not my kind of anime",
    "not my thing",
    "do not show me things like this",
    "don't show me things like this",
    "not interested in this kind",
)

NEW_BATCH_PATTERNS = (
    "new batch",
    "fresh batch",
    "another batch",
    "different batch",
    "more options",
    "more recommendations",
    "no repeats",
    "do not repeat",
    "don't repeat",
    "not repeat",
    "not repeated",
)

ONE_PER_SERIES_PATTERNS = (
    "one anime from each series",
    "one title from each series",
    "one from each series",
    "one anime from each serie",
    "one title from each serie",
    "one from each serie",
    "only recommend one",
    "not in series",
    "no sequels",
    "no sequel",
    "standalone",
)

GENRE_ALIASES = {
    "rommance": "Romance",
    "romance": "Romance",
    "romantic": "Romance",
    "love": "Romance",
    "relationship": "Romance",
    "relationships": "Romance",
    "mystery": "Mystery",
    "mysteries": "Mystery",
    "mecha": "Mecha",
    "robot": "Mecha",
    "robots": "Mecha",
    "slice of life": "Slice of Life",
    "healing": "Iyashikei",
    "comfort": "Iyashikei",
    "relaxing": "Iyashikei",
    "drama": "Drama",
    "action": "Action",
    "comedy": "Comedy",
    "school": "School",
    "ghost": "Supernatural",
    "ghosts": "Supernatural",
    "spirit": "Supernatural",
    "spirits": "Supernatural",
    "paranormal": "Supernatural",
    "occult": "Supernatural",
    "curse": "Supernatural",
    "curses": "Supernatural",
}

TITLE_ALIASES = {}

GENERIC_TITLE_TOKENS = {
    "anime",
    "each",
    "from",
    "give",
    "more",
    "one",
    "only",
    "option",
    "options",
    "recommend",
    "recommendations",
    "serie",
    "series",
    "title",
    "titles",
    "you",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


class AnimeAgent:
    def __init__(
        self,
        recommender: AnimeRecommender,
        client: OllamaClient | None = None,
        get_session_profile: Callable[[str | None], dict[str, Any]] | None = None,
        update_session_preferences: Callable[[str | None, dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.recommender = recommender
        self.client = client or OllamaClient()
        self.entity_resolver = EntityResolver(recommender.catalog)
        self.get_session_profile = get_session_profile or (lambda _session_id: {})
        self.update_session_preferences = update_session_preferences or (lambda _session_id, _patch: {})
        self.tools: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search_anime": self._tool_search,
            "rank_catalog": self._tool_rank_catalog,
            "search_entities": self._tool_search_entities,
            "recommend_anime": self._tool_recommend,
            "get_anime_details": self._tool_details,
            "anime_details": self._tool_details,
            "update_session_preferences": self._tool_update_session,
        }

    def status(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.client.model,
            "base_url": self.client.base_url,
            "available": self.client.is_available(),
        }

    def respond(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        history = history or []
        if not self._is_supported_english(message):
            return {
                "mode": "unsupported_language",
                "answer": "Anime Compass currently understands English requests only.",
                "trace": [],
                "agent": self.status(),
            }

        session = self.get_session_profile(session_id)
        intent, parser_mode, parser_error = self._parse_intent(message, history, session)
        try:
            response = self._respond_from_intent(intent, message, history, session_id, session)
        except (KeyError, TypeError, ValueError, OllamaUnavailable) as exc:
            if parser_mode != "qwen_structured":
                raise
            parser_mode = "rule_fallback"
            parser_error = f"Structured tool orchestration failed: {exc}"
            intent = self._rule_based_intent(message, history)
            response = self._respond_from_intent(intent, message, history, session_id, session)
        internal_debug = response.pop("_debug", {})
        if debug:
            response["parser_mode"] = parser_mode
            response["validated_intent"] = intent.to_dict()
            debug_payload = {
                "parser_mode": parser_mode,
                "parser_error": parser_error,
                "validated_intent": intent.to_dict(),
            }
            debug_payload.update(internal_debug)
            response["debug"] = debug_payload
            for field_name in (
                "selected_tool",
                "resolved_entity",
                "resolved_entity_id",
                "entity_type",
                "candidate_count_before_filter",
                "candidate_count_after_voice_actor_filter",
                "verified_voice_actor_matches",
                "ignored_session_fields",
            ):
                if field_name in debug_payload:
                    response[field_name] = debug_payload[field_name]
        return response

    def _parse_intent(
        self,
        message: str,
        history: list[dict[str, str]],
        session: dict[str, Any],
    ) -> tuple[StructuredIntent, str, str | None]:
        if self.client.is_available():
            try:
                meta = self.recommender.meta()
                messages = [
                    {
                        "role": "system",
                        "content": intent_parser_prompt(meta["genres"], meta["types"], session, history),
                    },
                    {"role": "user", "content": message},
                ]
                chat_json = getattr(self.client, "chat_json", None)
                raw = chat_json(messages) if callable(chat_json) else self.client.chat(messages)
                parsed = parse_structured_intent(raw, meta["genres"], meta["types"])
                return parsed, "qwen_structured", None
            except (IntentValidationError, OllamaUnavailable, TypeError, ValueError) as exc:
                parser_error = str(exc)
            except Exception as exc:
                parser_error = f"Intent parser failed: {exc}"
        else:
            parser_error = "Ollama is unavailable"
        return self._rule_based_intent(message, history), "rule_fallback", parser_error

    def _rule_based_intent(self, message: str, history: list[dict[str, str]]) -> StructuredIntent:
        source = self._recommendation_source_message(message, history)
        intent_name = "conversational"
        if self._looks_like_introduction_request(message):
            intent_name = "details"
        elif self._looks_like_catalog_ranking(message):
            intent_name = "rank_catalog"
        elif self._looks_like_negative_feedback(message) and not re.search(
            r"\b(give|recommend|suggest|find|more|another|different|new batch)\b",
            message,
            re.IGNORECASE,
        ):
            intent_name = "update_preferences"
        elif self._should_use_catalog_recommender(message, history):
            intent_name = "recommend"
        elif self._extract_preference_patch(message):
            intent_name = "update_preferences"
        elif self._has_catalog_signal(message) or any(
            value in message.casefold() for value in ("something short", "nothing too long", "finish this weekend")
        ):
            intent_name = "recommend"
        elif re.search(r"\b(search|look up|who voices|who made|cast|staff)\b", message, re.IGNORECASE):
            intent_name = "search"

        patch = self._extract_preference_patch(message)
        message_key = source.casefold()
        excluded_genres = list(patch.get("excluded_genres", []))
        included_genres, explicitly_excluded_genres = self._matched_genre_constraints(source)
        if explicitly_excluded_genres:
            excluded_genres = explicitly_excluded_genres
        preference = PreferenceUpdate(
            liked_titles=list(patch.get("liked_titles", [])),
            disliked_titles=list(patch.get("disliked_titles", [])),
            watched_titles=list(patch.get("seen_titles", [])),
            excluded_titles=list(patch.get("excluded_titles", [])),
            preferred_genres=list(patch.get("preferred_genres", [])),
            excluded_genres=excluded_genres,
        )
        novelty = (
            "less_famous"
            if any(value in message_key for value in ("less famous", "hidden gem", "underrated", "obscure"))
            else "neutral"
        )
        inferred = []
        if (
            any(value in message_key for value in ("short", "not too long", "over a weekend", "quick watch"))
            and self._requested_max_episodes(source) is None
        ):
            inferred.append(InferredConstraint("max_episodes", 24, 0.55, "vague length preference"))

        mentions: list[EntityMention] = []
        for match in re.finditer(r"\b(?:the\s+)?director\s+of\s+([^,?.]+)", message, re.IGNORECASE):
            mentions.append(EntityMention(match.group(1).strip(), "anime", "director_of"))
        for title in self._mentioned_catalog_titles(source):
            mentions.append(EntityMention(title, "anime", "reference"))
        required_voice_actors = self._extract_required_voice_actors(message)
        for actor in required_voice_actors:
            mentions.append(EntityMention(actor, "voice_actor", "direct"))
        explicit_relationship = bool(
            re.search(
                r"\b(?:anime|shows?|titles?)\b.{0,80}\b(?:from|by|with|featuring|involving)\b", message, re.IGNORECASE
            )
        )
        named_studios = self._named_catalog_entities(message, "studio")
        named_staff = self._named_catalog_entities(message, "staff")
        named_characters = self._named_catalog_entities(message, "character")
        named_producers = self._named_catalog_entities(message, "producer")
        named_directors = self._named_catalog_entities(message, "director")
        named_original_creators = self._named_catalog_entities(message, "original_creator")
        named_themes = self._named_catalog_entities(message, "theme")
        named_demographics = self._named_catalog_entities(message, "demographic")
        required_studios = named_studios if explicit_relationship else []
        required_staff = named_staff if explicit_relationship and re.search(r"\bby\b", message, re.IGNORECASE) else []
        required_characters = named_characters if explicit_relationship else []
        for name in required_studios:
            mentions.append(EntityMention(name, "studio", "direct"))
        for name in required_staff:
            mentions.append(EntityMention(name, "staff", "direct"))
        for name in required_characters:
            mentions.append(EntityMention(name, "character", "direct"))
        for entity_type, names in (
            ("producer", named_producers),
            ("director", named_directors),
            ("original_creator", named_original_creators),
            ("theme", named_themes),
            ("demographic", named_demographics),
        ):
            for name in names:
                mentions.append(EntityMention(name, entity_type, "direct"))

        return StructuredIntent(
            intent=intent_name,
            catalog_query=self._catalog_ranking_query(message) if intent_name == "rank_catalog" else "",
            rank_by=self._requested_rank_field(message),
            sort_order=self._requested_sort_order(message),
            reference_titles=self._mentioned_catalog_titles(source),
            entity_mentions=mentions,
            include_genres=included_genres,
            exclude_genres=preference.excluded_genres,
            required_studios=required_studios,
            preferred_studios=[] if required_studios else named_studios,
            required_staff=required_staff,
            preferred_staff=[] if required_staff else named_staff,
            required_characters=required_characters,
            preferred_characters=[] if required_characters else named_characters,
            required_voice_actors=required_voice_actors,
            min_score=self._requested_min_score(source),
            min_year=self._requested_min_year(source),
            max_year=self._requested_max_year(source),
            max_episodes=self._requested_max_episodes(source),
            excluded_titles=self._collect_negative_feedback_titles(message, history),
            seen_titles=preference.watched_titles,
            exclude_related_series=True,
            one_per_series=self._requested_one_per_series(message, history),
            top_k=self._requested_limit(message) or self._requested_limit(source) or 5,
            formats=self._matched_formats(message),
            free_text_preferences=source if intent_name == "recommend" else "",
            novelty_preference=novelty,
            inferred_constraints=inferred,
            preference_update=preference,
        )

    def _named_catalog_entities(self, message: str, entity_type: str) -> list[str]:
        message_key = self._entity_name_key(message)
        matches: list[str] = []
        for record in self.entity_resolver.by_type.get(entity_type, []):
            variants = [variant for variant in record.variants if len(variant) >= 4]
            if any(re.search(rf"(?:^|\s){re.escape(variant)}(?:$|\s)", message_key) for variant in variants):
                matches.append(record.name)
        return self._dedupe_entity_names(matches)

    def _respond_from_intent(
        self,
        intent: StructuredIntent,
        message: str,
        history: list[dict[str, str]],
        session_id: str | None,
        session: dict[str, Any],
    ) -> dict[str, Any]:
        context_error = self._materialize_result_references(intent, message, session)
        if context_error:
            return {
                "mode": "catalog_context_error",
                "answer": context_error,
                "trace": [],
                "agent": self.status(),
            }
        self._normalize_explicit_request(intent, message)
        if self._looks_like_series_lookup_request(message):
            intent.intent = "search"
            return self._respond_with_title_family(intent, message)
        if intent.intent == "recommend":
            return self._respond_with_structured_recommendations(intent, message, history, session_id, session)
        if intent.intent == "rank_catalog":
            return self._respond_with_catalog_ranking(intent, message)
        if intent.intent == "details":
            return self._respond_with_structured_details(intent, message)
        if intent.intent == "search":
            return self._respond_with_structured_search(intent, message)
        if intent.intent == "update_preferences":
            patch = self._preference_patch_from_intent(intent)
            result = self._tool_update_session({"session_id": session_id, **patch})
            return {
                "mode": "session_update",
                "answer": "I updated this session's preferences and will use them in later recommendations.",
                "trace": [{"tool": "update_session_preferences", "arguments": patch, "result": result}],
                "agent": self.status(),
            }
        return self._conversational_response(message, history)

    def _normalize_explicit_request(self, intent: StructuredIntent, message: str) -> None:
        """Make literal user constraints authoritative over probabilistic parsing."""
        explicit_formats = self._matched_formats(message)
        include_genres, exclude_genres = self._matched_genre_constraints(message)
        requested_limit = self._requested_limit(message)
        min_score = self._requested_min_score(message)
        min_year = self._requested_min_year(message)
        max_year = self._requested_max_year(message)
        max_episodes = self._requested_max_episodes(message)

        valid_genres = {value.casefold(): value for value in self.recommender.meta()["genres"]}
        intent.include_genres = [
            valid_genres[value.casefold()] for value in intent.include_genres if value.casefold() in valid_genres
        ]
        intent.exclude_genres = [
            valid_genres[value.casefold()] for value in intent.exclude_genres if value.casefold() in valid_genres
        ]
        if include_genres or exclude_genres:
            intent.include_genres = include_genres
            intent.exclude_genres = exclude_genres
        if explicit_formats:
            intent.formats = explicit_formats
        if requested_limit is not None:
            intent.top_k = requested_limit
        if min_score is not None:
            intent.min_score = min_score
        if min_year is not None:
            intent.min_year = min_year
        if max_year is not None:
            intent.max_year = max_year
        if max_episodes is not None:
            intent.max_episodes = max_episodes

        explicitly_excluded = self._explicitly_excluded_titles(message)
        if explicitly_excluded:
            excluded_keys = {value.casefold() for value in explicitly_excluded}
            intent.excluded_titles = list(dict.fromkeys([*intent.excluded_titles, *explicitly_excluded]))
            intent.reference_titles = [
                value for value in intent.reference_titles if value.casefold() not in excluded_keys
            ]
            intent.entity_mentions = [
                mention
                for mention in intent.entity_mentions
                if not (mention.entity_type == "anime" and self._canonical_title_key(mention.text) in excluded_keys)
            ]
            intent.preference_update.excluded_titles = list(
                dict.fromkeys([*intent.preference_update.excluded_titles, *explicitly_excluded])
            )

        named_studios = self._named_catalog_entities(message, "studio")
        asks_for_catalog_items = bool(
            re.search(
                r"\b(?:recommend|suggest|find|show|give|list|rank|sort|looking for|want|need)\b",
                message,
                re.IGNORECASE,
            )
        )
        if named_studios and asks_for_catalog_items:
            intent.required_studios = named_studios
            intent.preferred_studios = []
            existing = {
                (mention.entity_type, self._entity_name_key(mention.text)) for mention in intent.entity_mentions
            }
            for studio in named_studios:
                key = ("studio", self._entity_name_key(studio))
                if key not in existing:
                    intent.entity_mentions.append(EntityMention(studio, "studio", "direct"))
                    existing.add(key)

        if self._looks_like_catalog_ranking(message):
            intent.intent = "rank_catalog"
            intent.catalog_query = self._catalog_ranking_query(message)
            intent.rank_by = self._requested_rank_field(message)
            intent.sort_order = self._requested_sort_order(message)
        elif (
            asks_for_catalog_items
            and intent.intent in {"search", "conversational"}
            and (
                requested_limit is not None
                or explicit_formats
                or include_genres
                or min_score is not None
                or min_year is not None
                or max_year is not None
                or max_episodes is not None
                or named_studios
            )
        ):
            intent.intent = "recommend"
            intent.free_text_preferences = intent.free_text_preferences or message
        elif (
            self._looks_like_negative_feedback(message)
            and not asks_for_catalog_items
            and not self._looks_like_new_batch_request(message)
        ):
            intent.intent = "update_preferences"

        format_keys = {value.casefold() for value in self.recommender.meta()["types"]}
        intent.include_genres = [
            value for value in dict.fromkeys(intent.include_genres) if value.casefold() not in format_keys
        ]
        intent.exclude_genres = [
            value for value in dict.fromkeys(intent.exclude_genres) if value.casefold() not in format_keys
        ]

    def _respond_with_catalog_ranking(self, intent: StructuredIntent, message: str) -> dict[str, Any]:
        requested_limit = self._requested_limit(message)
        if requested_limit is not None:
            intent.top_k = requested_limit
        arguments = {
            "query": intent.catalog_query,
            "include_genres": intent.include_genres,
            "exclude_genres": intent.exclude_genres,
            "formats": intent.formats,
            "required_studios": intent.required_studios,
            "min_score": intent.min_score,
            "min_year": intent.min_year,
            "max_year": intent.max_year,
            "max_episodes": intent.max_episodes,
            "excluded_titles": intent.excluded_titles,
            "sort_by": intent.rank_by,
            "sort_order": intent.sort_order,
            "top_k": intent.top_k,
        }
        result = self._tool_rank_catalog(arguments)
        results = result["results"]
        return {
            "mode": "catalog_ranking",
            "answer": format_catalog_ranking_answer(results, result["diagnostics"]),
            "trace": [{"tool": "rank_catalog", "arguments": arguments, "result": compact_result(result)}],
            "agent": self.status(),
            "_debug": {
                "selected_tool": "rank_catalog",
                "candidate_count_before_filter": len(self.recommender.catalog),
                "candidate_count_after_filter": result["diagnostics"]["candidate_count"],
                "ignored_session_fields": ["all"],
            },
        }

    def _respond_with_structured_recommendations(
        self,
        intent: StructuredIntent,
        message: str,
        history: list[dict[str, str]],
        session_id: str | None,
        session: dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce_explicit_entity_constraints(intent, message)
        requested_limit = self._requested_limit(message)
        if requested_limit is not None:
            intent.top_k = requested_limit
        intent = self._merge_followup_intent(intent, message, history, session)
        required_mentions = self._required_entity_mentions(intent, message)
        ignored_session_fields: list[str] = []
        if self._has_required_entity_constraint(intent) or required_mentions:
            ignored_session_fields = [key for key, value in session.items() if value]
            intent.reference_titles = []
            intent.reference_result_indices = []
        resolutions, resolution_debug = self._resolve_intent_entities(intent, session, required_mentions)
        if resolution_debug["errors"]:
            error = resolution_debug["errors"][0]
            if error["type"] == "missing_director_credit":
                answer = (
                    f"I resolved **{error['input']}**, but this catalog has no Director credit for it. "
                    "I cannot reliably recommend works by the same director without that relationship data."
                )
            elif error["type"].startswith("ambiguous_"):
                choices = ", ".join(
                    f"{match['matched_name']} (ID {match.get('entity_id')})" for match in error.get("matches", [])
                )
                entity_label = str(error.get("entity_type") or "entity").replace("_", " ")
                answer = f"I found multiple {entity_label} matches: {choices}. Please specify which one you mean."
            else:
                entity_label = str(error.get("entity_type") or "entity").replace("_", " ")
                answer = f"I could not resolve that {entity_label} in the local catalog. Try its full credited name."
            return {
                "mode": "catalog_constraint_error",
                "answer": answer,
                "trace": resolutions,
                "agent": self.status(),
                "_debug": {
                    "selected_tool": "search_entities",
                    **resolution_debug,
                    "candidate_count_before_filter": len(self.recommender.catalog),
                    "candidate_count_after_voice_actor_filter": 0,
                    "verified_voice_actor_matches": 0,
                    "ignored_session_fields": ignored_session_fields,
                },
            }
        preference_patch = self._preference_patch_from_intent(intent)
        has_required_voice_actor = bool(intent.required_voice_actors)
        has_required_entity = bool(
            self._has_required_entity_constraint(intent) or resolution_debug["required_entity_constraints"]
        )

        arguments = {
            "reference_titles": intent.reference_titles,
            "excluded_titles": intent.excluded_titles,
            "seen_titles": intent.seen_titles,
            "include_genres": intent.include_genres,
            "exclude_genres": intent.exclude_genres,
            "required_studios": intent.required_studios,
            "preferred_studios": intent.preferred_studios,
            "required_staff": intent.required_staff,
            "preferred_staff": intent.preferred_staff,
            "required_characters": intent.required_characters,
            "preferred_characters": intent.preferred_characters,
            "required_voice_actors": intent.required_voice_actors,
            "required_voice_actor_ids": resolution_debug["required_voice_actor_ids"],
            "preferred_voice_actors": intent.preferred_voice_actors,
            "required_entity_constraints": resolution_debug["required_entity_constraints"],
            "formats": intent.formats,
            "min_score": intent.min_score,
            "min_year": intent.min_year,
            "max_year": intent.max_year,
            "max_episodes": intent.max_episodes,
            "query": intent.free_text_preferences or ("" if has_required_entity else message),
            "free_text_preferences": intent.free_text_preferences,
            "novelty_preference": intent.novelty_preference,
            "exclude_related_series": intent.exclude_related_series,
            "one_per_series": intent.one_per_series,
            "top_k": intent.top_k,
            "session_id": session_id,
            "ignore_session_preferences": has_required_entity,
        }
        result = self._tool_recommend(arguments)
        if preference_patch:
            self._tool_update_session({"session_id": session_id, **preference_patch})
        recommendation_step = {"tool": "recommend_anime", "arguments": arguments, "result": compact_result(result)}
        trace = [*resolutions, recommendation_step] if has_required_entity else [recommendation_step, *resolutions]
        result_titles = [item["title"] for item in result["results"]]
        self._tool_update_session(
            {
                "session_id": session_id,
                "last_recommendation_intent": intent.to_dict(),
                "last_recommendations": result_titles,
            }
        )
        fallback = format_recommendation_answer(result["results"], limit=intent.top_k)
        if intent.excluded_titles:
            fallback = "I will avoid the titles you excluded, including related entries when requested.\n\n" + fallback
        content = fallback
        if self.client.is_available() and result["results"] and not has_required_voice_actor:
            try:
                candidate = self.client.chat(
                    [
                        {"role": "system", "content": catalog_summary_prompt(intent.top_k)},
                        {
                            "role": "user",
                            "content": f"User request:\n{message}\n\nCatalog recommendation JSON:\n"
                            + json.dumps(catalog_response_payload(result["results"]), ensure_ascii=False)[:15000],
                        },
                    ]
                )
                if self._valid_recommendation_answer(
                    candidate,
                    result_titles,
                    intent.excluded_titles,
                    intent.top_k,
                    intent.required_voice_actors,
                ):
                    content = candidate
            except OllamaUnavailable:
                pass
        diagnostics = result.get("diagnostics", {})
        response_mode = (
            "catalog_verified"
            if has_required_voice_actor
            else ("ollama" if content != fallback else "catalog_fallback")
        )
        return {
            "mode": response_mode,
            "answer": content,
            "trace": trace,
            "agent": self.status(),
            "_debug": {
                "selected_tool": "recommend_anime",
                **resolution_debug,
                "candidate_count_before_filter": diagnostics.get("candidate_count_before_filter", 0),
                "candidate_count_after_voice_actor_filter": diagnostics.get(
                    "candidate_count_after_voice_actor_filter", 0
                ),
                "verified_voice_actor_matches": diagnostics.get("verified_voice_actor_matches", 0),
                "ignored_session_fields": ignored_session_fields,
            },
        }

    def _respond_with_structured_details(self, intent: StructuredIntent, message: str) -> dict[str, Any]:
        target = intent.reference_titles[0] if intent.reference_titles else ""
        if not target:
            target = next((mention.text for mention in intent.entity_mentions if mention.entity_type == "anime"), "")
        match = self.entity_resolver.resolve(target or self._introduction_search_query(message), "anime")
        if not match or match["confidence"] < 0.62:
            return self._respond_with_structured_search(intent, target or message)
        details = self.recommender.details(int(match["anime_id"]))
        if not details:
            return {
                "mode": "catalog_fallback",
                "answer": "I could not resolve that title in the local catalog.",
                "trace": [],
                "agent": self.status(),
            }
        fallback = self._format_requested_details(details, message)
        content = fallback
        if self.client.is_available():
            try:
                safe_details = {**details, "synopsis": spoiler_safe_synopsis(details.get("synopsis"))}
                candidate = self.client.chat(
                    [
                        {"role": "system", "content": introduction_prompt()},
                        {
                            "role": "user",
                            "content": f"User question:\n{message}\n\nTrusted catalog details JSON:\n"
                            + json.dumps(safe_details, ensure_ascii=False)[:14000],
                        },
                    ]
                )
                if self._valid_introduction_answer(candidate, details["title"]):
                    content = candidate
            except OllamaUnavailable:
                pass
        return {
            "mode": "ollama" if content != fallback else "catalog_introduction",
            "answer": content,
            "trace": [
                {
                    "tool": "get_anime_details",
                    "arguments": {"anime_id": details["id"]},
                    "result": compact_result({"result": details}),
                }
            ],
            "agent": self.status(),
        }

    def _respond_with_title_family(
        self,
        intent: StructuredIntent,
        message: str,
    ) -> dict[str, Any]:
        mentioned = [
            *intent.reference_titles,
            *self._mentioned_catalog_titles(message),
        ]
        target_id = next(iter(self.recommender.resolve_titles(mentioned)), None)
        if target_id is None:
            return {
                "mode": "catalog_search",
                "answer": "I could not resolve the title whose sequel or franchise entries you want.",
                "trace": [],
                "agent": self.status(),
            }
        target = self.recommender.by_id[target_id]
        family_key = series_key(target["title"])
        family = [
            item
            for item in self.recommender.catalog
            if int(item["id"]) != target_id and series_key(item["title"]) == family_key
        ]
        family.sort(
            key=lambda item: (
                item.get("start_year") is None,
                int(item.get("start_year") or 10**9),
                str(item.get("title") or "").casefold(),
            )
        )
        limit = self._requested_limit(message) or intent.top_k
        results = [self.recommender.public_item(item) for item in family[:limit]]
        if results:
            lines = [
                (
                    f"I found these catalog titles in the same title family as "
                    f"**{target['title']}**. The dataset does not contain a reliable sequel graph, "
                    "so this is a title-family match rather than a claim that every entry is a direct sequel:"
                )
            ]
            lines.extend(
                f"{index}. **{item['title']}**" + (f" ({item['start_year']})" if item.get("start_year") else "")
                for index, item in enumerate(results, start=1)
            )
            answer = "\n".join(lines)
        else:
            answer = (
                f"I found **{target['title']}**, but no other catalog titles share its title-family key. "
                "The dataset does not include a reliable sequel relationship graph."
            )
        return {
            "mode": "catalog_search",
            "answer": answer,
            "trace": [
                {
                    "tool": "search_anime",
                    "arguments": {"query": target["title"], "limit": limit},
                    "result": {"results": results},
                }
            ],
            "agent": self.status(),
        }

    @staticmethod
    def _format_requested_details(details: dict[str, Any], message: str) -> str:
        message_key = message.casefold()
        facts: list[str] = []
        if re.search(r"\b(?:score|rating|rated)\b", message_key):
            score = details.get("score")
            facts.append(
                f"Its catalog score is {float(score):.2f}/10."
                if score is not None
                else "The catalog has no score for it."
            )
        if re.search(r"\b(?:studio|who made|animated by)\b", message_key):
            studios = [str(value) for value in details.get("studios", []) if value]
            facts.append(
                "Its credited studio is " + ", ".join(studios) + "."
                if studios
                else "The catalog has no studio credit for it."
            )
        if re.search(r"\b(?:episodes?|length|how long)\b", message_key):
            episodes = details.get("episodes")
            facts.append(
                f"It has {int(episodes)} episodes."
                if episodes is not None
                else "The catalog has no episode count for it."
            )
        if re.search(r"\b(?:who voices|voice actors?|cast)\b", message_key):
            roles = details.get("voice_actor_roles", []) or []
            cast = [
                f"{role.get('voice_actor')} as {role.get('character')}" for role in roles if role.get("voice_actor")
            ]
            facts.append(
                "Catalog voice credits include " + ", ".join(cast[:5]) + "."
                if cast
                else "The catalog has no voice-cast credits for it."
            )
        if re.search(r"\b(?:director|directed)\b", message_key):
            directors = [
                str(person.get("name"))
                for person in details.get("staff", [])
                if person.get("name") and "director" in str(person.get("role") or "").casefold()
            ]
            facts.append(
                "Its credited director is " + ", ".join(directors) + "."
                if directors
                else "The catalog has no director credit for it."
            )
        if not facts:
            return format_anime_introduction(details)
        return f"**{details['title']}** — " + " ".join(facts)

    def _respond_with_structured_search(self, intent: StructuredIntent, message: str) -> dict[str, Any]:
        query = next((mention.text for mention in intent.entity_mentions if mention.text), "") or message
        requested_types = [
            mention.entity_type for mention in intent.entity_mentions if mention.entity_type != "result_index"
        ]
        matches = self.entity_resolver.search(query, requested_types or None, limit=intent.top_k)
        if not matches:
            answer = "I could not find a confident catalog match. Try the full title or person's full name."
        else:
            lines = ["Here are the closest catalog matches:"]
            for match in matches:
                related = len(match["related_anime_ids"])
                lines.append(f"- {match['matched_name']} ({match['entity_type']}, {related} linked anime)")
            answer = "\n".join(lines)
        result = {"results": matches}
        return {
            "mode": "catalog_search",
            "answer": answer,
            "trace": [
                {
                    "tool": "search_entities",
                    "arguments": {"query": query, "entity_types": requested_types},
                    "result": result,
                }
            ],
            "agent": self.status(),
        }

    def _conversational_response(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        if not self.client.is_available():
            answer = "The local language model is offline. I can still search the catalog or rank anime if you ask for a title, genre, or mood."
            return {"mode": "offline", "answer": answer, "trace": [], "agent": self.status()}
        try:
            content = self.client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:] + [{"role": "user", "content": message}]
            )
        except OllamaUnavailable:
            content = "The local language model could not complete that request."
        return {"mode": "ollama", "answer": content, "trace": [], "agent": self.status()}

    def _merge_followup_intent(
        self,
        intent: StructuredIntent,
        message: str,
        history: list[dict[str, str]],
        session: dict[str, Any],
    ) -> StructuredIntent:
        if self._has_required_entity_constraint(intent):
            return intent
        explicit_titles = self._mentioned_catalog_titles(message)
        if (
            self._looks_like_catalog_ranking(message)
            or (
                explicit_titles
                and not re.search(
                    r"\b(?:this|that|it|these|those|option|result|number)\b",
                    message,
                    re.IGNORECASE,
                )
            )
            or (
                re.search(
                    r"^\s*(?:please\s+)?(?:find|recommend|suggest|show|give|list)\b",
                    message,
                    re.IGNORECASE,
                )
                and not re.search(
                    r"\b(?:more|another|different|same constraints|like (?:this|that|it))\b",
                    message,
                    re.IGNORECASE,
                )
            )
        ):
            return intent
        previous = session.get("last_recommendation_intent")
        is_followup = self._looks_like_recommendation_followup(message) or bool(
            intent.reference_result_indices or intent.watched_result_indices
        )
        if not is_followup or not isinstance(previous, dict):
            return intent

        list_fields = (
            "reference_titles",
            "include_genres",
            "exclude_genres",
            "required_studios",
            "preferred_studios",
            "required_staff",
            "preferred_staff",
            "required_characters",
            "preferred_characters",
            "required_voice_actors",
            "preferred_voice_actors",
            "formats",
        )
        for field_name in list_fields:
            if not getattr(intent, field_name):
                values = previous.get(field_name)
                if isinstance(values, list):
                    setattr(intent, field_name, [str(value) for value in values if value])
        for field_name in ("min_score", "min_year", "max_year", "max_episodes"):
            if getattr(intent, field_name) is None and previous.get(field_name) is not None:
                setattr(intent, field_name, previous[field_name])
        if self._requested_limit(message) is None and previous.get("top_k"):
            intent.top_k = max(1, min(int(previous["top_k"]), 50))
        previous_text = str(previous.get("free_text_preferences") or "").strip()
        if previous_text and previous_text.casefold() not in intent.free_text_preferences.casefold():
            intent.free_text_preferences = f"{previous_text}. {intent.free_text_preferences or message}".strip()
        return intent

    def _materialize_result_references(
        self,
        intent: StructuredIntent,
        message: str,
        session: dict[str, Any],
    ) -> str | None:
        """Resolve ordinal references before dispatching any intent/tool."""
        previous_results = [str(value) for value in session.get("last_recommendations", []) if value]
        classified_references, classified_watched = self._classified_result_indices(message)
        watched_indices = list(dict.fromkeys([*intent.watched_result_indices, *classified_watched]))
        reference_indices = list(
            dict.fromkeys(
                [
                    *intent.reference_result_indices,
                    *(index for index in classified_references if index not in watched_indices),
                ]
            )
        )
        if not watched_indices and not reference_indices:
            return None
        if not previous_results:
            return (
                "I do not have a previous result list in this session yet. "
                "Ask for recommendations first, then refer to an item by position."
            )
        invalid = [
            index for index in [*watched_indices, *reference_indices] if index < 1 or index > len(previous_results)
        ]
        if invalid:
            return f"The previous list has {len(previous_results)} items, so I cannot resolve position {invalid[0]}."

        for index in watched_indices:
            title = previous_results[index - 1]
            intent.seen_titles.append(title)
            intent.preference_update.watched_titles.append(title)
        for index in reference_indices:
            intent.reference_titles.append(previous_results[index - 1])
        detail_request = bool(
            re.search(
                r"\b(?:tell me about|introduce|overview|score|rating|studio|"
                r"who voices|voice actors?|cast|episodes?|director|who made)\b",
                message,
                re.IGNORECASE,
            )
        )
        recommendation_request = bool(
            re.search(
                r"\b(?:more like|similar to|recommend|suggest|find)\b",
                message,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:show|give)\b.{0,40}\b(?:anime|titles?|more|options?|recommendations?)\b",
                message,
                re.IGNORECASE,
            )
        )
        if reference_indices and detail_request and not recommendation_request:
            intent.intent = "details"
        elif watched_indices and not recommendation_request:
            intent.intent = "update_preferences"
        return None

    @staticmethod
    def _classified_result_indices(message: str) -> tuple[list[int], list[int]]:
        ordinal_pattern = "|".join(sorted((re.escape(value) for value in ORDINAL_WORDS), key=len, reverse=True))
        pattern = re.compile(
            rf"\b(?:the\s+)?(?P<word>{ordinal_pattern})\b"
            r"|\b(?:option|result|number)\s*(?P<number>\d{1,2})\b"
            r"|(?P<hash>#\s*(?P<hash_number>\d{1,2}))",
            re.IGNORECASE,
        )
        references: list[int] = []
        watched: list[int] = []
        message_key = message.casefold()
        for match in pattern.finditer(message):
            suffix = message_key[match.end() : match.end() + 32]
            if re.match(
                r"\s+(?:season|part|cour|movie|film|episode|highest|lowest|best|ranked)\b",
                suffix,
            ):
                continue
            if match.group("word"):
                index = ORDINAL_WORDS[match.group("word").casefold()]
            else:
                index = int(match.group("number") or match.group("hash_number"))
            if not 1 <= index <= 50:
                continue
            prefix = message_key[max(0, match.start() - 80) : match.start()]
            clause = re.split(r"[;,.!?]", prefix)[-1]
            watched_position = max(
                (clause.rfind(value) for value in ("watched", "seen", "finished", "saw")),
                default=-1,
            )
            reference_position = max(
                (
                    clause.rfind(value)
                    for value in (
                        "more like",
                        "similar to",
                        "tell me about",
                        "score",
                        "rating",
                        "studio",
                        "introduce",
                    )
                ),
                default=-1,
            )
            target = watched if watched_position > reference_position else references
            target.append(index)
        return (
            list(dict.fromkeys(references)),
            list(dict.fromkeys(watched)),
        )

    def _resolve_intent_entities(
        self,
        intent: StructuredIntent,
        session: dict[str, Any],
        required_mentions: set[tuple[str, str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        required_mentions = required_mentions or set()
        trace: list[dict[str, Any]] = []
        debug: dict[str, Any] = {
            "resolved_entity": None,
            "resolved_entities": [],
            "resolved_entity_id": None,
            "entity_type": None,
            "required_voice_actor_ids": [],
            "required_entity_constraints": [],
            "errors": [],
        }
        previous_results = [str(value) for value in session.get("last_recommendations", []) if value]

        for index in intent.reference_result_indices:
            if 1 <= index <= len(previous_results):
                intent.reference_titles.append(previous_results[index - 1])
        for index in intent.watched_result_indices:
            if 1 <= index <= len(previous_results):
                title = previous_results[index - 1]
                intent.seen_titles.append(title)
                intent.preference_update.watched_titles.append(title)

        canonical_required_actors: list[str] = []
        for actor_name in intent.required_voice_actors:
            matches = self.entity_resolver.search(
                actor_name,
                entity_types=["voice_actor"],
                limit=10,
                minimum_confidence=0.58,
            )
            match = self.entity_resolver.resolve(actor_name, "voice_actor", minimum_confidence=0.58)
            trace.append(
                {
                    "tool": "search_entities",
                    "arguments": {"query": actor_name, "entity_types": ["voice_actor"]},
                    "result": {"results": [self._compact_entity_resolution(value) for value in matches]},
                }
            )
            if not match:
                debug["errors"].append(
                    {"type": "unknown_voice_actor", "input": actor_name, "entity_type": "voice_actor"}
                )
                continue
            if match.get("ambiguous"):
                debug["errors"].append(
                    {
                        "type": "ambiguous_voice_actor",
                        "input": actor_name,
                        "entity_type": "voice_actor",
                        "matches": match.get("alternatives", []),
                    }
                )
                continue
            canonical_required_actors.append(match["matched_name"])
            if match.get("entity_id") is not None:
                debug["required_voice_actor_ids"].append(int(match["entity_id"]))
            debug.update(
                {
                    "resolved_entity": self._compact_entity_resolution(match),
                    "resolved_entity_id": match.get("entity_id"),
                    "entity_type": "voice_actor",
                }
            )
            debug["resolved_entities"].append(self._compact_entity_resolution(match))
        if canonical_required_actors:
            intent.required_voice_actors = canonical_required_actors

        required_field_specs = (
            ("required_studios", "studio"),
            ("required_staff", "staff"),
            ("required_characters", "character"),
        )
        for field_name, entity_type in required_field_specs:
            canonical_names: list[str] = []
            for name in getattr(intent, field_name):
                matches = self.entity_resolver.search(
                    name,
                    entity_types=[entity_type],
                    limit=10,
                    minimum_confidence=0.58,
                )
                match = self.entity_resolver.resolve(name, entity_type, minimum_confidence=0.58)
                trace.append(
                    {
                        "tool": "search_entities",
                        "arguments": {"query": name, "entity_types": [entity_type]},
                        "result": {"results": [self._compact_entity_resolution(value) for value in matches]},
                    }
                )
                if not match:
                    debug["errors"].append(
                        {
                            "type": "unknown_required_entity",
                            "input": name,
                            "entity_type": entity_type,
                        }
                    )
                    continue
                if match.get("ambiguous"):
                    debug["errors"].append(
                        {
                            "type": "ambiguous_required_entity",
                            "input": name,
                            "entity_type": entity_type,
                            "matches": match.get("alternatives", []),
                        }
                    )
                    continue
                canonical_names.append(match["matched_name"])
                debug["required_entity_constraints"].append(self._entity_constraint(match))
                debug["resolved_entities"].append(self._compact_entity_resolution(match))
                if debug["resolved_entity"] is None:
                    debug.update(
                        {
                            "resolved_entity": self._compact_entity_resolution(match),
                            "resolved_entity_id": match.get("entity_id"),
                            "entity_type": entity_type,
                        }
                    )
            if canonical_names:
                setattr(intent, field_name, canonical_names)

        for title in list(intent.reference_titles):
            match = self.entity_resolver.resolve(title, "anime")
            if match:
                trace.append(
                    {"tool": "resolve_entity", "arguments": {"query": title, "entity_type": "anime"}, "result": match}
                )
                if match["confidence"] >= 0.68:
                    intent.reference_titles.append(match["matched_name"])

        for mention in intent.entity_mentions:
            if mention.entity_type == "result_index":
                continue
            match = self.entity_resolver.resolve(mention.text, mention.entity_type)
            mention_key = (mention.entity_type, self._entity_name_key(mention.text))
            is_required = mention_key in required_mentions
            if (
                mention.entity_type == "voice_actor"
                and match
                and match.get("entity_id") in debug["required_voice_actor_ids"]
            ):
                continue
            trace.append(
                {
                    "tool": "resolve_entity",
                    "arguments": {
                        "query": mention.text,
                        "entity_type": mention.entity_type,
                        "relation": mention.relation,
                    },
                    "result": match,
                }
            )
            if not match or match["confidence"] < 0.68:
                if is_required:
                    debug["errors"].append(
                        {
                            "type": "unknown_required_entity",
                            "input": mention.text,
                            "entity_type": mention.entity_type,
                        }
                    )
                continue
            if is_required and match.get("ambiguous"):
                debug["errors"].append(
                    {
                        "type": "ambiguous_required_entity",
                        "input": mention.text,
                        "entity_type": mention.entity_type,
                        "matches": match.get("alternatives", []),
                    }
                )
                continue
            name = match["matched_name"]
            if is_required:
                debug["required_entity_constraints"].append(self._entity_constraint(match))
                debug["resolved_entities"].append(self._compact_entity_resolution(match))
                if debug["resolved_entity"] is None:
                    debug.update(
                        {
                            "resolved_entity": self._compact_entity_resolution(match),
                            "resolved_entity_id": match.get("entity_id"),
                            "entity_type": mention.entity_type,
                        }
                    )
                if mention.entity_type in {"genre", "theme", "demographic"}:
                    intent.include_genres.append(name)
                continue
            if mention.entity_type == "anime":
                if mention.relation == "exclude":
                    intent.excluded_titles.append(name)
                elif mention.relation == "watched":
                    intent.seen_titles.append(name)
                    intent.preference_update.watched_titles.append(name)
                elif mention.relation == "director_of":
                    details = self.recommender.details(int(match["anime_id"])) or {}
                    directors = [
                        person.get("name")
                        for person in details.get("staff", [])
                        if "director" in str(person.get("role") or "").casefold() and person.get("name")
                    ]
                    intent.preferred_staff.extend(directors)
                    if not directors:
                        debug["errors"].append(
                            {
                                "type": "missing_director_credit",
                                "input": name,
                                "entity_type": "director",
                            }
                        )
                    trace.append(
                        {
                            "tool": "get_anime_details",
                            "arguments": {"anime_id": match["anime_id"], "field": "director"},
                            "result": {"title": name, "directors": directors},
                        }
                    )
                else:
                    intent.reference_titles.append(name)
            elif mention.entity_type == "studio":
                intent.preferred_studios.append(name)
            elif mention.entity_type in {"staff", "producer", "director", "original_creator"}:
                intent.preferred_staff.append(name)
            elif mention.entity_type == "character":
                intent.preferred_characters.append(name)
            elif mention.entity_type == "voice_actor":
                intent.preferred_voice_actors.append(name)
            elif mention.entity_type == "genre" and name not in intent.include_genres:
                intent.include_genres.append(name)
            elif mention.entity_type in {"theme", "demographic"}:
                intent.free_text_preferences = f"{intent.free_text_preferences} {name}".strip()

        for field_name in (
            "reference_titles",
            "excluded_titles",
            "seen_titles",
            "required_studios",
            "preferred_studios",
            "required_staff",
            "preferred_staff",
            "required_characters",
            "preferred_characters",
            "required_voice_actors",
            "preferred_voice_actors",
        ):
            setattr(intent, field_name, list(dict.fromkeys(value for value in getattr(intent, field_name) if value)))
        intent.reference_titles = [
            title
            for title in intent.reference_titles
            if title.casefold() not in {value.casefold() for value in intent.excluded_titles}
        ]
        debug["required_voice_actor_ids"] = list(dict.fromkeys(debug["required_voice_actor_ids"]))
        unique_constraints: list[dict[str, Any]] = []
        constraint_keys: set[tuple[str, int | None, str]] = set()
        for constraint in debug["required_entity_constraints"]:
            key = (
                constraint["entity_type"],
                constraint.get("entity_id"),
                self._entity_name_key(constraint["matched_name"]),
            )
            if key not in constraint_keys:
                unique_constraints.append(constraint)
                constraint_keys.add(key)
        debug["required_entity_constraints"] = unique_constraints
        return trace, debug

    @staticmethod
    def _has_required_entity_constraint(intent: StructuredIntent) -> bool:
        return any(
            (
                intent.required_studios,
                intent.required_staff,
                intent.required_characters,
                intent.required_voice_actors,
            )
        )

    def _preference_patch_from_intent(self, intent: StructuredIntent) -> dict[str, Any]:
        update = intent.preference_update
        patch = {
            "liked_titles": update.liked_titles,
            "disliked_titles": update.disliked_titles,
            "seen_titles": list(dict.fromkeys(update.watched_titles + intent.seen_titles)),
            "excluded_titles": list(dict.fromkeys(update.excluded_titles + intent.excluded_titles)),
            "preferred_genres": update.preferred_genres,
            "excluded_genres": list(dict.fromkeys(update.excluded_genres + intent.exclude_genres)),
        }
        return {key: value for key, value in patch.items() if value}

    def _enforce_explicit_entity_constraints(self, intent: StructuredIntent, message: str) -> None:
        self._enforce_explicit_voice_actor_constraint(intent, message)
        self._recover_explicit_entity_mentions(intent, message)
        explicit_types = self._explicit_entity_types(message)
        explicit_names = {
            entity_type: {self._entity_name_key(value) for value in self._named_catalog_entities(message, entity_type)}
            for entity_type in (
                "studio",
                "staff",
                "character",
                "voice_actor",
                "producer",
                "director",
                "original_creator",
                "genre",
                "theme",
                "demographic",
            )
        }
        explicit_names["staff"].update(explicit_names["director"])
        explicit_names["staff"].update(explicit_names["original_creator"])
        if explicit_types:
            allowed_direct_types = {*explicit_types, "voice_actor"}
            intent.entity_mentions = [
                mention
                for mention in intent.entity_mentions
                if mention.relation != "direct"
                or (
                    mention.entity_type in allowed_direct_types
                    and (
                        self._entity_name_key(mention.text) in explicit_names.get(mention.entity_type, set())
                        or self._unresolved_entity_is_explicit(mention.text, mention.entity_type, message)
                    )
                )
            ]
            if "studio" not in explicit_types:
                intent.required_studios = []
            if "staff" not in explicit_types:
                intent.required_staff = []
            if "character" not in explicit_types:
                intent.required_characters = []
        else:
            intent.entity_mentions = [
                mention
                for mention in intent.entity_mentions
                if mention.relation != "direct"
                or mention.entity_type == "anime"
                or self._entity_name_key(mention.text) in explicit_names.get(mention.entity_type, set())
                or self._unresolved_entity_is_explicit(mention.text, mention.entity_type, message)
            ]

        intent.required_studios = [
            value
            for value in intent.required_studios
            if self._entity_name_key(value) in explicit_names["studio"]
            or self._unresolved_entity_is_explicit(value, "studio", message)
        ]
        intent.required_staff = [
            value
            for value in intent.required_staff
            if self._entity_name_key(value) in explicit_names["staff"]
            or self._unresolved_entity_is_explicit(value, "staff", message)
        ]
        intent.required_characters = [
            value
            for value in intent.required_characters
            if self._entity_name_key(value) in explicit_names["character"]
            or self._unresolved_entity_is_explicit(value, "character", message)
        ]
        intent.required_voice_actors = [
            value
            for value in intent.required_voice_actors
            if self._entity_name_key(value) in explicit_names["voice_actor"]
            or self._unresolved_entity_is_explicit(value, "voice_actor", message)
        ]
        required_mentions = self._required_entity_mentions(intent, message)
        for mention in intent.entity_mentions:
            key = (mention.entity_type, self._entity_name_key(mention.text))
            if key not in required_mentions:
                continue
            if mention.entity_type == "studio":
                intent.required_studios.append(mention.text)
            elif mention.entity_type == "staff":
                intent.required_staff.append(mention.text)
            elif mention.entity_type == "character":
                intent.required_characters.append(mention.text)
            elif mention.entity_type == "voice_actor":
                intent.required_voice_actors.append(mention.text)

        for required_name, preferred_name in (
            ("required_studios", "preferred_studios"),
            ("required_staff", "preferred_staff"),
            ("required_characters", "preferred_characters"),
            ("required_voice_actors", "preferred_voice_actors"),
        ):
            required_values = self._dedupe_entity_names(getattr(intent, required_name))
            required_keys = {self._entity_name_key(value) for value in required_values}
            setattr(intent, required_name, required_values)
            setattr(
                intent,
                preferred_name,
                [
                    value
                    for value in self._dedupe_entity_names(getattr(intent, preferred_name))
                    if self._entity_name_key(value) not in required_keys
                ],
            )

    def _recover_explicit_entity_mentions(self, intent: StructuredIntent, message: str) -> None:
        existing = {
            (mention.entity_type, self._entity_name_key(mention.text), mention.relation)
            for mention in intent.entity_mentions
        }
        for match in re.finditer(
            r"\b(?:the\s+)?director\s+of\s+([^,?.;]+)",
            message,
            re.IGNORECASE,
        ):
            candidate = re.sub(
                r"\s+(?:and|then|who|that|with|for)\b.*$",
                "",
                match.group(1),
                flags=re.IGNORECASE,
            ).strip()
            resolved_titles = self._mentioned_catalog_titles(candidate)
            if not resolved_titles:
                resolved = self.entity_resolver.resolve(candidate, "anime")
                if resolved and resolved.get("anime_id") is not None:
                    resolved_titles = [str(resolved["matched_name"])]
            for title in resolved_titles[:1]:
                key = ("anime", self._entity_name_key(title), "director_of")
                if key not in existing:
                    intent.entity_mentions.append(EntityMention(title, "anime", "director_of"))
                    existing.add(key)

        entity_types = self._explicit_entity_types(message)
        if not entity_types:
            return

        direct_existing = {
            (mention.entity_type, self._entity_name_key(mention.text)) for mention in intent.entity_mentions
        }
        for entity_type in entity_types:
            for name in self._named_catalog_entities(message, entity_type):
                key = (entity_type, self._entity_name_key(name))
                if key in direct_existing:
                    continue
                intent.entity_mentions.append(EntityMention(name, entity_type, "direct"))
                direct_existing.add(key)

    def _unresolved_entity_is_explicit(self, value: str, entity_type: str, message: str) -> bool:
        value_key = self._entity_name_key(value)
        if len(value_key.split()) < 2 or value_key not in self._entity_name_key(message):
            return False
        patterns = {
            "voice_actor": r"\b(?:voice[- ]?actor|voiced by|voice of)\b",
            "producer": r"\b(?:producer|produced by)\b",
            "director": r"\b(?:director|directed by)\b",
            "original_creator": r"\b(?:creator|created by|author|manga by)\b",
            "studio": r"\b(?:studio|from|made by|animated by)\b",
            "staff": r"\b(?:staff|written by|writer|music by|composer|director|creator)\b",
            "character": r"\b(?:character|featuring|contains?)\b",
            "genre": r"\bgenre\b",
            "theme": r"\btheme\b",
            "demographic": r"\b(?:demographic|audience)\b",
        }
        pattern = patterns.get(entity_type)
        return bool(pattern and re.search(pattern, message, re.IGNORECASE))

    def _explicit_entity_types(self, message: str) -> set[str]:
        if not self._explicit_relationship_request(message):
            return set()

        message_key = " ".join(message.casefold().split())
        if self._named_catalog_entities(message, "voice_actor") and re.search(
            r"\b(?:voice[- ]?actor|voiced|with|featuring|involving|involved)\b",
            message_key,
        ):
            return {"voice_actor"}
        entity_patterns = {
            "voice_actor": r"\bvoice[- ]?actor\b",
            "producer": r"\b(?:produced by|producer)\b",
            "director": r"\b(?:directed by|director)\b",
            "original_creator": r"\b(?:created by|original creator|manga by|author)\b",
            "studio": r"\b(?:studio|from)\b",
            "staff": r"\b(?:staff|written by|writer|music by|composed by|composer)\b",
            "character": r"\b(?:character|featuring|contains?)\b",
            "genre": r"\bgenre\b",
            "theme": r"\btheme\b",
            "demographic": r"\b(?:demographic|audience)\b",
        }
        entity_types = {
            entity_type for entity_type, pattern in entity_patterns.items() if re.search(pattern, message_key)
        }
        if entity_types:
            return entity_types
        if re.search(r"\bby\b", message_key):
            return {"studio", "producer", "staff", "director", "original_creator"}
        if re.search(r"\b(?:with|featuring|involving|contains?|has|have)\b", message_key):
            return {"staff", "character", "genre", "theme", "demographic"}
        return {
            "studio",
            "producer",
            "staff",
            "director",
            "original_creator",
            "character",
            "genre",
            "theme",
            "demographic",
        }

    def _required_entity_mentions(
        self,
        intent: StructuredIntent,
        message: str,
    ) -> set[tuple[str, str]]:
        required_names = {
            "studio": {self._entity_name_key(value) for value in intent.required_studios},
            "staff": {self._entity_name_key(value) for value in intent.required_staff},
            "director": {self._entity_name_key(value) for value in intent.required_staff},
            "original_creator": {self._entity_name_key(value) for value in intent.required_staff},
            "character": {self._entity_name_key(value) for value in intent.required_characters},
            "voice_actor": {self._entity_name_key(value) for value in intent.required_voice_actors},
        }
        explicit_relationship = self._explicit_relationship_request(message)
        explicit_types = self._explicit_entity_types(message)
        required: set[tuple[str, str]] = set()
        for mention in intent.entity_mentions:
            if mention.entity_type in {"anime", "result_index"} or mention.relation == "director_of":
                continue
            name_key = self._entity_name_key(mention.text)
            if name_key in required_names.get(mention.entity_type, set()) or (
                explicit_relationship and mention.relation == "direct" and mention.entity_type in explicit_types
            ):
                required.add((mention.entity_type, name_key))
        return required

    @staticmethod
    def _explicit_relationship_request(message: str) -> bool:
        message_key = " ".join(message.casefold().split())
        action = re.search(
            r"\b(?:recommend|suggest|find|show|give|list|looking for|want|need)\b",
            message_key,
        )
        relationship = re.search(
            r"\b(?:with|featuring|involving|from|by|made by|produced by|created by|has|have|contains?)\b",
            message_key,
        )
        return bool(action and relationship)

    @staticmethod
    def _entity_constraint(match: dict[str, Any]) -> dict[str, Any]:
        return {
            "entity_type": match.get("entity_type"),
            "entity_id": match.get("entity_id"),
            "matched_name": match.get("matched_name"),
            "related_anime_ids": [int(value) for value in match.get("related_anime_ids", [])],
        }

    def _enforce_explicit_voice_actor_constraint(self, intent: StructuredIntent, message: str) -> None:
        extracted = self._extract_required_voice_actors(message)
        if self._requires_voice_actor_membership(message):
            extracted.extend(
                mention.text
                for mention in intent.entity_mentions
                if mention.entity_type == "voice_actor" and mention.text
            )
            extracted.extend(intent.preferred_voice_actors)
        intent.required_voice_actors = self._dedupe_entity_names([*intent.required_voice_actors, *extracted])
        required_keys = {self._entity_name_key(value) for value in intent.required_voice_actors}
        intent.preferred_voice_actors = [
            value for value in intent.preferred_voice_actors if self._entity_name_key(value) not in required_keys
        ]
        if intent.required_voice_actors and " ".join(intent.free_text_preferences.casefold().split()) == " ".join(
            message.casefold().split()
        ):
            intent.free_text_preferences = ""

    def _extract_required_voice_actors(self, message: str) -> list[str]:
        if not self._requires_voice_actor_membership(message):
            return []
        names: list[str] = self._named_catalog_entities(message, "voice_actor")
        patterns = (
            r"\bvoice[- ]?actor\s+(?:called|named)\s+(.+?)(?=[.!?]|$)",
            r"\bvoiced\s+by\s+(.+?)(?=[.!?]|$)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                candidate = re.sub(
                    r"\s+(?:could|can|please|who|that)\b.*$",
                    "",
                    match.group(1),
                    flags=re.IGNORECASE,
                ).strip(" ,")
                if candidate:
                    names.append(candidate)
        return self._dedupe_entity_names(names)

    def _requires_voice_actor_membership(self, message: str) -> bool:
        explicit_pattern = bool(
            re.search(
                r"\b(?:anime|shows?|titles?)\b.{0,100}\b(?:have|has|with|featuring|involving)\b"
                r".{0,60}\b(?:him|her|them|voice[- ]?actor|involved)\b",
                message,
                re.IGNORECASE | re.DOTALL,
            )
            or re.search(r"\b(?:voiced by|featuring the voice of)\b", message, re.IGNORECASE)
        )
        named_actor_relationship = bool(
            self._named_catalog_entities(message, "voice_actor")
            and re.search(
                r"\b(?:recommend|suggest|find|show|give|list|with|featuring|involving|involved|voiced)\b",
                message,
                re.IGNORECASE,
            )
        )
        return explicit_pattern or named_actor_relationship

    @classmethod
    def _dedupe_entity_names(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = cls._entity_name_key(text)
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    @staticmethod
    def _entity_name_key(value: str) -> str:
        return " ".join("".join(character if character.isalnum() else " " for character in value.casefold()).split())

    @staticmethod
    def _compact_entity_resolution(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_text": value.get("input_text"),
            "entity_type": value.get("entity_type"),
            "matched_name": value.get("matched_name"),
            "entity_id": value.get("entity_id"),
            "anime_id": value.get("anime_id"),
            "confidence": value.get("confidence"),
            "resolution_method": value.get("resolution_method"),
            "ambiguous": bool(value.get("ambiguous")),
            "related_anime_count": len(value.get("related_anime_ids", [])),
        }

    def _is_supported_english(self, message: str) -> bool:
        letters = [character for character in message if character.isalpha()]
        if not letters:
            return True
        unsupported = sum(1 for character in letters if ord(character) > 591)
        return unsupported / len(letters) < 0.2

    def _looks_like_recommendation_request(self, message: str) -> bool:
        tokens = set(re.findall(r"[a-z]+", message.casefold()))
        active_request_words = tokens.intersection(
            {"find", "give", "recommend", "suggest", "show", "similar", "watch", "watching", "want", "looking"}
        )
        if self._looks_like_negative_feedback(message) and not active_request_words:
            return False
        return bool(tokens.intersection(RECOMMENDATION_INTENTS))

    def _looks_like_introduction_request(self, message: str) -> bool:
        message_key = " ".join(message.casefold().split())
        return any(pattern in message_key for pattern in INTRODUCTION_PATTERNS)

    def _respond_with_catalog_introduction(self, message: str) -> dict[str, Any] | None:
        target = self._resolve_introduction_target(message)
        if not target:
            return None

        details = self.recommender.details(int(target["id"]))
        if not details:
            return None

        trace = [
            {
                "tool": "get_anime_details",
                "arguments": {"anime_id": details["id"]},
                "result": compact_result({"result": details}),
            }
        ]
        fallback = format_anime_introduction(details)

        if not self.client.is_available():
            return {
                "mode": "catalog_introduction",
                "answer": fallback,
                "trace": trace,
                "agent": self.status(),
            }

        try:
            safe_details = {**details, "synopsis": spoiler_safe_synopsis(details.get("synopsis"))}
            content = self.client.chat(
                [
                    {"role": "system", "content": introduction_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"User question:\n{message}\n\n"
                            "Trusted catalog details JSON:\n" + json.dumps(safe_details, ensure_ascii=False)[:14000]
                        ),
                    },
                ]
            )
        except OllamaUnavailable:
            content = fallback

        if not self._valid_introduction_answer(content, details["title"]):
            content = fallback

        return {
            "mode": "ollama" if content != fallback else "catalog_introduction",
            "answer": content,
            "trace": trace,
            "agent": self.status(),
        }

    def _resolve_introduction_target(self, message: str) -> dict[str, Any] | None:
        mentioned_titles = self._mentioned_catalog_titles(message)
        if mentioned_titles:
            anime_ids = self.recommender.resolve_titles([mentioned_titles[0]])
            if anime_ids:
                return self.recommender.details(anime_ids[0])

        query = self._introduction_search_query(message)
        if len(tokenize(query)) < 2:
            return None

        matches = self.recommender.search(query, limit=3)
        return matches[0] if matches else None

    def _introduction_search_query(self, message: str) -> str:
        query = " ".join(message.strip().rstrip("?!. ").split())
        prefixes = (
            r"^(?:please\s+)?introduce(?:\s+the\s+anime)?\s+",
            r"^(?:please\s+)?tell\s+me\s+about(?:\s+the\s+anime)?\s+",
            r"^(?:please\s+)?give\s+me\s+an\s+overview\s+of(?:\s+the\s+anime)?\s+",
            r"^(?:what\s+is|what's)(?:\s+the\s+anime)?\s+",
            r"^(?:please\s+)?(?:describe|explain)(?:\s+the\s+anime)?\s+",
            r"^(?:synopsis|plot|cast)\s+of\s+",
        )
        for pattern in prefixes:
            cleaned = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
            if cleaned != query:
                return cleaned
        return query

    def _valid_introduction_answer(self, content: str, title: str) -> bool:
        if not content or len(content.strip()) < 80:
            return False
        return title.casefold() in content.casefold()

    def _should_use_catalog_recommender(self, message: str, history: list[dict[str, str]]) -> bool:
        return (
            self._looks_like_recommendation_request(message)
            or (self._looks_like_negative_feedback(message) and self._has_recent_recommendations(history))
            or (self._looks_like_recommendation_followup(message) and self._has_recent_recommendations(history))
        )

    def _looks_like_recommendation_followup(self, message: str) -> bool:
        message_key = " ".join(message.casefold().split())
        deictic = re.search(
            r"\b(?:more like (?:that|this|it|those|these)|another one|"
            r"(?:the )?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)|"
            r"(?:option|result|number|#)\s*\d{1,2})\b",
            message_key,
        )
        if deictic:
            return True
        if re.search(r"\bmore like\s+\S", message_key):
            return False
        if re.match(
            r"^(?:please\s+)?(?:give|show|recommend|suggest)\b.{0,40}\b"
            r"(?:more|another|new batch|fresh batch|different (?:ones|options))\b",
            message_key,
        ):
            return True
        if re.search(
            r"\b(?:recommend|suggest|find|show|give|list|similar to|looking for|want|need)\b",
            message_key,
        ):
            return False
        if re.match(
            r"^(?:please\s+)?(?:more|another|other|different|new batch|fresh batch|"
            r"no repeats|do not repeat|don't repeat)\b",
            message_key,
        ):
            return True
        return bool(
            re.search(
                r"\b(?:only|instead|this time|score|rating|at least|above|or later|"
                r"shorter|not too long|weekend)\b",
                message_key,
            )
        )

    def _looks_like_negative_feedback(self, message: str) -> bool:
        message_key = message.casefold()
        return any(pattern in message_key for pattern in NEGATIVE_FEEDBACK_PATTERNS)

    def _looks_like_same_series_feedback(self, message: str) -> bool:
        message_key = message.casefold()
        return "same series" in message_key or "same franchise" in message_key

    @staticmethod
    def _looks_like_series_lookup_request(message: str) -> bool:
        message_key = " ".join(message.casefold().split())
        if re.search(
            r"\b(?:no|without|avoid|exclude|don't want|do not want)\s+(?:any\s+)?sequels?\b",
            message_key,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:sequels?\s+(?:to|of|for)|"
                r"(?:other|more)\s+(?:entries|titles)\s+(?:in|from)\s+(?:the\s+)?same\s+franchise|"
                r"(?:what|which)\s+comes\s+(?:next|after))\b",
                message_key,
            )
        )

    def _looks_like_new_batch_request(self, message: str) -> bool:
        message_key = message.casefold()
        return any(pattern in message_key for pattern in NEW_BATCH_PATTERNS)

    def _respond_with_catalog_recommendations(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        history = history or []
        source_message = self._recommendation_source_message(message, history)
        excluded_titles = self._collect_negative_feedback_titles(message, history)
        excluded_ids = set(self.recommender.resolve_titles(excluded_titles))
        preference_patch = self._extract_preference_patch(message)
        preference_result = None
        if preference_patch:
            preference_result = self._tool_update_session({"session_id": session_id, **preference_patch})
        seed_titles = [
            title
            for title in self._mentioned_catalog_titles(source_message)
            if not set(self.recommender.resolve_titles([title])).intersection(excluded_ids)
        ]
        matched_genres = self._matched_genres(source_message)
        requested_limit = self._requested_limit(message) or self._requested_limit(source_message) or 5
        min_score = self._requested_min_score(message)
        if min_score is None:
            min_score = self._requested_min_score(source_message)
        if min_score is None:
            min_score = 7.0
        min_year = self._requested_min_year(message)
        if min_year is None:
            min_year = self._requested_min_year(source_message)
        max_episodes = self._requested_max_episodes(message)
        if max_episodes is None:
            max_episodes = self._requested_max_episodes(source_message)
        one_per_series = self._requested_one_per_series(message, history)
        tool_arguments = {
            "reference_titles": seed_titles,
            "excluded_titles": excluded_titles,
            "include_genres": matched_genres,
            "query": source_message,
            "min_score": min_score,
            "min_year": min_year,
            "max_episodes": max_episodes,
            "one_per_series": one_per_series,
            "top_k": requested_limit,
            "session_id": session_id,
        }
        result = self._tool_recommend(tool_arguments)
        trace = [{"tool": "recommend_anime", "arguments": tool_arguments, "result": compact_result(result)}]
        if preference_result is not None:
            trace.append(
                {
                    "tool": "update_session_preferences",
                    "arguments": preference_patch,
                    "result": preference_result,
                }
            )
        fallback = format_recommendation_answer(result["results"], limit=requested_limit)
        if excluded_titles:
            fallback = "I will avoid the titles you rejected in this chat.\n\n" + fallback

        try:
            content = self.client.chat(
                [
                    {"role": "system", "content": catalog_summary_prompt(requested_limit)},
                    {
                        "role": "user",
                        "content": (
                            f"User request:\n{message}\n\n"
                            f"Recommendation context:\n{source_message}\n\n"
                            "Titles to avoid because the user rejected them:\n"
                            + json.dumps(excluded_titles, ensure_ascii=False)
                            + "\n\n"
                            "Catalog recommendation JSON:\n" + json.dumps(result, ensure_ascii=False)[:12000]
                        ),
                    },
                ]
            )
        except OllamaUnavailable:
            return self._agent_error_response(
                "Ollama is reachable, but the chat request failed or timed out. "
                "I used the local catalog recommender instead.",
                fallback,
                trace,
            )

        result_titles = [item["title"] for item in result["results"]]
        if not self._valid_recommendation_answer(content, result_titles, excluded_titles, requested_limit):
            content = fallback

        return {
            "mode": "ollama",
            "answer": content,
            "trace": trace,
            "agent": self.status(),
        }

    def _agent_error_response(self, prefix: str, fallback: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "mode": "catalog_fallback",
            "answer": f"{prefix}\n\n{fallback}",
            "trace": trace,
            "agent": self.status(),
        }

    def _recommendation_source_message(self, message: str, history: list[dict[str, str]]) -> str:
        if not self._looks_like_negative_feedback(message) and not self._looks_like_recommendation_followup(message):
            return message

        context = self._latest_base_recommendation_request(history)
        if context:
            return f"{context}\n{message}"

        return message

    def _latest_base_recommendation_request(self, history: list[dict[str, str]]) -> str:
        for item in reversed(history[-10:]):
            if item.get("role") != "user":
                continue
            content = item.get("content", "")
            if (
                content
                and self._looks_like_recommendation_request(content)
                and not self._looks_like_negative_feedback(content)
            ):
                if self._looks_like_one_per_series_request(content) and not self._has_catalog_signal(content):
                    continue
                if self._looks_like_recommendation_followup(content) and not self._has_catalog_signal(content):
                    continue
                return content
        return ""

    def _has_recent_recommendations(self, history: list[dict[str, str]]) -> bool:
        return any(
            item.get("role") == "assistant" and self._extract_recommended_titles(item.get("content", ""))
            for item in history[-6:]
        )

    def _has_catalog_signal(self, message: str) -> bool:
        has_genre = bool(self._matched_genres(message))
        return has_genre or bool(self._mentioned_catalog_titles(message))

    def _collect_negative_feedback_titles(self, message: str, history: list[dict[str, str]]) -> list[str]:
        titles: list[str] = []
        previous_assistant = ""

        for item in history[-12:]:
            role = item.get("role")
            content = item.get("content", "")
            if not content:
                continue
            if role == "assistant":
                previous_assistant = content
                continue
            if role == "user" and (
                self._looks_like_negative_feedback(content)
                or self._looks_like_same_series_feedback(content)
                or self._looks_like_new_batch_request(content)
            ):
                previous_titles = self._extract_recommended_titles(previous_assistant)
                titles.extend(self._negative_feedback_targets(content, previous_titles))

        if (
            self._looks_like_negative_feedback(message)
            or self._looks_like_same_series_feedback(message)
            or self._looks_like_new_batch_request(message)
        ):
            previous_titles = self._extract_recommended_titles(previous_assistant)
            titles.extend(self._negative_feedback_targets(message, previous_titles))

        return list(dict.fromkeys(title for title in titles if title))

    def _negative_feedback_targets(self, message: str, previous_titles: list[str]) -> list[str]:
        targets = self._mentioned_catalog_titles(message)
        for index in self._referenced_result_indices(message):
            if 1 <= index <= len(previous_titles):
                targets.append(previous_titles[index - 1])

        message_key = message.casefold()
        entire_batch = self._looks_like_new_batch_request(message) or any(
            phrase in message_key
            for phrase in (
                "none of these",
                "not these",
                "bad recommendations",
                "same recommendations",
                "different options",
                "different ones",
            )
        )
        if entire_batch:
            targets.extend(previous_titles)
        if self._looks_like_same_series_feedback(message):
            series_targets = targets or previous_titles
            targets.extend(self._same_series_titles(series_targets))
        return list(dict.fromkeys(targets))

    @staticmethod
    def _referenced_result_indices(message: str) -> list[int]:
        message_key = message.casefold()
        indices = [
            int(match.group(1)) for match in re.finditer(r"\b(?:option|result|number|#)\s*(\d{1,2})\b", message_key)
        ]
        for word, value in ORDINAL_WORDS.items():
            if re.search(rf"\b(?:the\s+)?{word}\b", message_key):
                indices.append(value)
        return list(dict.fromkeys(index for index in indices if 1 <= index <= 50))

    def _same_series_titles(self, titles: list[str]) -> list[str]:
        keys = {series_key(title) for title in titles if title}
        if not keys:
            return []

        matches: list[str] = []
        for item in self.recommender.catalog:
            if series_key(item["title"]) in keys:
                matches.append(item["title"])
        return matches

    def _shared_title_prefix(self, titles: list[str]) -> list[str]:
        token_lists = [tokenize(title) for title in titles if title]
        if len(token_lists) < 2:
            return []

        prefix: list[str] = []
        for values in zip(*token_lists, strict=False):
            if len(set(values)) != 1:
                break
            prefix.append(values[0])

        return prefix

    def _requested_limit(self, message: str) -> int | None:
        message_key = message.casefold()
        limit_text = re.sub(
            r"\bonly\s+recommend\s+one\s+(?:anime|title)?\s*from\s+each\s+series?\b",
            "",
            message_key,
        )
        patterns = (
            r"\b(?:give|recommend|suggest|find|show)\s+(?:me\s+)?(?:the\s+)?(\d{1,2})\b",
            r"\btop\s+(\d{1,2})\b",
            r"\banother\s+(\d{1,2})\b",
            r"\b(\d{1,2})\s+(?:anime|animes|titles|recommendations|options)\b",
            r"\b(\d{1,2})\s+more\b",
        )
        for pattern in patterns:
            match = re.search(pattern, limit_text)
            if match:
                return max(1, min(int(match.group(1)), 50))

        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\s+(?:anime|animes|titles|recommendations|options|more)\b", limit_text):
                return value

        return None

    def _requested_min_score(self, message: str) -> float | None:
        message_key = message.casefold()
        strict = re.search(
            r"\b(?:score|rating|rated)[^\d]{0,30}?(?:above|over|>)\s*(\d+(?:\.\d+)?)",
            message_key,
        )
        if strict:
            score = float(strict.group(1))
            if 0 <= score <= 10:
                return min(10.0, score + 0.01)
        patterns = (
            r"\b(?:score|rating|rated)[^\d]{0,30}(at least|minimum|min|>=|above|over|>|or above)?\s*"
            r"(\d+(?:\.\d+)?)",
            r"\b(\d+(?:\.\d+)?)\s*(or above|and above|\+)\b",
        )
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, message_key)
            if match:
                qualifier = (match.group(1) if index == 0 else match.group(2)) or ""
                score = float(match.group(2) if index == 0 else match.group(1))
                if 0 <= score <= 10:
                    if qualifier in {"above", "over", ">"}:
                        return min(10.0, score + 0.01)
                    return score
        return None

    def _requested_min_year(self, message: str) -> int | None:
        message_key = message.casefold()
        patterns = (
            (r"\b(19\d{2}|20\d{2})\s*(?:or later|and later|\+|onward|onwards)\b", 0),
            (r"\b(?:from|since)\s+(19\d{2}|20\d{2})\b", 0),
            (r"\bafter\s+(19\d{2}|20\d{2})\b", 1),
        )
        for pattern, offset in patterns:
            match = re.search(pattern, message_key)
            if match:
                return int(match.group(1)) + offset

        return None

    @staticmethod
    def _requested_max_year(message: str) -> int | None:
        message_key = message.casefold()
        patterns = (
            (r"\b(?:through|until|up to|no later than|at most)\s+(19\d{2}|20\d{2})\b", 0),
            (r"\b(?:before|older than)\s+(19\d{2}|20\d{2})\b", -1),
            (r"\b(19\d{2}|20\d{2})\s+or earlier\b", 0),
        )
        for pattern, offset in patterns:
            match = re.search(pattern, message_key)
            if match:
                return int(match.group(1)) + offset
        return None

    def _requested_max_episodes(self, message: str) -> int | None:
        message_key = message.casefold()
        patterns = (
            (r"\b(\d{1,3})\s+episodes?\s+(?:or\s+)?(?:fewer|less|under|max|maximum|at most)\b", 0),
            (r"\b(\d{1,3})\s+or\s+fewer\s+episodes?\b", 0),
            (r"\b(?:at most|max|maximum)\s+(\d{1,3})\s+episodes?\b", 0),
            (r"\b(?:under|fewer than|less than)\s+(\d{1,3})\s+episodes?\b", -1),
            (r"\bno more than\s+(\d{1,3})\s+episodes?\b", 0),
            (r"\b(?:keep it|stay)\s+(?:under|below)\s+(\d{1,3})\s+episodes?\b", -1),
        )
        for pattern, offset in patterns:
            match = re.search(pattern, message_key)
            if match:
                return max(1, int(match.group(1)) + offset)

        return None

    def _requested_one_per_series(self, message: str, history: list[dict[str, str]]) -> bool:
        texts = [message] + [item.get("content", "") for item in history[-12:] if item.get("role") == "user"]
        return any(self._looks_like_one_per_series_request(text) for text in texts)

    def _looks_like_one_per_series_request(self, message: str) -> bool:
        text_key = message.casefold()
        if any(pattern in text_key for pattern in ONE_PER_SERIES_PATTERNS):
            return True
        if "only" in text_key and "one" in text_key and ("series" in text_key or "serie" in text_key):
            return True
        return False

    @staticmethod
    def _looks_like_catalog_ranking(message: str) -> bool:
        text = message.casefold()
        return bool(
            re.search(
                r"\b(?:top|highest|lowest|best|worst|most popular|least popular|newest|latest|oldest|earliest)\b",
                text,
            )
            and re.search(r"\b(?:anime|movie|movies|ona|ova|special|tv|title|titles|series)\b", text)
        ) or bool(re.search(r"\b(?:rank|sort)\b.+\b(?:score|rating|popularity|members|year|title)\b", text))

    @staticmethod
    def _requested_rank_field(message: str) -> str:
        text = message.casefold()
        if "popular" in text or "popularity" in text:
            return "popularity"
        if "member" in text:
            return "members"
        if re.search(r"\b(?:newest|latest|oldest|earliest|year)\b", text):
            return "start_year"
        if re.search(r"\b(?:alphabetical|alphabetically|title)\b", text) and re.search(r"\b(?:rank|sort)\b", text):
            return "title"
        if re.search(r"\bcatalog rank\b|\brank(?:ed)? by rank\b", text):
            return "rank"
        return "score"

    @classmethod
    def _requested_sort_order(cls, message: str) -> str | None:
        text = message.casefold()
        if re.search(r"\b(?:ascending|lowest|worst|least popular|oldest|earliest)\b", text):
            return "asc"
        if re.search(r"\b(?:descending|highest|best|most popular|newest|latest)\b", text):
            return "desc"
        return None

    def _matched_formats(self, message: str) -> list[str]:
        found = []
        for media_type in self.recommender.meta()["types"]:
            if re.search(rf"(?<![a-z0-9]){re.escape(media_type.casefold())}(?![a-z0-9])", message.casefold()):
                found.append(media_type)
        return found

    def _catalog_ranking_query(self, message: str) -> str:
        text = message
        removable = [
            *self._matched_formats(message),
            *self._matched_genres(message),
            *self._named_catalog_entities(message, "studio"),
        ]
        for value in sorted(removable, key=len, reverse=True):
            text = re.sub(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", " ", text, flags=re.IGNORECASE)
        excluded_numbers = {
            str(value)
            for value in (
                self._requested_limit(message),
                self._requested_min_year(message),
                self._requested_max_year(message),
                self._requested_max_episodes(message),
            )
            if value is not None
        }
        excluded_numbers.update(re.findall(r"\b(?:19|20)\d{2}\b", message))
        text = re.sub(
            r"\b(?:show|find|list|give|tell|rank|ranked|ranking|sort|sorted|me|the|a|an|by|of|on|catalog|"
            r"top|highest|lowest|best|worst|most|least|popular|popularity|score|scored|rating|rated|"
            r"scores|members?|newest|latest|oldest|earliest|year|ascending|descending|anime|titles?|series|"
            r"movies?|after|before|since|through|until|later|earlier|episodes?|under|above|over|minimum|"
            r"maximum|max|released?|aired|airing|based|according|using|with|that|match|matching|only|"
            r"want|need|please|could|would|can|you|i)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        query_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text) if token not in excluded_numbers]
        return " ".join(query_tokens).strip()

    def _matched_genres(self, message: str) -> list[str]:
        message_key = f" {message.casefold()} "
        found: list[str] = []

        for genre in self.recommender.meta()["genres"]:
            if genre.casefold() in message_key and genre not in found:
                found.append(genre)

        for alias, genre in GENRE_ALIASES.items():
            alias_key = alias.casefold()
            if alias_key in message_key and genre not in found:
                found.append(genre)

        return found

    def _matched_genre_constraints(self, message: str) -> tuple[list[str], list[str]]:
        message_key = message.casefold()
        included: list[str] = []
        excluded: list[str] = []
        aliases_by_genre: dict[str, list[str]] = {}
        for alias, genre in GENRE_ALIASES.items():
            aliases_by_genre.setdefault(genre, []).append(alias)

        for genre in self._matched_genres(message):
            terms = [genre, *aliases_by_genre.get(genre, [])]
            negative = False
            for term in sorted(set(terms), key=len, reverse=True):
                for match in re.finditer(
                    rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])",
                    message_key,
                ):
                    prefix = message_key[max(0, match.start() - 60) : match.start()]
                    prefix = re.split(r"[,.;]|\bbut\b", prefix)[-1]
                    if re.search(
                        r"(?:without|anything except|except|avoid|exclude|no|not interested in|"
                        r"don't like|do not like|dislike|hate)(?:[\s,;/&]+[a-z-]+){0,5}[\s,;/&]*$",
                        prefix,
                    ):
                        negative = True
            target = excluded if negative else included
            if genre not in target:
                target.append(genre)

        excluded_keys = {value.casefold() for value in excluded}
        included = [value for value in included if value.casefold() not in excluded_keys]
        return included, excluded

    def _valid_recommendation_answer(
        self,
        content: str,
        result_titles: list[str],
        excluded_titles: list[str],
        requested_limit: int,
        required_voice_actors: list[str] | None = None,
    ) -> bool:
        if not content or not result_titles:
            return False

        content_key = content.casefold()
        if any(title.casefold() in content_key for title in excluded_titles):
            return False
        if any(
            term in content_key
            for term in (
                "dense text",
                "embedding match",
                "tf-idf",
                "latent semantic",
                "lsa similarity",
                "metadata similarity",
                "hybrid score",
                "session profile",
            )
        ):
            return False
        if any(marker in content_key for marker in ("###", "shared elements", "differences", "breakdown")):
            return False

        expected = min(requested_limit, len(result_titles))
        catalog_lines = 0
        for line in content.splitlines():
            if not re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
                continue
            matched_title = self._title_at_list_item_start(line, result_titles)
            if matched_title:
                if required_voice_actors and not all(
                    actor.casefold() in line.casefold() for actor in required_voice_actors
                ):
                    return False
                catalog_lines += 1
            elif re.match(r"^\s*\d+[.)]\s+", line) or not self._allowed_explanatory_bullet(line):
                return False

        return catalog_lines >= expected

    @staticmethod
    def _title_at_list_item_start(line: str, result_titles: list[str]) -> str | None:
        payload = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        payload = payload.lstrip("*`").casefold()
        for title in sorted(result_titles, key=len, reverse=True):
            title_key = title.casefold()
            if not payload.startswith(title_key):
                continue
            remainder = payload[len(title_key) :]
            if not remainder or re.match(r"^(?:\*\*|``)?\s*(?:\(|:|-|–|—)", remainder):
                return title
        return None

    @staticmethod
    def _allowed_explanatory_bullet(line: str) -> bool:
        payload = re.sub(r"^\s*[-*]\s+", "", line).strip().casefold()
        return bool(
            re.match(
                r"(?:why (?:these|it) fit|what to expect|note|constraint|common thread|best for|tone|themes?)\b",
                payload,
            )
        )

    def _extract_recommended_titles(self, text: str) -> list[str]:
        titles: list[str] = []
        for line in text.splitlines():
            match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", line)
            if not match:
                continue

            candidate = match.group(1).strip()
            bold = re.match(r"\*\*(.+?)\*\*", candidate)
            if bold:
                candidate = bold.group(1)
            else:
                candidate = re.split(r"\s+\(|:|\s+-\s+", candidate, maxsplit=1)[0]
            candidate = candidate.strip(" *`")

            title = self._canonical_catalog_title(candidate)
            if title:
                titles.append(title)

        return list(dict.fromkeys(titles))

    def _mentioned_catalog_titles(self, text: str) -> list[str]:
        text_key = text.casefold()
        text_tokens = set(tokenize(text))
        titles: list[str] = []

        for alias, title in TITLE_ALIASES.items():
            if alias.casefold() in text_key and self.recommender.resolve_titles([title]):
                titles.append(title)

        for item in self.recommender.search(text, limit=10):
            title = item["title"]
            variants = [str(value) for value in [title, *(item.get("aliases") or [])] if value]
            mentioned = False
            for variant in variants:
                variant_tokens = {
                    token for token in tokenize(variant) if len(token) > 2 and token not in GENERIC_TITLE_TOKENS
                }
                if variant.casefold() in text_key or (variant_tokens and variant_tokens.issubset(text_tokens)):
                    mentioned = True
                    break
            if mentioned:
                titles.append(title)

        return list(dict.fromkeys(titles))

    def _canonical_title_key(self, value: str) -> str:
        resolved = self.recommender.resolve_titles([value])
        if resolved:
            return str(self.recommender.by_id[resolved[0]]["title"]).casefold()
        return str(value).strip().casefold()

    def _explicitly_excluded_titles(self, message: str) -> list[str]:
        message_key = self._entity_name_key(message)
        excluded: list[str] = []
        for title in self._mentioned_catalog_titles(message):
            title_key = self._entity_name_key(title)
            positions = [
                match.start()
                for match in re.finditer(
                    rf"(?:^|\s){re.escape(title_key)}(?:$|\s)",
                    message_key,
                )
            ]
            for position in positions:
                prefix = message_key[max(0, position - 90) : position]
                prefix = re.split(r"[,.;]|\bbut\b", prefix)[-1]
                if re.search(
                    r"(?:not|no|without|except|anything but|exclude|excluding|avoid|"
                    r"don't include|do not include|don't want|do not want)"
                    r"(?:\s+[a-z0-9'-]+){0,6}\s*$",
                    prefix,
                ):
                    excluded.append(title)
                    break
        return list(dict.fromkeys(excluded))

    def _canonical_catalog_title(self, candidate: str) -> str | None:
        if not candidate:
            return None

        matches = self.recommender.search(candidate, limit=1)
        if not matches:
            return None

        title = matches[0]["title"]
        candidate_tokens = set(tokenize(candidate))
        title_tokens = set(tokenize(title))
        if not candidate_tokens or candidate_tokens.isdisjoint(title_tokens):
            return None

        return title

    def _offline_response(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        history = history or []
        source_message = self._recommendation_source_message(message, history)
        excluded_titles = self._collect_negative_feedback_titles(message, history)
        seed_titles = [
            title for title in self._mentioned_catalog_titles(source_message) if title not in excluded_titles
        ]
        preference_patch = self._extract_preference_patch(message)
        if preference_patch:
            self._tool_update_session({"session_id": session_id, **preference_patch})
        matched_genres = self._matched_genres(source_message)
        requested_limit = self._requested_limit(message) or self._requested_limit(source_message) or 5
        min_score = self._requested_min_score(message)
        if min_score is None:
            min_score = self._requested_min_score(source_message)
        if min_score is None:
            min_score = 7.0
        min_year = self._requested_min_year(message)
        if min_year is None:
            min_year = self._requested_min_year(source_message)
        max_episodes = self._requested_max_episodes(message)
        if max_episodes is None:
            max_episodes = self._requested_max_episodes(source_message)
        one_per_series = self._requested_one_per_series(message, history)
        picks = self.recommender.recommend(
            reference_titles=seed_titles,
            excluded_titles=excluded_titles,
            include_genres=matched_genres,
            min_score=min_score,
            min_year=min_year,
            max_episodes=max_episodes,
            query=source_message,
            one_per_series=one_per_series,
            session_profile=self.get_session_profile(session_id),
            limit=requested_limit,
        )
        lines = [
            "The local LLM agent is offline. Start Ollama and run `ollama pull gemma3:12b` to enable the full tool-calling agent.",
            "",
            format_recommendation_answer(picks, limit=requested_limit),
        ]
        if excluded_titles:
            lines.insert(2, "I will avoid the titles you rejected in this chat.")
            lines.insert(3, "")

        return {
            "mode": "offline",
            "answer": "\n".join(lines),
            "trace": [],
            "agent": self.status(),
        }

    def _tool_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        limit = int(arguments.get("limit") or 5)
        return {"results": self.recommender.search(query, limit=max(1, min(limit, 50)))}

    def _tool_rank_catalog(self, arguments: dict[str, Any]) -> dict[str, Any]:
        results, diagnostics = self.recommender.rank_catalog(
            query=str(arguments.get("query") or ""),
            include_genres=[str(value) for value in arguments.get("include_genres") or []],
            exclude_genres=[str(value) for value in arguments.get("exclude_genres") or []],
            formats=[str(value) for value in arguments.get("formats") or []],
            required_studios=[str(value) for value in arguments.get("required_studios") or []],
            min_score=float(arguments["min_score"]) if arguments.get("min_score") is not None else None,
            min_year=int(arguments["min_year"]) if arguments.get("min_year") is not None else None,
            max_year=int(arguments["max_year"]) if arguments.get("max_year") is not None else None,
            max_episodes=int(arguments["max_episodes"]) if arguments.get("max_episodes") is not None else None,
            excluded_titles=[str(value) for value in arguments.get("excluded_titles") or []],
            sort_by=str(arguments.get("sort_by") or "score"),
            sort_order=str(arguments["sort_order"]) if arguments.get("sort_order") else None,
            limit=max(1, min(int(arguments.get("top_k") or 10), 50)),
        )
        return {"results": results, "diagnostics": diagnostics}

    def _tool_search_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        limit = max(1, min(int(arguments.get("limit") or 10), 20))
        entity_types = [str(value) for value in arguments.get("entity_types") or []]
        return {"results": self.entity_resolver.search(query, entity_types or None, limit=limit)}

    def _extract_preference_patch(self, message: str) -> dict[str, Any]:
        message_key = message.casefold()
        patch: dict[str, Any] = {}
        matched_titles = self._mentioned_catalog_titles(message)
        matched_genres, excluded_genres = self._matched_genre_constraints(message)

        if "reset" in message_key and (
            "session" in message_key or "preferences" in message_key or "profile" in message_key
        ):
            return {"reset": True}

        enjoyment_phrases = (
            "i liked",
            "i like",
            "i love",
            "i enjoyed",
            "i enjoy",
            "enjoy watching",
            "enjoyed watching",
        )
        watched_phrases = (
            "i watched",
            "i enjoyed",
            "i liked",
            "already watched",
            "i have watched",
            "i've watched",
            "i have seen",
            "i've seen",
            "i saw",
            "enjoy watching",
            "enjoyed watching",
        )

        if any(phrase in message_key for phrase in enjoyment_phrases):
            if matched_titles:
                patch["liked_titles"] = matched_titles
            if matched_genres:
                patch["preferred_genres"] = matched_genres

        if any(phrase in message_key for phrase in ("i prefer", "prefer ", "favorite genre", "favourite genre")):
            if matched_genres:
                patch["preferred_genres"] = matched_genres

        if any(phrase in message_key for phrase in watched_phrases):
            if matched_titles:
                patch["seen_titles"] = matched_titles

        if any(
            phrase in message_key
            for phrase in ("don't like", "do not like", "dislike", "hate", "avoid", "exclude", "not interested in")
        ):
            if matched_titles:
                patch["disliked_titles"] = matched_titles
                patch["excluded_titles"] = matched_titles
            if excluded_genres:
                patch["excluded_genres"] = excluded_genres

        return patch

    def _tool_recommend(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = arguments.get("session_id")
        top_k = arguments.get("top_k", arguments.get("limit", 5))
        diagnostics: dict[str, Any] = {}
        session_profile = (
            {}
            if arguments.get("ignore_session_preferences")
            else self.get_session_profile(str(session_id) if session_id else None)
        )
        results = self.recommender.recommend(
            reference_titles=[str(value) for value in arguments.get("reference_titles") or []],
            liked_ids=[int(value) for value in arguments.get("liked_ids") or []],
            liked_titles=[str(value) for value in arguments.get("liked_titles") or []],
            excluded_ids=[int(value) for value in arguments.get("excluded_ids") or []],
            excluded_titles=[str(value) for value in arguments.get("excluded_titles") or []],
            seen_titles=[str(value) for value in arguments.get("seen_titles") or []],
            genres=[str(value) for value in arguments.get("genres") or []],
            include_genres=[str(value) for value in arguments.get("include_genres") or []],
            exclude_genres=[str(value) for value in arguments.get("exclude_genres") or []],
            media_type=arguments.get("media_type") or None,
            formats=[str(value) for value in arguments.get("formats") or []],
            min_score=float(arguments["min_score"]) if arguments.get("min_score") is not None else None,
            min_year=int(arguments["min_year"]) if arguments.get("min_year") is not None else None,
            max_year=int(arguments["max_year"]) if arguments.get("max_year") is not None else None,
            max_episodes=int(arguments["max_episodes"]) if arguments.get("max_episodes") is not None else None,
            query=arguments.get("query") or None,
            free_text_preferences=str(arguments.get("free_text_preferences") or ""),
            preferred_studios=[str(value) for value in arguments.get("preferred_studios") or []],
            required_studios=[str(value) for value in arguments.get("required_studios") or []],
            preferred_staff=[str(value) for value in arguments.get("preferred_staff") or []],
            required_staff=[str(value) for value in arguments.get("required_staff") or []],
            preferred_characters=[str(value) for value in arguments.get("preferred_characters") or []],
            required_characters=[str(value) for value in arguments.get("required_characters") or []],
            required_voice_actors=[str(value) for value in arguments.get("required_voice_actors") or []],
            required_voice_actor_ids=[int(value) for value in arguments.get("required_voice_actor_ids") or []],
            preferred_voice_actors=[str(value) for value in arguments.get("preferred_voice_actors") or []],
            required_entity_constraints=[
                dict(value) for value in arguments.get("required_entity_constraints") or [] if isinstance(value, dict)
            ],
            novelty_preference=str(arguments.get("novelty_preference") or "neutral"),
            exclude_related_series=bool(arguments.get("exclude_related_series", True)),
            one_per_series=bool(arguments.get("one_per_series")),
            session_profile=session_profile,
            diversity_strength=float(arguments.get("diversity_strength") or 0.12),
            limit=max(1, min(int(top_k or 5), 50)),
            diagnostics=diagnostics,
        )
        return {"results": results, "diagnostics": diagnostics}

    def _tool_details(self, arguments: dict[str, Any]) -> dict[str, Any]:
        anime_id = int(arguments.get("anime_id"))
        return {"result": self.recommender.details(anime_id)}

    def _tool_update_session(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = arguments.get("session_id")
        patch = {
            key: arguments.get(key)
            for key in (
                "liked_titles",
                "disliked_titles",
                "seen_titles",
                "excluded_titles",
                "preferred_genres",
                "excluded_genres",
                "preferred_studios",
                "preferred_staff",
                "preferred_characters",
                "preferred_voice_actors",
                "temporary_ratings",
                "previous_reference_titles",
                "last_recommendations",
                "last_recommendation_intent",
                "reset",
            )
            if key in arguments
        }
        return {"profile": self.update_session_preferences(str(session_id) if session_id else None, patch)}


def parse_tool_call(content: str) -> dict[str, Any] | None:
    content = content.strip()
    candidates = [content]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    inline = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if inline:
        candidates.append(inline.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("tool"):
            return parsed

    return None


def compact_result(result: Any) -> Any:
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        return {
            "diagnostics": result.get("diagnostics", {}),
            "total_results": len(result["results"]),
            "result_titles": [
                str(item.get("title") or item.get("matched_name"))
                for item in result["results"][:50]
                if item.get("title") or item.get("matched_name")
            ],
            "results": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "score": item.get("score"),
                    "start_year": item.get("start_year"),
                    "type": item.get("type"),
                    "episodes": item.get("episodes"),
                    "genres": item.get("genres", []),
                    "studios": item.get("studios", []),
                    "producers": item.get("producers", []),
                    "hybrid_scores": item.get("hybrid_scores", {}),
                    "explanation_data": item.get("explanation_data", {}),
                    "reasons": item.get("reasons", []),
                    "ranking": item.get("ranking", {}),
                    "matched_voice_actors": item.get("matched_voice_actors", []),
                    "voice_actor_roles": item.get("voice_actor_roles", []),
                    "entity_relationships": item.get("entity_relationships", []),
                    "matched_entities": item.get("matched_entities", []),
                    "matched_producers": item.get("matched_producers", []),
                    "matched_directors": item.get("matched_directors", []),
                    "matched_original_creators": item.get("matched_original_creators", []),
                    "matched_themes": item.get("matched_themes", []),
                    "matched_demographics": item.get("matched_demographics", []),
                    "matched_required_studios": item.get("matched_required_studios", []),
                    "matched_required_staff": item.get("matched_required_staff", []),
                    "matched_required_characters": item.get("matched_required_characters", []),
                    "recommendation_mode": item.get("recommendation_mode"),
                    "active_channels": item.get("active_channels", []),
                    "inactive_channel_reasons": item.get("inactive_channel_reasons", {}),
                    "configured_weights": item.get("configured_weights", {}),
                    "effective_weights": item.get("effective_weights", {}),
                    "weighted_contributions": item.get("weighted_contributions", {}),
                    "score_breakdown": item.get("score_breakdown", {}),
                    "pre_diversity_score": item.get("pre_diversity_score"),
                    "diversity_adjustment": item.get("diversity_adjustment"),
                    "final_score": item.get("final_score"),
                }
                for item in result["results"][:10]
            ],
        }
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        item = result["result"]
        return {
            "result": {
                "id": item.get("id"),
                "title": item.get("title"),
                "score": item.get("score"),
                "start_year": item.get("start_year"),
                "type": item.get("type"),
                "episodes": item.get("episodes"),
                "genres": item.get("genres", []),
                "studios": item.get("studios", []),
                "synopsis": item.get("synopsis", ""),
                "characters": item.get("characters", [])[:5],
                "staff": item.get("staff", [])[:5],
                "voice_actors": item.get("voice_actors", [])[:5],
            }
        }
    return result


def catalog_response_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "results": [
            {
                "title": item.get("title"),
                "score": item.get("score"),
                "type": item.get("type"),
                "start_year": item.get("start_year"),
                "episodes": item.get("episodes"),
                "genres": item.get("genres", []),
                "studios": item.get("studios", []),
                "producers": item.get("producers", []),
                "synopsis": item.get("synopsis", ""),
                "reasons": item.get("reasons", []),
                "matched_voice_actors": item.get("matched_voice_actors", []),
                "voice_actor_roles": item.get("voice_actor_roles", []),
                "entity_relationships": item.get("entity_relationships", []),
                "matched_entities": item.get("matched_entities", []),
                "matched_producers": item.get("matched_producers", []),
                "matched_directors": item.get("matched_directors", []),
                "matched_original_creators": item.get("matched_original_creators", []),
                "matched_themes": item.get("matched_themes", []),
                "matched_demographics": item.get("matched_demographics", []),
                "matched_required_studios": item.get("matched_required_studios", []),
                "matched_required_staff": item.get("matched_required_staff", []),
                "matched_required_characters": item.get("matched_required_characters", []),
                "explanation_data": item.get("explanation_data", {}),
            }
            for item in results
        ]
    }


def catalog_summary_prompt(limit: int = 5) -> str:
    return f"""You are Anime Compass, a friendly and proactive anime guide.

Write a concise recommendation answer using only the provided catalog JSON.
Do not invent titles, scores, studios, genres, or people.
Include exactly {limit} numbered recommendations if the catalog JSON contains that many results.
Each recommendation must be one numbered line with the anime title and one concrete,
viewer-facing reason. Explain the connection through story themes, premise, genres,
studio, creator, staff, format, or tone that is explicitly present in the JSON.
When matched_voice_actors or voice_actor_roles are present, every recommendation line
must name the verified voice actor and role before mentioning any other connection.
Never expose implementation terms such as embeddings, TF-IDF, LSA, dense similarity,
metadata similarity, hybrid scores, channels, vectors, or session profiles.
Do not write franchise analysis or detail pages.
If the catalog results are weak, say so briefly and still use only the provided titles.
Open with one natural sentence that reflects the user's request. End with one brief,
specific next step, such as offering a narrower mood, format, or era. Avoid generic filler.
"""


def introduction_prompt() -> str:
    return """You are Anime Compass, a friendly and knowledgeable anime guide.

Answer the user's question using only the trusted catalog details JSON. Never add a
story event, character, voice actor, studio, score, date, or episode count that is not
present in the JSON.

Use this compact response structure:
1. Start with a one-sentence, spoiler-light hook naming the anime.
2. Explain the premise in two or three clear sentences based on the synopsis.
3. Describe the tone, themes, and likely audience using only the supplied genres and synopsis.
4. Give practical details naturally: format, year, episode count, score, and studio when present.
5. Mention at most three notable characters, staff, or voice actors, and only when present.
6. End with one useful next step related to the user's question.

Sound warm and conversational, not promotional. Keep the answer under 260 words unless
the user explicitly asks for a detailed explanation. Avoid major spoilers and do not use
empty phrases such as "great choice" or "excellent question."
"""


def format_recommendation_answer(results: list[dict[str, Any]], limit: int | None = None) -> str:
    if not results:
        return "I could not find catalog matches for that request. Try adding a genre, title, or format."

    verified_actor_results = bool(results) and all(item.get("matched_voice_actors") for item in results[:limit])
    lines = [
        "Here are the catalog titles with verified voice-actor credits:"
        if verified_actor_results
        else "Here is a focused set from the catalog:"
    ]
    for item in results[:limit]:
        score = f"{item['score']:.2f}" if item.get("score") else "unrated"
        reasons = item.get("reasons") or []
        reason = reasons[0] if reasons else "Its story and catalog details align with the titles you referenced."
        lines.append(f"- {item['title']} ({item.get('type') or 'Unknown'}, score {score}): {reason}")
    lines.append("Tell me which one catches your eye, and I can introduce it without spoilers.")
    return "\n".join(lines)


def format_catalog_ranking_answer(results: list[dict[str, Any]], diagnostics: dict[str, Any]) -> str:
    if not results:
        return "No catalog titles matched all of those ranking filters."
    field = str(diagnostics.get("sort_by") or "score").replace("_", " ")
    order = "highest first" if diagnostics.get("sort_order") == "desc" else "lowest first"
    lines = [f"Here are the matching catalog titles ranked by {field}, {order}:"]
    for item in results:
        ranking = item.get("ranking", {})
        value = ranking.get("value")
        details = [str(item.get("type"))] if item.get("type") else []
        if item.get("score") is not None:
            details.append(f"score {float(item['score']):.2f}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"{ranking.get('position', len(lines))}. {item['title']}{suffix}: {field} = {value}")
    return "\n".join(lines)


def format_anime_introduction(item: dict[str, Any]) -> str:
    title = item.get("title") or "This anime"
    synopsis = spoiler_safe_synopsis(item.get("synopsis"))
    genres = ", ".join((item.get("genres") or [])[:4])
    details = [value for value in (item.get("type"), item.get("start_year")) if value]
    if item.get("episodes"):
        details.append(f"{item['episodes']} episodes")
    if item.get("score"):
        details.append(f"catalog score {item['score']:.2f}")

    lines = [
        f"Here is the spoiler-light introduction to {title}.",
        "",
        synopsis,
    ]
    if genres:
        lines.extend(["", f"Its catalog genres are {genres}, which gives you a useful picture of its overall tone."])
    if details:
        lines.extend(["", "Quick details: " + " / ".join(str(value) for value in details) + "."])

    studios = item.get("studios") or []
    if studios:
        lines.append("Studio: " + ", ".join(studios[:2]) + ".")

    characters = [person.get("name") for person in item.get("characters", []) if person.get("name")][:3]
    if characters:
        lines.append("Notable characters in the catalog: " + ", ".join(characters) + ".")

    lines.extend(["", "I can also break down its characters, cast, themes, or find similar anime."])
    return "\n".join(lines)


def spoiler_safe_synopsis(value: Any, max_characters: int = 420) -> str:
    fallback = "The catalog does not include a synopsis for this title."
    text = re.sub(r"\[(?:Written by|Source:)[^\]]*\]", "", str(value or ""), flags=re.IGNORECASE)
    text = " ".join(text.split()).strip()
    if not text:
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*selected, sentence]).strip()
        if selected and len(candidate) > max_characters:
            break
        selected.append(sentence)
        if len(selected) >= 2 or len(candidate) >= max_characters:
            break
    summary = " ".join(selected).strip()
    if len(summary) > max_characters:
        summary = summary[: max_characters - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return summary or fallback
