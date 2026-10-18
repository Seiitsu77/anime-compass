"""Franchise ("title family") constraints, verified against catalog titles.

Two English sentences can both name Gundam and mean different things:

    "I like Gundam. Recommend similar anime."   -- Gundam is a taste signal
    "Are there any Gundam movies after 2020?"   -- Gundam is the subject

Only the first reading was ever implemented. A franchise name reached the
recommender as `free_text_preferences`, which is one more similarity token, so
the second question was answered by whatever ranked well among 2021+ movies:
Gintama, Evangelion, Free!. Padding a Top-N with plausible neighbours is right
for a taste signal and wrong for a subject, and the language model knew it --
it wrote "While not a Gundam title..." beside an item the backend had handed it
as a verified candidate.

This module supplies the missing distinction. It decides, deterministically and
from the raw message, when a name is a *required* franchise constraint, and it
re-checks every candidate against that constraint before any of them reach the
model.

Evidence
--------
The MAL-derived catalog has no franchise, series-family, or related-works
column, so franchise membership cannot be looked up. The only evidence the data
carries is the title itself, and that is all this module uses: a candidate
belongs to the "gundam" family when its own normalised title, or one of its
catalogued aliases, contains that token sequence.

What deliberately does *not* count as franchise evidence:

* **Studio.** Sunrise animates Gundam and much else; Sunrise is not Gundam. A
  candidate phrase that names a studio or producer is rejected outright rather
  than promoted, which keeps the inference symmetric -- a studio word never
  becomes a franchise, and a franchise word never widens to a studio.
* **Genre, theme, format.** Mecha, sci-fi, military and "movie" describe what a
  title is like. They are similarity features, and every one of them is in the
  non-family vocabulary below.

Because the evidence is weaker than a real franchise column, the trigger is
narrow on purpose. When it does not fire, a request behaves exactly as it did
before; when it fires, results are restricted to titles that literally carry the
name, which is a claim the catalog can actually support.

Membership details (`AnimeRecommender.title_family_members`): the name may sit
anywhere in a title or alias, not only at the start; a name that is also an
ordinary word ("fate") must open the title or use the franchise's slash form;
and an English name reaches the entries that only carry the romaji series name
through an alias bridge ("Demon Slayer" -> "Kimetsu no Yaiba").

Where a constraint comes from
-----------------------------
Two sources, combined by `resolve_required_title_families`:

* **The model.** Gemini reads the request and names the franchises results must
  belong to (`required_franchises`). It understands phrasings no pattern lists:
  "anything Gundam from 2022?", "only Gundam please".
* **This module's own reading of the message**, for the attributive pattern
  ("Gundam movies", "the Naruto franchise") and explicit restriction ("only
  Gundam"). It holds when the model is unavailable or misses the name.

Either source may add a family. Neither can add one the catalog does not
support, and the model's proposals are checked against the conversation: a name
the user never typed is refused, and a name the message asks to resemble
("similar to Gundam", "movies like Gundam") is a reference, not a constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakKeyDictionary

from backend.anime_agent.recommender import family_key, normalize_label

# Nouns that turn a preceding name into "things of that name": the attributive
# reading ("Gundam movies") rather than the comparative one ("like Gundam").
CATALOG_ITEM_NOUNS = frozenset(
    {
        "anime",
        "entries",
        "entry",
        "film",
        "films",
        "movie",
        "movies",
        "ona",
        "onas",
        "ova",
        "ovas",
        "season",
        "seasons",
        "series",
        "show",
        "shows",
        "special",
        "specials",
        "stuff",
        "things",
        "title",
        "titles",
        "tv",
        "works",
    }
)

# "Not Gundam", "except Naruto", "without Gundam": a franchise put out of bounds.
NEGATION_WORDS = frozenset({"not", "no", "except", "excluding", "exclude", "without", "avoid", "besides"})
NEGATION_FILLERS = frozenset({"the", "any", "anything", "for", "from", "of", "more"})
# Two-word cues: "anything but Gundam", "other than Naruto".
NEGATION_PAIRS = frozenset({("anything", "but"), ("everything", "but"), ("other", "than")})

# "Only Gundam", "just Naruto": the user restricting the answer outright. This
# outranks a similarity phrase elsewhere in the same message.
RESTRICTION_WORDS = frozenset({"only", "just", "exclusively", "strictly", "purely"})

# "X franchise" / "X universe" states the relationship outright.
FRANCHISE_NOUNS = frozenset({"franchise", "franchises", "saga", "universe"})

# Phrasings that ask for neighbours of a title rather than members of it. A
# bare "like" is not one of them: "I like watching Gundam movies" is a statement
# of taste attached to an attributive request, not a similarity ask.
SIMILARITY_REQUEST_RE = re.compile(
    r"\b(?:similar"
    r"|remind(?:s)?\s+me\s+of"
    r"|comparable"
    r"|in\s+the\s+(?:same\s+)?(?:vein|style|spirit)"
    r"|same\s+(?:vibe|feel|energy)"
    r"|(?:something|anything|stuff|more|other|others|anime|show|shows|series|title|titles|movie|movies)\s+like"
    r"|like\s+(?:this|that|these|those|it)\b"
    r")",
    re.IGNORECASE,
)

# Request vocabulary that can never identify a franchise on its own. Catalog
# genres and formats are added per call, because they come from the catalog.
NON_FAMILY_TOKENS = frozenset(
    {
        "a",
        "after",
        "all",
        "also",
        "an",
        "and",
        "another",
        "any",
        "anything",
        "are",
        "available",
        "before",
        "best",
        "both",
        "but",
        "can",
        "catalog",
        "classic",
        "come",
        "could",
        "different",
        "do",
        "does",
        "else",
        "every",
        "favorite",
        "favourite",
        "find",
        "for",
        "from",
        "give",
        "good",
        "great",
        "has",
        "have",
        "her",
        "here",
        "his",
        "i",
        "in",
        "into",
        "is",
        "it",
        "just",
        "latest",
        "least",
        "list",
        "long",
        "longer",
        "look",
        "looking",
        "lot",
        "many",
        "me",
        "modern",
        "more",
        "most",
        "much",
        "my",
        "need",
        "new",
        "newer",
        "newest",
        "nice",
        "no",
        "not",
        "old",
        "older",
        "oldest",
        "on",
        "one",
        "only",
        "other",
        "our",
        "out",
        "please",
        "popular",
        "prefer",
        "preferably",
        "rank",
        "ranked",
        "rated",
        "rating",
        "really",
        "recent",
        "released",
        "score",
        "scored",
        "short",
        "shorter",
        "show",
        "since",
        "some",
        "something",
        "sort",
        "sorted",
        "suggest",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "top",
        "two",
        "until",
        "upcoming",
        "want",
        "was",
        "watch",
        "watched",
        "watching",
        "well",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "worst",
        "would",
        "year",
        "years",
        "you",
        "your",
    }
)

TOKEN_RE = re.compile(r"[a-z0-9]+")

# The longest name this will treat as one franchise, in tokens. Long enough for
# "kidou senshi gundam", short enough that a whole clause cannot become a name.
MAX_FAMILY_TOKENS = 4

# A family has to be a family: one title is a title, and a request for that is
# already served by title search.
MIN_FAMILY_TITLES = 2

# Above this share of the catalog a token describes anime in general rather than
# naming one franchise. Gundam is 0.76% of this catalog; "movie" is 4.75%.
MAX_FAMILY_DF_RATIO = 0.01

# The ratio above is a statement about vocabulary mass, which a fixture-sized
# catalog does not have. Below this size the ceiling is not applied, because a
# name covering three of eight rows is not evidence of a stopword.
MIN_CATALOG_FOR_DF_CEILING = 500


def _message_tokens(message: str) -> list[str]:
    return TOKEN_RE.findall(str(message or "").casefold())


# `meta()` rebuilds the genre and format vocabulary by walking the whole catalog,
# and this runs on every message. The catalog does not change under a live
# recommender, so the derived vocabulary is cached against it and released with
# it rather than recomputed per request.
_VOCABULARY_CACHE: WeakKeyDictionary[Any, frozenset[str]] = WeakKeyDictionary()


def _non_family_vocabulary(recommender: Any) -> frozenset[str]:
    """Request words plus this catalog's own genre and format vocabulary."""
    cached = _VOCABULARY_CACHE.get(recommender)
    if cached is not None:
        return cached
    meta = recommender.meta()
    vocabulary = NON_FAMILY_TOKENS | {
        token for label in [*meta.get("genres", []), *meta.get("types", [])] for token in normalize_label(label).split()
    }
    try:
        _VOCABULARY_CACHE[recommender] = vocabulary
    except TypeError:  # pragma: no cover - a recommender stand-in without weakrefs
        pass
    return vocabulary


def _verified_family(phrase_key: str, recommender: Any, vocabulary: frozenset[str]) -> bool:
    """Is `phrase_key` a name this catalog actually uses for a family of titles?"""
    if not phrase_key:
        return False
    tokens = phrase_key.split()
    if all(token in vocabulary for token in tokens):
        return False
    # The last token is the head of the name. "gundam tv" is not a franchise
    # whose name ends in a format word; the franchise is "gundam".
    if tokens[-1] in vocabulary:
        return False
    if phrase_key in recommender.studio_name_phrases():
        return False
    # "Cat anime", "ninja movies" and "dragon movies" fit the same pattern as
    # "Gundam movies". What separates them is how the catalog uses the word:
    # as the name of a family of titles, or as an ordinary word. "Fate" is an
    # ordinary word too, but its titles are written "Fate/..." -- and then its
    # membership is the strict leading form, so "Bloody Fate" stays out.
    if not (
        recommender.title_family_reads_as_name(phrase_key) or recommender.title_family_uses_slash_notation(phrase_key)
    ):
        return False
    frequency = recommender.title_family_document_frequency(phrase_key)
    if frequency < MIN_FAMILY_TITLES:
        return False
    catalog_size = len(recommender.catalog)
    if catalog_size >= MIN_CATALOG_FOR_DF_CEILING and frequency > catalog_size * MAX_FAMILY_DF_RATIO:
        return False
    return True


def extract_required_title_families(message: str, recommender: Any) -> list[str]:
    """Franchise names this message requires results to belong to.

    Returns normalised keys, as a disjunction: a candidate qualifies by matching
    any one of them. Empty means the message did not ask for a franchise, which
    is the common case and leaves every other path untouched.
    """
    text = str(message or "")
    tokens = _message_tokens(text)
    if not text.strip() or not tokens:
        return []

    vocabulary = _non_family_vocabulary(recommender)
    found: list[str] = []
    # Explicit restriction first: "only Gundam" stands even beside a similarity
    # phrase ("only Gundam, nothing merely similar").
    for position, token in enumerate(tokens):
        if token not in RESTRICTION_WORDS:
            continue
        for start in range(position + 1, min(position + 4, len(tokens))):
            for length in range(min(MAX_FAMILY_TOKENS, len(tokens) - start), 0, -1):
                phrase_key = family_key(" ".join(tokens[start : start + length]))
                if _verified_family(phrase_key, recommender, vocabulary):
                    if phrase_key not in found:
                        found.append(phrase_key)
                    break
            else:
                continue
            break
    if SIMILARITY_REQUEST_RE.search(text):
        return found

    for position, token in enumerate(tokens):
        if token not in CATALOG_ITEM_NOUNS and token not in FRANCHISE_NOUNS:
            continue
        for length in range(min(MAX_FAMILY_TOKENS, position), 0, -1):
            phrase_key = family_key(" ".join(tokens[position - length : position]))
            if _verified_family(phrase_key, recommender, vocabulary):
                if phrase_key not in found:
                    found.append(phrase_key)
                break
    return found


def extract_excluded_title_families(message: str, recommender: Any) -> list[str]:
    """Franchises this message puts out of bounds ("mecha anime, but not Gundam").

    The catalog-title evidence that decides membership for a required franchise
    decides it here too, so "not Gundam" removes every Gundam entry rather than
    the one title that happens to be called "Gundam".
    """
    tokens = _message_tokens(message)
    if not tokens:
        return []
    vocabulary = _non_family_vocabulary(recommender)
    found: list[str] = []
    for position, token in enumerate(tokens):
        pair = (tokens[position - 1], token) if position else ("", token)
        if token not in NEGATION_WORDS and pair not in NEGATION_PAIRS:
            continue
        start = position + 1
        while start < len(tokens) and tokens[start] in NEGATION_FILLERS:
            start += 1
        for length in range(min(MAX_FAMILY_TOKENS, len(tokens) - start), 0, -1):
            phrase_key = family_key(" ".join(tokens[start : start + length]))
            if _verified_family(phrase_key, recommender, vocabulary):
                if phrase_key not in found:
                    found.append(phrase_key)
                break
    return found


def resolve_excluded_title_families(
    message: str,
    recommender: Any,
    *,
    required: list[str] | tuple[str, ...] = (),
    earlier_user_messages: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Franchises excluded now or earlier in the conversation.

    An exclusion stated earlier ("mecha, but not Gundam") still holds for a
    follow-up such as "what about movies?". A franchise the current message
    asks for outright wins over an earlier exclusion of it.
    """
    found = extract_excluded_title_families(message, recommender)
    for earlier in earlier_user_messages:
        for key in extract_excluded_title_families(earlier, recommender):
            if key not in found and key not in required:
                found.append(key)
    return found


@dataclass(frozen=True)
class FamilyResolution:
    """Which franchise constraints a request carries, and why the rest do not."""

    families: tuple[str, ...] = ()
    from_message: tuple[str, ...] = ()
    from_model: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    inherited: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "required_title_families": list(self.families),
            "detected_in_message": list(self.from_message),
            "accepted_from_model": list(self.from_model),
            "inherited_from_earlier_turn": list(self.inherited),
            "refused": [{"name": name, "reason": reason} for name, reason in self.refused],
        }


def _grounded_key(key: str, texts: list[str]) -> str:
    """The longest tail of `key` the user actually typed, or "" if none.

    The model may expand "Gundam" to "Mobile Suit Gundam"; the words the user
    wrote are what the constraint may rest on.
    """
    tokens = key.split()
    haystacks = [f" {family_key(text)} " for text in texts if text]
    for start in range(len(tokens)):
        candidate = " ".join(tokens[start:])
        if any(f" {candidate} " in haystack for haystack in haystacks):
            return candidate
    return ""


def resolve_required_title_families(
    message: str,
    recommender: Any,
    *,
    proposed: list[str] | tuple[str, ...] = (),
    earlier_user_messages: list[str] | tuple[str, ...] = (),
    follow_up: bool = False,
) -> FamilyResolution:
    """Combine the model's proposed franchises with the message's own reading.

    A proposal is accepted when the user typed the name (now, or earlier in the
    conversation for a follow-up such as "what about TV series?"), the message
    does not ask to resemble it, it is not a studio or a genre, and the catalog
    has entries for it. A typed, explicitly requested franchise the catalog has
    no entries for is still kept: it then matches nothing, and the honest answer
    is that the catalog holds nothing for it -- not titles from elsewhere.

    A `follow_up` that names no franchise of its own ("what about TV series
    instead?") keeps the one from the most recent earlier message that named
    one. This is what the model is asked to do too; doing it here as well means
    the answer does not change when the model is unavailable.
    """
    excluded = set(extract_excluded_title_families(message, recommender))
    detected = [key for key in extract_required_title_families(message, recommender) if key not in excluded]
    families = list(detected)
    accepted: list[str] = []
    refused: list[tuple[str, str]] = []
    vocabulary = _non_family_vocabulary(recommender)
    text = str(message or "")
    tokens = _message_tokens(text)
    restricted = {
        family_key(" ".join(tokens[index + 1 : index + 1 + length]))
        for index, token in enumerate(tokens)
        if token in RESTRICTION_WORDS
        for length in range(1, MAX_FAMILY_TOKENS + 1)
    }
    similarity = bool(SIMILARITY_REQUEST_RE.search(text))
    for name in proposed:
        raw_key = family_key(name)
        if not raw_key:
            continue
        key = _grounded_key(raw_key, [text]) or _grounded_key(raw_key, list(earlier_user_messages))
        if not key:
            refused.append((str(name), "not named by the user"))
            continue
        if key in families:
            continue
        if key in excluded:
            refused.append((str(name), "the message excludes it"))
            continue
        if similarity and key not in restricted:
            refused.append((str(name), "named as a similarity reference"))
            continue
        key_tokens = key.split()
        if all(token in vocabulary for token in key_tokens) or key_tokens[-1] in vocabulary:
            refused.append((str(name), "a genre, format or request word, not a franchise"))
            continue
        if key in recommender.studio_name_phrases():
            refused.append((str(name), "a studio; handled as a studio requirement"))
            continue
        families.append(key)
        accepted.append(key)
    inherited: list[str] = []
    if follow_up and not families and not similarity:
        for earlier in reversed(list(earlier_user_messages)):
            earlier_families = extract_required_title_families(earlier, recommender)
            if earlier_families:
                inherited = [key for key in earlier_families if key not in excluded]
                break
        families.extend(inherited)
    return FamilyResolution(
        families=tuple(families),
        from_message=tuple(detected),
        from_model=tuple(accepted),
        refused=tuple(refused),
        inherited=tuple(inherited),
    )


@dataclass(frozen=True)
class EligibilityConstraints:
    """What a result must satisfy to be allowed into an answer.

    Built from the validated intent rather than from the raw message, so the
    check sees exactly the constraints the pipeline ran with -- including any a
    replanning step deliberately dropped.
    """

    title_families: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    min_year: int | None = None
    max_year: int | None = None
    min_score: float | None = None
    max_episodes: int | None = None
    required_studios: tuple[str, ...] = ()
    required_staff: tuple[str, ...] = ()
    required_characters: tuple[str, ...] = ()
    required_voice_actors: tuple[str, ...] = ()
    excluded_ids: frozenset[int] = frozenset()
    excluded_families: tuple[str, ...] = ()

    @property
    def is_constrained(self) -> bool:
        """True when any hard constraint applies, so results must be re-checked."""
        return bool(
            self.pins_subject
            or self.formats
            or self.min_year is not None
            or self.max_year is not None
            or self.min_score is not None
            or self.max_episodes is not None
            or self.excluded_ids
            or self.excluded_families
        )

    @property
    def pins_subject(self) -> bool:
        """True when the request named what results must *be*, not resemble."""
        return bool(
            self.title_families
            or self.required_studios
            or self.required_staff
            or self.required_characters
            or self.required_voice_actors
        )

    def describe(self) -> list[str]:
        parts: list[str] = []
        if self.title_families:
            parts.append("title family " + " or ".join(sorted(self.title_families)))
        if self.formats:
            parts.append("type " + " or ".join(self.formats))
        if self.min_year is not None:
            parts.append(f"start_year >= {self.min_year}")
        if self.max_year is not None:
            parts.append(f"start_year <= {self.max_year}")
        if self.min_score is not None:
            parts.append(f"score >= {self.min_score}")
        if self.max_episodes is not None:
            parts.append(f"episodes <= {self.max_episodes}")
        if self.excluded_ids:
            parts.append(f"not one of {len(self.excluded_ids)} excluded titles")
        if self.excluded_families:
            parts.append("not in title family " + " or ".join(sorted(self.excluded_families)))
        return parts


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def constraints_from_intent(intent: Any, excluded_ids: frozenset[int] = frozenset()) -> EligibilityConstraints:
    """Read the required constraint set off a validated or legacy intent."""
    families = tuple(
        key
        for key in (family_key(value) for value in _string_tuple(getattr(intent, "required_title_families", ())))
        if key
    )
    return EligibilityConstraints(
        title_families=families,
        formats=_string_tuple(getattr(intent, "formats", ())),
        min_year=getattr(intent, "min_year", None),
        max_year=getattr(intent, "max_year", None),
        min_score=getattr(intent, "min_score", None),
        max_episodes=getattr(intent, "max_episodes", None),
        required_studios=_string_tuple(getattr(intent, "required_studios", ())),
        required_staff=_string_tuple(getattr(intent, "required_staff", ())),
        required_characters=_string_tuple(getattr(intent, "required_characters", ())),
        required_voice_actors=_string_tuple(getattr(intent, "required_voice_actors", ())),
        excluded_ids=excluded_ids,
        excluded_families=tuple(
            key
            for key in (family_key(value) for value in _string_tuple(getattr(intent, "excluded_title_families", ())))
            if key
        ),
    )


def _item_staff_keys(item: dict[str, Any]) -> set[str]:
    return {
        normalize_label(person.get("name"))
        for person in item.get("staff_relationships", item.get("staff", []))
        if person.get("name")
    }


def _item_character_keys(item: dict[str, Any]) -> set[str]:
    named = {normalize_label(value) for value in item.get("character_names", []) if value}
    if named:
        return named
    return {normalize_label(person.get("name")) for person in item.get("characters", []) if person.get("name")}


def is_eligible(recommender: Any, constraints: EligibilityConstraints, row: dict[str, Any]) -> bool:
    """Does this result row still satisfy every required constraint?

    Checked against the catalog record the row's id points at, not against the
    row's own fields. A row is a claim; the catalog is the evidence.
    """
    try:
        anime_id = int(str(row.get("id")))
    except (TypeError, ValueError):
        return False
    item = recommender.by_id.get(anime_id)
    if item is None:
        return False
    if anime_id in constraints.excluded_ids:
        return False
    if any(anime_id in recommender.title_family_members(family) for family in constraints.excluded_families):
        return False

    if constraints.title_families and not any(
        anime_id in recommender.title_family_members(family) for family in constraints.title_families
    ):
        return False
    if constraints.formats:
        allowed = {value.casefold() for value in constraints.formats}
        if str(item.get("type") or "").casefold() not in allowed:
            return False

    year = item.get("start_year")
    if constraints.min_year is not None and (year is None or int(year) < int(constraints.min_year)):
        return False
    if constraints.max_year is not None and (year is None or int(year) > int(constraints.max_year)):
        return False

    score = item.get("score")
    if constraints.min_score is not None and (score is None or float(score) < float(constraints.min_score)):
        return False

    episodes = item.get("episodes")
    if constraints.max_episodes is not None and (episodes is None or int(episodes) > int(constraints.max_episodes)):
        return False

    if constraints.required_studios:
        wanted = {normalize_label(value) for value in constraints.required_studios}
        if not wanted.issubset({normalize_label(value) for value in item.get("studios", []) if value}):
            return False
    if constraints.required_staff:
        wanted = {normalize_label(value) for value in constraints.required_staff}
        if not wanted.issubset(_item_staff_keys(item)):
            return False
    if constraints.required_characters:
        wanted = {normalize_label(value) for value in constraints.required_characters}
        if not wanted.issubset(_item_character_keys(item)):
            return False
    if constraints.required_voice_actors and not row.get("matched_voice_actors"):
        # The recommender fills `matched_voice_actors` only from verified role
        # records, so its absence means this row was never verified -- which is
        # exactly what this gate exists to catch.
        return False
    return True


# Tools whose results are catalog rows the model may name in an answer.
GROUNDED_RESULT_TOOLS = frozenset({"recommend_anime", "rank_catalog", "search_anime"})


def enforce_result_eligibility(
    recommender: Any,
    constraints: EligibilityConstraints,
    response: dict[str, Any],
) -> list[str]:
    """Drop ineligible rows from a tool trace in place; return what was dropped.

    This is the last gate before candidate ids reach the language model. It is
    deliberately independent of how they were produced: ranking, hybrid scoring,
    diversity reranking, replanning and follow-up merging all run upstream of it,
    and none of them can put an ineligible title back.
    """
    if not constraints.is_constrained:
        return []

    dropped: list[str] = []
    for step in response.get("trace") or []:
        if not isinstance(step, dict) or step.get("tool") not in GROUNDED_RESULT_TOOLS:
            continue
        result = step.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            continue
        rows = result["results"]
        kept = []
        for row in rows:
            if isinstance(row, dict) and is_eligible(recommender, constraints, row):
                kept.append(row)
            else:
                dropped.append(str(row.get("title") or "") if isinstance(row, dict) else "")
        if len(kept) == len(rows):
            continue
        result["results"] = kept
        if "total_results" in result:
            result["total_results"] = len(kept)
        if isinstance(result.get("result_titles"), list):
            result["result_titles"] = [
                str(row.get("title") or row.get("matched_name"))
                for row in kept
                if row.get("title") or row.get("matched_name")
            ]
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["final_eligibility_dropped"] = len(rows) - len(kept)
            diagnostics["final_eligibility_constraints"] = constraints.describe()
    return dropped


def without_relaxed_fields(constraints: EligibilityConstraints, relaxed: frozenset[str]) -> EligibilityConstraints:
    """The same constraints minus ones a replanning step deliberately dropped."""
    updates: dict[str, Any] = {}
    for field_name in ("min_year", "max_year", "min_score", "max_episodes"):
        if field_name in relaxed:
            updates[field_name] = None
    if "formats" in relaxed:
        updates["formats"] = ()
    return replace(constraints, **updates) if updates else constraints
