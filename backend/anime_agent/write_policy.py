"""Deterministic write eligibility for session preferences.

A language model proposes an intent and a preference update. Neither is enough
to change what a session remembers: a model that mislabels "Introduce Cowboy
Bebop without spoilers" as feedback must not be able to exclude Cowboy Bebop.
So a preference field may be written only when the user's own message states
that kind of preference, and that is decided here, from the text alone, before
any state-changing tool runs.

The boundary is deliberately asymmetric. A phrase this module does not
recognise costs one clarifying reply; a phrase it wrongly recognises changes
every later recommendation silently. Patterns are therefore word-bounded and
first-person where the verb is ambiguous, and anything unlisted is denied.

This module holds patterns and one small value object and imports nothing from
the rest of the package, so the orchestrated path and the legacy deterministic
path enforce the same rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

LIKED = "liked"
WATCHED = "watched"
DISLIKED = "disliked"
EXCLUDED = "excluded"
RESET = "reset"

# What the backend records about its own results so "the second one" resolves
# on the next turn. This is conversation context, not taste: it is written by
# the retrieval tools themselves and needs no statement from the user.
CONTEXT_FIELDS = frozenset({"last_recommendation_intent", "last_recommendations", "previous_reference_titles"})

# Which stated preference authorizes which stored field. A field that is absent
# here can never be written from a conversation.
FIELD_SIGNALS: dict[str, frozenset[str]] = {
    "liked_titles": frozenset({LIKED}),
    "preferred_genres": frozenset({LIKED}),
    "preferred_studios": frozenset({LIKED}),
    "preferred_staff": frozenset({LIKED}),
    "preferred_characters": frozenset({LIKED}),
    "preferred_voice_actors": frozenset({LIKED}),
    "temporary_ratings": frozenset({LIKED, DISLIKED}),
    # Saying a title was enjoyed says it was watched.
    "seen_titles": frozenset({WATCHED, LIKED}),
    "disliked_titles": frozenset({DISLIKED}),
    "excluded_titles": frozenset({DISLIKED, EXCLUDED}),
    "excluded_genres": frozenset({DISLIKED, EXCLUDED}),
    "reset": frozenset({RESET}),
}

_SUBJECT = r"(?<![a-z0-9'])(?:i|we)"
_ADVERBS = (
    r"(?:\s+(?:really|truly|absolutely|totally|definitely|genuinely|also|still|just|always|"
    r"actually|kinda|quite|both|all|do|did))*"
)
_WATCH_ADVERBS = r"(?:\s+(?:already|just|recently|also|finally|only|actually|previously|both|all))*"
_NEGATION = r"(?:don't|dont|do\s+not|didn't|didnt|did\s+not|never)"
# "Would I like Monster?" asks; it does not tell.
_NOT_A_QUESTION = r"(?<!would )(?<!will )(?<!might )(?<!could )(?<!should )(?<!if )(?<!whether )"
_REQUEST_VERB = (
    r"(?:recommend(?:ing)?|suggest(?:ing)?|show(?:ing)?|includ(?:e|ing)|giv(?:e|ing)|"
    r"list(?:ing)?|mention(?:ing)?|bring(?:ing)?\s+up)"
)

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    LIKED: tuple(
        re.compile(pattern)
        for pattern in (
            rf"{_NOT_A_QUESTION}{_SUBJECT}(?:'ve|\s+have)?{_ADVERBS}\s+"
            r"(?:like|liked|love|loved|enjoy|enjoyed|adore|adored|prefer|preferred)\b"
            r"(?!\s+to\s+(?:know|see|hear|ask|learn|find|get|have)\b)",
            r"(?<!not )\b(?:my|our)\s+(?:all[- ]time\s+|absolute\s+|personal\s+|current\s+)?favou?rites?\b",
            rf"{_SUBJECT}(?:'m|\s+am)\s+(?:a\s+)?(?:big\s+|huge\s+|massive\s+|longtime\s+|real\s+)?(?:fan\s+of|into)\b",
            r"\b(?:add|put|save|mark)\b.{0,80}?\b(?:to|as|in|on)\s+(?:my\s+)?(?:likes?|liked|favou?rites?)\b",
        )
    ),
    WATCHED: tuple(
        re.compile(pattern)
        for pattern in (
            rf"{_SUBJECT}(?:'ve|\s+have|\s+had|'d)?{_WATCH_ADVERBS}\s+"
            r"(?:watched|seen|saw|finished|completed|rewatched|re-watched|binged)\b",
            r"\balready\s+(?:watched|seen|saw|finished|completed)\b",
            r"\b(?:mark|add|put|log|count)\b.{0,80}?\b(?:as|to|in|on)\s+(?:my\s+)?(?:watched|seen|completed|finished)\b",
            r"\b(?:finished|done|completed)\s+watching\b",
        )
    ),
    DISLIKED: tuple(
        re.compile(pattern)
        for pattern in (
            rf"\b{_NEGATION}(?:\s+(?:really|even|much|particularly))?\s+(?:like|liked|enjoy|enjoyed|love|loved|care\s+for)\b",
            r"\b(?:dislike|disliked|hate|hated|loathe|loathed|despise|despised|detest|detested)\b",
            rf"{_SUBJECT}\s+(?:can't|cannot|couldn't|could\s+not)\s+stand\b",
            r"\b(?:was|is|were|are)(?:n't|\s+not)\s+(?:really\s+|quite\s+)?"
            r"(?:for\s+me|my\s+(?:thing|kind|type|taste|cup\s+of\s+tea))\b",
            r"\bnot\s+(?:really\s+)?(?:for\s+me|my\s+(?:thing|kind|type|taste|cup\s+of\s+tea))\b",
            r"\bnot\s+(?:really\s+|that\s+)?interested\b",
            r"\bbad\s+recommendations?\b",
        )
    ),
    EXCLUDED: tuple(
        re.compile(pattern)
        for pattern in (
            rf"\b(?:don't|dont|do\s+not|never|stop|quit|no\s+longer)\s+(?:ever\s+)?{_REQUEST_VERB}\b",
            r"\b(?:don't|dont|do\s+not)\s+want\b",
            r"\b(?:avoid|exclude|excluding|skip|block|blacklist)\b",
            # "no more than 12 episodes" is a length limit, not a rejection.
            r"\bno\s+more\b(?!\s+than\b)",
            r"\b(?:none|not\s+any)\s+of\s+(?:these|those|them)\b",
            r"\bnot\s+(?:these|those)\b",
            r"\bsame\s+recommendations\b",
        )
    ),
    RESET: (
        re.compile(r"\breset\b.*\b(?:session|preferences|profile)\b|\b(?:session|preferences|profile)\b.*\breset\b"),
    ),
}

# A request verb that is itself negated ("don't recommend X") asks for nothing.
_NEGATED_REQUEST = re.compile(rf"\b(?:don't|dont|do\s+not|never|stop|quit|no\s+longer)\s+(?:ever\s+)?{_REQUEST_VERB}\b")


def _normalize(message: str) -> str:
    text = message.casefold().replace("’", "'").replace("‘", "'").replace("`", "'")
    return " ".join(text.split())


def preference_signals(message: str) -> frozenset[str]:
    """The kinds of preference the user's own words state. Empty for most messages."""
    text = _normalize(message)
    return frozenset(
        signal for signal, patterns in _PATTERNS.items() if any(pattern.search(text) for pattern in patterns)
    )


def states_negative_preference(message: str) -> bool:
    """Does the message reject or dislike something, in so many words?"""
    return bool(preference_signals(message) & {DISLIKED, EXCLUDED})


def without_negated_requests(message: str) -> str:
    """The message with negated request verbs removed, for "is anything being asked for?" checks."""
    return _NEGATED_REQUEST.sub(" ", _normalize(message))


@dataclass(frozen=True)
class PreferenceWriteAuthorization:
    """What one user message permits the session store to change.

    Built once per request from the raw message, never from the model's
    output, and consulted before every session write.
    """

    signals: frozenset[str] = frozenset()

    def permits(self, name: str) -> bool:
        return bool(FIELD_SIGNALS.get(name, frozenset()) & self.signals)

    def split(self, patch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition a proposed session patch into (authorized, denied).

        Context fields always pass. An empty value changes nothing and is
        dropped from both halves, so a caller can treat an empty authorized
        half as "there is nothing to write".
        """
        authorized: dict[str, Any] = {}
        denied: dict[str, Any] = {}
        for name, value in patch.items():
            if name in CONTEXT_FIELDS:
                authorized[name] = value
            elif not value:
                continue
            elif self.permits(name):
                authorized[name] = value
            else:
                denied[name] = value
        return authorized, denied

    def describe(self) -> dict[str, Any]:
        return {
            "signals": sorted(self.signals),
            "writable_fields": sorted(name for name in FIELD_SIGNALS if self.permits(name)),
        }


@dataclass
class PreferenceWriteGate:
    """One request's authorization, and a record of what it let through.

    Every session write passes through `admit`. A caller that has no gate gets
    the default one, which authorizes nothing, so a new code path cannot write
    a preference by forgetting to ask.
    """

    authorization: PreferenceWriteAuthorization = field(default_factory=PreferenceWriteAuthorization)
    written: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    # Set when a preference update was refused and the request was answered
    # some read-only way instead.
    rerouted_from: str | None = None

    def admit(self, patch: dict[str, Any]) -> dict[str, Any]:
        """The part of `patch` that may be written. Call it before writing, not after."""
        authorized, denied = self.authorization.split(patch)
        for name in authorized:
            if name not in CONTEXT_FIELDS and name not in self.written:
                self.written.append(name)
        for name in denied:
            if name not in self.denied:
                self.denied.append(name)
        return authorized

    def describe(self) -> dict[str, Any]:
        return {
            **self.authorization.describe(),
            "written_fields": sorted(self.written),
            "denied_fields": sorted(self.denied),
            "rerouted_from": self.rerouted_from,
        }
