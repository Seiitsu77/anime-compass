"""Choose between the fast personalized path and the constraint-rich hybrid.

Neither path is "better". They answer different questions:

* The **fast path** answers "what should this user watch next?" It is the
  default because standalone ALS matched the full hybrid's ranking quality
  (NDCG@10 0.2624 vs 0.2629, interval including zero) at 1/125th the latency.
* The **constraint-rich path** answers "what matches this specific set of
  catalog constraints?" It performs entity resolution, exact relationship
  joins, explicit sorting, and evidence-bearing explanations. The personalized
  benchmark never scored any of that, so the benchmark cannot argue against it.

Routing is on request *semantics*, never at random and never on load. A request
that names a studio, a voice actor, a year window, or asks for an explicit
ordering needs deterministic constraint handling, so it takes the rich path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RecommendationPath(Enum):
    """Which ranking pipeline serves this request."""

    FAST = "fast"
    CONSTRAINT_RICH = "constraint_rich"


# Fields whose presence means the request pins results to verified catalog
# relationships. These cannot be served approximately.
ENTITY_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "required_studios",
    "required_staff",
    "required_voice_actors",
    "required_characters",
    "preferred_studios",
    "preferred_staff",
    "preferred_voice_actors",
    "preferred_characters",
)

# Fields describing deterministic metadata filtering or explicit ordering.
METADATA_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "include_genres",
    "exclude_genres",
    "formats",
    "min_score",
    "min_year",
    "max_year",
    "max_episodes",
)

# Intents that are inherently constraint- or explanation-shaped.
CONSTRAINT_RICH_INTENTS = frozenset({"rank_catalog", "search", "details"})


@dataclass(frozen=True)
class PathDecision:
    """The chosen pipeline and the reason, for the trace."""

    path: RecommendationPath
    reason: str
    signals: tuple[str, ...]
    policy_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommendation_path": self.path.value,
            "path_reason": self.reason,
            "path_signals": list(self.signals),
            "path_policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class PathPolicy:
    """Configurable rules for path selection."""

    # Reference titles ("more like X") are similarity requests the content
    # channels handle explicitly, so they take the rich path.
    reference_titles_are_constraints: bool = True
    # Free-text preference prose is interpreted by the content channels; the
    # fast path has no text signal at all.
    free_text_is_constraint: bool = True
    version: str = "pathpolicy-v1"


def _has_value(intent: Mapping[str, Any] | Any, field: str) -> bool:
    value = intent.get(field) if isinstance(intent, Mapping) else getattr(intent, field, None)
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, frozenset, str)):
        return len(value) > 0
    return True


def choose_recommendation_path(
    intent: Mapping[str, Any] | Any,
    *,
    policy: PathPolicy | None = None,
) -> PathDecision:
    """Select the pipeline for one request from its structured intent."""
    policy = policy or PathPolicy()
    signals: list[str] = []

    intent_name = intent.get("intent") if isinstance(intent, Mapping) else getattr(intent, "intent", None)
    if intent_name in CONSTRAINT_RICH_INTENTS:
        signals.append(f"intent:{intent_name}")

    for field in ENTITY_CONSTRAINT_FIELDS:
        if _has_value(intent, field):
            signals.append(f"entity:{field}")
    for field in METADATA_CONSTRAINT_FIELDS:
        if _has_value(intent, field):
            signals.append(f"metadata:{field}")
    if policy.reference_titles_are_constraints and _has_value(intent, "reference_titles"):
        signals.append("similarity:reference_titles")
    if policy.free_text_is_constraint and _has_value(intent, "free_text_preferences"):
        signals.append("text:free_text_preferences")
    if _has_value(intent, "entity_mentions"):
        signals.append("entity:entity_mentions")

    if signals:
        return PathDecision(
            RecommendationPath.CONSTRAINT_RICH,
            "request carries catalog constraints the fast path cannot satisfy exactly",
            tuple(signals),
            policy.version,
        )
    return PathDecision(
        RecommendationPath.FAST,
        "unconstrained personalized request",
        (),
        policy.version,
    )


def required_entity_fields(intent: Mapping[str, Any] | Any) -> tuple[str, ...]:
    """Entity fields that must be satisfied exactly, never relaxed."""
    present: list[str] = []
    for field in ENTITY_CONSTRAINT_FIELDS:
        if field.startswith("required_") and _has_value(intent, field):
            present.append(field)
    return tuple(present)


def describe_paths() -> Sequence[Mapping[str, str]]:
    """Human-readable routing summary, for docs and debug payloads."""
    return (
        {
            "path": RecommendationPath.FAST.value,
            "serves": "unconstrained personalized recommendations",
            "ranking": "ALS (or sparse fallback) + optional item-item tail supplement",
            "typical_latency": "single-digit to low tens of milliseconds",
        },
        {
            "path": RecommendationPath.CONSTRAINT_RICH.value,
            "serves": "entity joins, metadata filters, explicit ordering, explanations",
            "ranking": "multi-channel hybrid with hand-set weights",
            "typical_latency": "hundreds of milliseconds to ~1 second",
        },
    )
