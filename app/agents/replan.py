"""Bounded constraint relaxation for catalog requests that return nothing.

A strict entity-and-metadata filter chain can be over-constrained: "a 2015-only
Madhouse isekai with at most 12 episodes rated above 8.5" is a reasonable
sentence that matches no catalog row. Returning an empty list is correct but
unhelpful, and asking the language model to retry is not, because it would be
free to invent titles.

Instead the backend replans deterministically. Constraints are dropped one at a
time, in a fixed order, and the catalog tools are re-executed. Every step is
recorded so the response can state exactly what was relaxed.

Two invariants keep this safe:

* **Required entity constraints are never relaxed.** A request for titles with a
  specific voice actor, studio, staff member, or character must either be
  satisfied exactly or return nothing. Relaxing those would silently break the
  grounding contract that the rest of the agent enforces.
* **The loop is bounded.** At most ``max_steps`` relaxations are attempted, so a
  request costs a predictable amount of work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.agents.runtime_state import ReplanState, ToolResultSummary
from app.agents.schemas import AgentIntent

# Intents whose value depends on returning candidates. `details`,
# `update_preferences`, and `conversation` are excluded: an empty result there
# is a genuine answer, not an over-constrained filter.
RETRIEVAL_INTENTS = frozenset({"recommend", "rank_catalog", "search"})


@dataclass(frozen=True)
class RelaxationStep:
    """One applied relaxation and the outcome it produced."""

    field: str
    description: str
    removed_value: Any
    result_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "description": self.description,
            "removed_value": self.removed_value,
            "result_count": self.result_count,
        }


def _clear_list(intent: AgentIntent, field: str) -> tuple[AgentIntent, Any] | None:
    current = list(getattr(intent, field, []) or [])
    if not current:
        return None
    return intent.model_copy(update={field: []}), current


def _clear_scalar(intent: AgentIntent, field: str) -> tuple[AgentIntent, Any] | None:
    current = getattr(intent, field, None)
    if current is None:
        return None
    return intent.model_copy(update={field: None}), current


# Ordered least-costly-to-lose first. Numeric thresholds are the most common
# cause of an empty result and the least central to what was asked for; genre
# preferences are dropped last because they usually carry the actual request.
RELAXATION_LADDER: tuple[tuple[str, str, Callable[[AgentIntent], tuple[AgentIntent, Any] | None]], ...] = (
    ("min_score", "minimum score filter", lambda intent: _clear_scalar(intent, "min_score")),
    ("max_episodes", "maximum episode count", lambda intent: _clear_scalar(intent, "max_episodes")),
    ("min_year", "earliest year filter", lambda intent: _clear_scalar(intent, "min_year")),
    ("max_year", "latest year filter", lambda intent: _clear_scalar(intent, "max_year")),
    ("formats", "format filter", lambda intent: _clear_list(intent, "formats")),
    ("preferred_studios", "preferred studios", lambda intent: _clear_list(intent, "preferred_studios")),
    ("preferred_staff", "preferred staff", lambda intent: _clear_list(intent, "preferred_staff")),
    (
        "preferred_voice_actors",
        "preferred voice actors",
        lambda intent: _clear_list(intent, "preferred_voice_actors"),
    ),
    ("preferred_characters", "preferred characters", lambda intent: _clear_list(intent, "preferred_characters")),
    ("exclude_genres", "excluded genres", lambda intent: _clear_list(intent, "exclude_genres")),
    ("include_genres", "required genres", lambda intent: _clear_list(intent, "include_genres")),
)

# Fields that pin the answer to verified catalog relationships. The ladder must
# never contain any of these; the test suite asserts that.
PROTECTED_FIELDS = frozenset(
    {
        "required_studios",
        "required_staff",
        "required_voice_actors",
        "required_characters",
        "entity_mentions",
        "intent",
    }
)


def result_count(response: dict[str, Any]) -> int:
    """Count catalog rows a tool trace actually produced."""
    total = 0
    for step in response.get("trace", []) or []:
        result = step.get("result")
        if not isinstance(result, dict):
            continue
        rows = result.get("results")
        if isinstance(rows, list):
            total += len(rows)
    return total


def has_required_entities(intent: AgentIntent) -> bool:
    """True when the request pins results to specific catalog entities."""
    return any(
        bool(getattr(intent, field, None))
        for field in ("required_studios", "required_staff", "required_voice_actors", "required_characters")
    )


def candidate_relaxations(intent: AgentIntent) -> list[str]:
    """Fields on this intent that the ladder could still relax, in order."""
    available: list[str] = []
    for field, _description, transform in RELAXATION_LADDER:
        if transform(intent) is not None:
            available.append(field)
    return available


@dataclass(frozen=True)
class ReplanOutcome:
    """Everything a replan produced, including its explicit end state."""

    intent: AgentIntent
    response: dict[str, Any]
    steps: list[RelaxationStep]
    state: ReplanState


def replan_with_state(
    intent: AgentIntent,
    execute: Callable[[AgentIntent, frozenset[str]], dict[str, Any]],
    *,
    max_steps: int = 2,
    initial_response: dict[str, Any] | None = None,
) -> ReplanOutcome:
    """`replan_until_results`, plus the runtime state it passed through.

    The state is not a log. `ReplanState.after_relaxing` re-derives the required
    constraints from the relaxed intent and refuses to advance if they changed,
    so the "required constraints survive replanning" invariant is enforced on
    every step rather than assumed from the shape of the ladder.
    """
    final_intent, response, steps = replan_until_results(
        intent,
        execute,
        max_steps=max_steps,
        initial_response=initial_response,
    )
    state = ReplanState.initial(intent, max_replans=max_steps)
    running = intent
    for step in steps:
        outcome = next(
            (transform(running) for field, _d, transform in RELAXATION_LADDER if field == step.field),
            None,
        )
        running = outcome[0] if outcome else running
        state = state.after_relaxing(
            step.field,
            running,
            summary=ToolResultSummary(result_count=step.result_count),
        )
    return ReplanOutcome(final_intent, response, steps, state)


def replan_until_results(
    intent: AgentIntent,
    execute: Callable[[AgentIntent, frozenset[str]], dict[str, Any]],
    *,
    max_steps: int = 2,
    initial_response: dict[str, Any] | None = None,
) -> tuple[AgentIntent, dict[str, Any], list[RelaxationStep]]:
    """Relax constraints until the catalog returns rows or the budget is spent.

    ``execute`` runs the real tool chain for a candidate intent and the set of
    fields relaxed so far, and returns its response payload. It must not
    re-derive those fields from the raw user message. Pass ``initial_response``
    when the caller has already run the unrelaxed plan, so it is not executed
    twice.

    Returns the intent that produced the final response, that response, and the
    ordered list of relaxations that were applied.
    """
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")

    current_intent = intent
    relaxed_fields: set[str] = set()
    current_response = execute(current_intent, frozenset()) if initial_response is None else initial_response
    applied: list[RelaxationStep] = []

    if intent.intent not in RETRIEVAL_INTENTS or max_steps == 0:
        return current_intent, current_response, applied

    for _attempt in range(max_steps):
        if result_count(current_response) > 0:
            break
        progressed = False
        for field, description, transform in RELAXATION_LADDER:
            outcome = transform(current_intent)
            if outcome is None:
                continue
            relaxed_intent, removed = outcome
            relaxed_fields.add(field)
            relaxed_response = execute(relaxed_intent, frozenset(relaxed_fields))
            current_intent = relaxed_intent
            current_response = relaxed_response
            applied.append(
                RelaxationStep(
                    field=field,
                    description=description,
                    removed_value=removed,
                    result_count=result_count(relaxed_response),
                )
            )
            progressed = True
            break
        if not progressed:
            # Nothing left that may be relaxed; the request is genuinely empty.
            break

    return current_intent, current_response, applied


def describe_relaxations(steps: Sequence[RelaxationStep]) -> str:
    """A short, user-facing sentence naming what was loosened."""
    if not steps:
        return ""
    descriptions = [step.description for step in steps]
    if len(descriptions) == 1:
        return f"No exact match, so the {descriptions[0]} was relaxed."
    listed = ", ".join(descriptions[:-1]) + f" and {descriptions[-1]}"
    return f"No exact match, so the {listed} were relaxed."
