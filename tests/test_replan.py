from __future__ import annotations

from typing import Any

import pytest

from app.agents.replan import (
    PROTECTED_FIELDS,
    RELAXATION_LADDER,
    RETRIEVAL_INTENTS,
    candidate_relaxations,
    describe_relaxations,
    has_required_entities,
    replan_until_results,
    result_count,
)
from app.agents.schemas import AgentIntent


def intent(**overrides: Any) -> AgentIntent:
    base: dict[str, Any] = {"intent": "recommend"}
    base.update(overrides)
    return AgentIntent(**base)


def response_with(count: int) -> dict[str, Any]:
    return {"trace": [{"tool": "recommend_anime", "result": {"results": [{"id": n} for n in range(count)]}}]}


def test_ladder_never_relaxes_a_grounding_constraint():
    """Required entity constraints are the agent's correctness contract."""
    ladder_fields = {field for field, _description, _transform in RELAXATION_LADDER}
    assert not ladder_fields & PROTECTED_FIELDS


def test_result_count_sums_rows_across_trace_steps():
    payload = {
        "trace": [
            {"tool": "search_anime", "result": {"results": [{"id": 1}, {"id": 2}]}},
            {"tool": "recommend_anime", "result": {"results": [{"id": 3}]}},
            {"tool": "resolve_entity", "result": {"entity": "x"}},
        ]
    }
    assert result_count(payload) == 3
    assert result_count({"trace": []}) == 0
    assert result_count({}) == 0


def test_candidate_relaxations_follow_ladder_order():
    over_constrained = intent(min_score=9.5, max_episodes=12, include_genres=["Action"])
    assert candidate_relaxations(over_constrained) == ["min_score", "max_episodes", "include_genres"]


def test_replan_stops_as_soon_as_results_appear():
    calls: list[tuple[AgentIntent, frozenset[str]]] = []

    def execute(candidate: AgentIntent, relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        calls.append((candidate, relaxed))
        # Recovers only once the score floor is gone.
        return response_with(0 if candidate.min_score is not None else 4)

    start = intent(min_score=9.9, max_episodes=12)
    final, payload, steps = replan_until_results(start, execute, max_steps=3)

    assert [step.field for step in steps] == ["min_score"]
    assert result_count(payload) == 4
    assert final.min_score is None
    # The episode cap was never reached because the first relaxation worked.
    assert final.max_episodes == 12
    # The executor is told which fields were dropped, so downstream
    # normalisation cannot reinstate them from the raw message.
    assert calls[-1][1] == frozenset({"min_score"})


def test_replan_respects_the_step_budget():
    def execute(_candidate: AgentIntent, _relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        return response_with(0)

    start = intent(min_score=9.9, max_episodes=1, min_year=2024, formats=["TV"])
    _final, _payload, steps = replan_until_results(start, execute, max_steps=2)
    assert len(steps) == 2


def test_replan_reuses_the_initial_response_without_re_executing():
    executions: list[AgentIntent] = []

    def execute(candidate: AgentIntent, relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        executions.append(candidate)
        return response_with(3)

    start = intent(min_score=8.0)
    _final, payload, steps = replan_until_results(
        start,
        execute,
        max_steps=2,
        initial_response=response_with(3),
    )
    assert executions == []
    assert steps == []
    assert result_count(payload) == 3


def test_replan_is_skipped_for_non_retrieval_intents():
    def execute(_candidate: AgentIntent, _relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        return response_with(0)

    start = intent(intent="conversation")
    _final, _payload, steps = replan_until_results(start, execute, max_steps=3)
    assert steps == []
    assert "conversation" not in RETRIEVAL_INTENTS


def test_replan_gives_up_when_nothing_remains_to_relax():
    def execute(_candidate: AgentIntent, _relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        return response_with(0)

    # Only a required voice actor, which the ladder must never touch.
    start = intent(required_voice_actors=["Matsuoka, Yoshitsugu"])
    final, _payload, steps = replan_until_results(start, execute, max_steps=3)
    assert steps == []
    assert final.required_voice_actors == ["Matsuoka, Yoshitsugu"]
    assert has_required_entities(final)


def test_max_steps_zero_disables_replanning():
    def execute(_candidate: AgentIntent, _relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        return response_with(0)

    _final, _payload, steps = replan_until_results(intent(min_score=9.9), execute, max_steps=0)
    assert steps == []


def test_negative_step_budget_is_rejected():
    with pytest.raises(ValueError):
        replan_until_results(intent(), lambda _candidate, _relaxed: response_with(1), max_steps=-1)


def test_describe_relaxations_reads_as_a_sentence():
    def execute(candidate: AgentIntent, relaxed: frozenset[str] = frozenset()) -> dict[str, Any]:
        return response_with(0 if candidate.max_episodes is not None else 2)

    _final, _payload, steps = replan_until_results(
        intent(min_score=9.9, max_episodes=6),
        execute,
        max_steps=3,
    )
    sentence = describe_relaxations(steps)
    assert sentence.startswith("No exact match")
    assert "minimum score filter" in sentence
    assert describe_relaxations([]) == ""
