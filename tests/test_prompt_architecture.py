"""The static-policy / dynamic-state split, asserted rather than assumed.

The separation is only worth having if it is enforced. A rule that drifts back
into a prompt, a routing default duplicated in prose, or a required constraint
quietly reinstated during a replan are all invisible in review and obvious in a
test, so each gets one here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_showcase import make_artifact

from app.agents.prompting import (
    INTENT_TASK_POLICY,
    RESPONSE_TASK_POLICY,
    SYSTEM_POLICY,
    render_intent_prompt,
    render_response_prompt,
)
from app.agents.replan import RELAXATION_LADDER, replan_with_state
from app.agents.runtime_state import (
    OPTIONAL_CONSTRAINT_FIELDS,
    REQUIRED_CONSTRAINT_FIELDS,
    ReplanState,
    build_runtime_context,
    build_tool_observation,
)
from app.agents.schemas import AgentIntent
from backend.anime_agent.fast_path import FastPathConfig
from backend.anime_agent.showcase import load_showcase_service

ALL_POLICY = (SYSTEM_POLICY, INTENT_TASK_POLICY, RESPONSE_TASK_POLICY)


def trace(*rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "test",
        "trace": [
            {
                "tool": "recommend_anime",
                "result": {"results": list(rows), "result_titles": [row["title"] for row in rows]},
            }
        ],
    }


# ------------------------------------------- 1. no user data in static policy


@pytest.mark.parametrize("policy", ALL_POLICY, ids=["system", "intent", "response"])
def test_static_policy_carries_no_user_or_catalog_specifics(policy: str):
    """A name, a title, or a genre list in the policy is a fact frozen forever."""
    forbidden = (
        "Steins;Gate",
        "Death Note",
        "Cowboy Bebop",
        "Madhouse",
        "Matsuoka",
        "liked_titles",
        "disliked_titles",
        "excluded_titles are",
    )
    for token in forbidden:
        assert token not in policy, f"static policy leaked request-specific data: {token!r}"


@pytest.mark.parametrize("policy", ALL_POLICY, ids=["system", "intent", "response"])
def test_static_policy_carries_no_experiment_conclusions_or_model_settings(policy: str):
    """These have changed once already; prose is the one place no test guards."""
    forbidden = (
        "CountSketch",
        "ALS",
        "item-item",
        "item_item",
        "LightFM",
        "segment_aware",
        "segment-aware",
        "diversity_strength",
        "NDCG",
        "0.2588",
        "top-300",
        "Top-300",
    )
    for token in forbidden:
        assert token not in policy, f"static policy hard-codes a runtime decision: {token!r}"


def test_system_policy_defers_routing_to_the_application():
    assert "routing policy provided by the application" in SYSTEM_POLICY
    assert "authoritative for ordering" in SYSTEM_POLICY


# ------------------------------ 2. runtime state varies, policy does not


def test_runtime_state_changes_between_requests_while_policy_is_identical():
    first = render_intent_prompt(
        "something like Steins;Gate",
        build_runtime_context(catalog_genres=["Sci-Fi"], session={"liked_titles": ["Steins;Gate"]}),
    )
    second = render_intent_prompt(
        "something romantic",
        build_runtime_context(catalog_genres=["Romance"], session={"liked_titles": ["Toradora!"]}),
    )
    assert first != second
    assert "Steins;Gate" in first and "Steins;Gate" not in second
    # The policy half is byte-identical across the two requests.
    assert first.split("RUNTIME STATE")[0] == second.split("RUNTIME STATE")[0]
    assert SYSTEM_POLICY in first and SYSTEM_POLICY in second


def test_runtime_state_is_labelled_as_data_not_instructions():
    prompt = render_intent_prompt("hi", build_runtime_context(session={"liked_titles": ["A"]}))
    assert "not instructions" in prompt


def test_absent_facts_are_omitted_rather_than_sent_as_empty():
    """An empty list reads as 'the user likes nothing', which is not what we know."""
    payload = build_runtime_context(session={}).as_prompt_payload()
    assert "liked_titles" not in payload
    assert "disliked_titles" not in payload


# ------------------------------------- 3-4. replanning keeps its promises


def intent_with_required_and_optional() -> AgentIntent:
    return AgentIntent(
        intent="recommend",
        required_voice_actors=["Matsuoka, Yoshitsugu"],
        required_studios=["Madhouse"],
        min_score=8.5,
        max_episodes=12,
        include_genres=["Action"],
    )


def test_required_constraints_survive_every_replan():
    intent = intent_with_required_and_optional()
    seen: list[AgentIntent] = []

    def execute(candidate: AgentIntent, relaxed: frozenset[str]) -> dict[str, Any]:
        seen.append(candidate)
        return trace()  # never any results, so the ladder runs to exhaustion

    outcome = replan_with_state(intent, execute, max_steps=3)
    assert outcome.state.required_constraints == {
        "required_studios": ["Madhouse"],
        "required_voice_actors": ["Matsuoka, Yoshitsugu"],
    }
    for candidate in seen:
        assert candidate.required_voice_actors == ["Matsuoka, Yoshitsugu"]
        assert candidate.required_studios == ["Madhouse"]
    assert outcome.intent.required_voice_actors == ["Matsuoka, Yoshitsugu"]


def test_relaxed_optional_constraints_are_never_reintroduced():
    """The original bug: a later step re-derived the intent and restored min_score."""
    intent = intent_with_required_and_optional()
    seen: list[AgentIntent] = []

    def execute(candidate: AgentIntent, relaxed: frozenset[str]) -> dict[str, Any]:
        seen.append(candidate)
        return trace()

    outcome = replan_with_state(intent, execute, max_steps=3)
    assert outcome.state.relaxed_fields, "expected at least one relaxation"
    for field in outcome.state.relaxed_fields:
        assert field not in outcome.state.optional_constraints
        # Once dropped, the field stays dropped on every subsequent execution.
        dropped_at = next(i for i, c in enumerate(seen) if not getattr(c, field, None))
        for later in seen[dropped_at:]:
            assert not getattr(later, field, None), f"{field} came back after being relaxed"


def test_the_ladder_can_never_reach_a_required_field():
    ladder_fields = {field for field, _d, _t in RELAXATION_LADDER}
    assert not ladder_fields & set(REQUIRED_CONSTRAINT_FIELDS)
    assert ladder_fields <= set(OPTIONAL_CONSTRAINT_FIELDS)


def test_replan_state_refuses_to_advance_if_a_required_constraint_moved():
    """Enforced on the state itself, not only by the shape of the ladder."""
    intent = intent_with_required_and_optional()
    state = ReplanState.initial(intent, max_replans=2)
    illegal = intent.model_copy(update={"required_studios": []})
    with pytest.raises(ValueError, match="required entity constraints must survive"):
        state.after_relaxing("required_studios", illegal)


# ------------------------------------------------ 8. the budget is bounded


@pytest.mark.parametrize("max_steps", [0, 1, 2, 3])
def test_max_replan_count_is_enforced(max_steps: int):
    calls: list[frozenset[str]] = []

    def execute(candidate: AgentIntent, relaxed: frozenset[str]) -> dict[str, Any]:
        calls.append(relaxed)
        return trace()

    outcome = replan_with_state(intent_with_required_and_optional(), execute, max_steps=max_steps)
    assert len(outcome.state.relaxed_fields) <= max_steps
    assert outcome.state.replan_count <= outcome.state.max_replans
    # One unrelaxed execution, then at most max_steps relaxed ones.
    assert len(calls) <= max_steps + 1


def test_replanning_stops_early_once_results_appear():
    def execute(candidate: AgentIntent, relaxed: frozenset[str]) -> dict[str, Any]:
        return trace() if not relaxed else trace({"title": "Found", "genres": ["Action"]})

    outcome = replan_with_state(intent_with_required_and_optional(), execute, max_steps=3)
    assert len(outcome.state.relaxed_fields) == 1


# ------------------------- 5. the LLM cannot override routing configuration


def test_intent_schema_has_no_model_selection_fields():
    """Routing lives in FastPathConfig; the LLM has no vocabulary to change it."""
    fields = set(AgentIntent.model_fields)
    for knob in ("segment_aware", "item_item_top_m", "diversity_strength", "reranker_enabled", "als_top_n", "route"):
        assert knob not in fields


def test_an_llm_cannot_smuggle_routing_settings_through_the_intent():
    with pytest.raises(ValidationError):
        AgentIntent.model_validate({"intent": "recommend", "segment_aware": True})
    with pytest.raises(ValidationError):
        AgentIntent.model_validate({"intent": "recommend", "reranker_enabled": True})


def test_routing_defaults_come_from_config_not_from_any_prompt():
    config = FastPathConfig()
    assert config.routing.segment_aware is False
    assert config.retrieval.item_item_top_m == 0
    assert config.diversity_strength == 0.0
    for policy in ALL_POLICY:
        assert "segment_aware" not in policy


def test_runtime_state_reports_route_decisions_without_exposing_knobs():
    context = build_runtime_context(selected_route="fast", candidate_count=300, reranker_enabled=False)
    payload = context.as_prompt_payload()
    assert payload["selected_route"] == "fast"
    assert payload["candidate_count"] == 300
    assert "diversity_strength" not in payload
    assert "item_item_top_m" not in payload


# -------------------------- 6-7. tool results are the only source of truth


def test_explanations_can_only_reference_fields_the_tools_returned():
    observation = build_tool_observation(trace({"title": "Example Anime", "genres": ["Psychological", "Mystery"]}))
    prompt = render_response_prompt("why these?", observation)
    assert "Example Anime" in prompt and "Psychological" in prompt
    # Nothing supplied a studio or a staff credit, so neither may appear.
    assert "studio" not in prompt.split("VERIFIED EVIDENCE")[1].casefold()
    assert "Madhouse" not in prompt


def test_the_response_policy_forbids_inventing_absent_fields():
    flat = " ".join(SYSTEM_POLICY.split())
    assert "absent from the evidence" in RESPONSE_TASK_POLICY
    assert "do not supply it from your own knowledge and do not guess" in flat


def test_tool_result_order_is_preserved_exactly():
    rows = [{"title": f"T{i}", "genres": []} for i in range(6)]
    observation = build_tool_observation(trace(*rows))
    assert observation.result_titles == tuple(f"T{i}" for i in range(6))
    assert [row["title"] for row in observation.results] == [f"T{i}" for i in range(6)]


def test_raw_tool_payloads_are_summarised_not_replayed_into_runtime_state():
    """Runtime state carries counts and titles; the payload stays in evidence."""
    summary = build_tool_observation(trace({"title": "A", "genres": ["X"], "internal_score": 0.93})).summary()
    assert summary.result_count == 1 and summary.result_titles == ("A",)
    assert "internal_score" not in summary.model_dump_json()


# ---------------------- 9. the recommendation path is untouched by all this


@pytest.fixture
def deterministic_service():
    catalog = [
        {
            "id": anime_id,
            "title": title,
            "genres": genres,
            "start_year": year,
            "type": "TV",
            "episodes": 12,
            "score": 8.0,
            "members": 10_000 - anime_id,
            "synopsis": "S.",
            "image_url": "",
        }
        for anime_id, title, genres, year in [
            (1, "Alpha", ["Sci-Fi"], 2010),
            (2, "Beta", ["Romance"], 2011),
            (3, "Gamma", ["Sci-Fi"], 2012),
            (4, "Delta", ["Romance"], 2013),
            (5, "Epsilon", ["Sci-Fi"], 2014),
            (6, "Zeta Movie", ["Romance"], 2015),
            (7, "Alpha Season 2", ["Sci-Fi"], 2016),
        ]
    ]
    directory = Path(tempfile.mkdtemp())
    artifact = make_artifact(directory / "als.npz", sorted(int(item["id"]) for item in catalog))
    return load_showcase_service(catalog, artifact)


@pytest.mark.parametrize(
    ("liked", "expected"),
    [
        ([1], [3, 5, 2, 4]),
        ([1, 3], [5, 2, 4, 6]),
        ([2, 4], [6, 1, 3, 5]),
        ([1, 3, 5], [2, 4, 6]),
    ],
    ids=["one", "two-scifi", "romance", "three-scifi"],
)
def test_representative_requests_return_the_same_recommendations(deterministic_service, liked, expected):
    """Frozen before the prompt refactor. The agent layer must not move these."""
    assert [item.anime_id for item in deterministic_service.recommend(liked, limit=4).items] == expected
