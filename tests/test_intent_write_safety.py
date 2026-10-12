"""Informational requests must not change what a session remembers.

The failure these tests pin down: "Introduce Cowboy Bebop without spoilers."
was answered with "I updated this session's preferences". The model had
classified the request correctly as `details`; the backend's own normalization
read "without" as negative feedback and rerouted it to a session write, and the
only tool-contract check ran after that write. On the rule path the same
message stored Cowboy Bebop as an excluded title.

Two properties are asserted throughout, at the HTTP boundary and against the
real session store:

* an information, search or recommendation request leaves the stored
  preferences exactly as they were, whatever intent the model proposes;
* a preference the user states in so many words is still stored.

The provider here is scripted rather than mocked to be helpful: most cases feed
the orchestrator a *wrong* structured intent on purpose, because the boundary
under test is the one that has to hold when the model is wrong.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.gemini_provider import GeminiAgentProvider
from app.agents.schemas import AgentIntent, IntentPreferenceUpdate, ProviderHealth, ProviderParseContext
from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.main import create_app
from app.repositories.session_repository import SESSION_CONTEXT_FIELDS, SQLiteSessionRepository
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.write_policy import (
    DISLIKED,
    EXCLUDED,
    LIKED,
    RESET,
    WATCHED,
    PreferenceWriteAuthorization,
    PreferenceWriteGate,
    preference_signals,
)
from tests.conftest import anime, wait_until_ready

READ_ONLY_TOOLS = {"get_anime_details", "anime_details", "search_anime", "search_entities", "resolve_entity"}
WRITE_TOOL = "update_session_preferences"
SEEDED_PREFERENCES = {"liked_titles": ["Ghost Hunt"], "excluded_genres": ["Romance"]}


def safety_catalog() -> list[dict[str, Any]]:
    return [
        anime(1, "Death Note", genres=["Supernatural", "Mystery"], synopsis="A dark psychological battle of wits."),
        anime(2, "Death Parade", genres=["Supernatural", "Drama"], synopsis="Souls are judged through bar games."),
        anime(3, "Ghost Hunt", genres=["Supernatural", "Mystery"], synopsis="A team investigates ghosts and spirits."),
        anime(4, "Quiet Romance", genres=["Romance"], synopsis="Two students build a gentle relationship."),
        anime(
            20,
            "Cowboy Bebop",
            genres=["Action", "Sci-Fi"],
            synopsis="A crew of bounty hunters drifts between planets chasing fugitives and their own pasts.",
            score=8.8,
        ),
        anime(21, "Trigun", genres=["Action", "Sci-Fi"], synopsis="A gunman with a bounty wanders a desert planet."),
        anime(22, "Space Dandy", genres=["Comedy", "Sci-Fi"], synopsis="An alien hunter drifts between planets."),
        anime(23, "Outlaw Star", genres=["Action", "Sci-Fi"], synopsis="A crew of outlaws chases a galactic treasure."),
        anime(24, "Piano Forest", genres=["Drama", "Music"], synopsis="Two boys and a piano hidden in the woods."),
    ]


class ScriptedProvider:
    """A provider that says exactly what a test tells it to, right or wrong."""

    name = "gemini"
    model = "scripted"

    def __init__(self) -> None:
        self.proposal: AgentIntent | Exception = AgentIntent(intent="conversation")

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        if isinstance(self.proposal, Exception):
            raise self.proposal
        return self.proposal.model_copy(deep=True)

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


class RecordingSessions(SQLiteSessionRepository):
    """The real store, plus a list of every patch anything asked it to apply."""

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)
        self.patches: list[dict[str, Any]] = []

    def update(self, session_id: str | None, patch: dict[str, Any]) -> dict[str, Any]:
        self.patches.append(dict(patch))
        return super().update(session_id, patch)

    def preference_patches(self) -> list[dict[str, Any]]:
        """Patches that touched a preference, as opposed to conversation context."""
        return [patch for patch in self.patches if set(patch) - SESSION_CONTEXT_FIELDS - {"replace"}]


class Harness:
    def __init__(self, client: TestClient, provider: ScriptedProvider, sessions: RecordingSessions) -> None:
        self.client = client
        self.provider = provider
        self.sessions = sessions

    def new_session(self, *, seeded: bool = True) -> str:
        session_id = "safety-" + uuid.uuid4().hex
        if seeded:
            self.sessions.update(session_id, dict(SEEDED_PREFERENCES))
        self.sessions.patches.clear()
        return session_id

    def preferences(self, session_id: str) -> dict[str, Any]:
        profile = self.client.get(f"/api/session/{session_id}").json()["profile"]
        return {key: value for key, value in profile.items() if key not in SESSION_CONTEXT_FIELDS}

    def chat(
        self,
        message: str,
        proposal: AgentIntent | Exception,
        session_id: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        self.provider.proposal = proposal
        # A scripted outage counts toward the circuit breaker like a real one.
        # Left open, it would quietly send every later case down the rule path
        # and these tests would stop exercising the proposals they name.
        agent = self.client.app.state.container.agent  # type: ignore[attr-defined]
        for name in agent.circuits:
            agent._record_success(name)
        response = self.client.post(
            "/api/chat",
            json={"message": message, "session_id": session_id, "history": history or [], "debug": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        expected_parser = None if isinstance(proposal, Exception) else "gemini"
        assert body["debug"]["intent_provider"] == expected_parser
        return body


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Harness]:
    database = tmp_path_factory.mktemp("write-safety") / "sessions.db"
    settings = Settings(
        database_url=f"sqlite:///{database.as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=10_000,
        startup_warm_grace_seconds=60,
        embedding_provider="none",
        collaborative_enabled=False,
        als_enabled=False,
        reranker_enabled=False,
    )
    provider = ScriptedProvider()
    sessions = RecordingSessions(settings.database_url)
    application = create_app(
        settings=settings,
        catalog=safety_catalog(),
        providers={"gemini": provider},
        session_repository=sessions,
    )
    with TestClient(application) as client:
        wait_until_ready(client)
        yield Harness(client, provider, sessions)


def tools(body: dict[str, Any]) -> list[str]:
    return [step["tool"] for step in body["trace"]]


def subject_update(**fields: list[str]) -> AgentIntent:
    """A wrong `update_preferences` intent that tries to write `fields`."""
    return AgentIntent(
        intent="update_preferences",
        excluded_titles=fields.get("excluded_titles", []),
        seen_titles=fields.get("watched_titles", []),
        preference_update=IntentPreferenceUpdate(**fields),
    )


# What a parser might say about an information request. The first is right; the
# rest are the mistakes the backend has to survive.
PROPOSALS: dict[str, AgentIntent | Exception] = {
    "correct_details": AgentIntent(intent="details", reference_titles=["Cowboy Bebop"]),
    "details_without_a_title": AgentIntent(intent="details"),
    "empty_update": AgentIntent(intent="update_preferences"),
    "update_that_excludes_the_subject": subject_update(
        excluded_titles=["Cowboy Bebop"], disliked_titles=["Cowboy Bebop"]
    ),
    "update_that_likes_the_subject": subject_update(liked_titles=["Cowboy Bebop"], watched_titles=["Cowboy Bebop"]),
    "update_that_excludes_a_genre": AgentIntent(
        intent="update_preferences",
        exclude_genres=["Mystery"],
        preference_update=IntentPreferenceUpdate(excluded_genres=["Mystery"]),
    ),
    "provider_unavailable": ProviderUnavailable("gemini", "scripted outage"),
}

INFORMATION_REQUESTS = (
    "Introduce Cowboy Bebop without spoilers.",
    "Tell me about Cowboy Bebop.",
    "What is Cowboy Bebop about?",
    "Who are the main characters in Cowboy Bebop?",
    "Give me a spoiler-free synopsis of Cowboy Bebop.",
    "Introduce Cowboy Bebop, no spoilers please.",
    "Introduce Cowboy Bebop but avoid spoilers.",
    "Tell me about Cowboy Bebop, I hate spoilers.",
    "Whatever, just tell me about Cowboy Bebop.",
)


@pytest.mark.parametrize("message", INFORMATION_REQUESTS)
@pytest.mark.parametrize("proposal_name", list(PROPOSALS))
def test_information_request_reads_the_catalog_and_writes_nothing(
    harness: Harness,
    message: str,
    proposal_name: str,
) -> None:
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat(message, PROPOSALS[proposal_name], session_id)

    assert body["validated_intent"]["intent"] == "details"
    assert tools(body) == ["get_anime_details"]
    assert body["trace"][0]["result"]["result"]["title"] == "Cowboy Bebop"
    assert "Cowboy Bebop" in body["answer"]
    assert "updated this session" not in body["answer"]
    assert body["debug"]["write_safety"]["written_fields"] == []
    assert harness.preferences(session_id) == before == {**before, **SEEDED_PREFERENCES}
    assert harness.sessions.preference_patches() == []


def test_the_reported_failure_is_explained_in_the_debug_payload(harness: Harness) -> None:
    session_id = harness.new_session()

    body = harness.chat(
        "Introduce Cowboy Bebop without spoilers.",
        PROPOSALS["update_that_excludes_the_subject"],
        session_id,
    )

    assert body["debug"]["proposed_intent"] == "update_preferences"
    assert body["validated_intent"]["intent"] == "details"
    safety = body["debug"]["write_safety"]
    assert safety["signals"] == []
    assert safety["rerouted_from"] == "update_preferences"
    assert set(safety["denied_fields"]) == {"disliked_titles", "excluded_titles"}
    assert {"phase": "write_eligibility", "provider": "backend", "outcome": "denied"}.items() <= next(
        attempt for attempt in body["debug"]["provider_attempts"] if attempt["phase"] == "write_eligibility"
    ).items()


def test_a_fresh_session_is_not_even_created_by_an_information_request(harness: Harness) -> None:
    """The old empty-patch write still created a session row and claimed an update."""
    session_id = harness.new_session(seeded=False)

    harness.chat("Introduce Cowboy Bebop without spoilers.", PROPOSALS["empty_update"], session_id)

    assert harness.sessions.patches == []


@pytest.mark.parametrize(
    ("message", "proposal"),
    [
        ("Search for Death Note.", AgentIntent(intent="search", catalog_query="Death Note")),
        ("Search for Death Note.", subject_update(liked_titles=["Death Note"])),
        ("Who voices the lead in Death Note?", AgentIntent(intent="search", catalog_query="Death Note")),
    ],
)
def test_search_request_is_read_only(harness: Harness, message: str, proposal: AgentIntent) -> None:
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat(message, proposal, session_id)

    assert tools(body) and set(tools(body)) <= READ_ONLY_TOOLS
    assert harness.preferences(session_id) == before
    assert harness.sessions.preference_patches() == []


@pytest.mark.parametrize(
    "proposal",
    [
        AgentIntent(intent="recommend", reference_titles=["Cowboy Bebop"], top_k=3),
        # "Similar to X" does not say X was liked, watched or unwanted, whatever the parser adds.
        AgentIntent(
            intent="recommend",
            reference_titles=["Cowboy Bebop"],
            top_k=3,
            seen_titles=["Cowboy Bebop"],
            preference_update=IntentPreferenceUpdate(
                liked_titles=["Cowboy Bebop"], watched_titles=["Cowboy Bebop"], excluded_genres=["Mystery"]
            ),
        ),
        subject_update(liked_titles=["Cowboy Bebop"]),
        ProviderUnavailable("gemini", "scripted outage"),
    ],
    ids=["correct", "recommend_with_invented_preferences", "misrouted_to_update", "provider_unavailable"],
)
def test_recommendation_request_keeps_context_but_not_preferences(
    harness: Harness,
    proposal: AgentIntent | Exception,
) -> None:
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat("Recommend anime similar to Cowboy Bebop.", proposal, session_id)

    assert body["validated_intent"]["intent"] == "recommend"
    assert "recommend_anime" in tools(body)
    assert WRITE_TOOL not in tools(body)
    recommended = next(step for step in body["trace"] if step["tool"] == "recommend_anime")["result"]["results"]
    assert recommended and all(item["title"] != "Cowboy Bebop" for item in recommended)
    assert harness.preferences(session_id) == before
    assert harness.sessions.preference_patches() == []
    # Conversation context is not a preference and is still recorded, so
    # "the second one" resolves on the next turn.
    profile = harness.client.get(f"/api/session/{session_id}").json()["profile"]
    assert profile["last_recommendations"] == [item["title"] for item in recommended]


EXPLICIT_UPDATES = (
    (
        "I like Cowboy Bebop.",
        subject_update(liked_titles=["Cowboy Bebop"]),
        {"liked_titles": ["Ghost Hunt", "Cowboy Bebop"]},
    ),
    (
        "I've already watched Cowboy Bebop.",
        subject_update(watched_titles=["Cowboy Bebop"]),
        {"seen_titles": ["Cowboy Bebop"]},
    ),
    (
        "Don't recommend Cowboy Bebop.",
        subject_update(excluded_titles=["Cowboy Bebop"]),
        {"excluded_titles": ["Cowboy Bebop"]},
    ),
    (
        "I do not like mystery.",
        AgentIntent(
            intent="update_preferences",
            exclude_genres=["Mystery"],
            preference_update=IntentPreferenceUpdate(excluded_genres=["Mystery"]),
        ),
        {"excluded_genres": ["Romance", "Mystery"]},
    ),
)


@pytest.mark.parametrize(("message", "proposal", "expected"), EXPLICIT_UPDATES)
def test_stated_preference_is_stored(
    harness: Harness,
    message: str,
    proposal: AgentIntent,
    expected: dict[str, list[str]],
) -> None:
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat(message, proposal, session_id)

    assert body["mode"] == "session_update"
    assert tools(body) == [WRITE_TOOL]
    assert body["debug"]["write_safety"]["rerouted_from"] is None
    assert harness.preferences(session_id) == {**before, **expected}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I like Cowboy Bebop.", {"liked_titles": ["Ghost Hunt", "Cowboy Bebop"]}),
        ("I've already watched Cowboy Bebop.", {"seen_titles": ["Cowboy Bebop"]}),
        ("Don't recommend Cowboy Bebop.", {"excluded_titles": ["Cowboy Bebop"]}),
    ],
)
def test_stated_preference_is_stored_without_the_model(
    harness: Harness,
    message: str,
    expected: dict[str, list[str]],
) -> None:
    """The deterministic fallback must keep working when the provider is down."""
    session_id = harness.new_session()

    harness.chat(message, ProviderUnavailable("gemini", "scripted outage"), session_id)

    stored = harness.preferences(session_id)
    for name, values in expected.items():
        assert stored[name] == values


def test_explicit_exclusion_does_not_depend_on_the_models_intent(harness: Harness) -> None:
    """A parser that answers "don't recommend X" with recommendations is overruled."""
    session_id = harness.new_session()

    body = harness.chat(
        "Don't recommend Cowboy Bebop.",
        AgentIntent(intent="recommend", excluded_titles=["Cowboy Bebop"], top_k=3),
        session_id,
    )

    assert body["mode"] == "session_update"
    assert tools(body) == [WRITE_TOOL]
    assert harness.preferences(session_id)["excluded_titles"] == ["Cowboy Bebop"]


def test_only_the_stated_part_of_a_proposed_update_is_stored(harness: Harness) -> None:
    session_id = harness.new_session()

    body = harness.chat(
        "I like Cowboy Bebop.",
        subject_update(liked_titles=["Cowboy Bebop"], excluded_titles=["Death Note"], disliked_titles=["Death Note"]),
        session_id,
    )

    stored = harness.preferences(session_id)
    assert stored["liked_titles"] == ["Ghost Hunt", "Cowboy Bebop"]
    assert stored["excluded_titles"] == []
    assert stored["disliked_titles"] == []
    assert body["trace"][0]["arguments"] == {"liked_titles": ["Cowboy Bebop"]}
    assert set(body["debug"]["write_safety"]["denied_fields"]) == {"disliked_titles", "excluded_titles"}


def test_taste_stated_inside_a_recommendation_request_is_still_stored(harness: Harness) -> None:
    session_id = harness.new_session()

    body = harness.chat(
        "I enjoyed Cowboy Bebop, recommend something similar but not Trigun.",
        AgentIntent(
            intent="recommend",
            reference_titles=["Cowboy Bebop"],
            excluded_titles=["Trigun"],
            seen_titles=["Cowboy Bebop"],
            top_k=3,
            preference_update=IntentPreferenceUpdate(liked_titles=["Cowboy Bebop"], watched_titles=["Cowboy Bebop"]),
        ),
        session_id,
    )

    stored = harness.preferences(session_id)
    assert "recommend_anime" in tools(body)
    assert stored["liked_titles"] == ["Ghost Hunt", "Cowboy Bebop"]
    assert stored["seen_titles"] == ["Cowboy Bebop"]
    assert stored["excluded_titles"] == ["Trigun"]


@pytest.mark.parametrize(
    "message",
    ["Cowboy Bebop", "Hmm, what do you think?", "Why do people hate Cowboy Bebop?"],
)
def test_ambiguous_message_gets_a_clarifying_reply_instead_of_a_write(harness: Harness, message: str) -> None:
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat(message, subject_update(liked_titles=["Cowboy Bebop"]), session_id)

    assert body["mode"] == "preference_clarification"
    assert body["trace"] == []
    assert "not changed" in body["answer"]
    assert body["validated_intent"]["intent"] == "conversation"
    assert harness.preferences(session_id) == before
    assert harness.sessions.preference_patches() == []


def test_follow_up_about_a_previous_result_is_read_only(harness: Harness) -> None:
    session_id = harness.new_session()
    first = harness.chat(
        "Recommend anime similar to Cowboy Bebop.",
        AgentIntent(intent="recommend", reference_titles=["Cowboy Bebop"], top_k=3),
        session_id,
    )
    listed = harness.client.get(f"/api/session/{session_id}").json()["profile"]["last_recommendations"]
    assert len(listed) >= 2
    history = [
        {"role": "user", "content": "Recommend anime similar to Cowboy Bebop."},
        {"role": "assistant", "content": first["answer"]},
    ]
    before = harness.preferences(session_id)

    # The parser points at the right position but labels it "watched".
    follow_up = harness.chat(
        "Tell me more about the second one.",
        AgentIntent(intent="update_preferences", watched_result_indices=[2]),
        session_id,
        history,
    )

    assert follow_up["validated_intent"]["intent"] == "details"
    assert tools(follow_up) == ["get_anime_details"]
    assert follow_up["trace"][0]["result"]["result"]["title"] == listed[1]
    assert harness.preferences(session_id) == before
    assert harness.sessions.preference_patches() == []

    # And a real statement about a position is still honoured.
    watched = harness.chat(
        "I watched the first one.",
        AgentIntent(intent="update_preferences", watched_result_indices=[1]),
        session_id,
        history,
    )

    assert watched["mode"] == "session_update"
    assert harness.preferences(session_id)["seen_titles"] == [listed[0]]


@pytest.mark.parametrize(
    "proposal",
    [
        AgentIntent(intent="details", reference_result_indices=[3]),
        # The parser repeats the title the list was built from.
        AgentIntent(intent="details", reference_titles=["Cowboy Bebop"], reference_result_indices=[3]),
        ProviderUnavailable("gemini", "scripted outage"),
    ],
    ids=["position_only", "position_and_seed_title", "provider_unavailable"],
)
def test_follow_up_by_position_describes_that_result_not_the_seed_title(
    harness: Harness,
    proposal: AgentIntent | Exception,
) -> None:
    """On the rule path this used to introduce Cowboy Bebop, the title the list was built from."""
    session_id = harness.new_session()
    first = harness.chat(
        "Recommend anime similar to Cowboy Bebop.",
        AgentIntent(intent="recommend", reference_titles=["Cowboy Bebop"], top_k=3),
        session_id,
    )
    listed = harness.client.get(f"/api/session/{session_id}").json()["profile"]["last_recommendations"]
    assert len(listed) >= 3
    history = [
        {"role": "user", "content": "Recommend anime similar to Cowboy Bebop."},
        {"role": "assistant", "content": first["answer"]},
    ]
    before = harness.preferences(session_id)

    body = harness.chat("What is the third one about?", proposal, session_id, history)

    assert tools(body) == ["get_anime_details"]
    assert body["trace"][0]["result"]["result"]["title"] == listed[2]
    assert harness.preferences(session_id) == before


@pytest.mark.parametrize(
    ("message", "proposal"),
    [
        ("Tell me more about the second one.", AgentIntent(intent="details", reference_result_indices=[2])),
        ("Tell me more about the second one.", AgentIntent(intent="recommend", reference_result_indices=[2])),
        ("Tell me more about the second one.", ProviderUnavailable("gemini", "scripted outage")),
        ("I watched the second one.", AgentIntent(intent="update_preferences", watched_result_indices=[2])),
    ],
    ids=["details", "recommend", "provider_unavailable", "watched"],
)
def test_follow_up_without_a_previous_list_asks_instead_of_failing(
    harness: Harness,
    message: str,
    proposal: AgentIntent | Exception,
) -> None:
    """This was HTTP 500: "no tool ran" failed the tool contract twice over."""
    session_id = harness.new_session()
    before = harness.preferences(session_id)

    body = harness.chat(message, proposal, session_id)

    assert body["mode"] == "catalog_context_error"
    assert body["trace"] == []
    assert "previous result list" in body["answer"]
    assert harness.preferences(session_id) == before
    assert harness.sessions.preference_patches() == []


# ------------------------------------------------------------- policy, directly


@pytest.mark.parametrize(
    "message",
    [
        "Introduce Cowboy Bebop without spoilers.",
        "Tell me about Frieren.",
        "What is Steins;Gate about?",
        "Who are the main characters in Cowboy Bebop?",
        "Recommend anime similar to Cowboy Bebop.",
        "Tell me more about the second one.",
        "Whatever, tell me about Frieren",
        "Anime with no more than 12 episodes please",
        "Would I like Monster?",
        "I would like to know about Cowboy Bebop",
        "Have you seen Cowboy Bebop?",
        "I haven't watched Cowboy Bebop",
        "Something like Cowboy Bebop but not like the others",
        "Is Cowboy Bebop unavoidable for a sci-fi fan?",
    ],
)
def test_messages_that_state_no_preference_authorize_no_write(message: str) -> None:
    assert preference_signals(message) == frozenset()
    authorized, denied = PreferenceWriteAuthorization(preference_signals(message)).split(
        {"liked_titles": ["X"], "seen_titles": ["X"], "excluded_titles": ["X"], "excluded_genres": ["Y"], "reset": True}
    )
    assert authorized == {}
    assert set(denied) == {"liked_titles", "seen_titles", "excluded_titles", "excluded_genres", "reset"}


@pytest.mark.parametrize(
    ("message", "signal"),
    [
        ("I like Cowboy Bebop.", LIKED),
        ("I really loved Cowboy Bebop", LIKED),
        ("Cowboy Bebop is my favourite", LIKED),
        ("I'm a huge fan of Cowboy Bebop", LIKED),
        ("I've already watched Cowboy Bebop.", WATCHED),
        ("I’ve already watched Cowboy Bebop.", WATCHED),
        ("I watched the third.", WATCHED),
        ("I just finished Frieren", WATCHED),
        ("Don't recommend Cowboy Bebop.", EXCLUDED),
        ("Stop recommending Cowboy Bebop", EXCLUDED),
        ("Never show me Cowboy Bebop again", EXCLUDED),
        ("I don't want Death Note.", EXCLUDED),
        ("No more isekai", EXCLUDED),
        ("None of these work.", EXCLUDED),
        ("I do not like mystery.", DISLIKED),
        ("I disliked the second one.", DISLIKED),
        ("This was not for me", DISLIKED),
        ("I am not interested in this kind of anime", DISLIKED),
        ("Please reset my preferences", RESET),
    ],
)
def test_stated_preferences_are_recognised(message: str, signal: str) -> None:
    assert signal in preference_signals(message)


def test_context_fields_never_need_authorization() -> None:
    gate = PreferenceWriteGate()

    admitted = gate.admit({"last_recommendations": ["A"], "last_recommendation_intent": {}, "liked_titles": ["A"]})

    assert admitted == {"last_recommendations": ["A"], "last_recommendation_intent": {}}
    assert gate.written == []
    assert gate.denied == ["liked_titles"]


def test_an_unknown_field_is_denied_by_default() -> None:
    everything = PreferenceWriteAuthorization(frozenset({LIKED, WATCHED, DISLIKED, EXCLUDED, RESET}))

    assert everything.split({"replace": True, "liked_titles": ["A"]}) == ({"liked_titles": ["A"]}, {"replace": True})


# -------------------------------------------------------------- agent, directly


@pytest.fixture(scope="module")
def recorded_agent() -> tuple[AnimeAgent, list[dict[str, Any]]]:
    writes: list[dict[str, Any]] = []

    def update(_session_id: str | None, patch: dict[str, Any]) -> dict[str, Any]:
        writes.append(dict(patch))
        return {}

    agent = AnimeAgent(AnimeRecommender(safety_catalog()), update_session_preferences=update)
    return agent, writes


def test_the_write_tool_refuses_preferences_without_a_gate(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
) -> None:
    agent, writes = recorded_agent
    writes.clear()

    agent._tool_update_session({"session_id": "s", "liked_titles": ["Cowboy Bebop"], "reset": True})
    assert writes == []

    agent._tool_update_session(
        {"session_id": "s", "liked_titles": ["Cowboy Bebop"], "last_recommendations": ["Trigun"]}
    )
    assert writes == [{"last_recommendations": ["Trigun"]}]


def test_the_write_tool_applies_the_gate_it_is_given(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
) -> None:
    agent, writes = recorded_agent
    writes.clear()
    gate = PreferenceWriteGate(agent._preference_write_authorization("I like Cowboy Bebop."))

    agent._tool_update_session(
        {"session_id": "s", "liked_titles": ["Cowboy Bebop"], "excluded_titles": ["Trigun"]},
        gate,
    )

    assert writes == [{"liked_titles": ["Cowboy Bebop"]}]
    assert gate.describe()["written_fields"] == ["liked_titles"]
    assert gate.describe()["denied_fields"] == ["excluded_titles"]


@pytest.mark.parametrize(
    ("message", "excluded"),
    [
        ("Recommend sci-fi anime but not Trigun", ["Trigun"]),
        ("Something like Cowboy Bebop without Trigun", ["Trigun"]),
        ("Don't recommend Cowboy Bebop.", ["Cowboy Bebop"]),
        ("I don't want Death Note.", ["Death Note"]),
        ("Introduce Cowboy Bebop without spoilers.", []),
        # The negation belongs to another clause.
        ("No horror, something like Cowboy Bebop", []),
        # "no" is not a word here.
        ("A piano drama like Cowboy Bebop", []),
        ("Is Cowboy Bebop a casino heist show like Trigun?", []),
    ],
)
def test_title_exclusion_needs_a_negation_in_the_same_clause(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
    message: str,
    excluded: list[str],
) -> None:
    agent, _ = recorded_agent

    assert agent._explicitly_excluded_titles(message) == excluded


@pytest.mark.parametrize(
    "message",
    [
        "Introduce Cowboy Bebop without spoilers.",
        "Tell me about Cowboy Bebop, I hate spoilers.",
        "A piano drama like Cowboy Bebop without too much talking",
        "Whatever you think is unavoidable",
    ],
)
def test_without_is_not_negative_feedback_about_a_title(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
    message: str,
) -> None:
    agent, _ = recorded_agent

    authorization = agent._preference_write_authorization(message)

    assert not authorization.permits("excluded_titles")
    assert not authorization.permits("disliked_titles")


def test_the_catalog_title_scan_runs_once_per_text(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
) -> None:
    """The write gate asks the same question the intent rules already asked; it must not pay for it again."""
    agent, _ = recorded_agent
    searches: list[str] = []
    original = agent.recommender.search

    def counting_search(query: str, *args: Any, **kwargs: Any) -> Any:
        searches.append(query)
        return original(query, *args, **kwargs)

    agent.recommender.search = counting_search  # type: ignore[method-assign]
    try:
        message = "A sentence nobody has asked before about Cowboy Bebop and Trigun."
        first = agent._mentioned_catalog_titles(message)
        first.append("a caller extending its own copy")
        agent._preference_write_authorization(message)
        second = agent._mentioned_catalog_titles(message)
    finally:
        agent.recommender.search = original  # type: ignore[method-assign]

    assert searches == [message]
    assert set(second) == {"Cowboy Bebop", "Trigun"}


def test_rule_fallback_no_longer_excludes_the_title_it_was_asked_to_introduce(
    recorded_agent: tuple[AnimeAgent, list[dict[str, Any]]],
) -> None:
    """The stored mutation the rule path used to make: excluded_titles == ["Cowboy Bebop"]."""
    agent, writes = recorded_agent
    writes.clear()

    intent = agent._rule_based_intent("Introduce Cowboy Bebop without spoilers.", [])
    response = agent.respond("Introduce Cowboy Bebop without spoilers.", session_id="s")

    assert intent.intent == "details"
    assert intent.excluded_titles == []
    assert response["mode"] == "catalog_introduction"
    assert response["trace"][0]["tool"] == "get_anime_details"
    assert writes == []


# ------------------------------------------------- the parser's own vocabulary

# Verbatim from a live Gemini response to "Recommend anime similar to Cowboy
# Bebop." The generation schema does not constrain `relation`, and this value
# is not one the backend defines.
LIVE_SIMILAR_TO_INTENT = {
    "intent": "recommend",
    "reference_titles": ["Cowboy Bebop"],
    "entity_mentions": [{"text": "Cowboy Bebop", "entity_type": "anime", "relation": "similar_to"}],
}


@pytest.mark.asyncio
async def test_an_unlisted_relation_word_no_longer_rejects_the_whole_intent() -> None:
    """It used to raise ProviderUnavailable, which also counts toward the circuit breaker."""
    body = {"candidates": [{"content": {"parts": [{"text": json.dumps(LIVE_SIMILAR_TO_INTENT)}]}}]}
    provider = GeminiAgentProvider(
        api_key="test-key",
        model="test-model",
        base_url="https://example.invalid/v1beta",
        timeout=2,
        max_output_tokens=500,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )

    intent = await provider.parse_intent(
        "Recommend anime similar to Cowboy Bebop.",
        ProviderParseContext(genres=[], formats=[]),
    )
    await provider.close()

    assert intent.intent == "recommend"
    assert intent.reference_titles == ["Cowboy Bebop"]
    assert [(mention.entity_type, mention.relation) for mention in intent.entity_mentions] == [("anime", "reference")]


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ({"text": "X", "entity_type": "anime", "relation": "similar"}, ("anime", "reference")),
        ({"text": "X", "entity_type": "anime", "relation": "Similar To"}, ("anime", "reference")),
        ({"text": "X", "entity_type": "anime", "relation": "some new wording"}, ("anime", "reference")),
        ({"text": "X", "entity_type": "anime", "relation": "excluded"}, ("anime", "exclude")),
        ({"text": "X", "entity_type": "anime", "relation": "director_of"}, ("anime", "director_of")),
        # An unknown relation on a studio is what omitting it means. Whether
        # that is a hard constraint is still read from the user's message.
        ({"text": "Madhouse", "entity_type": "studio", "relation": "produced_by"}, ("studio", "direct")),
        ({"text": "X"}, ("anime", "direct")),
    ],
)
def test_the_relation_label_is_read_tolerantly(mention: dict[str, Any], expected: tuple[str, str]) -> None:
    intent = AgentIntent.model_validate({"intent": "recommend", "entity_mentions": [mention]})

    assert [(item.entity_type, item.relation) for item in intent.entity_mentions] == [expected]


def test_everything_else_about_an_intent_is_still_strict() -> None:
    """Tolerance is for a label with a safe reading. These have none."""
    with pytest.raises(ValueError):
        AgentIntent.model_validate(
            {"intent": "recommend", "entity_mentions": [{"text": "Bebop", "entity_type": "spaceship"}]}
        )
    with pytest.raises(ValueError):
        AgentIntent.model_validate(
            {"intent": "recommend", "entity_mentions": [{"text": "X", "relation": "similar_to", "weight": 3}]}
        )


def test_an_invalid_intent_name_is_still_rejected() -> None:
    with pytest.raises(ValueError):
        AgentIntent.model_validate({"intent": "delete_everything"})
