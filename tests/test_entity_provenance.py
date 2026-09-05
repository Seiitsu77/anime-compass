"""Validated structured state is the source of truth for entities.

The defect these cover: `AgentIntent` validation removed the character "With"
(a stopword, and a junk record in the catalog), and `_named_catalog_entities`
then matched that record against the preposition in "something with dark
psychological mind games" and put it straight back. The required-character
filter that followed emptied the result set.

The fix is not a rule about that word. It is that the boundary where raw text
becomes a structured entity applies the same validity rule the validated intent
does, so recovery can still add an entity the parser missed but can never
reintroduce one validation rejected.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.schemas import AgentIntent
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.intent import StructuredIntent
from backend.anime_agent.recommender import AnimeRecommender, normalize_label


def character(anime_id: int, title: str, character_name: str, studio: str = "Madhouse") -> dict[str, Any]:
    return {
        "id": anime_id,
        "title": title,
        "genres": ["Psychological"],
        "members": 10_000 - anime_id,
        "score": 8.0,
        "type": "TV",
        "episodes": 12,
        "start_year": 2015,
        "synopsis": f"{title} is a story.",
        "studios": [studio],
        "characters": [{"id": 5000 + anime_id, "name": character_name, "role": "Main"}],
        "character_names": [character_name],
        "character_relationships": [{"id": 5000 + anime_id, "name": character_name, "role": "Main"}],
        "metadata_tokens": ["genre_Psychological"],
    }


@pytest.fixture
def catalog_with_junk_character() -> list[dict[str, Any]]:
    """The real catalog contains a character record literally named "With"."""
    return [
        character(1, "Alpha", "With"),
        character(2, "Beta", "Light Yagami"),
        character(3, "Gamma", "Lelouch Lamperouge"),
    ]


@pytest.fixture
def agent(catalog_with_junk_character) -> AnimeAgent:
    return AnimeAgent(AnimeRecommender(catalog_with_junk_character))


MESSAGE = "I want something with dark psychological mind games and unreliable narrators."


# ------------------------------------------------ 1. invalid entity removal


def test_validation_removes_an_entity_that_cannot_name_anything():
    intent = AgentIntent.model_validate(
        {
            "intent": "recommend",
            "entity_mentions": [{"text": "With", "entity_type": "character", "relation": "direct"}],
            "required_characters": ["With"],
        }
    )
    assert intent.entity_mentions == []
    assert intent.required_characters == []


def test_raw_text_recovery_cannot_reintroduce_it(agent):
    """The boundary applies the same rule, so the junk record is unreachable."""
    assert agent._named_catalog_entities(MESSAGE, "character") == []


def test_the_entity_resolver_is_never_asked_about_the_invalid_entity(agent, monkeypatch):
    """The original symptom was a lookup for "With" in the tool trace."""
    queries: list[str] = []
    original = agent.entity_resolver.resolve

    def spy(query, entity_type=None, *args, **kwargs):
        queries.append(str(query))
        return original(query, entity_type, *args, **kwargs)

    monkeypatch.setattr(agent.entity_resolver, "resolve", spy)

    intent = AgentIntent.model_validate(
        {
            "intent": "recommend",
            "entity_mentions": [{"text": "With", "entity_type": "character", "relation": "direct"}],
            "required_characters": ["With"],
            "include_genres": ["Psychological"],
        }
    ).to_legacy()
    agent._enforce_explicit_entity_constraints(intent, MESSAGE)

    assert not any(query.casefold() == "with" for query in queries), (
        f"the resolver was asked about the removed entity: {queries}"
    )
    assert intent.required_characters == []


def test_a_real_character_is_still_recovered_from_the_message(agent):
    """The guard must not disable legitimate recovery."""
    assert agent._named_catalog_entities("anime with Light Yagami in it", "character") == ["Light Yagami"]


def test_the_rule_is_general_not_a_word_list():
    """Any stopword-only name fails, because it normalises to nothing."""
    for value in ("With", "The", "A", "And", "For"):
        assert normalize_label(value) == ""
    for value in ("Madhouse", "Light Yagami", "Sato, Ken"):
        assert normalize_label(value)


# --------------------------------------- 2. relaxed constraints stay relaxed


def test_relaxed_numeric_constraints_are_not_re_derived_from_raw_text(agent):
    """Replanning drops these; the raw message must not put them back."""
    relaxed = StructuredIntent(intent="recommend", min_score=None, max_episodes=None, min_year=None)
    agent._enforce_explicit_entity_constraints(
        relaxed,
        "A 2015 Madhouse isekai with at most 3 episodes rated above 9.5",
    )
    assert relaxed.min_score is None
    assert relaxed.max_episodes is None
    assert relaxed.min_year is None


def test_relaxed_genre_constraints_are_not_re_derived_from_raw_text(agent):
    relaxed = StructuredIntent(intent="recommend", include_genres=[], exclude_genres=[])
    agent._enforce_explicit_entity_constraints(relaxed, "I want a psychological anime, no romance")
    assert relaxed.include_genres == []
    assert relaxed.exclude_genres == []


# ------------------------------------ 3. required constraints stay invariant


def test_a_valid_required_entity_survives_enforcement(agent):
    intent = AgentIntent(intent="recommend", required_studios=["Madhouse"]).to_legacy()
    agent._enforce_explicit_entity_constraints(intent, "Recommend anime from the studio Madhouse.")
    assert intent.required_studios == ["Madhouse"]


def test_a_valid_required_character_survives_enforcement(agent):
    intent = AgentIntent(intent="recommend", required_characters=["Light Yagami"]).to_legacy()
    agent._enforce_explicit_entity_constraints(intent, "anime with Light Yagami")
    assert intent.required_characters == ["Light Yagami"]


# ------------------------------------------------- 4. exclusions stay excluded


def test_excluded_titles_are_not_re_derived_into_references(agent):
    """An exclusion must not come back as something to recommend."""
    intent = AgentIntent(
        intent="recommend",
        excluded_titles=["Alpha"],
        reference_titles=["Beta"],
    ).to_legacy()
    agent._enforce_explicit_entity_constraints(intent, "Something like Beta but not Alpha")
    assert "Alpha" in intent.excluded_titles
    assert "Alpha" not in intent.reference_titles
