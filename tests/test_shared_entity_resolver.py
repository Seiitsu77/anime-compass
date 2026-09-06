"""One entity index per catalog, not one per component that wants it.

A memory profiling audit found `EntityResolver` constructed twice over the same
catalog -- once by the API container, once inside `AnimeAgent` -- for 197 MiB of
byte-identical indexes in a process resident at 1.98 GB. The two instances were
verified distinct objects built from the same list.

Sharing is safe because the resolver is read-only once built: every method after
`__init__` only reads. These tests pin that arrangement, and the equivalence
that made it a zero-behaviour change rather than a trade-off.
"""

from __future__ import annotations

import pytest

from backend.anime_agent.entities import EntityResolver
from backend.anime_agent.recommender import AnimeRecommender


@pytest.fixture
def catalog() -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "title": f"Title {index}",
            "genres": ["Action"] if index % 2 else ["Drama"],
            "studios": ["Madhouse"] if index % 3 else ["Bones"],
            "type": "TV",
            "episodes": 12,
            "start_year": 2000 + index,
            "score": 7.0 + (index % 3) / 10,
            "members": 10_000 - index,
            "synopsis": f"Synopsis for {index}.",
            "characters": [{"id": 900 + index, "name": f"Hero {index}", "role": "Main"}],
            "character_names": [f"Hero {index}"],
            "metadata_tokens": ["genre_Action"],
        }
        for index in range(1, 25)
    ]


def test_the_recommender_hands_out_one_resolver(catalog):
    recommender = AnimeRecommender(catalog)
    assert recommender.entity_resolver is recommender.entity_resolver


def test_the_agent_shares_the_recommenders_resolver(catalog):
    from backend.anime_agent.agent import AnimeAgent

    recommender = AnimeRecommender(catalog)
    agent = AnimeAgent(recommender)
    assert agent.entity_resolver is recommender.entity_resolver, (
        "the agent must not build a second index over the same catalog"
    )


def test_the_resolver_is_not_built_until_something_asks_for_it(catalog):
    """The fast recommendation path never resolves an entity."""
    recommender = AnimeRecommender(catalog)
    assert recommender._entity_resolver is None
    built = recommender.entity_resolver
    assert recommender._entity_resolver is built


def test_the_shared_resolver_answers_as_its_own_instance_would(catalog):
    """Sharing is only safe because the resolver is read-only after build."""
    recommender = AnimeRecommender(catalog)
    separate = EntityResolver(catalog)
    for query, entity_type in (("Madhouse", "studio"), ("Hero 3", "character"), ("Bones", None)):
        assert recommender.entity_resolver.resolve(query, entity_type) == separate.resolve(query, entity_type)
        assert recommender.entity_resolver.search(query, entity_type) == separate.search(query, entity_type)
