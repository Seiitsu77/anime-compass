"""The compact serving catalog must be a drop-in for the full one.

The deployment payload is dominated by catalog fields the demo never reads.
Dropping them is only safe if it is provably invisible, so these tests pin the
three properties serving depends on: the same recommendation IDs, the same
display and search behaviour, and the same catalog identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_showcase import make_artifact

from backend.anime_agent.als_serving import catalog_ids_digest
from backend.anime_agent.showcase import (
    SYNOPSIS_DISPLAY_CHARS,
    load_showcase_service,
    truncate_synopsis,
)
from scripts.build_serving_catalog import (
    SERVING_FIELDS,
    build_serving_catalog,
    check_identity,
    compact_entry,
)

# Fields the constraint-rich Hybrid uses and the fast path does not. A real
# catalog row carries megabytes of these; the serving catalog must shed them.
HYBRID_ONLY: dict[str, Any] = {
    "characters": [{"name": "Someone", "role": "Main"}] * 40,
    "voice_actors": ["A Person"] * 30,
    "voice_actor_roles": [{"actor": "A Person", "character": "Someone"}] * 30,
    "staff": ["A Director"] * 12,
    "studios": ["A Studio"],
    "producers": ["A Producer"],
    "metadata_tokens": ["token"] * 50,
    "score_distribution": {str(n): n for n in range(1, 11)},
    "aliases": ["An Alias"],
}


@pytest.fixture
def full_catalog() -> list[dict[str, Any]]:
    """Shaped like the real catalog: display fields plus heavy entity data."""

    def entry(anime_id: int, title: str, genres: list[str], year: int) -> dict[str, Any]:
        return {
            "id": anime_id,
            "title": title,
            "genres": genres,
            "start_year": year,
            "type": "TV",
            "episodes": 12,
            "score": 8.0,
            "members": 10_000 - anime_id,
            "image_url": f"https://example.invalid/{anime_id}.jpg",
            "synopsis": f"A synopsis for {title}. " + ("Detail. " * 60),
            **HYBRID_ONLY,
        }

    return [
        entry(1, "Alpha", ["Sci-Fi"], 2010),
        entry(2, "Beta", ["Romance"], 2011),
        entry(3, "Gamma", ["Sci-Fi"], 2012),
        entry(4, "Delta", ["Romance"], 2013),
        entry(5, "Epsilon", ["Sci-Fi"], 2014),
        entry(6, "Zeta Movie", ["Romance"], 2015),
        entry(7, "Alpha Season 2", ["Sci-Fi"], 2016),
    ]


@pytest.fixture
def paired_services(tmp_path: Path, full_catalog):
    """The same model served from the full catalog and from the compact one."""
    ids = sorted(int(item["id"]) for item in full_catalog)
    artifact = make_artifact(tmp_path / "als.npz", ids)
    compact = build_serving_catalog(full_catalog)
    return (
        load_showcase_service(full_catalog, artifact),
        load_showcase_service(compact, artifact),
    )


# ------------------------------------------------------------------- shape


def test_only_the_fields_the_demo_reads_survive(full_catalog):
    compact = build_serving_catalog(full_catalog)
    written = {key for item in compact for key in item}
    assert written <= set(SERVING_FIELDS)
    assert not written & set(HYBRID_ONLY), "hybrid-only entity data must not ship to the demo"


def test_mal_ids_are_preserved_exactly_and_in_order(full_catalog):
    compact = build_serving_catalog(full_catalog)
    assert [item["id"] for item in compact] == [item["id"] for item in full_catalog]
    assert all(isinstance(item["id"], int) for item in compact)


def test_the_catalog_identity_digest_is_unchanged(full_catalog):
    """The pinned ALS catalog digest must still validate against the compact catalog."""
    compact = build_serving_catalog(full_catalog)
    full_ids = sorted({int(item["id"]) for item in full_catalog})
    compact_ids = sorted({int(item["id"]) for item in compact})
    assert catalog_ids_digest(full_ids) == catalog_ids_digest(compact_ids)


def test_empty_values_are_dropped_but_identity_fields_are_not(full_catalog):
    sparse = {**full_catalog[0], "genres": [], "score": None, "image_url": "", "episodes": None}
    compact = compact_entry(sparse)
    assert "genres" not in compact and "score" not in compact
    assert compact["id"] == sparse["id"] and compact["title"] == sparse["title"]


def test_the_builder_refuses_to_write_a_catalog_that_drifts(full_catalog):
    compact = build_serving_catalog(full_catalog)
    assert check_identity(full_catalog, compact) == []
    compact[2]["id"] = 999999
    assert check_identity(full_catalog, compact), "a changed ID must be caught, not written"


# -------------------------------------------------------------- truncation


def test_synopsis_truncation_is_idempotent():
    """This is what makes a pre-trimmed catalog equivalent to trimming at render."""
    long_text = "word " * 500
    once = truncate_synopsis(long_text)
    assert len(once) <= SYNOPSIS_DISPLAY_CHARS
    assert truncate_synopsis(once) == once


def test_short_synopses_are_left_alone():
    assert truncate_synopsis("Short.") == "Short."
    assert truncate_synopsis(None) == ""


# ------------------------------------------------------------- equivalence


@pytest.mark.parametrize(
    "liked",
    [[1], [1, 3], [1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6, 7]],
    ids=["one", "two", "sci-fi", "romance", "everything"],
)
@pytest.mark.parametrize("limit", [4, 12])
def test_recommendation_ids_are_identical(paired_services, liked, limit):
    full, compact = paired_services
    assert [item.anime_id for item in full.recommend(liked, limit=limit).items] == [
        item.anime_id for item in compact.recommend(liked, limit=limit).items
    ]


def test_display_cards_are_identical(paired_services):
    full, compact = paired_services

    def card(item):
        return (
            item.anime_id,
            item.title,
            item.year,
            item.media_type,
            item.episodes,
            item.score,
            item.genres,
            item.image_url,
            item.synopsis,
            item.explanation,
            item.because_of,
        )

    assert [card(i) for i in full.recommend([1, 3], limit=6).items] == [
        card(i) for i in compact.recommend([1, 3], limit=6).items
    ]


def test_exclusions_behave_identically(paired_services):
    full, compact = paired_services
    a = full.recommend([1], disliked_ids=[3, 5], limit=6)
    b = compact.recommend([1], disliked_ids=[3, 5], limit=6)
    assert [i.anime_id for i in a.items] == [i.anime_id for i in b.items]
    assert not ({3, 5} & {i.anime_id for i in b.items})


def test_series_collapsing_behaves_identically(paired_services):
    """Collapsing reads titles, so it must survive the projection."""
    full, compact = paired_services
    assert 7 not in {i.anime_id for i in compact.recommend([1], limit=5).items}
    assert [i.anime_id for i in full.recommend([1], limit=5).items] == [
        i.anime_id for i in compact.recommend([1], limit=5).items
    ]


@pytest.mark.parametrize("query", ["alpha", "ALPHA", "gamma", "movie", "zzzz", ""])
def test_search_results_are_identical(paired_services, query):
    full, compact = paired_services
    assert [i["id"] for i in full.search(query)] == [i["id"] for i in compact.search(query)]


def test_popularity_ordering_is_identical(paired_services):
    full, compact = paired_services
    assert [i["id"] for i in full.popular(7)] == [i["id"] for i in compact.popular(7)]


def test_cold_start_accounting_is_identical(paired_services):
    full, compact = paired_services
    assert full.health.catalog_items == compact.health.catalog_items
    assert full.health.als_covered_items == compact.health.als_covered_items
    assert full.health.cold_start_items == compact.health.cold_start_items


# ---------------------------------------------------- against the real data


def test_the_real_serving_catalog_matches_the_real_full_catalog():
    """Runs only where both catalogs exist; the checkout does not ship them."""
    full_path = Path("data/processed/anime_catalog.json")
    serving_path = Path("data/processed/anime_catalog_serving.json")
    if not (full_path.exists() and serving_path.exists()):
        pytest.skip("processed catalogs are not present in this checkout")

    full = json.loads(full_path.read_text(encoding="utf-8"))
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    assert check_identity(full, serving) == []
    assert serving_path.stat().st_size < full_path.stat().st_size / 10
