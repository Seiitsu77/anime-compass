"""Showcase service and artifact bootstrap.

Streamlit internals are not tested. Everything the demo depends on lives in
`ShowcaseService` and `artifact_bootstrap`, so that is what is pinned here.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.artifact_bootstrap import ensure_production_artifact
from backend.anime_agent.showcase import (
    ShowcaseService,
    cold_start_ids,
    load_showcase_service,
)


def make_artifact(path: Path, anime_ids: list[int], *, role: str = "production", dimensions: int = 6) -> Path:
    """Artifact where items cluster by id parity, so results are predictable."""
    generator = np.random.default_rng(3)
    factors = generator.normal(0, 0.01, (len(anime_ids), dimensions)).astype(np.float32)
    for index, anime_id in enumerate(anime_ids):
        factors[index, anime_id % 2] = 1.0
        factors[index, 2] = index * 0.001
    metadata = {
        "artifact_version": 1,
        "artifact_role": role,
        "factors": dimensions,
        "alpha": 5.0,
        "regularization": 0.05,
        "iterations": 15,
        "ratings_used": 1234,
        "training_source": "unit-test fixture",
    }
    np.savez_compressed(
        path,
        anime_ids=np.asarray(anime_ids, dtype=np.int64),
        item_factors=factors,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return path


@pytest.fixture
def demo_catalog() -> list[dict[str, Any]]:
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
            "synopsis": "A synopsis.",
            "image_url": "",
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
def service(tmp_path: Path, demo_catalog) -> ShowcaseService:
    ids = sorted(int(item["id"]) for item in demo_catalog)
    return load_showcase_service(demo_catalog, make_artifact(tmp_path / "als.npz", ids))


# ------------------------------------------------------------------ health


def test_service_reports_production_als(service):
    health = service.health
    assert health.serving_production_als is True
    assert health.headline == "Production ALS"
    assert health.catalog_items == 7
    assert health.als_covered_items == 7
    assert health.cold_start_items == 0
    assert health.error is None


def test_missing_artifact_never_claims_production_als(tmp_path, demo_catalog):
    """The demo must not advertise the ALS benchmark while serving nothing."""
    service = load_showcase_service(demo_catalog, tmp_path / "absent.npz")
    assert service.health.serving_production_als is False
    assert service.health.headline == "Unavailable"
    assert service.health.error


def test_evaluation_artifact_is_refused_for_the_demo(tmp_path, demo_catalog):
    ids = sorted(int(item["id"]) for item in demo_catalog)
    artifact = make_artifact(tmp_path / "eval.npz", ids, role="evaluation")
    service = load_showcase_service(demo_catalog, artifact)
    assert service.health.serving_production_als is False
    assert "role mismatch" in (service.health.error or "")


def test_recommend_raises_when_the_model_is_unavailable(tmp_path, demo_catalog):
    from backend.anime_agent.als_serving import ALSArtifactError

    service = load_showcase_service(demo_catalog, tmp_path / "absent.npz")
    with pytest.raises(ALSArtifactError):
        service.recommend([1])


# ------------------------------------------------------------------ search


def test_search_prefers_prefix_matches_by_popularity(service):
    results = service.search("alpha")
    assert [item["title"] for item in results][:2] == ["Alpha", "Alpha Season 2"]


def test_search_is_case_and_accent_insensitive(service):
    assert service.search("ALPHA")
    assert service.search("gamma")[0]["title"] == "Gamma"


def test_empty_search_returns_nothing(service):
    assert service.search("") == []
    assert service.search("   ") == []


def test_unmatched_search_returns_nothing(service):
    assert service.search("zzzzz") == []


# ---------------------------------------------------------- recommendations


def test_recommendations_exclude_the_profile(service):
    result = service.recommend([1, 3], limit=5)
    assert result.items
    assert not ({1, 3} & {item.anime_id for item in result.items})


def test_recommendations_exclude_dislikes(service):
    result = service.recommend([1], disliked_ids=[3, 5], limit=5)
    assert not ({3, 5} & {item.anime_id for item in result.items})


def test_recommendations_use_the_fast_path(service):
    result = service.recommend([1, 3], limit=4)
    assert result.path == "fast"
    assert result.diagnostics["collaborative_route"] == "als"
    assert result.candidate_pool_size > 0


def test_empty_profile_returns_no_recommendations(service):
    result = service.recommend([], limit=5)
    assert result.items == []
    assert result.diagnostics["reason"] == "no_liked_titles"


def test_unknown_ids_are_ignored(service):
    result = service.recommend([1, 999999], limit=3)
    assert result.items


def test_same_series_entries_are_collapsed(service):
    """Recommending 'Alpha Season 2' to someone who liked 'Alpha' reads badly."""
    result = service.recommend([1], limit=5)
    assert 7 not in {item.anime_id for item in result.items}


def test_series_collapsing_can_be_disabled(service):
    collapsed = service.recommend([1], limit=6, one_per_series=True)
    raw = service.recommend([1], limit=6, one_per_series=False)
    assert len(raw.items) >= len(collapsed.items)


# ------------------------------------------------------------ explanations


def test_explanations_name_real_profile_titles(service):
    """Explanations must be grounded, never generated prose."""
    result = service.recommend([1, 3], limit=4)
    liked_titles = {"Alpha", "Gamma"}
    for item in result.items:
        assert item.explanation
        for named in item.because_of:
            assert named in liked_titles


def test_explanations_reference_genuinely_shared_genres(service):
    result = service.recommend([1], limit=4)
    for item in result.items:
        if "Shares" in item.explanation:
            claimed = item.explanation.split("Shares", 1)[1].strip(" .").split(", ")
            liked_genres = set(service.by_id[1]["genres"])
            assert set(claimed) <= liked_genres


def test_recommendation_cards_carry_display_fields(service):
    item = service.recommend([1, 3], limit=1).items[0]
    assert item.title and item.year and item.media_type
    assert isinstance(item.genres, tuple)
    assert item.synopsis


# ------------------------------------------------------------ example profiles


def test_example_profiles_only_offer_titles_in_this_catalog(service):
    # The real profile IDs are absent from this fixture catalog.
    assert service.example_profiles() == {}


def test_example_profiles_are_usable_against_the_real_catalog():
    catalog_path = Path("data/processed/anime_catalog.json")
    if not catalog_path.exists():
        pytest.skip("processed catalog is not present in this checkout")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ids = {int(item["id"]) for item in catalog}
    from backend.anime_agent.showcase import EXAMPLE_PROFILES

    for name, (_description, profile_ids) in EXAMPLE_PROFILES.items():
        present = [anime_id for anime_id in profile_ids if anime_id in ids]
        assert present, f"example profile {name!r} has no titles in the catalog"


# ------------------------------------------------------------- cold start


def test_cold_start_items_are_identified(service, demo_catalog):
    extended = [*demo_catalog, {"id": 9001, "title": "Brand New", "genres": [], "start_year": 2026}]
    assert 9001 in cold_start_ids(extended, service.als_index)
    assert 1 not in cold_start_ids(extended, service.als_index)


def test_cold_start_items_never_enter_recommendations(tmp_path, demo_catalog):
    """A catalog-only title has no factor and must not be given a score."""
    ids = sorted(int(item["id"]) for item in demo_catalog)
    artifact = make_artifact(tmp_path / "als.npz", ids)
    extended = [*demo_catalog, {"id": 9001, "title": "Brand New", "genres": [], "start_year": 2026}]
    service = load_showcase_service(extended, artifact)

    assert service.is_cold_start(9001) is True
    result = service.recommend([1, 3], limit=6)
    assert 9001 not in {item.anime_id for item in result.items}


# --------------------------------------------------------------- bootstrap


def test_present_artifact_with_matching_checksum_is_accepted(tmp_path):
    path = tmp_path / "artifact.npz"
    path.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    result = ensure_production_artifact(path, expected_sha256=digest)
    assert result.usable and not result.downloaded


def test_present_artifact_with_wrong_checksum_is_rejected(tmp_path):
    path = tmp_path / "artifact.npz"
    path.write_bytes(b"payload")
    result = ensure_production_artifact(path, expected_sha256="0" * 64)
    assert not result.usable
    assert "checksum mismatch" in result.detail


def test_missing_artifact_without_a_url_reports_clearly(tmp_path):
    result = ensure_production_artifact(tmp_path / "absent.npz")
    assert not result.usable
    assert "no ALS_ARTIFACT_URL" in result.detail


def test_download_failure_is_reported_not_raised(tmp_path, monkeypatch):
    import backend.anime_agent.artifact_bootstrap as bootstrap

    def fail(url, destination):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(bootstrap, "_download", fail)
    result = bootstrap.ensure_production_artifact(tmp_path / "absent.npz", url="https://example.invalid/a.npz")
    assert not result.usable
    assert "download failed" in result.detail


def test_downloaded_artifact_failing_verification_is_deleted(tmp_path, monkeypatch):
    import backend.anime_agent.artifact_bootstrap as bootstrap

    target = tmp_path / "artifact.npz"

    def write_wrong(url, destination):
        destination.write_bytes(b"not what was pinned")

    monkeypatch.setattr(bootstrap, "_download", write_wrong)
    result = bootstrap.ensure_production_artifact(target, url="https://example.invalid/a.npz", expected_sha256="0" * 64)
    assert not result.usable
    assert not target.exists(), "an unverified download must not be left in place"


def test_non_https_urls_are_refused(tmp_path):
    with pytest.raises(ValueError, match="https"):
        ensure_production_artifact(tmp_path / "absent.npz", url="http://example.invalid/a.npz")


def test_bootstrap_reads_the_environment(tmp_path, monkeypatch):
    import backend.anime_agent.artifact_bootstrap as bootstrap

    path = tmp_path / "artifact.npz"
    path.write_bytes(b"payload")
    monkeypatch.setenv("ALS_ARTIFACT_PATH", str(path))
    monkeypatch.delenv("ALS_ARTIFACT_URL", raising=False)
    monkeypatch.setenv("ALS_EXPECTED_SHA256", hashlib.sha256(b"payload").hexdigest())
    result = bootstrap.bootstrap_from_environment(tmp_path / "unused.npz")
    assert result.usable and result.path == path


# ----------------------------------------------------------- no LLM needed


def test_recommendations_need_no_llm_credentials(service, monkeypatch):
    """The core demo must work with no provider configured."""
    for name in ("GEMINI_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL"):
        monkeypatch.delenv(name, raising=False)
    result = service.recommend([1, 3], limit=3)
    assert result.items


def test_free_text_reports_the_path_it_would_take(service):
    plain = service.recommend([1, 3], limit=2)
    assert plain.diagnostics["would_route_to"] == "fast"

    constrained = service.recommend([1, 3], limit=2, free_text="something dark and psychological")
    assert constrained.diagnostics["would_route_to"] == "constraint_rich"
    # The fast path still served it, since the demo does not call an LLM.
    assert constrained.path == "fast"
