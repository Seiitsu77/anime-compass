"""End-to-end checks that the fast path never bypasses production controls."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import RecommendRequest
from app.repositories.session_repository import SQLiteSessionRepository
from app.services.recommendation_service import RecommendationService
from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.fast_path import FastPathConfig
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.routing import RoutingPolicy


class _RecordCollector(logging.Handler):
    """Collect log records directly.

    configure_logging() clears root handlers, which removes pytest's caplog
    handler, so these tests attach their own instead of relying on caplog.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def capture_logs():
    collector = _RecordCollector()
    root = logging.getLogger()
    root.addHandler(collector)
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:
        yield collector
    finally:
        root.removeHandler(collector)
        root.setLevel(previous)


def degradation_events(collector: _RecordCollector) -> list[logging.LogRecord]:
    return [r for r in collector.records if "als_artifact_degradation" in r.getMessage()]


def build_artifact(
    path: Path,
    anime_ids: list[int],
    dimensions: int = 4,
    role: str = "production",
) -> Path:
    """A hand-built ALS artifact: item i is closest to items with the same parity."""
    import json

    generator = np.random.default_rng(7)
    factors = generator.normal(0, 0.01, (len(anime_ids), dimensions)).astype(np.float32)
    for index, anime_id in enumerate(anime_ids):
        factors[index, anime_id % 2] = 1.0
        # Later IDs score slightly higher, giving a deterministic ordering.
        factors[index, 2] = index * 0.001
    metadata = {
        "artifact_version": 1,
        "method": "implicit-feedback ALS (conjugate gradient)",
        "factors": dimensions,
        "alpha": 5.0,
        "regularization": 0.05,
        "iterations": 15,
        "artifact_role": role,
        "split_sha256": "test-split",
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
def service(tmp_path: Path, catalog: list[dict[str, Any]]) -> RecommendationService:
    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids)
    index = ALSCollaborativeIndex.load(artifact, catalog)
    recommender = AnimeRecommender(catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}", retention_days=30)
    return RecommendationService(
        recommender,
        sessions,
        als_index=index,
        fast_path_config=FastPathConfig(routing=RoutingPolicy(medium_threshold=1)),
    )


@pytest.mark.asyncio
async def test_unconstrained_request_takes_the_fast_path(service, catalog):
    request = RecommendRequest(liked_ids=[1, 2], top_k=3)
    response = await service.recommend(request)
    assert response["diagnostics"]["recommendation_path"] == "fast"
    assert response["diagnostics"]["collaborative_route"] == "als"
    assert response["results"]


@pytest.mark.asyncio
async def test_entity_constrained_request_takes_the_rich_path(service):
    """A required voice actor must be satisfied exactly, not approximately."""
    request = RecommendRequest(
        liked_ids=[1, 2],
        required_voice_actors=["Matsuoka, Yoshitsugu"],
        top_k=3,
    )
    response = await service.recommend(request)
    assert response["diagnostics"].get("recommendation_path") != "fast"
    for item in response["results"]:
        names = {str(actor.get("name")) for actor in item.get("voice_actors", [])}
        assert "Matsuoka, Yoshitsugu" in names


@pytest.mark.asyncio
async def test_metadata_constrained_request_takes_the_rich_path(service):
    request = RecommendRequest(liked_ids=[1, 2], include_genres=["Supernatural"], top_k=3)
    response = await service.recommend(request)
    assert response["diagnostics"].get("recommendation_path") != "fast"


@pytest.mark.asyncio
async def test_fast_path_honours_explicit_exclusions(service):
    request = RecommendRequest(liked_ids=[1], excluded_ids=[2, 3], top_k=5)
    response = await service.recommend(request)
    assert response["diagnostics"]["recommendation_path"] == "fast"
    returned = {int(item["id"]) for item in response["results"]}
    assert not ({2, 3} & returned)


@pytest.mark.asyncio
async def test_fast_path_never_recommends_the_profile_back(service):
    request = RecommendRequest(liked_ids=[1, 2, 3], top_k=5)
    response = await service.recommend(request)
    returned = {int(item["id"]) for item in response["results"]}
    assert not ({1, 2, 3} & returned)


@pytest.mark.asyncio
async def test_fast_path_result_carries_candidate_provenance(service):
    request = RecommendRequest(liked_ids=[1, 2], top_k=3)
    response = await service.recommend(request)
    assert all("candidate_sources" in item for item in response["results"])
    assert all("als" in item["candidate_sources"] for item in response["results"])


@pytest.mark.asyncio
async def test_fast_path_result_carries_artifact_versions(service):
    request = RecommendRequest(liked_ids=[1, 2], top_k=3)
    response = await service.recommend(request)
    versions = response["diagnostics"]["artifact_versions"]
    for key in (
        "als_artifact_sha256",
        "als_split_sha256",
        "routing_config_version",
        "retrieval_config_version",
        "ranking_config_version",
    ):
        assert versions.get(key), f"missing version field: {key}"


@pytest.mark.asyncio
async def test_fast_path_records_stage_latency(service):
    request = RecommendRequest(liked_ids=[1, 2], top_k=3)
    response = await service.recommend(request)
    stages = response["diagnostics"]["stage_latency_ms"]
    assert set(stages) == {"routing", "retrieval", "filtering", "ranking", "learned_reranking", "reranking"}
    assert all(value >= 0 for value in stages.values())


@pytest.mark.asyncio
async def test_service_without_als_falls_back_to_the_hybrid(tmp_path, catalog):
    """No ALS artifact must degrade to previous behaviour, not fail."""
    recommender = AnimeRecommender(catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / 's.db').as_posix()}", retention_days=30)
    service = RecommendationService(recommender, sessions, als_index=None)
    response = await service.recommend(RecommendRequest(liked_ids=[1], top_k=3))
    assert response["diagnostics"].get("recommendation_path") != "fast"
    assert "results" in response


@pytest.mark.asyncio
async def test_sparse_user_routes_away_from_als(tmp_path, catalog):
    ids = sorted(int(item["id"]) for item in catalog)
    index = ALSCollaborativeIndex.load(build_artifact(tmp_path / "als.npz", ids), catalog)
    recommender = AnimeRecommender(catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / 's.db').as_posix()}", retention_days=30)
    service = RecommendationService(
        recommender,
        sessions,
        als_index=index,
        fallback_index=None,
        fast_path_config=FastPathConfig(routing=RoutingPolicy(segment_aware=True)),
    )
    response = await service.recommend(RecommendRequest(liked_ids=[1, 2], top_k=3))
    # With no fallback source configured the fast path yields nothing and the
    # request falls through to the hybrid, which is the intended degradation.
    assert response["diagnostics"].get("collaborative_route") in (None, "sparse_fallback")


@pytest.mark.asyncio
async def test_semantic_stays_retired_on_the_fast_path(service):
    """The retired channel must not reappear through the new path."""
    from backend.anime_agent.recommender import DEFAULT_CHANNEL_WEIGHTS

    assert DEFAULT_CHANNEL_WEIGHTS["semantic_embedding"] == 0.0
    response = await service.recommend(RecommendRequest(liked_ids=[1, 2], top_k=3))
    assert response["model_info"]["weights"]["semantic_embedding"] == 0.0


def test_catalog_mismatch_refuses_to_start(tmp_path, catalog):
    """The catalog moving out from under the model is not a quiet fallback.

    Degrading silently would keep serving recommendations from a model that no
    longer describes the catalog, so this escalates rather than falling back.
    """
    from app.core.config import Settings
    from app.main import _load_als_index
    from backend.anime_agent.als_serving import ALSCatalogMismatchError

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids)
    settings = Settings(_env_file=None, als_artifact_path=artifact)
    unrelated = [{"id": anime_id} for anime_id in range(9000, 9020)]

    with capture_logs() as logs, pytest.raises(ALSCatalogMismatchError):
        _load_als_index(settings, unrelated)
    events = degradation_events(logs)
    assert events, "a catalog mismatch must emit an explicit degradation event"
    assert events[0].context["severity"] == "critical"
    assert events[0].context["action"] == "refusing_startup"


def test_role_mismatch_degrades_and_emits_a_high_severity_event(tmp_path, catalog):
    """An evaluation artifact must never quietly serve production traffic."""
    from app.core.config import Settings
    from app.main import _load_als_index

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids, role="evaluation")
    settings = Settings(_env_file=None, als_artifact_path=artifact)

    with capture_logs() as logs:
        assert _load_als_index(settings, catalog) is None
    events = degradation_events(logs)
    assert events
    assert events[0].context["severity"] == "high"
    assert events[0].context["action"] == "degraded_to_countsketch"
    assert events[0].context["error_type"] == "ALSArtifactRoleError"


def test_invalid_artifact_refuses_to_start_in_strict_mode(tmp_path, catalog):
    from app.core.config import Settings
    from app.main import _load_als_index
    from backend.anime_agent.als_serving import ALSArtifactError

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids, role="evaluation")
    settings = Settings(_env_file=None, als_artifact_path=artifact, als_require_valid_artifact=True)

    with pytest.raises(ALSArtifactError):
        _load_als_index(settings, catalog)


def test_pinned_catalog_digest_mismatch_refuses(tmp_path, catalog):
    """The overlap ratio is fuzzy; a pinned catalog digest is exact."""
    from app.core.config import Settings
    from app.main import _load_als_index
    from backend.anime_agent.als_serving import ALSCatalogMismatchError

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids)
    settings = Settings(
        _env_file=None,
        als_artifact_path=artifact,
        als_expected_catalog_ids_sha256="0" * 64,
    )
    with pytest.raises(ALSCatalogMismatchError, match="Catalog identity mismatch"):
        _load_als_index(settings, catalog)


def test_pinned_checksum_mismatch_always_refuses(tmp_path, catalog):
    """A pinned hash that does not match is an integrity failure, not a
    missing optional model."""
    from app.core.config import Settings
    from app.main import _load_als_index
    from backend.anime_agent.als_serving import ALSArtifactError

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids)
    settings = Settings(_env_file=None, als_artifact_path=artifact, als_expected_sha256="0" * 64)

    with pytest.raises(ALSArtifactError, match="checksum mismatch"):
        _load_als_index(settings, catalog)


def test_missing_artifact_is_a_warning_not_an_error(tmp_path, catalog):
    from app.core.config import Settings
    from app.main import _load_als_index

    settings = Settings(_env_file=None, als_artifact_path=tmp_path / "absent.npz")
    assert _load_als_index(settings, catalog) is None


def test_als_can_be_disabled_by_configuration(tmp_path, catalog):
    from app.core.config import Settings
    from app.main import _load_als_index

    ids = sorted(int(item["id"]) for item in catalog)
    artifact = build_artifact(tmp_path / "als.npz", ids)
    settings = Settings(_env_file=None, als_artifact_path=artifact, als_enabled=False)
    assert _load_als_index(settings, catalog) is None
