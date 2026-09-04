"""Serving the learned reranker inside the fast path.

Two properties carry the integration. The reranker must reorder and do nothing
else -- no item appears, disappears, or escapes an exclusion because of it. And
every way it can fail must land on the ALS order, because the fast path worked
before the reranker existed and a missing improvement is not an outage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.anime_agent.fast_path import FastPathConfig, recommend_fast
from backend.anime_agent.reranker_serving import (
    EXPECTED_FEATURE_COUNT,
    RerankerUnavailable,
    load_reranker,
    try_load_reranker,
)

FEATURE_ARTIFACT = Path("data/processed/reranker_features.npz")
MODEL_ARTIFACT = Path("data/processed/reranker_lambdamart.txt")


class StubSource:
    """An ALS stand-in: fixed order, raw scores available."""

    def __init__(self, order: list[int]):
        self.order = order
        self.scores = {anime_id: float(len(order) - index) for index, anime_id in enumerate(order)}

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        blocked = set(excluded_ids)
        return [anime_id for anime_id in self.order if anime_id not in blocked][:limit]

    def raw_profile_scores(self, positive_ids):
        return np.asarray([self.scores.get(index, 0.0) for index in range(max(self.order) + 1)], dtype=np.float32)

    def quality_score(self, anime_id):
        return None


class ReverseReranker:
    """Deterministic stand-in that always reverses the candidate order."""

    def __init__(self, anime_ids: list[int]):
        self.index_by_id = {anime_id: index for index, anime_id in enumerate(anime_ids)}
        self.feature_space = object()

    def rerank(self, profile_rows, candidate_rows, als_scores):
        return list(reversed(list(candidate_rows)))


class BrokenReranker(ReverseReranker):
    def rerank(self, profile_rows, candidate_rows, als_scores):
        raise RuntimeError("model exploded")


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    return [{"id": anime_id, "title": f"T{anime_id}", "genres": ["Action"]} for anime_id in range(1, 9)]


@pytest.fixture
def catalog_by_id(catalog):
    return {int(item["id"]): item for item in catalog}


def run(catalog_by_id, *, reranker=None, excluded=(), limit=5):
    order = [1, 2, 3, 4, 5, 6, 7, 8]
    return recommend_fast(
        [1],
        catalog_by_id=catalog_by_id,
        als_source=StubSource(order),
        fallback_source=None,
        profile_rows=[1],
        excluded_ids=excluded,
        limit=limit,
        config=FastPathConfig(reranker=reranker),
    )


# ------------------------------------------------------ ALS-only fallback


def test_no_reranker_leaves_the_als_order_untouched(catalog_by_id):
    """The default path must be exactly what it was before reranking existed."""
    result = run(catalog_by_id)
    assert result.diagnostics["learned_reranker_applied"] is False
    assert result.anime_ids == [1, 2, 3, 4, 5]


def test_a_failing_reranker_degrades_to_the_als_order(catalog_by_id):
    baseline = run(catalog_by_id).anime_ids
    result = run(catalog_by_id, reranker=BrokenReranker(list(range(1, 9))))
    assert result.anime_ids == baseline
    assert result.diagnostics["learned_reranker_applied"] is False


def test_a_reranker_that_does_not_know_the_catalog_is_ignored(catalog_by_id):
    """An unknown candidate would receive an arbitrary score, so decline entirely."""
    partial = ReverseReranker([1, 2, 3])
    result = run(catalog_by_id, reranker=partial)
    assert result.diagnostics["learned_reranker_applied"] is False


def test_an_empty_profile_skips_reranking(catalog_by_id):
    result = recommend_fast(
        [],
        catalog_by_id=catalog_by_id,
        als_source=StubSource([1, 2, 3, 4, 5]),
        fallback_source=None,
        profile_rows=[],
        limit=3,
        config=FastPathConfig(reranker=ReverseReranker(list(range(1, 9)))),
    )
    assert result.diagnostics["learned_reranker_applied"] is False


# ---------------------------------------------------- reordering contract


def test_the_reranker_reorders_and_changes_nothing_else(catalog_by_id):
    """Reranking may change which candidates reach the top-N, never which exist.

    Changing the membership of the top-N is the point -- that is how a reranker
    improves NDCG. What it must never do is surface an item the retriever did
    not supply.
    """
    plain = run(catalog_by_id, limit=8)
    pool = set(plain.anime_ids)
    reranked = run(catalog_by_id, reranker=ReverseReranker(list(range(1, 9))), limit=7)
    assert reranked.diagnostics["learned_reranker_applied"] is True
    assert set(reranked.anime_ids) <= pool, "a reranker invented a candidate"
    assert len(set(reranked.anime_ids)) == len(reranked.anime_ids), "duplicated a candidate"
    assert reranked.anime_ids != run(catalog_by_id, limit=7).anime_ids


def test_exclusions_survive_reranking(catalog_by_id):
    """Reordering must not float an excluded item back into the results."""
    result = run(catalog_by_id, reranker=ReverseReranker(list(range(1, 9))), excluded=(3, 4), limit=6)
    assert not ({3, 4} & set(result.anime_ids))


def test_the_profile_is_never_recommended_back(catalog_by_id):
    result = run(catalog_by_id, reranker=ReverseReranker(list(range(1, 9))), excluded=(1,), limit=6)
    assert 1 not in result.anime_ids


def test_reranking_is_reported_as_its_own_stage(catalog_by_id):
    result = run(catalog_by_id, reranker=ReverseReranker(list(range(1, 9))))
    assert "learned_reranking" in result.diagnostics["stage_latency_ms"]


# -------------------------------------------------- artifact validation


def artifacts_present() -> bool:
    return FEATURE_ARTIFACT.exists() and MODEL_ARTIFACT.exists()


@pytest.fixture
def real_catalog():
    path = Path("data/processed/anime_catalog.json")
    if not path.exists():
        pytest.skip("processed catalog is not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_missing_artifact_returns_none_rather_than_raising(tmp_path, real_catalog):
    assert (
        try_load_reranker(
            tmp_path / "absent.npz",
            tmp_path / "absent.txt",
            real_catalog,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        is None
    )


def test_a_wrong_checksum_is_refused(real_catalog):
    if not artifacts_present():
        pytest.skip("reranker artifacts are not present in this checkout")
    with np.load(FEATURE_ARTIFACT, allow_pickle=False) as payload:
        anime_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
    with pytest.raises(RerankerUnavailable, match="checksum mismatch"):
        load_reranker(
            FEATURE_ARTIFACT,
            MODEL_ARTIFACT,
            real_catalog,
            anime_ids,
            expected_feature_sha256="0" * 64,
        )


def test_a_mismatched_item_set_is_refused(real_catalog):
    if not artifacts_present():
        pytest.skip("reranker artifacts are not present in this checkout")
    with pytest.raises(RerankerUnavailable, match="does not match the ALS item set"):
        load_reranker(FEATURE_ARTIFACT, MODEL_ARTIFACT, real_catalog, np.asarray([1, 2, 3], dtype=np.int64))


def test_the_real_reranker_reorders_without_changing_the_candidate_set(real_catalog):
    if not artifacts_present():
        pytest.skip("reranker artifacts are not present in this checkout")
    with np.load(FEATURE_ARTIFACT, allow_pickle=False) as payload:
        anime_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
    reranker = load_reranker(FEATURE_ARTIFACT, MODEL_ARTIFACT, real_catalog, anime_ids)
    assert reranker.model_info()["features"] == EXPECTED_FEATURE_COUNT

    profile = [0, 5, 11, 42]
    candidates = list(range(100, 400))
    scores = np.linspace(1.0, 0.1, len(candidates)).astype(np.float32)
    ordered = reranker.rerank(profile, candidates, scores)
    assert sorted(ordered) == sorted(candidates)
    assert ordered != candidates, "a trained reranker should disagree with ALS somewhere"
