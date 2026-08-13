from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from backend.anime_agent.collaborative import ARTIFACT_VERSION, CollaborativeIndex
from backend.anime_agent.recommender import DEFAULT_CHANNEL_WEIGHTS, MODEL_VERSION, AnimeRecommender

from .split import SplitStore, UserSplit, sha256_file


@dataclass(frozen=True)
class TrainStatistics:
    anime_ids: np.ndarray
    positive_count: np.ndarray
    observed_rating_count: np.ndarray
    observed_rating_sum: np.ndarray
    users: int

    @property
    def positive_total(self) -> int:
        return int(self.positive_count.sum())

    def positive_counts_by_id(self) -> dict[int, int]:
        return {
            int(anime_id): int(count)
            for anime_id, count in zip(self.anime_ids.tolist(), self.positive_count.tolist(), strict=True)
        }


@dataclass(frozen=True)
class OfflineRecommendation:
    anime_ids: list[int]
    diagnostics: dict[str, Any]


class OfflineEvaluationModel(Protocol):
    name: str
    version: str
    config: dict[str, Any]
    build_duration_seconds: float
    artifact_path: Path | None
    resident_array_bytes: int

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation: ...


def compute_train_statistics(store: SplitStore, catalog_ids: Sequence[int]) -> TrainStatistics:
    """Aggregate train-only item statistics in one bounded-memory pass."""
    anime_ids = np.asarray(sorted({int(value) for value in catalog_ids}), dtype=np.int64)
    index_by_id = {int(anime_id): index for index, anime_id in enumerate(anime_ids.tolist())}
    positive_count = np.zeros(len(anime_ids), dtype=np.int64)
    observed_count = np.zeros(len(anime_ids), dtype=np.int64)
    observed_sum = np.zeros(len(anime_ids), dtype=np.float64)
    users = 0
    for user in store.iter_users():
        users += 1
        for anime_id, _rating in user.train_positive:
            index = index_by_id.get(anime_id)
            if index is not None:
                positive_count[index] += 1
        for anime_id, rating in user.all_observed_training_ratings:
            index = index_by_id.get(anime_id)
            if index is not None:
                observed_count[index] += 1
                observed_sum[index] += rating
    return TrainStatistics(
        anime_ids=anime_ids,
        positive_count=positive_count,
        observed_rating_count=observed_count,
        observed_rating_sum=observed_sum,
        users=users,
    )


def save_popularity_artifact(
    statistics: TrainStatistics,
    output_path: Path,
    *,
    split_sha256: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_version": 1,
        "method": "training-positive interaction count",
        "split_sha256": split_sha256,
        "items": len(statistics.anime_ids),
        "training_positive_interactions": statistics.positive_total,
    }
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=statistics.anime_ids,
        positive_count=statistics.positive_count,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )
    temporary.replace(output_path)
    return metadata


class PopularityModel:
    name = "popularity"
    version = "train-positive-count-v1"

    def __init__(
        self,
        statistics: TrainStatistics,
        *,
        build_duration_seconds: float,
        artifact_path: Path | None,
    ):
        self.statistics = statistics
        self.build_duration_seconds = build_duration_seconds
        self.artifact_path = artifact_path
        self.config = {"signal": "positive training interaction count", "tie_breaker": "anime_id"}
        ordered = sorted(
            zip(statistics.anime_ids.tolist(), statistics.positive_count.tolist(), strict=True),
            key=lambda pair: (-int(pair[1]), int(pair[0])),
        )
        self.ordered_ids = np.asarray([anime_id for anime_id, _count in ordered], dtype=np.int64)
        self.resident_array_bytes = int(
            statistics.anime_ids.nbytes + statistics.positive_count.nbytes + self.ordered_ids.nbytes
        )

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        results: list[int] = []
        for anime_id in self.ordered_ids.tolist():
            if anime_id not in known:
                results.append(int(anime_id))
                if len(results) >= k:
                    break
        return OfflineRecommendation(results, {})


def _user_projection(user_id: int, projections: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The exact independent CountSketch projection used by the production builder."""
    mask = (1 << 64) - 1
    hashes: list[int] = []
    for projection in range(projections):
        value = (user_id * 0x9E3779B185EBCA87 + (projection + 1) * 0xC2B2AE3D27D4EB4F) & mask
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        hashes.append(value)
    buckets = np.asarray([value % width for value in hashes], dtype=np.int64)
    signs = np.asarray(
        [1.0 if (value >> 32) & 1 == 0 else -1.0 for value in hashes],
        dtype=np.float32,
    )
    return buckets, signs


def build_countsketch_artifact_from_split(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    projections: int = 3,
    width: int = 128,
) -> dict[str, Any]:
    """Rebuild the existing CountSketch model from leakage-safe train ratings.

    All observed train ratings are used because that is the current production
    algorithm.  Held-out positives are absent; explicit negatives remain real
    ratings and are never conflated with unobserved user-item pairs.
    """
    if projections < 1 or width < 8:
        raise ValueError("projections must be positive and width must be at least 8")
    started = time.perf_counter()
    anime_ids = np.asarray(sorted({int(item["id"]) for item in catalog}), dtype=np.int64)
    if not len(anime_ids):
        raise ValueError("Cannot train CountSketch with an empty catalog")
    index_by_id = {int(anime_id): index for index, anime_id in enumerate(anime_ids.tolist())}
    dimensions = projections * width
    vectors = np.zeros((len(anime_ids), dimensions), dtype=np.float32)
    counts = np.zeros(len(anime_ids), dtype=np.int64)
    sums = np.zeros(len(anime_ids), dtype=np.float64)
    offsets = np.arange(projections, dtype=np.int64) * width
    ratings_used = 0
    users_seen = 0

    for user in store.iter_users():
        observed = user.all_observed_training_ratings
        if not observed:
            continue
        kept = [(anime_id, rating) for anime_id, rating in observed if anime_id in index_by_id]
        if not kept:
            continue
        users_seen += 1
        indexes = np.asarray([index_by_id[anime_id] for anime_id, _rating in kept], dtype=np.int64)
        ratings = np.asarray([rating for _anime_id, rating in kept], dtype=np.float32)
        if len(indexes) != len(np.unique(indexes)):
            raise ValueError(f"Duplicate training item for user {user.user_id}")
        counts[indexes] += 1
        sums[indexes] += ratings
        ratings_used += len(indexes)

        residuals = ratings - float(np.mean(ratings))
        residuals /= max(float(np.std(ratings)), 1.0)
        if not np.any(np.abs(residuals) > 1e-7):
            continue
        buckets, signs = _user_projection(user.user_id, projections, width)
        columns = offsets + buckets
        vectors[indexes[:, None], columns[None, :]] += residuals[:, None] * signs[None, :]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 1e-8
    vectors[nonzero] /= norms[nonzero]
    rating_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    global_mean = float(sums.sum() / max(counts.sum(), 1))
    prior_weight = 50.0
    bayesian_score = np.divide(
        sums + prior_weight * global_mean,
        counts + prior_weight,
        out=np.full_like(sums, global_mean),
        where=(counts + prior_weight) > 0,
    )
    bayesian_score = np.clip(bayesian_score / 10.0, 0.0, 1.0).astype(np.float32)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "method": "user-centred CountSketch item similarity",
        "training_source": "personalized split train ratings",
        "split_sha256": sha256_file(store.path),
        "ratings_used": ratings_used,
        "users_seen": users_seen,
        "catalog_items": len(anime_ids),
        "items_with_ratings": int(np.count_nonzero(counts)),
        "projections": projections,
        "projection_width": width,
        "dimensions": dimensions,
        "global_rating_mean": round(global_mean, 6),
        "build_duration_seconds": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=anime_ids,
        vectors=vectors,
        rating_count=counts,
        rating_mean=rating_mean,
        bayesian_score=bayesian_score,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )
    temporary.replace(output_path)
    return metadata


def countsketch_artifact_matches(
    path: Path,
    *,
    split_sha256: str,
    projections: int,
    width: int,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("artifact_version") == ARTIFACT_VERSION
        and metadata.get("split_sha256") == split_sha256
        and metadata.get("projections") == projections
        and metadata.get("projection_width") == width
    )


class CountSketchModel:
    name = "countsketch_cf"
    version = "user-centred-countsketch-v1"

    def __init__(
        self,
        index: CollaborativeIndex,
        catalog_ids: Sequence[int],
        *,
        build_duration_seconds: float,
        artifact_path: Path,
    ):
        self.index = index
        self.catalog_ids = tuple(sorted({int(value) for value in catalog_ids}))
        self.build_duration_seconds = build_duration_seconds
        self.artifact_path = artifact_path
        self.config = {
            "method": index.metadata.get("method"),
            "dimensions": int(index.vectors.shape[1]),
            "profile_feedback": "positive training interactions",
            "training_feedback": "all observed train ratings, classes kept distinct",
        }
        self.resident_array_bytes = int(
            index.anime_ids.nbytes
            + index.vectors.nbytes
            + index.rating_count.nbytes
            + index.rating_mean.nbytes
            + index.bayesian_score.nbytes
        )

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        scores = self.index.profile_scores(positive_ids=user.train_positive_ids)
        results = heapq.nsmallest(
            k,
            (anime_id for anime_id in self.catalog_ids if anime_id not in known),
            key=lambda anime_id: (-scores.get(anime_id, 0.0), anime_id),
        )
        return OfflineRecommendation(results, {"profile_score_count": len(scores)})


def sanitize_catalog_with_training_statistics(
    catalog: Sequence[Mapping[str, Any]],
    statistics: TrainStatistics,
    collaborative_index: CollaborativeIndex,
) -> list[dict[str, Any]]:
    """Replace aggregate rating/popularity fields with train-only equivalents."""
    row_by_id = {int(anime_id): index for index, anime_id in enumerate(statistics.anime_ids.tolist())}
    popularity_order = sorted(
        statistics.anime_ids.tolist(),
        key=lambda anime_id: (-int(statistics.positive_count[row_by_id[int(anime_id)]]), int(anime_id)),
    )
    popularity_rank = {int(anime_id): position for position, anime_id in enumerate(popularity_order, start=1)}
    quality_by_id = {
        int(anime_id): collaborative_index.quality_score(int(anime_id)) for anime_id in statistics.anime_ids.tolist()
    }
    quality_order = sorted(
        statistics.anime_ids.tolist(),
        key=lambda anime_id: (
            -float(quality_by_id[int(anime_id)]) if quality_by_id[int(anime_id)] is not None else math.inf,
            int(anime_id),
        ),
    )
    quality_rank = {int(anime_id): position for position, anime_id in enumerate(quality_order, start=1)}

    result: list[dict[str, Any]] = []
    for source in catalog:
        item = dict(source)
        anime_id = int(item["id"])
        row = row_by_id.get(anime_id)
        observed_count = int(statistics.observed_rating_count[row]) if row is not None else 0
        observed_sum = float(statistics.observed_rating_sum[row]) if row is not None else 0.0
        item["score"] = round(observed_sum / observed_count, 6) if observed_count else None
        item["archive_score"] = item["score"]
        item["members"] = observed_count
        item["rating_count"] = observed_count
        item["score_distribution"] = {}
        item["watching_stats"] = {}
        item["favorites"] = 0
        item["collaborative_available"] = observed_count > 0
        item["popularity"] = popularity_rank.get(anime_id, len(popularity_rank) + 1)
        item["rank"] = quality_rank.get(anime_id, len(quality_rank) + 1)
        result.append(item)
    return result


def _recommender_array_bytes(recommender: AnimeRecommender) -> int:
    total = 0
    seen: set[int] = set()
    for value in vars(recommender).values():
        if isinstance(value, np.ndarray) and id(value) not in seen:
            total += int(value.nbytes)
            seen.add(id(value))
    if recommender.collaborative_index is not None:
        for value in vars(recommender.collaborative_index).values():
            if isinstance(value, np.ndarray) and id(value) not in seen:
                total += int(value.nbytes)
                seen.add(id(value))
    return total


class CurrentHybridModel:
    name = "current_hybrid"
    version = MODEL_VERSION

    def __init__(
        self,
        recommender: AnimeRecommender,
        *,
        build_duration_seconds: float,
        artifact_path: Path,
        catalog_artifact_path: Path | None = None,
    ):
        self.recommender = recommender
        self.build_duration_seconds = build_duration_seconds
        self.artifact_path = artifact_path
        self.catalog_artifact_path = catalog_artifact_path
        self.config = {
            "weights": dict(DEFAULT_CHANNEL_WEIGHTS),
            "diversity_strength": 0.12,
            "profile_feedback": "positive training interactions",
            "catalog_aggregate_fields": "rating-derived fields rebuilt or cleared from training data",
            "candidate_catalog": "full catalog minus exact training-known IDs",
            "llm_used": False,
        }
        self.resident_array_bytes = _recommender_array_bytes(recommender)

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        diagnostics: dict[str, Any] = {}
        liked = list(user.train_positive_ids)
        known_nonpositive = [anime_id for anime_id, _rating in (*user.explicit_negative, *user.neutral, *user.ignored)]
        results = self.recommender.recommend(
            liked_ids=liked,
            excluded_ids=known_nonpositive,
            session_profile={},
            diversity_strength=0.12,
            exclude_related_series=False,
            limit=k,
            diagnostics=diagnostics,
        )
        return OfflineRecommendation([int(item["id"]) for item in results], diagnostics)


def composite_artifact_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(value) for value in paths), key=lambda value: value.as_posix()):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
