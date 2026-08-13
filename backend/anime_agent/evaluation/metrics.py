from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    """Binary-relevance ranking metrics calculated for one user."""

    ndcg_at_10: float
    recall_at_10: float
    hit_rate_at_10: float
    ndcg_at_20: float
    recall_at_20: float
    mrr: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _unique_ranking(ranked_ids: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(anime_id) for anime_id in ranked_ids))


def recall_at_k(ranked_ids: Sequence[int], relevant_ids: Iterable[int], k: int) -> float:
    relevant = {int(anime_id) for anime_id in relevant_ids}
    if not relevant:
        return 0.0
    hits = relevant.intersection(_unique_ranking(ranked_ids)[:k])
    return len(hits) / len(relevant)


def hit_rate_at_k(ranked_ids: Sequence[int], relevant_ids: Iterable[int], k: int) -> float:
    relevant = {int(anime_id) for anime_id in relevant_ids}
    if not relevant:
        return 0.0
    return float(bool(relevant.intersection(_unique_ranking(ranked_ids)[:k])))


def ndcg_at_k(ranked_ids: Sequence[int], relevant_ids: Iterable[int], k: int) -> float:
    """Binary NDCG with an ideal ranking containing every held-out positive."""
    relevant = {int(anime_id) for anime_id in relevant_ids}
    if not relevant:
        return 0.0
    ranking = _unique_ranking(ranked_ids)[:k]
    dcg = sum(1.0 / math.log2(position + 2) for position, anime_id in enumerate(ranking) if anime_id in relevant)
    ideal_length = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(position + 2) for position in range(ideal_length))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(ranked_ids: Sequence[int], relevant_ids: Iterable[int], *, cutoff: int | None = None) -> float:
    relevant = {int(anime_id) for anime_id in relevant_ids}
    ranking = _unique_ranking(ranked_ids)
    if cutoff is not None:
        ranking = ranking[:cutoff]
    for position, anime_id in enumerate(ranking, start=1):
        if anime_id in relevant:
            return 1.0 / position
    return 0.0


def ranking_metrics(
    ranked_ids: Sequence[int],
    relevant_ids: Iterable[int],
    *,
    mrr_cutoff: int = 20,
) -> RankingMetrics:
    relevant = {int(anime_id) for anime_id in relevant_ids}
    return RankingMetrics(
        ndcg_at_10=ndcg_at_k(ranked_ids, relevant, 10),
        recall_at_10=recall_at_k(ranked_ids, relevant, 10),
        hit_rate_at_10=hit_rate_at_k(ranked_ids, relevant, 10),
        ndcg_at_20=ndcg_at_k(ranked_ids, relevant, 20),
        recall_at_20=recall_at_k(ranked_ids, relevant, 20),
        mrr=reciprocal_rank(ranked_ids, relevant, cutoff=mrr_cutoff),
    )


def user_activity_segment(train_positive_count: int) -> str:
    """Segment on positive interactions available to the model, not all ratings."""
    if train_positive_count < 1:
        return "none"
    if train_positive_count <= 4:
        return "sparse"
    if train_positive_count <= 19:
        return "medium"
    return "heavy"


def aggregate_user_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metric_names = (
        "ndcg_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_20",
        "recall_at_20",
        "mrr",
    )
    if not rows:
        return {name: 0.0 for name in metric_names}
    return {name: mean(float(row[name]) for row in rows) for name in metric_names}


def metrics_by_user_segment(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["user_segment"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for segment in ("sparse", "medium", "heavy"):
        segment_rows = grouped.get(segment, [])
        aggregate = aggregate_user_metrics(segment_rows)
        result[segment] = {
            "users": len(segment_rows),
            "ndcg_at_10": aggregate["ndcg_at_10"],
            "recall_at_10": aggregate["recall_at_10"],
            "hit_rate_at_10": aggregate["hit_rate_at_10"],
        }
    return result


def build_item_popularity_buckets(
    catalog_ids: Iterable[int],
    train_positive_counts: Mapping[int, int],
) -> dict[int, str]:
    """Assign reproducible item-count quantiles using training positives only.

    Head is the top 20% of catalog items by training-positive count, mid-tail
    the next 30%, and long-tail the bottom 50%.  Anime ID is the deterministic
    tie-breaker, and zero-interaction catalog items therefore remain long-tail.
    """
    ordered = sorted(
        {int(anime_id) for anime_id in catalog_ids},
        key=lambda anime_id: (-int(train_positive_counts.get(anime_id, 0)), anime_id),
    )
    if not ordered:
        return {}
    head_end = math.ceil(len(ordered) * 0.20)
    mid_end = math.ceil(len(ordered) * 0.50)
    return {
        anime_id: "head" if index < head_end else "mid_tail" if index < mid_end else "long_tail"
        for index, anime_id in enumerate(ordered)
    }


def catalog_coverage(recommendations: Sequence[Sequence[int]], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    exposed = {int(anime_id) for ranking in recommendations for anime_id in ranking}
    return len(exposed) / catalog_size


def item_novelty(anime_id: int, train_positive_counts: Mapping[int, int], total_train: int, catalog_size: int) -> float:
    """Self-information in bits with add-one smoothing from training only."""
    denominator = total_train + catalog_size
    if denominator <= 0:
        return 0.0
    probability = (int(train_positive_counts.get(int(anime_id), 0)) + 1) / denominator
    return -math.log2(probability)


def mean_novelty(
    recommendations: Sequence[Sequence[int]],
    train_positive_counts: Mapping[int, int],
    *,
    catalog_size: int,
) -> float:
    total_train = sum(int(value) for value in train_positive_counts.values())
    values = [
        item_novelty(anime_id, train_positive_counts, total_train, catalog_size)
        for ranking in recommendations
        for anime_id in ranking
    ]
    return mean(values) if values else 0.0


def normalized_log_popularity(anime_id: int, train_positive_counts: Mapping[int, int]) -> float:
    maximum = max((int(value) for value in train_positive_counts.values()), default=0)
    if maximum <= 0:
        return 0.0
    return math.log1p(int(train_positive_counts.get(int(anime_id), 0))) / math.log1p(maximum)


def popularity_bias(
    recommendations: Sequence[Sequence[int]],
    training_histories: Sequence[Sequence[int]],
    train_positive_counts: Mapping[int, int],
) -> float:
    """Mean per-user recommendation popularity minus profile popularity.

    Positive values indicate that recommendations are more popular than the
    user's observed positive history; negative values indicate the reverse.
    """
    differences: list[float] = []
    for ranking, history in zip(recommendations, training_histories, strict=True):
        if not ranking or not history:
            continue
        recommended = mean(normalized_log_popularity(anime_id, train_positive_counts) for anime_id in ranking)
        observed = mean(normalized_log_popularity(anime_id, train_positive_counts) for anime_id in history)
        differences.append(recommended - observed)
    return mean(differences) if differences else 0.0


def genre_jaccard_distance(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(value).casefold() for value in left if value}
    right_set = {str(value).casefold() for value in right if value}
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def intra_list_diversity(ranking: Sequence[int], genres_by_id: Mapping[int, Sequence[str]]) -> float:
    unique = _unique_ranking(ranking)
    if len(unique) < 2:
        return 0.0
    distances = [
        genre_jaccard_distance(genres_by_id.get(left, ()), genres_by_id.get(right, ()))
        for left_index, left in enumerate(unique)
        for right in unique[left_index + 1 :]
    ]
    return mean(distances) if distances else 0.0


def mean_intra_list_diversity(
    recommendations: Sequence[Sequence[int]],
    genres_by_id: Mapping[int, Sequence[str]],
) -> float:
    values = [intra_list_diversity(ranking, genres_by_id) for ranking in recommendations]
    return mean(values) if values else 0.0


def recommendation_exposure_by_bucket(
    recommendations: Sequence[Sequence[int]],
    bucket_by_id: Mapping[int, str],
) -> dict[str, float]:
    counts: Counter[str] = Counter(
        bucket_by_id.get(int(anime_id), "unknown") for ranking in recommendations for anime_id in ranking
    )
    total = sum(counts.values())
    return {
        bucket: counts.get(bucket, 0) / total if total else 0.0
        for bucket in ("head", "mid_tail", "long_tail", "unknown")
    }


def heldout_metrics_by_item_bucket(
    rankings_by_user: Mapping[int, Sequence[int]],
    relevant_by_user: Mapping[int, Sequence[int]],
    bucket_by_id: Mapping[int, str],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ("head", "mid_tail", "long_tail"):
        recalls: list[float] = []
        ndcgs: list[float] = []
        relevant_items = 0
        for user_id, relevant_ids in relevant_by_user.items():
            bucket_relevant = [anime_id for anime_id in relevant_ids if bucket_by_id.get(int(anime_id)) == bucket]
            if not bucket_relevant:
                continue
            ranking = rankings_by_user.get(int(user_id), ())
            recalls.append(recall_at_k(ranking, bucket_relevant, 10))
            ndcgs.append(ndcg_at_k(ranking, bucket_relevant, 10))
            relevant_items += len(bucket_relevant)
        result[bucket] = {
            "users": len(recalls),
            "heldout_items": relevant_items,
            "recall_at_10": mean(recalls) if recalls else 0.0,
            "ndcg_at_10": mean(ndcgs) if ndcgs else 0.0,
        }
    return result


def paired_bootstrap_difference(
    left_by_user: Mapping[int, float],
    right_by_user: Mapping[int, float],
    *,
    iterations: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired user-level percentile bootstrap for a mean metric difference."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    user_ids = sorted(set(left_by_user).intersection(right_by_user))
    if not user_ids:
        raise ValueError("paired bootstrap requires at least one shared user")
    differences = np.asarray(
        [float(left_by_user[user_id]) - float(right_by_user[user_id]) for user_id in user_ids],
        dtype=np.float64,
    )
    return paired_bootstrap_aligned(
        differences,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )


def paired_bootstrap_aligned(
    paired_differences: Sequence[float] | np.ndarray,
    *,
    iterations: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap pre-aligned per-user differences with bounded index memory."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    differences = np.asarray(paired_differences, dtype=np.float64)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired bootstrap requires at least one paired difference")
    if not np.isfinite(differences).all():
        raise ValueError("paired bootstrap differences must be finite")
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(iterations, dtype=np.float64)
    # Cap the temporary integer index matrix at roughly two million cells.
    # This remains bounded even for a full 300k-user evaluation.
    batch_size = min(256, iterations, max(1, 2_000_000 // len(differences)))
    written = 0
    while written < iterations:
        batch = min(batch_size, iterations - written)
        indexes = generator.integers(0, len(differences), size=(batch, len(differences)))
        bootstrap_means[written : written + batch] = differences[indexes].mean(axis=1)
        written += batch
    alpha = (1.0 - confidence) / 2.0
    return {
        "users": len(differences),
        "iterations": iterations,
        "seed": seed,
        "difference": float(differences.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, alpha)),
        "ci_upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
    }
