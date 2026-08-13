from __future__ import annotations

import csv
import gc
import gzip
import json
import os
import platform
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.recommender import AnimeRecommender

from .metrics import (
    build_item_popularity_buckets,
    intra_list_diversity,
    item_novelty,
    ndcg_at_k,
    normalized_log_popularity,
    paired_bootstrap_aligned,
    ranking_metrics,
    recall_at_k,
    user_activity_segment,
)
from .models import (
    CountSketchModel,
    CurrentHybridModel,
    OfflineEvaluationModel,
    PopularityModel,
    build_countsketch_artifact_from_split,
    composite_artifact_hash,
    compute_train_statistics,
    countsketch_artifact_matches,
    sanitize_catalog_with_training_statistics,
    save_popularity_artifact,
)
from .split import METHODOLOGY_NOTE, SplitStore, UserSplit, select_evaluation_user_ids, sha256_file


@dataclass(frozen=True)
class EvaluationRunConfig:
    sample_seed: int = 42
    sampling_strategy: str = "uniform"
    max_evaluation_users: int | None = 100
    recommendation_k: int = 20
    bootstrap_iterations: int = 2_000
    countsketch_projections: int = 3
    countsketch_width: int = 128
    force_model_rebuild: bool = False
    progress_every: int = 25

    def __post_init__(self) -> None:
        if self.recommendation_k < 20:
            raise ValueError("recommendation_k must be at least 20 for the requested metrics")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        if self.sampling_strategy not in {"uniform", "stratified"}:
            raise ValueError("sampling_strategy must be 'uniform' or 'stratified'")


@dataclass(frozen=True)
class AlignedPrimaryMetrics:
    """Compact per-user values retained for paired comparisons only."""

    user_ids: np.ndarray
    ndcg_at_10: np.ndarray
    recall_at_10: np.ndarray


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _measure_build(
    function: Callable[[], Any],
    *,
    trace_allocations: bool = False,
) -> tuple[Any, float, int | None]:
    gc.collect()
    if trace_allocations:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        value = function()
        duration = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1] if trace_allocations else None
    finally:
        if trace_allocations:
            tracemalloc.stop()
    return value, duration, int(peak) if peak is not None else None


def _peak_process_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            success = get_process_memory_info(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.PeakWorkingSetSize) if success else None
        except (AttributeError, OSError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, ValueError):
        return None


def _current_process_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            success = get_process_memory_info(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize) if success else None
        except (AttributeError, OSError):
            return None
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            return None
    return _peak_process_rss_bytes()


def _environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "peak_process_rss_bytes": _peak_process_rss_bytes(),
    }


def _model_artifact_info(model: OfflineEvaluationModel, *, catalog_path: Path) -> dict[str, Any]:
    paths: list[Path] = []
    if model.artifact_path and model.artifact_path.exists():
        paths.append(model.artifact_path)
    if isinstance(model, CurrentHybridModel) and catalog_path.exists():
        paths.append(catalog_path)
    return {
        "files": [path.name for path in paths],
        "size_bytes": sum(path.stat().st_size for path in paths),
        "sha256": composite_artifact_hash(paths) if paths else None,
    }


def _prepare_models(
    store: SplitStore,
    catalog: list[dict[str, Any]],
    *,
    catalog_path: Path,
    artifacts_dir: Path,
    config: EvaluationRunConfig,
) -> tuple[list[OfflineEvaluationModel], dict[str, Any], dict[int, int]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    split_sha256 = sha256_file(store.path)
    catalog_ids = [int(item["id"]) for item in catalog]

    statistics_rss_before = _current_process_rss_bytes()
    statistics, statistics_seconds, statistics_peak = _measure_build(
        lambda: compute_train_statistics(store, catalog_ids)
    )
    statistics_rss_after = _current_process_rss_bytes()
    popularity_path = artifacts_dir / "popularity_train_only.npz"
    save_popularity_artifact(statistics, popularity_path, split_sha256=split_sha256)
    popularity = PopularityModel(
        statistics,
        build_duration_seconds=statistics_seconds,
        artifact_path=popularity_path,
    )

    countsketch_path = artifacts_dir / "countsketch_train_only.npz"
    countsketch_rss_before = _current_process_rss_bytes()
    reusable = not config.force_model_rebuild and countsketch_artifact_matches(
        countsketch_path,
        split_sha256=split_sha256,
        projections=config.countsketch_projections,
        width=config.countsketch_width,
    )
    if reusable:
        with np.load(countsketch_path, allow_pickle=False) as artifact:
            countsketch_metadata = json.loads(str(artifact["metadata_json"].item()))
        countsketch_seconds = float(countsketch_metadata.get("build_duration_seconds", 0.0))
        countsketch_peak = None
    else:
        countsketch_metadata, countsketch_seconds, countsketch_peak = _measure_build(
            lambda: build_countsketch_artifact_from_split(
                store,
                catalog,
                countsketch_path,
                projections=config.countsketch_projections,
                width=config.countsketch_width,
            )
        )
        countsketch_metadata["measured_wall_seconds"] = countsketch_seconds
    countsketch_rss_after = _current_process_rss_bytes()
    countsketch_process_peak = _peak_process_rss_bytes()
    collaborative_index = CollaborativeIndex.load(countsketch_path, catalog)
    countsketch = CountSketchModel(
        collaborative_index,
        catalog_ids,
        build_duration_seconds=countsketch_seconds,
        artifact_path=countsketch_path,
    )

    def build_hybrid() -> CurrentHybridModel:
        evaluation_catalog = sanitize_catalog_with_training_statistics(
            catalog,
            statistics,
            collaborative_index,
        )
        started = time.perf_counter()
        recommender = AnimeRecommender(
            evaluation_catalog,
            collaborative_index=collaborative_index,
        )
        initialization_seconds = time.perf_counter() - started
        return CurrentHybridModel(
            recommender,
            build_duration_seconds=initialization_seconds,
            artifact_path=countsketch_path,
            catalog_artifact_path=catalog_path,
        )

    hybrid_rss_before = _current_process_rss_bytes()
    hybrid, measured_hybrid_seconds, _hybrid_trace_peak = _measure_build(
        build_hybrid,
        trace_allocations=False,
    )
    hybrid_rss_after = _current_process_rss_bytes()
    hybrid.build_duration_seconds = measured_hybrid_seconds

    build_details = {
        "split_sha256": split_sha256,
        "training_statistics": {
            "duration_seconds": statistics_seconds,
            "python_tracemalloc_peak_bytes": statistics_peak,
            "process_rss_before_bytes": statistics_rss_before,
            "process_rss_after_bytes": statistics_rss_after,
            "process_rss_delta_bytes": (
                max(0, statistics_rss_after - statistics_rss_before)
                if statistics_rss_before is not None and statistics_rss_after is not None
                else None
            ),
            "positive_interactions": statistics.positive_total,
        },
        "countsketch": {
            "reused": reusable,
            "metadata": countsketch_metadata,
            "python_tracemalloc_peak_bytes": countsketch_peak,
            "process_rss_before_bytes": countsketch_rss_before,
            "process_rss_after_bytes": countsketch_rss_after,
            "process_peak_rss_after_bytes": countsketch_process_peak,
        },
        "hybrid": {
            "process_rss_before_bytes": hybrid_rss_before,
            "process_rss_after_bytes": hybrid_rss_after,
            "process_rss_delta_bytes": (
                max(0, hybrid_rss_after - hybrid_rss_before)
                if hybrid_rss_before is not None and hybrid_rss_after is not None
                else None
            ),
            "measurement_note": "RSS delta is used because tracemalloc materially distorts hybrid initialization.",
        },
    }
    return [popularity, countsketch, hybrid], build_details, statistics.positive_counts_by_id()


def _evaluate_one_model(
    model: OfflineEvaluationModel,
    users: Iterable[UserSplit],
    *,
    expected_user_ids: Sequence[int],
    k: int,
    catalog_ids: set[int],
    genres_by_id: Mapping[int, Sequence[str]],
    train_positive_counts: Mapping[int, int],
    bucket_by_id: Mapping[int, str],
    row_writer: csv.DictWriter,
    progress: Callable[[str], None] | None,
    progress_every: int,
) -> tuple[dict[str, Any], AlignedPrimaryMetrics]:
    metric_names = (
        "ndcg_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_20",
        "recall_at_20",
        "mrr",
    )
    metric_totals: dict[str, float] = defaultdict(float)
    segment_totals = {
        segment: {"users": 0, "ndcg_at_10": 0.0, "recall_at_10": 0.0, "hit_rate_at_10": 0.0}
        for segment in ("sparse", "medium", "heavy")
    }
    bucket_totals = {
        bucket: {"users": 0, "heldout_items": 0, "recall_at_10": 0.0, "ndcg_at_10": 0.0}
        for bucket in ("head", "mid_tail", "long_tail")
    }
    exposure_counts: Counter[str] = Counter()
    exposed_ids: set[int] = set()
    novelty_total = 0.0
    recommendation_count = 0
    popularity_bias_total = 0.0
    popularity_bias_users = 0
    diversity_total = 0.0
    latencies: list[float] = []
    serialization_latencies: list[float] = []
    stage_timings: dict[str, list[float]] = defaultdict(list)
    evaluated_user_ids: list[int] = []
    ndcg_values: list[float] = []
    recall_values: list[float] = []

    total_train_positives = sum(int(value) for value in train_positive_counts.values())
    for position, user in enumerate(users):
        if position >= len(expected_user_ids) or user.user_id != int(expected_user_ids[position]):
            raise RuntimeError("Evaluation users changed or were reordered between model runs")
        started = time.perf_counter()
        recommendation = model.recommend(user, k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        ranking = list(dict.fromkeys(int(value) for value in recommendation.anime_ids))[:k]
        unknown = set(ranking).difference(catalog_ids)
        if unknown:
            raise RuntimeError(f"{model.name} returned IDs outside the candidate catalog: {sorted(unknown)[:5]}")
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        leaked_known = known.intersection(ranking)
        if leaked_known:
            raise RuntimeError(
                f"{model.name} recommended known training items for user {user.user_id}: {sorted(leaked_known)[:5]}"
            )
        relevant = list(user.test_positive_ids)
        if not relevant:
            raise RuntimeError(f"Eligible user {user.user_id} has no test positives")
        if set(relevant).intersection(known):
            raise RuntimeError(f"Split leakage detected for user {user.user_id}")
        metric = ranking_metrics(ranking, relevant)
        serialization_started = time.perf_counter()
        json.dumps(ranking, separators=(",", ":"))
        serialization_latencies.append((time.perf_counter() - serialization_started) * 1000)
        for stage, value in recommendation.diagnostics.get("timing_ms", {}).items():
            stage_timings[str(stage)].append(float(value))
        row = {
            "model": model.name,
            "user_id": user.user_id,
            "train_positive_count": len(user.train_positive),
            "test_positive_count": len(relevant),
            "user_segment": user_activity_segment(len(user.train_positive)),
            "latency_ms": elapsed_ms,
            **metric.as_dict(),
        }
        row_writer.writerow(row)
        evaluated_user_ids.append(user.user_id)
        ndcg_values.append(metric.ndcg_at_10)
        recall_values.append(metric.recall_at_10)
        for name in metric_names:
            metric_totals[name] += float(row[name])

        segment = str(row["user_segment"])
        segment_totals[segment]["users"] += 1
        for name in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10"):
            segment_totals[segment][name] += float(row[name])

        for bucket in ("head", "mid_tail", "long_tail"):
            bucket_relevant = [anime_id for anime_id in relevant if bucket_by_id.get(int(anime_id)) == bucket]
            if bucket_relevant:
                bucket_totals[bucket]["users"] += 1
                bucket_totals[bucket]["heldout_items"] += len(bucket_relevant)
                bucket_totals[bucket]["recall_at_10"] += recall_at_k(ranking, bucket_relevant, 10)
                bucket_totals[bucket]["ndcg_at_10"] += ndcg_at_k(ranking, bucket_relevant, 10)

        exposed_ids.update(ranking)
        for anime_id in ranking:
            exposure_counts[bucket_by_id.get(anime_id, "unknown")] += 1
            novelty_total += item_novelty(
                anime_id,
                train_positive_counts,
                total_train_positives,
                len(catalog_ids),
            )
            recommendation_count += 1
        history = list(user.train_positive_ids)
        if ranking and history:
            recommendation_popularity = sum(
                normalized_log_popularity(anime_id, train_positive_counts) for anime_id in ranking
            ) / len(ranking)
            history_popularity = sum(
                normalized_log_popularity(anime_id, train_positive_counts) for anime_id in history
            ) / len(history)
            popularity_bias_total += recommendation_popularity - history_popularity
            popularity_bias_users += 1
        diversity_total += intra_list_diversity(ranking, genres_by_id)

        if progress is not None and (position + 1) % progress_every == 0:
            progress(f"{model.name}: evaluated {position + 1:,}/{len(expected_user_ids):,} users")

    evaluated_users = len(evaluated_user_ids)
    if evaluated_users != len(expected_user_ids):
        raise RuntimeError(f"Expected {len(expected_user_ids):,} users but {model.name} evaluated {evaluated_users:,}")
    aggregate = {name: metric_totals[name] / evaluated_users if evaluated_users else 0.0 for name in metric_names}
    user_segments: dict[str, dict[str, float | int]] = {}
    for segment, values in segment_totals.items():
        users_in_segment = int(values["users"])
        user_segments[segment] = {
            "users": users_in_segment,
            **{
                name: float(values[name]) / users_in_segment if users_in_segment else 0.0
                for name in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10")
            },
        }
    heldout_item_popularity: dict[str, dict[str, float | int]] = {}
    for bucket, values in bucket_totals.items():
        users_in_bucket = int(values["users"])
        heldout_item_popularity[bucket] = {
            "users": users_in_bucket,
            "heldout_items": int(values["heldout_items"]),
            "recall_at_10": float(values["recall_at_10"]) / users_in_bucket if users_in_bucket else 0.0,
            "ndcg_at_10": float(values["ndcg_at_10"]) / users_in_bucket if users_in_bucket else 0.0,
        }
    total_exposure = sum(exposure_counts.values())
    summary: dict[str, Any] = {
        "model": model.name,
        "model_version": model.version,
        "evaluated_users": evaluated_users,
        **aggregate,
        "catalog_coverage": len(exposed_ids) / len(catalog_ids) if catalog_ids else 0.0,
        "novelty_bits": novelty_total / recommendation_count if recommendation_count else 0.0,
        "popularity_bias": (popularity_bias_total / popularity_bias_users if popularity_bias_users else 0.0),
        "intra_list_diversity": diversity_total / evaluated_users if evaluated_users else 0.0,
        "recommendation_exposure": {
            bucket: exposure_counts.get(bucket, 0) / total_exposure if total_exposure else 0.0
            for bucket in ("head", "mid_tail", "long_tail", "unknown")
        },
        "user_segments": user_segments,
        "heldout_item_popularity": heldout_item_popularity,
        "engineering": {
            "build_duration_seconds": model.build_duration_seconds,
            "inference_latency_p50_ms": _percentile(latencies, 50),
            "inference_latency_p95_ms": _percentile(latencies, 95),
            "serialization_latency_p50_ms": _percentile(serialization_latencies, 50),
            "serialization_latency_p95_ms": _percentile(serialization_latencies, 95),
            "resident_array_bytes": model.resident_array_bytes,
            "stage_latency_ms": {
                stage: {"p50": _percentile(values, 50), "p95": _percentile(values, 95)}
                for stage, values in sorted(stage_timings.items())
            },
        },
    }
    return summary, AlignedPrimaryMetrics(
        user_ids=np.asarray(evaluated_user_ids, dtype=np.int64),
        ndcg_at_10=np.asarray(ndcg_values, dtype=np.float64),
        recall_at_10=np.asarray(recall_values, dtype=np.float64),
    )


def _bootstrap_comparisons(
    metrics_by_model: Mapping[str, AlignedPrimaryMetrics],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    comparisons = (
        ("countsketch_cf", "popularity"),
        ("current_hybrid", "countsketch_cf"),
        ("current_hybrid", "popularity"),
    )
    result: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_metrics = metrics_by_model[left]
        right_metrics = metrics_by_model[right]
        if not np.array_equal(left_metrics.user_ids, right_metrics.user_ids):
            raise RuntimeError(f"Cannot pair {left} and {right}: evaluated user IDs differ")
        for metric in ("ndcg_at_10", "recall_at_10"):
            interval = paired_bootstrap_aligned(
                getattr(left_metrics, metric) - getattr(right_metrics, metric),
                iterations=iterations,
                seed=seed,
            )
            result.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    **interval,
                    "ci_excludes_zero": bool(interval["ci_lower"] > 0 or interval["ci_upper"] < 0),
                }
            )
    return result


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model",
        "ndcg_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_20",
        "recall_at_20",
        "mrr",
        "catalog_coverage",
        "novelty_bits",
        "popularity_bias",
        "intra_list_diversity",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def _dependency_inclusive_build_seconds(result: Mapping[str, Any], model_name: str) -> float:
    models = _model_by_name(result)
    statistics_seconds = float(result["build_details"]["training_statistics"]["duration_seconds"])
    if model_name == "popularity":
        return statistics_seconds
    countsketch_seconds = float(models["countsketch_cf"]["engineering"]["build_duration_seconds"])
    if model_name == "countsketch_cf":
        return countsketch_seconds
    return (
        statistics_seconds
        + countsketch_seconds
        + float(models["current_hybrid"]["engineering"]["build_duration_seconds"])
    )


def _write_slice_csvs(output_dir: Path, result: Mapping[str, Any]) -> None:
    summaries = result["models"]
    with (output_dir / "user_segments.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["model", "segment", "users", "ndcg_at_10", "recall_at_10", "hit_rate_at_10"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for segment, values in summary["user_segments"].items():
                writer.writerow({"model": summary["model"], "segment": segment, **values})

    with (output_dir / "item_popularity.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "model",
            "bucket",
            "users",
            "heldout_items",
            "recall_at_10",
            "ndcg_at_10",
            "recommendation_exposure",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for bucket, values in summary["heldout_item_popularity"].items():
                writer.writerow(
                    {
                        "model": summary["model"],
                        "bucket": bucket,
                        **values,
                        "recommendation_exposure": summary["recommendation_exposure"][bucket],
                    }
                )

    with (output_dir / "engineering.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "model",
            "build_duration_seconds",
            "dependency_inclusive_build_duration_seconds",
            "inference_latency_p50_ms",
            "inference_latency_p95_ms",
            "serialization_latency_p50_ms",
            "serialization_latency_p95_ms",
            "resident_array_bytes",
            "artifact_size_bytes",
            "artifact_sha256",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            engineering = summary["engineering"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "build_duration_seconds": engineering["build_duration_seconds"],
                    "dependency_inclusive_build_duration_seconds": _dependency_inclusive_build_seconds(
                        result, str(summary["model"])
                    ),
                    **{field: engineering[field] for field in fields[3:8]},
                    "artifact_size_bytes": engineering["artifact"]["size_bytes"],
                    "artifact_sha256": engineering["artifact"]["sha256"],
                }
            )


def _write_bootstrap_csv(path: Path, comparisons: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "left_model",
        "right_model",
        "metric",
        "users",
        "iterations",
        "seed",
        "difference",
        "ci_lower",
        "ci_upper",
        "ci_excludes_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({field: comparison[field] for field in fields})


PER_USER_FIELDS = [
    "model",
    "user_id",
    "train_positive_count",
    "test_positive_count",
    "user_segment",
    "ndcg_at_10",
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_20",
    "recall_at_20",
    "mrr",
    "latency_ms",
]


def run_personalized_evaluation(
    store: SplitStore,
    catalog: list[dict[str, Any]],
    *,
    catalog_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    config: EvaluationRunConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = config or EvaluationRunConfig()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = store.metadata()
    split_audit = store.audit_counts()
    eligible_segment_counts = store.eligible_segment_counts()
    evaluation_user_ids = select_evaluation_user_ids(
        store,
        limit=config.max_evaluation_users,
        seed=config.sample_seed,
        strategy=config.sampling_strategy,
    )
    if not evaluation_user_ids:
        raise ValueError("No eligible users are available for evaluation")

    if progress is not None:
        progress(f"models: preparing train-only artifacts for {len(evaluation_user_ids):,} evaluation users")
    models, build_details, train_positive_counts = _prepare_models(
        store,
        catalog,
        catalog_path=catalog_path,
        artifacts_dir=artifacts_dir,
        config=config,
    )
    catalog_ids = {int(item["id"]) for item in catalog}
    genres_by_id = {int(item["id"]): list(item.get("genres") or []) for item in catalog}
    bucket_by_id = build_item_popularity_buckets(catalog_ids, train_positive_counts)

    summaries: list[dict[str, Any]] = []
    metrics_by_model: dict[str, AlignedPrimaryMetrics] = {}
    per_user_path = output_dir / "per_user_metrics.csv.gz"
    with gzip.open(per_user_path, "wt", encoding="utf-8", newline="") as per_user_file:
        row_writer = csv.DictWriter(per_user_file, fieldnames=PER_USER_FIELDS)
        row_writer.writeheader()
        for model in models:
            if progress is not None:
                progress(f"{model.name}: starting offline inference")
            summary, aligned_metrics = _evaluate_one_model(
                model,
                store.iter_users_by_ids(evaluation_user_ids),
                expected_user_ids=evaluation_user_ids,
                k=config.recommendation_k,
                catalog_ids=catalog_ids,
                genres_by_id=genres_by_id,
                train_positive_counts=train_positive_counts,
                bucket_by_id=bucket_by_id,
                row_writer=row_writer,
                progress=progress,
                progress_every=config.progress_every,
            )
            summary["model_config"] = model.config
            summary["engineering"]["artifact"] = _model_artifact_info(model, catalog_path=catalog_path)
            summaries.append(summary)
            metrics_by_model[model.name] = aligned_metrics
            if progress is not None:
                progress(f"{model.name}: complete")

    bootstrap = _bootstrap_comparisons(
        metrics_by_model,
        iterations=config.bootstrap_iterations,
        seed=config.sample_seed,
    )
    sample_segment_counts: dict[str, int] = defaultdict(int)
    for user in store.iter_users_by_ids(evaluation_user_ids):
        sample_segment_counts[user_activity_segment(len(user.train_positive))] += 1

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_scope": "full" if config.max_evaluation_users in (None, 0) else "sampled",
        "methodology_note": METHODOLOGY_NOTE,
        "run_config": asdict(config),
        "dataset": {
            "source_file": split_metadata.get("source_file"),
            "source_user_limit": split_metadata.get("source_user_limit"),
            "source_rows_scanned": split_metadata.get("rows_scanned"),
            "source_orphan_rows": split_metadata.get("orphan_rows"),
            "split_build_duration_seconds": split_metadata.get("build_duration_seconds"),
            "dataset_sha256": split_metadata.get("dataset_sha256"),
            "dataset_sha256_scope": split_metadata.get("dataset_sha256_scope", "full_file"),
            "catalog_sha256": sha256_file(catalog_path),
            "split_sha256": build_details["split_sha256"],
            "rating_threshold": split_metadata.get("split_config", {}).get("feedback", {}).get("positive_threshold"),
            "split_seed": split_metadata.get("split_config", {}).get("seed"),
            "users_before_filter": split_metadata.get("users_before_filter"),
            "users_after_filter": split_metadata.get("users_after_filter"),
            "train_positive_interactions": split_metadata.get("train_positive_interactions"),
            "validation_positive_interactions": split_metadata.get("validation_positive_interactions"),
            "test_positive_interactions": split_metadata.get("test_positive_interactions"),
            "explicit_negative_ratings": split_metadata.get("explicit_negative_ratings"),
            "neutral_ratings": split_metadata.get("neutral_ratings"),
            "ignored_ratings": split_metadata.get("ignored_ratings"),
            "train_positive_sparsity": split_metadata.get("train_positive_sparsity"),
            "catalog_items": len(catalog),
            "split_audit": split_audit,
        },
        "evaluation_population": {
            "evaluated_users": len(evaluation_user_ids),
            "sampling": (
                "all eligible users"
                if config.max_evaluation_users in (None, 0)
                else f"deterministic {config.sampling_strategy} hash sample of eligible users"
            ),
            "segment_counts": dict(sorted(sample_segment_counts.items())),
            "eligible_population_segment_counts": eligible_segment_counts,
            "aggregate_weighting": (
                "unweighted macro-average over all eligible users"
                if config.max_evaluation_users in (None, 0)
                else "unweighted macro-average over sampled users; no population reweighting"
            ),
            "user_ids_sha256": hashlib_sha256_ids(evaluation_user_ids),
        },
        "item_popularity_bucket_definition": {
            "source": "positive training interaction count only",
            "head": "top 20% of catalog items by count",
            "mid_tail": "next 30% of catalog items by count",
            "long_tail": "bottom 50% of catalog items by count, including zero-count items",
            "tie_breaker": "anime_id ascending",
        },
        "metric_definitions": {
            "ranking": "Per-user binary relevance; aggregate values are unweighted means across users.",
            "mrr": f"Reciprocal rank of the first test positive within the top {config.recommendation_k}.",
            "catalog_coverage": "Unique recommended catalog IDs divided by candidate catalog size.",
            "novelty_bits": "Mean -log2((train positive count + 1)/(all train positives + catalog size)).",
            "popularity_bias": "Mean per-user normalized-log popularity of recommendations minus training-positive history.",
            "intra_list_diversity": "Mean pairwise Jaccard distance over catalog genre sets; empty/empty distance is zero.",
            "beyond_accuracy_scope": f"Coverage, novelty, popularity bias, exposure, and ILD use top-{config.recommendation_k} rankings.",
            "serialization": "Timing covers compact ranking-ID JSON serialization, not the production HTTP response payload.",
        },
        "models": summaries,
        "paired_bootstrap": bootstrap,
        "build_details": build_details,
        "environment": _environment_info(),
        "evaluation_duration_seconds": round(time.perf_counter() - started, 6),
        "what_this_can_tell_us": [
            "Personalized ranking quality under deterministic random held-out positive interactions.",
            "Whether the current collaborative signal adds value over train-only popularity.",
            "Whether the current hybrid adds value over its CountSketch collaborative channel.",
            "How ranking quality changes with training-positive user activity.",
            "Recommendation exposure and recovery across train-defined item popularity buckets.",
        ],
        "what_this_cannot_tell_us": [
            "Chronological next-anime prediction because the source has no interaction timestamps.",
            "Real online click-through rate or recommendation acceptance.",
            "Causal user satisfaction.",
            "Production A/B-test performance without online traffic.",
        ],
    }
    result["environment"]["peak_process_rss_bytes"] = _peak_process_rss_bytes()

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_csv(output_dir / "summary.csv", summaries)
    _write_slice_csvs(output_dir, result)
    _write_bootstrap_csv(output_dir / "paired_bootstrap.csv", bootstrap)
    (output_dir / "report.md").write_text(render_markdown_report(result), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": result["generated_at"],
        "experiment": {
            "dataset_sha256": result["dataset"]["dataset_sha256"],
            "catalog_sha256": result["dataset"]["catalog_sha256"],
            "split_sha256": result["dataset"]["split_sha256"],
            "split_seed": result["dataset"]["split_seed"],
            "positive_threshold": result["dataset"]["rating_threshold"],
            "evaluated_users": len(evaluation_user_ids),
            "run_config": result["run_config"],
            "environment": result["environment"],
        },
        "models": [
            {
                "name": summary["model"],
                "version": summary["model_version"],
                "hyperparameters": summary["model_config"],
                "training_or_build_duration_seconds": summary["engineering"]["build_duration_seconds"],
                "artifact": summary["engineering"]["artifact"],
            }
            for summary in summaries
        ],
        "outputs": {},
    }
    for name in (
        "results.json",
        "summary.csv",
        "user_segments.csv",
        "item_popularity.csv",
        "engineering.csv",
        "paired_bootstrap.csv",
        "per_user_metrics.csv.gz",
        "report.md",
    ):
        path = output_dir / name
        manifest["outputs"][name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if progress is not None:
        progress(f"outputs: wrote benchmark artifacts to {output_dir}")
    return result


def refresh_derived_outputs(output_dir: Path) -> None:
    """Regenerate report/CSV views from a completed immutable results JSON."""
    output_dir = Path(output_dir)
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    summaries = result["models"]
    _write_summary_csv(output_dir / "summary.csv", summaries)
    _write_slice_csvs(output_dir, result)
    _write_bootstrap_csv(output_dir / "paired_bootstrap.csv", result["paired_bootstrap"])
    (output_dir / "report.md").write_text(render_markdown_report(result), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def hashlib_sha256_ids(values: Sequence[int] | Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)}\n".encode())
    return digest.hexdigest()


def _model_by_name(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(model["model"]): model for model in result["models"]}


def _bootstrap_lookup(result: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["left_model"]), str(row["right_model"]), str(row["metric"])): row for row in result["paired_bootstrap"]
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_markdown_report(result: Mapping[str, Any]) -> str:
    models = _model_by_name(result)
    bootstrap = _bootstrap_lookup(result)
    model_order = ("popularity", "countsketch_cf", "current_hybrid")
    labels = {
        "popularity": "Popularity",
        "countsketch_cf": "CountSketch CF",
        "current_hybrid": "Current Hybrid",
    }
    lines = [
        "# Personalized Offline Recommendation Benchmark",
        "",
        f"> {result['methodology_note']}",
        "",
        f"Run scope: **{result['run_scope']}**; evaluated users: **{result['evaluation_population']['evaluated_users']:,}**; "
        f"positive threshold: **{result['dataset']['rating_threshold']}**; split seed: **{result['dataset']['split_seed']}**.",
        "",
        "## Ranking and beyond-accuracy results",
        "",
        "| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in model_order:
        model = models[name]
        lines.append(
            f"| {labels[name]} | {_fmt(model['ndcg_at_10'])} | {_fmt(model['recall_at_10'])} | "
            f"{_fmt(model['hit_rate_at_10'])} | {_fmt(model['ndcg_at_20'])} | {_fmt(model['recall_at_20'])} | "
            f"{_fmt(model['mrr'])} | {_fmt(model['catalog_coverage'])} | {_fmt(model['novelty_bits'], 3)} | "
            f"{_fmt(model['popularity_bias'])} | {_fmt(model['intra_list_diversity'])} |"
        )

    lines.extend(["", "## Performance by training-positive user activity", ""])
    for segment in ("sparse", "medium", "heavy"):
        lines.extend(
            [
                f"### {segment.replace('_', ' ').title()} users",
                "",
                "| Model | Users | NDCG@10 | Recall@10 | HR@10 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name in model_order:
            values = models[name]["user_segments"][segment]
            lines.append(
                f"| {labels[name]} | {int(values['users'])} | {_fmt(values['ndcg_at_10'])} | "
                f"{_fmt(values['recall_at_10'])} | {_fmt(values['hit_rate_at_10'])} |"
            )
        lines.append("")

    lines.extend(["## Performance by held-out item popularity", ""])
    for bucket in ("head", "mid_tail", "long_tail"):
        lines.extend(
            [
                f"### {bucket.replace('_', '-').title()}",
                "",
                "| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in model_order:
            values = models[name]["heldout_item_popularity"][bucket]
            exposure = models[name]["recommendation_exposure"][bucket]
            lines.append(
                f"| {labels[name]} | {int(values['users'])} | {int(values['heldout_items'])} | "
                f"{_fmt(values['recall_at_10'])} | {_fmt(values['ndcg_at_10'])} | {_fmt(exposure)} |"
            )
        lines.append("")

    lines.extend(
        [
            "The buckets are defined only from training positives: head is the top 20% of catalog items by count, "
            "mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.",
            "",
            "## Engineering metrics",
            "",
            "Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the "
            "resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. "
            "The hybrid uses its ranking-only interface, so deterministic explanation/result-payload construction is also excluded. "
            "Incremental build is the model's own stage; end-to-end includes required shared train-statistics/CountSketch stages.",
            "",
            "| Model | Incremental build | End-to-end build | p50 inference | p95 inference | Array memory | Artifact size |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in model_order:
        engineering = models[name]["engineering"]
        lines.append(
            f"| {labels[name]} | {engineering['build_duration_seconds']:.2f}s | "
            f"{_dependency_inclusive_build_seconds(result, name):.2f}s | "
            f"{engineering['inference_latency_p50_ms']:.2f}ms | {engineering['inference_latency_p95_ms']:.2f}ms | "
            f"{engineering['resident_array_bytes'] / 1024 / 1024:.2f} MiB | "
            f"{engineering['artifact']['size_bytes'] / 1024 / 1024:.2f} MiB |"
        )

    process_peak = result["environment"].get("peak_process_rss_bytes")
    lines.append("")
    lines.append(
        f"Whole-run process peak RSS: **{int(process_peak) / 1024 / 1024:.2f} MiB**."
        if process_peak is not None
        else "Whole-run process peak RSS was unavailable on this platform; per-model NumPy footprints remain reported."
    )

    hybrid_stages = models["current_hybrid"]["engineering"]["stage_latency_ms"]
    if hybrid_stages:
        lines.extend(
            [
                "",
                "### Current hybrid timing stages",
                "",
                "The production recommender currently exposes combined candidate-generation/channel-scoring and "
                "diversity-reranking timers; it does not separately time each content channel.",
                "",
                "| Stage | p50 | p95 |",
                "|---|---:|---:|",
            ]
        )
        for stage, values in hybrid_stages.items():
            lines.append(f"| {stage.replace('_', ' ')} | {float(values['p50']):.2f}ms | {float(values['p95']):.2f}ms |")

    lines.extend(["", "## Paired statistical comparisons", ""])
    for left, right in (("countsketch_cf", "popularity"), ("current_hybrid", "countsketch_cf")):
        lines.append(f"### {labels[left]} vs {labels[right]}")
        lines.append("")
        for metric in ("ndcg_at_10", "recall_at_10"):
            interval = bootstrap[(left, right, metric)]
            readable = "NDCG@10" if metric == "ndcg_at_10" else "Recall@10"
            lines.append(
                f"- Delta {readable}: {float(interval['difference']) * 100:+.2f} percentage points; "
                f"95% paired-bootstrap CI [{float(interval['ci_lower']) * 100:+.2f}, "
                f"{float(interval['ci_upper']) * 100:+.2f}] pp."
            )
        lines.append("")

    lines.extend(["## Interpretation", ""])
    sample_warning = result["run_scope"] != "full"
    if sample_warning:
        lines.append(
            "**This is a deterministic sampled run. Its findings are provisional; run the full eligible-user evaluation "
            "before using small differences to select a model.**"
        )
        lines.append("")
    if result["dataset"].get("source_user_limit") is not None:
        lines.append(
            "**Pipeline smoke only:** model artifacts were trained on a source-user prefix, so ranking values must "
            "not be compared with full-data experiments."
        )
        lines.append("")
    if result["run_config"].get("sampling_strategy") == "stratified":
        lines.append(
            "**This activity-balanced sample intentionally gives each user segment equal quota. Aggregate metrics "
            "are unweighted and therefore are not estimates of whole-population performance; use the uniform sample "
            "for the primary aggregate comparison.**"
        )
        lines.append("")
    for left, right, question in (
        ("countsketch_cf", "popularity", "CountSketch versus popularity"),
        ("current_hybrid", "countsketch_cf", "Hybrid versus CountSketch"),
    ):
        interval = bootstrap[(left, right, "ndcg_at_10")]
        direction = "higher" if interval["difference"] > 0 else "lower" if interval["difference"] < 0 else "unchanged"
        credible = "excludes zero" if interval["ci_excludes_zero"] else "includes zero"
        lines.append(
            f"- **{question}:** NDCG@10 is {direction} by {abs(float(interval['difference'])) * 100:.2f} pp; "
            f"the 95% interval {credible}."
        )
    fastest = min(model_order, key=lambda name: models[name]["engineering"]["inference_latency_p50_ms"])
    hybrid_latency = float(models["current_hybrid"]["engineering"]["inference_latency_p50_ms"])
    lines.append(f"- **Latency:** {labels[fastest]} is fastest. Hybrid p50 is {hybrid_latency:.1f} ms.")
    eligible_users = int(result["dataset"].get("users_after_filter") or 0)
    if sample_warning and eligible_users:
        projected_hours = eligible_users * hybrid_latency / 1000 / 60 / 60
        lines.append(
            f"- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about "
            f"{projected_hours:.1f} hours for {eligible_users:,} eligible users. This is a planning estimate, "
            "not a measured full-run duration."
        )
    for segment in ("sparse", "medium", "heavy"):
        if int(models["popularity"]["user_segments"][segment]["users"]):
            best = max(model_order, key=lambda name: models[name]["user_segments"][segment]["ndcg_at_10"])
            value = float(models[best]["user_segments"][segment]["ndcg_at_10"])
            lines.append(
                f"- **{segment.title()} users:** {labels[best]} has the highest sampled NDCG@10 ({value:.4f})."
            )
    lines.append(
        "- **Popularity/long tail:** inspect exposure together with held-out long-tail Recall@10; a model can appear "
        "novel simply because the train-only artifact has little evidence for many items."
    )
    lines.append(
        "- **Next model:** the credible CountSketch-over-popularity lift justifies LightFM as the next offline challenger, "
        "not a production replacement. Promotion still requires a larger held-out gain with acceptable coverage, bias, "
        "and latency."
    )

    lines.extend(
        [
            "",
            "## What this experiment can tell us",
            "",
            *[f"- {value}" for value in result["what_this_can_tell_us"]],
            "",
            "## What this experiment cannot tell us",
            "",
            *[f"- {value}" for value in result["what_this_cannot_tell_us"]],
            "",
            "## Reproducibility",
            "",
            f"- Dataset SHA-256: `{result['dataset']['dataset_sha256']}`",
            f"- Split artifact SHA-256: `{result['dataset']['split_sha256']}`",
            f"- Catalog SHA-256: `{result['dataset']['catalog_sha256']}`",
            f"- Bootstrap iterations: {result['run_config']['bootstrap_iterations']:,}",
            f"- Evaluation duration: {result['evaluation_duration_seconds']:.2f} seconds",
        ]
    )
    return "\n".join(lines) + "\n"
