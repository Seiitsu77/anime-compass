from __future__ import annotations

import csv
import gc
import gzip
import json
import os
import platform
import re
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
from backend.anime_agent.lightfm_serving import LightFMServingIndex
from backend.anime_agent.recommender import AnimeRecommender

from .collaborative_baselines import (
    ALSCollaborativeAdapter,
    ALSModel,
    ItemItemModel,
    OracleModel,
    RandomModel,
    build_als_artifact_from_split,
    build_item_item_artifact_from_split,
)
from .fusion import load_fusion_weights
from .metrics import (
    build_item_popularity_buckets,
    intra_list_diversity,
    item_novelty,
    ndcg_at_k,
    normalized_log_popularity,
    paired_bootstrap_aligned,
    ranking_metrics,
    recall_at_k,
    recommendation_popularity_concentration,
    user_activity_segment,
)
from .models import (
    CountSketchModel,
    CurrentHybridModel,
    LightFMModel,
    OfflineEvaluationModel,
    PopularityModel,
    build_countsketch_artifact_from_split,
    composite_artifact_hash,
    compute_train_statistics,
    countsketch_artifact_matches,
    sanitize_catalog_with_training_statistics,
    save_popularity_artifact,
)
from .split import METHODOLOGY_NOTE, SplitStore, UserSplit, select_evaluation_sample, sha256_file


@dataclass(frozen=True)
class EvaluationRunConfig:
    sample_seed: int = 42
    sampling_strategy: str = "uniform"
    max_evaluation_users: int | None = 100
    users_per_stratum: int | None = None
    model_names: tuple[str, ...] = ("popularity", "countsketch_cf", "current_hybrid")
    recommendation_k: int = 20
    bootstrap_iterations: int = 2_000
    countsketch_projections: int = 3
    countsketch_width: int = 128
    item_item_neighbors: int = 200
    fusion_weights_path: str | None = None
    semantic_artifact_path: str | None = None
    # User IDs any earlier experiment already inspected. Excluding them is what
    # makes a confirmation run genuinely held out after exploratory work.
    excluded_user_ids: tuple[int, ...] = ()
    als_factors: int = 64
    als_iterations: int = 15
    als_regularization: float = 0.05
    als_alpha: float = 40.0
    force_model_rebuild: bool = False
    progress_every: int = 25
    lightfm_penalties: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.recommendation_k < 20:
            raise ValueError("recommendation_k must be at least 20 for the requested metrics")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        allowed_sampling = {"uniform", "stratified", "activity_stratified", "popularity_stratified"}
        if self.sampling_strategy not in allowed_sampling:
            raise ValueError(f"sampling_strategy must be one of {sorted(allowed_sampling)}")
        if self.users_per_stratum is not None and self.users_per_stratum < 1:
            raise ValueError("users_per_stratum must be positive when provided")
        if self.item_item_neighbors < 1:
            raise ValueError("item_item_neighbors must be positive")
        if self.als_factors < 1 or self.als_iterations < 1:
            raise ValueError("als_factors and als_iterations must be positive")
        if self.als_regularization < 0.0 or self.als_alpha <= 0.0:
            raise ValueError("als_regularization must be non-negative and als_alpha positive")
        allowed_models = {
            "popularity",
            "random",
            "current_hybrid_als",
            "oracle",
            "countsketch_cf",
            "current_hybrid",
            "current_hybrid_learned",
            "item_item_cosine",
            "als",
        }
        names = tuple(self.model_names)
        invalid = [
            name for name in names if name not in allowed_models and re.fullmatch(r"lightfm_[a-z0-9_]+", name) is None
        ]
        if not names or len(names) != len(set(names)) or invalid:
            raise ValueError("model_names must be unique built-ins or names matching lightfm_[a-z0-9_]+")
        penalties = tuple((str(name), float(value)) for name, value in self.lightfm_penalties)
        if len(penalties) != len({name for name, _value in penalties}):
            raise ValueError("LightFM popularity penalties must have unique model names")
        if any(name not in names or not name.startswith("lightfm_") or value < 0.0 for name, value in penalties):
            raise ValueError("LightFM penalties require a configured LightFM model and a non-negative lambda")
        object.__setattr__(self, "model_names", names)
        object.__setattr__(self, "lightfm_penalties", penalties)


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

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # type: ignore[attr-defined]
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
    lightfm_artifacts: Mapping[str, Path] | None = None,
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
    countsketch.offline_peak_process_rss_bytes = countsketch_process_peak

    build_details: dict[str, Any] = {
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
        "lightfm": {},
    }
    models_by_name: dict[str, OfflineEvaluationModel] = {
        "popularity": popularity,
        "countsketch_cf": countsketch,
    }

    # Reference points: a floor and an analytic ceiling. Neither is deployable;
    # they exist so every other number has a known scale and so a metric bug
    # shows up as an implausible random or sub-ceiling oracle score.
    if "random" in config.model_names:
        models_by_name["random"] = RandomModel(catalog_ids, seed=config.sample_seed)
    if "oracle" in config.model_names:
        models_by_name["oracle"] = OracleModel(catalog_ids, holdout="test")

    semantic_index = None
    if config.semantic_artifact_path and {"current_hybrid", "current_hybrid_learned"} & set(config.model_names):
        # Semantic vectors come from synopsis text, not ratings, so including
        # them carries no held-out leakage -- the same argument that already
        # justifies the TF-IDF and LSA channels.
        semantic_index = _load_semantic_index_for_evaluation(Path(config.semantic_artifact_path), catalog)
        build_details["semantic"] = {
            "artifact_path": config.semantic_artifact_path,
            "available": semantic_index is not None,
        }

    if "current_hybrid" in config.model_names:

        def build_hybrid() -> CurrentHybridModel:
            evaluation_catalog = sanitize_catalog_with_training_statistics(
                catalog,
                statistics,
                collaborative_index,
            )
            recommender = AnimeRecommender(
                evaluation_catalog,
                collaborative_index=collaborative_index,
                semantic_index=semantic_index,
            )
            return CurrentHybridModel(
                recommender,
                build_duration_seconds=0.0,
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
        models_by_name["current_hybrid"] = hybrid
        build_details["hybrid"] = {
            "process_rss_before_bytes": hybrid_rss_before,
            "process_rss_after_bytes": hybrid_rss_after,
            "process_rss_delta_bytes": (
                max(0, hybrid_rss_after - hybrid_rss_before)
                if hybrid_rss_before is not None and hybrid_rss_after is not None
                else None
            ),
            "measurement_note": "RSS delta is used because tracemalloc materially distorts hybrid initialization.",
        }

    if "current_hybrid_als" in config.model_names:
        substitution_als = ALSModel(
            artifacts_dir / "als_train_only.npz",
            catalog_ids,
            build_duration_seconds=0.0,
        )
        if substitution_als.metadata.get("split_sha256") != split_sha256:
            raise ValueError("The ALS artifact was trained from a different personalized split")

        def build_hybrid_als() -> CurrentHybridModel:
            # The catalog is sanitized with the SAME CountSketch index the
            # baseline hybrid uses, and Bayesian quality statistics still come
            # from that index, so the quality channel is byte-identical across
            # both arms. The only substantive change is the collaborative
            # similarity signal.
            evaluation_catalog = sanitize_catalog_with_training_statistics(
                catalog,
                statistics,
                collaborative_index,
            )
            recommender = AnimeRecommender(
                evaluation_catalog,
                collaborative_index=ALSCollaborativeAdapter(substitution_als, quality_source=collaborative_index),
                semantic_index=semantic_index,
            )
            return CurrentHybridModel(
                recommender,
                build_duration_seconds=0.0,
                artifact_path=artifacts_dir / "als_train_only.npz",
                catalog_artifact_path=catalog_path,
                name="current_hybrid_als",
            )

        hybrid_als, hybrid_als_seconds, _peak = _measure_build(build_hybrid_als, trace_allocations=False)
        hybrid_als.build_duration_seconds = hybrid_als_seconds
        models_by_name["current_hybrid_als"] = hybrid_als
        build_details["current_hybrid_als"] = {
            "collaborative_channel": "implicit ALS (substituted for CountSketch)",
            "als_metadata": substitution_als.metadata,
        }

    if "current_hybrid_learned" in config.model_names:
        if config.fusion_weights_path is None:
            raise ValueError("current_hybrid_learned requires --fusion-weights")
        learned_weights = load_fusion_weights(Path(config.fusion_weights_path))

        def build_learned_hybrid() -> CurrentHybridModel:
            evaluation_catalog = sanitize_catalog_with_training_statistics(
                catalog,
                statistics,
                collaborative_index,
            )
            recommender = AnimeRecommender(
                evaluation_catalog,
                weights=dict(learned_weights),
                collaborative_index=collaborative_index,
                semantic_index=semantic_index,
            )
            return CurrentHybridModel(
                recommender,
                build_duration_seconds=0.0,
                artifact_path=countsketch_path,
                catalog_artifact_path=catalog_path,
                name="current_hybrid_learned",
                weights=learned_weights,
            )

        learned_hybrid, learned_seconds, _learned_peak = _measure_build(
            build_learned_hybrid,
            trace_allocations=False,
        )
        learned_hybrid.build_duration_seconds = learned_seconds
        models_by_name["current_hybrid_learned"] = learned_hybrid
        build_details["current_hybrid_learned"] = {
            "fusion_weights_path": str(config.fusion_weights_path),
            "weights": dict(learned_weights),
        }

    if "item_item_cosine" in config.model_names:
        item_item_path = artifacts_dir / "item_item_train_only.npz"
        item_item_rss_before = _current_process_rss_bytes()
        if config.force_model_rebuild or not item_item_path.exists():
            item_item_metadata, item_item_seconds, item_item_peak = _measure_build(
                lambda: build_item_item_artifact_from_split(
                    store,
                    catalog,
                    item_item_path,
                    neighbors=config.item_item_neighbors,
                )
            )
        else:
            with np.load(item_item_path, allow_pickle=False) as artifact:
                item_item_metadata = json.loads(str(artifact["metadata_json"].item()))
            item_item_seconds = float(item_item_metadata.get("build_duration_seconds", 0.0))
            item_item_peak = None
        if item_item_metadata.get("split_sha256") != split_sha256:
            raise ValueError("The item-item artifact was trained from a different personalized split")
        item_item = ItemItemModel(item_item_path, catalog_ids, build_duration_seconds=item_item_seconds)
        item_item.offline_peak_process_rss_bytes = _peak_process_rss_bytes()
        models_by_name["item_item_cosine"] = item_item
        build_details["item_item_cosine"] = {
            "metadata": item_item_metadata,
            "python_tracemalloc_peak_bytes": item_item_peak,
            "process_rss_before_bytes": item_item_rss_before,
            "process_rss_after_bytes": _current_process_rss_bytes(),
        }

    if "als" in config.model_names:
        als_path = artifacts_dir / "als_train_only.npz"
        als_rss_before = _current_process_rss_bytes()
        if config.force_model_rebuild or not als_path.exists():
            als_metadata, als_seconds, als_peak = _measure_build(
                lambda: build_als_artifact_from_split(
                    store,
                    catalog,
                    als_path,
                    factors=config.als_factors,
                    iterations=config.als_iterations,
                    regularization=config.als_regularization,
                    alpha=config.als_alpha,
                    seed=config.sample_seed,
                )
            )
        else:
            with np.load(als_path, allow_pickle=False) as artifact:
                als_metadata = json.loads(str(artifact["metadata_json"].item()))
            als_seconds = float(als_metadata.get("build_duration_seconds", 0.0))
            als_peak = None
        if als_metadata.get("split_sha256") != split_sha256:
            raise ValueError("The ALS artifact was trained from a different personalized split")
        als = ALSModel(als_path, catalog_ids, build_duration_seconds=als_seconds)
        als.offline_peak_process_rss_bytes = _peak_process_rss_bytes()
        models_by_name["als"] = als
        build_details["als"] = {
            "metadata": als_metadata,
            "python_tracemalloc_peak_bytes": als_peak,
            "process_rss_before_bytes": als_rss_before,
            "process_rss_after_bytes": _current_process_rss_bytes(),
        }

    configured_lightfm_paths = dict(lightfm_artifacts or {})
    lightfm_penalties = dict(config.lightfm_penalties)
    train_positive_counts = statistics.positive_counts_by_id()
    for name in (value for value in config.model_names if value.startswith("lightfm_")):
        artifact_path = Path(configured_lightfm_paths.get(name, artifacts_dir / "lightfm" / f"{name}.npz"))
        load_started = time.perf_counter()
        index = LightFMServingIndex.load(artifact_path, catalog)
        load_seconds = time.perf_counter() - load_started
        if index.metadata.get("split_sha256") != split_sha256:
            raise ValueError(f"{name} artifact was trained from a different personalized split")
        model = LightFMModel(
            index,
            name=name,
            artifact_path=artifact_path,
            train_positive_counts=train_positive_counts,
            popularity_penalty_lambda=lightfm_penalties.get(name, 0.0),
        )
        models_by_name[name] = model
        build_details["lightfm"][name] = {
            "artifact_load_duration_seconds": load_seconds,
            "training_duration_seconds": model.build_duration_seconds,
            "total_tuning_duration_seconds": model.config["total_tuning_duration_seconds"],
            "peak_training_process_rss_bytes": index.metadata.get("peak_process_rss_bytes"),
            "numpy_score_roundtrip_max_abs_error": index.metadata.get("numpy_score_roundtrip_max_abs_error"),
            "popularity_penalty_lambda": lightfm_penalties.get(name, 0.0),
        }

    return (
        [models_by_name[name] for name in config.model_names],
        build_details,
        train_positive_counts,
    )


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
    exposure_by_item: Counter[int] = Counter()
    exposed_ids: set[int] = set()
    novelty_total = 0.0
    recommendation_count = 0
    popularity_bias_total = 0.0
    popularity_bias_users = 0
    recommended_normalized_popularity_total = 0.0
    profile_normalized_popularity_total = 0.0
    recommended_raw_popularity_total = 0.0
    profile_raw_popularity_total = 0.0
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
            metric_totals[name] += float(row[name])  # type: ignore[arg-type]

        segment = str(row["user_segment"])
        segment_totals[segment]["users"] += 1
        for name in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10"):
            segment_totals[segment][name] += float(row[name])  # type: ignore[arg-type]

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
            exposure_by_item[anime_id] += 1
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
            recommendation_raw_popularity = sum(
                int(train_positive_counts.get(anime_id, 0)) for anime_id in ranking
            ) / len(ranking)
            history_raw_popularity = sum(int(train_positive_counts.get(anime_id, 0)) for anime_id in history) / len(
                history
            )
            popularity_bias_total += recommendation_popularity - history_popularity
            recommended_normalized_popularity_total += recommendation_popularity
            profile_normalized_popularity_total += history_popularity
            recommended_raw_popularity_total += recommendation_raw_popularity
            profile_raw_popularity_total += history_raw_popularity
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
    concentration = recommendation_popularity_concentration(
        exposure_by_item,
        catalog_ids,
        train_positive_counts,
    )
    summary: dict[str, Any] = {
        "model": model.name,
        "model_version": model.version,
        "evaluated_users": evaluated_users,
        **aggregate,
        "catalog_coverage": len(exposed_ids) / len(catalog_ids) if catalog_ids else 0.0,
        "novelty_bits": novelty_total / recommendation_count if recommendation_count else 0.0,
        "popularity_bias": (popularity_bias_total / popularity_bias_users if popularity_bias_users else 0.0),
        "recommended_normalized_popularity": (
            recommended_normalized_popularity_total / popularity_bias_users if popularity_bias_users else 0.0
        ),
        "profile_normalized_popularity": (
            profile_normalized_popularity_total / popularity_bias_users if popularity_bias_users else 0.0
        ),
        "recommended_training_popularity_count": (
            recommended_raw_popularity_total / popularity_bias_users if popularity_bias_users else 0.0
        ),
        "profile_training_popularity_count": (
            profile_raw_popularity_total / popularity_bias_users if popularity_bias_users else 0.0
        ),
        "popularity_concentration": concentration,
        "intra_list_diversity": diversity_total / evaluated_users if evaluated_users else 0.0,
        "recommendation_exposure": {
            bucket: exposure_counts.get(bucket, 0) / total_exposure if total_exposure else 0.0
            for bucket in ("head", "mid_tail", "long_tail", "unknown")
        },
        "user_segments": user_segments,
        "heldout_item_popularity": heldout_item_popularity,
        "engineering": {
            "build_duration_seconds": model.build_duration_seconds,
            "offline_training_duration_seconds": float(
                getattr(model, "offline_training_duration_seconds", model.build_duration_seconds)
            ),
            "offline_peak_process_rss_bytes": getattr(model, "offline_peak_process_rss_bytes", None),
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


def _load_semantic_index_for_evaluation(artifact_path: Path, catalog: list[dict[str, Any]]) -> Any:
    """Load the optional semantic index, or return None if it is unusable.

    The channel is optional by design: the recommender renormalises weights
    across active channels, so a missing artifact degrades the blend rather than
    failing the run. A stale or mismatched artifact is treated the same way,
    because silently scoring against the wrong catalog would be worse.
    """
    try:
        from app.core.config import get_settings
        from app.embeddings.index import SemanticEmbeddingIndex
        from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

        settings = get_settings()
        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            model_revision=settings.embedding_model_revision,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            local_files_only=True,
        )
        return SemanticEmbeddingIndex.load(
            artifact_path,
            provider,
            catalog,
            expected_dimension=settings.embedding_dimensions,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "channel unavailable"
        print(f"semantic: unavailable ({type(exc).__name__}); continuing without the channel")
        return None


def _bootstrap_comparisons(
    metrics_by_model: Mapping[str, AlignedPrimaryMetrics],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    lightfm_names = [name for name in metrics_by_model if name.startswith("lightfm_")]
    candidates: list[tuple[str, str]] = []
    if "countsketch_cf" in metrics_by_model and "popularity" in metrics_by_model:
        candidates.append(("countsketch_cf", "popularity"))
    if "countsketch_cf" in metrics_by_model:
        candidates.extend((name, "countsketch_cf") for name in lightfm_names)
    if len(lightfm_names) > 1:
        reference = lightfm_names[0]
        candidates.extend((name, reference) for name in lightfm_names[1:])
    if "current_hybrid" in metrics_by_model and "countsketch_cf" in metrics_by_model:
        candidates.append(("current_hybrid", "countsketch_cf"))
    if "current_hybrid" in metrics_by_model and "popularity" in metrics_by_model:
        candidates.append(("current_hybrid", "popularity"))
    # Reference collaborative baselines are compared against the production
    # channel, and the exact/sketched pair isolates the projection's cost.
    if "countsketch_cf" in metrics_by_model:
        candidates.extend((name, "countsketch_cf") for name in ("item_item_cosine", "als") if name in metrics_by_model)
    if "als" in metrics_by_model:
        candidates.extend((name, "als") for name in lightfm_names)
    # A learned blend is only interesting relative to the hand-set blend it replaces.
    if "current_hybrid_learned" in metrics_by_model and "current_hybrid" in metrics_by_model:
        candidates.append(("current_hybrid_learned", "current_hybrid"))
    comparisons = tuple(
        (left, right) for left, right in candidates if left in metrics_by_model and right in metrics_by_model
    )
    result: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_metrics = metrics_by_model[left]
        right_metrics = metrics_by_model[right]
        if not np.array_equal(left_metrics.user_ids, right_metrics.user_ids):
            raise RuntimeError(f"Cannot pair {left} and {right}: evaluated user IDs differ")
        for metric in ("ndcg_at_10", "recall_at_10"):
            right_values = getattr(right_metrics, metric)
            interval = paired_bootstrap_aligned(
                getattr(left_metrics, metric) - right_values,
                iterations=iterations,
                seed=seed,
            )
            baseline = float(np.mean(right_values))
            result.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    **interval,
                    "relative_difference": (
                        float(interval["difference"]) / baseline if abs(baseline) > 1e-12 else None
                    ),
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
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def _dependency_inclusive_build_seconds(result: Mapping[str, Any], model_name: str) -> float:
    models = _model_by_name(result)
    statistics_seconds = float(result["build_details"]["training_statistics"]["duration_seconds"])
    if model_name == "popularity":
        return statistics_seconds
    if model_name.startswith("lightfm_"):
        engineering = models[model_name]["engineering"]
        return float(
            engineering.get(
                "offline_training_duration_seconds",
                models[model_name]
                .get("model_config", {})
                .get(
                    "total_tuning_duration_seconds",
                    engineering["build_duration_seconds"],
                ),
            )
        )
    countsketch_seconds = float(
        models.get("countsketch_cf", {})
        .get("engineering", {})
        .get(
            "build_duration_seconds",
            result.get("build_details", {})
            .get("countsketch", {})
            .get("metadata", {})
            .get("build_duration_seconds", 0.0),
        )
    )
    if model_name == "countsketch_cf":
        return countsketch_seconds
    if model_name == "current_hybrid":
        return (
            statistics_seconds
            + countsketch_seconds
            + float(models["current_hybrid"]["engineering"]["build_duration_seconds"])
        )
    return float(models[model_name]["engineering"]["build_duration_seconds"])


def _offline_peak_rss_bytes(result: Mapping[str, Any], summary: Mapping[str, Any]) -> int | None:
    engineering = summary["engineering"]
    explicit = engineering.get("offline_peak_process_rss_bytes")
    if explicit is not None:
        return int(explicit)
    name = str(summary["model"])
    build_details = result.get("build_details", {})
    if name.startswith("lightfm_"):
        value = build_details.get("lightfm", {}).get(name, {}).get("peak_training_process_rss_bytes")
    elif name == "countsketch_cf":
        value = build_details.get("countsketch", {}).get("process_peak_rss_after_bytes")
    else:
        value = None
    return int(value) if value is not None else None


def _write_slice_csvs(output_dir: Path, result: Mapping[str, Any]) -> None:
    summaries = result["models"]
    with (output_dir / "user_segments.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["model", "segment", "users", "ndcg_at_10", "recall_at_10", "hit_rate_at_10"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
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
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
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

    with (output_dir / "popularity_concentration.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "model",
            "top_1_percent_share",
            "top_5_percent_share",
            "top_10_percent_share",
            "top_20_percent_share",
            "unique_recommended_items",
            "exposure_gini",
            "catalog_coverage",
            "average_training_popularity_count",
            "average_normalized_log_popularity",
            "recommended_training_popularity_count",
            "profile_training_popularity_count",
            "recommended_normalized_popularity",
            "profile_normalized_popularity",
            "popularity_bias",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            concentration = summary["popularity_concentration"]
            writer.writerow(
                {
                    "model": summary["model"],
                    **{field: concentration[field] for field in fields[1:10]},
                    **{field: summary[field] for field in fields[10:]},
                }
            )

    with (output_dir / "engineering.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "model",
            "build_duration_seconds",
            "offline_training_duration_seconds",
            "dependency_inclusive_build_duration_seconds",
            "offline_peak_process_rss_bytes",
            "inference_latency_p50_ms",
            "inference_latency_p95_ms",
            "serialization_latency_p50_ms",
            "serialization_latency_p95_ms",
            "resident_array_bytes",
            "artifact_size_bytes",
            "artifact_sha256",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            engineering = summary["engineering"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "build_duration_seconds": engineering["build_duration_seconds"],
                    "offline_training_duration_seconds": engineering.get(
                        "offline_training_duration_seconds",
                        summary.get("model_config", {}).get(
                            "total_tuning_duration_seconds",
                            engineering["build_duration_seconds"],
                        ),
                    ),
                    "dependency_inclusive_build_duration_seconds": _dependency_inclusive_build_seconds(
                        result, str(summary["model"])
                    ),
                    "offline_peak_process_rss_bytes": _offline_peak_rss_bytes(result, summary),
                    **{field: engineering[field] for field in fields[5:10]},
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
        "relative_difference",
        "ci_lower",
        "ci_upper",
        "ci_excludes_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({field: comparison.get(field) for field in fields})


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


def _channel_activity(model_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Report which blend channels carried weight, for models that have a blend.

    A channel can be present, wired, and configured yet contribute nothing --
    either because its weight is zero or because its artifact never loaded. That
    is exactly how the semantic channel stayed dead across every published
    benchmark, so the state is recorded per run rather than inferred later.
    """
    weights = model_config.get("weights")
    if not isinstance(weights, Mapping):
        return None
    weighted = {channel: float(value) for channel, value in weights.items() if float(value) > 0.0}
    zero_weight = sorted(channel for channel, value in weights.items() if float(value) <= 0.0)
    return {
        "weight_source": model_config.get("weight_source", "hand_set"),
        "weighted_channels": dict(sorted(weighted.items())),
        "zero_weight_channels": zero_weight,
        "semantic_artifact_loaded": model_config.get("semantic_embedding_available"),
    }


def _score_variance(metrics: AlignedPrimaryMetrics | None) -> dict[str, Any] | None:
    """Per-user spread of the primary metrics.

    A mean alone hides whether a model is uniformly mediocre or bimodal, and a
    near-zero variance is a signal that a component is inert rather than
    balanced.
    """
    if metrics is None:
        return None
    summary: dict[str, Any] = {"users": int(len(metrics.user_ids))}
    for name in ("ndcg_at_10", "recall_at_10"):
        values = np.asarray(getattr(metrics, name), dtype=np.float64)
        if not len(values):
            continue
        summary[name] = {
            "mean": float(values.mean()),
            "variance": float(values.var(ddof=1)) if len(values) > 1 else 0.0,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "zero_fraction": float(np.mean(values <= 0.0)),
        }
    return summary


def run_personalized_evaluation(
    store: SplitStore,
    catalog: list[dict[str, Any]],
    *,
    catalog_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    config: EvaluationRunConfig | None = None,
    lightfm_artifacts: Mapping[str, Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = config or EvaluationRunConfig()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = store.metadata()
    split_audit = store.audit_counts()
    eligible_segment_counts = store.eligible_segment_counts()

    if progress is not None:
        progress("models: preparing train-only artifacts")
    models, build_details, train_positive_counts = _prepare_models(
        store,
        catalog,
        catalog_path=catalog_path,
        artifacts_dir=artifacts_dir,
        config=config,
        lightfm_artifacts=lightfm_artifacts,
    )
    catalog_ids = {int(item["id"]) for item in catalog}
    genres_by_id = {int(item["id"]): list(item.get("genres") or []) for item in catalog}
    bucket_by_id = build_item_popularity_buckets(catalog_ids, train_positive_counts)
    sample = select_evaluation_sample(
        store,
        limit=config.max_evaluation_users,
        seed=config.sample_seed,
        strategy=config.sampling_strategy,
        users_per_stratum=config.users_per_stratum,
        bucket_by_id=bucket_by_id,
        excluded_user_ids=config.excluded_user_ids,
    )
    evaluation_user_ids = list(sample.user_ids)
    if not evaluation_user_ids:
        raise ValueError("No eligible users are available for evaluation")
    if progress is not None:
        progress(
            f"sample: selected {len(evaluation_user_ids):,} users using {sample.strategy}"
            + (" diagnostic sampling" if sample.diagnostic else " sampling")
        )

    summaries: list[dict[str, Any]] = []
    metrics_by_model: dict[str, AlignedPrimaryMetrics] = {}
    per_user_path = output_dir / "per_user_metrics.csv.gz"
    with gzip.open(per_user_path, "wt", encoding="utf-8", newline="") as per_user_file:
        row_writer = csv.DictWriter(per_user_file, fieldnames=PER_USER_FIELDS, lineterminator="\n")
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
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_scope": (
            "diagnostic" if sample.diagnostic else "full" if config.max_evaluation_users in (None, 0) else "sampled"
        ),
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
            "train_positive_density": split_metadata.get("train_positive_density"),
            "train_positive_matrix_sparsity": split_metadata.get("train_positive_matrix_sparsity"),
            "catalog_items": len(catalog),
            "split_audit": split_audit,
        },
        "evaluation_population": {
            "evaluated_users": len(evaluation_user_ids),
            "sampling": (
                "all eligible users"
                if not sample.diagnostic and config.max_evaluation_users in (None, 0)
                else f"deterministic {sample.strategy} hash sample"
            ),
            "sampling_strategy": sample.strategy,
            "diagnostic_sample": sample.diagnostic,
            "selection_target": sample.selection_target,
            "requested_total": sample.requested_total,
            "requested_per_stratum": sample.requested_per_stratum,
            "stratum_population_counts": sample.stratum_population_counts,
            "stratum_selected_counts": sample.stratum_selected_counts,
            "segment_counts": dict(sorted(sample_segment_counts.items())),
            "eligible_population_segment_counts": eligible_segment_counts,
            "aggregate_weighting": (
                "unweighted macro-average over all eligible users"
                if not sample.diagnostic and config.max_evaluation_users in (None, 0)
                else "diagnostic sample only; unweighted aggregate is not a population estimate"
                if sample.diagnostic
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
            "popularity_concentration": (
                "Recommendation-event share inside train-positive popularity top 1/5/10/20%; exposure Gini includes "
                "all catalog items, including zero-exposure items."
            ),
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
            "Whether LightFM collaborative challengers add value over CountSketch when included in the run.",
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
    _write_text_lf(results_path, json.dumps(result, indent=2, sort_keys=True))
    _write_summary_csv(output_dir / "summary.csv", summaries)
    _write_slice_csvs(output_dir, result)
    _write_bootstrap_csv(output_dir / "paired_bootstrap.csv", bootstrap)
    _write_text_lf(output_dir / "report.md", render_markdown_report(result))
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
            "excluded_user_count": len(config.excluded_user_ids),
            "sample_is_disjoint_from_excluded": not (set(evaluation_user_ids) & set(config.excluded_user_ids)),
            "run_config": result["run_config"],
            "environment": result["environment"],
        },
        "models": [
            {
                "name": summary["model"],
                "version": summary["model_version"],
                "hyperparameters": summary["model_config"],
                "selected_fit_or_build_duration_seconds": summary["engineering"]["build_duration_seconds"],
                "offline_training_duration_seconds": _dependency_inclusive_build_seconds(result, str(summary["model"])),
                "offline_peak_process_rss_bytes": _offline_peak_rss_bytes(result, summary),
                "training_or_build_duration_seconds": summary["engineering"]["build_duration_seconds"],
                "artifact": summary["engineering"]["artifact"],
                "channel_activity": _channel_activity(summary["model_config"]),
                "score_variance": _score_variance(metrics_by_model.get(str(summary["model"]))),
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
        "popularity_concentration.csv",
        "engineering.csv",
        "paired_bootstrap.csv",
        "per_user_metrics.csv.gz",
        "report.md",
    ):
        path = output_dir / name
        manifest["outputs"][name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    _write_text_lf(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if progress is not None:
        progress(f"outputs: wrote benchmark artifacts to {output_dir}")
    return result


def refresh_derived_outputs(output_dir: Path) -> None:
    """Regenerate normalized report/CSV views from completed results JSON."""
    output_dir = Path(output_dir)
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    dataset = result.get("dataset", {})
    legacy_density = float(dataset.get("train_positive_sparsity") or 0.0)
    dataset.setdefault("train_positive_density", legacy_density)
    dataset.setdefault("train_positive_matrix_sparsity", 1.0 - legacy_density)
    _write_text_lf(output_dir / "results.json", json.dumps(result, indent=2, sort_keys=True))
    summaries = result["models"]
    _write_summary_csv(output_dir / "summary.csv", summaries)
    _write_slice_csvs(output_dir, result)
    _write_bootstrap_csv(output_dir / "paired_bootstrap.csv", result["paired_bootstrap"])
    _write_text_lf(output_dir / "report.md", render_markdown_report(result))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Refreshing re-renders stored values; it never re-runs inference, so
    # per-user variance cannot be recomputed here. Carry the recorded values
    # forward instead of silently dropping them.
    recorded = {str(entry.get("name")): entry for entry in manifest.get("models", [])}
    manifest["models"] = [
        {
            "name": summary["model"],
            "version": summary["model_version"],
            "hyperparameters": summary["model_config"],
            "selected_fit_or_build_duration_seconds": summary["engineering"]["build_duration_seconds"],
            "offline_training_duration_seconds": _dependency_inclusive_build_seconds(result, str(summary["model"])),
            "offline_peak_process_rss_bytes": _offline_peak_rss_bytes(result, summary),
            "training_or_build_duration_seconds": summary["engineering"]["build_duration_seconds"],
            "artifact": summary["engineering"]["artifact"],
            "channel_activity": _channel_activity(summary["model_config"]),
            "score_variance": recorded.get(str(summary["model"]), {}).get("score_variance"),
        }
        for summary in summaries
    ]
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    _write_text_lf(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))


def _write_text_lf(path: Path, content: str) -> None:
    """Write deterministic UTF-8 text without platform newline translation."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


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
    model_order = tuple(str(model["model"]) for model in result["models"])
    labels = {
        "popularity": "Popularity",
        "countsketch_cf": "CountSketch CF",
        "current_hybrid": "Current Hybrid",
        "lightfm_id": "LightFM-ID",
        "lightfm_hybrid": "LightFM-Hybrid",
    }
    for name in model_order:
        labels.setdefault(name, "LightFM " + name.removeprefix("lightfm_").replace("_", " ").title())
    evaluation_population = result["evaluation_population"]
    sampling_strategy = str(
        evaluation_population.get("sampling_strategy")
        or result.get("run_config", {}).get("sampling_strategy", "uniform")
    )
    report_title = (
        "LightFM Offline Challenger Benchmark"
        if any(name.startswith("lightfm_") for name in model_order)
        else "Personalized Offline Recommendation Benchmark"
    )
    evaluation_label = {
        "uniform": "Evaluation A — representative uniform user sample",
        "stratified": "Evaluation B — activity-balanced diagnostic",
        "activity_stratified": "Evaluation B — activity-balanced diagnostic",
        "popularity_stratified": "Evaluation C — popularity-stratified diagnostic",
    }.get(sampling_strategy, sampling_strategy)
    lines = [
        f"# {report_title}",
        "",
        f"> {result['methodology_note']}",
        "",
        f"Evaluation: **{evaluation_label}**.",
        "",
        f"Run scope: **{result['run_scope']}**; evaluated users: **{evaluation_population['evaluated_users']:,}**; "
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

    lines.extend(
        [
            "",
            "## Recommendation popularity concentration",
            "",
            "Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes "
            "every catalog item, including items that receive zero recommendations.",
            "",
            "| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | "
            "Avg train count | Rec profile popularity | User profile popularity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in model_order:
        model = models[name]
        concentration = model["popularity_concentration"]
        lines.append(
            f"| {labels[name]} | {_fmt(concentration['top_1_percent_share'])} | "
            f"{_fmt(concentration['top_5_percent_share'])} | {_fmt(concentration['top_10_percent_share'])} | "
            f"{_fmt(concentration['top_20_percent_share'])} | "
            f"{int(concentration['unique_recommended_items']):,} | {_fmt(concentration['exposure_gini'])} | "
            f"{float(concentration['average_training_popularity_count']):.1f} | "
            f"{_fmt(model['recommended_normalized_popularity'])} | "
            f"{_fmt(model['profile_normalized_popularity'])} |"
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
            "The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not "
            "include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all "
            "validation-search candidates. Peak RSS is the trainer process peak where available.",
            "",
            "| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in model_order:
        engineering = models[name]["engineering"]
        peak_rss = _offline_peak_rss_bytes(result, models[name])
        peak_rss_label = f"{int(peak_rss) / 1024 / 1024:.2f} MiB" if peak_rss is not None else "n/a"
        lines.append(
            f"| {labels[name]} | {engineering['build_duration_seconds']:.2f}s | "
            f"{_dependency_inclusive_build_seconds(result, name):.2f}s | "
            f"{peak_rss_label} | "
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

    hybrid_stages = models["current_hybrid"]["engineering"]["stage_latency_ms"] if "current_hybrid" in models else {}
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

    lightfm_models = [name for name in model_order if name.startswith("lightfm_")]
    if lightfm_models:
        lines.extend(
            [
                "",
                "## Validation-only LightFM model selection",
                "",
                "The final configurations below were selected using validation positives only. Test positives were evaluated only "
                "after selection.",
                "",
                "| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for name in lightfm_models:
            config = models[name]["model_config"].get("selected_config", {})
            validation = models[name]["model_config"].get("selected_validation_metrics", {})
            lines.append(
                f"| {labels[name]} | {str(config.get('loss', 'unknown')).upper()} | "
                f"{int(config.get('no_components', 0))} | {int(config.get('epochs', 0))} | "
                f"{_fmt(validation.get('ndcg_at_10', 0.0))} | {_fmt(validation.get('recall_at_10', 0.0))} |"
            )

    lines.extend(["", "## Paired statistical comparisons", ""])
    comparison_pairs: list[tuple[str, str]] = []
    for row in result["paired_bootstrap"]:
        pair = (str(row["left_model"]), str(row["right_model"]))
        if pair not in comparison_pairs:
            comparison_pairs.append(pair)
    for left, right in comparison_pairs:
        lines.append(f"### {labels[left]} vs {labels[right]}")
        lines.append("")
        for metric in ("ndcg_at_10", "recall_at_10"):
            interval = bootstrap[(left, right, metric)]
            readable = "NDCG@10" if metric == "ndcg_at_10" else "Recall@10"
            relative = interval.get("relative_difference")
            relative_text = f"; relative delta {float(relative) * 100:+.2f}%" if relative is not None else ""
            lines.append(
                f"- Delta {readable}: {float(interval['difference']) * 100:+.2f} percentage points; "
                f"95% paired-bootstrap CI [{float(interval['ci_lower']) * 100:+.2f}, "
                f"{float(interval['ci_upper']) * 100:+.2f}] pp{relative_text}."
            )
        lines.append("")

    lines.extend(["## Interpretation", ""])
    diagnostic = bool(evaluation_population.get("diagnostic_sample"))
    sample_warning = result["run_scope"] != "full"
    if diagnostic:
        lines.append(
            "**Diagnostic sample:** its unweighted aggregate metrics are not estimates of whole-population performance. "
            "Use only the segment or popularity-stratum cuts that this evaluation was designed to inspect."
        )
        lines.append("")
    elif sample_warning:
        lines.append(
            "**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside "
            "this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population "
            "run is necessary.**"
        )
        lines.append("")
    if result["dataset"].get("source_user_limit") is not None:
        lines.append(
            "**Pipeline smoke only:** model artifacts were trained on a source-user prefix, so ranking values must "
            "not be compared with full-data experiments."
        )
        lines.append("")
    for left, right in comparison_pairs:
        interval = bootstrap[(left, right, "ndcg_at_10")]
        direction = "higher" if interval["difference"] > 0 else "lower" if interval["difference"] < 0 else "unchanged"
        evidence = (
            "excludes zero, providing evidence of a difference in this evaluation"
            if interval["ci_excludes_zero"]
            else "includes zero, so the observed difference is not statistically conclusive"
        )
        lines.append(
            f"- **{labels[left]} versus {labels[right]}:** NDCG@10 is {direction} by "
            f"{abs(float(interval['difference'])) * 100:.2f} pp; the 95% interval {evidence}."
        )
    fastest = min(model_order, key=lambda name: models[name]["engineering"]["inference_latency_p50_ms"])
    lines.append(
        f"- **Latency:** {labels[fastest]} has the lowest p50 in this run "
        f"({float(models[fastest]['engineering']['inference_latency_p50_ms']):.2f} ms)."
    )
    eligible_users = int(result["dataset"].get("users_after_filter") or 0)
    if sample_warning and eligible_users and "current_hybrid" in models:
        hybrid_latency = float(models["current_hybrid"]["engineering"]["inference_latency_p50_ms"])
        projected_hours = eligible_users * hybrid_latency / 1000 / 60 / 60
        lines.append(
            f"- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about "
            f"{projected_hours:.1f} hours for {eligible_users:,} eligible users. This is a planning estimate, "
            "not a measured full-run duration."
        )
    for segment in ("sparse", "medium", "heavy"):
        if int(models[model_order[0]]["user_segments"][segment]["users"]):
            best = max(model_order, key=lambda name: models[name]["user_segments"][segment]["ndcg_at_10"])
            value = float(models[best]["user_segments"][segment]["ndcg_at_10"])
            lines.append(
                f"- **{segment.title()} users:** {labels[best]} has the highest sampled NDCG@10 ({value:.4f})."
            )
    lines.append(
        "- **Popularity/long tail:** inspect exposure together with held-out long-tail Recall@10; a model can appear "
        "novel simply because the train-only artifact has little evidence for many items."
    )
    if lightfm_models:
        lines.append(
            "- **Promotion gate:** LightFM remains an offline challenger. Replacement requires a positive paired interval, "
            "roughly 5% relative NDCG@10 lift, no material Recall/coverage/diversity regression, and acceptable serving cost."
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
