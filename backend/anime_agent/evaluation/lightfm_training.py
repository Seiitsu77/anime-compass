from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import sparse

from backend.anime_agent.lightfm_serving import LIGHTFM_ARTIFACT_VERSION

from .metrics import (
    catalog_coverage,
    intra_list_diversity,
    item_novelty,
    ndcg_at_k,
    normalized_log_popularity,
    ranking_metrics,
    recall_at_k,
)
from .split import SplitStore, UserSplit, catalog_ids_sha256, select_evaluation_user_ids, sha256_file

LIGHTFM_FEATURE_SCHEMA_VERSION = 1
FORBIDDEN_METADATA_FIELDS = {
    "archive_score",
    "collaborative_available",
    "favorites",
    "members",
    "popularity",
    "rank",
    "rating_count",
    "score",
    "score_distribution",
    "watching_stats",
}
DEFAULT_INCLUDED_METADATA_FIELDS = (
    "genres",
    "type",
    "source",
    "studios",
    "start_year",
    "content_rating",
)
_TOKEN_SAFE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class LightFMCandidateConfig:
    loss: str
    no_components: int = 32
    learning_rate: float = 0.05
    epochs: int = 10
    item_alpha: float = 1e-6
    user_alpha: float = 1e-6
    max_sampled: int = 10

    def __post_init__(self) -> None:
        if self.loss not in {"warp", "bpr"}:
            raise ValueError("LightFM loss must be 'warp' or 'bpr'")
        if self.no_components < 1:
            raise ValueError("no_components must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.item_alpha < 0 or self.user_alpha < 0:
            raise ValueError("LightFM regularization values cannot be negative")
        if self.max_sampled < 1:
            raise ValueError("max_sampled must be positive")

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class LightFMSearchConfig:
    candidates: tuple[LightFMCandidateConfig, ...]
    validation_users: int = 300
    seed: int = 42
    num_threads: int = 1
    studio_min_frequency: int = 5
    require_both_losses: bool = True

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("LightFM search needs at least one candidate")
        if self.require_both_losses and {candidate.loss for candidate in self.candidates} != {"warp", "bpr"}:
            raise ValueError("LightFM search must include both WARP and BPR candidates")
        if self.validation_users < 1:
            raise ValueError("validation_users must be positive")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        if self.studio_min_frequency < 1:
            raise ValueError("studio_min_frequency must be positive")


@dataclass(frozen=True)
class LightFMTrainingData:
    user_ids: np.ndarray
    anime_ids: np.ndarray
    interactions: sparse.coo_matrix

    @property
    def positive_edges(self) -> int:
        return int(self.interactions.nnz)


@dataclass(frozen=True)
class LightFMItemFeatures:
    matrix: sparse.csr_matrix
    feature_names: tuple[str, ...]
    summary: dict[str, Any]


def default_search_candidates(profile: str = "standard") -> tuple[LightFMCandidateConfig, ...]:
    if profile == "smoke":
        return (
            LightFMCandidateConfig(loss="warp", no_components=16, epochs=3),
            LightFMCandidateConfig(loss="bpr", no_components=16, epochs=3),
        )
    if profile != "standard":
        raise ValueError("search profile must be 'smoke' or 'standard'")
    return (
        LightFMCandidateConfig(loss="warp", no_components=32, learning_rate=0.05, epochs=10),
        LightFMCandidateConfig(loss="warp", no_components=64, learning_rate=0.03, epochs=15),
        LightFMCandidateConfig(loss="bpr", no_components=32, learning_rate=0.05, epochs=10),
        LightFMCandidateConfig(loss="bpr", no_components=64, learning_rate=0.03, epochs=15),
    )


def _normalize_feature_value(value: Any) -> str:
    normalized = _TOKEN_SAFE.sub("-", str(value).casefold()).strip("-")
    return normalized or "unknown"


def _decade_token(value: Any) -> str:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return "decade:unknown"
    if year < 1900 or year > 2100:
        return "decade:unknown"
    return f"decade:{(year // 10) * 10}s"


def _studio_vocabulary(catalog: Sequence[Mapping[str, Any]], minimum_frequency: int) -> set[str]:
    counts: Counter[str] = Counter()
    for item in catalog:
        counts.update(
            _normalize_feature_value(value)
            for value in (item.get("studios") or [])
            if isinstance(value, str) and value.strip()
        )
    return {value for value, count in counts.items() if count >= minimum_frequency}


def lightfm_metadata_tokens(item: Mapping[str, Any], *, allowed_studios: set[str]) -> tuple[str, ...]:
    """Create static catalog-only feature tokens for one anime.

    Outcome-derived fields are intentionally never read here.  The identity
    feature is added separately by ``build_lightfm_item_features``.
    """
    tokens: set[str] = set()
    for genre in item.get("genres") or []:
        if isinstance(genre, str) and genre.strip():
            tokens.add(f"genre:{_normalize_feature_value(genre)}")
    tokens.add(f"type:{_normalize_feature_value(item.get('type') or 'unknown')}")
    tokens.add(f"source:{_normalize_feature_value(item.get('source') or 'unknown')}")
    tokens.add(f"content-rating:{_normalize_feature_value(item.get('content_rating') or 'unknown')}")
    tokens.add(_decade_token(item.get("start_year")))
    kept_studio = False
    for studio in item.get("studios") or []:
        if not isinstance(studio, str) or not studio.strip():
            continue
        normalized = _normalize_feature_value(studio)
        if normalized in allowed_studios:
            tokens.add(f"studio:{normalized}")
            kept_studio = True
    if not kept_studio:
        tokens.add("studio:unknown-or-rare")
    return tuple(sorted(tokens))


def build_lightfm_item_features(
    catalog: Sequence[Mapping[str, Any]],
    *,
    studio_min_frequency: int = 5,
) -> LightFMItemFeatures:
    """Build an L1-normalized sparse identity + metadata matrix."""
    if studio_min_frequency < 1:
        raise ValueError("studio_min_frequency must be positive")
    ordered = sorted((dict(item) for item in catalog), key=lambda item: int(item["id"]))
    anime_ids = [int(item["id"]) for item in ordered]
    if len(anime_ids) != len(set(anime_ids)):
        raise ValueError("Catalog anime IDs must be unique")
    allowed_studios = _studio_vocabulary(ordered, studio_min_frequency)
    metadata_by_row = [lightfm_metadata_tokens(item, allowed_studios=allowed_studios) for item in ordered]
    metadata_vocabulary = sorted({token for tokens in metadata_by_row for token in tokens})
    identity_names = [f"identity:{anime_id}" for anime_id in anime_ids]
    feature_names = tuple([*identity_names, *metadata_vocabulary])
    feature_index = {name: index for index, name in enumerate(feature_names)}

    indices: list[int] = []
    values: list[float] = []
    indptr = [0]
    for anime_id, metadata_tokens in zip(anime_ids, metadata_by_row, strict=True):
        row_tokens = (f"identity:{anime_id}", *metadata_tokens)
        weight = 1.0 / len(row_tokens)
        indices.extend(feature_index[token] for token in row_tokens)
        values.extend(weight for _token in row_tokens)
        indptr.append(len(indices))
    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(anime_ids), len(feature_names)),
        dtype=np.float32,
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("LightFM item features are not L1-normalized")

    completeness: dict[str, float] = {}
    for field in DEFAULT_INCLUDED_METADATA_FIELDS:
        present = sum(item.get(field) not in (None, "", [], {}) for item in ordered)
        completeness[field] = present / len(ordered) if ordered else 0.0
    summary = {
        "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
        "included_fields": list(DEFAULT_INCLUDED_METADATA_FIELDS),
        "excluded_outcome_fields": sorted(FORBIDDEN_METADATA_FIELDS),
        "feature_count": len(feature_names),
        "identity_feature_count": len(identity_names),
        "metadata_feature_count": len(metadata_vocabulary),
        "nonzero_assignments": int(matrix.nnz),
        "studio_min_frequency": studio_min_frequency,
        "eligible_studio_count": len(allowed_studios),
        "field_completeness": completeness,
        "normalization": "per-item L1; identity and static metadata are equally weighted tokens",
        "feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
    }
    return LightFMItemFeatures(matrix=matrix, feature_names=feature_names, summary=summary)


def build_lightfm_training_data(
    store: SplitStore,
    catalog_ids: Sequence[int],
    *,
    progress: Callable[[str], None] | None = None,
) -> LightFMTrainingData:
    """Build a bounded-memory positive-only COO matrix from train edges."""
    anime_ids = np.asarray(sorted({int(value) for value in catalog_ids}), dtype=np.int64)
    user_ids = np.asarray(store.training_user_ids(), dtype=np.int64)
    if not len(anime_ids) or not len(user_ids):
        raise ValueError("LightFM training requires non-empty user and item mappings")
    metadata = store.metadata()
    expected_edges = int(metadata["train_positive_interactions"])
    row_indexes = np.empty(expected_edges, dtype=np.int32)
    column_indexes = np.empty(expected_edges, dtype=np.int32)
    cursor = 0
    for row, user in enumerate(store.iter_users_by_ids(user_ids.tolist())):
        ids = np.asarray(user.train_positive_ids, dtype=np.int64)
        columns = np.searchsorted(anime_ids, ids)
        if np.any(columns >= len(anime_ids)) or np.any(anime_ids[columns] != ids):
            raise ValueError(f"User {user.user_id} contains an anime outside the LightFM catalog mapping")
        end = cursor + len(columns)
        row_indexes[cursor:end] = row
        column_indexes[cursor:end] = columns.astype(np.int32, copy=False)
        cursor = end
        if progress is not None and (row + 1) % 25_000 == 0:
            progress(f"lightfm matrix: encoded {row + 1:,}/{len(user_ids):,} users")
    if cursor != expected_edges:
        raise RuntimeError(f"LightFM edge accounting failed: encoded={cursor}, expected={expected_edges}")
    interactions = sparse.coo_matrix(
        (
            np.ones(expected_edges, dtype=np.float32),
            (row_indexes, column_indexes),
        ),
        shape=(len(user_ids), len(anime_ids)),
        dtype=np.float32,
    )
    if interactions.nnz != expected_edges:
        raise RuntimeError("LightFM interaction matrix changed edge count")
    return LightFMTrainingData(user_ids=user_ids, anime_ids=anime_ids, interactions=interactions)


def _top_k_ids(
    scores: np.ndarray,
    anime_ids: np.ndarray,
    *,
    known_ids: Sequence[int],
    k: int,
) -> list[int]:
    working = np.asarray(scores, dtype=np.float32).copy()
    candidate_mask = np.ones(len(anime_ids), dtype=bool)
    if len(known_ids):
        known = np.asarray(sorted({int(value) for value in known_ids}), dtype=np.int64)
        rows = np.searchsorted(anime_ids, known)
        valid = rows < len(anime_ids)
        rows = rows[valid]
        known = known[valid]
        rows = rows[anime_ids[rows] == known]
        candidate_mask[rows] = False
    candidates = np.flatnonzero(candidate_mask)
    order = np.lexsort((anime_ids[candidates], -working[candidates]))
    selected = candidates[order[: min(k, len(order))]]
    return [int(value) for value in anime_ids[selected].tolist()]


def _validation_metrics(
    model: Any,
    *,
    item_features: sparse.csr_matrix | None,
    training_data: LightFMTrainingData,
    validation_users: Sequence[UserSplit],
    train_positive_counts: Mapping[int, int],
    genres_by_id: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    item_biases, item_embeddings = model.get_item_representations(item_features)
    user_biases, user_embeddings = model.get_user_representations()
    rankings: list[list[int]] = []
    histories: list[list[int]] = []
    metric_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    exposed: set[int] = set()
    novelty_values: list[float] = []
    popularity_differences: list[float] = []
    diversity_values: list[float] = []
    total_train = sum(int(value) for value in train_positive_counts.values())
    for user in validation_users:
        user_row = int(np.searchsorted(training_data.user_ids, user.user_id))
        if user_row >= len(training_data.user_ids) or int(training_data.user_ids[user_row]) != user.user_id:
            raise RuntimeError(f"Validation user {user.user_id} is absent from LightFM training mappings")
        started = time.perf_counter()
        scores = item_embeddings @ user_embeddings[user_row]
        scores = scores + item_biases + user_biases[user_row]
        known = [anime_id for anime_id, _rating in user.all_observed_training_ratings]
        ranking = _top_k_ids(scores, training_data.anime_ids, known_ids=known, k=20)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = user.validation_positive_ids
        if not relevant:
            raise RuntimeError(f"Eligible validation user {user.user_id} has no validation positive")
        metric_rows.append(ranking_metrics(ranking, relevant).as_dict())
        rankings.append(ranking)
        history = list(user.train_positive_ids)
        histories.append(history)
        exposed.update(ranking)
        novelty_values.extend(
            item_novelty(anime_id, train_positive_counts, total_train, len(training_data.anime_ids))
            for anime_id in ranking
        )
        if history:
            recommended_popularity = mean(
                normalized_log_popularity(anime_id, train_positive_counts) for anime_id in ranking
            )
            history_popularity = mean(
                normalized_log_popularity(anime_id, train_positive_counts) for anime_id in history
            )
            popularity_differences.append(recommended_popularity - history_popularity)
        diversity_values.append(intra_list_diversity(ranking, genres_by_id))
    metric_names = ("ndcg_at_10", "recall_at_10", "hit_rate_at_10", "ndcg_at_20", "recall_at_20", "mrr")
    return {
        **{name: mean(float(row[name]) for row in metric_rows) if metric_rows else 0.0 for name in metric_names},
        "catalog_coverage": catalog_coverage(rankings, len(training_data.anime_ids)),
        "novelty_bits": mean(novelty_values) if novelty_values else 0.0,
        "popularity_bias": mean(popularity_differences) if popularity_differences else 0.0,
        "intra_list_diversity": mean(diversity_values) if diversity_values else 0.0,
        "inference_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "inference_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "validation_users": len(validation_users),
        "validation_target": "validation_positive only; test positives were not read",
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = result["validation_metrics"]
    return (
        float(metrics["ndcg_at_10"]),
        float(metrics["recall_at_10"]),
        float(metrics["catalog_coverage"]),
        -abs(float(metrics["popularity_bias"])),
        str(result["config_key"]),
    )


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
            if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, ValueError):
        return None


def _export_lightfm_artifact(
    model: Any,
    *,
    variant: str,
    item_features: sparse.csr_matrix | None,
    training_data: LightFMTrainingData,
    output_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    item_biases, item_embeddings = model.get_item_representations(item_features)
    user_biases, user_embeddings = model.get_user_representations()
    sample_user_row = 0
    sample_items = np.arange(min(32, len(training_data.anime_ids)), dtype=np.int32)
    native_scores = model.predict(
        sample_user_row,
        sample_items,
        item_features=item_features,
        num_threads=1,
    )
    numpy_scores = (
        item_embeddings[sample_items] @ user_embeddings[sample_user_row]
        + item_biases[sample_items]
        + user_biases[sample_user_row]
    )
    roundtrip_max_abs_error = float(np.max(np.abs(native_scores - numpy_scores))) if len(sample_items) else 0.0
    if roundtrip_max_abs_error > 1e-5:
        raise RuntimeError(f"LightFM NumPy export score mismatch: {roundtrip_max_abs_error}")

    payload_metadata = {
        **dict(metadata),
        "artifact_version": LIGHTFM_ARTIFACT_VERSION,
        "trainer": "lightfm",
        "variant": variant,
        "catalog_ids_sha256": catalog_ids_sha256(training_data.anime_ids.tolist()),
        "numpy_score_roundtrip_max_abs_error": roundtrip_max_abs_error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=training_data.anime_ids,
        user_ids=training_data.user_ids,
        item_embeddings=np.asarray(item_embeddings, dtype=np.float32),
        item_biases=np.asarray(item_biases, dtype=np.float32),
        user_embeddings=np.asarray(user_embeddings, dtype=np.float32),
        user_biases=np.asarray(user_biases, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True, separators=(",", ":"))),
    )
    temporary.replace(output_path)
    return payload_metadata


def train_lightfm_variant(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    training_data: LightFMTrainingData,
    *,
    variant: str,
    search: LightFMSearchConfig,
    output_path: Path,
    train_positive_counts: Mapping[int, int],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if variant not in {"lightfm_id", "lightfm_hybrid"}:
        raise ValueError("variant must be 'lightfm_id' or 'lightfm_hybrid'")
    try:
        import lightfm
        from lightfm import LightFM
    except ImportError as error:
        raise RuntimeError(
            "LightFM training dependency is unavailable. Use environment-lightfm.yml or requirements-lightfm.txt."
        ) from error

    catalog_by_id = {int(item["id"]): dict(item) for item in catalog}
    ordered_catalog = [catalog_by_id[int(anime_id)] for anime_id in training_data.anime_ids.tolist()]
    feature_bundle = (
        build_lightfm_item_features(ordered_catalog, studio_min_frequency=search.studio_min_frequency)
        if variant == "lightfm_hybrid"
        else None
    )
    item_features = feature_bundle.matrix if feature_bundle is not None else None
    validation_ids = select_evaluation_user_ids(
        store,
        limit=search.validation_users,
        seed=search.seed,
        strategy="uniform",
    )
    validation_users = list(store.iter_users_by_ids(validation_ids))
    genres_by_id = {int(item["id"]): tuple(item.get("genres") or ()) for item in catalog}
    tuning_results: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_result: dict[str, Any] | None = None
    variant_started = time.perf_counter()

    for position, candidate in enumerate(search.candidates, start=1):
        if progress is not None:
            progress(
                f"{variant}: training candidate {position}/{len(search.candidates)} "
                f"({candidate.loss}, {candidate.no_components}d, {candidate.epochs} epochs)"
            )
        model = LightFM(
            no_components=candidate.no_components,
            learning_rate=candidate.learning_rate,
            loss=candidate.loss,
            item_alpha=candidate.item_alpha,
            user_alpha=candidate.user_alpha,
            max_sampled=candidate.max_sampled,
            random_state=search.seed,
        )
        train_started = time.perf_counter()
        model.fit(
            training_data.interactions,
            item_features=item_features,
            epochs=candidate.epochs,
            num_threads=search.num_threads,
            verbose=False,
        )
        train_seconds = time.perf_counter() - train_started
        validation = _validation_metrics(
            model,
            item_features=item_features,
            training_data=training_data,
            validation_users=validation_users,
            train_positive_counts=train_positive_counts,
            genres_by_id=genres_by_id,
        )
        result = {
            "config_key": candidate.key,
            "config": asdict(candidate),
            "training_duration_seconds": round(train_seconds, 6),
            "validation_metrics": validation,
        }
        tuning_results.append(result)
        if best_result is None or _selection_key(result) > _selection_key(best_result):
            del best_model
            best_model = model
            best_result = result
        else:
            del model
        gc.collect()

    if best_model is None or best_result is None:
        raise RuntimeError("LightFM search did not produce a model")
    total_duration = time.perf_counter() - variant_started
    metadata = {
        "lightfm_version": str(lightfm.__version__),
        "split_sha256": sha256_file(store.path),
        "training_users": len(training_data.user_ids),
        "training_positive_edges": training_data.positive_edges,
        "catalog_items": len(training_data.anime_ids),
        "selected_config": best_result["config"],
        "selected_config_key": best_result["config_key"],
        "selection_rule": (
            "fixed preselected configuration; validation metrics are diagnostic only"
            if len(search.candidates) == 1
            else "maximize validation NDCG@10, then Recall@10, coverage, lower absolute popularity bias, config key"
        ),
        "selection_data": "validation positives only; test positives were not accessed",
        "selected_validation_metrics": best_result["validation_metrics"],
        "tuning_results": tuning_results,
        "search_config": {
            "validation_users": search.validation_users,
            "seed": search.seed,
            "num_threads": search.num_threads,
            "studio_min_frequency": search.studio_min_frequency,
            "require_both_losses": search.require_both_losses,
        },
        "feature_summary": (
            feature_bundle.summary
            if feature_bundle is not None
            else {
                "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
                "included_fields": [],
                "identity_feature_count": len(training_data.anime_ids),
                "metadata_feature_count": 0,
                "normalization": "identity matrix",
            }
        ),
        "total_search_duration_seconds": round(total_duration, 6),
        "selected_training_duration_seconds": best_result["training_duration_seconds"],
        "peak_process_rss_bytes": _peak_process_rss_bytes(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }
    exported = _export_lightfm_artifact(
        best_model,
        variant=variant,
        item_features=item_features,
        training_data=training_data,
        output_path=output_path,
        metadata=metadata,
    )
    return {
        "variant": variant,
        "artifact_path": str(output_path),
        "artifact_size_bytes": output_path.stat().st_size,
        "artifact_sha256": sha256_file(output_path),
        "metadata": exported,
    }


def train_lightfm_challengers(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    *,
    artifacts_dir: Path,
    search: LightFMSearchConfig,
    train_positive_counts: Mapping[int, int],
    variants: Sequence[str] = ("lightfm_id", "lightfm_hybrid"),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_variants = tuple(dict.fromkeys(variants))
    if not selected_variants or set(selected_variants) - {"lightfm_id", "lightfm_hybrid"}:
        raise ValueError("variants must contain lightfm_id and/or lightfm_hybrid")
    catalog_ids = [int(item["id"]) for item in catalog]
    training_data = build_lightfm_training_data(store, catalog_ids, progress=progress)
    outputs: list[dict[str, Any]] = []
    for variant in selected_variants:
        outputs.append(
            train_lightfm_variant(
                store,
                catalog,
                training_data,
                variant=variant,
                search=search,
                output_path=Path(artifacts_dir) / f"{variant}.npz",
                train_positive_counts=train_positive_counts,
                progress=progress,
            )
        )
    result = {
        "schema_version": 1,
        "generated_at_epoch_seconds": time.time(),
        "split_sha256": sha256_file(store.path),
        "catalog_ids_sha256": catalog_ids_sha256(catalog_ids),
        "search": {
            "candidates": [asdict(candidate) for candidate in search.candidates],
            "validation_users": search.validation_users,
            "seed": search.seed,
            "num_threads": search.num_threads,
            "studio_min_frequency": search.studio_min_frequency,
            "require_both_losses": search.require_both_losses,
        },
        "training_matrix": {
            "users": len(training_data.user_ids),
            "items": len(training_data.anime_ids),
            "positive_edges": training_data.positive_edges,
            "sparsity": 1.0
            - training_data.positive_edges / max(1, len(training_data.user_ids) * len(training_data.anime_ids)),
        },
        "variants": outputs,
        "total_duration_seconds": round(time.perf_counter() - started, 6),
    }
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / "lightfm_training.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(result, indent=2, sort_keys=True))
    return result


def validation_bucket_recall(
    ranking: Sequence[int],
    user: UserSplit,
    bucket_by_id: Mapping[int, str],
    bucket: str,
) -> float:
    """Small public helper used by tests for validation-only diagnostics."""
    relevant = [anime_id for anime_id in user.validation_positive_ids if bucket_by_id.get(anime_id) == bucket]
    return recall_at_k(ranking, relevant, 10) if relevant else math.nan


def validation_bucket_ndcg(
    ranking: Sequence[int],
    user: UserSplit,
    bucket_by_id: Mapping[int, str],
    bucket: str,
) -> float:
    relevant = [anime_id for anime_id in user.validation_positive_ids if bucket_by_id.get(anime_id) == bucket]
    return ndcg_at_k(ranking, relevant, 10) if relevant else math.nan
