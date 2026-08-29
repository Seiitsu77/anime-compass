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
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from backend.anime_agent.lightfm_serving import LIGHTFM_ARTIFACT_VERSION, LightFMServingIndex

from .metrics import (
    build_item_popularity_buckets,
    catalog_coverage,
    intra_list_diversity,
    item_novelty,
    ndcg_at_k,
    normalized_log_popularity,
    ranking_metrics,
    recall_at_k,
    recommendation_popularity_concentration,
    user_activity_segment,
)
from .split import SplitStore, UserSplit, catalog_ids_sha256, sha256_file

LIGHTFM_FEATURE_SCHEMA_VERSION = 2
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
    "decade",
    "content_rating",
    "studios",
)
SUPPORTED_METADATA_FIELDS = frozenset(DEFAULT_INCLUDED_METADATA_FIELDS)
_TOKEN_SAFE = re.compile(r"[^a-z0-9]+")
_VARIANT_SAFE = re.compile(r"^lightfm_[a-z0-9_]+$")


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
    validation_users_per_activity_segment: int = 0
    validation_users_per_popularity_bucket: int = 0
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
        if self.validation_users_per_activity_segment < 0:
            raise ValueError("validation_users_per_activity_segment cannot be negative")
        if self.validation_users_per_popularity_bucket < 0:
            raise ValueError("validation_users_per_popularity_bucket cannot be negative")
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


@dataclass(frozen=True)
class LightFMUserFeatures:
    matrix: sparse.csr_matrix
    feature_names: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LightFMVariantConfig:
    """Feature-family definition for one offline LightFM challenger."""

    name: str
    item_fields: tuple[str, ...] = ()
    user_fields: tuple[str, ...] = ()
    user_preference_mass: float = 0.5

    def __post_init__(self) -> None:
        if not _VARIANT_SAFE.fullmatch(self.name):
            raise ValueError("LightFM variant names must match lightfm_[a-z0-9_]+")
        item_fields = _validated_metadata_fields(self.item_fields)
        user_fields = _validated_metadata_fields(self.user_fields)
        if not 0.0 < self.user_preference_mass < 1.0:
            raise ValueError("user_preference_mass must be between zero and one")
        object.__setattr__(self, "item_fields", item_fields)
        object.__setattr__(self, "user_fields", user_fields)


def _validated_metadata_fields(fields: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(field) for field in fields)
    if len(selected) != len(set(selected)):
        raise ValueError("LightFM metadata fields must be unique")
    unknown = set(selected).difference(SUPPORTED_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported LightFM metadata fields: {sorted(unknown)}")
    order = {field: index for index, field in enumerate(DEFAULT_INCLUDED_METADATA_FIELDS)}
    if selected != tuple(sorted(selected, key=order.__getitem__)):
        raise ValueError("LightFM metadata fields must follow the documented forward-selection order")
    return selected


def default_variant_config(name: str) -> LightFMVariantConfig:
    if name == "lightfm_id":
        return LightFMVariantConfig(name=name)
    if name == "lightfm_hybrid":
        return LightFMVariantConfig(name=name, item_fields=DEFAULT_INCLUDED_METADATA_FIELDS)
    raise ValueError(f"Feature configuration is required for experimental variant {name!r}")


def item_metadata_ablation_configs() -> tuple[LightFMVariantConfig, ...]:
    """Return the bounded, documented forward-selection item ablation."""
    names = (
        "lightfm_id",
        "lightfm_item_genres",
        "lightfm_item_genres_type",
        "lightfm_item_genres_type_source",
        "lightfm_item_genres_type_source_decade",
        "lightfm_item_genres_type_source_decade_rating",
        "lightfm_item_full",
    )
    field_counts = (0, 1, 2, 3, 4, 5, 6)
    return tuple(
        LightFMVariantConfig(name=name, item_fields=DEFAULT_INCLUDED_METADATA_FIELDS[:count])
        for name, count in zip(names, field_counts, strict=True)
    )


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


def lightfm_metadata_tokens_by_field(
    item: Mapping[str, Any],
    *,
    allowed_studios: set[str],
    included_fields: Sequence[str] = DEFAULT_INCLUDED_METADATA_FIELDS,
) -> dict[str, tuple[str, ...]]:
    """Create static catalog tokens grouped by audited metadata field."""
    fields = _validated_metadata_fields(included_fields)
    result: dict[str, tuple[str, ...]] = {}
    if "genres" in fields:
        result["genres"] = tuple(
            sorted(
                {
                    f"genre:{_normalize_feature_value(genre)}"
                    for genre in item.get("genres") or []
                    if isinstance(genre, str) and genre.strip()
                }
            )
        )
    if "type" in fields:
        result["type"] = (f"type:{_normalize_feature_value(item.get('type') or 'unknown')}",)
    if "source" in fields:
        result["source"] = (f"source:{_normalize_feature_value(item.get('source') or 'unknown')}",)
    if "decade" in fields:
        result["decade"] = (_decade_token(item.get("start_year")),)
    if "content_rating" in fields:
        result["content_rating"] = (
            f"content-rating:{_normalize_feature_value(item.get('content_rating') or 'unknown')}",
        )
    if "studios" in fields:
        studios = {
            _normalize_feature_value(studio)
            for studio in item.get("studios") or []
            if isinstance(studio, str) and studio.strip()
        }
        kept = tuple(sorted(f"studio:{studio}" for studio in studios if studio in allowed_studios))
        result["studios"] = kept or ("studio:unknown-or-rare",)
    return result


def lightfm_metadata_tokens(
    item: Mapping[str, Any],
    *,
    allowed_studios: set[str],
    included_fields: Sequence[str] = DEFAULT_INCLUDED_METADATA_FIELDS,
) -> tuple[str, ...]:
    """Create static catalog-only feature tokens for one anime.

    Outcome-derived fields are intentionally never read here.  The identity
    feature is added separately by ``build_lightfm_item_features``.
    """
    by_field = lightfm_metadata_tokens_by_field(
        item,
        allowed_studios=allowed_studios,
        included_fields=included_fields,
    )
    return tuple(sorted({token for tokens in by_field.values() for token in tokens}))


def build_lightfm_item_features(
    catalog: Sequence[Mapping[str, Any]],
    *,
    studio_min_frequency: int = 5,
    included_fields: Sequence[str] = DEFAULT_INCLUDED_METADATA_FIELDS,
) -> LightFMItemFeatures:
    """Build an L1-normalized sparse identity + metadata matrix."""
    if studio_min_frequency < 1:
        raise ValueError("studio_min_frequency must be positive")
    fields = _validated_metadata_fields(included_fields)
    ordered = sorted((dict(item) for item in catalog), key=lambda item: int(item["id"]))
    anime_ids = [int(item["id"]) for item in ordered]
    if len(anime_ids) != len(set(anime_ids)):
        raise ValueError("Catalog anime IDs must be unique")
    allowed_studios = _studio_vocabulary(ordered, studio_min_frequency)
    metadata_by_row = [
        lightfm_metadata_tokens(item, allowed_studios=allowed_studios, included_fields=fields) for item in ordered
    ]
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
    source_field_by_feature = {"decade": "start_year"}
    for field in fields:
        source_field = source_field_by_feature.get(field, field)
        present = sum(item.get(source_field) not in (None, "", [], {}) for item in ordered)
        completeness[field] = present / len(ordered) if ordered else 0.0
    summary = {
        "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
        "included_fields": list(fields),
        "excluded_outcome_fields": sorted(FORBIDDEN_METADATA_FIELDS),
        "feature_count": len(feature_names),
        "identity_feature_count": len(identity_names),
        "metadata_feature_count": len(metadata_vocabulary),
        "nonzero_assignments": int(matrix.nnz),
        "studio_min_frequency": studio_min_frequency,
        "eligible_studio_count": len(allowed_studios),
        "field_completeness": completeness,
        "normalization": "per-item L1; identity and selected static metadata are equally weighted tokens",
        "feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
    }
    return LightFMItemFeatures(matrix=matrix, feature_names=feature_names, summary=summary)


def build_lightfm_user_features(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    training_data: LightFMTrainingData,
    *,
    included_fields: Sequence[str] = DEFAULT_INCLUDED_METADATA_FIELDS,
    studio_min_frequency: int = 5,
    preference_mass: float = 0.5,
    progress: Callable[[str], None] | None = None,
) -> LightFMUserFeatures:
    """Build identity plus train-history preference features for each user.

    For every active metadata field, token counts are transformed with
    ``log1p(count)`` and normalized within that field. Active fields receive
    equal mass. The complete metadata block receives ``preference_mass`` and
    the user identity feature receives the remaining mass. Only
    ``train_positive_ids`` are read; validation and test interactions never
    contribute to the representation.
    """
    fields = _validated_metadata_fields(included_fields)
    if studio_min_frequency < 1:
        raise ValueError("studio_min_frequency must be positive")
    if not 0.0 < preference_mass < 1.0:
        raise ValueError("preference_mass must be between zero and one")
    ordered_catalog = sorted((dict(item) for item in catalog), key=lambda item: int(item["id"]))
    catalog_ids = np.asarray([int(item["id"]) for item in ordered_catalog], dtype=np.int64)
    if not np.array_equal(catalog_ids, training_data.anime_ids):
        raise ValueError("User features require the same sorted catalog mapping as LightFM training data")
    allowed_studios = _studio_vocabulary(ordered_catalog, studio_min_frequency)
    tokens_by_anime: dict[int, dict[str, tuple[str, ...]]] = {
        int(item["id"]): lightfm_metadata_tokens_by_field(
            item,
            allowed_studios=allowed_studios,
            included_fields=fields,
        )
        for item in ordered_catalog
    }
    metadata_vocabulary = sorted(
        {token for field_tokens in tokens_by_anime.values() for tokens in field_tokens.values() for token in tokens}
    )
    identity_names = [f"identity-user:{int(user_id)}" for user_id in training_data.user_ids.tolist()]
    feature_names = tuple([*identity_names, *metadata_vocabulary])
    metadata_offset = len(identity_names)
    metadata_index = {name: metadata_offset + index for index, name in enumerate(metadata_vocabulary)}

    indices = array("i")
    values = array("f")
    indptr: npt.NDArray[Any] = np.empty(len(training_data.user_ids) + 1, dtype=np.int64)
    indptr[0] = 0
    users_seen = 0
    for row, user in enumerate(store.iter_users_by_ids(training_data.user_ids.tolist())):
        expected_user_id = int(training_data.user_ids[row])
        if user.user_id != expected_user_id:
            raise RuntimeError("User feature rows changed order during construction")
        counters: dict[str, Counter[str]] = {field: Counter() for field in fields}
        for anime_id in user.train_positive_ids:
            field_tokens = tokens_by_anime.get(int(anime_id))
            if field_tokens is None:
                raise ValueError(f"Training anime {anime_id} is absent from the user-feature catalog")
            for field in fields:
                counters[field].update(field_tokens.get(field, ()))
        active_fields = [field for field in fields if counters[field]]
        identity_weight = 1.0 - preference_mass if active_fields else 1.0
        indices.append(row)
        values.append(identity_weight)
        if active_fields:
            field_mass = preference_mass / len(active_fields)
            metadata_entries: list[tuple[int, float]] = []
            for field in active_fields:
                transformed = {token: math.log1p(count) for token, count in counters[field].items() if count > 0}
                denominator = sum(transformed.values())
                if denominator <= 0.0:
                    continue
                metadata_entries.extend(
                    (metadata_index[token], field_mass * weight / denominator) for token, weight in transformed.items()
                )
            for feature_index, weight in sorted(metadata_entries):
                indices.append(feature_index)
                values.append(weight)
        indptr[row + 1] = len(indices)
        users_seen += 1
        if progress is not None and users_seen % 25_000 == 0:
            progress(f"lightfm user features: encoded {users_seen:,}/{len(training_data.user_ids):,} users")
    if users_seen != len(training_data.user_ids):
        raise RuntimeError("User feature construction did not read every LightFM training user")
    matrix = sparse.csr_matrix(
        (
            np.frombuffer(values, dtype=np.float32),
            np.frombuffer(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(training_data.user_ids), len(feature_names)),
        dtype=np.float32,
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("LightFM user features are not L1-normalized")
    summary = {
        "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
        "included_fields": list(fields),
        "excluded_outcome_fields": sorted(FORBIDDEN_METADATA_FIELDS),
        "source_interactions": "training positive interactions only",
        "heldout_interactions_accessed": False,
        "feature_count": len(feature_names),
        "identity_feature_count": len(identity_names),
        "preference_feature_count": len(metadata_vocabulary),
        "nonzero_assignments": int(matrix.nnz),
        "preference_mass": preference_mass,
        "studio_min_frequency": studio_min_frequency,
        "eligible_studio_count": len(allowed_studios),
        "normalization": (
            "count tokens in train positives; log1p counts; L1 within field; equal mass across active fields; "
            f"metadata mass={preference_mass:.3f}, identity mass={1.0 - preference_mass:.3f}"
        ),
        "feature_names_sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
    }
    return LightFMUserFeatures(matrix=matrix, feature_names=feature_names, summary=summary)


def build_lightfm_training_data(
    store: SplitStore,
    catalog_ids: Sequence[int],
    *,
    user_ids: Sequence[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LightFMTrainingData:
    """Build a bounded-memory positive-only COO matrix from train edges."""
    anime_ids = np.asarray(sorted({int(value) for value in catalog_ids}), dtype=np.int64)
    selected_user_ids = store.training_user_ids() if user_ids is None else [int(value) for value in user_ids]
    if selected_user_ids != sorted(selected_user_ids) or len(selected_user_ids) != len(set(selected_user_ids)):
        raise ValueError("LightFM training user IDs must be unique and sorted")
    user_id_array = np.asarray(selected_user_ids, dtype=np.int64)
    if not len(anime_ids) or not len(user_id_array):
        raise ValueError("LightFM training requires non-empty user and item mappings")
    expected_edges = int(store.metadata()["train_positive_interactions"]) if user_ids is None else None
    row_indexes: Any = np.empty(expected_edges, dtype=np.int32) if expected_edges is not None else array("i")
    column_indexes: Any = np.empty(expected_edges, dtype=np.int32) if expected_edges is not None else array("i")
    cursor = 0
    users_seen = 0
    for row, user in enumerate(store.iter_users_by_ids(user_id_array.tolist())):
        users_seen += 1
        if not user.train_positive_ids:
            raise ValueError(f"Selected LightFM training user {user.user_id} has no positive training edge")
        ids = np.asarray(user.train_positive_ids, dtype=np.int64)
        columns: npt.NDArray[Any] = np.searchsorted(anime_ids, ids)
        if np.any(columns >= len(anime_ids)) or np.any(anime_ids[columns] != ids):
            raise ValueError(f"User {user.user_id} contains an anime outside the LightFM catalog mapping")
        end = cursor + len(columns)
        if expected_edges is None:
            row_indexes.extend([row] * len(columns))
            column_indexes.extend(columns.astype(np.int32, copy=False).tolist())
        else:
            row_indexes[cursor:end] = row
            column_indexes[cursor:end] = columns.astype(np.int32, copy=False)
        cursor = end
        if progress is not None and (row + 1) % 25_000 == 0:
            progress(f"lightfm matrix: encoded {row + 1:,}/{len(user_id_array):,} users")
    if users_seen != len(user_id_array):
        raise ValueError("At least one selected LightFM training user is absent from the split")
    if expected_edges is not None and cursor != expected_edges:
        raise RuntimeError(f"LightFM edge accounting failed: encoded={cursor}, expected={expected_edges}")
    row_array = row_indexes if isinstance(row_indexes, np.ndarray) else np.frombuffer(row_indexes, dtype=np.int32)
    column_array = (
        column_indexes if isinstance(column_indexes, np.ndarray) else np.frombuffer(column_indexes, dtype=np.int32)
    )
    interactions = sparse.coo_matrix(
        (
            np.ones(cursor, dtype=np.float32),
            (row_array, column_array),
        ),
        shape=(len(user_id_array), len(anime_ids)),
        dtype=np.float32,
    )
    if interactions.nnz != cursor:
        raise RuntimeError("LightFM interaction matrix changed edge count")
    return LightFMTrainingData(user_ids=user_id_array, anime_ids=anime_ids, interactions=interactions)


def _top_k_ids(
    scores: np.ndarray,
    anime_ids: np.ndarray,
    *,
    known_ids: Sequence[int],
    k: int,
) -> list[int]:
    working = np.asarray(scores, dtype=np.float32).copy()
    candidate_mask: npt.NDArray[Any] = np.ones(len(anime_ids), dtype=bool)
    if len(known_ids):
        known = np.asarray(sorted({int(value) for value in known_ids}), dtype=np.int64)
        rows: npt.NDArray[Any] = np.searchsorted(anime_ids, known)
        valid = rows < len(anime_ids)
        rows = rows[valid]
        known = known[valid]
        rows = rows[np.asarray(anime_ids)[rows] == known]
        candidate_mask[rows] = False
    candidates = np.flatnonzero(candidate_mask)
    order = np.lexsort((anime_ids[candidates], -working[candidates]))
    selected = candidates[order[: min(k, len(order))]]
    return [int(value) for value in anime_ids[selected].tolist()]


def _select_validation_user_ids(
    store: SplitStore,
    training_user_ids: Sequence[int],
    *,
    limit: int,
    seed: int,
) -> list[int]:
    """Select the existing uniform validation sample within trained users."""
    trained = {int(value) for value in training_user_ids}
    eligible = [user_id for user_id in store.eligible_user_ids() if user_id in trained]

    def sample_key(user_id: int) -> tuple[bytes, int]:
        digest = hashlib.blake2b(f"sample:{seed}:{user_id}".encode(), digest_size=8).digest()
        return digest, user_id

    if limit <= 0 or limit >= len(eligible):
        return eligible
    return sorted(sorted(eligible, key=sample_key)[:limit])


def _select_activity_validation_user_ids(
    store: SplitStore,
    training_user_ids: Sequence[int],
    *,
    users_per_segment: int,
    seed: int,
) -> list[int]:
    if users_per_segment <= 0:
        return []
    trained = {int(value) for value in training_user_ids}
    groups: dict[str, list[int]] = {segment: [] for segment in ("sparse", "medium", "heavy")}
    for user_id, count in store.eligible_user_activity():
        if user_id in trained:
            groups[user_activity_segment(count)].append(user_id)
    selected: set[int] = set()
    for segment, user_ids in groups.items():

        def segment_key(user_id: int, *, _segment: str = str(segment)) -> tuple[bytes, int]:
            digest = hashlib.blake2b(
                f"validation:{seed}:activity:{_segment}:{user_id}".encode(),
                digest_size=8,
            ).digest()
            return digest, user_id

        selected.update(sorted(user_ids, key=segment_key)[:users_per_segment])
    return sorted(selected)


def _select_popularity_validation_user_ids(
    store: SplitStore,
    training_user_ids: Sequence[int],
    bucket_by_id: Mapping[int, str],
    *,
    users_per_bucket: int,
    seed: int,
) -> list[int]:
    """Select a validation-positive tail diagnostic without reading test data."""
    if users_per_bucket <= 0:
        return []
    eligible = set(store.eligible_user_ids())
    candidate_ids = sorted(eligible.intersection(int(value) for value in training_user_ids))
    groups: dict[str, list[int]] = {bucket: [] for bucket in ("head", "mid_tail", "long_tail")}
    for user in store.iter_users_by_ids(candidate_ids):
        relevant_buckets = {
            bucket_by_id.get(int(anime_id))
            for anime_id in user.validation_positive_ids
            if bucket_by_id.get(int(anime_id)) in groups
        }
        for bucket in relevant_buckets:
            if bucket is not None:
                groups[bucket].append(user.user_id)
    selected: set[int] = set()
    for bucket in ("long_tail", "mid_tail", "head"):
        already = sum(user_id in selected for user_id in groups[bucket])
        needed = max(0, users_per_bucket - already)
        candidates = [user_id for user_id in groups[bucket] if user_id not in selected]

        def popularity_key(user_id: int, *, _bucket: str = str(bucket)) -> tuple[bytes, int]:
            digest = hashlib.blake2b(
                f"validation:{seed}:popularity:{_bucket}:{user_id}".encode(),
                digest_size=8,
            ).digest()
            return digest, user_id

        selected.update(sorted(candidates, key=popularity_key)[:needed])
    return sorted(selected)


def _validation_metrics(
    model: Any,
    *,
    item_features: sparse.csr_matrix | None,
    user_features: sparse.csr_matrix | None,
    training_data: LightFMTrainingData,
    validation_users: Sequence[UserSplit],
    train_positive_counts: Mapping[int, int],
    genres_by_id: Mapping[int, Sequence[str]],
    bucket_by_id: Mapping[int, str] | None = None,
    popularity_penalty_lambda: float = 0.0,
) -> dict[str, Any]:
    if popularity_penalty_lambda < 0.0:
        raise ValueError("popularity_penalty_lambda cannot be negative")
    item_biases, item_embeddings = model.get_item_representations(item_features)
    user_biases, user_embeddings = model.get_user_representations(user_features)
    rankings: list[list[int]] = []
    metric_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    exposure_by_item: Counter[int] = Counter()
    novelty_values: list[float] = []
    popularity_differences: list[float] = []
    recommended_popularity_values: list[float] = []
    profile_popularity_values: list[float] = []
    recommended_raw_popularity_values: list[float] = []
    profile_raw_popularity_values: list[float] = []
    diversity_values: list[float] = []
    segment_rows: dict[str, list[dict[str, float]]] = {segment: [] for segment in ("sparse", "medium", "heavy")}
    bucket_recalls: dict[str, list[float]] = {bucket: [] for bucket in ("head", "mid_tail", "long_tail")}
    bucket_ndcgs: dict[str, list[float]] = {bucket: [] for bucket in ("head", "mid_tail", "long_tail")}
    bucket_relevant_items: Counter[str] = Counter()
    total_train = sum(int(value) for value in train_positive_counts.values())
    popularity_vector = np.asarray(
        [normalized_log_popularity(int(anime_id), train_positive_counts) for anime_id in training_data.anime_ids],
        dtype=np.float32,
    )
    active_buckets = dict(bucket_by_id or build_item_popularity_buckets(training_data.anime_ids, train_positive_counts))
    for user in validation_users:
        user_row = int(np.searchsorted(training_data.user_ids, user.user_id))
        if user_row >= len(training_data.user_ids) or int(training_data.user_ids[user_row]) != user.user_id:
            raise RuntimeError(f"Validation user {user.user_id} is absent from LightFM training mappings")
        started = time.perf_counter()
        scores = item_embeddings @ user_embeddings[user_row]
        scores = scores + item_biases + user_biases[user_row]
        if popularity_penalty_lambda:
            scores = scores - popularity_penalty_lambda * popularity_vector
        known = [anime_id for anime_id, _rating in user.all_observed_training_ratings]
        ranking = _top_k_ids(scores, training_data.anime_ids, known_ids=known, k=20)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = user.validation_positive_ids
        if not relevant:
            raise RuntimeError(f"Eligible validation user {user.user_id} has no validation positive")
        metric = ranking_metrics(ranking, relevant).as_dict()
        metric_rows.append(metric)
        segment_rows[user_activity_segment(len(user.train_positive))].append(metric)
        rankings.append(ranking)
        history = list(user.train_positive_ids)
        exposure_by_item.update(ranking)
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
            recommended_raw_popularity = mean(int(train_positive_counts.get(anime_id, 0)) for anime_id in ranking)
            history_raw_popularity = mean(int(train_positive_counts.get(anime_id, 0)) for anime_id in history)
            popularity_differences.append(recommended_popularity - history_popularity)
            recommended_popularity_values.append(recommended_popularity)
            profile_popularity_values.append(history_popularity)
            recommended_raw_popularity_values.append(recommended_raw_popularity)
            profile_raw_popularity_values.append(history_raw_popularity)
        diversity_values.append(intra_list_diversity(ranking, genres_by_id))
        for bucket in ("head", "mid_tail", "long_tail"):
            bucket_relevant = [anime_id for anime_id in relevant if active_buckets.get(int(anime_id)) == bucket]
            if bucket_relevant:
                bucket_recalls[bucket].append(recall_at_k(ranking, bucket_relevant, 10))
                bucket_ndcgs[bucket].append(ndcg_at_k(ranking, bucket_relevant, 10))
                bucket_relevant_items[bucket] += len(bucket_relevant)
    metric_names = ("ndcg_at_10", "recall_at_10", "hit_rate_at_10", "ndcg_at_20", "recall_at_20", "mrr")
    concentration = recommendation_popularity_concentration(
        exposure_by_item,
        training_data.anime_ids,
        train_positive_counts,
    )
    return {
        **{name: mean(float(row[name]) for row in metric_rows) if metric_rows else 0.0 for name in metric_names},
        "catalog_coverage": catalog_coverage(rankings, len(training_data.anime_ids)),
        "novelty_bits": mean(novelty_values) if novelty_values else 0.0,
        "popularity_bias": mean(popularity_differences) if popularity_differences else 0.0,
        "recommended_normalized_popularity": (
            mean(recommended_popularity_values) if recommended_popularity_values else 0.0
        ),
        "profile_normalized_popularity": mean(profile_popularity_values) if profile_popularity_values else 0.0,
        "recommended_training_popularity_count": (
            mean(recommended_raw_popularity_values) if recommended_raw_popularity_values else 0.0
        ),
        "profile_training_popularity_count": (
            mean(profile_raw_popularity_values) if profile_raw_popularity_values else 0.0
        ),
        "intra_list_diversity": mean(diversity_values) if diversity_values else 0.0,
        "popularity_concentration": concentration,
        "user_segments": {
            segment: {
                "users": len(rows),
                **{
                    name: mean(float(row[name]) for row in rows) if rows else 0.0
                    for name in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10")
                },
            }
            for segment, rows in segment_rows.items()
        },
        "heldout_item_popularity": {
            bucket: {
                "users": len(bucket_recalls[bucket]),
                "heldout_items": int(bucket_relevant_items[bucket]),
                "recall_at_10": mean(bucket_recalls[bucket]) if bucket_recalls[bucket] else 0.0,
                "ndcg_at_10": mean(bucket_ndcgs[bucket]) if bucket_ndcgs[bucket] else 0.0,
            }
            for bucket in ("head", "mid_tail", "long_tail")
        },
        "inference_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "inference_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "popularity_penalty_lambda": popularity_penalty_lambda,
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

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # type: ignore[attr-defined]
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, ValueError):
        return None


def _export_lightfm_artifact(
    model: Any,
    *,
    variant: str,
    item_features: sparse.csr_matrix | None,
    user_features: sparse.csr_matrix | None = None,
    training_data: LightFMTrainingData,
    output_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    item_biases, item_embeddings = model.get_item_representations(item_features)
    user_biases, user_embeddings = model.get_user_representations(user_features)
    sample_user_row = 0
    sample_items: npt.NDArray[Any] = np.arange(min(32, len(training_data.anime_ids)), dtype=np.int32)
    native_scores = model.predict(
        sample_user_row,
        sample_items,
        item_features=item_features,
        user_features=user_features,
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
    feature_config: LightFMVariantConfig | None = None,
    search: LightFMSearchConfig,
    output_path: Path,
    train_positive_counts: Mapping[int, int],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved_features = feature_config or default_variant_config(variant)
    if resolved_features.name != variant:
        raise ValueError("LightFM feature configuration name must match the artifact variant")
    try:
        import lightfm
        from lightfm import LightFM
    except ImportError as error:
        raise RuntimeError(
            "LightFM training dependency is unavailable. Use environment-lightfm.yml or requirements-lightfm.txt."
        ) from error

    catalog_by_id = {int(item["id"]): dict(item) for item in catalog}
    ordered_catalog = [catalog_by_id[int(anime_id)] for anime_id in training_data.anime_ids.tolist()]
    item_feature_bundle = (
        build_lightfm_item_features(
            ordered_catalog,
            studio_min_frequency=search.studio_min_frequency,
            included_fields=resolved_features.item_fields,
        )
        if resolved_features.item_fields
        else None
    )
    item_features = item_feature_bundle.matrix if item_feature_bundle is not None else None
    user_feature_bundle = (
        build_lightfm_user_features(
            store,
            ordered_catalog,
            training_data,
            included_fields=resolved_features.user_fields,
            studio_min_frequency=search.studio_min_frequency,
            preference_mass=resolved_features.user_preference_mass,
            progress=progress,
        )
        if resolved_features.user_fields
        else None
    )
    user_features = user_feature_bundle.matrix if user_feature_bundle is not None else None
    validation_ids = _select_validation_user_ids(
        store,
        training_data.user_ids.tolist(),
        limit=search.validation_users,
        seed=search.seed,
    )
    validation_users = list(store.iter_users_by_ids(validation_ids))
    genres_by_id = {int(item["id"]): tuple(item.get("genres") or ()) for item in catalog}
    bucket_by_id = build_item_popularity_buckets(training_data.anime_ids, train_positive_counts)
    activity_validation_ids = _select_activity_validation_user_ids(
        store,
        training_data.user_ids.tolist(),
        users_per_segment=search.validation_users_per_activity_segment,
        seed=search.seed,
    )
    activity_validation_users = list(store.iter_users_by_ids(activity_validation_ids))
    popularity_validation_ids = _select_popularity_validation_user_ids(
        store,
        training_data.user_ids.tolist(),
        bucket_by_id,
        users_per_bucket=search.validation_users_per_popularity_bucket,
        seed=search.seed,
    )
    popularity_validation_users = list(store.iter_users_by_ids(popularity_validation_ids))
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
            user_features=user_features,
            epochs=candidate.epochs,
            num_threads=search.num_threads,
            verbose=False,
        )
        train_seconds = time.perf_counter() - train_started
        validation = _validation_metrics(
            model,
            item_features=item_features,
            user_features=user_features,
            training_data=training_data,
            validation_users=validation_users,
            train_positive_counts=train_positive_counts,
            genres_by_id=genres_by_id,
            bucket_by_id=bucket_by_id,
        )
        validation_diagnostics: dict[str, Any] = {}
        if activity_validation_users:
            validation_diagnostics["activity_balanced"] = _validation_metrics(
                model,
                item_features=item_features,
                user_features=user_features,
                training_data=training_data,
                validation_users=activity_validation_users,
                train_positive_counts=train_positive_counts,
                genres_by_id=genres_by_id,
                bucket_by_id=bucket_by_id,
            )
        if popularity_validation_users:
            validation_diagnostics["popularity_stratified"] = _validation_metrics(
                model,
                item_features=item_features,
                user_features=user_features,
                training_data=training_data,
                validation_users=popularity_validation_users,
                train_positive_counts=train_positive_counts,
                genres_by_id=genres_by_id,
                bucket_by_id=bucket_by_id,
            )
        result = {
            "config_key": candidate.key,
            "config": asdict(candidate),
            "training_duration_seconds": round(train_seconds, 6),
            "validation_metrics": validation,
            "validation_diagnostics": validation_diagnostics,
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
        "selected_validation_diagnostics": best_result["validation_diagnostics"],
        "tuning_results": tuning_results,
        "search_config": {
            "validation_users": search.validation_users,
            "validation_users_per_activity_segment": search.validation_users_per_activity_segment,
            "validation_users_per_popularity_bucket": search.validation_users_per_popularity_bucket,
            "seed": search.seed,
            "num_threads": search.num_threads,
            "studio_min_frequency": search.studio_min_frequency,
            "require_both_losses": search.require_both_losses,
        },
        "feature_summary": {
            "variant_family": (
                "user+item-hybrid"
                if resolved_features.item_fields and resolved_features.user_fields
                else "item-hybrid"
                if resolved_features.item_fields
                else "user-hybrid"
                if resolved_features.user_fields
                else "id"
            ),
            "item": (
                item_feature_bundle.summary
                if item_feature_bundle is not None
                else {
                    "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
                    "included_fields": [],
                    "identity_feature_count": len(training_data.anime_ids),
                    "metadata_feature_count": 0,
                    "normalization": "implicit item identity matrix",
                }
            ),
            "user": (
                user_feature_bundle.summary
                if user_feature_bundle is not None
                else {
                    "schema_version": LIGHTFM_FEATURE_SCHEMA_VERSION,
                    "included_fields": [],
                    "identity_feature_count": len(training_data.user_ids),
                    "preference_feature_count": 0,
                    "normalization": "implicit user identity matrix",
                }
            ),
        },
        "legacy_item_feature_summary": (
            item_feature_bundle.summary
            if item_feature_bundle is not None
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
        user_features=user_features,
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
    variants: Sequence[str | LightFMVariantConfig] = ("lightfm_id", "lightfm_hybrid"),
    training_user_ids: Sequence[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved_variants = tuple(
        value if isinstance(value, LightFMVariantConfig) else default_variant_config(str(value)) for value in variants
    )
    if not resolved_variants:
        raise ValueError("At least one LightFM variant is required")
    variant_names = [value.name for value in resolved_variants]
    if len(variant_names) != len(set(variant_names)):
        raise ValueError("LightFM variant names must be unique")
    catalog_ids = [int(item["id"]) for item in catalog]
    training_data = build_lightfm_training_data(
        store,
        catalog_ids,
        user_ids=training_user_ids,
        progress=progress,
    )
    outputs: list[dict[str, Any]] = []
    for variant_config in resolved_variants:
        variant = variant_config.name
        outputs.append(
            train_lightfm_variant(
                store,
                catalog,
                training_data,
                variant=variant,
                feature_config=variant_config,
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
            "validation_users_per_activity_segment": search.validation_users_per_activity_segment,
            "validation_users_per_popularity_bucket": search.validation_users_per_popularity_bucket,
            "seed": search.seed,
            "num_threads": search.num_threads,
            "studio_min_frequency": search.studio_min_frequency,
            "require_both_losses": search.require_both_losses,
            "feature_variants": [asdict(value) for value in resolved_variants],
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


class _ExportedLightFMRepresentations:
    def __init__(self, index: LightFMServingIndex):
        self.index = index

    def get_item_representations(self, _features: Any = None) -> tuple[np.ndarray, np.ndarray]:
        return self.index.item_biases, self.index.item_embeddings

    def get_user_representations(self, _features: Any = None) -> tuple[np.ndarray, np.ndarray]:
        return self.index.user_biases, self.index.user_embeddings


def evaluate_lightfm_artifact_on_validation(
    index: LightFMServingIndex,
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    train_positive_counts: Mapping[int, int],
    *,
    validation_users: int = 1_000,
    activity_users_per_segment: int = 100,
    popularity_users_per_bucket: int = 100,
    seed: int = 42,
    popularity_penalty_lambda: float = 0.0,
) -> dict[str, Any]:
    """Evaluate an exported candidate using validation positives only."""
    split_hash = sha256_file(store.path)
    if index.metadata.get("split_sha256") != split_hash:
        raise ValueError("LightFM artifact was trained from a different personalized split")
    catalog_by_id = {int(item["id"]): dict(item) for item in catalog}
    if set(catalog_by_id) != set(int(value) for value in index.anime_ids.tolist()):
        raise ValueError("LightFM validation catalog does not match the exported artifact")
    training_data = LightFMTrainingData(
        user_ids=index.user_ids,
        anime_ids=index.anime_ids,
        interactions=sparse.coo_matrix((len(index.user_ids), len(index.anime_ids)), dtype=np.float32),
    )
    bucket_by_id = build_item_popularity_buckets(index.anime_ids, train_positive_counts)
    genres_by_id = {anime_id: tuple(item.get("genres") or ()) for anime_id, item in catalog_by_id.items()}
    model = _ExportedLightFMRepresentations(index)

    def evaluate(user_ids: Sequence[int]) -> dict[str, Any]:
        return _validation_metrics(
            model,
            item_features=None,
            user_features=None,
            training_data=training_data,
            validation_users=list(store.iter_users_by_ids(user_ids)),
            train_positive_counts=train_positive_counts,
            genres_by_id=genres_by_id,
            bucket_by_id=bucket_by_id,
            popularity_penalty_lambda=popularity_penalty_lambda,
        )

    primary_ids = _select_validation_user_ids(
        store,
        index.user_ids.tolist(),
        limit=validation_users,
        seed=seed,
    )
    activity_ids = _select_activity_validation_user_ids(
        store,
        index.user_ids.tolist(),
        users_per_segment=activity_users_per_segment,
        seed=seed,
    )
    popularity_ids = _select_popularity_validation_user_ids(
        store,
        index.user_ids.tolist(),
        bucket_by_id,
        users_per_bucket=popularity_users_per_bucket,
        seed=seed,
    )
    return {
        "primary": evaluate(primary_ids),
        "activity_balanced": evaluate(activity_ids) if activity_ids else None,
        "popularity_stratified": evaluate(popularity_ids) if popularity_ids else None,
        "selection_data": "validation positives only; test positives were not accessed",
        "seed": seed,
        "popularity_penalty_lambda": popularity_penalty_lambda,
    }


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
