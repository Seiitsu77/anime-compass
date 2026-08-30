"""Reference collaborative baselines for the personalized offline benchmark.

The production channel approximates adjusted-cosine item similarity with a
CountSketch projection. Two baselines are needed to say what that choice costs:

* ``item_item_cosine`` computes the same similarity **exactly**, using an
  identical residual transform. The only difference from CountSketch is the
  absence of the random projection, so the gap between them isolates the
  sketching error rather than confounding it with a different model family.
* ``als`` is implicit-feedback alternating least squares (Hu, Koren, and
  Volinsky, 2008), the standard latent-factor baseline for this task. It learns
  factors under a ranking-agnostic squared loss, in contrast to CountSketch,
  which learns nothing and only compresses observed co-rating structure.

Both are trained from a leakage-safe split store and exported as NumPy arrays,
matching how the other benchmark artifacts are served. Neither adds a compiled
dependency: SciPy sparse plus NumPy is sufficient.
"""

from __future__ import annotations

import heapq
import json
import time
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .models import OfflineRecommendation
from .split import SplitStore, UserSplit, sha256_file

ITEM_ITEM_ARTIFACT_VERSION = 1
ALS_ARTIFACT_VERSION = 1

# Two ALS artifacts exist and must never be confused.
#
# "evaluation" is trained on a leakage-safe split that withholds each user's
# held-out positives. Every published holdout metric comes from it, so it must
# stay byte-stable and must never be rebuilt from full data.
#
# "production" is trained on all historically available positives. It is
# strictly better for serving and strictly invalid for measuring, because the
# users it trained on include the ones any holdout would score.
ARTIFACT_ROLE_EVALUATION = "evaluation"
ARTIFACT_ROLE_PRODUCTION = "production"


def _require_scipy() -> Any:
    """Import SciPy lazily so the web application never pays for it."""
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - exercised only without SciPy
        raise RuntimeError(
            "The item-item and ALS baselines require SciPy. Install it with "
            "`python -m pip install -r requirements-evaluation.txt`."
        ) from exc
    return sparse


CONFIDENCE_MAPPINGS: dict[str, dict[int, float]] = {
    # Binary: every positive is one observation. This is the frozen reference.
    "binary": {},
    # Linear intensity: a 10 counts as three observations of a 8.
    "linear": {8: 1.0, 9: 2.0, 10: 3.0},
    # Gentler: preference intensity exists but is sub-linear.
    "sqrt": {8: 1.0, 9: 1.414, 10: 1.732},
    # Log-scaled, the standard implicit-feedback alternative.
    "log": {8: 1.0, 9: 1.585, 10: 2.0},
}


def _confidence_weight(mapping: Mapping[int, float] | None, rating: int) -> float:
    """Observation weight for one positive rating.

    An unmapped rating falls back to 1.0, so a mapping only ever *raises* the
    weight of stronger ratings. It never introduces a negative or zero weight:
    that would silently reclassify a positive as unobserved or as a negative,
    and implicit ALS has no principled place for negative confidence.
    """
    if not mapping:
        return 1.0
    weight = float(mapping.get(int(rating), 1.0))
    if weight <= 0.0:
        raise ValueError("Confidence weights must be positive")
    return weight


def _residual_rating_matrix(
    store: SplitStore,
    anime_ids: npt.NDArray[Any],
    *,
    positives_only: bool,
    confidence: Mapping[int, float] | None = None,
) -> tuple[Any, int, int]:
    """Build a sparse user-by-item matrix of user-centred rating residuals.

    Centring happens only on observed entries, exactly as the CountSketch
    trainer does, so the matrix stays sparse and the two models see the same
    signal. Held-out positives are never read: only
    ``all_observed_training_ratings`` (or train positives) contribute.
    """
    sparse = _require_scipy()
    index_by_id = {int(anime_id): index for index, anime_id in enumerate(anime_ids.tolist())}

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    users_seen = 0
    ratings_used = 0

    for user in store.iter_users():
        observed = user.train_positive if positives_only else user.all_observed_training_ratings
        kept = [(anime_id, rating) for anime_id, rating in observed if anime_id in index_by_id]
        if not kept:
            continue
        row = users_seen
        users_seen += 1
        ratings = np.asarray([rating for _anime_id, rating in kept], dtype=np.float32)
        residuals: npt.NDArray[Any]
        if positives_only:
            residuals = np.asarray(
                [_confidence_weight(confidence, rating) for _anime_id, rating in kept],
                dtype=np.float32,
            )
        else:
            residuals = ratings - float(np.mean(ratings))
            residuals /= max(float(np.std(ratings)), 1.0)
            if not np.any(np.abs(residuals) > 1e-7):
                continue
        for (anime_id, _rating), residual in zip(kept, residuals.tolist(), strict=True):
            rows.append(row)
            columns.append(index_by_id[anime_id])
            values.append(float(residual))
        ratings_used += len(kept)

    if not values:
        raise ValueError("The split produced no usable training interactions")

    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
        ),
        shape=(users_seen, len(anime_ids)),
        dtype=np.float32,
    )
    return matrix, users_seen, ratings_used


# --------------------------------------------------------------------------
# Exact item-item cosine
# --------------------------------------------------------------------------


def build_item_item_artifact_from_split(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    neighbors: int = 200,
    block_size: int = 512,
) -> dict[str, Any]:
    """Compute exact adjusted-cosine item similarity and keep the top neighbours.

    The full 18k-by-18k similarity matrix is never materialised. Items are
    processed in blocks; each block yields a dense ``block x items`` slab from
    which only the strongest ``neighbors`` entries per row are retained.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    started = time.perf_counter()
    anime_ids = np.asarray(sorted({int(item["id"]) for item in catalog}), dtype=np.int64)
    if not len(anime_ids):
        raise ValueError("Cannot train item-item similarity with an empty catalog")

    matrix, users_seen, ratings_used = _residual_rating_matrix(store, anime_ids, positives_only=False)
    item_count = len(anime_ids)

    # L2-normalise item columns so a dot product is a cosine.
    normalized = matrix.tocsc()
    column_norms = np.sqrt(np.asarray(normalized.multiply(normalized).sum(axis=0)).ravel())
    safe_norms = np.where(column_norms > 1e-8, column_norms, 1.0).astype(np.float32)
    scale = (1.0 / safe_norms).astype(np.float32)
    sparse = _require_scipy()
    normalized = normalized @ sparse.diags(scale)
    normalized = normalized.tocsc()

    kept = min(neighbors, item_count - 1) if item_count > 1 else 0
    neighbor_indices: npt.NDArray[Any] = np.zeros((item_count, kept), dtype=np.int32)
    neighbor_scores: npt.NDArray[Any] = np.zeros((item_count, kept), dtype=np.float32)

    for start in range(0, item_count, block_size):
        stop = min(start + block_size, item_count)
        block = normalized[:, start:stop]
        # (block x items) similarity slab.
        similarity = np.asarray((block.T @ normalized).todense(), dtype=np.float32)
        # An item is never its own neighbour.
        for offset in range(stop - start):
            similarity[offset, start + offset] = -np.inf
        if kept:
            top = np.argpartition(-similarity, kth=kept - 1, axis=1)[:, :kept]
            top_scores = np.take_along_axis(similarity, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            neighbor_indices[start:stop] = np.take_along_axis(top, order, axis=1).astype(np.int32)
            neighbor_scores[start:stop] = np.take_along_axis(top_scores, order, axis=1)

    # Negative similarities are noise for top-N retrieval; clamp them away.
    neighbor_scores = np.where(np.isfinite(neighbor_scores), neighbor_scores, 0.0)
    neighbor_scores = np.maximum(neighbor_scores, 0.0).astype(np.float32)

    metadata = {
        "artifact_version": ITEM_ITEM_ARTIFACT_VERSION,
        "method": "exact user-centred adjusted-cosine item similarity",
        "training_source": "personalized split train ratings",
        "split_sha256": sha256_file(store.path),
        "catalog_items": item_count,
        "users_seen": users_seen,
        "ratings_used": ratings_used,
        "neighbors": int(kept),
        "block_size": int(block_size),
        "build_duration_seconds": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=anime_ids,
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )
    temporary.replace(output_path)
    return metadata


class ItemItemModel:
    """Classic item-KNN scoring over exact adjusted-cosine neighbours."""

    name = "item_item_cosine"
    version = "exact-adjusted-cosine-v1"

    def __init__(self, artifact_path: Path, catalog_ids: Sequence[int], *, build_duration_seconds: float):
        with np.load(artifact_path, allow_pickle=False) as artifact:
            required = {"anime_ids", "neighbor_indices", "neighbor_scores", "metadata_json"}
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError("Item-item artifact is missing arrays: " + ", ".join(sorted(missing)))
            self.anime_ids = np.asarray(artifact["anime_ids"], dtype=np.int64)
            self.neighbor_indices = np.asarray(artifact["neighbor_indices"], dtype=np.int32)
            self.neighbor_scores = np.asarray(artifact["neighbor_scores"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"].item()))

        if metadata.get("artifact_version") != ITEM_ITEM_ARTIFACT_VERSION:
            raise ValueError("Unsupported item-item artifact version")
        if self.neighbor_indices.shape != self.neighbor_scores.shape:
            raise ValueError("Item-item neighbour arrays are not aligned")
        if self.neighbor_indices.shape[0] != len(self.anime_ids):
            raise ValueError("Item-item neighbours are not aligned with anime IDs")
        if not np.isfinite(self.neighbor_scores).all():
            raise ValueError("Item-item artifact contains non-finite similarities")

        self.metadata = metadata
        self.catalog_ids = tuple(sorted({int(value) for value in catalog_ids}))
        self.index_by_id = {int(anime_id): index for index, anime_id in enumerate(self.anime_ids.tolist())}
        self.build_duration_seconds = build_duration_seconds
        self.artifact_path: Path | None = Path(artifact_path)
        self.offline_peak_process_rss_bytes: int | None = None
        self.config = {
            "method": metadata.get("method"),
            "neighbors": metadata.get("neighbors"),
            "profile_feedback": "positive training interactions",
            "training_feedback": "all observed train ratings, classes kept distinct",
        }
        self.resident_array_bytes = int(
            self.anime_ids.nbytes + self.neighbor_indices.nbytes + self.neighbor_scores.nbytes
        )

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        scores: npt.NDArray[Any] = np.zeros(len(self.anime_ids), dtype=np.float32)
        contributing = 0
        for anime_id in user.train_positive_ids:
            index = self.index_by_id.get(int(anime_id))
            if index is None:
                continue
            contributing += 1
            np.add.at(scores, self.neighbor_indices[index], self.neighbor_scores[index])

        score_by_id = {
            int(anime_id): float(score)
            for anime_id, score in zip(self.anime_ids.tolist(), scores.tolist(), strict=True)
            if score > 0.0
        }
        results = heapq.nsmallest(
            k,
            (anime_id for anime_id in self.catalog_ids if anime_id not in known),
            key=lambda anime_id: (-score_by_id.get(anime_id, 0.0), anime_id),
        )
        return OfflineRecommendation(results, {"profile_score_count": len(score_by_id), "profile_items": contributing})


# --------------------------------------------------------------------------
# Implicit-feedback ALS
# --------------------------------------------------------------------------


def _solve_factors(
    factors: npt.NDArray[Any],
    fixed: npt.NDArray[Any],
    indptr: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    regularization: float,
    alpha: float,
    cg_steps: int,
    data: npt.NDArray[Any] | None = None,
) -> None:
    """One ALS half-step, in place, using conjugate gradient.

    Solving each row exactly costs O(f^3) and is prohibitive at this scale, so
    this uses the conjugate-gradient formulation from Takács et al. (2011),
    which needs only matrix-vector products against the fixed factor matrix.

    ``data`` carries the per-entry preference intensity r_ui, giving confidence
    ``c_ui = 1 + alpha * r_ui``. When it is None every observed entry has
    r_ui = 1 and the arithmetic reduces exactly to the binary formulation, so
    the frozen binary reference is unaffected by this parameter existing.
    """
    gramian = fixed.T @ fixed + regularization * np.eye(fixed.shape[1], dtype=np.float32)

    for row in range(factors.shape[0]):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if start == stop:
            factors[row] = 0.0
            continue
        columns = indices[start:stop]
        liked = fixed[columns]
        weights = None if data is None else np.asarray(data[start:stop], dtype=np.float32)

        current = factors[row]
        # Residual of (YtY + Yu^T (C-I) Yu + reg I) x = Yu^T C p, with p = 1 and
        # C - I = alpha * r on observed entries.
        if weights is None:
            target = liked.sum(axis=0)
            weighted_product = liked.T @ (liked @ current)
        else:
            target = (liked * weights[:, None]).sum(axis=0)
            weighted_product = liked.T @ (weights * (liked @ current))
        residual = alpha * target - gramian @ current - alpha * weighted_product
        direction = residual.copy()
        residual_squared = float(residual @ residual)
        if residual_squared < 1e-10:
            continue

        for _step in range(cg_steps):
            if weights is None:
                curvature = liked.T @ (liked @ direction)
            else:
                curvature = liked.T @ (weights * (liked @ direction))
            product = gramian @ direction + alpha * curvature
            denominator = float(direction @ product)
            if abs(denominator) < 1e-12:
                break
            step_size = residual_squared / denominator
            current += step_size * direction
            residual -= step_size * product
            updated_squared = float(residual @ residual)
            if updated_squared < 1e-10:
                break
            direction = residual + (updated_squared / residual_squared) * direction
            residual_squared = updated_squared


def build_als_artifact_from_split(
    store: SplitStore,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    factors: int = 64,
    iterations: int = 15,
    regularization: float = 0.05,
    alpha: float = 40.0,
    cg_steps: int = 3,
    seed: int = 42,
    confidence_mapping: str = "binary",
) -> dict[str, Any]:
    """Train implicit-feedback ALS on train positives and export the factors.

    Only positive training interactions are treated as observed feedback, which
    is the standard implicit formulation and matches how the LightFM challenger
    was trained. Explicit negatives and neutral ratings are deliberately not
    folded into the unobserved mass; they are excluded as known items at
    ranking time instead.
    """
    if factors < 1 or iterations < 1:
        raise ValueError("factors and iterations must be positive")
    if regularization < 0.0 or alpha <= 0.0:
        raise ValueError("regularization must be non-negative and alpha positive")

    started = time.perf_counter()
    anime_ids = np.asarray(sorted({int(item["id"]) for item in catalog}), dtype=np.int64)
    if not len(anime_ids):
        raise ValueError("Cannot train ALS with an empty catalog")

    if confidence_mapping not in CONFIDENCE_MAPPINGS:
        raise ValueError(f"Unknown confidence mapping: {confidence_mapping}")
    mapping = CONFIDENCE_MAPPINGS[confidence_mapping]
    interactions, users_seen, ratings_used = _residual_rating_matrix(
        store, anime_ids, positives_only=True, confidence=mapping
    )
    user_major = interactions.tocsr()
    item_major = interactions.tocsc()
    # `weighted` is None for the binary reference so the solver takes the exact
    # code path it took before weighting existed.
    weighted = None if confidence_mapping == "binary" else mapping

    generator = np.random.default_rng(seed)
    user_factors: npt.NDArray[Any] = generator.normal(0.0, 0.01, size=(users_seen, factors)).astype(np.float32)
    item_factors: npt.NDArray[Any] = generator.normal(0.0, 0.01, size=(len(anime_ids), factors)).astype(np.float32)

    for _iteration in range(iterations):
        _solve_factors(
            user_factors,
            item_factors,
            user_major.indptr,
            user_major.indices,
            regularization,
            alpha,
            cg_steps,
            data=None if weighted is None else user_major.data,
        )
        _solve_factors(
            item_factors,
            user_factors,
            item_major.indptr,
            item_major.indices,
            regularization,
            alpha,
            cg_steps,
            data=None if weighted is None else item_major.data,
        )

    if not np.isfinite(item_factors).all() or not np.isfinite(user_factors).all():
        raise ValueError("ALS training diverged and produced non-finite factors")

    metadata = {
        "artifact_version": ALS_ARTIFACT_VERSION,
        "artifact_role": ARTIFACT_ROLE_EVALUATION,
        "method": "implicit-feedback ALS (conjugate gradient)",
        "training_source": "personalized split train positives",
        "split_sha256": sha256_file(store.path),
        "catalog_items": len(anime_ids),
        "users_seen": users_seen,
        "ratings_used": ratings_used,
        "factors": int(factors),
        "iterations": int(iterations),
        "regularization": float(regularization),
        "alpha": float(alpha),
        "cg_steps": int(cg_steps),
        "seed": int(seed),
        "confidence_mapping": confidence_mapping,
        "confidence_weights": {str(k): v for k, v in mapping.items()},
        "build_duration_seconds": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=anime_ids,
        item_factors=item_factors,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )
    temporary.replace(output_path)
    return metadata


class ALSModel:
    """Serve implicit ALS by folding a user's positives into the item space.

    Only item factors are exported. A user's representation is recomputed at
    request time from their training positives, so the model generalises to
    users who were absent from training and matches how a production service
    would score a live session.
    """

    name = "als"
    version = "implicit-als-v1"

    def __init__(self, artifact_path: Path, catalog_ids: Sequence[int], *, build_duration_seconds: float):
        with np.load(artifact_path, allow_pickle=False) as artifact:
            required = {"anime_ids", "item_factors", "metadata_json"}
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError("ALS artifact is missing arrays: " + ", ".join(sorted(missing)))
            self.anime_ids = np.asarray(artifact["anime_ids"], dtype=np.int64)
            self.item_factors = np.asarray(artifact["item_factors"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"].item()))

        if metadata.get("artifact_version") != ALS_ARTIFACT_VERSION:
            raise ValueError("Unsupported ALS artifact version")
        if self.item_factors.ndim != 2 or self.item_factors.shape[0] != len(self.anime_ids):
            raise ValueError("ALS item factors are not aligned with anime IDs")
        if not np.isfinite(self.item_factors).all():
            raise ValueError("ALS artifact contains non-finite factors")

        self.metadata = metadata
        self.regularization = float(metadata.get("regularization", 0.05))
        self.alpha = float(metadata.get("alpha", 40.0))
        self.catalog_ids = tuple(sorted({int(value) for value in catalog_ids}))
        self.index_by_id = {int(anime_id): index for index, anime_id in enumerate(self.anime_ids.tolist())}
        self.build_duration_seconds = build_duration_seconds
        self.artifact_path: Path | None = Path(artifact_path)
        self.offline_peak_process_rss_bytes: int | None = None
        self.config = {
            "method": metadata.get("method"),
            "factors": metadata.get("factors"),
            "iterations": metadata.get("iterations"),
            "regularization": self.regularization,
            "alpha": self.alpha,
            "profile_feedback": "positive training interactions folded into item space",
            "training_feedback": "train positives only, implicit formulation",
        }
        self.resident_array_bytes = int(self.anime_ids.nbytes + self.item_factors.nbytes)
        self._gramian = self.item_factors.T @ self.item_factors

    def _user_vector(self, positive_ids: Sequence[int]) -> npt.NDArray[Any] | None:
        rows = [self.index_by_id[int(value)] for value in positive_ids if int(value) in self.index_by_id]
        if not rows:
            return None
        liked = self.item_factors[np.asarray(rows, dtype=np.int64)]
        dimensions = self.item_factors.shape[1]
        matrix = (
            self._gramian + self.alpha * (liked.T @ liked) + self.regularization * np.eye(dimensions, dtype=np.float32)
        )
        target = self.alpha * liked.sum(axis=0)
        try:
            return np.linalg.solve(matrix, target).astype(np.float32)
        except np.linalg.LinAlgError:
            return None

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        vector = self._user_vector(user.train_positive_ids)
        if vector is None:
            return OfflineRecommendation(
                heapq.nsmallest(k, (a for a in self.catalog_ids if a not in known)),
                {"profile_score_count": 0},
            )
        scores = self.item_factors @ vector
        score_by_id = {
            int(anime_id): float(score)
            for anime_id, score in zip(self.anime_ids.tolist(), scores.tolist(), strict=True)
        }
        results = heapq.nsmallest(
            k,
            (anime_id for anime_id in self.catalog_ids if anime_id not in known),
            key=lambda anime_id: (-score_by_id.get(anime_id, 0.0), anime_id),
        )
        return OfflineRecommendation(results, {"profile_score_count": len(score_by_id)})


# --------------------------------------------------------------------------
# Sanity-check reference points
# --------------------------------------------------------------------------


class RandomModel:
    """Deterministic pseudo-random ranking: the metric floor.

    Not a baseline anyone would ship. It exists so every other number has a
    known lower reference, and so a metric bug that inflates scores shows up as
    an implausibly strong random model.
    """

    name = "random"
    version = "seeded-random-v1"

    def __init__(self, catalog_ids: Sequence[int], *, seed: int = 42):
        self.catalog_ids = tuple(sorted({int(value) for value in catalog_ids}))
        self.seed = int(seed)
        self.build_duration_seconds = 0.0
        self.artifact_path: Path | None = None
        self.offline_peak_process_rss_bytes: int | None = None
        self.config = {"method": "seeded pseudo-random ranking", "seed": self.seed}
        self.resident_array_bytes = 0

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        candidates = np.asarray([a for a in self.catalog_ids if a not in known], dtype=np.int64)
        generator = np.random.default_rng(self.seed + int(user.user_id))
        chosen = generator.permutation(len(candidates))[:k]
        return OfflineRecommendation([int(candidates[i]) for i in chosen], {"profile_score_count": 0})


class OracleModel:
    """Places held-out positives first: the attainable ranking ceiling.

    This reads the held-out labels and is therefore **not a model**. It verifies
    the metric implementation (a correct oracle must score at the analytic
    ceiling) and gives the other numbers an upper reference point.
    """

    name = "oracle"
    version = "label-reading-oracle-v1"

    def __init__(self, catalog_ids: Sequence[int], *, holdout: str = "test"):
        if holdout not in {"validation", "test"}:
            raise ValueError("holdout must be 'validation' or 'test'")
        self.catalog_ids = tuple(sorted({int(value) for value in catalog_ids}))
        self.holdout = holdout
        self.build_duration_seconds = 0.0
        self.artifact_path: Path | None = None
        self.offline_peak_process_rss_bytes: int | None = None
        self.config = {"method": f"oracle over {holdout} positives", "deployable": False}
        self.resident_array_bytes = 0

    def recommend(self, user: UserSplit, k: int) -> OfflineRecommendation:
        known = {anime_id for anime_id, _rating in user.all_observed_training_ratings}
        relevant = user.test_positive_ids if self.holdout == "test" else user.validation_positive_ids
        ranking = [int(a) for a in relevant if a not in known][:k]
        if len(ranking) < k:
            chosen = set(ranking)
            for anime_id in self.catalog_ids:
                if anime_id in known or anime_id in chosen:
                    continue
                ranking.append(anime_id)
                if len(ranking) >= k:
                    break
        return OfflineRecommendation(ranking[:k], {"profile_score_count": len(relevant)})


# --------------------------------------------------------------------------
# Serving adapter: ALS behind the CollaborativeIndex interface
# --------------------------------------------------------------------------


class ALSCollaborativeAdapter:
    """Expose an ALS artifact through the hybrid's collaborative-channel interface.

    The hybrid calls exactly three members on its collaborative index:
    `profile_scores`, `quality_score`, and `model_info`. This adapter satisfies
    them so ALS can be substituted for CountSketch with **no other change** to
    the hybrid, which is what makes the substitution experiment controlled.

    Bayesian quality statistics are not part of an ALS artifact, so they are
    supplied from the CountSketch artifact. That keeps the quality channel
    byte-identical across both arms of the experiment: the only thing that
    changes is the collaborative similarity signal.
    """

    def __init__(self, als: ALSModel, quality_source: Any | None = None):
        self.als = als
        self.quality_source = quality_source

    def profile_scores(
        self,
        positive_ids: Sequence[int] = (),
        negative_ids: Sequence[int] = (),
        explicit_ratings: Mapping[int, float] | None = None,
    ) -> dict[int, float]:
        liked = [int(value) for value in positive_ids]
        for anime_id, rating in (explicit_ratings or {}).items():
            if float(rating) >= 8.0:
                liked.append(int(anime_id))
        vector = self.als._user_vector(liked)
        if vector is None:
            return {}
        scores = self.als.item_factors @ vector
        # The hybrid blends non-negative normalized channel scores, so clamp and
        # rescale to [0, 1] exactly as the CountSketch channel does.
        scores = np.maximum(scores, 0.0)
        peak = float(scores.max()) if len(scores) else 0.0
        if peak <= 0.0:
            return {}
        scores = scores / peak
        return {
            int(anime_id): float(score)
            for anime_id, score in zip(self.als.anime_ids.tolist(), scores.tolist(), strict=True)
            if score > 0.0
        }

    def quality_score(self, anime_id: int) -> float | None:
        if self.quality_source is None:
            return None
        return self.quality_source.quality_score(anime_id)

    def model_info(self) -> dict[str, Any]:
        info = dict(self.als.config)
        info.update({"available": True, "method": "implicit ALS (hybrid collaborative channel)"})
        return info


# --------------------------------------------------------------------------
# Full-data production training
# --------------------------------------------------------------------------


def build_production_als_artifact(
    ratings_path: Path,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    positive_threshold: int = 8,
    factors: int = 128,
    iterations: int = 15,
    regularization: float = 0.05,
    alpha: float = 5.0,
    cg_steps: int = 3,
    seed: int = 42,
    row_limit: int | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train ALS on **all** historically available positives, for serving only.

    This reads the raw rating file rather than a split store, so no positives
    are withheld. The resulting artifact is therefore stronger for serving and
    invalid for measuring: any holdout scored against it would consist of
    interactions it already trained on. It is tagged
    ``artifact_role="production"`` so the two cannot be mixed up, and it carries
    no ``split_sha256`` because it belongs to no split.

    Hyperparameters default to the frozen validated configuration. They remain
    parameters so tests can train a tiny model, not so production can retune.
    """
    if factors < 1 or iterations < 1:
        raise ValueError("factors and iterations must be positive")
    if regularization < 0.0 or alpha <= 0.0:
        raise ValueError("regularization must be non-negative and alpha positive")
    if not 1 <= positive_threshold <= 10:
        raise ValueError("positive_threshold must be between 1 and 10")

    sparse = _require_scipy()
    started = time.perf_counter()
    anime_ids = np.asarray(sorted({int(item["id"]) for item in catalog}), dtype=np.int64)
    if not len(anime_ids):
        raise ValueError("Cannot train ALS with an empty catalog")
    column_by_id = {int(value): index for index, value in enumerate(anime_ids.tolist())}

    # The file is user-sorted, so a change of user id starts a new matrix row.
    # array("i") keeps the edge list compact without a Python int per edge.
    row_buffer = array("i")
    column_buffer = array("i")
    rows_seen = 0
    positives = 0
    orphan_rows = 0
    current_user: int | None = None
    previous_user = -1
    current_row = -1

    with Path(ratings_path).open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if header != ["user_id", "anime_id", "rating"]:
            raise ValueError(f"Unexpected rating file header: {header}")
        for line in handle:
            if row_limit is not None and rows_seen >= row_limit:
                break
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != 3:
                raise ValueError(f"Malformed rating row at line {rows_seen + 2}")
            try:
                user_id, anime_id, rating = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"Non-integer rating row at line {rows_seen + 2}") from exc
            if user_id < previous_user:
                raise ValueError("The rating file must be sorted by user_id")
            if rating < 1 or rating > 10:
                raise ValueError(f"Rating outside 1..10 at line {rows_seen + 2}")
            previous_user = user_id
            rows_seen += 1
            if progress is not None and rows_seen % 5_000_000 == 0:
                progress(f"scanned {rows_seen:,} rating rows")

            if rating < positive_threshold:
                continue
            column = column_by_id.get(anime_id)
            if column is None:
                orphan_rows += 1
                continue
            if user_id != current_user:
                current_user = user_id
                current_row += 1
            row_buffer.append(current_row)
            column_buffer.append(column)
            positives += 1

    if not positives:
        raise ValueError("No positive interactions were found")

    user_count = current_row + 1
    interactions = sparse.csr_matrix(
        (
            np.ones(positives, dtype=np.float32),
            (
                np.frombuffer(row_buffer, dtype=np.int32),
                np.frombuffer(column_buffer, dtype=np.int32),
            ),
        ),
        shape=(user_count, len(anime_ids)),
        dtype=np.float32,
    )
    del row_buffer, column_buffer
    if progress is not None:
        progress(f"built {positives:,} positive edges over {user_count:,} users")

    user_major = interactions.tocsr()
    item_major = interactions.tocsc()
    generator = np.random.default_rng(seed)
    user_factors: npt.NDArray[Any] = generator.normal(0.0, 0.01, size=(user_count, factors)).astype(np.float32)
    item_factors: npt.NDArray[Any] = generator.normal(0.0, 0.01, size=(len(anime_ids), factors)).astype(np.float32)

    for iteration in range(iterations):
        _solve_factors(
            user_factors,
            item_factors,
            user_major.indptr,
            user_major.indices,
            regularization,
            alpha,
            cg_steps,
        )
        _solve_factors(
            item_factors,
            user_factors,
            item_major.indptr,
            item_major.indices,
            regularization,
            alpha,
            cg_steps,
        )
        if progress is not None:
            progress(f"iteration {iteration + 1}/{iterations}")

    if not np.isfinite(item_factors).all() or not np.isfinite(user_factors).all():
        raise ValueError("ALS training diverged and produced non-finite factors")

    metadata = {
        "artifact_version": ALS_ARTIFACT_VERSION,
        "artifact_role": ARTIFACT_ROLE_PRODUCTION,
        "method": "implicit-feedback ALS (conjugate gradient)",
        "training_source": "all historically available positive ratings",
        "not_valid_for_holdout_evaluation": True,
        "ratings_file": Path(ratings_path).name,
        "ratings_sha256": sha256_file(Path(ratings_path)),
        "catalog_ids_sha256": _catalog_id_digest(anime_ids),
        "positive_threshold": int(positive_threshold),
        "rows_scanned": rows_seen,
        "catalog_items": len(anime_ids),
        "users_seen": user_count,
        "ratings_used": positives,
        "orphan_positive_rows": orphan_rows,
        "factors": int(factors),
        "iterations": int(iterations),
        "regularization": float(regularization),
        "alpha": float(alpha),
        "cg_steps": int(cg_steps),
        "seed": int(seed),
        "confidence_mapping": "binary",
        "build_duration_seconds": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        anime_ids=anime_ids,
        item_factors=item_factors,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )
    temporary.replace(output_path)
    return metadata


def _catalog_id_digest(anime_ids: npt.NDArray[Any]) -> str:
    """Hash the exact catalog ID set the artifact was trained against."""
    import hashlib

    digest = hashlib.sha256()
    for value in anime_ids.tolist():
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
