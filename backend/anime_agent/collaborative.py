from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1


class CollaborativeIndex:
    """Compact item embeddings learned from anonymous user ratings.

    The artifact stores normalized CountSketch projections of user-centred
    item-rating vectors. Dot products therefore approximate adjusted-cosine
    item similarity without loading the original ratings into the web process.
    """

    def __init__(
        self,
        anime_ids: np.ndarray,
        vectors: np.ndarray,
        rating_count: np.ndarray,
        rating_mean: np.ndarray,
        bayesian_score: np.ndarray,
        metadata: dict[str, Any],
    ):
        self.anime_ids = np.asarray(anime_ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.rating_count = np.asarray(rating_count, dtype=np.int64)
        self.rating_mean = np.asarray(rating_mean, dtype=np.float32)
        self.bayesian_score = np.asarray(bayesian_score, dtype=np.float32)
        self.metadata = dict(metadata)
        self.index_by_id = {int(anime_id): index for index, anime_id in enumerate(self.anime_ids.tolist())}

    @classmethod
    def load(
        cls,
        path: Path,
        catalog: Sequence[Mapping[str, Any]],
    ) -> CollaborativeIndex:
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "anime_ids",
                "vectors",
                "rating_count",
                "rating_mean",
                "bayesian_score",
                "metadata_json",
            }
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError("Collaborative artifact is missing arrays: " + ", ".join(sorted(missing)))
            anime_ids = np.asarray(artifact["anime_ids"], dtype=np.int64)
            vectors = np.asarray(artifact["vectors"], dtype=np.float32)
            rating_count = np.asarray(artifact["rating_count"], dtype=np.int64)
            rating_mean = np.asarray(artifact["rating_mean"], dtype=np.float32)
            bayesian_score = np.asarray(artifact["bayesian_score"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"].item()))

        row_count = len(anime_ids)
        if metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported collaborative artifact version")
        if anime_ids.ndim != 1 or len(set(anime_ids.tolist())) != row_count:
            raise ValueError("Collaborative anime IDs must be a unique one-dimensional array")
        if vectors.ndim != 2 or vectors.shape[0] != row_count or vectors.shape[1] < 8:
            raise ValueError("Collaborative vectors have an invalid shape")
        if any(len(array) != row_count for array in (rating_count, rating_mean, bayesian_score)):
            raise ValueError("Collaborative statistics are not aligned with anime IDs")
        if not (np.isfinite(vectors).all() and np.isfinite(rating_mean).all() and np.isfinite(bayesian_score).all()):
            raise ValueError("Collaborative artifact contains non-finite values")
        if np.any(rating_count < 0) or np.any((rating_mean < 0) | (rating_mean > 10)):
            raise ValueError("Collaborative rating statistics are outside their valid range")
        if np.any((bayesian_score < 0) | (bayesian_score > 1)):
            raise ValueError("Collaborative Bayesian scores must be between zero and one")

        catalog_ids = {int(item["id"]) for item in catalog}
        overlap = sum(int(anime_id) in catalog_ids for anime_id in anime_ids)
        if row_count and overlap / row_count < 0.90:
            raise ValueError("Collaborative artifact does not match the active catalog")

        return cls(
            anime_ids,
            vectors,
            rating_count,
            rating_mean,
            bayesian_score,
            metadata,
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "method": "user-centred CountSketch item similarity",
            "items": len(self.anime_ids),
            "dimensions": int(self.vectors.shape[1]),
            "ratings": int(self.metadata.get("ratings_used", 0)),
            "users": int(self.metadata.get("users_seen", 0)),
            "artifact_version": ARTIFACT_VERSION,
        }

    def profile_scores(
        self,
        positive_ids: Sequence[int] = (),
        negative_ids: Sequence[int] = (),
        explicit_ratings: Mapping[int, float] | None = None,
    ) -> dict[int, float]:
        weighted_rows: list[np.ndarray] = []
        weights: list[float] = []

        for anime_id in positive_ids:
            index = self.index_by_id.get(int(anime_id))
            if index is not None and np.any(self.vectors[index]):
                weighted_rows.append(self.vectors[index])
                weights.append(1.0)
        for anime_id in negative_ids:
            index = self.index_by_id.get(int(anime_id))
            if index is not None and np.any(self.vectors[index]):
                weighted_rows.append(self.vectors[index])
                weights.append(-1.0)
        for anime_id, rating in (explicit_ratings or {}).items():
            index = self.index_by_id.get(int(anime_id))
            if index is None or not np.any(self.vectors[index]):
                continue
            preference = max(-1.0, min(1.0, (float(rating) - 5.5) / 4.5))
            if abs(preference) < 0.10:
                continue
            weighted_rows.append(self.vectors[index])
            weights.append(preference)

        if not weighted_rows:
            return {}

        stacked_rows = np.stack(weighted_rows, axis=0)
        direction = np.sign(np.asarray(weights, dtype=np.float32))
        magnitude = np.abs(np.asarray(weights, dtype=np.float32))
        profile = np.sum(stacked_rows * direction[:, None] * magnitude[:, None], axis=0)
        norm = float(np.linalg.norm(profile))
        if not np.isfinite(norm) or norm <= 1e-8:
            return {}
        profile /= norm

        similarities = self.vectors @ profile
        confidence = np.clip(
            np.log1p(self.rating_count.astype(np.float32)) / np.log1p(5_000.0),
            0.0,
            1.0,
        )
        scores = np.maximum(similarities, 0.0) * (0.65 + 0.35 * confidence)
        return {
            int(anime_id): float(score)
            for anime_id, score in zip(self.anime_ids.tolist(), scores.tolist(), strict=True)
            if score > 0.0
        }

    def quality_score(self, anime_id: int) -> float | None:
        index = self.index_by_id.get(int(anime_id))
        if index is None or self.rating_count[index] <= 0:
            return None
        return float(self.bayesian_score[index])
