from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LIGHTFM_ARTIFACT_VERSION = 1


def _catalog_ids_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for anime_id in sorted({int(value) for value in values}):
        digest.update(f"{anime_id}\n".encode())
    return digest.hexdigest()


class LightFMServingIndex:
    """NumPy-only LightFM inference artifact.

    The LightFM package is required to train and export the arrays, but this
    loader deliberately depends only on NumPy.  FastAPI therefore does not
    need the native LightFM extension in a production environment.
    """

    def __init__(
        self,
        anime_ids: np.ndarray,
        user_ids: np.ndarray,
        item_embeddings: np.ndarray,
        item_biases: np.ndarray,
        user_embeddings: np.ndarray,
        user_biases: np.ndarray,
        metadata: Mapping[str, Any],
        *,
        artifact_path: Path | None = None,
    ):
        self.anime_ids = np.asarray(anime_ids, dtype=np.int64)
        self.user_ids = np.asarray(user_ids, dtype=np.int64)
        self.item_embeddings = np.asarray(item_embeddings, dtype=np.float32)
        self.item_biases = np.asarray(item_biases, dtype=np.float32)
        self.user_embeddings = np.asarray(user_embeddings, dtype=np.float32)
        self.user_biases = np.asarray(user_biases, dtype=np.float32)
        self.metadata = dict(metadata)
        self.artifact_path = Path(artifact_path) if artifact_path is not None else None

    @classmethod
    def load(
        cls,
        path: Path,
        catalog: Sequence[Mapping[str, Any]] | None = None,
    ) -> LightFMServingIndex:
        path = Path(path)
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "anime_ids",
                "user_ids",
                "item_embeddings",
                "item_biases",
                "user_embeddings",
                "user_biases",
                "metadata_json",
            }
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError("LightFM artifact is missing arrays: " + ", ".join(sorted(missing)))
            values = {name: np.asarray(artifact[name]) for name in required if name != "metadata_json"}
            metadata = json.loads(str(artifact["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("LightFM artifact metadata must be a JSON object")

        anime_ids = np.asarray(values["anime_ids"], dtype=np.int64)
        user_ids = np.asarray(values["user_ids"], dtype=np.int64)
        item_embeddings = np.asarray(values["item_embeddings"], dtype=np.float32)
        item_biases = np.asarray(values["item_biases"], dtype=np.float32)
        user_embeddings = np.asarray(values["user_embeddings"], dtype=np.float32)
        user_biases = np.asarray(values["user_biases"], dtype=np.float32)

        if metadata.get("artifact_version") != LIGHTFM_ARTIFACT_VERSION:
            raise ValueError("Unsupported LightFM serving artifact version")
        if metadata.get("trainer") != "lightfm":
            raise ValueError("Artifact was not exported by the LightFM trainer")
        if anime_ids.ndim != 1 or user_ids.ndim != 1:
            raise ValueError("LightFM ID arrays must be one-dimensional")
        if not np.array_equal(anime_ids, np.unique(anime_ids)):
            raise ValueError("LightFM anime IDs must be unique and sorted")
        if not np.array_equal(user_ids, np.unique(user_ids)):
            raise ValueError("LightFM user IDs must be unique and sorted")
        if item_embeddings.ndim != 2 or item_embeddings.shape[0] != len(anime_ids):
            raise ValueError("LightFM item embeddings are not aligned with anime IDs")
        if user_embeddings.ndim != 2 or user_embeddings.shape[0] != len(user_ids):
            raise ValueError("LightFM user embeddings are not aligned with user IDs")
        if item_embeddings.shape[1] != user_embeddings.shape[1] or item_embeddings.shape[1] < 1:
            raise ValueError("LightFM user/item embedding dimensions differ")
        if item_biases.shape != (len(anime_ids),) or user_biases.shape != (len(user_ids),):
            raise ValueError("LightFM bias arrays are not aligned with IDs")
        for array in (item_embeddings, item_biases, user_embeddings, user_biases):
            if not np.isfinite(array).all():
                raise ValueError("LightFM artifact contains non-finite values")

        stored_catalog_hash = metadata.get("catalog_ids_sha256")
        if stored_catalog_hash != _catalog_ids_sha256(anime_ids.tolist()):
            raise ValueError("LightFM artifact catalog checksum does not match its anime IDs")
        if catalog is not None:
            active_ids = [int(item["id"]) for item in catalog]
            if len(active_ids) != len(set(active_ids)):
                raise ValueError("Active catalog anime IDs must be unique")
            if stored_catalog_hash != _catalog_ids_sha256(active_ids):
                raise ValueError("LightFM artifact does not match the active catalog")

        return cls(
            anime_ids,
            user_ids,
            item_embeddings,
            item_biases,
            user_embeddings,
            user_biases,
            metadata,
            artifact_path=path,
        )

    @property
    def resident_array_bytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.anime_ids,
                self.user_ids,
                self.item_embeddings,
                self.item_biases,
                self.user_embeddings,
                self.user_biases,
            )
        )

    def _user_row(self, user_id: int) -> int:
        row = int(np.searchsorted(self.user_ids, int(user_id)))
        if row >= len(self.user_ids) or int(self.user_ids[row]) != int(user_id):
            raise KeyError(f"User {user_id} is not present in the LightFM artifact")
        return row

    def scores_for_user(self, user_id: int) -> np.ndarray:
        """Return raw LightFM scores aligned with ``anime_ids``."""
        row = self._user_row(user_id)
        scores = self.item_embeddings @ self.user_embeddings[row]
        scores = scores + self.item_biases + self.user_biases[row]
        return np.asarray(scores, dtype=np.float32)

    def score_pairs(self, user_id: int, anime_ids: Sequence[int]) -> np.ndarray:
        scores = self.scores_for_user(user_id)
        rows = np.searchsorted(self.anime_ids, np.asarray(anime_ids, dtype=np.int64))
        if np.any(rows >= len(self.anime_ids)) or np.any(self.anime_ids[rows] != np.asarray(anime_ids, dtype=np.int64)):
            raise KeyError("At least one requested anime ID is absent from the LightFM artifact")
        return scores[rows]

    def recommend(self, user_id: int, *, known_ids: Iterable[int] = (), k: int = 20) -> list[int]:
        if k < 1:
            return []
        scores = self.scores_for_user(user_id).copy()
        known = np.asarray(sorted({int(value) for value in known_ids}), dtype=np.int64)
        candidate_mask = np.ones(len(self.anime_ids), dtype=bool)
        if len(known):
            rows = np.searchsorted(self.anime_ids, known)
            valid = rows < len(self.anime_ids)
            rows = rows[valid]
            known = known[valid]
            rows = rows[self.anime_ids[rows] == known]
            candidate_mask[rows] = False
        candidates = np.flatnonzero(candidate_mask)
        order = np.lexsort((self.anime_ids[candidates], -scores[candidates]))
        selected = candidates[order[: min(k, len(order))]]
        return [int(value) for value in self.anime_ids[selected].tolist()]

    def model_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "method": self.metadata.get("variant"),
            "trainer": "LightFM",
            "loss": self.metadata.get("selected_config", {}).get("loss"),
            "items": len(self.anime_ids),
            "users": len(self.user_ids),
            "dimensions": int(self.item_embeddings.shape[1]),
            "artifact_version": LIGHTFM_ARTIFACT_VERSION,
        }
