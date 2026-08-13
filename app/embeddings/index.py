from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.anime_agent.recommender import story_text

from .base import EmbeddingProvider

ARTIFACT_FORMAT_VERSION = 1
TEXT_SCHEMA_VERSION = "anime-story-v1"
NORMALIZATION_TOLERANCE = 1e-4


def catalog_checksum(catalog: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(catalog, key=lambda value: int(value["id"])):
        digest.update(str(item["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(story_text(item).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


class SemanticEmbeddingIndex:
    def __init__(
        self,
        anime_ids: np.ndarray,
        matrix: np.ndarray,
        metadata: dict[str, Any],
        provider: EmbeddingProvider,
    ):
        self.anime_ids = np.asarray(anime_ids, dtype=np.int64)
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.metadata = metadata
        self.provider = provider
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = threading.Lock()
        self.row_by_id = {int(anime_id): index for index, anime_id in enumerate(self.anime_ids)}
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.anime_ids):
            raise ValueError("Semantic embedding artifact has inconsistent dimensions")

    @classmethod
    def load(
        cls,
        path: Path,
        provider: EmbeddingProvider,
        catalog: list[dict[str, Any]],
        *,
        expected_dimension: int | None = None,
    ) -> SemanticEmbeddingIndex:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                anime_ids = np.asarray(data["anime_ids"], dtype=np.int64)
                matrix = np.asarray(data["embeddings"], dtype=np.float32)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Semantic embedding artifact is unreadable or incomplete") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Semantic embedding artifact metadata must be an object")
        cls._validate_artifact(
            anime_ids,
            matrix,
            metadata,
            provider,
            catalog,
            expected_dimension=expected_dimension,
        )
        return cls(anime_ids, matrix, metadata, provider)

    @classmethod
    def build(
        cls,
        path: Path,
        provider: EmbeddingProvider,
        catalog: list[dict[str, Any]],
    ) -> SemanticEmbeddingIndex:
        texts = [story_text(item) for item in catalog]
        matrix = cls._normalize_matrix(provider.encode_documents(texts), expected_rows=len(catalog))
        anime_ids: np.ndarray = np.asarray(
            [int(item["id"]) for item in catalog],
            dtype=np.int64,
        )
        provider_info = provider.model_info()
        metadata = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "text_schema_version": TEXT_SCHEMA_VERSION,
            "provider": provider_info.get("provider"),
            "model_name": provider.model_name,
            "model_revision": getattr(provider, "model_revision", "default"),
            "model_version": provider_info.get("model_version") or provider_info.get("model"),
            "library_version": provider_info.get("library_version"),
            "pretrained": bool(provider_info.get("pretrained")),
            "max_sequence_length": provider_info.get("max_sequence_length"),
            "vector_dimension": int(matrix.shape[1]),
            "document_count": len(anime_ids),
            "dtype": "float32",
            "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_data_checksum": catalog_checksum(catalog),
            "normalized": True,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                anime_ids=anime_ids,
                embeddings=matrix,
                metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
            )
        temporary_path.replace(path)
        return cls(anime_ids, matrix, metadata, provider)

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray, *, expected_rows: int) -> np.ndarray:
        normalized = np.asarray(matrix, dtype=np.float32)
        if normalized.ndim != 2 or normalized.shape[0] != expected_rows or normalized.shape[1] <= 0:
            raise ValueError("Embedding provider returned an unexpected matrix shape")
        if not np.isfinite(normalized).all():
            raise ValueError("Embedding provider returned non-finite values")
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Embedding provider returned one or more zero vectors")
        return np.asarray(normalized / norms, dtype=np.float32)

    @classmethod
    def _validate_artifact(
        cls,
        anime_ids: np.ndarray,
        matrix: np.ndarray,
        metadata: dict[str, Any],
        provider: EmbeddingProvider,
        catalog: list[dict[str, Any]],
        *,
        expected_dimension: int | None,
    ) -> None:
        if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError("Semantic artifact format version is unsupported")
        if metadata.get("text_schema_version") != TEXT_SCHEMA_VERSION:
            raise ValueError("Semantic artifact text preprocessing schema is unsupported")
        if metadata.get("model_name") != provider.model_name:
            raise ValueError("Semantic artifact model does not match EMBEDDING_MODEL")
        if metadata.get("model_revision") != getattr(provider, "model_revision", "default"):
            raise ValueError("Semantic artifact revision does not match EMBEDDING_MODEL_REVISION")
        if metadata.get("source_data_checksum") != catalog_checksum(catalog):
            raise ValueError("Semantic artifact is stale for the current catalog")

        catalog_ids: np.ndarray = np.asarray(
            [int(item["id"]) for item in catalog],
            dtype=np.int64,
        )
        if anime_ids.ndim != 1 or len(anime_ids) != len(catalog_ids):
            raise ValueError("Semantic artifact anime ID mapping is incomplete")
        if len(np.unique(anime_ids)) != len(anime_ids) or set(anime_ids.tolist()) != set(catalog_ids.tolist()):
            raise ValueError("Semantic artifact anime ID mapping does not match the catalog")
        if matrix.ndim != 2 or matrix.shape[0] != len(anime_ids) or matrix.shape[1] <= 0:
            raise ValueError("Semantic artifact embedding matrix has inconsistent dimensions")
        if int(metadata.get("document_count") or -1) != len(anime_ids):
            raise ValueError("Semantic artifact document count is inconsistent")
        if int(metadata.get("vector_dimension") or -1) != matrix.shape[1]:
            raise ValueError("Semantic artifact vector dimension is inconsistent")
        if expected_dimension is not None and matrix.shape[1] != expected_dimension:
            raise ValueError("Semantic artifact vector dimension does not match EMBEDDING_DIMENSIONS")
        if metadata.get("normalized") is not True or not np.isfinite(matrix).all():
            raise ValueError("Semantic artifact vectors are invalid")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=NORMALIZATION_TOLERANCE):
            raise ValueError("Semantic artifact vectors are not normalized")

    def document_vector(self, anime_id: int) -> np.ndarray | None:
        row = self.row_by_id.get(int(anime_id))
        return self.matrix[row] if row is not None else None

    def encode_query(self, text: str) -> np.ndarray:
        cache_key = " ".join(text.split())
        with self._query_cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                return cached
        vector = np.asarray(self.provider.encode_query(text), dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.matrix.shape[1] or not np.isfinite(vector).all():
            raise ValueError("Embedding provider returned an invalid query vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Embedding provider returned a zero query vector")
        normalized = np.asarray(vector / norm, dtype=np.float32)
        with self._query_cache_lock:
            self._query_cache[cache_key] = normalized
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > 512:
                self._query_cache.popitem(last=False)
        return normalized

    def model_info(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "available": True,
            "document_count": len(self.anime_ids),
        }
