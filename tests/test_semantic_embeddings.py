from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.embeddings.index import SemanticEmbeddingIndex
from backend.anime_agent.recommender import AnimeRecommender


class FakeEmbeddingProvider:
    model_name = "fake-semantic-v1"

    def __init__(self, model_revision: str = "test-revision"):
        self.model_revision = model_revision
        self.document_encode_calls = 0
        self.query_encode_calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        values = np.zeros(8, dtype=np.float32)
        for token in text.casefold().split():
            values[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(values)] += 1
        norm = float(np.linalg.norm(values))
        return values / norm if norm else values

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_encode_calls += 1
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        self.query_encode_calls += 1
        return self._vector(text)

    def model_info(self) -> dict[str, object]:
        return {
            "provider": "fake",
            "model": self.model_name,
            "model_revision": self.model_revision,
            "library_version": "test",
            "dimension": 8,
            "pretrained": False,
        }


class ConceptEmbeddingProvider(FakeEmbeddingProvider):
    model_name = "concept-semantic-v1"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        text_key = text.casefold()
        vector = np.zeros(8, dtype=np.float32)
        concepts = (
            ({"ghost", "ghosts", "spirit", "spirits", "specter", "specters", "paranormal", "apparition"}, 0),
            ({"romance", "relationship", "love"}, 1),
            ({"psychological", "battle", "death"}, 2),
        )
        tokens = set(text_key.split())
        for words, index in concepts:
            if tokens.intersection(words):
                vector[index] = 1.0
        if not vector.any():
            vector[7] = 1.0
        return vector


def test_semantic_artifact_round_trip_and_channel_activation(
    tmp_path: Path,
    catalog: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "semantic_embeddings.npz"
    provider = FakeEmbeddingProvider()
    built = SemanticEmbeddingIndex.build(artifact, provider, catalog)
    loaded = SemanticEmbeddingIndex.load(artifact, provider, catalog)
    assert built.matrix.shape == (len(catalog), 8)
    assert loaded.model_info()["source_data_checksum"]
    assert loaded.model_info()["artifact_format_version"] == 1
    assert loaded.model_info()["model_revision"] == "test-revision"

    result = AnimeRecommender(catalog, semantic_index=loaded).recommend(
        free_text_preferences="ghosts spirits investigation",
        limit=2,
    )[0]
    channel = result["score_breakdown"]["channels"]["semantic_embedding"]
    assert channel["active"] is True
    assert channel["configured_weight"] > 0


def test_stale_semantic_artifact_is_rejected(tmp_path: Path, catalog: list[dict[str, object]]) -> None:
    artifact = tmp_path / "semantic_embeddings.npz"
    provider = FakeEmbeddingProvider()
    SemanticEmbeddingIndex.build(artifact, provider, catalog)
    changed_catalog = [dict(item) for item in catalog]
    changed_catalog[0] = {**changed_catalog[0], "synopsis": "changed source data"}
    with pytest.raises(ValueError, match="stale"):
        SemanticEmbeddingIndex.load(artifact, provider, changed_catalog)


def test_loading_artifact_never_recomputes_documents_and_query_cache_is_normalized(
    tmp_path: Path,
    catalog: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "semantic_embeddings.npz"
    provider = FakeEmbeddingProvider()
    SemanticEmbeddingIndex.build(artifact, provider, catalog)
    loaded = SemanticEmbeddingIndex.load(artifact, provider, catalog, expected_dimension=8)

    loaded.encode_query("ghosts and spirits")
    loaded.encode_query("ghosts   and   spirits")

    assert provider.document_encode_calls == 1
    assert provider.query_encode_calls == 1


def test_artifact_rejects_wrong_revision_dimension_and_duplicate_ids(
    tmp_path: Path,
    catalog: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "semantic_embeddings.npz"
    provider = FakeEmbeddingProvider()
    SemanticEmbeddingIndex.build(artifact, provider, catalog)

    with pytest.raises(ValueError, match="revision"):
        SemanticEmbeddingIndex.load(artifact, FakeEmbeddingProvider("different-revision"), catalog)
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        SemanticEmbeddingIndex.load(artifact, provider, catalog, expected_dimension=12)

    with np.load(artifact, allow_pickle=False) as data:
        anime_ids = data["anime_ids"].copy()
        embeddings = data["embeddings"].copy()
        metadata_json = str(data["metadata_json"].item())
    anime_ids[0] = anime_ids[1]
    with artifact.open("wb") as file:
        np.savez_compressed(
            file,
            anime_ids=anime_ids,
            embeddings=embeddings,
            metadata_json=np.asarray(json.dumps(json.loads(metadata_json))),
        )
    with pytest.raises(ValueError, match="ID mapping"):
        SemanticEmbeddingIndex.load(artifact, provider, catalog)


def test_semantic_channel_recovers_a_paraphrased_story_concept(
    tmp_path: Path,
    catalog: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "semantic_embeddings.npz"
    provider = ConceptEmbeddingProvider()
    semantic_index = SemanticEmbeddingIndex.build(artifact, provider, catalog)

    results = AnimeRecommender(catalog, semantic_index=semantic_index).recommend(
        free_text_preferences="paranormal specters",
        limit=3,
    )

    assert results[0]["title"] == "Ghost Hunt"
    channel = results[0]["score_breakdown"]["channels"]["semantic_embedding"]
    assert channel["active"] is True
    assert channel["raw_score"] > 0.99
