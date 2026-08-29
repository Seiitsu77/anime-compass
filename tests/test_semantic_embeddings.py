from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.embeddings.index import SemanticEmbeddingIndex
from backend.anime_agent.recommender import AnimeRecommender, experimental_semantic_weights


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

    # The channel is retired by default, so it is available but carries no weight.
    default_result = AnimeRecommender(catalog, semantic_index=loaded).recommend(
        free_text_preferences="ghosts spirits investigation",
        limit=2,
    )[0]
    default_channel = default_result["score_breakdown"]["channels"]["semantic_embedding"]
    assert default_channel["active"] is True
    assert default_channel["configured_weight"] == 0.0
    assert default_channel["weighted_contribution"] == 0.0

    # It can still be switched back on for experiments, which is what keeps the
    # retirement decision falsifiable.
    experimental = AnimeRecommender(
        catalog,
        weights=experimental_semantic_weights(),
        semantic_index=loaded,
    ).recommend(free_text_preferences="ghosts spirits investigation", limit=2)[0]
    experimental_channel = experimental["score_breakdown"]["channels"]["semantic_embedding"]
    assert experimental_channel["active"] is True
    assert experimental_channel["configured_weight"] > 0


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

    # This asserts the channel's own behaviour, so it must be weighted on: the
    # production default retires it to zero.
    results = AnimeRecommender(
        catalog,
        weights=experimental_semantic_weights(),
        semantic_index=semantic_index,
    ).recommend(free_text_preferences="paranormal specters", limit=3)

    assert results[0]["title"] == "Ghost Hunt"
    channel = results[0]["score_breakdown"]["channels"]["semantic_embedding"]
    assert channel["active"] is True
    assert channel["raw_score"] > 0.99


def test_semantic_is_retired_from_the_default_blend() -> None:
    """The default blend must not carry the retired channel's weight.

    Retirement was an evidence-based decision (8.9% relative NDCG@10 loss on 300
    held-out users). This pins it so the weight cannot drift back silently.
    """
    from backend.anime_agent.recommender import (
        DEFAULT_CHANNEL_WEIGHTS,
        SEMANTIC_EXPERIMENTAL_WEIGHT,
    )

    assert DEFAULT_CHANNEL_WEIGHTS["semantic_embedding"] == 0.0
    # Every other channel keeps its original value; removing 0.14 renormalizes
    # proportionally rather than introducing new hand-tuned constants.
    assert DEFAULT_CHANNEL_WEIGHTS["collaborative"] == 0.22
    assert DEFAULT_CHANNEL_WEIGHTS["metadata"] == 0.16
    # The experiment path still restores a positive weight.
    assert experimental_semantic_weights()["semantic_embedding"] == SEMANTIC_EXPERIMENTAL_WEIGHT
    assert experimental_semantic_weights(0.05)["semantic_embedding"] == 0.05


def test_experimental_semantic_weight_rejects_a_negative_value() -> None:
    with pytest.raises(ValueError):
        experimental_semantic_weights(-0.1)
