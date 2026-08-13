from __future__ import annotations

import numpy as np


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        device: str = "cpu",
        batch_size: int = 64,
        local_files_only: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer, __version__
        except ImportError as exc:
            raise RuntimeError("Install requirements-embeddings.txt to enable pretrained semantic embeddings") from exc
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.library_version = __version__
        self.model = SentenceTransformer(
            model_name,
            revision=model_revision,
            device=device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0],
            dtype=np.float32,
        )

    def model_info(self) -> dict[str, object]:
        dimension = int(self.model.get_embedding_dimension() or 0)
        return {
            "provider": "sentence_transformers",
            "model": self.model_name,
            "model_revision": self.model_revision,
            "library_version": self.library_version,
            "dimension": dimension,
            "max_sequence_length": int(self.model.max_seq_length),
            "pretrained": True,
            "device": self.device,
            "local_files_only": self.local_files_only,
        }
