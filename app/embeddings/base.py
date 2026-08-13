from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    model_name: str
    model_revision: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def model_info(self) -> dict[str, object]: ...
