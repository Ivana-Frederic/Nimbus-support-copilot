"""Embedding backend: a transformer sentence-encoder (all-MiniLM-L6-v2).

Defined behind a small Protocol so the API layer and tests can swap in a
fake, deterministic embedder without pulling in torch/sentence-transformers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Loaded lazily (inside __init__, not at import time) so importing this
    module doesn't drag in torch just to type-check something.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return SentenceTransformerEmbedder(model_name)
