"""Retriever with RL-tuned re-ranking. The actual arm-picking logic lives in
app.rl.bandit, this file just applies whichever weights it picks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorMatch, VectorStore

# Each arm is a (semantic_weight, recency_weight, popularity_weight) preset.
# Weights need not sum to 1; scores are all already normalized to [0, 1].
RANKING_ARMS: dict[str, tuple[float, float, float]] = {
    "pure_semantic": (1.0, 0.0, 0.0),
    "semantic_recency": (0.7, 0.3, 0.0),
    "semantic_popularity": (0.7, 0.0, 0.3),
    "balanced": (0.6, 0.2, 0.2),
    "recency_heavy": (0.4, 0.5, 0.1),
}


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    source: str
    text: str
    score: float


def _rerank(matches: list[VectorMatch], weights: tuple[float, float, float]) -> list[VectorMatch]:
    w_sem, w_recency, w_pop = weights

    def combined(m: VectorMatch) -> float:
        recency = float(m.metadata.get("recency_score", 0.5))
        popularity = float(m.metadata.get("popularity_score", 0.5))
        return w_sem * m.score + w_recency * recency + w_pop * popularity

    return sorted(matches, key=combined, reverse=True)


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, *, fetch_k: int = 12):
        self._embedder = embedder
        self._vector_store = vector_store
        self._fetch_k = fetch_k

    def has_index(self) -> bool:
        return self._vector_store.count() > 0

    def retrieve(self, query: str, k: int, arm: str) -> list[RetrievedChunk]:
        weights = RANKING_ARMS[arm]
        query_embedding = self._embedder.embed_one(query)
        candidates = self._vector_store.query(query_embedding, k=self._fetch_k)
        reranked = _rerank(candidates, weights)[:k]
        return [
            RetrievedChunk(
                doc_id=m.id,
                source=str(m.metadata.get("source", "unknown")),
                text=m.text,
                score=m.score,
            )
            for m in reranked
        ]
