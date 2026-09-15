"""Vector store backend is Chroma, persistent and embedded so there's no
extra service to run. Behind a Protocol so tests can swap in a plain
in-memory store instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict


@dataclass
class VectorMatch:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...

    def query(self, embedding: list[float], k: int) -> list[VectorMatch]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str = "kb_chunks"):
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata for r in records],
        )

    def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
        )
        matches = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            # chroma cosine "distance" = 1 - cosine_similarity
            similarity = 1.0 - dist
            matches.append(VectorMatch(id=id_, text=doc, metadata=meta, score=similarity))
        return matches

    def count(self) -> int:
        return self._collection.count()
