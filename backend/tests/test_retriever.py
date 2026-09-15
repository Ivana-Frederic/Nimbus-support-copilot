from app.rag.retriever import RANKING_ARMS, Retriever
from app.rag.vector_store import VectorMatch


class FakeEmbedder:
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    """Returns a fixed candidate set. One doc is most semantically similar
    but old and unpopular, the other is less similar but recent and
    popular, so we can check the rerank actually uses the metadata.
    """

    def __init__(self, matches: list[VectorMatch]):
        self._matches = matches

    def upsert(self, records):  # pragma: no cover - unused in these tests
        raise NotImplementedError

    def query(self, embedding, k):
        return self._matches[:k]

    def count(self) -> int:
        return len(self._matches)


def _matches():
    return [
        VectorMatch(
            id="stale::0",
            text="stale but semantically closest",
            metadata={"source": "Stale Doc", "recency_score": 0.1, "popularity_score": 0.1},
            score=0.95,
        ),
        VectorMatch(
            id="fresh::0",
            text="fresh and popular, slightly less similar",
            metadata={"source": "Fresh Doc", "recency_score": 1.0, "popularity_score": 1.0},
            score=0.80,
        ),
    ]


def test_pure_semantic_arm_ranks_by_similarity_only():
    retriever = Retriever(FakeEmbedder(), FakeVectorStore(_matches()))
    results = retriever.retrieve("anything", k=2, arm="pure_semantic")
    assert [r.doc_id for r in results] == ["stale::0", "fresh::0"]


def test_recency_heavy_arm_can_flip_the_ranking():
    retriever = Retriever(FakeEmbedder(), FakeVectorStore(_matches()))
    results = retriever.retrieve("anything", k=2, arm="recency_heavy")
    assert results[0].doc_id == "fresh::0"


def test_retrieve_respects_k():
    retriever = Retriever(FakeEmbedder(), FakeVectorStore(_matches()))
    results = retriever.retrieve("anything", k=1, arm="balanced")
    assert len(results) == 1


def test_has_index_reflects_vector_store_count():
    empty = Retriever(FakeEmbedder(), FakeVectorStore([]))
    assert empty.has_index() is False
    nonempty = Retriever(FakeEmbedder(), FakeVectorStore(_matches()))
    assert nonempty.has_index() is True


def test_all_configured_arms_are_usable():
    retriever = Retriever(FakeEmbedder(), FakeVectorStore(_matches()))
    for arm in RANKING_ARMS:
        results = retriever.retrieve("anything", k=2, arm=arm)
        assert len(results) == 2
