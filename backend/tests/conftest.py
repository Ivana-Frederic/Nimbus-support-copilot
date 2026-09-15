"""Only the Retriever and Ollama client are faked here. Everything else
(KeywordIntentClassifier, BanditPolicy, Database) is the real code, so this
actually tests the retrieval -> bandit -> feedback wiring instead of just
checking that mocks got called.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier.intent_classifier import KeywordIntentClassifier  # noqa: E402
from app.db import Database  # noqa: E402
from app.deps import (  # noqa: E402
    get_bandit_policy,
    get_db,
    get_intent_classifier,
    get_ollama_client,
    get_retriever,
)
from app.main import create_app  # noqa: E402
from app.rag.retriever import RetrievedChunk  # noqa: E402
from app.rl.policy_store import BanditPolicy  # noqa: E402


class FakeRetriever:
    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []

    def has_index(self) -> bool:
        return True

    def retrieve(self, query: str, k: int, arm: str) -> list[RetrievedChunk]:
        self.calls.append((query, k, arm))
        return [
            RetrievedChunk(doc_id="billing-faq::0", source="Billing FAQ", text="Refunds within 14 days.", score=0.9)
        ][:k]


class FakeOllamaClient:
    def __init__(self, canned_answer: str = "Here is a canned support answer."):
        self.canned_answer = canned_answer
        self.received_messages: list[list[dict]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.received_messages.append(messages)
        return self.canned_answer

    async def chat_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.received_messages.append(messages)
        for word in self.canned_answer.split(" "):
            yield word + " "

    async def health(self) -> bool:
        return True


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def real_db(tmp_path) -> Database:
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def real_bandit(tmp_path) -> BanditPolicy:
    return BanditPolicy(str(tmp_path / "bandit.json"))


@pytest.fixture
def client(fake_retriever, fake_ollama, real_db, real_bandit) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: fake_retriever
    app.dependency_overrides[get_ollama_client] = lambda: fake_ollama
    app.dependency_overrides[get_intent_classifier] = lambda: KeywordIntentClassifier()
    app.dependency_overrides[get_db] = lambda: real_db
    app.dependency_overrides[get_bandit_policy] = lambda: real_bandit
    # Not entering `with TestClient(app)` as a context manager deliberately
    # skips the lifespan startup, avoiding the need for chromadb/torch/etc.
    # to be installed just to run these tests.
    return TestClient(app)
