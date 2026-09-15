"""Builds the bandit's context feature vector and wraps a persisted
LinUCBBandit singleton so routers don't have to know about file I/O.
"""

from __future__ import annotations

import threading

import numpy as np

from app.classifier.labels import INTENTS
from app.rag.retriever import RANKING_ARMS
from app.rl.bandit import LinUCBBandit

_LENGTH_BUCKETS = ("short", "medium", "long")
CONTEXT_DIM = len(INTENTS) + len(_LENGTH_BUCKETS) + 1  # + bias term


def _length_bucket(query: str) -> str:
    n_words = len(query.split())
    if n_words <= 5:
        return "short"
    if n_words <= 15:
        return "medium"
    return "long"


def build_context(query: str, intent: str) -> list[float]:
    intent_one_hot = [1.0 if intent == i else 0.0 for i in INTENTS]
    bucket = _length_bucket(query)
    length_one_hot = [1.0 if bucket == b else 0.0 for b in _LENGTH_BUCKETS]
    return [*intent_one_hot, *length_one_hot, 1.0]  # trailing 1.0 is the bias term


class BanditPolicy:
    """Thread-safe wrapper: select an arm, later update it from feedback."""

    def __init__(self, state_path: str, alpha: float = 1.0):
        self._state_path = state_path
        self._lock = threading.Lock()
        self._bandit = LinUCBBandit.load_or_create(
            state_path, arms=list(RANKING_ARMS.keys()), context_dim=CONTEXT_DIM, alpha=alpha
        )

    def select_arm(self, query: str, intent: str) -> tuple[str, list[float]]:
        context = build_context(query, intent)
        with self._lock:
            arm = self._bandit.select_arm(np.array(context))
        return arm, context

    def record_reward(self, arm: str, context: list[float], reward: float) -> None:
        with self._lock:
            self._bandit.update(arm, np.array(context), reward)
            self._bandit.save(self._state_path)

    def snapshot(self) -> LinUCBBandit:
        with self._lock:
            return self._bandit
