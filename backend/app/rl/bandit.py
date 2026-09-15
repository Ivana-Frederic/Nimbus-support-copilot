"""LinUCB contextual bandit, based on the algorithm from Li et al. 2010's
personalized news recommendation paper. I'm using it here to pick a
retrieval ranking strategy instead of a news article.

For each arm we keep a ridge-regression estimate of the reward (that's the
A and b below), plus an extra term that grows when we haven't tried this
arm/context combo much yet. That extra term is what makes it explore arms
it hasn't tried instead of just sticking with whatever worked first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinUCBBandit:
    def __init__(self, arms: list[str], context_dim: int, alpha: float = 1.0):
        if not arms:
            raise ValueError("LinUCBBandit needs at least one arm")
        self.arms = list(arms)
        self.context_dim = context_dim
        self.alpha = alpha
        self._A = {arm: np.eye(context_dim) for arm in self.arms}
        self._b = {arm: np.zeros(context_dim) for arm in self.arms}

    def select_arm(self, context: np.ndarray) -> str:
        context = np.asarray(context, dtype=float).reshape(-1)
        best_arm = self.arms[0]
        best_score = -np.inf
        for arm in self.arms:
            A_inv = np.linalg.inv(self._A[arm])
            theta = A_inv @ self._b[arm]
            mean = float(theta @ context)
            exploration = self.alpha * float(np.sqrt(context @ A_inv @ context))
            score = mean + exploration
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        context = np.asarray(context, dtype=float).reshape(-1)
        self._A[arm] += np.outer(context, context)
        self._b[arm] += reward * context

    def expected_rewards(self, context: np.ndarray) -> dict[str, float]:
        context = np.asarray(context, dtype=float).reshape(-1)
        out = {}
        for arm in self.arms:
            A_inv = np.linalg.inv(self._A[arm])
            theta = A_inv @ self._b[arm]
            out[arm] = float(theta @ context)
        return out

    def to_state(self) -> dict:
        return {
            "arms": self.arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "A": {arm: self._A[arm].tolist() for arm in self.arms},
            "b": {arm: self._b[arm].tolist() for arm in self.arms},
        }

    @classmethod
    def from_state(cls, state: dict) -> "LinUCBBandit":
        bandit = cls(state["arms"], state["context_dim"], state["alpha"])
        for arm in bandit.arms:
            bandit._A[arm] = np.array(state["A"][arm])
            bandit._b[arm] = np.array(state["b"][arm])
        return bandit

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_state()))

    @classmethod
    def load_or_create(
        cls, path: str, arms: list[str], context_dim: int, alpha: float = 1.0
    ) -> "LinUCBBandit":
        p = Path(path)
        if p.exists():
            try:
                state = json.loads(p.read_text())
                loaded = cls.from_state(state)
                if loaded.arms == list(arms) and loaded.context_dim == context_dim:
                    return loaded
            except (json.JSONDecodeError, KeyError):
                pass
        return cls(arms, context_dim, alpha)
