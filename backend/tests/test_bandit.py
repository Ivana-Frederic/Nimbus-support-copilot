import json

import numpy as np
import pytest

from app.rl.bandit import LinUCBBandit


def test_select_arm_returns_one_of_the_configured_arms():
    bandit = LinUCBBandit(arms=["a", "b", "c"], context_dim=3)
    arm = bandit.select_arm(np.array([1.0, 0.0, 0.0]))
    assert arm in {"a", "b", "c"}


def test_converges_to_better_arm_for_a_given_context():
    """Arm 'good' always rewards 1.0 for this context, 'bad' always rewards
    0.0. After enough rounds the bandit should mostly pick 'good'."""
    bandit = LinUCBBandit(arms=["good", "bad"], context_dim=2, alpha=0.5)
    context = np.array([1.0, 0.0])

    for _ in range(200):
        arm = bandit.select_arm(context)
        reward = 1.0 if arm == "good" else 0.0
        bandit.update(arm, context, reward)

    # after learning, exploitation should dominate: check the *greedy*
    # (alpha=0) choice, which isolates the learned mean from exploration bonus
    greedy = LinUCBBandit.from_state(bandit.to_state())
    greedy.alpha = 0.0
    choices = [greedy.select_arm(context) for _ in range(20)]
    assert choices.count("good") >= 18


def test_reward_isolated_per_context():
    """A context that never received reward for an arm should not be
    affected by rewards observed under a different, orthogonal context."""
    bandit = LinUCBBandit(arms=["x", "y"], context_dim=2, alpha=0.0)
    ctx_a = np.array([1.0, 0.0])
    ctx_b = np.array([0.0, 1.0])

    for _ in range(50):
        bandit.update("x", ctx_a, 1.0)
        bandit.update("y", ctx_a, 0.0)

    rewards_b = bandit.expected_rewards(ctx_b)
    # untouched context should stay near its ridge-regression prior (~0)
    assert abs(rewards_b["x"]) < 0.3
    assert abs(rewards_b["y"]) < 0.3


def test_state_roundtrip_preserves_behavior():
    bandit = LinUCBBandit(arms=["a", "b"], context_dim=2, alpha=1.0)
    context = np.array([0.5, 0.5])
    bandit.update("a", context, 1.0)
    bandit.update("b", context, 0.0)

    restored = LinUCBBandit.from_state(bandit.to_state())
    assert restored.arms == bandit.arms
    assert restored.expected_rewards(context) == pytest.approx(
        bandit.expected_rewards(context)
    )


def test_save_and_load_or_create_roundtrip(tmp_path):
    path = tmp_path / "bandit.json"
    bandit = LinUCBBandit(arms=["a", "b"], context_dim=2, alpha=1.0)
    bandit.update("a", np.array([1.0, 0.0]), 1.0)
    bandit.save(str(path))

    assert path.exists()
    loaded = LinUCBBandit.load_or_create(str(path), arms=["a", "b"], context_dim=2)
    assert loaded.expected_rewards(np.array([1.0, 0.0])) == pytest.approx(
        bandit.expected_rewards(np.array([1.0, 0.0]))
    )


def test_load_or_create_falls_back_on_arm_mismatch(tmp_path):
    path = tmp_path / "bandit.json"
    path.write_text(json.dumps({"arms": ["only_old_arm"], "context_dim": 2, "alpha": 1.0,
                                 "A": {"only_old_arm": [[1, 0], [0, 1]]},
                                 "b": {"only_old_arm": [0, 0]}}))
    loaded = LinUCBBandit.load_or_create(str(path), arms=["a", "b"], context_dim=2)
    assert loaded.arms == ["a", "b"]


def test_rejects_empty_arms():
    with pytest.raises(ValueError):
        LinUCBBandit(arms=[], context_dim=2)
