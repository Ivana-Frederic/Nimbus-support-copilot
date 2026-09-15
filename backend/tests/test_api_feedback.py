def _post_chat(client, message="Can I cancel and get a refund?"):
    resp = client.post("/api/chat", json={"session_id": "s1", "message": message})
    assert resp.status_code == 200
    return resp.json()


def test_feedback_updates_bandit_and_is_reflected_in_stats(client):
    chat = _post_chat(client)

    resp = client.post(
        "/api/feedback", json={"interaction_id": chat["interaction_id"], "rating": "up"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bandit_updated"] is True
    assert body["rating"] == "up"

    stats = client.get("/api/admin/bandit-stats").json()
    assert stats["total_interactions"] == 1
    assert stats["total_feedback"] == 1
    chosen_arm_stats = next(a for a in stats["arms"] if a["arm"] == chat["arm"])
    assert chosen_arm_stats["n_selected"] == 1
    assert chosen_arm_stats["n_positive"] == 1


def test_feedback_on_unknown_interaction_returns_404(client):
    resp = client.post("/api/feedback", json={"interaction_id": "does-not-exist", "rating": "up"})
    assert resp.status_code == 404


def test_feedback_cannot_be_submitted_twice_for_same_interaction(client):
    chat = _post_chat(client)
    first = client.post(
        "/api/feedback", json={"interaction_id": chat["interaction_id"], "rating": "up"}
    )
    assert first.status_code == 200

    second = client.post(
        "/api/feedback", json={"interaction_id": chat["interaction_id"], "rating": "down"}
    )
    assert second.status_code == 404


def test_feedback_rejects_invalid_rating(client):
    chat = _post_chat(client)
    resp = client.post(
        "/api/feedback", json={"interaction_id": chat["interaction_id"], "rating": "meh"}
    )
    assert resp.status_code == 422


def test_bandit_reward_shrinks_toward_zero_after_negative_feedback(real_bandit):
    """Checks the same RL update that POST /api/feedback triggers. An
    initial up-vote (reward=1) should raise the expected reward for that
    (arm, context) above zero, and later down-votes (reward=0) should pull
    it back down. See routers/feedback.py for where this actually runs.
    """
    from app.rl.policy_store import build_context

    bandit = real_bandit.snapshot()
    context = build_context("Can I cancel and get a refund?", "refund")
    arm = "balanced"

    bandit.update(arm, context, 1.0)
    after_up = bandit.expected_rewards(context)[arm]
    assert after_up > 0

    for _ in range(20):
        bandit.update(arm, context, 0.0)
    after_downvotes = bandit.expected_rewards(context)[arm]

    assert after_downvotes < after_up


def test_stats_aggregate_correctly_across_multiple_interactions(client):
    for i in range(3):
        chat = _post_chat(client, message=f"Can I cancel and get a refund? ({i})")
        rating = "up" if i % 2 == 0 else "down"
        client.post("/api/feedback", json={"interaction_id": chat["interaction_id"], "rating": rating})

    stats = client.get("/api/admin/bandit-stats").json()
    assert stats["total_interactions"] == 3
    assert stats["total_feedback"] == 3
    assert sum(a["n_selected"] for a in stats["arms"]) == 3
    for a in stats["arms"]:
        assert a["n_positive"] + a["n_negative"] <= a["n_selected"]
