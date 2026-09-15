from app.rag.retriever import RANKING_ARMS


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_answer_sources_and_arm(client, fake_ollama):
    resp = client.post(
        "/api/chat", json={"session_id": "s1", "message": "Can I cancel and get a refund?"}
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == fake_ollama.canned_answer
    assert body["intent"] == "refund"
    assert body["arm"] in RANKING_ARMS
    assert body["latency_ms"] > 0
    assert len(body["sources"]) == 1
    assert body["sources"][0]["doc_id"] == "billing-faq::0"
    assert body["interaction_id"]


def test_chat_passes_retrieved_context_into_llm_prompt(client, fake_ollama):
    client.post("/api/chat", json={"session_id": "s1", "message": "What's my refund policy?"})
    assert len(fake_ollama.received_messages) == 1
    user_message = fake_ollama.received_messages[0][1]["content"]
    assert "Refunds within 14 days" in user_message


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/chat", json={"session_id": "s1", "message": ""})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post("/api/chat", json={"session_id": "s1", "message": "hello"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "copilot_chat_requests_total" in resp.text
