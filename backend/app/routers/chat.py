from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.classifier.intent_classifier import IntentClassifier, normalize
from app.config import Settings, get_settings
from app.db import Database
from app.deps import get_bandit_policy, get_db, get_intent_classifier, get_ollama_client, get_retriever
from app.llm.ollama_client import OllamaClient
from app.llm.prompt import build_messages
from app.metrics import BANDIT_ARM_SELECTED, CHAT_LATENCY, CHAT_REQUESTS, LLM_LATENCY, RETRIEVAL_LATENCY
from app.rag.retriever import Retriever
from app.rl.policy_store import BanditPolicy
from app.schemas import ChatRequest, ChatResponse, SourceChunk

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    ollama: OllamaClient = Depends(get_ollama_client),
    intent_classifier: IntentClassifier = Depends(get_intent_classifier),
    bandit: BanditPolicy = Depends(get_bandit_policy),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    start = time.perf_counter()
    query = normalize(body.message)

    intent = intent_classifier.predict(query)
    arm, context = bandit.select_arm(query, intent)
    BANDIT_ARM_SELECTED.labels(arm=arm).inc()

    t0 = time.perf_counter()
    chunks = retriever.retrieve(query, k=settings.top_k, arm=arm)
    RETRIEVAL_LATENCY.observe(time.perf_counter() - t0)

    messages = build_messages(query, chunks)

    t1 = time.perf_counter()
    answer = await ollama.chat(messages)
    LLM_LATENCY.observe(time.perf_counter() - t1)

    interaction_id = str(uuid.uuid4())
    latency_ms = (time.perf_counter() - start) * 1000

    db.save_interaction(
        interaction_id=interaction_id,
        session_id=body.session_id,
        query=query,
        intent=intent,
        arm=arm,
        context=context,
        source_doc_ids=[c.doc_id for c in chunks],
        answer=answer,
        latency_ms=latency_ms,
    )

    CHAT_REQUESTS.labels(intent=intent).inc()
    CHAT_LATENCY.observe(latency_ms / 1000)

    return ChatResponse(
        interaction_id=interaction_id,
        answer=answer,
        intent=intent,
        sources=[SourceChunk(doc_id=c.doc_id, source=c.source, text=c.text, score=c.score) for c in chunks],
        arm=arm,
        latency_ms=latency_ms,
    )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    ollama: OllamaClient = Depends(get_ollama_client),
    intent_classifier: IntentClassifier = Depends(get_intent_classifier),
    bandit: BanditPolicy = Depends(get_bandit_policy),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Server-Sent Events stream: a `meta` event with interaction id/sources
    first, then incremental `token` events as the LLM generates, then `done`.
    """
    start = time.perf_counter()
    query = normalize(body.message)
    intent = intent_classifier.predict(query)
    arm, context = bandit.select_arm(query, intent)
    BANDIT_ARM_SELECTED.labels(arm=arm).inc()
    chunks = retriever.retrieve(query, k=settings.top_k, arm=arm)
    messages = build_messages(query, chunks)
    interaction_id = str(uuid.uuid4())

    async def event_stream():
        meta = {
            "interaction_id": interaction_id,
            "intent": intent,
            "arm": arm,
            "sources": [
                {"doc_id": c.doc_id, "source": c.source, "text": c.text, "score": c.score} for c in chunks
            ],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        full_answer = []
        async for piece in ollama.chat_stream(messages):
            full_answer.append(piece)
            yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"

        answer = "".join(full_answer)
        latency_ms = (time.perf_counter() - start) * 1000
        db.save_interaction(
            interaction_id=interaction_id,
            session_id=body.session_id,
            query=query,
            intent=intent,
            arm=arm,
            context=context,
            source_doc_ids=[c.doc_id for c in chunks],
            answer=answer,
            latency_ms=latency_ms,
        )
        CHAT_REQUESTS.labels(intent=intent).inc()
        CHAT_LATENCY.observe(latency_ms / 1000)
        yield f"event: done\ndata: {json.dumps({'latency_ms': latency_ms})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
