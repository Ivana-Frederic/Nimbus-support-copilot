from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.classifier.intent_classifier import load_intent_classifier
from app.config import get_settings
from app.db import Database
from app.llm.ollama_client import OllamaClient
from app.rag.embeddings import get_embedder
from app.rag.ingest import ingest_kb_dir
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore
from app.rl.policy_store import BanditPolicy
from app.routers import admin, chat, feedback, health

logger = logging.getLogger("copilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    embedder = get_embedder(settings.embedding_model_name)
    vector_store = ChromaVectorStore(settings.chroma_path)
    if vector_store.count() == 0:
        logger.info("Vector store empty, auto-ingesting knowledge base from %s", settings.kb_dir)
        n = ingest_kb_dir(settings.kb_dir, embedder, vector_store)
        logger.info("Ingested %d chunks", n)

    app.state.retriever = Retriever(embedder, vector_store)
    app.state.ollama_client = OllamaClient(
        settings.ollama_base_url, settings.ollama_model, settings.ollama_request_timeout_s
    )
    app.state.intent_classifier = load_intent_classifier(settings.intent_model_dir)
    app.state.bandit_policy = BanditPolicy(settings.bandit_state_path, settings.bandit_alpha)
    app.state.db = Database(settings.db_path)

    logger.info("Copilot backend ready (intent classifier: %s)", type(app.state.intent_classifier).__name__)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Nimbus Support Copilot", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(feedback.router)
    app.include_router(admin.router)
    return app


app = create_app()
