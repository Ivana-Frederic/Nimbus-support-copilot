"""The heavy singletons live on app.state (set up in main.py's lifespan) and
get handed out here via Depends. Routers never import them directly, which
is what lets tests swap in fakes through app.dependency_overrides.
"""

from __future__ import annotations

from fastapi import Request

from app.classifier.intent_classifier import IntentClassifier
from app.db import Database
from app.llm.ollama_client import OllamaClient
from app.rag.retriever import Retriever
from app.rl.policy_store import BanditPolicy


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_intent_classifier(request: Request) -> IntentClassifier:
    return request.app.state.intent_classifier


def get_bandit_policy(request: Request) -> BanditPolicy:
    return request.app.state.bandit_policy


def get_db(request: Request) -> Database:
    return request.app.state.db
