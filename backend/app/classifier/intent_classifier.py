"""Two intent classifiers behind the same Protocol: a fine-tuned DistilBERT
(see scripts/train_intent_classifier.py) and a keyword-matching fallback for
when nobody's run the training script yet. main.py picks whichever's
available at startup.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from app.classifier.labels import INTENTS

_KEYWORDS: dict[str, list[str]] = {
    "billing": ["invoice", "charge", "charged", "billing", "payment", "price", "plan", "subscription"],
    "technical_api": ["api", "endpoint", "error", "500", "rate limit", "token", "sdk", "webhook", "integration"],
    "outage": ["down", "outage", "incident", "unavailable", "status page", "degraded"],
    "security": ["security", "breach", "password", "2fa", "mfa", "compliance", "soc 2", "gdpr", "encryption"],
    "onboarding": ["get started", "onboarding", "setup", "set up", "new account", "first time", "how do i start"],
    "refund": ["refund", "cancel", "cancellation", "money back", "chargeback"],
}


class IntentClassifier(Protocol):
    def predict(self, text: str) -> str: ...


class KeywordIntentClassifier:
    """Deterministic fallback: scores each intent by keyword hits."""

    def predict(self, text: str) -> str:
        lowered = text.lower()
        best_intent = "other"
        best_hits = 0
        for intent, keywords in _KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits > best_hits:
                best_hits = hits
                best_intent = intent
        return best_intent


class TransformerIntentClassifier:
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        self._id2label = self._model.config.id2label

    def predict(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        pred_id = int(logits.argmax(dim=-1).item())
        return self._id2label[pred_id]


def load_intent_classifier(model_dir: str) -> IntentClassifier:
    weights_present = Path(model_dir).exists() and any(
        p.suffix in {".safetensors", ".bin"} for p in Path(model_dir).glob("*")
    )
    if weights_present:
        try:
            return TransformerIntentClassifier(model_dir)
        except Exception:
            pass
    return KeywordIntentClassifier()


_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


assert set(_KEYWORDS.keys()) <= set(INTENTS)
