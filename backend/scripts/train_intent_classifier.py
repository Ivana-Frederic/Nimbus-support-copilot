#!/usr/bin/env python
"""Fine-tunes distilbert-base-uncased into a 7-way intent classifier.

backend/data/intents/train.csv is a small synthetic set I made up, just
enough to run the pipeline end to end. Wouldn't trust the accuracy for
anything real, you'd want actual labeled tickets for that.

Usage (from backend/, needs torch + transformers + scikit-learn):
    python -m scripts.train_intent_classifier
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_intent_classifier")


def load_dataset(csv_path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def main() -> None:
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from app.classifier.labels import INTENTS
    from app.config import get_settings

    settings = get_settings()
    label2id = {label: i for i, label in enumerate(INTENTS)}
    id2label = {i: label for label, i in label2id.items()}

    csv_path = Path(__file__).resolve().parent.parent / "data" / "intents" / "train.csv"
    texts, labels = load_dataset(csv_path)
    label_ids = [label2id[label] for label in labels]

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
    )

    logger.info("Loading base model %s ...", settings.intent_base_model)
    tokenizer = AutoTokenizer.from_pretrained(settings.intent_base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.intent_base_model,
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id,
    )

    class IntentDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int]):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = IntentDataset(train_texts, train_labels)
    eval_ds = IntentDataset(eval_texts, eval_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = float((preds == labels).mean())
        return {"accuracy": accuracy}

    output_dir = Path(settings.intent_model_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "_checkpoints"),
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        learning_rate=5e-5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved fine-tuned intent classifier to %s", output_dir)


if __name__ == "__main__":
    main()
