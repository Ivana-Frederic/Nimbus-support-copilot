"""sqlite3 wrapper for interaction/feedback logging. Kept it plain stdlib
since this is just a low-volume log, didn't feel like pulling in an ORM
for something this small.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    intent TEXT NOT NULL,
    arm TEXT NOT NULL,
    context_json TEXT NOT NULL,
    source_doc_ids_json TEXT NOT NULL,
    answer TEXT NOT NULL,
    latency_ms REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    interaction_id TEXT PRIMARY KEY,
    rating TEXT NOT NULL,
    ts REAL NOT NULL,
    FOREIGN KEY (interaction_id) REFERENCES interactions (id)
);
"""


class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_interaction(
        self,
        *,
        interaction_id: str,
        session_id: str,
        query: str,
        intent: str,
        arm: str,
        context: list[float],
        source_doc_ids: list[str],
        answer: str,
        latency_ms: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO interactions
                   (id, session_id, ts, query, intent, arm, context_json,
                    source_doc_ids_json, answer, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction_id,
                    session_id,
                    time.time(),
                    query,
                    intent,
                    arm,
                    json.dumps(context),
                    json.dumps(source_doc_ids),
                    answer,
                    latency_ms,
                ),
            )

    def get_interaction(self, interaction_id: str) -> dict | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
            return dict(row) if row else None

    def save_feedback(self, interaction_id: str, rating: str) -> bool:
        """Returns False if the interaction id is unknown or already rated."""
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
            if not exists:
                return False
            already_rated = conn.execute(
                "SELECT 1 FROM feedback WHERE interaction_id = ?", (interaction_id,)
            ).fetchone()
            if already_rated:
                return False
            conn.execute(
                "INSERT INTO feedback (interaction_id, rating, ts) VALUES (?, ?, ?)",
                (interaction_id, rating, time.time()),
            )
            return True

    def bandit_arm_counts(self) -> dict[str, dict[str, int]]:
        """arm -> {selected, positive, negative} joined against feedback."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT i.arm AS arm, f.rating AS rating, COUNT(*) AS n
                   FROM interactions i
                   LEFT JOIN feedback f ON f.interaction_id = i.id
                   GROUP BY i.arm, f.rating"""
            ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for row in rows:
            arm = row["arm"]
            counts.setdefault(arm, {"selected": 0, "positive": 0, "negative": 0})
            n = row["n"]
            counts[arm]["selected"] += n
            if row["rating"] == "up":
                counts[arm]["positive"] += n
            elif row["rating"] == "down":
                counts[arm]["negative"] += n
        return counts

    def counts(self) -> tuple[int, int]:
        with self._connect() as conn:
            total_interactions = conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]
            total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return total_interactions, total_feedback
