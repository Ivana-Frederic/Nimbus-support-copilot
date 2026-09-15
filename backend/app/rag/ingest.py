"""Loads the markdown KB docs, chunks and embeds them, and upserts into the
vector store. Runs automatically at app startup if the index is empty, and
scripts/ingest_kb.py calls this too for a manual re-ingest after editing docs.

The recency_score/popularity_score in each doc's front matter are just
numbers I made up to stand in for "how fresh is this" and "how often does
this get used". app.rag.retriever is where those get combined with the
similarity score.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.rag.chunking import chunk_markdown
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorRecord, VectorStore

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta_block, body = match.groups()
    meta = {}
    for line in meta_block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body


def ingest_kb_dir(kb_dir: str, embedder: Embedder, vector_store: VectorStore) -> int:
    """Returns the number of chunks upserted."""
    records: list[VectorRecord] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        source = meta.get("source", path.stem.replace("-", " ").title())
        recency = float(meta.get("recency_score", 0.5))
        popularity = float(meta.get("popularity_score", 0.5))

        chunks = chunk_markdown(body)
        if not chunks:
            continue
        embeddings = embedder.embed([c.text for c in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = f"{path.stem}::{chunk.index}"
            records.append(
                VectorRecord(
                    id=doc_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        "source": source,
                        "heading": chunk.heading,
                        "file": path.name,
                        "recency_score": recency,
                        "popularity_score": popularity,
                    },
                )
            )

    vector_store.upsert(records)
    return len(records)
