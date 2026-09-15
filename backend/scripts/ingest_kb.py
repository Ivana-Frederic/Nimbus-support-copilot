#!/usr/bin/env python
"""CLI: (re-)ingest the markdown knowledge base into the vector store.

Usage (from backend/):
    python -m scripts.ingest_kb
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.ingest import ingest_kb_dir  # noqa: E402
from app.rag.vector_store import ChromaVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_kb")


def main() -> None:
    settings = get_settings()
    logger.info("Loading embedding model %s ...", settings.embedding_model_name)
    embedder = get_embedder(settings.embedding_model_name)
    vector_store = ChromaVectorStore(settings.chroma_path)

    logger.info("Ingesting KB docs from %s ...", settings.kb_dir)
    n = ingest_kb_dir(settings.kb_dir, embedder, vector_store)
    logger.info("Done. Upserted %d chunks. Vector store now has %d chunks total.", n, vector_store.count())


if __name__ == "__main__":
    main()
