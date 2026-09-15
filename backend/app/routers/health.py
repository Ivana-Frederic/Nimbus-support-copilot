from fastapi import APIRouter, Depends

from app.deps import get_ollama_client, get_retriever
from app.llm.ollama_client import OllamaClient
from app.rag.retriever import Retriever

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe: process is up and serving requests."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    retriever: Retriever = Depends(get_retriever),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> dict:
    """Readiness probe: dependencies (vector store populated, LLM reachable)
    are actually usable, not just that the process started."""
    kb_ready = retriever.has_index()
    llm_ready = await ollama.health()
    ready_ = kb_ready and llm_ready
    body = {"ready": ready_, "kb_ready": kb_ready, "llm_ready": llm_ready}
    return body
