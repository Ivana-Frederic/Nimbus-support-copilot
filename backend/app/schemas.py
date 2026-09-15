from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Opaque client-generated session identifier")
    message: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    doc_id: str
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    interaction_id: str
    answer: str
    intent: str
    sources: list[SourceChunk]
    arm: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    interaction_id: str
    rating: str
    bandit_updated: bool


class BanditArmStats(BaseModel):
    arm: str
    n_selected: int
    n_positive: int
    n_negative: int
    empirical_reward: float


class BanditStatsResponse(BaseModel):
    arms: list[BanditArmStats]
    total_interactions: int
    total_feedback: int
