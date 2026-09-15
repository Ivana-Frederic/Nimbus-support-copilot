from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.db import Database
from app.deps import get_bandit_policy, get_db
from app.rag.retriever import RANKING_ARMS
from app.rl.policy_store import BanditPolicy
from app.schemas import BanditArmStats, BanditStatsResponse

router = APIRouter(tags=["admin"])


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/admin/bandit-stats", response_model=BanditStatsResponse)
async def bandit_stats(
    db: Database = Depends(get_db),
    bandit: BanditPolicy = Depends(get_bandit_policy),
) -> BanditStatsResponse:
    """Shows what the RL policy has learned so far: how often each retrieval
    re-ranking arm was picked and its empirical positive-feedback rate.
    """
    counts = db.bandit_arm_counts()
    total_interactions, total_feedback = db.counts()

    arms = []
    for arm in RANKING_ARMS:
        c = counts.get(arm, {"selected": 0, "positive": 0, "negative": 0})
        rated = c["positive"] + c["negative"]
        reward = c["positive"] / rated if rated > 0 else 0.0
        arms.append(
            BanditArmStats(
                arm=arm,
                n_selected=c["selected"],
                n_positive=c["positive"],
                n_negative=c["negative"],
                empirical_reward=reward,
            )
        )

    return BanditStatsResponse(
        arms=arms, total_interactions=total_interactions, total_feedback=total_feedback
    )
