from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from app.db import Database
from app.deps import get_bandit_policy, get_db
from app.metrics import FEEDBACK_TOTAL
from app.rl.policy_store import BanditPolicy
from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api", tags=["feedback"])

_REWARD = {"up": 1.0, "down": 0.0}


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    db: Database = Depends(get_db),
    bandit: BanditPolicy = Depends(get_bandit_policy),
) -> FeedbackResponse:
    saved = db.save_feedback(body.interaction_id, body.rating)
    if not saved:
        raise HTTPException(
            status_code=404, detail="Unknown interaction id, or feedback already recorded for it"
        )

    interaction = db.get_interaction(body.interaction_id)
    bandit_updated = False
    if interaction is not None:
        context = json.loads(interaction["context_json"])
        reward = _REWARD[body.rating]
        bandit.record_reward(interaction["arm"], context, reward)
        bandit_updated = True

    FEEDBACK_TOTAL.labels(rating=body.rating).inc()

    return FeedbackResponse(
        interaction_id=body.interaction_id, rating=body.rating, bandit_updated=bandit_updated
    )
