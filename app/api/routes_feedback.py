"""Feedback routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_job_store
from app.api.schemas import FeedbackRequest, FeedbackResponse
from app.services.job_store import JobStore

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    payload: FeedbackRequest,
    job_store: JobStore = Depends(get_job_store),
) -> FeedbackResponse:
    """Store frontend feedback against an existing job."""

    record = job_store.get(payload.job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    updated = job_store.append_feedback(
        payload.job_id,
        {
            "accepted": payload.accepted,
            "rating": payload.rating,
            "feedback_text": payload.feedback_text,
            "manual_adjustments": payload.manual_adjustments,
        },
    )
    return FeedbackResponse(
        job_id=payload.job_id,
        saved=True,
        feedback_count=len(updated.feedback),
    )
