"""Human review resume routes."""

from fastapi import APIRouter


router = APIRouter()


@router.post("/resume-review")
async def resume_review() -> dict:
    """Resume a paused review flow."""
    return {"message": "Not implemented yet"}
