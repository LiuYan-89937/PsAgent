"""Feedback routes."""

from fastapi import APIRouter


router = APIRouter()


@router.post("/feedback")
async def feedback() -> dict:
    """Handle edit feedback."""
    return {"message": "Not implemented yet"}
