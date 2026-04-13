"""Edit request routes."""

from fastapi import APIRouter


router = APIRouter()


@router.post("/edit")
async def edit() -> dict:
    """Handle image edit requests."""
    return {"message": "Not implemented yet"}
