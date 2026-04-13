"""Normalize routing-related state before executor dispatch."""

from app.graph.state import EditState


def route_executor(state: EditState) -> dict:
    """Prepare execution routing state."""

    return {
        "approval_required": state.get("approval_required", False),
    }
