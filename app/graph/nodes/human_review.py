"""Human review checkpoint node."""

from app.graph.state import EditState


def human_review(state: EditState) -> dict:
    """Pause or resume high-risk edits."""

    return {
        "approval_payload": state.get("approval_payload"),
    }
