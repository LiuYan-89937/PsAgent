"""Evaluate edit outputs against quality and preference constraints."""

from app.graph.state import EditState


def evaluate_result(state: EditState) -> dict:
    """Produce a result evaluation report."""

    return {
        "eval_report": state.get("eval_report"),
        "approval_required": state.get("approval_required", False),
    }
