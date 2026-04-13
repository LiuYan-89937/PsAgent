"""Build a structured edit plan."""

from app.graph.state import EditState


def build_plan(state: EditState) -> dict:
    """Merge image analysis, user input, and preferences into a plan."""

    return {
        "edit_plan": state.get("edit_plan"),
    }
