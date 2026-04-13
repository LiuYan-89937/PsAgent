"""Analyze input image content and issues."""

from app.graph.state import EditState


def analyze_image(state: EditState) -> dict:
    """Analyze image domain, tags, and quality issues."""

    return {
        "image_analysis": state.get("image_analysis"),
    }
