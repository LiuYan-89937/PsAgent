"""Build a stable edit profile for stage activation and tool exposure."""

from __future__ import annotations

from app.graph.state import EditState, coerce_image_analysis, coerce_request_intent
from app.services.edit_profile import build_edit_profile as infer_edit_profile


def build_edit_profile(state: EditState) -> dict:
    """Build an edit profile from request intent and image analysis."""

    profile = infer_edit_profile(
        request_text=str(state.get("request_text") or ""),
        request_intent=coerce_request_intent(state.get("request_intent")),
        image_analysis=coerce_image_analysis(state.get("image_analysis")),
    )
    return {"edit_profile": profile.model_dump(mode="json")}
