"""Build the round-search objective card."""

from __future__ import annotations

from app.graph.state import EditState, coerce_image_analysis, coerce_request_intent
from app.services.search_agent import build_objective_card as infer_objective_card


def build_objective(state: EditState) -> dict[str, object]:
    """Build a search objective from request intent and image analysis."""

    objective = infer_objective_card(
        request_text=str(state.get("request_text") or ""),
        request_intent=coerce_request_intent(state.get("request_intent")),
        image_analysis=coerce_image_analysis(state.get("image_analysis")),
        mode="auto",
    )
    return {"objective_card": objective.model_dump(mode="json")}
