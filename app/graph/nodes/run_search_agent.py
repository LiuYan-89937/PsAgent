"""Run the round-first search agent."""

from __future__ import annotations

from langgraph.config import get_stream_writer

from app.graph.state import EditState, ObjectiveCard, coerce_mask_catalog, coerce_request_intent
from app.services.search_agent import run_search_first_agent


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def run_search_agent(state: EditState) -> dict[str, object]:
    """Execute explicit direct mode or auto search-first orchestration."""

    input_images = list(state.get("input_images") or [])
    current_image = str(state.get("selected_output") or (input_images[0] if input_images else ""))
    if not current_image:
        raise ValueError("run_search_agent requires at least one input image.")

    objective_payload = state.get("objective_card")
    objective = objective_payload if isinstance(objective_payload, ObjectiveCard) else ObjectiveCard.model_validate(objective_payload or {})
    result = run_search_first_agent(
        input_image_path=current_image,
        objective=objective,
        request_intent=coerce_request_intent(state.get("request_intent")),
        mode=str(state.get("mode") or "explicit"),
        mask_catalog=coerce_mask_catalog(state.get("mask_catalog")),
        writer=_safe_stream_writer(),
    )
    result["objective_card"] = objective.model_dump(mode="json")
    if result.get("rounds"):
        last_round = result["rounds"][-1]
        result["current_round"] = last_round.id
        result["current_focus"] = last_round.focus
    return result
