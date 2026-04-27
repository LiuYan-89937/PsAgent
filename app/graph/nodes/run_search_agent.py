"""Run the round-first search agent."""

from __future__ import annotations

from langgraph.config import get_stream_writer

from app.graph.state import (
    EditPlan,
    EditState,
    ObjectiveCard,
    coerce_edit_plan,
    coerce_execution_trace,
    coerce_mask_catalog,
    coerce_request_intent,
    coerce_search_rounds,
    coerce_segmentation_trace,
)
from app.services.search_agent import run_search_first_agent
from app.services.search_agent.config import resolve_search_round_limits


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
    existing_rounds = coerce_search_rounds(state.get("rounds") or [])
    existing_candidate_outputs = list(state.get("candidate_outputs") or [])
    existing_execution_trace = coerce_execution_trace(state.get("execution_trace") or [])
    existing_final_execution_trace = coerce_execution_trace(state.get("final_execution_trace") or state.get("execution_trace") or [])
    existing_segmentation_trace = coerce_segmentation_trace(state.get("segmentation_trace") or [])
    existing_fallback_trace = list(state.get("fallback_trace") or [])
    mode = str(state.get("mode") or "explicit")
    round_offset = len(existing_rounds) if mode == "auto" else 0
    cycle_round_offset = 0
    if mode == "auto":
        round_limits = resolve_search_round_limits(state.get("search_effort"))
        if state.get("human_review_continuation") and not state.get("search_cycle_round_offset"):
            cycle_round_offset = len(existing_rounds)
        else:
            try:
                cycle_round_offset = int(state.get("search_cycle_round_offset") or 0)
            except (TypeError, ValueError):
                cycle_round_offset = 0
        cycle_round_count = max(len(existing_rounds) - max(cycle_round_offset, 0), 0)
        remaining_rounds = max(round_limits.max_rounds - cycle_round_count, 0)
        remaining_min_rounds = max(round_limits.min_rounds - cycle_round_count, 0)
    else:
        remaining_rounds = None
        remaining_min_rounds = None
    result = run_search_first_agent(
        input_image_path=current_image,
        objective=objective,
        request_intent=coerce_request_intent(state.get("request_intent")),
        mode=mode,
        mask_catalog=coerce_mask_catalog(state.get("mask_catalog")),
        tool_catalog=[item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in state.get("tool_catalog") or []],
        writer=_safe_stream_writer(),
        round_index_offset=round_offset,
        min_rounds=remaining_min_rounds,
        max_rounds=remaining_rounds,
    )
    new_rounds = coerce_search_rounds(result.get("rounds") or [])
    result["rounds"] = existing_rounds + new_rounds
    result["candidate_outputs"] = existing_candidate_outputs + list(result.get("candidate_outputs") or [])
    result["execution_trace"] = existing_execution_trace + coerce_execution_trace(result.get("execution_trace") or [])
    result["final_execution_trace"] = existing_final_execution_trace + coerce_execution_trace(result.get("final_execution_trace") or [])
    result["segmentation_trace"] = existing_segmentation_trace + coerce_segmentation_trace(result.get("segmentation_trace") or [])
    result["fallback_trace"] = existing_fallback_trace + list(result.get("fallback_trace") or [])

    previous_plan = coerce_edit_plan(state.get("edit_plan"))
    current_plan = coerce_edit_plan(result.get("edit_plan"))
    if previous_plan is not None and current_plan is not None:
        result["edit_plan"] = EditPlan(
            mode=current_plan.mode,
            domain=current_plan.domain,
            executor=current_plan.executor,
            preserve=list(dict.fromkeys([*previous_plan.preserve, *current_plan.preserve])),
            operations=[*previous_plan.operations, *current_plan.operations],
            should_write_memory=previous_plan.should_write_memory or current_plan.should_write_memory,
            memory_candidates=[*previous_plan.memory_candidates, *current_plan.memory_candidates],
            needs_confirmation=previous_plan.needs_confirmation or current_plan.needs_confirmation,
        )
    result["objective_card"] = objective.model_dump(mode="json")
    result["needs_search_continuation"] = False
    result["search_continuation_reason"] = None
    result["human_review_continuation"] = False
    result["search_cycle_round_offset"] = cycle_round_offset
    result["approval_required"] = False
    result["approval_payload"] = None
    if result.get("rounds"):
        last_round = result["rounds"][-1]
        result["current_round"] = last_round.id
        result["current_focus"] = last_round.focus
    return result
