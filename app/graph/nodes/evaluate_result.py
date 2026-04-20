"""Final review node with one final critic call."""

from __future__ import annotations

from typing import Any
from pydantic import ValidationError

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import (
    CriticResult,
    EditState,
    EvaluationReport,
    coerce_approval_payload,
    coerce_edit_plan,
    coerce_execution_trace,
    coerce_image_analysis,
    coerce_phase_artifacts,
)
from app.services.critic_model import critic_model_available, evaluate_edit_result_with_qwen
from app.services.stage_policy import STAGE_ORDER


def _safe_image_analysis(state: EditState):
    value = state.get("image_analysis")
    if value is None:
        return None
    try:
        return coerce_image_analysis(value)
    except ValidationError:
        return None


def _safe_edit_plan(state: EditState):
    value = state.get("edit_plan")
    if value is None:
        return None
    try:
        return coerce_edit_plan(value)
    except ValidationError:
        return None


def _build_base_report(state: EditState) -> dict[str, Any]:
    """Build execution-fact based evaluation fields."""

    execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("execution_trace") or [])]
    return {
        "selected_output": state.get("selected_output"),
        "num_operations": len(execution_trace),
        "success_count": sum(1 for item in execution_trace if item.get("ok")),
        "failure_count": sum(1 for item in execution_trace if item.get("ok") is False),
        "fallback_count": sum(1 for item in execution_trace if item.get("fallback_used")),
        "has_output": bool(state.get("selected_output")),
    }


def _run_critic(state: EditState) -> tuple[CriticResult | None, str | None]:
    """Run the final critic model when enough inputs are present."""

    input_images = state.get("input_images") or []
    selected_output = state.get("selected_output")
    if not (critic_model_available() and input_images and selected_output):
        if input_images and selected_output:
            return None, "critic unavailable"
        return None, None

    edit_plan = _safe_edit_plan(state)
    image_analysis = _safe_image_analysis(state)
    execution_trace = coerce_execution_trace(state.get("execution_trace") or [])
    try:
        model_report = evaluate_edit_result_with_qwen(
            original_image_path=input_images[0],
            edited_image_path=selected_output,
            request_text=str(state.get("request_text") or ""),
            edit_plan=edit_plan.model_dump(mode="json") if edit_plan is not None else {},
            image_analysis=image_analysis.model_dump(mode="json") if image_analysis is not None else {},
            execution_trace=[item.model_dump(mode="json") for item in execution_trace],
        )
    except RuntimeError as error:
        return None, str(error)

    return CriticResult.model_validate(model_report), None


def final_review(state: EditState) -> dict[str, Any]:
    """Produce the final evaluation report after all stages finish."""

    base_report = _build_base_report(state)
    critic, critic_error = _run_critic(state)
    fallback_trace = list(state.get("fallback_trace") or [])
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))
    elif critic_error:
        fallback_trace = append_fallback_trace(
            fallback_trace,
            stage="final_review",
            source="critic_model",
            location="eval_report",
            strategy="execution_only_evaluation",
            message="结果评估模型不可用，改用执行事实评估。",
            error=critic_error,
        )

    report = EvaluationReport.model_validate(base_report)
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    last_stage = next((stage for stage in reversed(STAGE_ORDER) if stage in phases), None)
    if last_stage is not None:
        phase = phases[last_stage]
        phase.eval_report = report
        phases[last_stage] = phase

    memory_candidates: list[dict[str, Any]] = []
    for stage_key in STAGE_ORDER:
        phase = phases.get(stage_key)
        if phase is None or phase.plan is None or not phase.plan.should_write_memory:
            continue
        for item in phase.plan.memory_candidates:
            if item not in memory_candidates:
                memory_candidates.append(item)

    return {
        "eval_report": report,
        "phases": phases,
        "approval_required": bool(state.get("approval_required")) or bool(report.should_request_review),
        "approval_payload": coerce_approval_payload(state.get("approval_payload")),
        "fallback_trace": fallback_trace,
        "memory_write_candidates": memory_candidates,
    }
