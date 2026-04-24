"""Final review node with one final critic call."""

from __future__ import annotations

from pathlib import Path
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
)
from app.services.image_metrics import compute_image_metrics
from app.services.critic_model import critic_model_available, evaluate_edit_result_with_qwen


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
    report = {
        "selected_output": state.get("selected_output"),
        "num_operations": len(execution_trace),
        "success_count": sum(1 for item in execution_trace if item.get("ok")),
        "failure_count": sum(1 for item in execution_trace if item.get("ok") is False),
        "fallback_count": sum(1 for item in execution_trace if item.get("fallback_used")),
        "has_output": bool(state.get("selected_output")),
    }
    input_images = list(state.get("input_images") or [])
    selected_output = state.get("selected_output")
    if input_images and selected_output and Path(str(input_images[0])).exists() and Path(str(selected_output)).exists():
        # 最终质检补一层确定性前后对比，专门捕捉“自动美化过头”这类模型 critic 可能漏掉的问题。
        original_metrics = compute_image_metrics(str(input_images[0]))
        edited_metrics = compute_image_metrics(str(selected_output))
        warnings: list[str] = []
        brightness_delta = edited_metrics["brightness_mean"] - original_metrics["brightness_mean"]
        shadow_drop = original_metrics["shadow_ratio"] - edited_metrics["shadow_ratio"]
        highlight_delta = edited_metrics["highlight_ratio"] - original_metrics["highlight_ratio"]
        saturation_delta = edited_metrics["saturation_mean"] - original_metrics["saturation_mean"]
        if brightness_delta > 18.0 and shadow_drop > 0.08:
            warnings.append("输出黑位被明显抬高，可能出现暗部发灰或原图氛围丢失。")
        if brightness_delta > 16.0 and saturation_delta < -0.03:
            warnings.append("输出亮度上升但饱和度下降，存在奶白/灰雾风险。")
        if highlight_delta > 0.08:
            warnings.append("输出高光面积明显扩大，可能存在高光扩散或白场过推。")
        if warnings:
            report["warnings"] = warnings
            report["should_request_review"] = True
    return report


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
    """Produce the final evaluation report after search rounds finish."""

    base_report = _build_base_report(state)
    deterministic_warnings = list(base_report.get("warnings") or [])
    deterministic_review_required = bool(base_report.get("should_request_review"))
    critic, critic_error = _run_critic(state)
    fallback_trace = list(state.get("fallback_trace") or [])
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))
        merged_warnings = list(base_report.get("warnings") or [])
        for warning in deterministic_warnings:
            if warning not in merged_warnings:
                merged_warnings.append(warning)
        base_report["warnings"] = merged_warnings
        base_report["should_request_review"] = bool(base_report.get("should_request_review")) or deterministic_review_required
    elif critic_error:
        fallback_trace = append_fallback_trace(
            fallback_trace,
            round_id=None,
            focus=None,
            source="critic_model",
            location="eval_report",
            strategy="execution_only_evaluation",
            message="结果评估模型不可用，改用执行事实评估。",
            error=critic_error,
        )

    report = EvaluationReport.model_validate(base_report)
    memory_candidates: list[dict[str, Any]] = []
    edit_plan = _safe_edit_plan(state)
    if edit_plan is not None and edit_plan.should_write_memory:
        for item in edit_plan.memory_candidates:
            if item not in memory_candidates:
                memory_candidates.append(item)

    return {
        "eval_report": report,
        "final_review": report,
        "approval_required": bool(state.get("approval_required")) or bool(report.should_request_review),
        "approval_payload": coerce_approval_payload(state.get("approval_payload")),
        "fallback_trace": fallback_trace,
        "memory_write_candidates": memory_candidates,
    }
