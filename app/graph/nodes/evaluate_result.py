"""Evaluate edit outputs against quality and preference constraints."""

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
    coerce_request_intent,
)
from app.services.critic_model import critic_model_available, evaluate_edit_result_with_qwen


STYLE_KEYWORDS = ("质感", "氛围", "色调", "夏日", "夏天", "通透", "空气感", "胶片", "明媚")
REPAIR_KEYWORDS = ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色", "压暗")
PORTRAIT_KEYWORDS = ("人像", "脸", "面部", "肤色", "皮肤", "眼下", "发丝", "婚纱", "少女")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether a text contains any keyword."""

    return any(keyword in text for keyword in keywords)


def _safe_request_intent(state: EditState):
    """Return a typed request intent when the payload is complete enough."""

    value = state.get("request_intent")
    if value is None:
        return None
    try:
        return coerce_request_intent(value)
    except ValidationError:
        return None


def _safe_image_analysis(state: EditState):
    """Return a typed image analysis when the payload is complete enough."""

    value = state.get("image_analysis")
    if value is None:
        return None
    try:
        return coerce_image_analysis(value)
    except ValidationError:
        return None


def _safe_edit_plan(state: EditState):
    """Return a typed edit plan when the payload is complete enough."""

    value = state.get("edit_plan")
    if value is None:
        return None
    try:
        return coerce_edit_plan(value)
    except ValidationError:
        return None


def _should_continue_round_2_by_rules(state: EditState, report: EvaluationReport) -> bool:
    """Decide whether round 2 should run when the critic is absent or too optimistic."""

    if not report.has_output:
        return False

    request_text = str(state.get("request_text") or "")
    request_intent = _safe_request_intent(state)
    image_analysis = _safe_image_analysis(state)
    mode = str((request_intent.mode if request_intent is not None else None) or state.get("mode") or "")
    constraints = {str(item) for item in (request_intent.constraints if request_intent is not None else [])}
    requested_packages = list(request_intent.requested_packages if request_intent is not None else [])
    issues = list(image_analysis.issues if image_analysis is not None else [])
    domain = str((image_analysis.domain if image_analysis is not None else None) or "general")
    fallback_trace = list(state.get("fallback_trace") or [])

    if report.fallback_count > 0 or report.failure_count > 0:
        return True
    if any(
        item.get("stage") == "plan_execute_round_1" and item.get("strategy") == "finish_current_round"
        for item in fallback_trace
    ):
        return True

    if constraints & {"needs_layered_refinement", "repair_backlighting", "build_summer_mood", "match_reference_style"}:
        return True
    if mode == "auto" and report.success_count > 0:
        return True
    if domain == "portrait" and report.success_count > 0 and (requested_packages or issues or _contains_any(request_text, PORTRAIT_KEYWORDS)):
        return True
    if _contains_any(request_text, STYLE_KEYWORDS) and _contains_any(request_text, REPAIR_KEYWORDS):
        return True
    if len(requested_packages) >= 3:
        return True
    if len(issues) >= 2 and report.num_operations >= 2:
        return True
    return False


def _run_critic(state: EditState) -> tuple[CriticResult | None, str | None]:
    """Run the critic model when enough inputs are present."""

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
        # critic 只是结果评估层，不应该因为模型响应异常就打断整条修图流程。
        return None, str(error)

    return CriticResult.model_validate(model_report), None


def _build_base_report(state: EditState) -> dict[str, Any]:
    """Build execution-fact based evaluation fields."""

    execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("execution_trace") or [])]
    success_count = sum(1 for item in execution_trace if item.get("ok"))
    failure_count = sum(1 for item in execution_trace if item.get("ok") is False)
    fallback_count = sum(1 for item in execution_trace if item.get("fallback_used"))
    selected_output = state.get("selected_output")

    return {
        "selected_output": selected_output,
        "num_operations": len(execution_trace),
        "success_count": success_count,
        "failure_count": failure_count,
        "fallback_count": fallback_count,
        "has_output": bool(selected_output),
    }


def _round_key(state: EditState) -> str:
    """Return the active round key."""

    return f"round_{int(state.get('current_round') or 1)}"


def evaluate_round_1(state: EditState) -> dict[str, Any]:
    """Evaluate the first round and decide whether a second round is needed."""

    base_report = _build_base_report(state)
    critic, critic_error = _run_critic(state)
    fallback_trace = list(state.get("fallback_trace") or [])
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))
    elif critic_error:
        fallback_trace = append_fallback_trace(
            fallback_trace,
            stage="evaluate_round_1",
            source="critic_model",
            location="eval_report",
            strategy="execution_only_evaluation",
            message="结果评估模型不可用，改用执行事实评估。",
            error=critic_error,
        )

    report = EvaluationReport.model_validate(base_report)
    round_key = _round_key(state)
    round_eval_reports = dict(state.get("round_eval_reports") or {})
    round_eval_reports[round_key] = report.model_dump(mode="json")
    approval_payload = coerce_approval_payload(state.get("approval_payload"))

    continue_to_round_2 = (
        bool(critic.should_continue_editing) if critic is not None else False
    ) or _should_continue_round_2_by_rules(state, report)
    approval_required = bool(state.get("approval_required")) or bool(report.should_request_review)
    return {
        "round_eval_reports": round_eval_reports,
        "eval_report": report.model_dump(mode="json"),
        "continue_to_round_2": continue_to_round_2,
        "approval_required": approval_required,
        "approval_payload": approval_payload.model_dump(mode="json") if approval_payload is not None else None,
        "fallback_trace": fallback_trace,
    }


def finalize_round_1_result(state: EditState) -> dict[str, Any]:
    """Promote round 1 outputs to final results when no second round is required."""

    round_1_report = dict((state.get("round_eval_reports") or {}).get("round_1", {}))
    validated = EvaluationReport.model_validate(round_1_report or _build_base_report(state))
    approval_payload = coerce_approval_payload(state.get("approval_payload"))
    return {
        "eval_report": validated.model_dump(mode="json"),
        "approval_required": bool(state.get("approval_required")) or bool(validated.should_request_review),
        "approval_payload": approval_payload.model_dump(mode="json") if approval_payload is not None else None,
        "continue_to_round_2": False,
    }


def evaluate_result(state: EditState) -> dict[str, Any]:
    """Produce the final evaluation report after the last round."""

    base_report = _build_base_report(state)
    critic, critic_error = _run_critic(state)
    fallback_trace = list(state.get("fallback_trace") or [])
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))
    elif critic_error:
        fallback_trace = append_fallback_trace(
            fallback_trace,
            stage="evaluate_result_final",
            source="critic_model",
            location="eval_report",
            strategy="execution_only_evaluation",
            message="结果评估模型不可用，改用执行事实评估。",
            error=critic_error,
        )

    report = EvaluationReport.model_validate(base_report)
    round_key = _round_key(state)
    round_eval_reports = dict(state.get("round_eval_reports") or {})
    round_eval_reports[round_key] = report.model_dump(mode="json")
    approval_payload = coerce_approval_payload(state.get("approval_payload"))

    return {
        "eval_report": report.model_dump(mode="json"),
        "round_eval_reports": round_eval_reports,
        "approval_required": bool(state.get("approval_required")) or bool(report.should_request_review),
        "approval_payload": approval_payload.model_dump(mode="json") if approval_payload is not None else None,
        "continue_to_round_2": False,
        "fallback_trace": fallback_trace,
    }
