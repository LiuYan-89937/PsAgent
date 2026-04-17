"""Evaluate edit outputs against quality and preference constraints."""

from __future__ import annotations

from typing import Any

from app.graph.state import CriticResult, EditState, EvaluationReport
from app.services.critic_model import critic_model_available, evaluate_edit_result_with_qwen


STYLE_KEYWORDS = ("质感", "氛围", "色调", "夏日", "夏天", "通透", "空气感", "胶片", "明媚")
REPAIR_KEYWORDS = ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色", "压暗")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether a text contains any keyword."""

    return any(keyword in text for keyword in keywords)


def _should_continue_round_2_by_rules(state: EditState, report: EvaluationReport) -> bool:
    """Decide whether round 2 should run when the critic is absent or too optimistic."""

    if not report.has_output:
        return False

    request_text = str(state.get("request_text") or "")
    request_intent = state.get("request_intent") or {}
    constraints = {str(item) for item in request_intent.get("constraints", [])}
    requested_packages = list(request_intent.get("requested_packages", []))
    issues = list((state.get("image_analysis") or {}).get("issues", []))

    if constraints & {"needs_layered_refinement", "repair_backlighting", "build_summer_mood", "match_reference_style"}:
        return True
    if _contains_any(request_text, STYLE_KEYWORDS) and _contains_any(request_text, REPAIR_KEYWORDS):
        return True
    if len(requested_packages) >= 3:
        return True
    if len(issues) >= 2 and report.num_operations >= 2:
        return True
    return False


def _run_critic(state: EditState) -> CriticResult | None:
    """Run the critic model when enough inputs are present."""

    input_images = state.get("input_images") or []
    selected_output = state.get("selected_output")
    if not (critic_model_available() and input_images and selected_output):
        return None

    model_report = evaluate_edit_result_with_qwen(
        original_image_path=input_images[0],
        edited_image_path=selected_output,
        request_text=str(state.get("request_text") or ""),
        edit_plan=state.get("edit_plan") or {},
        image_analysis=state.get("image_analysis") or {},
        execution_trace=state.get("execution_trace") or [],
    )
    return CriticResult.model_validate(model_report)


def _build_base_report(state: EditState) -> dict[str, Any]:
    """Build execution-fact based evaluation fields."""

    execution_trace = state.get("execution_trace") or []
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
    critic = _run_critic(state)
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))

    report = EvaluationReport.model_validate(base_report)
    round_key = _round_key(state)
    round_eval_reports = dict(state.get("round_eval_reports") or {})
    round_eval_reports[round_key] = report.model_dump(mode="json")

    continue_to_round_2 = (
        bool(critic.should_continue_editing) if critic is not None else False
    ) or _should_continue_round_2_by_rules(state, report)
    approval_required = bool(state.get("approval_required")) or bool(report.should_request_review)
    return {
        "round_eval_reports": round_eval_reports,
        "eval_report": report.model_dump(mode="json"),
        "continue_to_round_2": continue_to_round_2,
        "approval_required": approval_required,
        "approval_payload": state.get("approval_payload"),
    }


def finalize_round_1_result(state: EditState) -> dict[str, Any]:
    """Promote round 1 outputs to final results when no second round is required."""

    round_1_report = dict((state.get("round_eval_reports") or {}).get("round_1", {}))
    validated = EvaluationReport.model_validate(round_1_report or _build_base_report(state))
    return {
        "eval_report": validated.model_dump(mode="json"),
        "approval_required": bool(state.get("approval_required")) or bool(validated.should_request_review),
        "approval_payload": state.get("approval_payload"),
        "continue_to_round_2": False,
    }


def evaluate_result(state: EditState) -> dict[str, Any]:
    """Produce the final evaluation report after the last round."""

    base_report = _build_base_report(state)
    critic = _run_critic(state)
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))

    report = EvaluationReport.model_validate(base_report)
    round_key = _round_key(state)
    round_eval_reports = dict(state.get("round_eval_reports") or {})
    round_eval_reports[round_key] = report.model_dump(mode="json")

    return {
        "eval_report": report.model_dump(mode="json"),
        "round_eval_reports": round_eval_reports,
        "approval_required": bool(state.get("approval_required")) or bool(report.should_request_review),
        "approval_payload": state.get("approval_payload"),
        "continue_to_round_2": False,
    }
