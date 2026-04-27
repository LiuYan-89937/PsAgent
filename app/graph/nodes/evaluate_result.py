"""Final review node with one final critic call."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4
from pydantic import ValidationError

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import (
    ApprovalPayload,
    CriticResult,
    EditState,
    EvaluationReport,
    FocusKey,
    ObjectiveCard,
    ObjectiveGap,
    ReviewDecision,
    coerce_approval_payload,
    coerce_edit_plan,
    coerce_execution_trace,
    coerce_image_analysis,
    coerce_objective_card,
    coerce_search_rounds,
)
from app.services.image_metrics import compute_image_metrics
from app.services.critic_model import critic_model_available, evaluate_edit_result
from app.services.search_agent.config import resolve_search_round_limits


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_structured_decision(
    report: dict[str, Any],
    *,
    decision: ReviewDecision,
    reason: str,
    next_focus: FocusKey | None = None,
    correction_objective: str = "",
) -> None:
    report["decision"] = decision
    report["decision_reason"] = reason
    report["next_focus"] = next_focus
    report["correction_objective"] = correction_objective


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
        "decision": "accept",
        "next_focus": None,
        "correction_objective": "",
        "decision_reason": "execution facts did not require routing intervention",
    }
    input_images = list(state.get("input_images") or [])
    selected_output = state.get("selected_output")
    if input_images and selected_output and Path(str(input_images[0])).exists() and Path(str(selected_output)).exists():
        original_path = Path(str(input_images[0]))
        edited_path = Path(str(selected_output))
        original_metrics = compute_image_metrics(str(original_path))
        edited_metrics = compute_image_metrics(str(edited_path))
        warnings: list[str] = []
        brightness_delta = edited_metrics["brightness_mean"] - original_metrics["brightness_mean"]
        shadow_drop = original_metrics["shadow_ratio"] - edited_metrics["shadow_ratio"]
        highlight_delta = edited_metrics["highlight_ratio"] - original_metrics["highlight_ratio"]
        saturation_delta = edited_metrics["saturation_mean"] - original_metrics["saturation_mean"]
        local_contrast_delta = abs(float(edited_metrics.get("local_contrast_mean", 0.0)) - float(original_metrics.get("local_contrast_mean", 0.0)))
        same_file = original_path.resolve() == edited_path.resolve()
        same_bytes = False if same_file else original_path.stat().st_size == edited_path.stat().st_size and _file_digest(original_path) == _file_digest(edited_path)
        visually_unchanged = (
            abs(brightness_delta) < 0.6
            and abs(saturation_delta) < 0.004
            and abs(highlight_delta) < 0.003
            and abs(shadow_drop) < 0.003
            and local_contrast_delta < 0.003
        )
        if same_file or same_bytes or visually_unchanged:
            warnings.append("输出与原图几乎一致，修图流程未产生有效修改。")
            _set_structured_decision(
                report,
                decision="continue_auto",
                next_focus="global_tone",
                correction_objective="输出与原图几乎一致，重新生成有效的曝光、色调和主体可读性调整。",
                reason="deterministic unchanged-output guard",
            )
        if brightness_delta > 18.0 and shadow_drop > 0.08:
            warnings.append("输出黑位被明显抬高，可能出现暗部发灰或原图氛围丢失。")
        if brightness_delta > 16.0 and saturation_delta < -0.03:
            warnings.append("输出亮度上升但饱和度下降，存在奶白/灰雾风险。")
        if highlight_delta > 0.08:
            warnings.append("输出高光面积明显扩大，可能存在高光扩散或白场过推。")
        if warnings:
            report["warnings"] = warnings
            if report.get("decision") == "accept":
                _set_structured_decision(
                    report,
                    decision="request_human_review",
                    reason="deterministic quality guard found a risk that lacks an automatic correction objective",
                )
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
        model_report = evaluate_edit_result(
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


def _auto_should_continue_before_review(
    *,
    state: EditState,
    report_payload: dict[str, Any],
    cycle_round_count: int,
    min_rounds: int,
    max_rounds: int,
) -> bool:
    """Keep auto search moving before sending uncertain quality reviews to humans."""

    if cycle_round_count >= max_rounds:
        return False
    if report_payload.get("decision") == "continue_auto":
        return True
    if report_payload.get("decision") == "request_human_review":
        return cycle_round_count < min_rounds
    return False


def _append_continuation_gap(state: EditState, report_payload: dict[str, Any]) -> ObjectiveCard | None:
    """Append a final-review corrective gap to the existing objective card."""

    objective = coerce_objective_card(state.get("objective_card"))
    if objective is None:
        return None
    text = str(report_payload.get("correction_objective") or report_payload.get("summary") or "Final review requested one more corrective round.").strip()
    focus = report_payload.get("next_focus") or "global_tone"
    gap = ObjectiveGap(
        id=f"final_review_{focus}_{uuid4().hex[:8]}",
        focus=focus,  # type: ignore[arg-type]
        description=text,
        priority=96,
        target_region="person area" if focus == "subject_separation" else "face and skin area" if focus == "subject_cleanup" else "whole_image",
        desired_delta=text,
        constraints=["final_review_continuation"],
    )
    updated_gaps = list(objective.gaps)
    updated_gaps.append(gap)
    return objective.model_copy(update={"gaps": updated_gaps})


def _cycle_round_count(state: EditState, rounds: list[Any]) -> int:
    try:
        offset = int(state.get("search_cycle_round_offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    return max(len(rounds) - max(offset, 0), 0)


def final_review(state: EditState) -> dict[str, Any]:
    """Produce the final evaluation report after search rounds finish."""

    base_report = _build_base_report(state)
    deterministic_warnings = list(base_report.get("warnings") or [])
    deterministic_decision = base_report.get("decision")
    critic, critic_error = _run_critic(state)
    fallback_trace = list(state.get("fallback_trace") or [])
    if critic is not None:
        base_report.update(critic.model_dump(mode="json"))
        merged_warnings = list(base_report.get("warnings") or [])
        for warning in deterministic_warnings:
            if warning not in merged_warnings:
                merged_warnings.append(warning)
        base_report["warnings"] = merged_warnings
        if deterministic_decision == "request_human_review" and base_report.get("decision") == "accept":
            _set_structured_decision(
                base_report,
                decision="request_human_review",
                reason="deterministic quality guard found a risk that the critic did not route",
            )
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

    rounds = coerce_search_rounds(state.get("rounds") or [])
    round_limits = resolve_search_round_limits(state.get("search_effort"))
    cycle_round_count = _cycle_round_count(state, rounds)
    continuation_candidate = _auto_should_continue_before_review(
        state=state,
        report_payload=base_report,
        cycle_round_count=cycle_round_count,
        min_rounds=round_limits.min_rounds,
        max_rounds=round_limits.max_rounds,
    )
    objective_update = _append_continuation_gap(state, base_report) if continuation_candidate else None
    continuation_requested = objective_update is not None

    max_rounds_exhausted = (
        cycle_round_count >= round_limits.max_rounds
        and base_report.get("decision") in {"continue_auto", "request_human_review"}
    )
    if max_rounds_exhausted:
        _set_structured_decision(
            base_report,
            decision="request_human_review",
            reason="auto search reached max rounds before the structured critic decision could be satisfied",
        )

    report = EvaluationReport.model_validate(base_report)
    memory_candidates: list[dict[str, Any]] = []
    edit_plan = _safe_edit_plan(state)
    if edit_plan is not None and edit_plan.should_write_memory:
        for item in edit_plan.memory_candidates:
            if item not in memory_candidates:
                memory_candidates.append(item)

    approval_payload = coerce_approval_payload(state.get("approval_payload"))
    if max_rounds_exhausted and approval_payload is None:
        approval_payload = ApprovalPayload(
            reason="final_review_unresolved_after_max_rounds",
            summary=report.summary,
            suggested_action="已达到自动搜索最大轮数，需要人工确认是否继续增强。",
            metadata={
                "search_effort": state.get("search_effort") or "standard",
                "cycle_round_count": cycle_round_count,
                "min_rounds": round_limits.min_rounds,
                "max_rounds": round_limits.max_rounds,
            },
        )

    update: dict[str, Any] = {
        "eval_report": report,
        "final_review": report,
        "needs_search_continuation": continuation_requested,
        "search_continuation_reason": report.summary if continuation_requested else None,
        "approval_required": bool(state.get("approval_required"))
        or (report.decision == "request_human_review" and not continuation_requested),
        "approval_payload": approval_payload,
        "fallback_trace": fallback_trace,
        "memory_write_candidates": memory_candidates,
    }
    if objective_update is not None:
        update["objective_card"] = objective_update.model_dump(mode="json")
    return update
