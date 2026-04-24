"""Deterministic candidate and round review for search-first orchestration."""

from __future__ import annotations

from app.graph.state import (
    CandidatePreviewExecution,
    CandidateProgram,
    CandidateReview,
    RoundReview,
)


def review_candidate(*, program: CandidateProgram, execution: CandidatePreviewExecution) -> CandidateReview:
    """Score one preview candidate using execution facts."""

    success_count = sum(1 for item in execution.execution_trace if item.ok)
    failure_count = sum(1 for item in execution.execution_trace if item.ok is False)
    fallback_count = len(execution.fallback_trace) + sum(1 for item in execution.execution_trace if item.fallback_used)
    issues: list[str] = []
    warnings: list[str] = []
    if failure_count:
        issues.append("候选预览存在失败步骤")
    if fallback_count:
        warnings.append("候选预览触发 fallback")
    if not program.steps:
        warnings.append("0-step candidate")

    score = (
        2.0
        + success_count * 0.5
        - failure_count * 2.0
        - fallback_count * 0.7
        - (0.25 if not program.steps else 0.0)
    )
    action = "keep"
    if failure_count:
        action = "recover_same_round"
    elif fallback_count:
        action = "recover_same_round"
    elif not program.steps:
        action = "stop_round"

    return CandidateReview(
        overall_ok=not issues,
        preserve_ok=not issues,
        style_ok=True,
        artifact_ok=not issues,
        issues=issues,
        warnings=warnings,
        summary=f"{program.label} 预览完成，得分 {score:.2f}。",
        recommended_action=action,  # type: ignore[arg-type]
        score=round(score, 4),
    )


def review_round(*, selected_review: CandidateReview, full_execution: CandidatePreviewExecution) -> RoundReview:
    """Review the committed round and decide whether recovery is needed."""

    fallback_count = len(full_execution.fallback_trace) + sum(1 for item in full_execution.execution_trace if item.fallback_used)
    failure_count = sum(1 for item in full_execution.execution_trace if item.ok is False)
    issues = list(selected_review.issues)
    warnings = list(selected_review.warnings)
    if failure_count and "正式执行存在失败步骤" not in issues:
        issues.append("正式执行存在失败步骤")
    if fallback_count and "正式执行触发 fallback" not in warnings:
        warnings.append("正式执行触发 fallback")

    action = selected_review.recommended_action
    if failure_count or fallback_count:
        action = "recover_same_round"
    elif action == "stop_round":
        action = "stop_round"
    else:
        action = "keep"

    score = selected_review.score - failure_count * 2.0 - fallback_count * 0.7
    return RoundReview(
        overall_ok=not issues,
        issues=issues,
        warnings=warnings,
        summary="本轮正式提交完成。" if action == "keep" else "本轮需要恢复搜索。" if action == "recover_same_round" else "本轮建议停止。",
        recommended_action=action,
        score=round(score, 4),
    )
