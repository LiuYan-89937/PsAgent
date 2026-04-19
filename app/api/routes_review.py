"""Human review resume routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command

from app.api.deps import get_graph_app, get_job_store
from app.api.runtime import append_job_event, build_graph_config, collect_terminal_status, make_event, read_final_state
from app.api.runtime import build_error_detail
from app.api.schemas import ResumeReviewRequest, ResumeReviewResponse
from app.services.job_store import JobStore

router = APIRouter(tags=["review"])


@router.post("/resume-review", response_model=ResumeReviewResponse)
async def resume_review(
    payload: ResumeReviewRequest,
    graph=Depends(get_graph_app),
    job_store: JobStore = Depends(get_job_store),
) -> ResumeReviewResponse:
    """Acknowledge a future interrupt/resume review flow.

    当前直接基于 thread_id 恢复图执行。
    """

    record = job_store.get(payload.job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not record.approval_required:
        raise HTTPException(status_code=400, detail="Job is not waiting for review")

    config = build_graph_config(record.thread_id)
    try:
        for _ in graph.stream(
            Command(resume={"approved": payload.approved, "note": payload.note}),
            config=config,
            stream_mode=["tasks", "updates", "custom"],
            version="v2",
        ):
            pass
    except Exception as exc:
        job_store.set_status(
            payload.job_id,
            "failed",
            error=str(exc),
            error_detail=build_error_detail(exc, stage="human_review", node="human_review"),
            current_stage="failed",
            current_message="审核恢复失败",
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final_state = read_final_state(graph, config)
    status = "completed" if payload.approved else "failed"
    completed = job_store.set_execution_result(
        payload.job_id,
        status=collect_terminal_status(final_state) if payload.approved else status,
        edit_plan=final_state.get("edit_plan"),
        eval_report=final_state.get("eval_report"),
        execution_trace=final_state.get("execution_trace") or [],
        segmentation_trace=final_state.get("segmentation_trace") or [],
        fallback_trace=final_state.get("fallback_trace") or [],
        round_plans=final_state.get("round_plans") or {},
        round_eval_reports=final_state.get("round_eval_reports") or {},
        round_execution_traces=final_state.get("round_execution_traces") or {},
        round_segmentation_traces=final_state.get("round_segmentation_traces") or {},
        approval_required=bool(final_state.get("approval_required")),
        request_text=final_state.get("request_text"),
        current_stage="completed" if payload.approved else "failed",
        current_message="审核通过，任务完成" if payload.approved else "审核拒绝，任务结束",
    )
    append_job_event(
        job_store,
        payload.job_id,
        make_event(
            "review_resumed",
            stage="human_review",
            message="审核恢复完成",
            approved=payload.approved,
            note=payload.note,
        ),
    )
    return ResumeReviewResponse(
        job_id=payload.job_id,
        accepted=payload.approved,
        implemented=True,
        status=completed.status,
        message="Graph review resume completed.",
    )
