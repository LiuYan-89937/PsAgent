"""Human review resume routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command

from app.api.deps import get_asset_store, get_graph_app, get_job_store
from app.api.runtime import append_job_event, build_graph_config, collect_terminal_status, iter_graph_events, make_event, read_final_state
from app.api.runtime import build_error_detail
from app.api.schemas import ResumeReviewRequest, ResumeReviewResponse
from app.services.asset_store import AssetStore
from app.services.edit_runner import finalize_edit_run
from app.services.job_store import JobStore

router = APIRouter(tags=["review"])


@router.post("/resume-review", response_model=ResumeReviewResponse)
async def resume_review(
    payload: ResumeReviewRequest,
    graph=Depends(get_graph_app),
    asset_store: AssetStore = Depends(get_asset_store),
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
        for _ in iter_graph_events(
            graph=graph,
            graph_input=Command(
                resume={
                    "approved": payload.approved,
                    "note": payload.note,
                    "search_effort": payload.search_effort,
                }
            ),
            config=config,
            job_store=job_store,
            job_id=payload.job_id,
        ):
            pass
    except Exception as exc:
        job_store.set_status(
            payload.job_id,
            "failed",
            error=str(exc),
            error_detail=build_error_detail(exc, node="human_review"),
            current_round="failed",
            current_focus=None,
            current_message="审核恢复失败",
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final_state = read_final_state(graph, config)
    status = collect_terminal_status(final_state) if payload.approved else "failed"
    current_round = "completed" if status == "completed" else "human_review" if status == "review_required" else "failed"
    current_message = "任务完成" if status == "completed" else "等待人工确认" if status == "review_required" else "审核拒绝，任务结束"
    finalized = finalize_edit_run(
        job_store=job_store,
        asset_store=asset_store,
        job_id=payload.job_id,
        final_state=final_state,
        status=status,
        current_round=current_round,
        current_focus=None,
        current_message=current_message,
    )
    completed = finalized.job
    append_job_event(
        job_store,
        payload.job_id,
        make_event(
            "review_resumed",
            node="human_review",
            message="审核恢复完成",
            approved=payload.approved,
            note=payload.note,
        ),
    )
    if status == "review_required":
        append_job_event(
            job_store,
            payload.job_id,
            make_event(
                "job_interrupted",
                job_id=payload.job_id,
                round="human_review",
                message="任务已暂停，等待人工确认",
                approval_payload=final_state.get("approval_payload"),
            ),
        )
    elif status == "completed":
        append_job_event(
            job_store,
            payload.job_id,
            make_event(
                "job_completed",
                job_id=payload.job_id,
                round="completed",
                message="任务处理完成",
                selected_output_asset_id=completed.output_asset_ids[-1] if completed.output_asset_ids else None,
            ),
        )
    else:
        append_job_event(
            job_store,
            payload.job_id,
            make_event("job_failed", job_id=payload.job_id, round="failed", message=current_message),
        )
    return ResumeReviewResponse(
        job_id=payload.job_id,
        accepted=payload.approved,
        implemented=True,
        status=completed.status,
        message="Graph review resume completed.",
    )
