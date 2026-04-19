"""Edit request routes."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_asset_store, get_graph_app, get_job_store
from app.api.runtime import (
    append_job_event,
    build_error_detail,
    collect_terminal_status,
    compute_stage_timings,
    format_sse,
    iter_graph_events,
    make_event,
    read_final_state,
)
from app.api.routes_assets import _build_asset_response
from app.api.routes_jobs import (
    _build_execution_trace_payload,
    _dump_fallback_trace,
    _dump_segmentation_trace,
    _build_job_summary,
    _build_round_execution_trace_payloads,
    _build_round_output_assets,
)
from app.api.schemas import EditRequest, EditResponse
from app.services.asset_store import AssetStore
from app.services.edit_runner import finalize_edit_run, prepare_edit_run
from app.services.job_store import JobStore

router = APIRouter(tags=["edit"])


@router.post("/edit", response_model=EditResponse)
async def edit(
    payload: EditRequest,
    request: Request,
    graph=Depends(get_graph_app),
    job_store: JobStore = Depends(get_job_store),
    asset_store: AssetStore = Depends(get_asset_store),
) -> EditResponse:
    """Run the current graph synchronously and persist a frontend-friendly job record."""

    try:
        run = prepare_edit_run(payload, asset_store=asset_store, job_store=job_store)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        graph.invoke(run.graph_input, config=run.config)
        final_state = read_final_state(graph, run.config)
    except Exception as exc:
        failed = job_store.set_status(
            run.job.job_id,
            "failed",
            error=str(exc),
            error_detail=build_error_detail(exc, stage=job_store.require(run.job.job_id).current_stage),
        )
        raise HTTPException(status_code=500, detail=failed.error) from exc

    status = "review_required" if final_state.get("approval_required") else "completed"
    finalized = finalize_edit_run(
        job_store=job_store,
        asset_store=asset_store,
        job_id=run.job.job_id,
        final_state=final_state,
        status=status,
        current_stage="completed" if status == "completed" else "human_review",
        current_message="任务完成" if status == "completed" else "等待人工确认",
    )
    completed = finalized.job

    output_records = list(finalized.output_records_by_path.values())
    selected_output_response = None
    selected_output = final_state.get("selected_output")
    if selected_output:
        selected_record = finalized.output_records_by_path.get(selected_output)
        if selected_record is not None:
            selected_output_response = _build_asset_response(request, selected_record)

    return EditResponse(
        job=_build_job_summary(completed),
        selected_output=selected_output_response,
        candidate_outputs=[
            _build_asset_response(request, record)
            for record in output_records
        ],
        edit_plan=completed.edit_plan.model_dump(mode="json") if completed.edit_plan is not None else None,
        eval_report=completed.eval_report.model_dump(mode="json") if completed.eval_report is not None else None,
        execution_trace=_build_execution_trace_payload(request, completed.execution_trace, asset_store),
        segmentation_trace=_dump_segmentation_trace(completed.segmentation_trace),
        fallback_trace=_dump_fallback_trace(completed.fallback_trace),
        round_outputs=_build_round_output_assets(request, completed, asset_store),
        round_plans=completed.round_plans,
        round_eval_reports=completed.round_eval_reports,
        round_execution_traces=_build_round_execution_trace_payloads(request, completed.round_execution_traces, asset_store),
        round_segmentation_traces=completed.round_segmentation_traces,
        events=completed.events,
        stage_timings=compute_stage_timings(completed.events),
    )


@router.post("/edit/stream")
async def edit_stream(
    payload: EditRequest,
    request: Request,
    graph=Depends(get_graph_app),
    job_store: JobStore = Depends(get_job_store),
    asset_store: AssetStore = Depends(get_asset_store),
) -> StreamingResponse:
    """Run the graph and stream frontend-friendly progress events."""

    try:
        run = prepare_edit_run(payload, asset_store=asset_store, job_store=job_store)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    def event_stream() -> Iterator[str]:
        created_event = make_event(
            "job_created",
            job_id=run.job.job_id,
            stage="queued",
            message="任务已创建，准备开始处理",
        )
        created_event = append_job_event(job_store, run.job.job_id, created_event)
        yield format_sse("job_created", created_event)

        try:
            for event in iter_graph_events(
                graph=graph,
                graph_input=run.graph_input,
                config=run.config,
                job_store=job_store,
                job_id=run.job.job_id,
            ):
                yield format_sse(event["event"], event)
        except Exception as exc:
            current = job_store.require(run.job.job_id)
            error_detail = build_error_detail(
                exc,
                stage=current.current_stage,
                node=current.current_stage,
                extra={
                    "last_event": current.events[-1] if current.events else None,
                },
            )
            failed = job_store.set_status(
                run.job.job_id,
                "failed",
                error=str(exc),
                error_detail=error_detail,
                current_stage="failed",
                current_message="任务执行失败",
            )
            error_event = make_event(
                "job_failed",
                job_id=run.job.job_id,
                stage="failed",
                message="任务执行失败",
                error=failed.error,
                error_detail=error_detail,
            )
            error_event = append_job_event(job_store, run.job.job_id, error_event)
            yield format_sse("job_failed", error_event)
            return

        final_state = read_final_state(graph, run.config)
        status = collect_terminal_status(final_state)
        finalized = finalize_edit_run(
            job_store=job_store,
            asset_store=asset_store,
            job_id=run.job.job_id,
            final_state=final_state,
            status=status,
            current_stage="completed" if status == "completed" else "human_review" if status == "review_required" else "failed",
            current_message="任务完成" if status == "completed" else "等待人工确认" if status == "review_required" else "任务失败",
        )
        completed = finalized.job

        if status == "review_required":
            event = make_event(
                "job_interrupted",
                job_id=completed.job_id,
                stage="human_review",
                message="任务已暂停，等待人工确认",
                approval_payload=final_state.get("approval_payload"),
            )
            event = append_job_event(job_store, completed.job_id, event)
            yield format_sse("job_interrupted", event)
            return

        event = make_event(
            "job_completed",
            job_id=completed.job_id,
            stage="completed",
            message="任务处理完成",
            selected_output_asset_id=completed.output_asset_ids[-1] if completed.output_asset_ids else None,
        )
        event = append_job_event(job_store, completed.job_id, event)
        yield format_sse("job_completed", event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
