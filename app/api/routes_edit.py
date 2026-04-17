"""Edit request routes."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_asset_store, get_graph_app, get_job_store
from app.api.runtime import (
    append_job_event,
    build_graph_config,
    build_error_detail,
    collect_terminal_status,
    compute_stage_timings,
    format_sse,
    iter_graph_events,
    make_event,
    read_final_state,
)
from app.api.routes_assets import _build_asset_response
from app.api.routes_jobs import _build_job_summary, _build_round_output_assets
from app.api.schemas import EditRequest, EditResponse
from app.services.asset_store import AssetStore
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

    input_asset_ids = list(payload.input_asset_ids)
    input_image_paths = list(payload.input_image_paths)
    for asset_id in input_asset_ids:
        input_image_paths.append(asset_store.require(asset_id).local_path)

    if not input_image_paths:
        raise HTTPException(status_code=400, detail="No input images provided")

    thread_id = payload.thread_id or f"thread-{uuid4().hex}"
    job = job_store.create_job(
        user_id=payload.user_id,
        thread_id=thread_id,
        request_text=payload.instruction,
        input_asset_ids=input_asset_ids,
    )
    job_store.update(job.job_id, status="running")

    try:
        result = graph.invoke(
            {
                "user_id": payload.user_id,
                "thread_id": thread_id,
                "input_images": input_image_paths,
                "request_text": payload.instruction or "",
                "mode": "auto" if payload.auto_mode else "explicit",
            },
        )
    except Exception as exc:
        failed = job_store.update(
            job.job_id,
            status="failed",
            error=str(exc),
            error_detail=build_error_detail(exc, stage=job_store.require(job.job_id).current_stage),
        )
        raise HTTPException(status_code=500, detail=failed.error) from exc

    output_records_by_path: dict[str, object] = {}
    output_asset_ids: list[str] = []
    for image_path in result.get("candidate_outputs") or []:
        if image_path in output_records_by_path:
            continue
        record = asset_store.save_generated(image_path)
        output_records_by_path[image_path] = record
        output_asset_ids.append(record.asset_id)

    selected_output = result.get("selected_output")
    if selected_output and selected_output not in output_records_by_path:
        record = asset_store.save_generated(selected_output)
        output_records_by_path[selected_output] = record
        output_asset_ids.append(record.asset_id)

    round_output_asset_ids: dict[str, str] = {}
    for round_key, image_path in dict(result.get("round_outputs") or {}).items():
        if not image_path:
            continue
        if image_path not in output_records_by_path:
            record = asset_store.save_generated(image_path)
            output_records_by_path[image_path] = record
            output_asset_ids.append(record.asset_id)
        round_output_asset_ids[round_key] = output_records_by_path[image_path].asset_id

    status = "review_required" if result.get("approval_required") else "completed"
    completed = job_store.update(
        job.job_id,
        status=status,
        output_asset_ids=output_asset_ids,
        round_output_asset_ids=round_output_asset_ids,
        edit_plan=result.get("edit_plan"),
        eval_report=result.get("eval_report"),
        execution_trace=result.get("execution_trace") or [],
        segmentation_trace=result.get("segmentation_trace") or [],
        round_plans=result.get("round_plans") or {},
        round_eval_reports=result.get("round_eval_reports") or {},
        round_execution_traces=result.get("round_execution_traces") or {},
        round_segmentation_traces=result.get("round_segmentation_traces") or {},
        approval_required=bool(result.get("approval_required")),
    )

    output_records = list(output_records_by_path.values())
    selected_output_response = None
    if selected_output:
        selected_record = output_records_by_path.get(selected_output)
        if selected_record is not None:
            selected_output_response = _build_asset_response(request, selected_record)

    return EditResponse(
        job=_build_job_summary(completed),
        selected_output=selected_output_response,
        candidate_outputs=[
            _build_asset_response(request, record)
            for record in output_records
        ],
        edit_plan=completed.edit_plan,
        eval_report=completed.eval_report,
        execution_trace=completed.execution_trace,
        segmentation_trace=completed.segmentation_trace,
        round_outputs=_build_round_output_assets(request, completed, asset_store),
        round_plans=completed.round_plans,
        round_eval_reports=completed.round_eval_reports,
        round_execution_traces=completed.round_execution_traces,
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

    input_asset_ids = list(payload.input_asset_ids)
    input_image_paths = list(payload.input_image_paths)
    for asset_id in input_asset_ids:
        input_image_paths.append(asset_store.require(asset_id).local_path)

    if not input_image_paths:
        raise HTTPException(status_code=400, detail="No input images provided")

    thread_id = payload.thread_id or f"thread-{uuid4().hex}"
    job = job_store.create_job(
        user_id=payload.user_id,
        thread_id=thread_id,
        request_text=payload.instruction,
        input_asset_ids=input_asset_ids,
    )
    job_store.update(job.job_id, status="running")

    def event_stream() -> Iterator[str]:
        created_event = make_event(
            "job_created",
            job_id=job.job_id,
            stage="queued",
            message="任务已创建，准备开始处理",
        )
        created_event = append_job_event(job_store, job.job_id, created_event)
        yield format_sse("job_created", created_event)

        config = build_graph_config(thread_id)
        graph_input = {
            "user_id": payload.user_id,
            "thread_id": thread_id,
            "input_images": input_image_paths,
            "request_text": payload.instruction or "",
            "mode": "auto" if payload.auto_mode else "explicit",
        }

        try:
            for event in iter_graph_events(
                graph=graph,
                graph_input=graph_input,
                config=config,
                job_store=job_store,
                job_id=job.job_id,
            ):
                yield format_sse(event["event"], event)
        except Exception as exc:
            current = job_store.require(job.job_id)
            error_detail = build_error_detail(
                exc,
                stage=current.current_stage,
                node=current.current_stage,
                extra={
                    "last_event": current.events[-1] if current.events else None,
                },
            )
            failed = job_store.update(
                job.job_id,
                status="failed",
                error=str(exc),
                error_detail=error_detail,
                current_stage="failed",
                current_message="任务执行失败",
            )
            error_event = make_event(
                "job_failed",
                job_id=job.job_id,
                stage="failed",
                message="任务执行失败",
                error=failed.error,
                error_detail=error_detail,
            )
            error_event = append_job_event(job_store, job.job_id, error_event)
            yield format_sse("job_failed", error_event)
            return

        final_state = read_final_state(graph, config)
        output_asset_ids: list[str] = []
        for image_path in final_state.get("candidate_outputs") or []:
            record = asset_store.save_generated(image_path)
            output_asset_ids.append(record.asset_id)

        selected_output = final_state.get("selected_output")
        if selected_output:
            selected_record = asset_store.save_generated(selected_output)
            if selected_record.asset_id not in output_asset_ids:
                output_asset_ids.append(selected_record.asset_id)

        status = collect_terminal_status(final_state)
        output_records_by_path: dict[str, object] = {}
        completed = job_store.update(
            job.job_id,
            status=status,
            output_asset_ids=output_asset_ids,
            edit_plan=final_state.get("edit_plan"),
            eval_report=final_state.get("eval_report"),
            execution_trace=final_state.get("execution_trace") or [],
            segmentation_trace=final_state.get("segmentation_trace") or [],
            round_plans=final_state.get("round_plans") or {},
            round_eval_reports=final_state.get("round_eval_reports") or {},
            round_execution_traces=final_state.get("round_execution_traces") or {},
            round_segmentation_traces=final_state.get("round_segmentation_traces") or {},
            approval_required=bool(final_state.get("approval_required")),
            current_stage="completed" if status == "completed" else "human_review" if status == "review_required" else "failed",
            current_message="任务完成" if status == "completed" else "等待人工确认" if status == "review_required" else "任务失败",
        )

        round_output_asset_ids: dict[str, str] = {}
        for asset_id in completed.output_asset_ids:
            try:
                record = asset_store.require(asset_id)
                output_records_by_path[record.local_path] = record
            except KeyError:
                continue
        for round_key, image_path in dict(final_state.get("round_outputs") or {}).items():
            if not image_path:
                continue
            if image_path not in output_records_by_path:
                record = asset_store.save_generated(image_path)
                output_records_by_path[image_path] = record
                output_asset_ids.append(record.asset_id)
            round_output_asset_ids[round_key] = output_records_by_path[image_path].asset_id
        if round_output_asset_ids:
            completed = job_store.update(
                job.job_id,
                output_asset_ids=output_asset_ids,
                round_output_asset_ids=round_output_asset_ids,
            )

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
