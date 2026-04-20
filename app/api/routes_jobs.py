"""Job query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_asset_store, get_job_store
from app.api.routes_assets import _build_asset_response
from app.api.runtime import compute_stage_timings
from app.api.schemas import JobDetailResponse, JobSummaryResponse
from app.graph.state import (
    ExecutionTraceItem,
    FallbackTraceItem,
    FeedbackItem,
    JobEvent,
    PhaseArtifact,
    SegmentationTraceItem,
)
from app.services.asset_store import AssetStore
from app.services.job_store import JobRecord, JobStore

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _build_job_summary(record: JobRecord) -> JobSummaryResponse:
    """Convert a stored job to summary response."""

    return JobSummaryResponse(
        job_id=record.job_id,
        status=record.status,
        user_id=record.user_id,
        thread_id=record.thread_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        approval_required=record.approval_required,
        request_text=record.request_text,
        current_stage=record.current_stage,
        current_message=record.current_message,
        error=record.error,
        error_detail=record.error_detail,
    )


def _build_execution_trace_payload(
    request: Request,
    trace: list[ExecutionTraceItem | dict[str, object]],
    asset_store: AssetStore,
) -> list[dict[str, object]]:
    """Expand execution trace items with frontend-safe output asset payloads."""

    payload: list[dict[str, object]] = []
    for item in trace:
        trace_item = item.model_dump(mode="json") if isinstance(item, ExecutionTraceItem) else dict(item)
        output_asset_id = trace_item.get("output_asset_id")
        output_asset = None
        if isinstance(output_asset_id, str):
            try:
                output_asset = _build_asset_response(request, asset_store.require(output_asset_id))
            except KeyError:
                output_asset = None
        trace_item["output_asset"] = output_asset
        payload.append(trace_item)
    return payload


def _dump_segmentation_trace(trace: list[SegmentationTraceItem | dict[str, object]]) -> list[dict[str, object]]:
    """Dump typed segmentation trace items into JSON-safe dict payloads."""

    return [item.model_dump(mode="json") if isinstance(item, SegmentationTraceItem) else dict(item) for item in trace]


def _build_segmentation_trace_payload(
    request: Request,
    trace: list[SegmentationTraceItem | dict[str, object]],
    asset_store: AssetStore,
) -> list[dict[str, object]]:
    """Expand segmentation trace items with frontend-safe mask/preview asset payloads."""

    payload: list[dict[str, object]] = []
    for item in trace:
        trace_item = item.model_dump(mode="json") if isinstance(item, SegmentationTraceItem) else dict(item)
        mask_asset = None
        mask_asset_id = trace_item.get("mask_asset_id")
        if isinstance(mask_asset_id, str):
            try:
                mask_asset = _build_asset_response(request, asset_store.require(mask_asset_id))
            except KeyError:
                mask_asset = None

        preview_asset = None
        preview_asset_id = trace_item.get("preview_asset_id")
        if isinstance(preview_asset_id, str):
            try:
                preview_asset = _build_asset_response(request, asset_store.require(preview_asset_id))
            except KeyError:
                preview_asset = None

        trace_item["mask_asset"] = mask_asset
        trace_item["preview_asset"] = preview_asset
        payload.append(trace_item)
    return payload


def _dump_fallback_trace(trace: list[FallbackTraceItem | dict[str, object]]) -> list[dict[str, object]]:
    """Dump typed fallback trace items into JSON-safe dict payloads."""

    return [item.model_dump(mode="json") if isinstance(item, FallbackTraceItem) else dict(item) for item in trace]


def _dump_job_events(events: list[JobEvent]) -> list[dict[str, object]]:
    """Dump persisted job events into JSON-safe dict payloads."""

    return [item.model_dump(mode="json") for item in events]


def _dump_feedback_items(items: list[FeedbackItem]) -> list[dict[str, object]]:
    """Dump feedback items into JSON-safe dict payloads."""

    return [item.model_dump(mode="json") for item in items]


def _build_phase_payloads(
    request: Request,
    phases: dict[str, PhaseArtifact],
    asset_store: AssetStore,
) -> dict[str, dict[str, object]]:
    """Expand grouped phase artifacts into frontend payloads."""

    payload: dict[str, dict[str, object]] = {}
    for phase_key, phase in phases.items():
        phase_payload = phase.model_dump(mode="json")
        output_payload = phase_payload.get("output")
        output_asset = None
        asset_id = output_payload.get("asset_id") if isinstance(output_payload, dict) else None
        if isinstance(asset_id, str):
            try:
                output_asset = _build_asset_response(request, asset_store.require(asset_id))
            except KeyError:
                output_asset = None
        payload[phase_key] = {
            "plan": phase_payload.get("plan"),
            "execution_trace": _build_execution_trace_payload(request, phase.execution_trace, asset_store),
            "segmentation_trace": _build_segmentation_trace_payload(request, phase.segmentation_trace, asset_store),
            "eval_report": phase_payload.get("eval_report"),
            "output": output_asset,
            "summary": phase_payload.get("summary"),
            "skipped": bool(phase_payload.get("skipped")),
            "skip_reason": phase_payload.get("skip_reason"),
            "trigger_reasons": phase_payload.get("trigger_reasons", []),
            "stopped_early": bool(phase_payload.get("stopped_early")),
        }
    return payload


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(
    request: Request,
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    asset_store: AssetStore = Depends(get_asset_store),
) -> JobDetailResponse:
    """Return a full job payload for frontend polling."""

    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    input_assets = [
        _build_asset_response(request, asset_store.require(asset_id))
        for asset_id in record.input_asset_ids
    ]
    output_assets = [
        _build_asset_response(request, asset_store.require(asset_id))
        for asset_id in record.output_asset_ids
    ]
    selected_output = output_assets[-1] if output_assets else None

    return JobDetailResponse(
        job=_build_job_summary(record),
        input_assets=input_assets,
        selected_output=selected_output,
        candidate_outputs=output_assets,
        edit_plan=record.edit_plan.model_dump(mode="json") if record.edit_plan is not None else None,
        eval_report=record.eval_report.model_dump(mode="json") if record.eval_report is not None else None,
        execution_trace=_build_execution_trace_payload(request, record.execution_trace, asset_store),
        segmentation_trace=_build_segmentation_trace_payload(request, record.segmentation_trace, asset_store),
        fallback_trace=_dump_fallback_trace(record.fallback_trace),
        phases=_build_phase_payloads(request, record.phases, asset_store),
        events=_dump_job_events(record.events),
        stage_timings=compute_stage_timings(record.events),
        feedback=_dump_feedback_items(record.feedback),
    )
