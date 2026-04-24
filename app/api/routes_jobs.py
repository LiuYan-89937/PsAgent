"""Job query routes."""

from __future__ import annotations

import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_asset_store, get_job_store
from app.api.routes_assets import _build_asset_response
from app.api.runtime import compute_round_timings, format_sse, make_event
from app.api.schemas import JobDetailResponse, JobSummaryResponse
from app.graph.state import (
    ExecutionTraceItem,
    FallbackTraceItem,
    FeedbackItem,
    JobEvent,
    SearchRoundArtifact,
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
        current_round=record.current_round,
        current_focus=record.current_focus,
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
    return [item.model_dump(mode="json") if isinstance(item, FallbackTraceItem) else dict(item) for item in trace]


def _dump_job_events(events: list[JobEvent]) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in events]


def _dump_feedback_items(items: list[FeedbackItem]) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in items]


def _hydrate_candidate_execution(request: Request, execution_payload: dict[str, object], asset_store: AssetStore) -> dict[str, object]:
    output_asset = None
    output_asset_id = execution_payload.get("output_asset_id")
    if isinstance(output_asset_id, str):
        try:
            output_asset = _build_asset_response(request, asset_store.require(output_asset_id))
        except KeyError:
            output_asset = None
    execution_payload["output_asset"] = output_asset
    raw_execution_trace = execution_payload.get("execution_trace")
    if isinstance(raw_execution_trace, list):
        execution_payload["execution_trace"] = _build_execution_trace_payload(request, raw_execution_trace, asset_store)
    raw_segmentation_trace = execution_payload.get("segmentation_trace")
    if isinstance(raw_segmentation_trace, list):
        execution_payload["segmentation_trace"] = _build_segmentation_trace_payload(request, raw_segmentation_trace, asset_store)
    return execution_payload


def _build_round_payloads(
    request: Request,
    rounds: list[SearchRoundArtifact],
    asset_store: AssetStore,
) -> list[dict[str, object]]:
    """Expand round artifacts into frontend payloads."""

    payload: list[dict[str, object]] = []
    for round_item in rounds:
        round_payload = round_item.model_dump(mode="json")
        output_asset = None
        output_asset_id = round_payload.get("output_asset_id")
        if isinstance(output_asset_id, str):
            try:
                output_asset = _build_asset_response(request, asset_store.require(output_asset_id))
            except KeyError:
                output_asset = None
        round_payload["output_asset"] = output_asset
        full_execution = round_payload.get("selected_full_execution")
        if isinstance(full_execution, dict):
            round_payload["selected_full_execution"] = _hydrate_candidate_execution(request, full_execution, asset_store)
        for key in ("candidates", "recovery_candidates"):
            raw_candidates = round_payload.get(key)
            if not isinstance(raw_candidates, list):
                continue
            candidates_payload = []
            for candidate in raw_candidates:
                candidate_payload = dict(candidate) if isinstance(candidate, dict) else {}
                execution_payload = candidate_payload.get("preview_execution")
                if isinstance(execution_payload, dict):
                    candidate_payload["preview_execution"] = _hydrate_candidate_execution(request, execution_payload, asset_store)
                candidates_payload.append(candidate_payload)
            round_payload[key] = candidates_payload
        payload.append(round_payload)
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
        objective_card=record.objective_card,
        rounds=_build_round_payloads(request, record.rounds, asset_store),
        selected_candidate_id=record.selected_candidate_id,
        final_review=record.final_review,
        final_execution_trace=_build_execution_trace_payload(request, record.final_execution_trace, asset_store),
        events=_dump_job_events(record.events),
        round_timings=compute_round_timings(record.events),
        feedback=_dump_feedback_items(record.feedback),
    )


@router.get("/{job_id}/events/stream")
async def stream_job_events(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> StreamingResponse:
    """Stream persisted job events so the frontend can reconnect after review/resume."""

    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream() -> Iterator[str]:
        sent = 0
        idle_ticks = 0
        while idle_ticks < 160:
            record = job_store.require(job_id)
            events = record.events
            while sent < len(events):
                event = events[sent]
                sent += 1
                yield format_sse(event.event, event)

            if record.status in {"completed", "failed", "review_required"}:
                terminal = make_event(
                    "job_status",
                    job_id=job_id,
                    round=record.current_round,
                    focus=record.current_focus,
                    message=record.current_message or record.status,
                    ok=record.status == "completed",
                    payload={"status": record.status},
                )
                yield format_sse("job_status", terminal)
                return

            idle_ticks += 1
            time.sleep(0.75)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
