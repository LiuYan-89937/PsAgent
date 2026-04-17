"""Job query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_asset_store, get_job_store
from app.api.routes_assets import _build_asset_response
from app.api.runtime import compute_stage_timings
from app.api.schemas import JobDetailResponse, JobSummaryResponse
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
        current_stage=record.current_stage,
        current_message=record.current_message,
        error=record.error,
        error_detail=record.error_detail,
    )


def _build_round_output_assets(
    request: Request,
    record: JobRecord,
    asset_store: AssetStore,
) -> dict[str, object | None]:
    """Convert stored round output asset ids into frontend asset payloads."""

    payload: dict[str, object | None] = {}
    for round_key, asset_id in record.round_output_asset_ids.items():
        try:
            payload[round_key] = _build_asset_response(request, asset_store.require(asset_id))
        except KeyError:
            payload[round_key] = None
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
        edit_plan=record.edit_plan,
        eval_report=record.eval_report,
        execution_trace=record.execution_trace,
        segmentation_trace=record.segmentation_trace,
        round_outputs=_build_round_output_assets(request, record, asset_store),
        round_plans=record.round_plans,
        round_eval_reports=record.round_eval_reports,
        round_execution_traces=record.round_execution_traces,
        round_segmentation_traces=record.round_segmentation_traces,
        events=record.events,
        stage_timings=compute_stage_timings(record.events),
        feedback=record.feedback,
    )
