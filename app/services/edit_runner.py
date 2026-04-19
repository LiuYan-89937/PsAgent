"""Shared orchestration helpers for sync and streaming edit routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.api.runtime import build_graph_config
from app.api.schemas import EditRequest
from app.services.asset_store import AssetStore
from app.services.job_store import JobRecord, JobStore


@dataclass(slots=True)
class PreparedEditRun:
    """Normalized edit run inputs shared by sync and streaming routes."""

    payload: EditRequest
    thread_id: str
    graph_input: dict[str, Any]
    config: dict[str, Any]
    job: JobRecord
    input_asset_ids: list[str]
    input_image_paths: list[str]


@dataclass(slots=True)
class FinalizedEditRun:
    """Finalized job payload plus persisted generated assets."""

    job: JobRecord
    output_records_by_path: dict[str, object]


def _attach_output_asset_ids_to_trace(
    trace: list[dict[str, object]],
    output_records_by_path: dict[str, object],
) -> list[dict[str, object]]:
    """Annotate trace items with persisted output asset ids."""

    payload: list[dict[str, object]] = []
    for item in trace:
        trace_item = dict(item)
        output_image = trace_item.get("output_image")
        if isinstance(output_image, str):
            record = output_records_by_path.get(output_image)
            asset_id = getattr(record, "asset_id", None)
            if isinstance(asset_id, str):
                trace_item["output_asset_id"] = asset_id
        payload.append(trace_item)
    return payload


def _attach_output_asset_ids_to_round_traces(
    round_execution_traces: dict[str, object],
    output_records_by_path: dict[str, object],
) -> dict[str, object]:
    """Annotate round execution traces with persisted output asset ids."""

    payload: dict[str, object] = {}
    for round_key, items in round_execution_traces.items():
        if isinstance(items, list):
            payload[round_key] = _attach_output_asset_ids_to_trace(items, output_records_by_path)
        else:
            payload[round_key] = items
    return payload


def collect_input_image_paths(payload: EditRequest, asset_store: AssetStore) -> tuple[list[str], list[str]]:
    """Resolve uploaded asset ids and raw image paths into concrete input paths."""

    input_asset_ids = list(payload.input_asset_ids)
    input_image_paths = list(payload.input_image_paths)
    for asset_id in input_asset_ids:
        input_image_paths.append(asset_store.require(asset_id).local_path)
    return input_asset_ids, input_image_paths


def prepare_edit_run(
    payload: EditRequest,
    *,
    asset_store: AssetStore,
    job_store: JobStore,
) -> PreparedEditRun:
    """Create a job record and normalized graph input for an edit run."""

    input_asset_ids, input_image_paths = collect_input_image_paths(payload, asset_store)
    if not input_image_paths:
        raise ValueError("No input images provided")

    raw_instruction = (payload.instruction or "").strip() or None
    thread_id = payload.thread_id or f"thread-{uuid4().hex}"
    job = job_store.create_job(
        user_id=payload.user_id,
        thread_id=thread_id,
        request_text=raw_instruction,
        input_asset_ids=input_asset_ids,
    )
    job_store.set_status(job.job_id, "running")

    graph_input = {
        "user_id": payload.user_id,
        "thread_id": thread_id,
        "input_images": input_image_paths,
        "request_text": raw_instruction or "",
        "planner_thinking_mode": bool(payload.planner_thinking_mode),
        "mode": "auto" if payload.auto_mode else "explicit",
    }
    return PreparedEditRun(
        payload=payload,
        thread_id=thread_id,
        graph_input=graph_input,
        config=build_graph_config(thread_id),
        job=job,
        input_asset_ids=input_asset_ids,
        input_image_paths=input_image_paths,
    )


def finalize_edit_run(
    *,
    job_store: JobStore,
    asset_store: AssetStore,
    job_id: str,
    final_state: dict[str, Any],
    current_stage: str | None,
    current_message: str | None,
    status: str,
) -> FinalizedEditRun:
    """Persist final graph outputs, traces, and generated assets for a completed run."""

    output_records_by_path: dict[str, object] = {}
    output_asset_ids: list[str] = []
    for image_path in final_state.get("candidate_outputs") or []:
        if image_path in output_records_by_path:
            continue
        record = asset_store.save_generated(image_path)
        output_records_by_path[image_path] = record
        output_asset_ids.append(record.asset_id)

    selected_output = final_state.get("selected_output")
    if selected_output and selected_output not in output_records_by_path:
        record = asset_store.save_generated(selected_output)
        output_records_by_path[selected_output] = record
        output_asset_ids.append(record.asset_id)

    round_output_asset_ids: dict[str, str] = {}
    for round_key, image_path in dict(final_state.get("round_outputs") or {}).items():
        if not image_path:
            continue
        if image_path not in output_records_by_path:
            record = asset_store.save_generated(image_path)
            output_records_by_path[image_path] = record
            output_asset_ids.append(record.asset_id)
        round_output_asset_ids[round_key] = output_records_by_path[image_path].asset_id

    execution_trace = _attach_output_asset_ids_to_trace(
        final_state.get("execution_trace") or [],
        output_records_by_path,
    )
    round_execution_traces = _attach_output_asset_ids_to_round_traces(
        final_state.get("round_execution_traces") or {},
        output_records_by_path,
    )

    completed = job_store.set_execution_result(
        job_id,
        status=status,  # type: ignore[arg-type]
        output_asset_ids=output_asset_ids,
        round_output_asset_ids=round_output_asset_ids,
        edit_plan=final_state.get("edit_plan"),
        eval_report=final_state.get("eval_report"),
        execution_trace=execution_trace,
        segmentation_trace=final_state.get("segmentation_trace") or [],
        fallback_trace=final_state.get("fallback_trace") or [],
        round_plans=final_state.get("round_plans") or {},
        round_eval_reports=final_state.get("round_eval_reports") or {},
        round_execution_traces=round_execution_traces,
        round_segmentation_traces=final_state.get("round_segmentation_traces") or {},
        approval_required=bool(final_state.get("approval_required")),
        approval_payload=final_state.get("approval_payload"),
        request_text=final_state.get("request_text"),
        current_stage=current_stage,
        current_message=current_message,
    )
    return FinalizedEditRun(job=completed, output_records_by_path=output_records_by_path)
