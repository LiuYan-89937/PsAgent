"""Shared orchestration helpers for sync and streaming edit routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.api.runtime import build_graph_config
from app.api.schemas import EditRequest
from app.graph.state import SearchRoundArtifact, coerce_search_rounds
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
    trace: list[object],
    output_records_by_path: dict[str, object],
) -> list[dict[str, object]]:
    """Annotate trace items with persisted output asset ids."""

    payload: list[dict[str, object]] = []
    for item in trace:
        trace_item = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        output_image = trace_item.get("output_image")
        if isinstance(output_image, str):
            record = output_records_by_path.get(output_image)
            asset_id = getattr(record, "asset_id", None)
            if isinstance(asset_id, str):
                trace_item["output_asset_id"] = asset_id
        payload.append(trace_item)
    return payload


def _attach_output_asset_ids_to_segmentation_trace(
    trace: list[object],
    asset_store: AssetStore,
    output_records_by_path: dict[str, object],
) -> list[dict[str, object]]:
    """Persist segmentation mask/preview images and annotate trace items with asset ids."""

    payload: list[dict[str, object]] = []
    for item in trace:
        trace_item = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        mask_path = trace_item.get("mask_path")
        preview_path = trace_item.get("preview_path")
        if isinstance(mask_path, str):
            record = output_records_by_path.get(mask_path)
            if record is None:
                record = asset_store.save_generated(mask_path)
                output_records_by_path[mask_path] = record
            asset_id = getattr(record, "asset_id", None)
            if isinstance(asset_id, str):
                trace_item["mask_asset_id"] = asset_id
        if isinstance(preview_path, str):
            record = output_records_by_path.get(preview_path)
            if record is None:
                record = asset_store.save_generated(preview_path)
                output_records_by_path[preview_path] = record
            asset_id = getattr(record, "asset_id", None)
            if isinstance(asset_id, str):
                trace_item["preview_asset_id"] = asset_id
        payload.append(trace_item)
    return payload


def _attach_output_asset_ids_to_rounds(
    rounds: list[SearchRoundArtifact | dict[str, object]],
    asset_store: AssetStore,
    output_records_by_path: dict[str, object],
) -> list[dict[str, object]]:
    """Persist round outputs, candidate previews, and traces into JSON-safe payloads."""

    payload: list[dict[str, object]] = []
    for round_value in coerce_search_rounds(rounds):
        round_payload = round_value.model_dump(mode="json")
        output_image_path = round_payload.get("output_image_path")
        if isinstance(output_image_path, str):
            record = output_records_by_path.get(output_image_path)
            if record is None:
                record = asset_store.save_generated(output_image_path)
                output_records_by_path[output_image_path] = record
            round_payload["output_asset_id"] = getattr(record, "asset_id", None)

        full_execution = round_payload.get("selected_full_execution")
        if isinstance(full_execution, dict):
            _hydrate_execution_payload(full_execution, asset_store, output_records_by_path)
            round_payload["selected_full_execution"] = full_execution

        for key in ("candidates", "recovery_candidates"):
            raw_candidates = round_payload.get(key)
            if not isinstance(raw_candidates, list):
                continue
            hydrated_candidates: list[dict[str, object]] = []
            for candidate in raw_candidates:
                candidate_payload = dict(candidate) if isinstance(candidate, dict) else {}
                execution_payload = candidate_payload.get("preview_execution")
                if isinstance(execution_payload, dict):
                    _hydrate_execution_payload(execution_payload, asset_store, output_records_by_path)
                    candidate_payload["preview_execution"] = execution_payload
                hydrated_candidates.append(candidate_payload)
            round_payload[key] = hydrated_candidates
        payload.append(round_payload)
    return payload


def _hydrate_execution_payload(
    execution_payload: dict[str, object],
    asset_store: AssetStore,
    output_records_by_path: dict[str, object],
) -> None:
    """Persist and annotate one candidate execution payload in-place."""

    output_image_path = execution_payload.get("output_image_path")
    if isinstance(output_image_path, str):
        record = output_records_by_path.get(output_image_path)
        if record is None:
            record = asset_store.save_generated(output_image_path)
            output_records_by_path[output_image_path] = record
        execution_payload["output_asset_id"] = getattr(record, "asset_id", None)
    candidate_execution_trace = execution_payload.get("execution_trace")
    if isinstance(candidate_execution_trace, list):
        execution_payload["execution_trace"] = _attach_output_asset_ids_to_trace(
            candidate_execution_trace,
            output_records_by_path,
        )
    candidate_seg_trace = execution_payload.get("segmentation_trace")
    if isinstance(candidate_seg_trace, list):
        execution_payload["segmentation_trace"] = _attach_output_asset_ids_to_segmentation_trace(
            candidate_seg_trace,
            asset_store,
            output_records_by_path,
        )


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
    current_round: str | None,
    current_focus: str | None,
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

    execution_trace = _attach_output_asset_ids_to_trace(
        final_state.get("execution_trace") or [],
        output_records_by_path,
    )
    segmentation_trace = _attach_output_asset_ids_to_segmentation_trace(
        final_state.get("segmentation_trace") or [],
        asset_store,
        output_records_by_path,
    )
    final_execution_trace = _attach_output_asset_ids_to_trace(
        final_state.get("final_execution_trace") or final_state.get("execution_trace") or [],
        output_records_by_path,
    )
    rounds = _attach_output_asset_ids_to_rounds(
        final_state.get("rounds") or [],
        asset_store,
        output_records_by_path,
    )

    completed = job_store.set_execution_result(
        job_id,
        status=status,  # type: ignore[arg-type]
        output_asset_ids=output_asset_ids,
        edit_plan=final_state.get("edit_plan"),
        eval_report=final_state.get("eval_report"),
        execution_trace=execution_trace,
        final_execution_trace=final_execution_trace,
        segmentation_trace=segmentation_trace,
        fallback_trace=final_state.get("fallback_trace") or [],
        objective_card=final_state.get("objective_card"),
        rounds=rounds,
        selected_candidate_id=final_state.get("selected_candidate_id"),
        final_review=final_state.get("final_review"),
        approval_required=bool(final_state.get("approval_required")),
        approval_payload=final_state.get("approval_payload"),
        request_text=final_state.get("request_text"),
        current_round=current_round,
        current_focus=current_focus,
        current_message=current_message,
    )
    return FinalizedEditRun(job=completed, output_records_by_path=output_records_by_path)
