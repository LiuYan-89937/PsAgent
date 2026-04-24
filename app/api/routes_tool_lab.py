"""Tool-lab routes for deterministic tool experimentation and visual comparison."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_asset_store
from app.api.routes_assets import _build_asset_response
from app.api.schemas import (
    ToolLabMaskRequest,
    ToolLabMaskResponse,
    ToolLabRunRequest,
    ToolLabRunResponse,
    ToolLabStepResultResponse,
)
from app.services.asset_store import AssetStore
from app.services.planner_param_codec import normalize_runtime_tool_params
from app.services.tool_runtime import execute_tool_lab_chain, generate_mask
from app.tools import require_tool_spec


router = APIRouter(prefix="/tool-lab", tags=["tool-lab"])


def _tool_lab_work_dir(asset_store: AssetStore, *, purpose: str) -> Path:
    """Return a stable local work directory for temporary tool-lab artifacts."""

    work_dir = Path(asset_store.root_dir) / "_tool_lab" / purpose / uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


@router.post("/masks", response_model=ToolLabMaskResponse)
async def generate_tool_lab_mask(
    request: Request,
    payload: ToolLabMaskRequest,
    asset_store: AssetStore = Depends(get_asset_store),
) -> ToolLabMaskResponse:
    """Generate a SAM/segmentation mask for one uploaded image."""

    try:
        input_record = asset_store.require(payload.input_asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Input asset not found.") from exc

    work_dir = _tool_lab_work_dir(asset_store, purpose="masks")
    try:
        result = generate_mask(
            input_record.local_path,
            region=payload.prompt,
            mask_params={
                "mask_provider": payload.provider,
                "mask_prompt": payload.prompt,
                "mask_semantic_type": True,
            },
            output_dir=str(work_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Mask generation failed: {exc}") from exc

    mask_record = asset_store.save_generated(
        result.binary_mask_path,
        filename=f"mask_{payload.prompt.replace(' ', '_')}.png",
        media_type="image/png",
    )
    preview_record = (
        asset_store.save_generated(
            result.segmentation_rgba_path,
            filename=f"mask_preview_{payload.prompt.replace(' ', '_')}.png",
            media_type="image/png",
        )
        if result.segmentation_rgba_path
        else None
    )
    return ToolLabMaskResponse(
        mask_asset=_build_asset_response(request, mask_record),
        preview_asset=_build_asset_response(request, preview_record) if preview_record is not None else None,
        provider=result.provider,
        requested_provider=result.requested_provider,
        prompt=payload.prompt,
        effective_prompt=result.effective_prompt,
        fallback_used=bool(result.fallback_used),
        attempt_strategy=result.attempt_strategy,
        attempt_index=result.attempt_index,
        target_label=result.target_label,
        revert_mask=result.revert_mask,
    )


@router.post("/run", response_model=ToolLabRunResponse)
async def run_tool_lab_pipeline(
    request: Request,
    payload: ToolLabRunRequest,
    asset_store: AssetStore = Depends(get_asset_store),
) -> ToolLabRunResponse:
    """Run a custom deterministic tool chain on one uploaded image."""

    try:
        input_record = asset_store.require(payload.input_asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Input asset not found.") from exc

    mask_records_by_path: dict[str, object] = {}
    runtime_steps: list[dict[str, object]] = []
    step_results: list[ToolLabStepResultResponse] = []

    for index, step in enumerate(payload.steps):
        try:
            tool_spec = require_tool_spec(step.tool_name)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {step.tool_name}") from exc

        mask_record = None
        if step.mask_asset_id:
            if not tool_spec.supports_mask:
                raise HTTPException(status_code=400, detail=f"{step.tool_name} does not support masks.")
            try:
                mask_record = asset_store.require(step.mask_asset_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Mask asset not found: {step.mask_asset_id}") from exc
        elif tool_spec.requires_mask or not tool_spec.supports_whole_image:
            recommended_prompt = tool_spec.recommended_mask_prompt or "matching local region"
            raise HTTPException(
                status_code=400,
                detail=f"{step.tool_name} requires a mask. Generate or upload a '{recommended_prompt}' mask before running it.",
            )

        runtime_params = normalize_runtime_tool_params(step.tool_name, step.params)
        tool_params = dict(runtime_params)
        if mask_record is not None:
            tool_params["mask_path"] = mask_record.local_path
            mask_records_by_path[mask_record.local_path] = mask_record
        runtime_steps.append(
            {
                "op": step.tool_name,
                "region": "masked_region" if mask_record is not None else "whole_image",
                "params": tool_params,
                "priority": index,
            }
        )

    final_output_path, runtime_results = execute_tool_lab_chain(
        input_image_path=input_record.local_path,
        steps=runtime_steps,
        writer=lambda *_args, **_kwargs: None,
    )
    records_by_path: dict[str, object] = {input_record.local_path: input_record}
    final_record = input_record
    for result in runtime_results:
        input_asset = records_by_path.get(result.input_image_path, final_record)
        output_asset = None
        if result.output_image_path:
            output_record = asset_store.save_generated(
                result.output_image_path,
                filename=Path(result.output_image_path).name,
                media_type="image/png",
            )
            records_by_path[result.output_image_path] = output_record
            output_asset = _build_asset_response(request, output_record)
            final_record = output_record
        mask_record = mask_records_by_path.get(result.mask_path or "")
        step_results.append(
            ToolLabStepResultResponse(
                index=result.index,
                tool_name=result.tool_name,
                ok=result.ok,
                input_asset=_build_asset_response(request, input_asset),
                output_asset=output_asset,
                mask_asset=_build_asset_response(request, mask_record) if mask_record is not None else None,
                applied_params=result.applied_params,
                warnings=list(result.warnings),
                artifacts=dict(result.artifacts),
                fallback_used=bool(result.fallback_used),
                error=result.error,
            )
        )
        if not result.ok:
            break

    return ToolLabRunResponse(
        input_asset=_build_asset_response(request, input_record),
        final_output_asset=_build_asset_response(request, final_record),
        steps=step_results,
    )
