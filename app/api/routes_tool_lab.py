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
from app.tools import require_tool, require_tool_spec
from app.tools.common import ToolExecutionResult
from app.tools.segmentation_tools import resolve_region_mask


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
        # 这里显式走 resolve_region_mask，
        # 让前端测试页也复用正式的分割 provider 选择与 fallback 逻辑。
        result = resolve_region_mask(
            input_record.local_path,
            payload.prompt,
            provider=payload.provider,
            prompt=payload.prompt,
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

    current_record = input_record
    step_results: list[ToolLabStepResultResponse] = []

    for index, step in enumerate(payload.steps):
        try:
            tool = require_tool(step.tool_name)
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
        tool_input = {
            "image_path": current_record.local_path,
            **runtime_params,
        }
        if mask_record is not None:
            tool_input["mask_path"] = mask_record.local_path

        try:
            # 工具实验室保持最直接的串行语义：
            # 上一步输出图就是下一步输入图，便于肉眼对比每一步效果。
            raw_result = tool.invoke(tool_input)
            result = ToolExecutionResult.model_validate(raw_result)
        except Exception as exc:
            step_results.append(
                ToolLabStepResultResponse(
                    index=index,
                    tool_name=step.tool_name,
                    ok=False,
                    input_asset=_build_asset_response(request, current_record),
                    output_asset=None,
                    mask_asset=_build_asset_response(request, mask_record) if mask_record is not None else None,
                    applied_params=runtime_params,
                    warnings=[],
                    artifacts={},
                    fallback_used=True,
                    error=str(exc),
                )
            )
            break

        if not result.ok or not result.output_image:
            step_results.append(
                ToolLabStepResultResponse(
                    index=index,
                    tool_name=step.tool_name,
                    ok=False,
                    input_asset=_build_asset_response(request, current_record),
                    output_asset=None,
                    mask_asset=_build_asset_response(request, mask_record) if mask_record is not None else None,
                    applied_params=result.applied_params,
                    warnings=list(result.warnings),
                    artifacts=dict(result.artifacts),
                    fallback_used=bool(result.fallback_used),
                    error=result.error or f"{step.tool_name} did not produce an output image.",
                )
            )
            break

        output_record = asset_store.save_generated(
            result.output_image,
            filename=Path(result.output_image).name,
            media_type="image/png",
        )
        step_results.append(
            ToolLabStepResultResponse(
                index=index,
                tool_name=step.tool_name,
                ok=True,
                input_asset=_build_asset_response(request, current_record),
                output_asset=_build_asset_response(request, output_record),
                mask_asset=_build_asset_response(request, mask_record) if mask_record is not None else None,
                applied_params=result.applied_params,
                warnings=list(result.warnings),
                artifacts=dict(result.artifacts),
                fallback_used=bool(result.fallback_used),
                error=result.error,
            )
        )
        current_record = output_record

    return ToolLabRunResponse(
        input_asset=_build_asset_response(request, input_record),
        final_output_asset=_build_asset_response(request, current_record),
        steps=step_results,
    )
