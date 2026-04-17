"""Hybrid editing subgraph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.config import get_stream_writer

from app.graph.state import EditPlan, EditState, ExecutionTraceItem, SegmentationTraceItem
from app.tools.packages import OperationContext, build_default_package_registry
from app.tools.packages.macros import expand_macro_operations, operations_require_hybrid
from app.tools.segmentation_tools import resolve_region_mask


def _build_operation_context(state: EditState, image_path: str) -> OperationContext:
    """Create the shared operation context for package execution."""

    return OperationContext(
        image_path=image_path,
        image_analysis=state.get("image_analysis") or {},
        retrieved_prefs=state.get("retrieved_prefs") or [],
        masks=state.get("masks") or {},
        thread_id=state.get("thread_id"),
        audit={},
    )


def _normalize_runtime_plan(state: EditState) -> dict[str, Any]:
    """Normalize a runtime plan before hybrid executor consumption."""

    raw_plan = dict(state.get("edit_plan") or {})
    raw_plan.setdefault("mode", str(state.get("mode") or "auto"))
    raw_plan.setdefault("domain", str((state.get("image_analysis") or {}).get("domain", "general")))
    raw_plan["executor"] = "hybrid" if operations_require_hybrid(raw_plan.get("operations", [])) else "deterministic"
    return EditPlan.model_validate(raw_plan).model_dump(mode="json")


def _safe_stream_writer():
    """Return a stream writer when inside LangGraph runtime, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def _build_mask_cache_key(region: str, mask_options: dict[str, Any]) -> str:
    """Build a stable cache key for region masks, including semantic mask hints."""

    if not mask_options:
        return region

    parts = [region]
    for key in sorted(mask_options):
        serialized = json.dumps(mask_options[key], sort_keys=True, ensure_ascii=True)
        parts.append(f"{key}={serialized}")
    return "|".join(parts)


def execute_hybrid(state: EditState) -> dict:
    """Run local-region aware deterministic editing with realtime segmentation.

    当前阶段的 hybrid 先聚焦“局部参数增强”：
    1. 遇到局部 region 时先请求阿里云主体分割；
    2. 把 mask 写回 state 级缓存；
    3. 再按确定性工具包顺序执行。
    """

    input_images = state.get("input_images") or []
    if not input_images:
        return {
            "candidate_outputs": state.get("candidate_outputs", []),
            "execution_trace": [
                {
                    "stage": "execute_hybrid",
                    "ok": False,
                    "error": "No input image available.",
                },
            ],
            "selected_output": None,
        }

    registry = build_default_package_registry()
    current_round = int(state.get("current_round") or 1)
    round_key = f"round_{current_round}"
    current_image = str(state.get("selected_output") or input_images[0]) if current_round > 1 else input_images[0]
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = list(state.get("execution_trace") or [])
    round_execution_trace: list[dict[str, Any]] = []
    segmentation_trace = list(state.get("segmentation_trace") or [])
    round_segmentation_trace: list[dict[str, Any]] = []
    round_outputs = dict(state.get("round_outputs") or {})
    round_execution_traces = dict(state.get("round_execution_traces") or {})
    round_segmentation_traces = dict(state.get("round_segmentation_traces") or {})
    mask_cache = dict(state.get("masks") or {})
    plan = _normalize_runtime_plan(state)
    expanded_operations = expand_macro_operations(plan.get("operations", []))
    writer = _safe_stream_writer()

    writer(
        {
            "event": "round_started",
            "stage": "execute_hybrid",
            "round": round_key,
            "message": f"开始执行 {round_key}",
        },
    )

    for index, operation in enumerate(expanded_operations):
        package = registry.require(operation["op"])
        region = operation.get("region", "whole_image")
        mask_options = package.get_mask_runtime_options(operation)
        requires_mask = bool(mask_options)
        mask_cache_key = _build_mask_cache_key(region, mask_options)
        current_mask_path = mask_cache.get(mask_cache_key) if requires_mask else None
        writer(
            {
                "event": "package_started",
                "stage": "execute_hybrid",
                "round": round_key,
                "op": operation["op"],
                "region": region,
                "message": f"正在执行 {operation['op']}",
            },
        )

        if requires_mask and current_mask_path is None:
            requested_provider = str(mask_options.get("provider") or "auto")
            requested_target = str(mask_options.get("prompt") or region)
            writer(
                {
                    "event": "segmentation_started",
                    "stage": "execute_hybrid",
                    "round": round_key,
                    "region": region,
                    "provider": requested_provider,
                    "prompt": mask_options.get("prompt"),
                    "negative_prompt": mask_options.get("negative_prompt"),
                    "message": f"正在准备 {requested_target} 的区域遮罩",
                },
            )
            mask_output_dir = str(
                Path(current_image).resolve().parent / "output" / f"{Path(current_image).stem}_{region}_mask"
            )
            try:
                segmentation_result = resolve_region_mask(
                    current_image,
                    region,
                    output_dir=mask_output_dir,
                    **mask_options,
                )
            except Exception as error:
                writer(
                    {
                        "event": "segmentation_failed",
                        "stage": "execute_hybrid",
                        "round": round_key,
                        "region": region,
                        "provider": requested_provider,
                        "prompt": mask_options.get("prompt"),
                        "negative_prompt": mask_options.get("negative_prompt"),
                        "message": f"{requested_target} 的区域遮罩生成失败",
                        "error": str(error),
                    },
                )
                raise

            current_mask_path = segmentation_result.binary_mask_path
            mask_cache[mask_cache_key] = current_mask_path
            segmentation_item = SegmentationTraceItem(
                index=len(segmentation_trace),
                stage=round_key,
                source_op=operation["op"],
                region=region,
                provider=segmentation_result.provider,
                requested_provider=segmentation_result.requested_provider or requested_provider,
                target_label=segmentation_result.target_label or requested_target,
                prompt=segmentation_result.prompt,
                negative_prompt=segmentation_result.negative_prompt,
                semantic_type=segmentation_result.semantic_type,
                ok=True,
                fallback_used=segmentation_result.fallback_used,
                mask_path=current_mask_path,
                request_id=segmentation_result.request_id,
                api_chain=list(segmentation_result.api_chain),
            ).model_dump(mode="json")
            segmentation_trace.append(segmentation_item)
            round_segmentation_trace.append(segmentation_item)
            writer(
                {
                    "event": "segmentation_finished",
                    "stage": "execute_hybrid",
                    "round": round_key,
                    "region": region,
                    "provider": segmentation_result.provider,
                    "requested_provider": segmentation_result.requested_provider or requested_provider,
                    "target_label": segmentation_result.target_label or requested_target,
                    "prompt": segmentation_result.prompt,
                    "negative_prompt": segmentation_result.negative_prompt,
                    "fallback_used": segmentation_result.fallback_used,
                    "mask_path": current_mask_path,
                    "message": f"{segmentation_result.target_label or requested_target} 的区域遮罩已生成",
                },
            )

        if requires_mask and current_mask_path is not None:
            mask_cache[region] = current_mask_path

        context = _build_operation_context(
            {
                **state,
                "masks": {
                    **mask_cache,
                    **({region: current_mask_path} if requires_mask and current_mask_path else {}),
                },
            },
            current_image,
        )
        result = package.execute(operation, context)

        trace_item = ExecutionTraceItem(
            index=index,
            stage=round_key,
            op=operation["op"],
            region=region,
            ok=result.ok,
            fallback_used=result.fallback_used,
            error=result.error,
            output_image=result.output_image,
            applied_params=result.applied_params,
            mask_path=current_mask_path if requires_mask else None,
        )
        if requires_mask:
            trace_item.mask_path = current_mask_path
        round_trace_item = trace_item.model_dump(mode="json")
        execution_trace.append(round_trace_item)
        round_execution_trace.append(round_trace_item)

        if result.ok and result.output_image:
            current_image = result.output_image
            candidate_outputs.append(result.output_image)

        writer(
            {
                "event": "package_finished" if result.ok else "package_failed",
                "stage": "execute_hybrid",
                "round": round_key,
                "op": operation["op"],
                "region": region,
                "ok": result.ok,
                "message": f"{operation['op']} 执行完成" if result.ok else f"{operation['op']} 执行失败",
                "error": result.error,
                "warnings": result.warnings,
                "artifacts": result.artifacts,
                "mask_path": current_mask_path if requires_mask else None,
            },
        )

    round_outputs[round_key] = current_image
    round_execution_traces[round_key] = round_execution_trace
    round_segmentation_traces[round_key] = round_segmentation_trace
    writer(
        {
            "event": "round_completed",
            "stage": "execute_hybrid",
            "round": round_key,
            "message": f"{round_key} 执行完成",
        },
    )

    return {
        "candidate_outputs": candidate_outputs,
        "execution_trace": execution_trace,
        "segmentation_trace": segmentation_trace,
        "selected_output": current_image,
        "masks": mask_cache,
        "round_outputs": round_outputs,
        "round_execution_traces": round_execution_traces,
        "round_segmentation_traces": round_segmentation_traces,
    }
