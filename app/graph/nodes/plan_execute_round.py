"""Realtime planner tool-calling nodes for round execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.config import get_stream_writer

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import ApprovalPayload, EditOperation, EditPlan, EditState, ExecutionTraceItem, SegmentationTraceItem
from app.services.planner_tool_model import (
    FinishRoundParams,
    build_operation_from_tool_call,
    call_planner_tool_turn,
    extract_single_tool_call,
    planner_tool_model_available,
    resolve_planner_tool_name,
)
from app.tools.packages import OperationContext, build_default_package_registry
from app.tools.packages.base import WHOLE_IMAGE_REGION, strip_mask_params
from app.tools.packages.macros import expand_macro_operation, is_macro_tool, operations_require_hybrid
from app.tools.segmentation_tools import is_recoverable_empty_segmentation_error, resolve_region_mask


def _safe_stream_writer():
    """Return a stream writer when inside LangGraph runtime, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether the text contains any keyword."""

    return any(keyword in text for keyword in keywords)


def _build_operation_context(state: EditState, image_path: str, mask_cache: dict[str, str]) -> OperationContext:
    """Create the shared operation context for package execution."""

    return OperationContext(
        image_path=image_path,
        image_analysis=state.get("image_analysis") or {},
        retrieved_prefs=state.get("retrieved_prefs") or [],
        masks=mask_cache,
        thread_id=state.get("thread_id"),
        audit={},
    )


def _build_mask_cache_key(region: str, mask_options: dict[str, Any]) -> str:
    """Build a stable cache key for region masks within the same source image."""

    if not mask_options:
        return region

    parts = [region]
    for key in sorted(mask_options):
        serialized = json.dumps(mask_options[key], sort_keys=True, ensure_ascii=True)
        parts.append(f"{key}={serialized}")
    return "|".join(parts)


def _choose_executor(operations: list[dict[str, Any]]) -> str:
    """Pick the minimal executor implied by executed operations."""

    return "hybrid" if operations_require_hybrid(operations) else "deterministic"


def _compact_tool_result(
    operation: dict[str, Any],
    result,
    *,
    segmentation_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact result summary for the next planner step."""

    payload = {
        "op": operation["op"],
        "region": operation.get("region", "whole_image"),
        "ok": result.ok,
        "fallback_used": result.fallback_used,
        "error": result.error,
    }
    normalized_params = result.applied_params.get("params") if isinstance(result.applied_params.get("params"), dict) else result.applied_params
    if isinstance(normalized_params, dict):
        compact_params: dict[str, Any] = {}
        for key, value in normalized_params.items():
            if value in (None, "", False):
                continue
            compact_params[key] = value
            if len(compact_params) >= 6:
                break
        if compact_params:
            payload["params"] = compact_params
    if segmentation_item is not None:
        payload["segmentation"] = {
            "target_label": segmentation_item.get("target_label"),
            "prompt": segmentation_item.get("prompt"),
            "negative_prompt": segmentation_item.get("negative_prompt"),
            "provider": segmentation_item.get("provider"),
        }
    return payload


def _execute_single_tool_call(
    *,
    state: EditState,
    node_name: str,
    round_key: str,
    current_image: str,
    operation: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    round_execution_trace: list[dict[str, Any]],
    segmentation_trace: list[dict[str, Any]],
    round_segmentation_trace: list[dict[str, Any]],
    fallback_trace: list[dict[str, Any]],
    candidate_outputs: list[str],
    mask_cache: dict[str, str],
) -> tuple[str, dict[str, Any] | None]:
    """Execute one package tool call against the current image."""

    if is_macro_tool(str(operation.get("op") or "")):
        writer = _safe_stream_writer()
        expanded_operations = expand_macro_operation(
            operation,
            context=_build_operation_context(state, current_image, mask_cache),
        )
        writer(
            {
                "event": "planner_tool_expanded",
                "stage": node_name,
                "round": round_key,
                "op": operation["op"],
                "region": operation.get("region", "whole_image"),
                "message": f"规划器宏工具已展开：{operation['op']}",
                "expanded_ops": [item["op"] for item in expanded_operations],
            },
        )
        latest_summary: dict[str, Any] | None = None
        next_image = current_image
        macro_steps: list[dict[str, Any]] = []
        for sub_operation in expanded_operations:
            next_image, latest_summary = _execute_single_tool_call(
                state=state,
                node_name=node_name,
                round_key=round_key,
                current_image=next_image,
                operation=sub_operation,
                execution_trace=execution_trace,
                round_execution_trace=round_execution_trace,
                segmentation_trace=segmentation_trace,
                round_segmentation_trace=round_segmentation_trace,
                fallback_trace=fallback_trace,
                candidate_outputs=candidate_outputs,
                mask_cache=mask_cache,
            )
            if latest_summary is not None:
                macro_steps.append(latest_summary)

        return next_image, {
            "op": operation["op"],
            "region": operation.get("region", "whole_image"),
            "ok": True,
            "macro_steps": macro_steps,
        }

    writer = _safe_stream_writer()
    registry = build_default_package_registry()
    package = registry.require(operation["op"])
    region = operation.get("region", "whole_image")
    mask_options = package.get_mask_runtime_options(operation)
    requires_mask = bool(mask_options)

    writer(
        {
            "event": "planner_tool_called",
            "stage": node_name,
            "round": round_key,
            "op": operation["op"],
            "region": region,
            "message": f"规划器选择 {operation['op']}",
        },
    )
    writer(
        {
            "event": "package_started",
            "stage": node_name,
            "round": round_key,
            "op": operation["op"],
            "region": region,
            "message": f"正在执行 {operation['op']}",
        },
    )

    segmentation_item: dict[str, Any] | None = None
    current_mask_path: str | None = None
    if requires_mask:
        mask_cache_key = _build_mask_cache_key(region, mask_options)
        current_mask_path = mask_cache.get(mask_cache_key)

        if current_mask_path is None:
            requested_provider = str(mask_options.get("provider") or "auto")
            requested_target = str(mask_options.get("prompt") or region)
            segmentation_result = None
            writer(
                {
                    "event": "segmentation_started",
                    "stage": node_name,
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
                        "stage": node_name,
                        "round": round_key,
                        "region": region,
                        "provider": requested_provider,
                        "prompt": mask_options.get("prompt"),
                        "negative_prompt": mask_options.get("negative_prompt"),
                        "message": f"{requested_target} 的区域遮罩生成失败",
                        "error": str(error),
                    },
                )
                if package.spec.mask_policy != "required" and is_recoverable_empty_segmentation_error(error):
                    segmentation_item = SegmentationTraceItem(
                        index=len(segmentation_trace),
                        stage=round_key,
                        source_op=operation["op"],
                        region=region,
                        provider=requested_provider,
                        requested_provider=requested_provider,
                        target_label=requested_target,
                        prompt=str(mask_options.get("prompt") or "") or None,
                        negative_prompt=str(mask_options.get("negative_prompt") or "") or None,
                        semantic_type=bool(mask_options.get("semantic_type")) if "semantic_type" in mask_options else None,
                        ok=False,
                        fallback_used=True,
                        error=str(error),
                        mask_path=None,
                        api_chain=[],
                        attempt_index=getattr(error, "attempts", [{}])[-1].get("attempt_index") if getattr(error, "attempts", None) else None,
                        attempt_strategy=getattr(error, "attempts", [{}])[-1].get("attempt_strategy") if getattr(error, "attempts", None) else None,
                        requested_prompt=getattr(error, "attempts", [{}])[-1].get("requested_prompt") if getattr(error, "attempts", None) else None,
                        effective_prompt=getattr(error, "attempts", [{}])[-1].get("effective_prompt") if getattr(error, "attempts", None) else None,
                        revert_mask=getattr(error, "attempts", [{}])[-1].get("revert_mask") if getattr(error, "attempts", None) else None,
                        attempts=list(getattr(error, "attempts", []) or []),
                    ).model_dump(mode="json")
                    segmentation_trace.append(segmentation_item)
                    round_segmentation_trace.append(segmentation_item)
                    writer(
                        {
                            "event": "segmentation_skipped",
                            "stage": node_name,
                            "round": round_key,
                            "region": region,
                            "provider": requested_provider,
                            "message": f"{requested_target} 未返回可用遮罩，跳过该局部步骤",
                            "error": str(error),
                        },
                    )
                    fallback_trace[:] = append_fallback_trace(
                        fallback_trace,
                        stage=node_name,
                        source="segmentation_provider",
                        location=operation["op"],
                        strategy="skip_local_operation",
                        message="局部分割未返回可用遮罩，已跳过该局部步骤。",
                        error=str(error),
                    )
                    skipped_trace = ExecutionTraceItem(
                        index=len(execution_trace),
                        stage=round_key,
                        op=operation["op"],
                        region=region,
                        ok=False,
                        fallback_used=True,
                        error="Skipped: segmentation returned no usable mask.",
                        output_image=current_image,
                        applied_params={"params": dict(operation.get("params") or {})},
                        mask_path=None,
                    ).model_dump(mode="json")
                    execution_trace.append(skipped_trace)
                    round_execution_trace.append(skipped_trace)
                    writer(
                        {
                            "event": "package_skipped",
                            "stage": node_name,
                            "round": round_key,
                            "op": operation["op"],
                            "region": region,
                            "message": f"{operation['op']} 因分割无结果被跳过",
                            "error": str(error),
                        },
                    )
                    return current_image, {
                        "op": operation["op"],
                        "region": region,
                        "ok": False,
                        "fallback_used": True,
                        "error": "Skipped: segmentation returned no usable mask.",
                    }
                else:
                    raise RuntimeError(f"Planner tool {operation['op']} segmentation failed: {error}") from error

            if segmentation_result is not None:
                current_mask_path = segmentation_result.binary_mask_path
                mask_cache[mask_cache_key] = current_mask_path
                mask_cache[region] = current_mask_path
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
                    attempt_index=segmentation_result.attempt_index,
                    attempt_strategy=segmentation_result.attempt_strategy,
                    requested_prompt=segmentation_result.requested_prompt,
                    effective_prompt=segmentation_result.effective_prompt,
                    revert_mask=segmentation_result.revert_mask,
                    attempts=list(segmentation_result.attempts),
                ).model_dump(mode="json")
                segmentation_trace.append(segmentation_item)
                round_segmentation_trace.append(segmentation_item)
                writer(
                    {
                        "event": "segmentation_finished",
                        "stage": node_name,
                        "round": round_key,
                        "region": region,
                        "provider": segmentation_result.provider,
                        "requested_provider": segmentation_result.requested_provider or requested_provider,
                        "target_label": segmentation_item["target_label"],
                        "prompt": segmentation_item["prompt"],
                        "negative_prompt": segmentation_item["negative_prompt"],
                        "fallback_used": segmentation_item["fallback_used"],
                        "mask_path": current_mask_path,
                        "message": f"{segmentation_item['target_label']} 的区域遮罩已生成",
                    },
                )

    context = _build_operation_context(state, current_image, mask_cache)
    result = package.execute(operation, context)
    if not result.ok:
        writer(
            {
                "event": "package_failed",
                "stage": node_name,
                "round": round_key,
                "op": operation["op"],
                "region": region,
                "ok": False,
                "message": f"{operation['op']} 执行失败",
                "error": result.error,
                "warnings": result.warnings,
                "artifacts": result.artifacts,
                "mask_path": current_mask_path if requires_mask else None,
            },
        )
        writer(
            {
                "event": "planner_tool_failed",
                "stage": node_name,
                "round": round_key,
                "op": operation["op"],
                "region": region,
                "message": f"规划器调用 {operation['op']} 失败",
                "error": result.error,
            },
        )
        fallback_trace[:] = append_fallback_trace(
            fallback_trace,
            stage=node_name,
            source="package_execute",
            location=operation["op"],
            strategy="keep_current_image",
            message=f"{operation['op']} 执行失败，保留当前结果继续流程。",
            error=result.error or "unknown error",
        )
        return current_image, _compact_tool_result(operation, result, segmentation_item=segmentation_item)

    trace_item = ExecutionTraceItem(
        index=len(execution_trace),
        stage=round_key,
        op=operation["op"],
        region=region,
        ok=result.ok,
        fallback_used=result.fallback_used,
        error=result.error,
        output_image=result.output_image,
        applied_params=result.applied_params,
        mask_path=current_mask_path if requires_mask else None,
    ).model_dump(mode="json")
    execution_trace.append(trace_item)
    round_execution_trace.append(trace_item)

    writer(
        {
            "event": "package_finished",
            "stage": node_name,
            "round": round_key,
            "op": operation["op"],
            "region": region,
            "ok": True,
            "message": f"{operation['op']} 执行完成",
            "warnings": result.warnings,
            "artifacts": result.artifacts,
            "mask_path": current_mask_path if requires_mask else None,
        },
    )
    writer(
        {
            "event": "planner_tool_finished",
            "stage": node_name,
            "round": round_key,
            "op": operation["op"],
            "region": region,
            "message": f"规划器完成 {operation['op']}",
        },
    )

    next_image = current_image
    if result.output_image:
        next_image = result.output_image
        candidate_outputs.append(result.output_image)

    return next_image, _compact_tool_result(operation, result, segmentation_item=segmentation_item)


def _fallback_rule_execute_round(
    state: EditState,
    *,
    round_index: int,
    fallback_trace: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Fallback to a small local rule planner when realtime tool-calling is unavailable."""

    def build_rule_operations() -> list[dict[str, Any]]:
        registry = build_default_package_registry()
        request_intent = dict(state.get("request_intent") or {"mode": state.get("mode") or "auto"})
        requested_packages = list(request_intent.get("requested_packages", []))
        image_analysis = dict(state.get("image_analysis") or {})
        issues = set(str(item) for item in image_analysis.get("issues", []))
        request_text = str(state.get("request_text") or "")
        constraints = {str(item) for item in request_intent.get("constraints", [])}
        operations: list[dict[str, Any]] = []

        def append_operation(op: str, *, region: str = "whole_image", params: dict[str, Any] | None = None, strength: float | None = None) -> None:
            if any(item["op"] == op and item.get("region", "whole_image") == region for item in operations):
                return
            package = registry.require(op)
            payload = {
                "op": op,
                "region": region,
                "strength": strength,
                "params": package.get_operation_params({"op": op, "region": region, "strength": strength, "params": params or {}}),
                "constraints": [],
            }
            operations.append(payload)

        if round_index == 1 and requested_packages:
            for item in requested_packages[:3]:
                op = str(item.get("op") or "")
                if not op or registry.get(op) is None:
                    continue
                append_operation(
                    op,
                    region=str(item.get("region") or "whole_image"),
                    params=dict(item.get("params") or {}),
                    strength=item.get("strength"),
                )

        if not operations:
            if "underexposed" in issues or _contains_any(request_text, ("提亮", "曝光", "逆光", "背光")):
                append_operation("adjust_exposure", strength=0.22, params={"max_stops": 1.45})
            if {"crushed_shadows", "clipped_highlights"} & issues or _contains_any(request_text, ("高光", "阴影", "层次")):
                append_operation("adjust_highlights_shadows", strength=0.18, params={"tone_amount": 0.3})
            if "flat_contrast" in issues or _contains_any(request_text, ("对比", "层次感")):
                append_operation("adjust_contrast", strength=0.16)
            if round_index == 1 and ("build_summer_mood" in constraints or _contains_any(request_text, ("夏日", "通透", "空气感"))):
                append_operation("adjust_white_balance", strength=0.1)
                append_operation("adjust_vibrance_saturation", strength=0.16)

        if round_index == 2 and not operations:
            if image_analysis.get("domain") == "portrait" or _contains_any(request_text, ("人像", "肤色", "脸", "皮肤")):
                append_operation("point_color", params={"target_color": "skin", "strength": 0.16, "luminance_shift": 0.08})
                append_operation("skin_texture_reduce", strength=0.1)
            elif _contains_any(request_text, ("质感", "通透", "清晰")):
                append_operation("adjust_clarity", strength=0.08)
                append_operation("adjust_texture", strength=0.08)

        return operations[:3]

    stage_name = f"plan_execute_round_{round_index}"
    round_key = f"round_{round_index}"
    traced = append_fallback_trace(
        fallback_trace,
        stage=stage_name,
        source="planner_tool_model",
        location="round_execution",
        strategy="rule_plan_execution",
        message="实时规划不可用，改用规则规划并继续执行。",
        error=reason,
    )
    input_images = state.get("input_images") or []
    if not input_images:
        raise ValueError("No input image available.")

    current_image = str(state.get("selected_output") or input_images[0]) if round_index > 1 else input_images[0]
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = list(state.get("execution_trace") or [])
    segmentation_trace = list(state.get("segmentation_trace") or [])
    round_execution_trace: list[dict[str, Any]] = []
    round_segmentation_trace: list[dict[str, Any]] = []
    round_outputs = dict(state.get("round_outputs") or {})
    round_plans = dict(state.get("round_plans") or {})
    round_execution_traces = dict(state.get("round_execution_traces") or {})
    round_segmentation_traces = dict(state.get("round_segmentation_traces") or {})
    mask_cache: dict[str, str] = {}
    writer = _safe_stream_writer()
    round_operations = build_rule_operations()
    latest_result: dict[str, Any] | None = None

    writer(
        {
            "event": "planner_started",
            "stage": stage_name,
            "round": round_key,
            "message": f"实时规划不可用，正在使用规则规划执行 {round_key}",
        }
    )
    for operation in round_operations:
        current_image, latest_result = _execute_single_tool_call(
            state=state,
            node_name=stage_name,
            round_key=round_key,
            current_image=current_image,
            operation=operation,
            execution_trace=execution_trace,
            round_execution_trace=round_execution_trace,
            segmentation_trace=segmentation_trace,
            round_segmentation_trace=round_segmentation_trace,
            fallback_trace=traced,
            candidate_outputs=candidate_outputs,
            mask_cache=mask_cache,
        )

    return _complete_current_round(
        state=state,
        round_index=round_index,
        current_image=current_image,
        candidate_outputs=candidate_outputs,
        execution_trace=execution_trace,
        segmentation_trace=segmentation_trace,
        round_outputs=round_outputs,
        round_plans=round_plans,
        round_execution_traces=round_execution_traces,
        round_segmentation_traces=round_segmentation_traces,
        round_operations=round_operations,
        round_execution_trace=round_execution_trace,
        round_segmentation_trace=round_segmentation_trace,
        fallback_trace=traced,
        planner_summary="实时规划不可用，已使用规则规划完成本轮。",
    )


def _complete_current_round(
    *,
    state: EditState,
    round_index: int,
    current_image: str,
    candidate_outputs: list[str],
    execution_trace: list[dict[str, Any]],
    segmentation_trace: list[dict[str, Any]],
    round_outputs: dict[str, Any],
    round_plans: dict[str, Any],
    round_execution_traces: dict[str, Any],
    round_segmentation_traces: dict[str, Any],
    round_operations: list[dict[str, Any]],
    round_execution_trace: list[dict[str, Any]],
    round_segmentation_trace: list[dict[str, Any]],
    fallback_trace: list[dict[str, Any]],
    planner_summary: str,
) -> dict[str, Any]:
    """Build the common return payload for a completed round."""

    round_key = f"round_{round_index}"
    request_intent = dict(state.get("request_intent") or {"mode": state.get("mode") or "auto"})
    image_analysis = dict(state.get("image_analysis") or {})
    plan = EditPlan(
        mode=str(state.get("mode") or request_intent.get("mode") or "auto"),
        domain=str(image_analysis.get("domain") or "general"),
        executor=_choose_executor(round_operations),
        preserve=list(request_intent.get("constraints", [])),
        operations=[EditOperation.model_validate({**item, "priority": index}) for index, item in enumerate(round_operations)],
        should_write_memory=False,
        memory_candidates=[],
        needs_confirmation=False,
    )

    round_plan_payload = plan.model_dump(mode="json")
    round_plan_payload["planner_summary"] = planner_summary
    round_plans[round_key] = round_plan_payload
    round_outputs[round_key] = current_image
    round_execution_traces[round_key] = round_execution_trace
    round_segmentation_traces[round_key] = round_segmentation_trace

    return {
        "current_round": round_index,
        "selected_output": current_image,
        "candidate_outputs": candidate_outputs,
        "execution_trace": execution_trace,
        "segmentation_trace": segmentation_trace,
        "fallback_trace": fallback_trace,
        "round_outputs": round_outputs,
        "edit_plan": plan.model_dump(mode="json"),
        "round_plans": round_plans,
        "round_execution_traces": round_execution_traces,
        "round_segmentation_traces": round_segmentation_traces,
        "memory_write_candidates": [],
        "approval_required": bool(state.get("approval_required")),
        "approval_payload": state.get("approval_payload"),
        "masks": {},
    }


def _run_round(state: EditState, *, round_index: int) -> dict[str, Any]:
    """Run one realtime planner round with per-step tool execution."""

    if not planner_tool_model_available():
        return _fallback_rule_execute_round(
            state,
            round_index=round_index,
            fallback_trace=list(state.get("fallback_trace") or []),
            reason="Planner tool-calling model is unavailable.",
        )

    input_images = state.get("input_images") or []
    if not input_images:
        raise ValueError("No input image available.")

    node_name = f"plan_execute_round_{round_index}"
    round_key = f"round_{round_index}"
    request_text = str(state.get("request_text") or "")
    request_intent = dict(state.get("request_intent") or {"mode": state.get("mode") or "auto"})
    image_analysis = dict(state.get("image_analysis") or {})
    retrieved_prefs = list(state.get("retrieved_prefs") or [])
    previous_plan = (state.get("round_plans") or {}).get("round_1") if round_index == 2 else None
    previous_execution_trace = (state.get("round_execution_traces") or {}).get("round_1") if round_index == 2 else None
    previous_eval_report = (state.get("round_eval_reports") or {}).get("round_1") if round_index == 2 else None
    planner_thinking_mode = bool(state.get("planner_thinking_mode"))

    current_image = str(state.get("selected_output") or input_images[0]) if round_index > 1 else input_images[0]
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = list(state.get("execution_trace") or [])
    segmentation_trace = list(state.get("segmentation_trace") or [])
    fallback_trace = list(state.get("fallback_trace") or [])
    round_execution_trace: list[dict[str, Any]] = []
    round_segmentation_trace: list[dict[str, Any]] = []
    round_outputs = dict(state.get("round_outputs") or {})
    round_plans = dict(state.get("round_plans") or {})
    round_execution_traces = dict(state.get("round_execution_traces") or {})
    round_segmentation_traces = dict(state.get("round_segmentation_traces") or {})
    mask_cache: dict[str, str] = {}
    writer = _safe_stream_writer()

    writer(
        {
            "event": "round_started",
            "stage": node_name,
            "round": round_key,
            "message": f"开始执行 {round_key}",
        },
    )
    writer(
        {
            "event": "planner_started",
            "stage": node_name,
            "round": round_key,
            "message": f"规划器开始执行 {round_key}",
        },
    )

    round_operations: list[dict[str, Any]] = []
    latest_result: dict[str, Any] | None = None

    step = 1
    while True:
        try:
            message = call_planner_tool_turn(
                request_text=request_text,
                request_intent=request_intent,
                image_analysis=image_analysis,
                retrieved_prefs=retrieved_prefs,
                current_image_path=current_image,
                round_name=round_key,
                current_step=step,
                round_operations=round_operations,
                latest_result=latest_result,
                planner_thinking_mode=planner_thinking_mode,
                previous_plan=previous_plan,
                previous_execution_trace=previous_execution_trace,
                previous_eval_report=previous_eval_report,
            )
            tool_name, arguments = extract_single_tool_call(message)
        except RuntimeError as error:
            if round_operations:
                fallback_trace = append_fallback_trace(
                    fallback_trace,
                    stage=node_name,
                    source="planner_tool_model",
                    location="round_execution",
                    strategy="finish_current_round",
                    message="实时规划中断，保留当前轮已完成结果。",
                    error=str(error),
                )
                return _complete_current_round(
                    state=state,
                    round_index=round_index,
                    current_image=current_image,
                    candidate_outputs=candidate_outputs,
                    execution_trace=execution_trace,
                    segmentation_trace=segmentation_trace,
                    round_outputs=round_outputs,
                    round_plans=round_plans,
                    round_execution_traces=round_execution_traces,
                    round_segmentation_traces=round_segmentation_traces,
                    round_operations=round_operations,
                    round_execution_trace=round_execution_trace,
                    round_segmentation_trace=round_segmentation_trace,
                    fallback_trace=fallback_trace,
                    planner_summary="实时规划中断，已保留当前轮结果。",
                )
            return _fallback_rule_execute_round(
                state,
                round_index=round_index,
                fallback_trace=fallback_trace,
                reason=str(error),
            )

        if tool_name == "finish_round":
            finish_payload = FinishRoundParams.model_validate(arguments)

            writer(
                {
                    "event": "planner_round_finished",
                    "stage": node_name,
                    "round": round_key,
                    "message": f"{round_key} 规划执行完成",
                    "summary": finish_payload.summary,
                },
            )
            writer(
                {
                    "event": "round_completed",
                    "stage": node_name,
                    "round": round_key,
                    "message": f"{round_key} 执行完成",
                },
            )
            completed = _complete_current_round(
                state=state,
                round_index=round_index,
                current_image=current_image,
                candidate_outputs=candidate_outputs,
                execution_trace=execution_trace,
                segmentation_trace=segmentation_trace,
                round_outputs=round_outputs,
                round_plans=round_plans,
                round_execution_traces=round_execution_traces,
                round_segmentation_traces=round_segmentation_traces,
                round_operations=round_operations,
                round_execution_trace=round_execution_trace,
                round_segmentation_trace=round_segmentation_trace,
                fallback_trace=fallback_trace,
                planner_summary=finish_payload.summary,
            )
            completed["memory_write_candidates"] = list(finish_payload.memory_candidates) if finish_payload.should_write_memory else []
            completed["approval_required"] = bool(finish_payload.needs_confirmation)
            completed["approval_payload"] = (
                ApprovalPayload(
                    reason="planner_requested_confirmation",
                    summary=finish_payload.summary,
                    suggested_action="请人工确认当前轮结果是否可接受。",
                    metadata={"round": round_key},
                ).model_dump(mode="json")
                if finish_payload.needs_confirmation
                else state.get("approval_payload")
            )
            return completed

        resolved_tool_name, resolution = resolve_planner_tool_name(tool_name, arguments)
        if resolved_tool_name != tool_name:
            writer(
                {
                    "event": "planner_tool_resolved",
                    "stage": node_name,
                    "round": round_key,
                    "tool_name": tool_name,
                    "resolved_tool_name": resolved_tool_name,
                    "strategy": resolution.get("strategy"),
                    "score": resolution.get("score"),
                    "candidates": resolution.get("candidates"),
                    "message": f"规划器工具名已纠正：{tool_name} -> {resolved_tool_name}",
                },
            )

        operation = build_operation_from_tool_call(resolved_tool_name, arguments)
        current_image, latest_result = _execute_single_tool_call(
            state=state,
            node_name=node_name,
            round_key=round_key,
            current_image=current_image,
            operation=operation,
            execution_trace=execution_trace,
            round_execution_trace=round_execution_trace,
            segmentation_trace=segmentation_trace,
            round_segmentation_trace=round_segmentation_trace,
            fallback_trace=fallback_trace,
            candidate_outputs=candidate_outputs,
            mask_cache=mask_cache,
        )
        round_operations.append(operation)
        step += 1


def plan_execute_round_1(state: EditState) -> dict[str, Any]:
    """Plan and execute the first round in realtime via tool calling."""

    return _run_round(state, round_index=1)


def plan_execute_round_2(state: EditState) -> dict[str, Any]:
    """Plan and execute the second round in realtime via tool calling."""

    return _run_round(state, round_index=2)
