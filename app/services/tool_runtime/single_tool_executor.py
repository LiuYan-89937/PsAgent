"""Single ToolNode execution with neutral round/candidate tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import (
    ExecutionTraceItem,
    FocusKey,
    MaskCatalog,
    SegmentationTraceItem,
)
from app.services.planner_param_codec import (
    extract_runtime_mask_params,
    normalize_runtime_tool_params,
    strip_runtime_mask_params,
)
from app.services.tool_runtime.mask_runtime import (
    evaluate_generated_mask,
    generate_mask,
    normalized_mask_signature,
    record_mask_catalog_item,
)
from app.tools import build_default_tool_node, require_tool_spec
from app.tools.common import ToolExecutionResult, ToolSpec, WHOLE_IMAGE_REGION


def parse_tool_message_payload(payload: Any) -> dict[str, Any]:
    """Parse ToolNode output payload into a JSON object."""

    if isinstance(payload, str):
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise RuntimeError("ToolNode returned non-object JSON payload.")
        return parsed
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        joined = "".join(part if isinstance(part, str) else str(part) for part in payload)
        parsed = json.loads(joined)
        if not isinstance(parsed, dict):
            raise RuntimeError("ToolNode returned non-object JSON payload.")
        return parsed
    raise RuntimeError(f"Unsupported ToolNode payload type: {type(payload).__name__}")


def invoke_tool_node(*, tool_name: str, tool_args: dict[str, Any], writer) -> ToolExecutionResult:
    """Invoke the shared ToolNode for one sequential tool call."""

    tool_node = build_default_tool_node()
    tool_call_id = f"{tool_name}_{abs(hash(json.dumps(tool_args, sort_keys=True, default=str)))}"
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": tool_args,
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )
    result = tool_node.invoke(
        {"messages": [ai_message]},
        config={"configurable": {}},
        runtime=Runtime(context=None, store=None, stream_writer=writer),
    )
    tool_messages = list(result.get("messages") or [])
    if not tool_messages:
        raise RuntimeError("ToolNode returned no ToolMessage.")
    payload = parse_tool_message_payload(tool_messages[-1].content)
    return ToolExecutionResult.model_validate(payload)


def _append_execution_skip(
    *,
    round_id: str | None,
    focus: FocusKey | None,
    candidate_id: str | None,
    execution_trace: list[dict[str, Any]],
    current_image: str,
    op_name: str,
    region: str,
    params: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    """Append a normalized skipped execution trace item."""

    skipped = ExecutionTraceItem(
        index=len(execution_trace),
        round_id=round_id,
        focus=focus,
        candidate_id=candidate_id,
        op=op_name,
        region=region,
        ok=False,
        fallback_used=True,
        error=error,
        output_image=current_image,
        applied_params={"params": dict(params)},
        mask_path=None,
    ).model_dump(mode="json")
    execution_trace.append(skipped)
    return skipped


@dataclass(slots=True)
class ToolCallContext:
    """Mutable trace and event state for one tool call."""

    current_image: str
    execution_trace: list[dict[str, Any]]
    segmentation_trace: list[dict[str, Any]]
    fallback_trace: list[dict[str, Any]]
    candidate_outputs: list[str]
    mask_catalog: MaskCatalog
    writer: Any
    round_id: str | None
    focus: FocusKey | None
    candidate_id: str | None


@dataclass(slots=True)
class PreparedToolCall:
    """Normalized operation metadata used by the runtime."""

    op_name: str
    tool_spec: ToolSpec
    region: str
    normalized_params: dict[str, Any]
    mask_params: dict[str, Any]
    direct_mask_path: str | None
    has_local_target: bool
    requires_mask: bool


@dataclass(slots=True)
class MaskResolution:
    """Result of resolving or generating the mask for a tool call."""

    mask_path: str | None
    mask_catalog: MaskCatalog
    skip_status: dict[str, Any] | None = None


def _prepare_tool_call(ctx: ToolCallContext, operation: dict[str, Any]) -> PreparedToolCall:
    op_name = str(operation["op"])
    tool_spec = require_tool_spec(op_name)
    region = str(operation.get("region") or WHOLE_IMAGE_REGION)
    raw_params = dict(operation.get("params") or {})
    direct_mask_path = raw_params.pop("mask_path", None)
    if operation.get("strength") is not None and raw_params.get("strength") is None:
        raw_params["strength"] = operation["strength"]

    normalized_params = normalize_runtime_tool_params(op_name, raw_params)
    mask_params = extract_runtime_mask_params(normalized_params)
    has_local_target = region != WHOLE_IMAGE_REGION or bool(mask_params) or bool(direct_mask_path)
    requires_mask = bool(tool_spec.supports_mask and (tool_spec.requires_mask or not tool_spec.supports_whole_image or has_local_target))

    if tool_spec.requires_mask and tool_spec.recommended_mask_prompt and not direct_mask_path:
        recommended_prompt = tool_spec.recommended_mask_prompt
        current_prompt = str(mask_params.get("mask_prompt") or "").strip()
        if current_prompt != recommended_prompt:
            normalized_params["mask_prompt"] = recommended_prompt
            mask_params["mask_prompt"] = recommended_prompt
            ctx.fallback_trace[:] = append_fallback_trace(
                ctx.fallback_trace,
                round_id=ctx.round_id,
                focus=ctx.focus,
                candidate_id=ctx.candidate_id,
                source="tool_mask_policy",
                location=op_name,
                strategy="use_recommended_mask_prompt",
                message=f"{op_name} switched to recommended mask prompt: {recommended_prompt}.",
                error=f"planner prompt={current_prompt or 'empty'}",
            )
        if str(mask_params.get("mask_provider") or "").strip() == "aliyun":
            normalized_params["mask_provider"] = "fal_sam3"
            mask_params["mask_provider"] = "fal_sam3"
        if not has_local_target:
            region = recommended_prompt
            has_local_target = True
            requires_mask = True

    return PreparedToolCall(
        op_name=op_name,
        tool_spec=tool_spec,
        region=region,
        normalized_params=normalized_params,
        mask_params=mask_params,
        direct_mask_path=str(direct_mask_path) if direct_mask_path else None,
        has_local_target=has_local_target,
        requires_mask=requires_mask,
    )


def _record_execution_skip(ctx: ToolCallContext, call: PreparedToolCall, *, error: str) -> dict[str, Any]:
    return _append_execution_skip(
        round_id=ctx.round_id,
        focus=ctx.focus,
        candidate_id=ctx.candidate_id,
        execution_trace=ctx.execution_trace,
        current_image=ctx.current_image,
        op_name=call.op_name,
        region=call.region,
        params=call.normalized_params,
        error=error,
    )


def _skip_with_fallback(
    ctx: ToolCallContext,
    call: PreparedToolCall,
    *,
    execution_error: str,
    status_error: str,
    source: str,
    strategy: str,
    message: str,
    fallback_error: str | None = None,
    mask_catalog: MaskCatalog | None = None,
) -> tuple[str, dict[str, Any], MaskCatalog]:
    _record_execution_skip(ctx, call, error=execution_error)
    ctx.fallback_trace[:] = append_fallback_trace(
        ctx.fallback_trace,
        round_id=ctx.round_id,
        focus=ctx.focus,
        candidate_id=ctx.candidate_id,
        source=source,
        location=call.op_name,
        strategy=strategy,
        message=message,
        error=fallback_error,
    )
    return ctx.current_image, {"op": call.op_name, "ok": False, "fallback_used": True, "error": status_error}, mask_catalog or ctx.mask_catalog


def _emit_tool_started(ctx: ToolCallContext, call: PreparedToolCall) -> None:
    ctx.writer(
        {
            "event": "tool_started",
            "round": ctx.round_id,
            "focus": ctx.focus,
            "op": call.op_name,
            "region": call.region,
            "message": f"正在执行 {call.op_name}",
        }
    )


def _mask_options(call: PreparedToolCall) -> dict[str, Any]:
    return {
        "provider": call.mask_params.get("mask_provider", "auto"),
        "prompt": call.mask_params.get("mask_prompt") or call.region,
        "negative_prompt": call.mask_params.get("mask_negative_prompt"),
        "semantic_type": bool(call.mask_params.get("mask_semantic_type", False)),
        "revert_mask": bool(call.mask_params.get("mask_revert", False)),
    }


def _resolve_cached_mask(
    ctx: ToolCallContext,
    call: PreparedToolCall,
    *,
    signature: str,
    signature_payload: dict[str, Any] | None,
    mask_catalog: MaskCatalog,
) -> MaskResolution | None:
    if signature not in mask_catalog.items or not mask_catalog.items[signature].mask_path:
        return None

    entry = mask_catalog.items[signature]
    if entry.rejected:
        _record_execution_skip(ctx, call, error="Skipped: cached mask was rejected by quality checks.")
        ctx.fallback_trace[:] = append_fallback_trace(
            ctx.fallback_trace,
            round_id=ctx.round_id,
            focus=ctx.focus,
            candidate_id=ctx.candidate_id,
            source="mask_quality",
            location=call.op_name,
            strategy="skip_rejected_cached_mask",
            message="Cached mask was rejected; local step skipped.",
            error=", ".join(entry.quality_flags),
        )
        return MaskResolution(
            mask_path=None,
            mask_catalog=mask_catalog,
            skip_status={"op": call.op_name, "ok": False, "fallback_used": True, "error": "rejected cached mask"},
        )

    updated_catalog = record_mask_catalog_item(
        mask_catalog,
        signature=signature,
        payload=signature_payload or {},
        focus=ctx.focus,
        op_name=call.op_name,
        region_label=call.region,
        mask_path=entry.mask_path,
        preview_path=entry.preview_path,
        quality=entry.quality,
    )
    return MaskResolution(mask_path=entry.mask_path, mask_catalog=updated_catalog)


def _record_segmentation_failure(
    ctx: ToolCallContext,
    call: PreparedToolCall,
    *,
    requested_provider: str,
    requested_target: str,
    error: Exception,
) -> None:
    segmentation_item = SegmentationTraceItem(
        index=len(ctx.segmentation_trace),
        round_id=ctx.round_id,
        focus=ctx.focus,
        candidate_id=ctx.candidate_id,
        source_op=call.op_name,
        region=call.region,
        provider=requested_provider,
        requested_provider=requested_provider,
        target_label=requested_target,
        prompt=str(call.mask_params.get("mask_prompt") or "") or None,
        semantic_type=bool(call.mask_params.get("mask_semantic_type")) if "mask_semantic_type" in call.mask_params else None,
        ok=False,
        fallback_used=True,
        error=str(error),
        attempts=list(getattr(error, "attempts", []) or []),
    ).model_dump(mode="json")
    ctx.segmentation_trace.append(segmentation_item)


def _generate_runtime_mask(
    ctx: ToolCallContext,
    call: PreparedToolCall,
    *,
    signature: str | None,
    signature_payload: dict[str, Any] | None,
    mask_catalog: MaskCatalog,
) -> MaskResolution:
    requested_provider = str(call.mask_params.get("mask_provider") or "auto")
    requested_target = str(call.mask_params.get("mask_prompt") or call.region)
    ctx.writer(
        {
            "event": "segmentation_started",
            "round": ctx.round_id,
            "focus": ctx.focus,
            "region": call.region,
            "provider": requested_provider,
            "prompt": requested_target,
            "message": f"正在准备 {requested_target} 的区域遮罩",
        }
    )
    mask_output_dir = str(Path(ctx.current_image).resolve().parent / "output" / f"{Path(ctx.current_image).stem}_mask")
    try:
        segmentation_result = generate_mask(
            ctx.current_image,
            region=call.region,
            mask_params=call.mask_params,
            output_dir=mask_output_dir,
        )
    except Exception as error:
        _record_segmentation_failure(
            ctx,
            call,
            requested_provider=requested_provider,
            requested_target=requested_target,
            error=error,
        )
        ctx.fallback_trace[:] = append_fallback_trace(
            ctx.fallback_trace,
            round_id=ctx.round_id,
            focus=ctx.focus,
            candidate_id=ctx.candidate_id,
            source="segmentation_provider",
            location=call.op_name,
            strategy="skip_local_operation",
            message="Segmentation did not return a usable mask; local step skipped.",
            error=str(error),
        )
        _record_execution_skip(ctx, call, error="Skipped: segmentation returned no usable mask.")
        ctx.writer(
            {
                "event": "segmentation_skipped",
                "round": ctx.round_id,
                "focus": ctx.focus,
                "region": call.region,
                "provider": requested_provider,
                "message": f"{requested_target} 未返回可用遮罩，跳过该局部步骤",
                "error": str(error),
            }
        )
        return MaskResolution(
            mask_path=None,
            mask_catalog=mask_catalog,
            skip_status={"op": call.op_name, "ok": False, "fallback_used": True, "error": "segmentation skipped"},
        )

    mask_quality = evaluate_generated_mask(segmentation_result.binary_mask_path)
    if signature and signature_payload is not None:
        mask_catalog = record_mask_catalog_item(
            mask_catalog,
            signature=signature,
            payload=signature_payload,
            focus=ctx.focus,
            op_name=call.op_name,
            region_label=call.region,
            mask_path=segmentation_result.binary_mask_path,
            preview_path=segmentation_result.segmentation_rgba_path,
            quality=mask_quality,
        )
    segmentation_item = SegmentationTraceItem(
        index=len(ctx.segmentation_trace),
        round_id=ctx.round_id,
        focus=ctx.focus,
        candidate_id=ctx.candidate_id,
        source_op=call.op_name,
        region=call.region,
        provider=segmentation_result.provider,
        requested_provider=segmentation_result.requested_provider or requested_provider,
        target_label=segmentation_result.target_label or requested_target,
        prompt=segmentation_result.prompt,
        negative_prompt=segmentation_result.negative_prompt,
        semantic_type=segmentation_result.semantic_type,
        ok=not mask_quality.rejected,
        fallback_used=segmentation_result.fallback_used or mask_quality.rejected,
        error="Rejected: low quality mask." if mask_quality.rejected else None,
        mask_path=segmentation_result.binary_mask_path,
        preview_path=segmentation_result.segmentation_rgba_path,
        request_id=segmentation_result.request_id,
        api_chain=list(segmentation_result.api_chain),
        attempt_index=segmentation_result.attempt_index,
        attempt_strategy=segmentation_result.attempt_strategy,
        requested_prompt=segmentation_result.requested_prompt,
        effective_prompt=segmentation_result.effective_prompt,
        revert_mask=segmentation_result.revert_mask,
        attempts=list(segmentation_result.attempts),
        quality_score=mask_quality.score,
        quality_flags=list(mask_quality.flags),
        rejected=mask_quality.rejected,
    ).model_dump(mode="json")
    ctx.segmentation_trace.append(segmentation_item)

    if mask_quality.rejected:
        ctx.fallback_trace[:] = append_fallback_trace(
            ctx.fallback_trace,
            round_id=ctx.round_id,
            focus=ctx.focus,
            candidate_id=ctx.candidate_id,
            source="mask_quality",
            location=call.op_name,
            strategy="skip_low_quality_mask",
            message="Mask quality was too low; local step skipped.",
            error=", ".join(mask_quality.flags),
        )
        _record_execution_skip(ctx, call, error="Skipped: mask quality rejected.")
        ctx.writer(
            {
                "event": "segmentation_skipped",
                "round": ctx.round_id,
                "focus": ctx.focus,
                "region": call.region,
                "provider": segmentation_item["provider"],
                "prompt": segmentation_item["prompt"],
                "message": f"{segmentation_item['target_label']} 的遮罩质量不足，跳过该局部步骤",
            }
        )
        return MaskResolution(
            mask_path=None,
            mask_catalog=mask_catalog,
            skip_status={"op": call.op_name, "ok": False, "fallback_used": True, "error": "mask quality rejected"},
        )

    ctx.writer(
        {
            "event": "segmentation_finished",
            "round": ctx.round_id,
            "focus": ctx.focus,
            "region": call.region,
            "provider": segmentation_item["provider"],
            "prompt": segmentation_item["prompt"],
            "message": f"{segmentation_item['target_label']} 的区域遮罩已生成",
        }
    )
    return MaskResolution(mask_path=segmentation_result.binary_mask_path, mask_catalog=mask_catalog)


def _resolve_mask(ctx: ToolCallContext, call: PreparedToolCall) -> MaskResolution:
    if call.direct_mask_path or not call.requires_mask:
        return MaskResolution(mask_path=call.direct_mask_path, mask_catalog=ctx.mask_catalog)

    try:
        mask_options = _mask_options(call)
    except Exception as error:
        _, status, mask_catalog = _skip_with_fallback(
            ctx,
            call,
            execution_error="Skipped: invalid mask parameters.",
            status_error="invalid mask params",
            source="tool_runtime",
            strategy="skip_invalid_mask_params",
            message="Mask parameters were invalid.",
            fallback_error=str(error),
        )
        return MaskResolution(mask_path=None, mask_catalog=mask_catalog, skip_status=status)

    signature_info = normalized_mask_signature(mask_options, region=call.region)
    signature, signature_payload = signature_info if signature_info is not None else (None, None)
    if signature:
        cached = _resolve_cached_mask(
            ctx,
            call,
            signature=signature,
            signature_payload=signature_payload,
            mask_catalog=ctx.mask_catalog,
        )
        if cached is not None:
            return cached
    return _generate_runtime_mask(
        ctx,
        call,
        signature=signature,
        signature_payload=signature_payload,
        mask_catalog=ctx.mask_catalog,
    )


def _invoke_and_record_tool(ctx: ToolCallContext, call: PreparedToolCall, *, mask_path: str | None) -> tuple[str, dict[str, Any], MaskCatalog]:
    tool_args = strip_runtime_mask_params(call.normalized_params)
    tool_args["image_path"] = ctx.current_image
    if mask_path:
        tool_args["mask_path"] = mask_path

    try:
        result = invoke_tool_node(tool_name=call.op_name, tool_args=tool_args, writer=ctx.writer)
    except Exception as error:
        result = ToolExecutionResult(
            ok=False,
            tool=call.op_name,
            output_image=ctx.current_image,
            applied_params={"params": call.normalized_params},
            fallback_used=True,
            error=str(error),
        )

    trace_item = ExecutionTraceItem(
        index=len(ctx.execution_trace),
        round_id=ctx.round_id,
        focus=ctx.focus,
        candidate_id=ctx.candidate_id,
        op=call.op_name,
        region=call.region,
        ok=result.ok,
        fallback_used=result.fallback_used,
        error=result.error,
        output_image=result.output_image or ctx.current_image,
        applied_params=result.applied_params or {"params": call.normalized_params},
        mask_path=mask_path,
        warnings=list(result.warnings),
        artifacts=dict(result.artifacts),
    ).model_dump(mode="json")
    ctx.execution_trace.append(trace_item)

    output_image = ctx.current_image
    if result.ok and result.output_image:
        output_image = result.output_image
        ctx.candidate_outputs.append(result.output_image)

    ctx.writer(
        {
            "event": "tool_finished" if result.ok else "tool_failed",
            "round": ctx.round_id,
            "focus": ctx.focus,
            "op": call.op_name,
            "region": call.region,
            "message": f"{call.op_name} {'执行完成' if result.ok else '执行失败'}",
            "error": result.error,
        }
    )
    return output_image, {"op": call.op_name, "ok": result.ok, "fallback_used": result.fallback_used, "error": result.error, "mask_path": mask_path}, ctx.mask_catalog


def execute_single_tool_call(
    *,
    current_image: str,
    operation: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    segmentation_trace: list[dict[str, Any]],
    fallback_trace: list[dict[str, Any]],
    candidate_outputs: list[str],
    mask_catalog: MaskCatalog,
    writer,
    round_id: str | None = None,
    focus: FocusKey | None = None,
    candidate_id: str | None = None,
    mode: str = "explicit",
) -> tuple[str, dict[str, Any], MaskCatalog]:
    """Execute one tool call against the current image."""

    ctx = ToolCallContext(
        current_image=current_image,
        execution_trace=execution_trace,
        segmentation_trace=segmentation_trace,
        fallback_trace=fallback_trace,
        candidate_outputs=candidate_outputs,
        mask_catalog=mask_catalog,
        writer=writer,
        round_id=round_id,
        focus=focus,
        candidate_id=candidate_id,
    )
    call = _prepare_tool_call(ctx, operation)

    if call.tool_spec.requires_mask and not call.has_local_target:
        return _skip_with_fallback(
            ctx,
            call,
            execution_error="Skipped: this tool requires a mask.",
            status_error="required mask missing",
            source="tool_runtime",
            strategy="skip_missing_required_mask",
            message=f"{call.op_name} requires a mask and no mask was supplied.",
        )

    _emit_tool_started(ctx, call)
    mask_resolution = _resolve_mask(ctx, call)
    if mask_resolution.skip_status is not None:
        return current_image, mask_resolution.skip_status, mask_resolution.mask_catalog
    ctx.mask_catalog = mask_resolution.mask_catalog
    return _invoke_and_record_tool(ctx, call, mask_path=mask_resolution.mask_path)
