"""Single ToolNode execution with neutral round/candidate tracing."""

from __future__ import annotations

import json
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
from app.tools.common import ToolExecutionResult, WHOLE_IMAGE_REGION


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
            fallback_trace[:] = append_fallback_trace(
                fallback_trace,
                round_id=round_id,
                focus=focus,
                candidate_id=candidate_id,
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

    if tool_spec.requires_mask and not has_local_target:
        _append_execution_skip(
            round_id=round_id,
            focus=focus,
            candidate_id=candidate_id,
            execution_trace=execution_trace,
            current_image=current_image,
            op_name=op_name,
            region=region,
            params=normalized_params,
            error="Skipped: this tool requires a mask.",
        )
        fallback_trace[:] = append_fallback_trace(
            fallback_trace,
            round_id=round_id,
            focus=focus,
            candidate_id=candidate_id,
            source="tool_runtime",
            location=op_name,
            strategy="skip_missing_required_mask",
            message=f"{op_name} requires a mask and no mask was supplied.",
        )
        return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "required mask missing"}, mask_catalog

    writer(
        {
            "event": "tool_started",
            "round": round_id,
            "focus": focus,
            "op": op_name,
            "region": region,
            "message": f"正在执行 {op_name}",
        }
    )

    current_mask_path: str | None = str(direct_mask_path) if direct_mask_path else None
    updated_catalog = mask_catalog
    if requires_mask and not current_mask_path:
        try:
            mask_options = {
                "provider": mask_params.get("mask_provider", "auto"),
                "prompt": mask_params.get("mask_prompt") or region,
                "negative_prompt": mask_params.get("mask_negative_prompt"),
                "semantic_type": bool(mask_params.get("mask_semantic_type", False)),
                "revert_mask": bool(mask_params.get("mask_revert", False)),
            }
        except Exception as error:
            _append_execution_skip(
                round_id=round_id,
                focus=focus,
                candidate_id=candidate_id,
                execution_trace=execution_trace,
                current_image=current_image,
                op_name=op_name,
                region=region,
                params=normalized_params,
                error="Skipped: invalid mask parameters.",
            )
            fallback_trace[:] = append_fallback_trace(
                fallback_trace,
                round_id=round_id,
                focus=focus,
                candidate_id=candidate_id,
                source="tool_runtime",
                location=op_name,
                strategy="skip_invalid_mask_params",
                message="Mask parameters were invalid.",
                error=str(error),
            )
            return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "invalid mask params"}, updated_catalog

        signature_info = normalized_mask_signature(mask_options, region=region)
        signature, signature_payload = signature_info if signature_info is not None else (None, None)
        if signature and signature in updated_catalog.items and updated_catalog.items[signature].mask_path:
            entry = updated_catalog.items[signature]
            if entry.rejected:
                _append_execution_skip(
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    execution_trace=execution_trace,
                    current_image=current_image,
                    op_name=op_name,
                    region=region,
                    params=normalized_params,
                    error="Skipped: cached mask was rejected by quality checks.",
                )
                fallback_trace[:] = append_fallback_trace(
                    fallback_trace,
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    source="mask_quality",
                    location=op_name,
                    strategy="skip_rejected_cached_mask",
                    message="Cached mask was rejected; local step skipped.",
                    error=", ".join(entry.quality_flags),
                )
                return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "rejected cached mask"}, updated_catalog
            current_mask_path = entry.mask_path
            updated_catalog = record_mask_catalog_item(
                updated_catalog,
                signature=signature,
                payload=signature_payload or {},
                focus=focus,
                op_name=op_name,
                region_label=region,
                mask_path=entry.mask_path,
                preview_path=entry.preview_path,
                quality=entry.quality,
            )
        else:
            requested_provider = str(mask_params.get("mask_provider") or "auto")
            requested_target = str(mask_params.get("mask_prompt") or region)
            writer(
                {
                    "event": "segmentation_started",
                    "round": round_id,
                    "focus": focus,
                    "region": region,
                    "provider": requested_provider,
                    "prompt": requested_target,
                    "message": f"正在准备 {requested_target} 的区域遮罩",
                }
            )
            mask_output_dir = str(Path(current_image).resolve().parent / "output" / f"{Path(current_image).stem}_mask")
            try:
                segmentation_result = generate_mask(
                    current_image,
                    region=region,
                    mask_params=mask_params,
                    output_dir=mask_output_dir,
                )
            except Exception as error:
                segmentation_item = SegmentationTraceItem(
                    index=len(segmentation_trace),
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    source_op=op_name,
                    region=region,
                    provider=requested_provider,
                    requested_provider=requested_provider,
                    target_label=requested_target,
                    prompt=str(mask_params.get("mask_prompt") or "") or None,
                    semantic_type=bool(mask_params.get("mask_semantic_type")) if "mask_semantic_type" in mask_params else None,
                    ok=False,
                    fallback_used=True,
                    error=str(error),
                    attempts=list(getattr(error, "attempts", []) or []),
                ).model_dump(mode="json")
                segmentation_trace.append(segmentation_item)
                fallback_trace[:] = append_fallback_trace(
                    fallback_trace,
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    source="segmentation_provider",
                    location=op_name,
                    strategy="skip_local_operation",
                    message="Segmentation did not return a usable mask; local step skipped.",
                    error=str(error),
                )
                _append_execution_skip(
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    execution_trace=execution_trace,
                    current_image=current_image,
                    op_name=op_name,
                    region=region,
                    params=normalized_params,
                    error="Skipped: segmentation returned no usable mask.",
                )
                writer(
                    {
                        "event": "segmentation_skipped",
                        "round": round_id,
                        "focus": focus,
                        "region": region,
                        "provider": requested_provider,
                        "message": f"{requested_target} 未返回可用遮罩，跳过该局部步骤",
                        "error": str(error),
                    }
                )
                return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "segmentation skipped"}, updated_catalog

            current_mask_path = segmentation_result.binary_mask_path
            mask_quality = evaluate_generated_mask(current_mask_path)
            if signature and signature_payload is not None:
                updated_catalog = record_mask_catalog_item(
                    updated_catalog,
                    signature=signature,
                    payload=signature_payload,
                    focus=focus,
                    op_name=op_name,
                    region_label=region,
                    mask_path=segmentation_result.binary_mask_path,
                    preview_path=segmentation_result.segmentation_rgba_path,
                    quality=mask_quality,
                )
            segmentation_item = SegmentationTraceItem(
                index=len(segmentation_trace),
                round_id=round_id,
                focus=focus,
                candidate_id=candidate_id,
                source_op=op_name,
                region=region,
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
            segmentation_trace.append(segmentation_item)
            if mask_quality.rejected:
                fallback_trace[:] = append_fallback_trace(
                    fallback_trace,
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    source="mask_quality",
                    location=op_name,
                    strategy="skip_low_quality_mask",
                    message="Mask quality was too low; local step skipped.",
                    error=", ".join(mask_quality.flags),
                )
                _append_execution_skip(
                    round_id=round_id,
                    focus=focus,
                    candidate_id=candidate_id,
                    execution_trace=execution_trace,
                    current_image=current_image,
                    op_name=op_name,
                    region=region,
                    params=normalized_params,
                    error="Skipped: mask quality rejected.",
                )
                writer(
                    {
                        "event": "segmentation_skipped",
                        "round": round_id,
                        "focus": focus,
                        "region": region,
                        "provider": segmentation_item["provider"],
                        "prompt": segmentation_item["prompt"],
                        "message": f"{segmentation_item['target_label']} 的遮罩质量不足，跳过该局部步骤",
                    }
                )
                return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "mask quality rejected"}, updated_catalog
            writer(
                {
                    "event": "segmentation_finished",
                    "round": round_id,
                    "focus": focus,
                    "region": region,
                    "provider": segmentation_item["provider"],
                    "prompt": segmentation_item["prompt"],
                    "message": f"{segmentation_item['target_label']} 的区域遮罩已生成",
                }
            )

    tool_args = strip_runtime_mask_params(normalized_params)
    tool_args["image_path"] = current_image
    if current_mask_path:
        tool_args["mask_path"] = current_mask_path

    try:
        result = invoke_tool_node(tool_name=op_name, tool_args=tool_args, writer=writer)
    except Exception as error:
        result = ToolExecutionResult(
            ok=False,
            tool=op_name,
            output_image=current_image,
            applied_params={"params": normalized_params},
            fallback_used=True,
            error=str(error),
        )

    trace_item = ExecutionTraceItem(
        index=len(execution_trace),
        round_id=round_id,
        focus=focus,
        candidate_id=candidate_id,
        op=op_name,
        region=region,
        ok=result.ok,
        fallback_used=result.fallback_used,
        error=result.error,
        output_image=result.output_image or current_image,
        applied_params=result.applied_params or {"params": normalized_params},
        mask_path=current_mask_path,
        warnings=list(result.warnings),
        artifacts=dict(result.artifacts),
    ).model_dump(mode="json")
    execution_trace.append(trace_item)

    if result.ok and result.output_image:
        current_image = result.output_image
        candidate_outputs.append(result.output_image)

    writer(
        {
            "event": "tool_finished" if result.ok else "tool_failed",
            "round": round_id,
            "focus": focus,
            "op": op_name,
            "region": region,
            "message": f"{op_name} {'执行完成' if result.ok else '执行失败'}",
            "error": result.error,
        }
    )
    return current_image, {"op": op_name, "ok": result.ok, "fallback_used": result.fallback_used, "error": result.error, "mask_path": current_mask_path}, updated_catalog
