"""Execution helpers for stage pipeline tool calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import (
    ExecutionTraceItem,
    MaskCatalog,
    SegmentationTraceItem,
    StageKey,
    StagePolicy,
)
from app.tools import build_default_tool_node, require_tool_spec
from app.tools.common import MaskParams, ToolExecutionResult, WHOLE_IMAGE_REGION
from app.tools.segmentation_tools import normalize_segmentation_prompt_label


def normalized_mask_signature(mask_options: dict[str, Any], *, region: str) -> tuple[str, dict[str, Any]] | None:
    """Build a reusable mask signature independent of free-form region labels."""

    # region 可以是中文自由文本，所以真正用于复用判定的关键
    # 不能直接拿原始 region，而要落到规范化后的 prompt + provider 组合。
    prompt_source = str(mask_options.get("prompt") or region or "").strip()
    if not prompt_source:
        return None
    normalized_prompt = normalize_segmentation_prompt_label(prompt_source, region=region)
    signature_payload = {
        "provider": str(mask_options.get("provider") or "auto"),
        "normalized_mask_prompt": normalized_prompt,
        "semantic_type": bool(mask_options.get("semantic_type", False)),
        "revert_mask": bool(mask_options.get("revert_mask", False)),
    }
    signature = json.dumps(signature_payload, sort_keys=True, ensure_ascii=True)
    return signature, signature_payload


def record_mask_catalog_item(
    mask_catalog: MaskCatalog,
    *,
    signature: str,
    payload: dict[str, Any],
    stage_key: StageKey,
    op_name: str,
    region_label: str,
    mask_path: str | None,
    preview_path: str | None,
) -> MaskCatalog:
    """Insert or update a reusable mask entry."""

    # MaskCatalog 是跨阶段共享缓存。
    # 同一个 signature 命中时只增加复用计数，不重新生成遮罩。
    items = dict(mask_catalog.items)
    existing = items.get(signature)
    if existing is None:
        from app.graph.state import MaskCatalogItem

        items[signature] = MaskCatalogItem(
            signature=signature,
            provider=payload["provider"],
            mask_prompt=payload["normalized_mask_prompt"],
            normalized_mask_prompt=payload["normalized_mask_prompt"],
            semantic_type=bool(payload["semantic_type"]),
            revert_mask=bool(payload["revert_mask"]),
            mask_path=mask_path,
            preview_path=preview_path,
            source_stage=stage_key,
            source_op=op_name,
            region_labels=[region_label],
            reuse_count=0,
        )
    else:
        updated = existing.model_copy(deep=True)
        if region_label not in updated.region_labels:
            updated.region_labels.append(region_label)
        updated.reuse_count += 1
        if not updated.mask_path and mask_path:
            updated.mask_path = mask_path
        if not updated.preview_path and preview_path:
            updated.preview_path = preview_path
        items[signature] = updated
    return MaskCatalog(items=items)


def append_execution_skip(
    *,
    stage_key: StageKey,
    execution_trace: list[dict[str, Any]],
    stage_execution_trace: list[dict[str, Any]],
    current_image: str,
    op_name: str,
    region: str,
    params: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    """Append a normalized skipped execution trace item."""

    # “跳过”也要落 execution_trace，
    # 这样前端和评估层才能看到这一步为什么没有真正执行。
    skipped = ExecutionTraceItem(
        index=len(execution_trace),
        stage=stage_key,
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
    stage_execution_trace.append(skipped)
    return skipped


def parse_tool_message_payload(payload: Any) -> dict[str, Any]:
    """Parse ToolNode output payload into a JSON object."""

    # ToolNode 返回的是 ToolMessage.content。
    # 这里统一把 str / dict / list 三种常见形态都收敛成 dict。
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


def invoke_tool_node(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    writer,
) -> ToolExecutionResult:
    """Invoke the shared ToolNode for one sequential tool call."""

    # 虽然 ToolNode 支持批量 tool call，
    # 但当前阶段链路要求“上一步输出图作为下一步输入图”，
    # 所以这里固定按单 step 顺序调用。
    tool_node = build_default_tool_node()
    tool_call_id = f"{tool_name}_{abs(hash(json.dumps(tool_args, sort_keys=True, default=str)))}"
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                # 这里只构造一条临时 AIMessage，
                # 把当前 step 翻译成一个标准 tool_call 给 ToolNode。
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
    # ToolNode 最终吐回 ToolMessage，我们再统一校验为 ToolExecutionResult。
    payload = parse_tool_message_payload(tool_messages[-1].content)
    return ToolExecutionResult.model_validate(payload)


def execute_single_tool_call(
    *,
    stage_key: StageKey,
    stage_policy: StagePolicy,
    current_image: str,
    operation: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    stage_execution_trace: list[dict[str, Any]],
    segmentation_trace: list[dict[str, Any]],
    stage_segmentation_trace: list[dict[str, Any]],
    fallback_trace: list[dict[str, Any]],
    candidate_outputs: list[str],
    mask_catalog: MaskCatalog,
    normalize_runtime_tool_params: Callable[[str, dict[str, Any] | None], dict[str, Any]],
    extract_runtime_mask_params: Callable[[dict[str, Any]], dict[str, Any]],
    strip_runtime_mask_params: Callable[[dict[str, Any]], dict[str, Any]],
    segmentation_resolver: Callable[..., Any],
    writer,
) -> tuple[str, dict[str, Any], MaskCatalog]:
    """Execute one tool call against the current image under a stage policy."""

    op_name = str(operation["op"])
    # 第一步先做阶段白名单校验，保证 planner 就算漂移也不会执行越权工具。
    if op_name not in set(stage_policy.visible_tools):
        fallback_trace[:] = append_fallback_trace(
            fallback_trace,
            stage=stage_key,
            source="stage_policy",
            location=op_name,
            strategy="skip_disallowed_tool",
            message=f"{op_name} 不在 {stage_key} 阶段白名单内，已跳过。",
            error=None,
        )
        append_execution_skip(
            stage_key=stage_key,
            execution_trace=execution_trace,
            stage_execution_trace=stage_execution_trace,
            current_image=current_image,
            op_name=op_name,
            region=str(operation.get("region") or WHOLE_IMAGE_REGION),
            params=dict(operation.get("params") or {}),
            error="Skipped: disallowed tool.",
        )
        return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "disallowed tool"}, mask_catalog

    tool_spec = require_tool_spec(op_name)
    region = str(operation.get("region") or WHOLE_IMAGE_REGION)
    params = dict(operation.get("params") or {})
    if operation.get("strength") is not None and params.get("strength") is None:
        # 兼容保留字段：如果上层把 strength 放在 step 顶层，这里并回 params。
        params["strength"] = operation["strength"]
    # 统一做默认值补齐，并忽略显式 None，避免 planner 把默认值“冲掉”。
    normalized_params = normalize_runtime_tool_params(op_name, params)
    mask_params = extract_runtime_mask_params(normalized_params)
    has_local_target = region != WHOLE_IMAGE_REGION or bool(mask_params)
    requires_mask = tool_spec.supports_mask and (
        tool_spec.requires_mask or not tool_spec.supports_whole_image or has_local_target
    )

    if tool_spec.requires_mask and not has_local_target:
        append_execution_skip(
            stage_key=stage_key,
            execution_trace=execution_trace,
            stage_execution_trace=stage_execution_trace,
            current_image=current_image,
            op_name=op_name,
            region=region,
            params=normalized_params,
            error="Skipped: this tool requires a mask.",
        )
        fallback_trace[:] = append_fallback_trace(
            fallback_trace,
            stage=stage_key,
            source="stage_runner",
            location=op_name,
            strategy="skip_missing_required_mask",
            message=f"{op_name} 是局部专属工具，缺少必需遮罩，已跳过。",
            error=None,
        )
        return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "required mask missing"}, mask_catalog

    if requires_mask and not stage_policy.mask_allowed:
        # 阶段 policy 不允许 mask 时，不报硬错，走可解释的 skip/fallback。
        append_execution_skip(
            stage_key=stage_key,
            execution_trace=execution_trace,
            stage_execution_trace=stage_execution_trace,
            current_image=current_image,
            op_name=op_name,
            region=region,
            params=normalized_params,
            error="Skipped: masks are not allowed in this stage.",
        )
        fallback_trace[:] = append_fallback_trace(
            fallback_trace,
            stage=stage_key,
            source="stage_policy",
            location=op_name,
            strategy="skip_masked_step",
            message=f"{op_name} 在当前阶段不允许使用局部遮罩，已跳过。",
            error=None,
        )
        return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "mask disallowed"}, mask_catalog

    writer(
        {
            "event": "package_started",
            "stage": stage_key,
            "op": op_name,
            "region": region,
            "message": f"正在执行 {op_name}",
        },
    )

    current_mask_path: str | None = None
    updated_catalog = mask_catalog
    if requires_mask:
        try:
            # 先用公共 mask contract 校验参数，再转成分割 provider 的 runtime kwargs。
            mask_options = MaskParams.model_validate(mask_params).to_runtime_options() if mask_params else {}
        except Exception as error:
            append_execution_skip(
                stage_key=stage_key,
                execution_trace=execution_trace,
                stage_execution_trace=stage_execution_trace,
                current_image=current_image,
                op_name=op_name,
                region=region,
                params=normalized_params,
                error="Skipped: invalid mask parameters.",
            )
            fallback_trace[:] = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="stage_runner",
                location=op_name,
                strategy="skip_invalid_mask_params",
                message="局部步骤的 mask 参数无效，已跳过。",
                error=str(error),
            )
            return current_image, {"op": op_name, "ok": False, "fallback_used": True, "error": "invalid mask params"}, updated_catalog

        signature_info = normalized_mask_signature(mask_options, region=region)
        signature, signature_payload = signature_info if signature_info is not None else (None, None)
        if signature and signature in updated_catalog.items and updated_catalog.items[signature].mask_path:
            # 命中缓存时直接复用旧 mask，避免重复分割和重复计费。
            entry = updated_catalog.items[signature]
            current_mask_path = entry.mask_path
            updated_catalog = record_mask_catalog_item(
                updated_catalog,
                signature=signature,
                payload=signature_payload or {},
                stage_key=stage_key,
                op_name=op_name,
                region_label=region,
                mask_path=entry.mask_path,
                preview_path=entry.preview_path,
            )
        else:
            # 没命中缓存时，才真正发起一次新的分割请求。
            requested_provider = str(mask_options.get("provider") or "auto")
            requested_target = str(mask_options.get("prompt") or region)
            writer(
                {
                    "event": "segmentation_started",
                    "stage": stage_key,
                    "region": region,
                    "provider": requested_provider,
                    "prompt": mask_options.get("prompt"),
                    "message": f"正在准备 {requested_target} 的区域遮罩",
                },
            )
            mask_output_dir = str(
                Path(current_image).resolve().parent / "output" / f"{Path(current_image).stem}_{stage_key}_mask"
            )
            try:
                segmentation_result = segmentation_resolver(
                    current_image,
                    region,
                    output_dir=mask_output_dir,
                    **mask_options,
                )
            except Exception as error:
                # 局部分割失败时，这一步整体降级为“跳过”，
                # 不让错误直接炸穿整条 stage pipeline。
                segmentation_item = SegmentationTraceItem(
                    index=len(segmentation_trace),
                    stage=stage_key,
                    source_op=op_name,
                    region=region,
                    provider=requested_provider,
                    requested_provider=requested_provider,
                    target_label=requested_target,
                    prompt=str(mask_options.get("prompt") or "") or None,
                    semantic_type=bool(mask_options.get("semantic_type")) if "semantic_type" in mask_options else None,
                    ok=False,
                    fallback_used=True,
                    error=str(error),
                    mask_path=None,
                    preview_path=None,
                    api_chain=[],
                    attempts=list(getattr(error, "attempts", []) or []),
                ).model_dump(mode="json")
                segmentation_trace.append(segmentation_item)
                stage_segmentation_trace.append(segmentation_item)
                fallback_trace[:] = append_fallback_trace(
                    fallback_trace,
                    stage=stage_key,
                    source="segmentation_provider",
                    location=op_name,
                    strategy="skip_local_operation",
                    message="局部分割未返回可用遮罩，已跳过该局部步骤。",
                    error=str(error),
                )
                append_execution_skip(
                    stage_key=stage_key,
                    execution_trace=execution_trace,
                    stage_execution_trace=stage_execution_trace,
                    current_image=current_image,
                    op_name=op_name,
                    region=region,
                    params=normalized_params,
                    error="Skipped: segmentation returned no usable mask.",
                )
                writer(
                    {
                        "event": "segmentation_skipped",
                        "stage": stage_key,
                        "region": region,
                        "provider": requested_provider,
                        "message": f"{requested_target} 未返回可用遮罩，跳过该局部步骤",
                        "error": str(error),
                    },
                )
                return current_image, {
                    "op": op_name,
                    "ok": False,
                    "fallback_used": True,
                    "error": "segmentation skipped",
                    "segmentation_skipped": True,
                }, updated_catalog

            current_mask_path = segmentation_result.binary_mask_path
            if signature and signature_payload is not None:
                # 新分割成功后立刻写入 MaskCatalog，供后续阶段或后续步骤复用。
                updated_catalog = record_mask_catalog_item(
                    updated_catalog,
                    signature=signature,
                    payload=signature_payload,
                    stage_key=stage_key,
                    op_name=op_name,
                    region_label=region,
                    mask_path=segmentation_result.binary_mask_path,
                    preview_path=segmentation_result.segmentation_rgba_path,
                )
            segmentation_item = SegmentationTraceItem(
                index=len(segmentation_trace),
                stage=stage_key,
                source_op=op_name,
                region=region,
                provider=segmentation_result.provider,
                requested_provider=segmentation_result.requested_provider or requested_provider,
                target_label=segmentation_result.target_label or requested_target,
                prompt=segmentation_result.prompt,
                negative_prompt=segmentation_result.negative_prompt,
                semantic_type=segmentation_result.semantic_type,
                ok=True,
                fallback_used=segmentation_result.fallback_used,
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
            ).model_dump(mode="json")
            segmentation_trace.append(segmentation_item)
            stage_segmentation_trace.append(segmentation_item)
            writer(
                {
                    "event": "segmentation_finished",
                    "stage": stage_key,
                    "region": region,
                    "provider": segmentation_item["provider"],
                    "prompt": segmentation_item["prompt"],
                    "message": f"{segmentation_item['target_label']} 的区域遮罩已生成",
                },
            )

    tool_args = strip_runtime_mask_params(normalized_params)
    # image_path / mask_path 属于 runtime-only 参数，不会出现在 planner schema，
    # 但在真正调用工具时一定要由 stage 层显式注入。
    tool_args["image_path"] = current_image
    if current_mask_path:
        tool_args["mask_path"] = current_mask_path

    try:
        # 这里才真正进入 ToolNode，前面的流程都只是“为 ToolNode 准备调用条件”。
        result = invoke_tool_node(
            tool_name=op_name,
            tool_args=tool_args,
            writer=writer,
        )
    except Exception as error:
        # ToolNode 或工具本身出错时，收敛成统一结果对象，避免上层到处 try/except。
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
        stage=stage_key,
        op=op_name,
        region=region,
        ok=result.ok,
        fallback_used=result.fallback_used,
        error=result.error,
        output_image=result.output_image or current_image,
        applied_params=result.applied_params or {"params": normalized_params},
        mask_path=current_mask_path,
    ).model_dump(mode="json")
    execution_trace.append(trace_item)
    stage_execution_trace.append(trace_item)

    if result.ok and result.output_image:
        # 成功时更新当前工作图，保证下一步串行接着处理最新结果。
        current_image = result.output_image
        candidate_outputs.append(result.output_image)

    writer(
        {
            "event": "package_finished" if result.ok else "package_failed",
            "stage": stage_key,
            "op": op_name,
            "region": region,
            "message": f"{op_name} {'执行完成' if result.ok else '执行失败'}",
            "error": result.error,
        },
    )
    return current_image, {
        "op": op_name,
        "ok": result.ok,
        "fallback_used": result.fallback_used,
        "error": result.error,
        "mask_path": current_mask_path,
    }, updated_catalog
