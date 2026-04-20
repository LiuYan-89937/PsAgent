"""Shared stage-pipeline nodes and execution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from PIL import Image

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import (
    ApprovalPayload,
    EditOperation,
    EditPlan,
    EditProfile,
    EditState,
    EvaluationReport,
    ExecutionTraceItem,
    MaskCatalog,
    PhaseArtifact,
    PhaseOutputArtifact,
    PlannerExecutionPlan,
    PlannerExecutionStep,
    SegmentationTraceItem,
    StageContextEnvelope,
    StageKey,
    StagePolicy,
    StageSummary,
    coerce_edit_profile,
    coerce_execution_trace,
    coerce_image_analysis,
    coerce_mask_catalog,
    coerce_phase_artifacts,
    coerce_request_intent,
    coerce_segmentation_trace,
    coerce_stage_context,
    coerce_stage_policy,
)
from app.services.model_context import compact_execution_trace_for_model
from app.services.planner_param_codec import (
    extract_runtime_mask_params,
    normalize_runtime_tool_params,
    strip_runtime_mask_params,
)
from app.services.planner_execution_model import (
    generate_stage_execution_plan_with_qwen,
    planner_execution_model_available,
)
from app.services.stage_context import build_stage_context, summarize_edit_profile_for_model
from app.services.stage_policy import STAGE_LABELS, STAGE_ORDER, resolve_stage_policy, summarize_policy_constraints
from app.tools.segmentation_tools import normalize_segmentation_prompt_label, resolve_region_mask
from app.tools.tool_registry import build_default_tool_node, build_default_tool_registry
from app.tools.tool_specs import MaskParams, ToolExecutionResult, WHOLE_IMAGE_REGION


TONE_STACK_OPS = {
    "adjust_exposure",
    "adjust_contrast",
    "adjust_vibrance_saturation",
}


def _safe_stream_writer():
    """Return a stream writer when inside LangGraph runtime, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def _current_image_path(state: EditState) -> str:
    """Return the current working image path."""

    input_images = state.get("input_images") or []
    if not input_images and not state.get("selected_output"):
        raise ValueError("No input image available.")
    return str(state.get("selected_output") or input_images[0])


def _compute_image_metrics(image_path: str, *, mask_path: str | None = None) -> dict[str, float]:
    """Compute lightweight brightness and clipping statistics for stage guards."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])

    if mask_path:
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        selector = mask > 0
        if selector.any():
            gray = gray[selector]
            rgb = image_np[selector]
        else:
            rgb = image_np.reshape(-1, 3)
    else:
        rgb = image_np.reshape(-1, 3)

    if gray.size == 0:
        return {
            "brightness_mean": 0.0,
            "shadow_ratio": 0.0,
            "highlight_ratio": 0.0,
            "saturation_mean": 0.0,
        }

    max_rgb = rgb.max(axis=1)
    min_rgb = rgb.min(axis=1)
    saturation = np.where(max_rgb == 0, 0.0, (max_rgb - min_rgb) / np.maximum(max_rgb, 1.0))
    return {
        "brightness_mean": float(gray.mean()),
        "shadow_ratio": float((gray < 28).mean()),
        "highlight_ratio": float((gray > 235).mean()),
        "saturation_mean": float(saturation.mean()),
    }


def _base_phase_artifact(stage_key: StageKey) -> PhaseArtifact:
    """Return a default phase artifact for a stage."""

    return PhaseArtifact(key=stage_key, label=STAGE_LABELS[stage_key])


def _merge_stage_plans_into_edit_plan(phases: dict[str, PhaseArtifact], *, mode: str, domain: str) -> EditPlan:
    """Aggregate all stage plans into one top-level display plan."""

    operations: list[EditOperation] = []
    preserve: list[str] = []
    memory_candidates: list[dict[str, Any]] = []
    should_write_memory = False
    needs_confirmation = False
    next_priority = 0

    for stage_key in STAGE_ORDER:
        phase = phases.get(stage_key)
        if phase is None or phase.plan is None:
            continue
        plan = phase.plan
        for item in plan.preserve:
            if item not in preserve:
                preserve.append(item)
        for candidate in plan.memory_candidates:
            if candidate not in memory_candidates:
                memory_candidates.append(candidate)
        should_write_memory = should_write_memory or bool(plan.should_write_memory)
        needs_confirmation = needs_confirmation or bool(plan.needs_confirmation)
        for step in plan.steps:
            operations.append(
                EditOperation(
                    op=step.op,
                    region=step.region,
                    strength=step.strength,
                    params=dict(step.params),
                    constraints=list(step.constraints),
                    priority=next_priority,
                )
            )
            next_priority += 1

    return EditPlan(
        mode=mode if mode in {"explicit", "auto"} else "auto",
        domain=domain if domain in {"portrait", "landscape", "food", "document", "general"} else "general",
        executor="deterministic",
        preserve=preserve,
        operations=operations,
        should_write_memory=should_write_memory,
        memory_candidates=memory_candidates,
        needs_confirmation=needs_confirmation,
    )


def _normalized_mask_signature(mask_options: dict[str, Any], *, region: str) -> tuple[str, dict[str, Any]] | None:
    """Build a reusable mask signature independent of free-form region labels."""

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


def _record_mask_catalog_item(
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


def _append_execution_skip(
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


def _parse_tool_message_payload(payload: Any) -> dict[str, Any]:
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


def _invoke_tool_node(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    writer,
) -> ToolExecutionResult:
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
    payload = _parse_tool_message_payload(tool_messages[-1].content)
    return ToolExecutionResult.model_validate(payload)


def _should_skip_stage(state: EditState, *, stage_key: StageKey) -> tuple[bool, str | None]:
    """Return whether a stage should be skipped deterministically."""

    edit_profile = coerce_edit_profile(state.get("edit_profile"))
    if edit_profile is None:
        return False, None

    request_constraints = {
        constraint
        for constraint in (coerce_request_intent(state.get("request_intent")).constraints if state.get("request_intent") else [])
    }

    if stage_key == "technical_prep" and not edit_profile.technical_issues and not (
        request_constraints & {"fix_technical_issues", "straighten", "denoise"}
    ):
        return True, "当前没有明显技术预处理需求。"
    if stage_key == "local_balance" and not edit_profile.local_balance_needed:
        return True, "当前没有明显局部平衡需求。"
    if stage_key == "subject_refine" and not edit_profile.subject_refine_needed:
        return True, "当前主体不需要额外细化。"
    return False, None


def _build_rule_stage_plan(
    *,
    stage_key: StageKey,
    stage_policy: StagePolicy,
    state: EditState,
    stage_context: StageContextEnvelope,
) -> PlannerExecutionPlan:
    """Build a small deterministic fallback plan when stage LLM planning is unavailable."""

    request_text = str(state.get("request_text") or "")
    analysis = coerce_image_analysis(state.get("image_analysis"))

    def make_step(op: str, *, region: str = WHOLE_IMAGE_REGION, params: dict[str, Any] | None = None) -> PlannerExecutionStep:
        return PlannerExecutionStep(op=op, region=region, params=dict(params or {}), constraints=[], priority=0)

    steps: list[PlannerExecutionStep] = []
    if stage_key == "technical_prep":
        steps = []
    elif stage_key == "global_base":
        if analysis and ("underexposed" in analysis.issues or "偏暗" in request_text):
            steps.append(make_step("adjust_exposure", params={"strength": 0.45, "max_stops": 2.0}))
        if analysis and any(issue in analysis.issues for issue in ("flat", "low_contrast", "hazy")):
            steps.append(make_step("adjust_contrast", params={"strength": 0.38}))
        if any(word in request_text for word in ("色彩", "饱和", "鲜艳", "通透", "空气感", "夏日")):
            steps.append(make_step("adjust_vibrance_saturation", params={"strength": 0.3}))
    elif stage_key == "local_balance":
        if stage_context.available_masks:
            mask = stage_context.available_masks[0]
            steps.append(
                make_step(
                    "adjust_exposure",
                    region=(mask.get("region_labels") or ["局部区域"])[0],
                    params={
                        "strength": 0.28,
                        "mask_provider": mask.get("provider") or "fal_sam3",
                        "mask_prompt": mask.get("mask_prompt") or "subject",
                        "mask_semantic_type": bool(mask.get("semantic_type")),
                    },
                )
            )
    elif stage_key == "subject_refine":
        if stage_context.available_masks:
            mask = stage_context.available_masks[0]
            steps.append(
                make_step(
                    "adjust_contrast",
                    region=(mask.get("region_labels") or ["主体区域"])[0],
                    params={
                        "strength": 0.22,
                        "mask_provider": mask.get("provider") or "fal_sam3",
                        "mask_prompt": mask.get("mask_prompt") or "subject",
                        "mask_semantic_type": bool(mask.get("semantic_type")),
                    },
                )
            )
    elif stage_key == "finish_output":
        steps = []

    trimmed = [
        step.model_copy(update={"priority": index})
        for index, step in enumerate(steps[: stage_policy.step_budget])
    ]
    return PlannerExecutionPlan(
        mode=str(state.get("mode") or "auto"),
        domain=str((analysis.domain if analysis is not None else None) or "general"),
        executor="deterministic",
        preserve=[],
        steps=trimmed,
        step_budget=stage_policy.step_budget,
        summary=f"{STAGE_LABELS[stage_key]}规则计划",
        should_write_memory=False,
        memory_candidates=[],
        needs_confirmation=False,
    )


def prepare_stage_context(state: EditState, *, stage_key: StageKey) -> dict[str, Any]:
    """Prepare the minimal context and effective policy for one stage."""

    current_image_path = _current_image_path(state)
    edit_profile = coerce_edit_profile(state.get("edit_profile"))
    image_analysis = coerce_image_analysis(state.get("image_analysis"))
    request_intent = coerce_request_intent(state.get("request_intent"))
    stage_policy = resolve_stage_policy(stage_key, edit_profile)
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    should_skip, skip_reason = _should_skip_stage(state, stage_key=stage_key)
    stage_context = build_stage_context(
        stage_key=stage_key,
        request_text=str(
            state.get("request_text")
            or (request_intent.goal_summary if request_intent is not None else "")
        ),
        current_image_path=current_image_path,
        image_analysis=image_analysis,
        edit_profile=edit_profile,
        mask_catalog=state.get("mask_catalog"),
        phases={key: value.model_dump(mode="json") for key, value in phases.items()},
        stage_constraints=[
            *(request_intent.constraints if request_intent is not None else []),
            *summarize_policy_constraints(stage_policy),
        ],
    )

    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    phase.skipped = should_skip
    phase.skip_reason = skip_reason
    phases[stage_key] = phase

    writer = _safe_stream_writer()
    writer(
        {
            "event": "stage_started",
            "stage": stage_key,
            "message": f"开始{STAGE_LABELS[stage_key]}",
        },
    )

    return {
        "current_stage": stage_key,
        "stage_policy": stage_policy.model_dump(mode="json"),
        "stage_context": stage_context.model_dump(mode="json"),
        "stage_plan": None,
        "phases": phases,
        "mask_catalog": coerce_mask_catalog(state.get("mask_catalog")).model_dump(mode="json"),
    }


def should_skip_stage(state: EditState, *, stage_key: StageKey) -> str:
    """Route within the stage subgraph based on deterministic skip decision."""

    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    return "skip" if phase.skipped else "run"


def build_stage_plan(state: EditState, *, stage_key: StageKey) -> dict[str, Any]:
    """Build a bounded stage plan with model-first and rule fallback behavior."""

    stage_policy = coerce_stage_policy(state.get("stage_policy"))
    stage_context = coerce_stage_context(state.get("stage_context"))
    if stage_policy is None or stage_context is None:
        raise ValueError(f"Missing stage policy/context for {stage_key}")

    tool_catalog = state.get("tool_catalog") or []
    fallback_trace = list(state.get("fallback_trace") or [])
    current_image_path = _current_image_path(state)

    writer = _safe_stream_writer()
    writer(
        {
            "event": "planner_started",
            "stage": stage_key,
            "message": f"正在生成{STAGE_LABELS[stage_key]}计划",
        },
    )

    if stage_policy.llm_enabled and planner_execution_model_available():
        try:
            stage_plan = generate_stage_execution_plan_with_qwen(
                stage_policy=stage_policy,
                stage_context=stage_context,
                tool_catalog=tool_catalog,
                current_image_path=current_image_path,
                fallback_mode=str(state.get("mode") or "auto"),
                fallback_domain=str((coerce_image_analysis(state.get("image_analysis")).domain if state.get("image_analysis") else None) or "general"),
            )
        except Exception as error:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="planner_model",
                location=stage_key,
                strategy="rule_based_plan",
                message="阶段规划失败，改用规则计划。",
                error=str(error),
            )
            stage_plan = _build_rule_stage_plan(
                stage_key=stage_key,
                stage_policy=stage_policy,
                state=state,
                stage_context=stage_context,
            )
    else:
        stage_plan = _build_rule_stage_plan(
            stage_key=stage_key,
            stage_policy=stage_policy,
            state=state,
            stage_context=stage_context,
        )

    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    phase.plan = stage_plan
    phases[stage_key] = phase
    writer(
        {
            "event": "planner_finished",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}计划已生成",
            "summary": stage_plan.summary,
            "num_steps": len(stage_plan.steps),
        },
    )
    return {
        "stage_plan": stage_plan.model_dump(mode="json"),
        "phases": phases,
        "fallback_trace": fallback_trace,
    }


def _execute_single_tool_call(
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
) -> tuple[str, dict[str, Any], MaskCatalog]:
    """Execute one tool call against the current image under a stage policy."""

    op_name = str(operation["op"])
    writer = _safe_stream_writer()
    registry = build_default_tool_registry()

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
        _append_execution_skip(
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

    registered_tool = registry.require(op_name)
    region = str(operation.get("region") or WHOLE_IMAGE_REGION)
    params = dict(operation.get("params") or {})
    if operation.get("strength") is not None and params.get("strength") is None:
        params["strength"] = operation["strength"]
    normalized_params = normalize_runtime_tool_params(op_name, params)
    mask_params = extract_runtime_mask_params(normalized_params)
    requires_mask = registered_tool.spec.supports_mask and (
        region != WHOLE_IMAGE_REGION or bool(mask_params)
    )

    if requires_mask and not stage_policy.mask_allowed:
        _append_execution_skip(
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
            mask_options = MaskParams.model_validate(mask_params).to_runtime_options() if mask_params else {}
        except Exception as error:
            _append_execution_skip(
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

        signature_info = _normalized_mask_signature(mask_options, region=region)
        signature, signature_payload = signature_info if signature_info is not None else (None, None)
        if signature and signature in updated_catalog.items and updated_catalog.items[signature].mask_path:
            entry = updated_catalog.items[signature]
            current_mask_path = entry.mask_path
            updated_catalog = _record_mask_catalog_item(
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
            # Build a stable local output directory near the current image.
            mask_output_dir = str(
                Path(current_image).resolve().parent / "output" / f"{Path(current_image).stem}_{stage_key}_mask"
            )
            try:
                segmentation_result = resolve_region_mask(
                    current_image,
                    region,
                    output_dir=mask_output_dir,
                    **mask_options,
                )
            except Exception as error:
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
                _append_execution_skip(
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
                updated_catalog = _record_mask_catalog_item(
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
    tool_args["image_path"] = current_image
    if current_mask_path:
        tool_args["mask_path"] = current_mask_path

    try:
        result = _invoke_tool_node(
            tool_name=op_name,
            tool_args=tool_args,
            writer=writer,
        )
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


def execute_stage_plan(state: EditState, *, stage_key: StageKey) -> dict[str, Any]:
    """Execute one stage plan sequentially with stage-local guard semantics."""

    stage_policy = coerce_stage_policy(state.get("stage_policy"))
    stage_plan = state.get("stage_plan")
    stage_plan_obj = PlannerExecutionPlan.model_validate(stage_plan) if stage_plan is not None else None
    if stage_policy is None or stage_plan_obj is None:
        raise ValueError(f"Missing stage policy/plan for {stage_key}")

    current_image = _current_image_path(state)
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("execution_trace") or [])]
    segmentation_trace = [item.model_dump(mode="json") for item in coerce_segmentation_trace(state.get("segmentation_trace") or [])]
    fallback_trace = list(state.get("fallback_trace") or [])
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    mask_catalog = coerce_mask_catalog(state.get("mask_catalog"))
    stage_execution_trace: list[dict[str, Any]] = []
    stage_segmentation_trace: list[dict[str, Any]] = []
    trigger_reasons = list(phase.trigger_reasons)
    stopped_early = False
    op_counts: dict[str, int] = {}
    tone_stack_count = 0

    writer = _safe_stream_writer()
    writer(
        {
            "event": "stage_execution_started",
            "stage": stage_key,
            "message": f"开始执行{STAGE_LABELS[stage_key]}",
        },
    )

    for step in stage_plan_obj.steps[: stage_policy.step_budget]:
        op_name = step.op
        op_counts[op_name] = op_counts.get(op_name, 0)
        if op_counts[op_name] >= stage_policy.tool_repeat_limit:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message=f"{op_name} 已达到当前阶段单工具上限，提前结束阶段。",
                error=None,
            )
            trigger_reasons.append("tool_budget_exceeded")
            stopped_early = True
            break
        if stage_policy.tone_stack_limit is not None and op_name in TONE_STACK_OPS and tone_stack_count >= stage_policy.tone_stack_limit:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message="当前阶段 tone stack 已达到上限，提前结束阶段。",
                error=None,
            )
            trigger_reasons.append("tone_budget_exceeded")
            stopped_early = True
            break

        current_image, result_summary, mask_catalog = _execute_single_tool_call(
            stage_key=stage_key,
            stage_policy=stage_policy,
            current_image=current_image,
            operation=step.model_dump(mode="json"),
            execution_trace=execution_trace,
            stage_execution_trace=stage_execution_trace,
            segmentation_trace=segmentation_trace,
            stage_segmentation_trace=stage_segmentation_trace,
            fallback_trace=fallback_trace,
            candidate_outputs=candidate_outputs,
            mask_catalog=mask_catalog,
        )
        op_counts[op_name] += 1
        if op_name in TONE_STACK_OPS:
            tone_stack_count += 1

        if stage_key == "global_base":
            metrics = _compute_image_metrics(current_image)
            brightness_limit = stage_policy.guard_thresholds.get("brightness_mean_max")
            highlight_limit = stage_policy.guard_thresholds.get("highlight_ratio_max")
            if (
                (brightness_limit is not None and metrics["brightness_mean"] > brightness_limit)
                or (highlight_limit is not None and metrics["highlight_ratio"] > highlight_limit)
            ):
                fallback_trace = append_fallback_trace(
                    fallback_trace,
                    stage=stage_key,
                    source="execution_guard",
                    location=op_name,
                    strategy="stop_current_stage",
                    message="检测到全图过曝风险，提前结束当前阶段。",
                    error=None,
                )
                trigger_reasons.append("highlight_guard_triggered")
                stopped_early = True
                break

        if stage_key == "subject_refine" and result_summary.get("mask_path"):
            metrics = _compute_image_metrics(current_image, mask_path=str(result_summary["mask_path"]))
            if (
                metrics["brightness_mean"] > stage_policy.guard_thresholds.get("human_subject_brightness_mean_max", 242.0)
                or metrics["highlight_ratio"] > stage_policy.guard_thresholds.get("human_subject_highlight_ratio_max", 0.42)
                or metrics["saturation_mean"] < stage_policy.guard_thresholds.get("human_subject_saturation_mean_min", 0.04)
            ):
                fallback_trace = append_fallback_trace(
                    fallback_trace,
                    stage=stage_key,
                    source="execution_guard",
                    location=op_name,
                    strategy="stop_current_stage",
                    message="检测到主体局部异常风险，提前结束当前阶段。",
                    error=None,
                )
                trigger_reasons.append("subject_guard_triggered")
                stopped_early = True
                break

        if stage_key == "finish_output" and any(item.get("mask_path") for item in stage_execution_trace):
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message="finish_output 不允许新增局部遮罩，提前结束当前阶段。",
                error=None,
            )
            trigger_reasons.append("mask_not_allowed")
            stopped_early = True
            break

    phase.execution_trace = coerce_execution_trace(stage_execution_trace)
    phase.segmentation_trace = coerce_segmentation_trace(stage_segmentation_trace)
    phase.output = PhaseOutputArtifact(image_path=current_image)
    phase.trigger_reasons = trigger_reasons
    phase.stopped_early = stopped_early
    phases[stage_key] = phase
    writer(
        {
            "event": "stage_execution_completed",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}执行完成",
        },
    )

    return {
        "selected_output": current_image,
        "candidate_outputs": candidate_outputs,
        "execution_trace": coerce_execution_trace(execution_trace),
        "segmentation_trace": coerce_segmentation_trace(segmentation_trace),
        "fallback_trace": fallback_trace,
        "phases": phases,
        "mask_catalog": mask_catalog.model_dump(mode="json"),
    }


def stage_guard(state: EditState, *, stage_key: StageKey) -> dict[str, Any]:
    """Compute deterministic stage evaluation facts after execution or skip."""

    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    if phase.skipped:
        report = EvaluationReport(
            selected_output=state.get("selected_output"),
            num_operations=0,
            success_count=0,
            failure_count=0,
            fallback_count=0,
            has_output=bool(state.get("selected_output")),
            overall_ok=True,
            artifact_ok=True,
            summary=phase.skip_reason or f"{STAGE_LABELS[stage_key]}已跳过。",
            should_continue_editing=False,
            should_request_review=False,
        )
    else:
        trace_items = [item.model_dump(mode="json") for item in phase.execution_trace]
        success_count = sum(1 for item in trace_items if item.get("ok"))
        failure_count = sum(1 for item in trace_items if item.get("ok") is False)
        fallback_count = sum(1 for item in trace_items if item.get("fallback_used"))
        issues = []
        warnings = []
        if failure_count:
            issues.append("当前阶段存在失败步骤")
        if fallback_count:
            warnings.append("当前阶段触发了自动降级")
        if phase.stopped_early:
            warnings.append("当前阶段提前结束")
        report = EvaluationReport(
            selected_output=phase.output.image_path if phase.output is not None else state.get("selected_output"),
            num_operations=len(trace_items),
            success_count=success_count,
            failure_count=failure_count,
            fallback_count=fallback_count,
            has_output=bool(phase.output and phase.output.image_path),
            overall_ok=not issues,
            artifact_ok=not issues,
            issues=issues,
            warnings=warnings,
            summary=f"{STAGE_LABELS[stage_key]}已完成。",
            should_continue_editing=False,
            should_request_review=False,
        )
    phase.eval_report = report
    phases[stage_key] = phase
    return {"phases": phases}


def summarize_stage(state: EditState, *, stage_key: StageKey) -> dict[str, Any]:
    """Summarize one stage and update the aggregated edit plan."""

    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or _base_phase_artifact(stage_key)
    request_intent = coerce_request_intent(state.get("request_intent"))
    image_analysis = coerce_image_analysis(state.get("image_analysis"))

    if phase.skipped:
        phase.summary = StageSummary(
            stage=stage_key,
            summary=phase.skip_reason or f"{STAGE_LABELS[stage_key]}已跳过。",
            used_tools=[],
            key_changes=[],
            remaining_issues=[],
        )
    else:
        used_tools = [item.op for item in phase.execution_trace if item.op]
        remaining_issues = list((phase.eval_report.issues if phase.eval_report is not None else []) + (phase.eval_report.warnings if phase.eval_report is not None else []))
        plan_summary = phase.plan.summary if phase.plan is not None else ""
        phase.summary = StageSummary(
            stage=stage_key,
            summary=plan_summary or (phase.eval_report.summary if phase.eval_report is not None else f"{STAGE_LABELS[stage_key]}已完成。"),
            used_tools=used_tools,
            key_changes=used_tools[:3],
            remaining_issues=remaining_issues[:3],
        )
    phases[stage_key] = phase

    edit_plan = _merge_stage_plans_into_edit_plan(
        phases,
        mode=str(state.get("mode") or (request_intent.mode if request_intent is not None else "auto")),
        domain=str((image_analysis.domain if image_analysis is not None else None) or "general"),
    )
    writer = _safe_stream_writer()
    writer(
        {
            "event": "stage_completed",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}已完成",
        },
    )
    return {
        "current_stage": stage_key,
        "phases": phases,
        "edit_plan": edit_plan,
        "stage_context": None,
        "stage_plan": None,
        "stage_policy": None,
    }
