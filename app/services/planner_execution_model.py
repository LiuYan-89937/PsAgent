"""Stage-aware planner service for bounded parameter-retouch plans."""

from __future__ import annotations

from typing import Any

from app.graph.state import (
    PlannerExecutionPlan,
    PlannerExecutionStep,
    StageContextEnvelope,
    StagePolicy,
    coerce_stage_context,
    coerce_stage_policy,
    ToolCatalogItem,
    coerce_tool_catalog,
)
from app.services.model_context import compact_tool_catalog_for_model, shared_mask_params_for_model
from app.services.planner_runtime_helpers import build_operation_from_tool_call, resolve_planner_tool_name
from app.services.qwen_model import DEFAULT_TEXT_MODEL, call_qwen_for_json, qwen_model_available
from app.tools import TOOL_SPECS_BY_NAME, WHOLE_IMAGE_REGION


def planner_execution_model_available() -> bool:
    """Return whether the stage planner model can be called."""

    return qwen_model_available()


def filter_stage_tool_catalog(
    tool_catalog: list[ToolCatalogItem | dict[str, Any]],
    *,
    visible_tools: list[str],
) -> list[ToolCatalogItem]:
    """Filter the exported tool catalog down to the stage-visible tools."""

    allowed = set(visible_tools)
    return [item for item in coerce_tool_catalog(tool_catalog) if item.name in allowed]


def _normalize_raw_step_payload(
    step: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Normalize a loose model-returned step into PlannerExecutionStep-compatible shape."""

    op = step.get("op")
    if not isinstance(op, str) or not op.strip():
        tool = step.get("tool")
        if isinstance(tool, str) and tool.strip():
            op = tool.strip()

    params = step.get("params")
    if not isinstance(params, dict):
        parameters = step.get("parameters")
        if isinstance(parameters, dict):
            params = dict(parameters)
        else:
            params = {}

    structural_keys = {
        "op",
        "tool",
        "region",
        "constraints",
        "priority",
        "params",
        "parameters",
        "strength",
    }
    for key, value in step.items():
        if key in structural_keys:
            continue
        params.setdefault(key, value)

    normalized: dict[str, Any] = {
        "op": op,
        "region": step.get("region") or WHOLE_IMAGE_REGION,
        "params": params,
        "constraints": list(step.get("constraints") or []),
        "priority": step.get("priority", index),
    }
    if "strength" in step:
        strength_value = step.get("strength")
        if isinstance(strength_value, (int, float)) and -1.0 <= float(strength_value) <= 1.0:
            normalized["strength"] = strength_value
        else:
            normalized["params"].setdefault("strength", strength_value)
    return normalized


def _normalize_raw_plan_payload(
    payload: dict[str, Any],
    *,
    stage_policy: StagePolicy,
    fallback_mode: str,
    fallback_domain: str,
) -> dict[str, Any]:
    """Normalize a loose model JSON payload into PlannerExecutionPlan-compatible shape."""

    candidate = dict(payload)
    nested_plan = candidate.get("plan")
    if isinstance(nested_plan, dict):
        merged = dict(nested_plan)
        for key, value in candidate.items():
            merged.setdefault(key, value)
        candidate = merged

    raw_steps = candidate.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = candidate.get("operations")
    if not isinstance(raw_steps, list):
        raw_steps = []

    normalized_steps = [
        _normalize_raw_step_payload(step if isinstance(step, dict) else {}, index=index)
        for index, step in enumerate(raw_steps[: stage_policy.step_budget])
    ]

    executor = "deterministic"

    mode = candidate.get("mode")
    if not isinstance(mode, str) or mode not in {"explicit", "auto"}:
        mode = fallback_mode if fallback_mode in {"explicit", "auto"} else "auto"

    domain = candidate.get("domain")
    if not isinstance(domain, str) or domain not in {"portrait", "landscape", "food", "document", "general"}:
        domain = fallback_domain if fallback_domain in {"portrait", "landscape", "food", "document", "general"} else "general"

    summary = candidate.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        for fallback_key in ("stage_summary", "reason", "message"):
            value = candidate.get(fallback_key)
            if isinstance(value, str) and value.strip():
                summary = value
                break
    if not isinstance(summary, str) or not summary.strip():
        summary = f"{stage_policy.label}计划"

    return {
        "mode": mode,
        "domain": domain,
        "executor": executor,
        "preserve": list(candidate.get("preserve") or []),
        "steps": normalized_steps,
        "step_budget": stage_policy.step_budget,
        "summary": summary,
        "should_write_memory": bool(candidate.get("should_write_memory", False)),
        "memory_candidates": list(candidate.get("memory_candidates") or []),
        "needs_confirmation": bool(candidate.get("needs_confirmation", False)),
    }


def _normalize_plan_steps(
    *,
    steps: list[PlannerExecutionStep | dict[str, Any]],
    stage_policy: StagePolicy,
) -> list[PlannerExecutionStep]:
    """Resolve tool names, decode planner params, and enforce stage tool visibility."""

    allowed = set(stage_policy.visible_tools)
    normalized_steps: list[PlannerExecutionStep] = []
    for index, step in enumerate(steps[: stage_policy.step_budget]):
        validated = step if isinstance(step, PlannerExecutionStep) else PlannerExecutionStep.model_validate(step)
        raw_arguments = dict(validated.params or {})
        raw_arguments["region"] = validated.region or WHOLE_IMAGE_REGION
        resolved_tool_name, _ = resolve_planner_tool_name(validated.op, raw_arguments)
        if resolved_tool_name not in allowed:
            raise RuntimeError(f"Planner selected disallowed tool for {stage_policy.key}: {resolved_tool_name}")
        if resolved_tool_name not in TOOL_SPECS_BY_NAME:
            raise RuntimeError(f"Planner selected unknown tool: {resolved_tool_name}")
        runtime_operation = build_operation_from_tool_call(resolved_tool_name, raw_arguments)
        normalized_steps.append(
            PlannerExecutionStep(
                op=resolved_tool_name,
                region=runtime_operation["region"],
                strength=runtime_operation.get("strength"),
                params=dict(runtime_operation.get("params") or {}),
                constraints=list(validated.constraints or []),
                priority=index,
            )
        )
    return normalized_steps


def _build_stage_planner_payload(
    *,
    stage_policy: StagePolicy,
    stage_context: StageContextEnvelope,
    tool_catalog: list[ToolCatalogItem],
) -> dict[str, Any]:
    """Build the planner request payload for a single stage."""

    context = coerce_stage_context(stage_context)
    return {
        "当前阶段": stage_policy.key,
        "阶段目标": stage_policy.label,
        "请求摘要": context.request_summary,
        "当前图片": context.current_image_path,
        "修图画像": context.edit_profile_summary,
        "相关图像信息": context.relevant_image_analysis,
        "可用遮罩": context.available_masks,
        "前序阶段摘要": context.previous_stage_summaries,
        "阶段约束": context.stage_constraints,
        "工具目录": compact_tool_catalog_for_model(
            [item.model_dump(mode="json") for item in tool_catalog],
            include_params=True,
        ),
        "共享遮罩参数": shared_mask_params_for_model([item.model_dump(mode="json") for item in tool_catalog]),
        "步数预算": stage_policy.step_budget,
        "补充要求": [
            "只规划当前阶段，不要尝试解决后续阶段问题。",
            "steps 数量不能超过步数预算。",
            "只能选择当前阶段工具目录中的工具。",
            "只在需要局部处理时才填写 mask_* 参数。",
            "mask_prompt 必须是单个英文词汇，只写一个可见主体或物体。",
            "不要输出 mask_negative_prompt。",
            "所有数值参数只允许填写 0-100 整数，不要输出小数。",
            "如果当前阶段无需处理，可以返回 0-step plan。",
        ],
    }


def generate_stage_execution_plan_with_qwen(
    *,
    stage_policy: StagePolicy | dict[str, Any],
    stage_context: StageContextEnvelope | dict[str, Any],
    tool_catalog: list[ToolCatalogItem | dict[str, Any]],
    current_image_path: str,
    fallback_mode: str = "auto",
    fallback_domain: str = "general",
) -> PlannerExecutionPlan:
    """Generate a bounded stage plan using the stage-specific prompt and tool list."""

    policy = coerce_stage_policy(stage_policy)
    context = coerce_stage_context(stage_context)
    if policy is None or context is None:
        raise RuntimeError("Stage policy and stage context are required for stage planning.")

    filtered_catalog = filter_stage_tool_catalog(tool_catalog, visible_tools=policy.visible_tools)
    payload = call_qwen_for_json(
        prompt_name=policy.prompt_name,
        user_payload=_build_stage_planner_payload(
            stage_policy=policy,
            stage_context=context,
            tool_catalog=filtered_catalog,
        ),
        model_env_name="DASHSCOPE_PLANNER_MODEL",
        default_model=DEFAULT_TEXT_MODEL,
        image_paths=[current_image_path],
        temperature=0.1,
    )
    normalized_payload = _normalize_raw_plan_payload(
        payload if isinstance(payload, dict) else {},
        stage_policy=policy,
        fallback_mode=fallback_mode,
        fallback_domain=fallback_domain,
    )
    plan = PlannerExecutionPlan.model_validate(normalized_payload)
    normalized_steps = _normalize_plan_steps(steps=plan.steps, stage_policy=policy)
    return PlannerExecutionPlan(
        mode=plan.mode,
        domain=plan.domain,
        executor=plan.executor,
        preserve=list(plan.preserve),
        steps=normalized_steps,
        step_budget=policy.step_budget,
        summary=plan.summary,
        should_write_memory=plan.should_write_memory,
        memory_candidates=list(plan.memory_candidates),
        needs_confirmation=plan.needs_confirmation,
    )
