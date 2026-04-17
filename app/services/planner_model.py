"""Planner model client for DashScope-compatible Qwen calls."""

from __future__ import annotations

from typing import Any

from app.graph.state import EditPlan
from app.services.model_context import (
    compact_image_analysis_for_model,
    compact_package_catalog_for_model,
    compact_plan_for_model,
    compact_preferences_for_model,
    compact_request_intent_for_model,
    shared_mask_params_for_model,
)
from app.services.qwen_model import DEFAULT_TEXT_MODEL, call_qwen_for_json, qwen_model_available
from app.tools.packages import build_default_package_registry


def planner_model_available() -> bool:
    """Return whether the planner model can be called in the current env."""

    return qwen_model_available()


def _choose_executor(operations: list[dict[str, Any]]) -> str:
    """Pick an executor from raw planner operations."""

    if any(operation.get("region", "whole_image") != "whole_image" for operation in operations):
        return "hybrid"
    return "deterministic"


def _summarize_previous_execution_trace(trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return a compact execution summary for planner context."""

    summary: list[dict[str, Any]] = []
    for item in trace or []:
        summary.append(
            {
                "op": item.get("op"),
                "region": item.get("region"),
                "ok": item.get("ok"),
                "fallback_used": item.get("fallback_used"),
                "error": item.get("error"),
            }
        )
    return summary


def _summarize_previous_eval_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Return a compact evaluation summary for planner context."""

    payload = dict(report or {})
    return {
        "summary": payload.get("summary"),
        "issues": payload.get("issues", []),
        "warnings": payload.get("warnings", []),
        "should_continue_editing": payload.get("should_continue_editing"),
        "should_request_review": payload.get("should_request_review"),
    }


def _normalize_plan_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    mode: str,
    domain: str,
) -> EditPlan:
    """Normalize the raw model payload into a validated EditPlan."""

    registry = build_default_package_registry()

    if isinstance(payload, list):
        # 兼容模型直接返回 operations 数组的情况：
        # [
        #   {"op": "adjust_exposure", ...},
        #   {"op": "adjust_contrast", ...}
        # ]
        plan = {"operations": list(payload)}
    elif isinstance(payload, dict):
        if "operations" in payload:
            plan = dict(payload)
        elif "plan" in payload:
            plan = {"operations": payload.get("plan", [])}
            for key in ("mode", "domain", "executor", "preserve", "should_write_memory", "memory_candidates", "needs_confirmation"):
                if key in payload:
                    plan[key] = payload[key]
        elif "edit_plan" in payload and isinstance(payload["edit_plan"], dict):
            plan = dict(payload["edit_plan"])
        else:
            plan = dict(payload)
    else:
        raise RuntimeError(f"Planner returned unsupported payload type: {type(payload).__name__}")

    operations = []
    for priority, operation in enumerate(plan.get("operations", [])):
        operation_payload = dict(operation)
        package_name = operation_payload.get("op")
        package = registry.get(str(package_name)) if package_name else None
        params = package.get_operation_params(operation_payload) if package is not None else dict(operation_payload.get("params", {}))
        operation_payload.setdefault("region", "whole_image")
        operation_payload["params"] = params
        operation_payload.setdefault("constraints", [])
        operation_payload.setdefault("priority", priority)
        operations.append(operation_payload)

    plan["operations"] = operations
    plan.setdefault("mode", mode)
    plan.setdefault("domain", domain or "general")
    plan.setdefault("executor", _choose_executor(operations))
    plan.setdefault("preserve", [])
    plan.setdefault("should_write_memory", False)
    plan.setdefault("memory_candidates", [])
    plan.setdefault("needs_confirmation", False)
    validated = EditPlan.model_validate(plan)
    if not validated.operations:
        raise RuntimeError("Planner returned an empty operations list.")
    return validated


def generate_edit_plan_with_qwen(
    *,
    request_text: str,
    request_intent: dict[str, Any],
    image_analysis: dict[str, Any],
    package_catalog: list[dict[str, Any]],
    retrieved_prefs: list[dict[str, Any]],
    image_paths: list[str] | None = None,
    round_name: str = "round_1",
    previous_plan: dict[str, Any] | None = None,
    previous_execution_trace: list[dict[str, Any]] | None = None,
    previous_eval_report: dict[str, Any] | None = None,
) -> EditPlan:
    """Call DashScope-compatible Qwen to produce a structured edit plan."""

    payload = call_qwen_for_json(
        prompt_name="planner.txt",
        user_payload={
            "用户需求": request_text,
            "需求意图": compact_request_intent_for_model(request_intent),
            "图像分析": compact_image_analysis_for_model(image_analysis),
            "长期偏好": compact_preferences_for_model(retrieved_prefs),
            "工具目录": compact_package_catalog_for_model(package_catalog, include_params=True),
            "局部分割公共参数": shared_mask_params_for_model(package_catalog),
            "当前轮次": round_name,
            "上一轮计划": compact_plan_for_model(previous_plan),
            "上一轮执行摘要": _summarize_previous_execution_trace(previous_execution_trace),
            "上一轮评估摘要": _summarize_previous_eval_report(previous_eval_report),
            "补充要求": [
                "op 必须来自 工具目录",
                "params 必须符合对应工具的参数说明",
                "局部遮罩相关字段优先参考 局部分割公共参数",
                "全图操作的 region 写 whole_image，局部操作的 region 写动态区域标签",
                "局部操作优先补齐 mask_provider、mask_prompt、mask_negative_prompt、mask_semantic_type",
                "优先保守、分层、少而准的修图策略",
                "如果是 round_2，不要简单重复 round_1",
                "只返回 JSON",
            ],
        },
        model_env_name="DASHSCOPE_PLANNER_MODEL",
        default_model=DEFAULT_TEXT_MODEL,
        image_paths=image_paths,
        temperature=0.1,
    )

    mode = str(request_intent.get("mode") or ("explicit" if request_text else "auto"))
    domain = str(image_analysis.get("domain") or "general")
    return _normalize_plan_payload(payload, mode=mode, domain=domain)
