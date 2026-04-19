"""Planner tool-calling helpers for realtime round execution."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.services.model_context import (
    compact_image_analysis_for_model,
    compact_plan_for_model,
    compact_preferences_for_model,
    compact_request_intent_for_model,
)
from app.services.planner_param_codec import (
    decode_planner_operation_params,
    parse_repaired_tool_arguments,
    planner_param_spec,
)
from app.services.qwen_model import DEFAULT_TEXT_MODEL, call_qwen_for_tool_message, qwen_model_available
from app.tools.packages import PackageRegistry, build_default_package_registry
from app.tools.packages.base import WHOLE_IMAGE_REGION


class FinishRoundParams(BaseModel):
    """Planner-issued payload that ends the current round."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1)
    should_write_memory: bool = False
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    needs_confirmation: bool = False


def planner_tool_model_available() -> bool:
    """Return whether the realtime planner model can be called."""

    return qwen_model_available()


def _normalize_tool_name(value: str) -> str:
    """Normalize a planner-returned tool name into a stable comparison key."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _generated_aliases(package_name: str) -> set[str]:
    """Generate zero-cost aliases from a registered package name."""

    aliases = {_normalize_tool_name(package_name)}
    if package_name.startswith("adjust_"):
        aliases.add(_normalize_tool_name(package_name.removeprefix("adjust_")))
    else:
        aliases.add(_normalize_tool_name(f"adjust_{package_name}"))
    return aliases


def _char_trigram_vector(text: str) -> Counter[str]:
    """Build a light-weight character trigram vector."""

    normalized = f"  {_normalize_tool_name(text)}  "
    return Counter(normalized[index : index + 3] for index in range(max(len(normalized) - 2, 0)))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Return cosine similarity for sparse trigram vectors."""

    keys = set(left) | set(right)
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tool_candidate_text(package) -> str:
    """Build a compact retrieval document for one package."""

    fields = [
        package.name,
        package.spec.description,
        " ".join(package.spec.supported_regions),
        " ".join(package.spec.supported_domains),
        " ".join(sorted(_generated_aliases(package.name))),
    ]
    return " ".join(part for part in fields if part)


def should_attach_planner_image(
    *,
    current_step: int,
    latest_result: dict[str, Any] | None,
) -> bool:
    """Only attach the current image on the first planner step."""

    return current_step <= 1 or latest_result is None


def resolve_planner_tool_name(
    raw_tool_name: str,
    arguments: dict[str, Any],
    registry: PackageRegistry | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a planner-returned tool name with exact-first fallback to light similarity."""

    active_registry = registry or build_default_package_registry()
    normalized_name = _normalize_tool_name(raw_tool_name)
    if active_registry.get(raw_tool_name) is not None:
        return raw_tool_name, {"strategy": "exact", "score": 1.0}
    if active_registry.get(normalized_name) is not None:
        return normalized_name, {"strategy": "normalized", "score": 1.0}

    for package in active_registry.list():
        if normalized_name in _generated_aliases(package.name):
            return package.name, {"strategy": "alias", "score": 1.0}

    raw_query_vector = _char_trigram_vector(raw_tool_name)
    query_fragments = [raw_tool_name]
    region = arguments.get("region")
    if isinstance(region, str) and region:
        query_fragments.append(region)
    for key in sorted(arguments):
        if key != "region":
            query_fragments.append(key)
    query_vector = _char_trigram_vector(" ".join(query_fragments))

    scored_candidates: list[tuple[str, float]] = []
    for package in active_registry.list():
        candidate_vector = _char_trigram_vector(_tool_candidate_text(package))
        candidate_score = max(
            _cosine_similarity(raw_query_vector, candidate_vector),
            _cosine_similarity(query_vector, candidate_vector),
        )
        scored_candidates.append((package.name, candidate_score))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    if not scored_candidates:
        raise RuntimeError(f"Planner returned unknown tool: {raw_tool_name}")

    best_name, best_score = scored_candidates[0]
    second_score = scored_candidates[1][1] if len(scored_candidates) > 1 else 0.0
    if best_score < 0.42 or best_score - second_score < 0.05:
        top_candidates = [
            {"name": name, "score": round(score, 4)}
            for name, score in scored_candidates[:3]
        ]
        raise RuntimeError(
            f"Planner returned unknown tool '{raw_tool_name}', and similarity match was inconclusive: {top_candidates}"
        )

    return best_name, {
        "strategy": "similarity",
        "score": round(best_score, 4),
        "candidates": [
            {"name": name, "score": round(score, 4)}
            for name, score in scored_candidates[:3]
        ],
    }


def _build_region_schema() -> dict[str, Any]:
    """Return the common region schema shared by all package tools."""

    return {
        "type": "string",
        "description": (
            "可选字段。需要局部遮罩时填写动态区域标签，例如 逆光脸部和颈部皮肤、白裙区域、远处灰雾背景；不填或无 mask 参数时按全图处理。"
        ),
    }


def _package_tool_parameters(package) -> dict[str, Any]:
    """Build tool parameters for a package tool."""

    params_schema = package.get_params_schema()
    original_properties = dict(params_schema.get("properties", {}))
    properties = {
        key: planner_param_spec(value) if isinstance(value, dict) else value
        for key, value in original_properties.items()
    }
    properties["region"] = _build_region_schema()
    required = list(params_schema.get("required", []))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_planner_tools(registry: PackageRegistry | None = None) -> list[dict[str, Any]]:
    """Build planner tool schemas from the current package registry."""

    active_registry = registry or build_default_package_registry()
    tools: list[dict[str, Any]] = []
    for package in active_registry.list():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": package.name,
                    "description": package.spec.description,
                    "parameters": _package_tool_parameters(package),
                },
            }
        )

    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish_round",
                "description": "结束当前轮次，不再继续调用修图工具。",
                "parameters": FinishRoundParams.model_json_schema(),
            },
        }
    )
    return tools


def _build_round_step_payload(
    *,
    request_text: str,
    request_intent: dict[str, Any],
    image_analysis: dict[str, Any],
    retrieved_prefs: list[dict[str, Any]],
    round_name: str,
    current_step: int,
    round_operations: list[dict[str, Any]],
    latest_result: dict[str, Any] | None,
    previous_plan: dict[str, Any] | None = None,
    previous_execution_trace: list[dict[str, Any]] | None = None,
    previous_eval_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-step planner payload for realtime tool calling."""

    payload: dict[str, Any] = {
        "用户需求": request_text,
        "需求意图": compact_request_intent_for_model(request_intent),
        "图像分析": compact_image_analysis_for_model(image_analysis),
        "长期偏好": compact_preferences_for_model(retrieved_prefs),
        "当前轮次": round_name,
        "当前步数": current_step,
        "本轮已执行操作": compact_plan_for_model({"operations": round_operations}).get("operations", []),
        "最近一步结果": latest_result,
        "补充要求": [
            "不要输出 JSON。",
            "每次只能调用一个工具，或者调用 finish_round。",
            "如果当前图片还需要继续调整，就调用下一个最合适的工具。",
            "如果当前轮目标已经完成，就调用 finish_round。",
            "没有 mask 参数时默认按全图处理。",
            "只有需要局部处理时才补齐 mask_provider、mask_prompt、mask_semantic_type。",
            "mask_prompt 必须是单个英文词汇，只写一个可见主体或物体，不要写中文，不要写句子。",
            "所有数值参数只允许填写 0-100 整数，不要输出小数；后端会再映射回真实范围。",
            "局部操作时 region 作为区域标签，可填写动态区域标签；不需要局部时 region 可以省略。",
        ],
    }

    if previous_plan:
        payload["上一轮计划"] = compact_plan_for_model(previous_plan)
    if previous_execution_trace:
        payload["上一轮执行摘要"] = [
            {
                "op": item.get("op"),
                "region": item.get("region"),
                "ok": item.get("ok"),
                "fallback_used": item.get("fallback_used"),
                "error": item.get("error"),
            }
            for item in previous_execution_trace
        ]
    if previous_eval_report:
        payload["上一轮评估摘要"] = {
            "summary": previous_eval_report.get("summary"),
            "issues": previous_eval_report.get("issues", []),
            "warnings": previous_eval_report.get("warnings", []),
            "should_continue_editing": previous_eval_report.get("should_continue_editing"),
            "should_request_review": previous_eval_report.get("should_request_review"),
        }
    return payload


def call_planner_tool_turn(
    *,
    request_text: str,
    request_intent: dict[str, Any],
    image_analysis: dict[str, Any],
    retrieved_prefs: list[dict[str, Any]],
    current_image_path: str,
    round_name: str,
    current_step: int,
    round_operations: list[dict[str, Any]],
    latest_result: dict[str, Any] | None,
    planner_thinking_mode: bool = False,
    previous_plan: dict[str, Any] | None = None,
    previous_execution_trace: list[dict[str, Any]] | None = None,
    previous_eval_report: dict[str, Any] | None = None,
    registry: PackageRegistry | None = None,
) -> dict[str, Any]:
    """Request the next realtime planner tool call for the current round."""

    tools = build_planner_tools(registry)
    image_paths = [current_image_path] if should_attach_planner_image(current_step=current_step, latest_result=latest_result) else None
    tool_choice = "auto" if planner_thinking_mode else "required"
    enable_thinking = True if planner_thinking_mode else False
    return call_qwen_for_tool_message(
        prompt_name="planner.txt",
        user_payload=_build_round_step_payload(
            request_text=request_text,
            request_intent=request_intent,
            image_analysis=image_analysis,
            retrieved_prefs=retrieved_prefs,
            round_name=round_name,
            current_step=current_step,
            round_operations=round_operations,
            latest_result=latest_result,
            previous_plan=previous_plan,
            previous_execution_trace=previous_execution_trace,
            previous_eval_report=previous_eval_report,
        ),
        model_env_name="DASHSCOPE_PLANNER_MODEL",
        default_model=DEFAULT_TEXT_MODEL,
        tools=tools,
        image_paths=image_paths,
        temperature=0.1,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
    )


def extract_single_tool_call(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract exactly one tool call from an assistant message."""

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = message.get("content")
        raise RuntimeError(f"Planner did not return any tool call. content={content!r}")
    if len(tool_calls) != 1:
        raise RuntimeError(f"Planner returned {len(tool_calls)} tool calls in one step; expected exactly 1.")

    tool_call = tool_calls[0] or {}
    function_payload = tool_call.get("function") or {}
    tool_name = function_payload.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise RuntimeError("Planner returned a tool call without a valid function name.")

    raw_arguments = function_payload.get("arguments") or "{}"
    if not isinstance(raw_arguments, str):
        raise RuntimeError(f"Planner returned invalid tool arguments for {tool_name}.")
    try:
        parsed_arguments = parse_repaired_tool_arguments(raw_arguments)
    except (json.JSONDecodeError, RuntimeError) as exc:
        raise RuntimeError(f"Planner returned invalid JSON arguments for {tool_name}: {raw_arguments}") from exc

    if not isinstance(parsed_arguments, dict):
        raise RuntimeError(f"Planner arguments for {tool_name} must be an object.")
    return tool_name, parsed_arguments


def build_operation_from_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert a package tool call into a runtime operation dict."""

    region, params, strength = decode_planner_operation_params(tool_name, arguments)
    return {
        "op": tool_name,
        "region": region,
        "strength": strength if isinstance(strength, (int, float)) else None,
        "params": params,
        "constraints": [],
    }
