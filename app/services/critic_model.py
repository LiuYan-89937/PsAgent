"""Vision-model powered result evaluation service."""

from __future__ import annotations

from typing import Any

from app.services.model_context import (
    compact_execution_trace_for_model,
    compact_image_analysis_for_model,
    compact_plan_for_model,
)
from app.services.qwen_model import DEFAULT_CRITIC_MODEL, call_qwen_for_json, qwen_model_available


def critic_model_available() -> bool:
    """Return whether the critic model can be called."""

    return qwen_model_available()


def evaluate_edit_result_with_qwen(
    *,
    original_image_path: str,
    edited_image_path: str,
    request_text: str,
    edit_plan: dict[str, Any],
    image_analysis: dict[str, Any],
    execution_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call a multimodal Qwen critic and return a structured evaluation."""

    payload = call_qwen_for_json(
        prompt_name="critic.txt",
        user_payload={
            "用户需求": request_text,
            "修图计划": compact_plan_for_model(edit_plan),
            "图像分析": compact_image_analysis_for_model(image_analysis),
            "执行摘要": compact_execution_trace_for_model(execution_trace),
            "补充要求": [
                "必须比较原图和结果图",
                "重点评估自然度、主体保留、需求命中和修图痕迹",
                "不要过度表扬结果，要指出真实问题",
                "只返回 JSON",
            ],
        },
        model_env_name="DASHSCOPE_CRITIC_MODEL",
        default_model=DEFAULT_CRITIC_MODEL,
        image_paths=[original_image_path, edited_image_path],
        temperature=0.1,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Critic model did not return a JSON object.")
    return payload
