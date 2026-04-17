"""Vision-model powered image analysis service."""

from __future__ import annotations

from typing import Any

from app.services.qwen_model import DEFAULT_VISION_MODEL, call_qwen_for_json, qwen_model_available


def analyze_image_model_available() -> bool:
    """Return whether the image-analysis model can be called."""

    return qwen_model_available()


def generate_image_analysis_with_qwen(
    *,
    image_path: str,
    request_text: str,
    basic_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Call a multimodal Qwen model and return structured image analysis."""

    payload = call_qwen_for_json(
        prompt_name="analyze_image.txt",
        user_payload={
            "用户需求": request_text,
            "基础指标": basic_metrics,
            "补充要求": [
                "所有标签都必须基于图中真实可见内容",
                "优先输出对修图有帮助的信息",
                "subjects 和 segmentation_hints 要有实际可编辑意义",
                "只返回 JSON",
            ],
        },
        model_env_name="DASHSCOPE_VISION_MODEL",
        default_model=DEFAULT_VISION_MODEL,
        image_paths=[image_path],
        temperature=0.1,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Image analysis model did not return a JSON object.")
    return payload
