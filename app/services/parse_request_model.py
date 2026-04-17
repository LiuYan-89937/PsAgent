"""Text-only request parsing service powered by Qwen."""

from __future__ import annotations

from app.graph.state import RequestIntent
from app.services.model_context import compact_package_catalog_for_model
from app.services.qwen_model import DEFAULT_TEXT_MODEL, call_qwen_for_json, qwen_model_available


def parse_request_model_available() -> bool:
    """Return whether the request-parser model can be called."""

    return qwen_model_available()


def generate_request_intent_with_qwen(
    *,
    request_text: str,
    package_catalog: list[dict],
) -> RequestIntent:
    """Call Qwen to normalize the user's instruction into a request intent."""

    payload = call_qwen_for_json(
        prompt_name="parse_request.txt",
        user_payload={
            "用户需求": request_text,
            "工具目录": compact_package_catalog_for_model(package_catalog, include_params=False),
            "补充要求": [
                "只做需求归一化，不要生成最终 edit plan",
                "全图请求用 whole_image，局部请求用动态区域标签",
                "requested_packages 只保留高层意图，不要过度展开",
                "只返回 JSON",
            ],
        },
        model_env_name="DASHSCOPE_REQUEST_MODEL",
        default_model=DEFAULT_TEXT_MODEL,
        temperature=0.1,
    )
    return RequestIntent.model_validate(payload)
