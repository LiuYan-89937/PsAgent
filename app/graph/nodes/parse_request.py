"""Parse user instruction and choose explicit or auto mode."""

from __future__ import annotations

from typing import Any

from app.graph.state import EditState, PackageCatalogItem, RequestIntent, RequestPackageHint
from app.tools import PARSE_REQUEST_KEYWORDS, WHOLE_IMAGE_ONLY_TOOL_NAMES
from app.services.parse_request_model import (
    generate_request_intent_with_qwen,
    parse_request_model_available,
)


def _extract_text_from_message_content(content: Any) -> str:
    """Normalize LangChain-style message content into plain text."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        return " ".join(part for part in parts if part)
    return ""


def _extract_latest_user_text(state: EditState) -> str:
    """Read the latest user-authored text from graph state."""

    if state.get("request_text"):
        return str(state["request_text"]).strip()

    messages = state.get("messages") or []
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        if message_type == "human":
            return _extract_text_from_message_content(getattr(message, "content", ""))
        if isinstance(message, dict) and message.get("type") in {"human", "user"}:
            return _extract_text_from_message_content(message.get("content", ""))
    return ""


def _infer_region(text: str) -> str:
    """Infer a coarse dynamic region label from the request text."""

    if any(keyword in text for keyword in ("脸", "面部", "肤色", "皮肤", "脸部")):
        return "face and skin area"
    if any(keyword in text for keyword in ("头发", "发丝", "发型")):
        return "hair area"
    if any(keyword in text for keyword in ("裙", "衣服", "服装", "连衣裙", "婚纱", "外套")):
        return "clothing area"
    if "背景" in text:
        return "background area"
    if "人物" in text or "人像" in text or "模特" in text:
        return "person area"
    if "主体" in text:
        return "subject area"
    return "whole_image"


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether the request contains any keyword."""

    return any(keyword in text for keyword in keywords)


def _estimate_strength(text: str) -> float:
    """Map qualitative adverbs to a conservative default strength."""

    if any(word in text for word in ("轻微", "稍微", "一点", "自然")):
        return 0.15
    if any(word in text for word in ("明显", "加强", "增强")):
        return 0.35
    if any(word in text for word in ("强烈", "大幅", "很")):
        return 0.55
    return 0.25


def _append_package_request(
    requests: list[dict[str, Any]],
    *,
    op: str,
    region: str,
    strength: float,
    params: dict[str, Any] | None = None,
) -> None:
    """Append a coarse package request if the same op+region is not already present."""

    if any(item["op"] == op and item.get("region", "whole_image") == region for item in requests):
        return
    requests.append(
        {
            "op": op,
            "region": region,
            "strength": strength,
            "params": params or {},
        },
    )


def _infer_requested_packages(text: str) -> list[dict[str, Any]]:
    """Infer coarse package requests from user wording.

    这里只做关键词级别的最小解析，为下一步 build_plan 提供稳定输入；
    真正的参数填充和排序仍然由 build_plan 负责。
    """

    region = _infer_region(text)
    strength = _estimate_strength(text)
    requests: list[dict[str, Any]] = []

    for op_name, keywords in PARSE_REQUEST_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            _append_package_request(
                requests,
                op=op_name,
                region="whole_image" if op_name in WHOLE_IMAGE_ONLY_TOOL_NAMES else region,
                strength=strength,
            )

    if _contains_any(text, ("逆光", "背光")):
        _append_package_request(
            requests,
            op="adjust_exposure",
            region="backlit subject area",
            strength=max(strength, 0.3),
        )
        _append_package_request(
            requests,
            op="adjust_highlights_shadows",
            region="backlit subject area",
            strength=max(strength, 0.24),
        )
        _append_package_request(
            requests,
            op="adjust_curves",
            region="whole_image",
            strength=0.2,
            params={
                "shadow_lift": 0.06,
                "midtone_gamma": 0.97,
                "highlight_compress": 0.08,
                "contrast_bias": 0.08,
            },
        )

    if _contains_any(text, ("夏日", "夏天", "夏日感", "阳光感", "清透", "通透", "空气感", "明媚")):
        _append_package_request(
            requests,
            op="adjust_white_balance",
            region="whole_image",
            strength=0.16,
        )
        _append_package_request(
            requests,
            op="adjust_vibrance_saturation",
            region="whole_image",
            strength=max(strength, 0.18),
        )
        _append_package_request(
            requests,
            op="adjust_dehaze",
            region="background area",
            strength=0.14,
            params={"amount": 0.14, "feather_radius": 22.0},
        )

    return requests


def _infer_constraints(text: str) -> list[str]:
    """Infer high-level planning constraints from the request."""

    constraints: list[str] = []
    if any(word in text for word in ("自然", "不要过度", "别太过")):
        constraints.append("avoid_overediting")
    if "保留" in text and any(word in text for word in ("主体", "人物", "人像", "肤色")):
        constraints.append("preserve_subject")
    if _contains_any(text, ("逆光", "背光")):
        constraints.append("repair_backlighting")
    if _contains_any(text, ("夏日", "夏天", "夏日感", "阳光感", "清透", "通透", "空气感", "明媚")):
        constraints.append("build_summer_mood")
    if _contains_any(text, ("参考", "像第二张", "像参考图", "同款", "一样的感觉")):
        constraints.append("match_reference_style")

    has_repair_goal = _contains_any(text, ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色"))
    has_style_goal = _contains_any(text, ("质感", "氛围", "色调", "夏日", "通透", "空气感", "胶片", "明媚"))
    if has_repair_goal and has_style_goal:
        constraints.append("needs_layered_refinement")
    return constraints


def parse_request(state: EditState) -> dict:
    """Determine request mode and extract a planner-friendly request intent."""

    request_text = _extract_latest_user_text(state)
    package_catalog = [
        PackageCatalogItem.model_validate(item).model_dump(mode="json")
        for item in state.get("package_catalog", [])
    ]

    if parse_request_model_available() and request_text:
        validated_intent = generate_request_intent_with_qwen(
            request_text=request_text,
            package_catalog=package_catalog,
        )
        return {
            "request_text": request_text,
            "mode": state.get("mode", validated_intent.mode),
            "request_intent": validated_intent.model_dump(mode="json"),
        }

    auto_markers = ("自动", "你看着修", "帮我修", "随便修", "auto")
    explicit_requests = _infer_requested_packages(request_text)
    mode = "auto" if not request_text or any(marker in request_text for marker in auto_markers) else "explicit"

    constraints = _infer_constraints(request_text)

    validated_intent = RequestIntent(
        mode=mode,
        requested_packages=[
            RequestPackageHint.model_validate(request)
            for request in explicit_requests
        ],
        constraints=constraints,
    )

    return {
        "request_text": request_text,
        "mode": state.get("mode", mode),
        "request_intent": validated_intent.model_dump(mode="json"),
    }
