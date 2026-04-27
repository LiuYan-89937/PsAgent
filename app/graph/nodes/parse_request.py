"""Parse user instruction and choose explicit or auto mode."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import EditState, RequestGoal, RequestIntent, RequestToolHint, ToolCatalogItem
from app.tools import PARSE_REQUEST_KEYWORDS, WHOLE_IMAGE_ONLY_TOOL_NAMES
from app.services.parse_request_model import (
    generate_request_intent,
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


def _has_explicit_style_goal(text: str) -> bool:
    """Return whether the request clearly asks for stylized finishing."""

    # “氛围”“质感”在智能美化提示词里经常只是保留原图气质，
    # 不能直接推断为要加柔光、胶片或强滤镜。
    explicit_style_keywords = (
        "胶片",
        "复古",
        "电影感",
        "电影色",
        "梦幻",
        "柔光",
        "发光",
        "强滤镜",
        "风格化",
        "小红书",
        "日系",
        "港风",
        "赛博",
        "ins风",
        "同款风格",
        "同款色调",
    )
    if _contains_any(text, explicit_style_keywords):
        return True
    return "色调" in text and _contains_any(text, ("统一成", "调成", "改成", "偏冷", "偏暖"))


def _estimate_strength(text: str) -> float:
    """Map qualitative adverbs to a more visible default strength."""

    if any(word in text for word in ("轻微", "稍微", "一点", "自然")):
        return 0.22
    if any(word in text for word in ("明显", "加强", "增强")):
        return 0.45
    if any(word in text for word in ("强烈", "大幅", "很")):
        return 0.68
    return 0.32


def _append_tool_request(
    requests: list[dict[str, Any]],
    *,
    op: str,
    region: str,
    strength: float,
    params: dict[str, Any] | None = None,
) -> None:
    """Append a coarse tool request if the same op+region is not already present."""

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


def _infer_requested_tools(text: str) -> list[dict[str, Any]]:
    """Infer explicit tool requests only when the user names a concrete tool."""

    region = _infer_region(text)
    strength = _estimate_strength(text)
    requests: list[dict[str, Any]] = []

    for op_name, keywords in PARSE_REQUEST_KEYWORDS:
        # 普通自然语言不要直接硬映射成工具。
        # 只有用户明确写出工具函数名，或者表达“用 XXX 工具”时，才保留 requested_tools。
        explicitly_named = op_name in text
        explicitly_asked_tool = any(f"用{keyword}工具" in text or f"{keyword}工具" in text for keyword in keywords)
        if explicitly_named or explicitly_asked_tool:
            _append_tool_request(
                requests,
                op=op_name,
                region="whole_image" if op_name in WHOLE_IMAGE_ONLY_TOOL_NAMES else region,
                strength=strength,
            )

    return requests


def _append_goal(
    goals: list[dict[str, Any]],
    *,
    kind: str,
    target_region: str,
    priority: int,
    intensity: float | None = None,
    constraints: list[str] | None = None,
) -> None:
    """Append a goal-level request without duplicating kind+region."""

    if any(item["kind"] == kind and item.get("target_region") == target_region for item in goals):
        return
    goals.append(
        {
            "kind": kind,
            "target_region": target_region,
            "priority": priority,
            "intensity": intensity,
            "constraints": list(constraints or []),
            "source": "heuristic",
        }
    )


def _infer_goals(text: str) -> list[dict[str, Any]]:
    """Infer goal-level editing intent without binding it to concrete tools."""

    region = _infer_region(text)
    strength = _estimate_strength(text)
    goals: list[dict[str, Any]] = []

    if not text:
        _append_goal(goals, kind="auto_enhance", target_region="whole_image", priority=50, intensity=0.28)
        return goals

    if _contains_any(text, ("提亮", "偏暗", "变亮", "亮一点", "逆光", "背光")):
        target_region = region if region != "whole_image" else ("person area" if _contains_any(text, ("逆光", "背光")) else "whole_image")
        _append_goal(
            goals,
            kind="lift_luminance",
            target_region=target_region,
            priority=82,
            intensity=max(strength, 0.32),
            constraints=["protect_highlights"],
        )
    if _contains_any(text, ("压高光", "高光", "过曝", "太亮")):
        _append_goal(
            goals,
            kind="recover_highlights",
            target_region=region,
            priority=78,
            intensity=max(strength, 0.3),
            constraints=["protect_detail"],
        )
    if _contains_any(text, ("对比", "层次", "灰", "发灰", "低反差")):
        _append_goal(
            goals,
            kind="improve_tonal_separation",
            target_region=region,
            priority=68,
            intensity=strength,
            constraints=["avoid_crushed_shadows"],
        )
    if _contains_any(text, ("色彩", "饱和", "颜色", "色调", "偏色")):
        _append_goal(
            goals,
            kind="balance_color",
            target_region=region,
            priority=64,
            intensity=strength,
            constraints=["preserve_skin_tone"] if _contains_any(text, ("肤色", "皮肤", "人像")) else [],
        )
    if _contains_any(text, ("肤色", "皮肤", "脸色", "面部")):
        _append_goal(
            goals,
            kind="natural_skin_tone",
            target_region="face and skin area",
            priority=86,
            intensity=min(max(strength, 0.22), 0.45),
            constraints=["avoid_over_smoothing", "preserve_identity"],
        )
    if _contains_any(text, ("背景",)):
        _append_goal(
            goals,
            kind="background_balance",
            target_region="background area",
            priority=58,
            intensity=strength,
            constraints=["do_not_affect_subject"],
        )
    if _contains_any(text, ("噪点", "降噪", "颗粒太多")):
        _append_goal(goals, kind="reduce_noise", target_region=region, priority=70, intensity=strength)
    if _contains_any(text, ("清晰", "锐", "细节", "质感")):
        _append_goal(
            goals,
            kind="improve_detail",
            target_region=region,
            priority=62,
            intensity=strength,
            constraints=["avoid_halo"],
        )

    if not goals:
        _append_goal(goals, kind="auto_enhance", target_region="whole_image", priority=45, intensity=0.25)
    return goals


def _infer_constraints(text: str) -> list[str]:
    """Infer high-level planning constraints from the request."""

    constraints: list[str] = []
    if any(word in text for word in ("自然", "不要过度", "别太过")):
        constraints.append("avoid_overediting")
    if "保留" in text and any(word in text for word in ("原图", "光线", "影调", "氛围", "黑位", "暗背景", "逆光")):
        constraints.append("preserve_original_mood")
    if "保留" in text and any(word in text for word in ("主体", "人物", "人像", "肤色")):
        constraints.append("preserve_subject")
    if _contains_any(text, ("逆光", "背光")):
        constraints.append("repair_backlighting")
    if _contains_any(text, ("参考", "像第二张", "像参考图", "同款", "一样的感觉")):
        constraints.append("match_reference_style")

    has_repair_goal = _contains_any(text, ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色"))
    has_style_goal = _has_explicit_style_goal(text)
    if has_repair_goal and has_style_goal:
        constraints.append("needs_layered_refinement")
    return constraints


def _infer_goal_summary(text: str) -> str:
    """Build a short goal summary for downstream planning."""

    normalized = " ".join(text.strip().split())
    if not normalized:
        return "智能美化并提升整体观感"
    return normalized[:120]


def _stabilize_request_intent(intent: RequestIntent, request_text: str) -> RequestIntent:
    """Apply deterministic guardrails to model-produced request intent."""

    # 模型有时会把“保留氛围/保留影调”误读成风格化诉求。
    # 这里用硬规则兜住：只有明确风格词才允许 wants_style=true。
    updates: dict[str, Any] = {}
    constraints = list(intent.constraints)
    inferred_constraints = _infer_constraints(request_text)
    for constraint in inferred_constraints:
        if constraint not in constraints:
            constraints.append(constraint)
    if intent.wants_style and not _has_explicit_style_goal(request_text):
        updates["wants_style"] = False
    if constraints != list(intent.constraints):
        updates["constraints"] = constraints
    return intent.model_copy(update=updates) if updates else intent


def _infer_requires_local_editing(text: str, requested_tools: list[dict[str, Any]]) -> bool:
    """Return whether the request clearly needs local operations."""

    if any((item.get("region") or "whole_image") != "whole_image" for item in requested_tools):
        return True
    return _contains_any(text, ("脸", "面部", "肤色", "皮肤", "头发", "发丝", "背景", "主体", "人物", "人像"))


def parse_request(state: EditState) -> dict:
    """Determine request mode and extract a planner-friendly request intent."""

    request_text = _extract_latest_user_text(state)
    tool_catalog = [
        ToolCatalogItem.model_validate(item).model_dump(mode="json")
        for item in state.get("tool_catalog", [])
    ]

    if parse_request_model_available() and request_text:
        try:
            validated_intent = generate_request_intent(
                request_text=request_text,
                tool_catalog=tool_catalog,
            )
            validated_intent = _stabilize_request_intent(validated_intent, request_text)
            return {
                "request_text": request_text,
                "mode": state.get("mode", validated_intent.mode),
                "request_intent": validated_intent.model_dump(mode="json"),
            }
        except (RuntimeError, ValidationError) as error:
            fallback_trace = append_fallback_trace(
                state.get("fallback_trace"),
                round_id=None,
                focus=None,
                source="parse_request_model",
                location="request_intent",
                strategy="heuristic_request_intent",
                message="需求理解模型输出无效，改用规则归一化。",
                error=str(error),
            )
        else:
            fallback_trace = state.get("fallback_trace", [])
    else:
        fallback_trace = list(state.get("fallback_trace", []))
        if request_text:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                round_id=None,
                focus=None,
                source="parse_request_model",
                location="request_intent",
                strategy="heuristic_request_intent",
                message="需求理解模型不可用，改用规则归一化。",
                error=None,
            )

    auto_markers = ("自动", "你看着修", "帮我修", "随便修", "auto")
    explicit_requests = _infer_requested_tools(request_text)
    goals = _infer_goals(request_text)
    mode = "auto" if not request_text or any(marker in request_text for marker in auto_markers) else "explicit"

    constraints = _infer_constraints(request_text)

    validated_intent = RequestIntent(
        mode=mode,
        goals=[
            RequestGoal.model_validate(goal)
            for goal in goals
        ],
        requested_tools=[
            RequestToolHint.model_validate(request)
            for request in explicit_requests
        ],
        constraints=constraints,
        goal_summary=_infer_goal_summary(request_text),
        wants_repair=_contains_any(request_text, ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色", "降噪", "去瑕疵")),
        wants_style=_has_explicit_style_goal(request_text),
        requires_local_editing=_infer_requires_local_editing(request_text, explicit_requests),
    )

    return {
        "request_text": request_text,
        "mode": state.get("mode", mode),
        "request_intent": validated_intent.model_dump(mode="json"),
        "fallback_trace": fallback_trace,
    }
