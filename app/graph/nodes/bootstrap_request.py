"""Bootstrap the request text before the main edit graph starts."""

from __future__ import annotations

from langgraph.config import get_stream_writer

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import EditState
from app.services.auto_instruction_model import (
    auto_instruction_model_available,
    generate_auto_beautify_instruction_with_qwen,
)


AUTO_BEAUTIFY_FALLBACK_INSTRUCTION = (
    "请把这张图明显往更好看的成片效果推进，优先改善主体表现、画面亮度、层次、通透感和整体色调，"
    "让结果更干净、更亮、更有质感，同时保持自然。"
)


def _safe_stream_writer():
    """Return a stream writer when running inside LangGraph, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def bootstrap_request(state: EditState) -> dict[str, object]:
    """Resolve the effective request text inside the graph bootstrap phase."""

    writer = _safe_stream_writer()
    raw_instruction = str(state.get("request_text") or "").strip()
    mode = str(state.get("mode") or "explicit")
    input_images = list(state.get("input_images") or [])
    fallback_trace = list(state.get("fallback_trace") or [])

    if raw_instruction and mode != "auto":
        writer(
            {
                "event": "bootstrap_finished",
                "stage": "bootstrap_request",
                "message": "已使用用户输入的修图需求",
            }
        )
        return {"request_text": raw_instruction, "fallback_trace": fallback_trace}

    image_path = input_images[0] if input_images else None
    if image_path and auto_instruction_model_available():
        writer(
            {
                "event": "bootstrap_started",
                "stage": "bootstrap_request",
                "message": "正在生成智能美化提示词",
            }
        )
        try:
            resolved_request = generate_auto_beautify_instruction_with_qwen(image_path=image_path)
            writer(
                {
                    "event": "bootstrap_finished",
                    "stage": "bootstrap_request",
                    "message": "智能美化提示词已生成",
                }
            )
            return {"request_text": resolved_request, "fallback_trace": fallback_trace}
        except RuntimeError as error:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage="bootstrap_request",
                source="auto_instruction_model",
                location="request_text",
                strategy="generic_auto_instruction",
                message="自动美化提示词生成失败，改用通用美化提示词。",
                error=str(error),
            )
            writer(
                {
                    "event": "bootstrap_failed",
                    "stage": "bootstrap_request",
                    "message": "智能美化提示词生成失败，已回退到通用提示词",
                    "error": str(error),
                }
            )
            return {
                "request_text": AUTO_BEAUTIFY_FALLBACK_INSTRUCTION,
                "fallback_trace": fallback_trace,
            }

    if image_path:
        fallback_trace = append_fallback_trace(
            fallback_trace,
            stage="bootstrap_request",
            source="auto_instruction_model",
            location="request_text",
            strategy="generic_auto_instruction",
            message="自动美化提示词模型不可用，改用通用美化提示词。",
            error=None,
        )
        writer(
            {
                "event": "bootstrap_finished",
                "stage": "bootstrap_request",
                "message": "智能美化模型不可用，已使用通用提示词",
            }
        )
        return {
            "request_text": AUTO_BEAUTIFY_FALLBACK_INSTRUCTION,
            "fallback_trace": fallback_trace,
        }

    return {
        "request_text": raw_instruction or AUTO_BEAUTIFY_FALLBACK_INSTRUCTION,
        "fallback_trace": fallback_trace,
    }
