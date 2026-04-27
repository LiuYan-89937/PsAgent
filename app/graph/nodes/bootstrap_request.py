"""Bootstrap the request text before the main edit graph starts."""

from __future__ import annotations

from langgraph.config import get_stream_writer

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import EditState
from app.services.auto_instruction_model import (
    auto_instruction_model_available,
    generate_auto_beautify_instruction,
)


AUTO_BEAUTIFY_FALLBACK_INSTRUCTION = (
    "请先保留原图已有的光线、影调、主体关系和氛围优势，只修正最影响观感的问题。"
    "以自然克制的智能美化为目标，改善主体可读性、基础层次、肤色和颜色干净度，"
    "避免默认大幅提亮、加柔光、制造奶白灰雾、抬灰暗部或过度风格化。"
)


def _safe_stream_writer():
    """Return a stream writer when running inside LangGraph, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def bootstrap_request(state: EditState) -> dict[str, object]:
    """Resolve the effective request text inside graph bootstrap."""

    writer = _safe_stream_writer()
    raw_instruction = str(state.get("request_text") or "").strip()
    input_images = list(state.get("input_images") or [])
    fallback_trace = list(state.get("fallback_trace") or [])

    if raw_instruction:
        writer(
            {
                "event": "bootstrap_finished",
                "node": "bootstrap_request",
                "message": "已使用用户输入的修图需求",
            }
        )
        return {"request_text": raw_instruction, "fallback_trace": fallback_trace}

    image_path = input_images[0] if input_images else None
    if image_path and auto_instruction_model_available():
        writer(
            {
                "event": "bootstrap_started",
                "node": "bootstrap_request",
                "message": "正在生成智能美化提示词",
            }
        )
        try:
            resolved_request = generate_auto_beautify_instruction(image_path=image_path)
            writer(
                {
                    "event": "bootstrap_finished",
                    "node": "bootstrap_request",
                    "message": "智能美化提示词已生成",
                }
            )
            return {"request_text": resolved_request, "fallback_trace": fallback_trace}
        except RuntimeError as error:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                round_id=None,
                focus=None,
                source="auto_instruction_model",
                location="request_text",
                strategy="generic_auto_instruction",
                message="自动美化提示词生成失败，改用通用美化提示词。",
                error=str(error),
            )
            writer(
                {
                    "event": "bootstrap_failed",
                    "node": "bootstrap_request",
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
            round_id=None,
            focus=None,
            source="auto_instruction_model",
            location="request_text",
            strategy="generic_auto_instruction",
            message="自动美化提示词模型不可用，改用通用美化提示词。",
            error=None,
        )
        writer(
            {
                "event": "bootstrap_finished",
                "node": "bootstrap_request",
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
