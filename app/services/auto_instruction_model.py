"""Vision-model helper that generates a concise beautify instruction."""

from __future__ import annotations

from app.services.qwen_model import DEFAULT_VISION_MODEL, call_qwen_for_json, qwen_model_available


def auto_instruction_model_available() -> bool:
    """Return whether the auto-instruction model can be called."""

    return qwen_model_available()


def generate_auto_beautify_instruction_with_qwen(*, image_path: str) -> str:
    """Generate a concise beautify instruction from image content."""

    payload = call_qwen_for_json(
        prompt_name="auto_beautify.txt",
        user_payload={
            "任务": "请根据图片内容生成一段明确、结果导向、适合直接执行的中文美化提示词。",
            "补充要求": [
                "结果只描述目标效果，不要解释原因",
                "要强调明显、有效、结果导向，不要写成泛泛建议",
                "优先写最关键的 2 到 4 个改善目标",
                "如果图片明显有改进空间，要允许做更有力度的增强",
                "适合直接作为用户输入给修图代理",
                "只返回 JSON",
            ],
        },
        model_env_name="DASHSCOPE_REQUEST_MODEL",
        default_model=DEFAULT_VISION_MODEL,
        image_paths=[image_path],
        temperature=0.2,
    )

    instruction = payload.get("instruction") if isinstance(payload, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("Auto beautify prompt model did not return a valid instruction.")
    return instruction.strip()
