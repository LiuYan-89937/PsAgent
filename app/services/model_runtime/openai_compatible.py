"""LangChain runtime for OpenAI-compatible text and vision models."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from app.services.env import load_project_env


DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MODEL = "gpt-4o-mini"
DEFAULT_CRITIC_MODEL = "gpt-4o-mini"
DEFAULT_CANDIDATE_REVIEW_MODEL = "qwen3-vl-flash"
DEFAULT_TIMEOUT_SECONDS = 300.0


def model_available() -> bool:
    """Return whether OpenAI-compatible credentials are configured."""

    load_project_env()
    return bool(os.getenv("OPENAI_API_KEY"))


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from `app/prompts/`."""

    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / prompt_name
    return prompt_path.read_text(encoding="utf-8").strip()


def strip_json_fence(text: str) -> str:
    """Strip markdown fences when a model wraps JSON in a code block."""

    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return content


def extract_message_text(message: BaseMessage) -> str:
    """Normalize LangChain message content into a text string."""

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return ""


def encode_image_as_data_url(image_path: str, *, max_side: int = 1280) -> str:
    """Encode a local image into a compact data URL for vision model calls."""

    image_file = Path(image_path)
    image = Image.open(image_file)
    image.load()

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    else:
        resized = image

    buffer = BytesIO()
    if resized.mode in {"RGBA", "LA"}:
        mime_type = "image/png"
        resized.save(buffer, format="PNG", optimize=True)
    else:
        mime_type, _ = mimetypes.guess_type(image_file.name)
        if mime_type not in {"image/jpeg", "image/jpg"}:
            mime_type = "image/jpeg"
        resized.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_user_content(
    *,
    user_payload: dict[str, Any],
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal user content."""

    content: list[dict[str, Any]] = []
    for image_path in image_paths or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(image_path)},
            }
        )
    content.append(
        {
            "type": "text",
            "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    )
    return content


def build_chat_model(
    *,
    model_env_name: str,
    default_model: str,
    temperature: float = 0.1,
    enable_thinking: bool | None = None,
) -> ChatOpenAI:
    """Build a LangChain chat model for an OpenAI-compatible endpoint."""

    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    model = os.getenv(model_env_name) or os.getenv("OPENAI_MODEL") or default_model
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    extra_body = {"enable_thinking": enable_thinking} if enable_thinking is not None else None
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=temperature,
        timeout=timeout_seconds,
        max_retries=1,
        extra_body=extra_body,
    )


def invoke_chat(
    *,
    prompt_name: str,
    user_payload: dict[str, Any],
    model_env_name: str,
    default_model: str,
    image_paths: list[str] | None = None,
    temperature: float = 0.1,
    response_format: dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
) -> AIMessage:
    """Invoke a LangChain chat model and return the assistant message."""

    model = build_chat_model(
        model_env_name=model_env_name,
        default_model=default_model,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    runnable = model.bind(response_format=response_format) if response_format is not None else model
    result = runnable.invoke(
        [
            SystemMessage(content=load_prompt(prompt_name)),
            HumanMessage(content=build_multimodal_user_content(user_payload=user_payload, image_paths=image_paths)),
        ]
    )
    if not isinstance(result, AIMessage):
        return AIMessage(content=result.content)
    return result


def invoke_json(
    *,
    prompt_name: str,
    user_payload: dict[str, Any],
    model_env_name: str,
    default_model: str,
    image_paths: list[str] | None = None,
    temperature: float = 0.1,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Invoke a model and parse the response content as JSON."""

    message = invoke_chat(
        prompt_name=prompt_name,
        user_payload=user_payload,
        model_env_name=model_env_name,
        default_model=default_model,
        image_paths=image_paths,
        temperature=temperature,
        response_format={"type": "json_object"},
        enable_thinking=enable_thinking,
    )
    content = strip_json_fence(extract_message_text(message))
    if not content:
        raise RuntimeError("Model returned empty content.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON: {content}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output JSON is not an object.")
    return parsed
