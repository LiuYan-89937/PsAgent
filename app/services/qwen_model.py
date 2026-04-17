"""Shared DashScope-compatible Qwen helpers for text and vision calls."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error, request

from app.services.env import load_project_env
from PIL import Image


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL = "qwen3.6-plus"
DEFAULT_VISION_MODEL = "qwen3.6-plus"
DEFAULT_CRITIC_MODEL = "qwen3.6-plus"
DEFAULT_DASHSCOPE_TIMEOUT_SECONDS = 300


def qwen_model_available() -> bool:
    """Return whether DashScope credentials are configured."""

    load_project_env()
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from `app/prompts/`."""

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / prompt_name
    return prompt_path.read_text(encoding="utf-8").strip()


def strip_json_fence(text: str) -> str:
    """Strip markdown fences when the model wraps JSON in a code block."""

    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return content


def extract_message_text(message: dict[str, Any]) -> str:
    """Normalize OpenAI-compatible message content into a text string."""

    content = message.get("content", "")
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
    """Encode a local image into a compact Data URL for vision calls.

    为了避免多图 critic 调用超时，这里会先做两件事：
    1. 长边限制在 `max_side`；
    2. 默认压成高质量 JPEG，明显减少 base64 体积。
    """

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
        rgb_image = resized.convert("RGB")
        rgb_image.save(buffer, format="JPEG", quality=88, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_user_content(
    *,
    user_payload: dict[str, Any],
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal OpenAI-compatible user content."""

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


def call_qwen_chat_completion_raw(
    *,
    messages: list[dict[str, Any]],
    model_env_name: str,
    default_model: str,
    temperature: float = 0.1,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
) -> dict[str, Any]:
    """Call DashScope-compatible chat completions with raw messages."""

    load_project_env()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured.")

    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL).rstrip("/")
    model = os.getenv(model_env_name, default_model)
    timeout_seconds = float(os.getenv("DASHSCOPE_TIMEOUT_SECONDS", str(DEFAULT_DASHSCOPE_TIMEOUT_SECONDS)))

    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    req = request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DashScope request failed: {detail}") from exc
    except error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"DashScope request failed: {exc}") from exc


def call_qwen_chat_completion(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    model_env_name: str,
    default_model: str,
    image_paths: list[str] | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Call DashScope-compatible chat completions and return parsed JSON response payload."""

    return call_qwen_chat_completion_raw(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_multimodal_user_content(
                    user_payload=user_payload,
                    image_paths=image_paths,
                ),
            },
        ],
        model_env_name=model_env_name,
        default_model=default_model,
        temperature=temperature,
        response_format={"type": "json_object"},
    )


def call_qwen_for_tool_message(
    *,
    prompt_name: str,
    user_payload: dict[str, Any],
    model_env_name: str,
    default_model: str,
    tools: list[dict[str, Any]],
    image_paths: list[str] | None = None,
    temperature: float = 0.1,
    tool_choice: Any = "required",
) -> dict[str, Any]:
    """Call Qwen in tool-calling mode and return the assistant message."""

    response_payload = call_qwen_chat_completion_raw(
        messages=[
            {"role": "system", "content": load_prompt(prompt_name)},
            {
                "role": "user",
                "content": build_multimodal_user_content(
                    user_payload=user_payload,
                    image_paths=image_paths,
                ),
            },
        ],
        model_env_name=model_env_name,
        default_model=default_model,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
    )

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("DashScope returned no choices.")

    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError("DashScope returned an invalid assistant message.")
    return message


def call_qwen_for_json(
    *,
    prompt_name: str,
    user_payload: dict[str, Any],
    model_env_name: str,
    default_model: str,
    image_paths: list[str] | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Call Qwen and parse the top-level message content as JSON."""

    response_payload = call_qwen_chat_completion(
        system_prompt=load_prompt(prompt_name),
        user_payload=user_payload,
        model_env_name=model_env_name,
        default_model=default_model,
        image_paths=image_paths,
        temperature=temperature,
    )

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("DashScope returned no choices.")

    message = choices[0].get("message") or {}
    content = strip_json_fence(extract_message_text(message))
    if not content:
        raise RuntimeError("DashScope returned empty content.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON: {content}") from exc
