"""Shared helpers for native tool wrappers."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.common.contracts import ToolExecutionResult


def temp_output_path(prefix: str) -> str:
    """Build a temporary PNG output path for deterministic edits."""

    return tempfile.mktemp(prefix=prefix, suffix=".png")


def require_mask_path(tool_name: str, mask_path: str | None, *, recommended_prompt: str | None = None) -> None:
    """Raise when a local-only tool is invoked without a runtime mask."""

    if mask_path:
        return
    prompt_hint = f" Generate or pass a '{recommended_prompt}' mask first." if recommended_prompt else ""
    raise ValueError(f"{tool_name} requires mask_path for local-only execution.{prompt_hint}")


def build_result(
    *,
    tool_name: str,
    output_image: str,
    applied_params: dict[str, Any],
    image_path: str,
    mask_path: str | None = None,
    warnings: list[str] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Build a normalized ToolExecutionResult payload."""

    payload_artifacts = {
        "input_image": image_path,
        "mask_path": mask_path,
    }
    if artifacts:
        payload_artifacts.update(artifacts)

    return ToolExecutionResult(
        ok=True,
        tool=tool_name,
        output_image=output_image,
        applied_params=applied_params,
        warnings=list(warnings or []),
        artifacts=payload_artifacts,
    ).model_dump(mode="json")
