"""Centralized parser metadata derived from the native tool catalog."""

from __future__ import annotations

from app.tools.catalog import TOOL_SPECS


WHOLE_IMAGE_ONLY_TOOL_NAMES = frozenset(
    spec.name for spec in TOOL_SPECS if spec.supports_whole_image and not spec.supports_mask
)
PARSE_REQUEST_KEYWORDS = tuple((spec.name, tuple(spec.keywords)) for spec in TOOL_SPECS if spec.keywords)


def validate_tool_name(name: str) -> str:
    """Validate that a planner-facing tool name is registered."""

    if not any(spec.name == name for spec in TOOL_SPECS):
        raise ValueError(f"Unsupported tool name: {name}")
    return name
