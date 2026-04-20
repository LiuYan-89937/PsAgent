"""Centralized tool metadata derived from the native tool registry."""

from __future__ import annotations

from dataclasses import dataclass

from app.tools.tool_registry import build_default_tool_registry


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Normalized metadata for one planner-visible tool."""

    name: str
    status_label: str
    keywords: tuple[str, ...] = ()
    macro: bool = False
    whole_image_only: bool = False


def _all_tool_metadata() -> tuple[ToolMetadata, ...]:
    """Build immutable metadata records from the native registry."""

    metadata: list[ToolMetadata] = []
    for registered in build_default_tool_registry().list():
        spec = registered.spec
        metadata.append(
            ToolMetadata(
                name=spec.name,
                status_label=spec.status_label or f"正在执行 {spec.label}",
                keywords=tuple(spec.keywords),
                macro=False,
                whole_image_only=spec.supports_whole_image and not spec.supports_mask,
            )
        )
    return tuple(metadata)


ALL_TOOL_METADATA: tuple[ToolMetadata, ...] = _all_tool_metadata()
TOOL_METADATA_BY_NAME = {item.name: item for item in ALL_TOOL_METADATA}
ALL_TOOL_NAMES: tuple[str, ...] = tuple(item.name for item in ALL_TOOL_METADATA)
MACRO_TOOL_NAMES = frozenset(item.name for item in ALL_TOOL_METADATA if item.macro)
WHOLE_IMAGE_ONLY_TOOL_NAMES = frozenset(item.name for item in ALL_TOOL_METADATA if item.whole_image_only)
TOOL_STATUS_LABELS = {item.name: item.status_label for item in ALL_TOOL_METADATA}
PARSE_REQUEST_KEYWORDS = tuple((item.name, item.keywords) for item in ALL_TOOL_METADATA if item.keywords)


def validate_tool_name(name: str) -> str:
    """Validate that a planner-facing tool name is registered."""

    if build_default_tool_registry().get(name) is None:
        raise ValueError(f"Unsupported tool name: {name}")
    return name
