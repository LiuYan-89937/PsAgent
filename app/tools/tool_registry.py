"""Registry for native @tool tools and their ToolSpec metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool

from app.tools.native_tools import adjust_contrast, adjust_exposure, adjust_vibrance_saturation
from app.tools.tool_specs import (
    ToolSpec,
    build_planner_schema,
    execution_modes_for_spec,
)


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Couple a ToolSpec with its LangChain tool callable."""

    spec: ToolSpec
    tool: BaseTool


class ToolRegistry:
    """Collect and export native tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, tool: BaseTool) -> None:
        """Register a native tool with its spec."""

        planner_schema = spec.planner_schema or build_planner_schema(
            tool,
            supports_mask=spec.supports_mask,
            excluded_fields={"image_path", "mask_path"},
        )
        registered = RegisteredTool(
            spec=spec.model_copy(update={"planner_schema": planner_schema}),
            tool=tool,
        )
        self._tools[registered.spec.name] = registered

    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered tool by name if present."""

        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        """Return a registered tool or raise a clear error."""

        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool

    def list(self) -> list[RegisteredTool]:
        """Return registered tools in insertion order."""

        return list(self._tools.values())

    def export_catalog(self) -> list[dict[str, Any]]:
        """Return the planner/API-visible tool catalog."""

        items: list[dict[str, Any]] = []
        for registered in self.list():
            spec = registered.spec
            mask_policy = "optional" if spec.supports_mask and spec.supports_whole_image else (
                "required" if spec.supports_mask else "none"
            )
            items.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "description": spec.description,
                    "family": spec.family,
                    "stage_affinity": list(spec.stage_affinity),
                    "supports_mask": spec.supports_mask,
                    "supports_whole_image": spec.supports_whole_image,
                    "default_params": dict(spec.default_params),
                    "planner_schema": spec.planner_schema,
                    "primary_param": spec.primary_param,
                    "supported_regions": execution_modes_for_spec(spec),
                    "mask_policy": mask_policy,
                    "supported_domains": ["general"],
                    "risk_level": spec.risk_level,
                    "params_schema": spec.planner_schema,
                }
            )
        return items


@lru_cache(maxsize=1)
def build_default_tool_registry() -> ToolRegistry:
    """Build the default native tool registry for the current app version."""

    registry = ToolRegistry()
    all_stages = [
        "technical_prep",
        "global_base",
        "local_balance",
        "subject_refine",
        "finish_output",
    ]
    registry.register(
        ToolSpec(
            name="adjust_exposure",
            label="曝光",
            description="Adjust whole-image or masked exposure with stable strength semantics.",
            family="tone",
            stage_affinity=all_stages,
            supports_mask=True,
            supports_whole_image=True,
            default_params={
                "strength": 0.5,
                "max_stops": 2.0,
                "feather_radius": 18.0,
            },
            primary_param="strength",
            risk_level="low",
            status_label="正在调整曝光",
            keywords=("曝光", "提亮", "压暗", "变亮", "变暗"),
        ),
        adjust_exposure,
    )
    registry.register(
        ToolSpec(
            name="adjust_contrast",
            label="对比度",
            description="Adjust whole-image or masked contrast around a controlled luminance pivot.",
            family="tone",
            stage_affinity=all_stages,
            supports_mask=True,
            supports_whole_image=True,
            default_params={
                "strength": 0.5,
                "contrast_scale": 1.0,
                "pivot": 0.5,
                "feather_radius": 18.0,
            },
            primary_param="strength",
            risk_level="low",
            status_label="正在调整对比度",
            keywords=("对比度", "层次", "反差"),
        ),
        adjust_contrast,
    )
    registry.register(
        ToolSpec(
            name="adjust_vibrance_saturation",
            label="自然饱和度",
            description="Adjust whole-image or masked vibrance and saturation in color space while protecting highlights, skin, and shadows.",
            family="color",
            stage_affinity=all_stages,
            supports_mask=True,
            supports_whole_image=True,
            default_params={
                "strength": 0.5,
                "vibrance_scale": 0.8,
                "saturation_scale": 0.28,
                "protect_highlights": 0.26,
                "protect_skin": 0.34,
                "protect_shadows": 0.24,
                "feather_radius": 18.0,
            },
            primary_param="strength",
            risk_level="low",
            status_label="正在调整色彩",
            keywords=("饱和度", "色彩", "鲜艳", "自然饱和度", "vibrance"),
        ),
        adjust_vibrance_saturation,
    )
    return registry


@lru_cache(maxsize=1)
def build_default_tool_node() -> ToolNode:
    """Build the shared ToolNode backed by the default native tools."""

    registry = build_default_tool_registry()
    return ToolNode([item.tool for item in registry.list()])


def exported_tool_names() -> tuple[str, ...]:
    """Return registered tool names in a stable tuple."""

    return tuple(item.spec.name for item in build_default_tool_registry().list())


__all__ = [
    "RegisteredTool",
    "ToolRegistry",
    "build_default_tool_node",
    "build_default_tool_registry",
    "exported_tool_names",
]
