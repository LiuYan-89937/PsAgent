"""Common tool contracts and planner schema helpers shared by native tools."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


MaskPolicy = Literal["none", "optional", "required"]
RiskLevel = Literal["low", "medium", "high"]
RegionExecutionMode = Literal["whole_image", "masked_region"]

WHOLE_IMAGE_REGION: RegionExecutionMode = "whole_image"
MASKED_REGION_MODE: RegionExecutionMode = "masked_region"


class ToolSpec(BaseModel):
    """Static declaration for one native tool."""

    # 这层是“系统认识工具”的公共 contract。
    # 以后无论 planner、API 还是前端，都尽量围绕这些字段协作。
    name: str
    label: str
    description: str
    family: str
    stage_affinity: list[str] = Field(default_factory=list)
    supports_mask: bool = False
    requires_mask: bool = False
    supports_whole_image: bool = True
    recommended_mask_prompt: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    planner_schema: dict[str, Any] = Field(default_factory=dict)
    primary_param: str = "strength"
    risk_level: RiskLevel = "low"
    status_label: str = ""
    keywords: tuple[str, ...] = ()


class ToolExecutionResult(BaseModel):
    """Standard execution result returned by native tools."""

    # 所有工具都统一返回这一个结果 shape，
    # 这样 ToolNode、trace、API 不需要再为每个工具写分支。
    ok: bool
    tool: str
    output_image: str | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    error: str | None = None


def execution_modes_for_spec(spec: ToolSpec) -> list[RegionExecutionMode]:
    """Return the runtime execution modes implied by a tool spec."""

    # 外部目录里仍然沿用 supported_regions 这类字段，
    # 这里统一把 spec 上的布尔能力翻译成可展示的执行模式列表。
    modes: list[RegionExecutionMode] = []
    if spec.supports_whole_image:
        modes.append(WHOLE_IMAGE_REGION)
    if spec.supports_mask:
        modes.append(MASKED_REGION_MODE)
    return modes


def merge_object_schemas(primary: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge two object schemas without mutating either input."""

    merged: dict[str, Any] = dict(primary)
    merged.setdefault("type", "object")
    merged_properties = dict(primary.get("properties") or {})
    merged_properties.update(extra.get("properties") or {})
    merged["properties"] = merged_properties

    required: list[str] = []
    for source_list in (primary.get("required") or [], extra.get("required") or []):
        for item in source_list:
            if item not in required:
                required.append(item)
    if required:
        merged["required"] = required

    if "description" not in merged and extra.get("description"):
        merged["description"] = extra["description"]
    if "title" not in merged and extra.get("title"):
        merged["title"] = extra["title"]
    return merged


def strip_object_schema_fields(schema: dict[str, Any], *, excluded_fields: set[str]) -> dict[str, Any]:
    """Return a copy of an object schema with selected fields removed."""

    sanitized = dict(schema)
    properties = dict(schema.get("properties") or {})
    for field_name in excluded_fields:
        properties.pop(field_name, None)
    sanitized["properties"] = properties

    required = [item for item in list(schema.get("required") or []) if item not in excluded_fields]
    if required:
        sanitized["required"] = required
    else:
        sanitized.pop("required", None)
    return sanitized


def build_planner_schema(
    tool: BaseTool,
    *,
    supports_mask: bool,
    mask_schema: dict[str, Any] | None = None,
    excluded_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Build planner-visible schema from a native @tool object."""

    # 先取 @tool 自己的输入 schema，
    # 再剔除 runtime-only 参数，并按需要合并共享 mask schema。
    tool_schema = tool.get_input_schema().model_json_schema()
    if excluded_fields:
        tool_schema = strip_object_schema_fields(tool_schema, excluded_fields=excluded_fields)
    if not supports_mask or not mask_schema:
        return tool_schema
    return merge_object_schemas(tool_schema, mask_schema)
