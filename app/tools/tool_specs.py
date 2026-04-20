"""Native tool specs, planner schema helpers, and mask parameter definitions."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MaskPolicy = Literal["none", "optional", "required"]
RiskLevel = Literal["low", "medium", "high"]
RegionExecutionMode = Literal["whole_image", "masked_region"]
MaskProvider = Literal["aliyun", "fal_sam3"]

WHOLE_IMAGE_REGION: RegionExecutionMode = "whole_image"
MASKED_REGION_MODE: RegionExecutionMode = "masked_region"

MASK_PARAM_TO_RUNTIME_KEY = {
    "mask_provider": "provider",
    "mask_prompt": "prompt",
    "mask_negative_prompt": "negative_prompt",
    "mask_semantic_type": "semantic_type",
    "mask_fill_holes": "fill_holes",
    "mask_expand": "expand_mask",
    "mask_blur": "blur_mask",
    "mask_use_grounding_dino": "use_grounding_dino",
    "mask_revert": "revert_mask",
    "mask_start_timeout_seconds": "start_timeout_seconds",
    "mask_client_timeout_seconds": "client_timeout_seconds",
}
MASK_PARAM_KEYS = frozenset(MASK_PARAM_TO_RUNTIME_KEY)


class MaskParams(BaseModel):
    """Shared planner-facing mask parameters handled by the stage runner."""

    model_config = ConfigDict(extra="forbid")

    mask_provider: MaskProvider | None = Field(
        default=None,
        description="Optional segmentation backend override.",
    )
    mask_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Single visible English subject term for segmentation, e.g. face, hair, dress, background.",
    )
    mask_negative_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Optional exclusion prompt for text-guided segmentation.",
    )
    mask_semantic_type: bool | None = Field(
        default=None,
        description="Hint that the target is semantic rather than purely visual.",
    )
    mask_fill_holes: bool | None = Field(
        default=None,
        description="Whether to fill holes in the generated mask.",
    )
    mask_expand: int | None = Field(
        default=None,
        ge=0,
        le=64,
        description="Optional mask expansion radius in pixels.",
    )
    mask_blur: bool | None = Field(
        default=None,
        description="Whether to blur the raw mask before binarization.",
    )
    mask_use_grounding_dino: bool | None = Field(
        default=None,
        description="Whether to enable GroundingDINO assistance when supported.",
    )
    mask_revert: bool | None = Field(
        default=None,
        description="Whether to invert the generated mask result.",
    )
    mask_start_timeout_seconds: float | None = Field(
        default=None,
        ge=10.0,
        le=600.0,
        description="Provider start timeout in seconds.",
    )
    mask_client_timeout_seconds: float | None = Field(
        default=None,
        ge=10.0,
        le=900.0,
        description="Provider client timeout in seconds.",
    )

    @field_validator("mask_prompt", "mask_negative_prompt", mode="before")
    @classmethod
    def _normalize_prompt_text(cls, value: Any) -> Any:
        """Trim and normalize mask prompt whitespace."""

        if value is None or not isinstance(value, str):
            return value
        normalized = " ".join(value.strip().split())
        return normalized or None

    @model_validator(mode="after")
    def _validate_mask_constraints(self) -> "MaskParams":
        """Validate provider-specific mask constraints."""

        if self.mask_negative_prompt and not self.mask_prompt:
            raise ValueError("mask_negative_prompt requires mask_prompt.")
        if self.mask_use_grounding_dino and not self.mask_prompt:
            raise ValueError("mask_use_grounding_dino requires mask_prompt.")

        if self.mask_provider == "aliyun":
            unsupported = [
                field_name
                for field_name in (
                    "mask_prompt",
                    "mask_negative_prompt",
                    "mask_semantic_type",
                    "mask_fill_holes",
                    "mask_expand",
                    "mask_blur",
                    "mask_use_grounding_dino",
                    "mask_revert",
                )
                if getattr(self, field_name) not in (None, False, 0)
            ]
            if unsupported:
                joined = ", ".join(unsupported)
                raise ValueError(f"Aliyun segmentation does not support: {joined}")
        return self

    def to_runtime_options(self) -> dict[str, Any]:
        """Convert validated mask params into segmentation runtime kwargs."""

        payload = self.model_dump(exclude_none=True)
        return {
            runtime_key: payload[param_key]
            for param_key, runtime_key in MASK_PARAM_TO_RUNTIME_KEY.items()
            if param_key in payload
        }


MASK_PARAMS_SCHEMA = MaskParams.model_json_schema()


class ToolSpec(BaseModel):
    """Static declaration for one native tool."""

    name: str
    label: str
    description: str
    family: str
    stage_affinity: list[str] = Field(default_factory=list)
    supports_mask: bool = False
    supports_whole_image: bool = True
    default_params: dict[str, Any] = Field(default_factory=dict)
    planner_schema: dict[str, Any] = Field(default_factory=dict)
    primary_param: str = "strength"
    risk_level: RiskLevel = "low"
    status_label: str = ""
    keywords: tuple[str, ...] = ()


class ToolExecutionResult(BaseModel):
    """Standard execution result returned by native tools."""

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
    excluded_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Build the planner-visible schema from a native @tool plus ToolSpec flags."""

    tool_schema = tool.get_input_schema().model_json_schema()
    if excluded_fields:
        tool_schema = strip_object_schema_fields(tool_schema, excluded_fields=excluded_fields)
    if not supports_mask:
        return tool_schema
    return merge_object_schemas(tool_schema, MASK_PARAMS_SCHEMA)


def extract_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Pick only shared mask params from a merged tool params payload."""

    return {
        key: value
        for key, value in params.items()
        if key in MASK_PARAM_KEYS and value is not None and value != ""
    }


def strip_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove shared mask params from a merged tool params payload."""

    return {key: value for key, value in params.items() if key not in MASK_PARAM_KEYS}
