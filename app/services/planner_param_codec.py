"""Shared planner-facing parameter codec for 0-100 integer slider values."""

from __future__ import annotations

import json
import re
from typing import Any

from app.tools.segmentation_tools import normalize_segmentation_prompt_label
from app.tools.tool_registry import build_default_tool_registry
from app.tools.tool_specs import MASK_PARAM_KEYS, WHOLE_IMAGE_REGION


def schema_non_null_variants(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return schema variants excluding explicit nulls."""

    variants = spec.get("anyOf") if isinstance(spec.get("anyOf"), list) else [spec]
    non_null = [item for item in variants if isinstance(item, dict) and item.get("type") != "null"]
    return non_null or [spec]


def schema_primary_type(spec: dict[str, Any]) -> str | None:
    """Return the first non-null schema type."""

    for variant in schema_non_null_variants(spec):
        variant_type = variant.get("type")
        if isinstance(variant_type, str):
            return variant_type
    variant_type = spec.get("type")
    return variant_type if isinstance(variant_type, str) else None


def schema_bound(spec: dict[str, Any], key: str) -> Any:
    """Read a numeric or length bound from the first schema variant that defines it."""

    for variant in schema_non_null_variants(spec):
        if key in variant:
            return variant[key]
    return spec.get(key)


def planner_integer_slider_description(spec: dict[str, Any]) -> str:
    """Build a planner-facing integer-slider description from the original schema."""

    description = str(spec.get("description") or "").strip()
    minimum = schema_bound(spec, "minimum")
    maximum = schema_bound(spec, "maximum")
    if minimum is not None and maximum is not None:
        return f"{description}；仅填 0-100 整数，0=最小值 {minimum}，100=最大值 {maximum}".strip("；")
    return f"{description}；仅填 0-100 整数".strip("；")


def planner_param_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a runtime param spec into a planner-stable spec."""

    primary_type = schema_primary_type(spec)
    if primary_type in {"number", "integer"}:
        return {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": planner_integer_slider_description(spec),
        }
    return dict(spec)


def repair_tool_arguments_json(raw_arguments: str) -> str:
    """Repair a few common planner JSON mistakes before parsing."""

    repaired = raw_arguments
    repaired = re.sub(r'([:\[,]\s*)0+(\d+\.\d+)', r"\1\2", repaired)
    repaired = re.sub(r'([:\[,]\s*)0+(\d+)(?=\s*[,}\]])', r"\1\2", repaired)
    repaired = re.sub(
        r'([:\[,]\s*)(True|False|None)(?=\s*[,}\]])',
        lambda match: match.group(1) + {"True": "true", "False": "false", "None": "null"}[match.group(2)],
        repaired,
    )
    return repaired


def parse_repaired_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse tool arguments with lightweight repair for common model mistakes."""

    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        parsed_arguments = json.loads(repair_tool_arguments_json(raw_arguments))

    if not isinstance(parsed_arguments, dict):
        raise RuntimeError("Planner arguments must be an object.")
    return parsed_arguments


def coerce_bool_like(value: Any) -> Any:
    """Coerce common string/int bool representations into booleans."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return value


def decode_planner_numeric_value(value: Any, spec: dict[str, Any]) -> Any:
    """Decode planner-facing 0-100 integer slider values back into the real schema range."""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            value = float(stripped) if "." in stripped else int(stripped)

    if not isinstance(value, (int, float)):
        return value

    slider_value = max(0.0, min(100.0, float(value)))
    minimum = schema_bound(spec, "minimum")
    maximum = schema_bound(spec, "maximum")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)) and maximum != minimum:
        decoded = float(minimum) + (slider_value / 100.0) * (float(maximum) - float(minimum))
    else:
        decoded = slider_value

    if schema_primary_type(spec) == "integer":
        return int(round(decoded))
    return decoded


def decode_planner_argument_value(value: Any, spec: dict[str, Any]) -> Any:
    """Decode planner-facing values into runtime values using the original schema."""

    primary_type = schema_primary_type(spec)
    if primary_type in {"number", "integer"}:
        return decode_planner_numeric_value(value, spec)
    if primary_type == "boolean":
        return coerce_bool_like(value)
    return value


def decode_planner_operation_params(tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any], float | None]:
    """Decode planner arguments into runtime params for a concrete tool."""

    registry = build_default_tool_registry()
    registered_tool = registry.require(tool_name)
    params_schema = registered_tool.spec.planner_schema
    schema_properties = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
    region = str(arguments.get("region") or WHOLE_IMAGE_REGION)
    params: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "region":
            continue
        spec = schema_properties.get(key) if isinstance(schema_properties, dict) else None
        if isinstance(spec, dict):
            params[key] = decode_planner_argument_value(value, spec)

    mask_prompt = params.get("mask_prompt")
    if isinstance(mask_prompt, str) and mask_prompt.strip():
        params["mask_prompt"] = normalize_segmentation_prompt_label(mask_prompt, region=region)
    params.pop("mask_negative_prompt", None)

    strength = params.get("strength", arguments.get("strength"))
    normalized_strength = float(strength) if isinstance(strength, (int, float)) else None
    return region, params, normalized_strength


def normalize_runtime_tool_params(tool_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Apply ToolSpec defaults and ignore explicit null overrides for runtime execution."""

    registry = build_default_tool_registry()
    registered_tool = registry.require(tool_name)
    schema_properties = dict(registered_tool.spec.planner_schema.get("properties") or {})
    merged = dict(registered_tool.spec.default_params)
    for key, value in dict(params or {}).items():
        if value is None:
            continue
        if key in schema_properties:
            merged[key] = value
    return merged


def extract_runtime_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return only shared mask parameters from a runtime param payload."""

    return {
        key: value
        for key, value in params.items()
        if key in MASK_PARAM_KEYS and value is not None and value != ""
    }


def strip_runtime_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return runtime tool params without shared mask fields."""

    return {key: value for key, value in params.items() if key not in MASK_PARAM_KEYS}
