"""Helpers for building compact model-facing context payloads."""

from __future__ import annotations

from typing import Any

from app.graph.state import AnalyzeImageResult, EditPlan, ExecutionTraceItem, PreferenceMemory, RequestIntent, ToolCatalogItem
from app.services.planner_param_codec import planner_param_spec
from app.tools.common.mask_contracts import MASK_PARAM_KEYS, MASK_PARAMS_SCHEMA


_BOUND_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
)


def _compact_param_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON-schema property into a compact param descriptor."""

    planner_spec = planner_param_spec(spec)
    variants = spec.get("anyOf") if isinstance(spec.get("anyOf"), list) else [spec]
    non_null_variants = [
        item
        for item in variants
        if isinstance(item, dict) and item.get("type") != "null"
    ]
    if not non_null_variants:
        non_null_variants = [spec]

    types: list[str] = []
    enum_values: list[Any] = []
    compact: dict[str, Any] = {}

    for variant in non_null_variants:
        variant_type = variant.get("type")
        if isinstance(variant_type, str) and variant_type not in types:
            types.append(variant_type)

        variant_enum = variant.get("enum")
        if isinstance(variant_enum, list):
            for value in variant_enum:
                if value not in enum_values:
                    enum_values.append(value)

        for key in _BOUND_KEYS:
            if key in variant and key not in compact:
                compact[key] = variant[key]

    description = planner_spec.get("description")
    if isinstance(description, str) and description:
        compact["description"] = description

    default_value = planner_spec.get("default", spec.get("default"))
    if default_value not in (None, "", [], {}):
        compact["default"] = default_value

    planner_type = planner_spec.get("type")
    if isinstance(planner_type, str):
        compact["type"] = planner_type
    elif types:
        compact["type"] = types[0] if len(types) == 1 else "|".join(types)
    if isinstance(planner_spec.get("minimum"), (int, float)):
        compact["minimum"] = planner_spec["minimum"]
    if isinstance(planner_spec.get("maximum"), (int, float)):
        compact["maximum"] = planner_spec["maximum"]
    if enum_values:
        compact["enum"] = enum_values

    return compact


def compact_request_intent_for_model(request_intent: RequestIntent | dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact request-intent summary for model consumption."""

    if isinstance(request_intent, RequestIntent):
        payload = request_intent.model_dump(mode="json")
    else:
        payload = dict(request_intent or {})
    compact: dict[str, Any] = {
        "mode": payload.get("mode"),
        "constraints": payload.get("constraints", []),
        "goal_summary": payload.get("goal_summary"),
        "wants_repair": payload.get("wants_repair"),
        "wants_style": payload.get("wants_style"),
        "requires_local_editing": payload.get("requires_local_editing"),
    }

    requested_packages = []
    for item in payload.get("requested_packages", []) or []:
        requested_packages.append(
            {
                "op": item.get("op"),
                "region": item.get("region"),
                "strength": item.get("strength"),
                "params": item.get("params", {}),
            }
        )
    compact["requested_packages"] = requested_packages
    return compact


def compact_image_analysis_for_model(image_analysis: AnalyzeImageResult | dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact image-analysis summary for model consumption."""

    if isinstance(image_analysis, AnalyzeImageResult):
        payload = image_analysis.model_dump(mode="json")
    else:
        payload = dict(image_analysis or {})
    metrics = dict(payload.get("metrics") or {})
    compact_metrics = {
        key: metrics[key]
        for key in ("brightness_mean", "brightness_std", "shadow_ratio", "highlight_ratio")
        if key in metrics
    }
    compact: dict[str, Any] = {
        "domain": payload.get("domain"),
        "summary": payload.get("summary"),
        "scene_tags": payload.get("scene_tags", []),
        "issues": payload.get("issues", []),
        "main_issues": payload.get("main_issues", []),
        "subjects": payload.get("subjects", []),
        "primary_subject": payload.get("primary_subject"),
        "segmentation_hints": payload.get("segmentation_hints", []),
        "has_portrait": payload.get("has_portrait"),
        "needs_local_editing": payload.get("needs_local_editing"),
        "has_background_distraction": payload.get("has_background_distraction"),
    }
    if compact_metrics:
        compact["metrics"] = compact_metrics
    return compact


def compact_preferences_for_model(
    retrieved_prefs: list[PreferenceMemory | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build a compact preference list for model consumption."""

    compact_items: list[dict[str, Any]] = []
    for item in retrieved_prefs or []:
        payload = item.model_dump(mode="json") if isinstance(item, PreferenceMemory) else item
        compact_item = {
            "key": payload.get("key"),
            "value": payload.get("value"),
        }
        if payload.get("confidence") is not None:
            compact_item["confidence"] = payload.get("confidence")
        if payload.get("source") is not None:
            compact_item["source"] = payload.get("source")
        compact_items.append(compact_item)
    return compact_items


def compact_plan_for_model(plan: EditPlan | dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact plan summary for model consumption."""

    if isinstance(plan, EditPlan):
        payload = plan.model_dump(mode="json")
    else:
        payload = dict(plan or {})
    operations = []
    for item in payload.get("operations", []) or []:
        operations.append(
            {
                "op": item.get("op"),
                "region": item.get("region"),
                "params": item.get("params", {}),
            }
        )

    compact: dict[str, Any] = {"operations": operations}
    for key in ("mode", "domain", "executor", "preserve"):
        if key in payload:
            compact[key] = payload.get(key)
    return compact


def compact_execution_trace_for_model(
    execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build a compact execution trace summary for model consumption."""

    compact_items: list[dict[str, Any]] = []
    for item in execution_trace or []:
        payload = item.model_dump(mode="json") if isinstance(item, ExecutionTraceItem) else item
        compact_item: dict[str, Any] = {
            "op": payload.get("op"),
            "region": payload.get("region"),
            "ok": payload.get("ok"),
        }

        if payload.get("fallback_used"):
            compact_item["fallback_used"] = payload.get("fallback_used")
        if payload.get("error"):
            compact_item["error"] = payload.get("error")

        applied_params = payload.get("applied_params")
        if isinstance(applied_params, dict):
            source_params = applied_params.get("params") if isinstance(applied_params.get("params"), dict) else applied_params
            compact_params: dict[str, Any] = {}
            for key, value in source_params.items():
                if value in (None, "", False):
                    continue
                compact_params[key] = value
                if len(compact_params) >= 6:
                    break
            if compact_params:
                compact_item["params"] = compact_params

        compact_items.append(compact_item)

    return compact_items


def shared_mask_params_for_model(
    tool_catalog: list[ToolCatalogItem | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract shared mask param definitions once for planner context."""

    supports_mask = False
    for item in tool_catalog:
        payload = item.model_dump(mode="json") if isinstance(item, ToolCatalogItem) else item
        if payload.get("supports_mask") or payload.get("mask_policy") in {"optional", "required"}:
            supports_mask = True
            break
    if not supports_mask:
        return []

    properties = MASK_PARAMS_SCHEMA.get("properties", {})
    if not isinstance(properties, dict):
        return []

    shared_params: list[dict[str, Any]] = []
    for param_name, spec in properties.items():
        if param_name not in MASK_PARAM_KEYS or not isinstance(spec, dict):
            continue
        compact_param = {"name": param_name}
        compact_param.update(_compact_param_spec(spec))
        shared_params.append(compact_param)
    return shared_params


def compact_tool_catalog_for_model(
    tool_catalog: list[ToolCatalogItem | dict[str, Any]],
    *,
    include_params: bool,
) -> list[dict[str, Any]]:
    """Build a compact tool catalog tailored for model consumption."""

    compact_items: list[dict[str, Any]] = []
    for item in tool_catalog:
        payload = item.model_dump(mode="json") if isinstance(item, ToolCatalogItem) else item
        compact_item: dict[str, Any] = {
            "name": payload.get("name"),
            "label": payload.get("label"),
            "description": payload.get("description"),
            "family": payload.get("family"),
            "execution_modes": payload.get("supported_regions", []),
        }

        mask_policy = payload.get("mask_policy")
        if mask_policy and mask_policy != "none":
            compact_item["mask_policy"] = mask_policy

        risk_level = payload.get("risk_level")
        if risk_level:
            compact_item["risk_level"] = risk_level

        supported_domains = payload.get("supported_domains") or []
        if supported_domains:
            compact_item["supported_domains"] = supported_domains

        if include_params:
            schema = payload.get("planner_schema") or payload.get("params_schema") or {}
            properties = schema.get("properties") if isinstance(schema, dict) else {}
            required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
            compact_params: list[dict[str, Any]] = []

            if isinstance(properties, dict):
                for param_name, spec in properties.items():
                    if not isinstance(spec, dict):
                        continue
                    if param_name in MASK_PARAM_KEYS:
                        continue
                    compact_param = {"name": param_name}
                    compact_param.update(_compact_param_spec(spec))
                    if param_name in required:
                        compact_param["required"] = True
                    compact_params.append(compact_param)

            compact_item["params"] = compact_params
            if payload.get("primary_param"):
                compact_item["primary_param"] = payload["primary_param"]

        compact_items.append(compact_item)

    return compact_items
