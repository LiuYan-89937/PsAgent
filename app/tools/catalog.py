"""Aggregated native tool catalog and ToolNode helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from app.tools.adjust_black_white import ADJUST_BLACK_WHITE_SPEC, adjust_black_white
from app.tools.adjust_brightness import ADJUST_BRIGHTNESS_SPEC, adjust_brightness
from app.tools.adjust_channel_mixer import ADJUST_CHANNEL_MIXER_SPEC, adjust_channel_mixer
from app.tools.adjust_chromatic_aberration import (
    ADJUST_CHROMATIC_ABERRATION_SPEC,
    adjust_chromatic_aberration,
)
from app.tools.adjust_clarity import ADJUST_CLARITY_SPEC, adjust_clarity
from app.tools.adjust_color_balance import ADJUST_COLOR_BALANCE_SPEC, adjust_color_balance
from app.tools.adjust_color_cleanup import ADJUST_COLOR_CLEANUP_SPEC, adjust_color_cleanup
from app.tools.adjust_color_grading import ADJUST_COLOR_GRADING_SPEC, adjust_color_grading
from app.tools.adjust_color_mixer import ADJUST_COLOR_MIXER_SPEC, adjust_color_mixer
from app.tools.adjust_color_noise_reduction import (
    ADJUST_COLOR_NOISE_REDUCTION_SPEC,
    adjust_color_noise_reduction,
)
from app.tools.adjust_color_overlay import ADJUST_COLOR_OVERLAY_SPEC, adjust_color_overlay
from app.tools.adjust_contrast import ADJUST_CONTRAST_SPEC, adjust_contrast
from app.tools.adjust_curves import ADJUST_CURVES_SPEC, adjust_curves
from app.tools.adjust_defringe import ADJUST_DEFRINGE_SPEC, adjust_defringe
from app.tools.adjust_dehaze import ADJUST_DEHAZE_SPEC, adjust_dehaze
from app.tools.adjust_exposure import ADJUST_EXPOSURE_SPEC, adjust_exposure
from app.tools.adjust_face_color_cleanup import ADJUST_FACE_COLOR_CLEANUP_SPEC, adjust_face_color_cleanup
from app.tools.adjust_glow_highlights import ADJUST_GLOW_HIGHLIGHTS_SPEC, adjust_glow_highlights
from app.tools.adjust_grain import ADJUST_GRAIN_SPEC, adjust_grain
from app.tools.adjust_hair_enhance import ADJUST_HAIR_ENHANCE_SPEC, adjust_hair_enhance
from app.tools.adjust_highlights_shadows import ADJUST_HIGHLIGHTS_SHADOWS_SPEC, adjust_highlights_shadows
from app.tools.adjust_hue_saturation import ADJUST_HUE_SATURATION_SPEC, adjust_hue_saturation
from app.tools.adjust_levels import ADJUST_LEVELS_SPEC, adjust_levels
from app.tools.adjust_local_contrast import ADJUST_LOCAL_CONTRAST_SPEC, adjust_local_contrast
from app.tools.adjust_midtones import ADJUST_MIDTONES_SPEC, adjust_midtones
from app.tools.adjust_moire_reduction import ADJUST_MOIRE_REDUCTION_SPEC, adjust_moire_reduction
from app.tools.adjust_neutral_clean_tone import ADJUST_NEUTRAL_CLEAN_TONE_SPEC, adjust_neutral_clean_tone
from app.tools.adjust_noise_reduction import ADJUST_NOISE_REDUCTION_SPEC, adjust_noise_reduction
from app.tools.adjust_point_color import ADJUST_POINT_COLOR_SPEC, adjust_point_color
from app.tools.adjust_selective_color import ADJUST_SELECTIVE_COLOR_SPEC, adjust_selective_color
from app.tools.adjust_sharpness import ADJUST_SHARPNESS_SPEC, adjust_sharpness
from app.tools.adjust_single_color_shift import ADJUST_SINGLE_COLOR_SHIFT_SPEC, adjust_single_color_shift
from app.tools.adjust_skin_brightness import ADJUST_SKIN_BRIGHTNESS_SPEC, adjust_skin_brightness
from app.tools.adjust_skin_smooth import ADJUST_SKIN_SMOOTH_SPEC, adjust_skin_smooth
from app.tools.adjust_skin_texture_reduce import ADJUST_SKIN_TEXTURE_REDUCE_SPEC, adjust_skin_texture_reduce
from app.tools.adjust_skin_tone_balance import ADJUST_SKIN_TONE_BALANCE_SPEC, adjust_skin_tone_balance
from app.tools.adjust_soft_glow import ADJUST_SOFT_GLOW_SPEC, adjust_soft_glow
from app.tools.adjust_soften_local_contrast import (
    ADJUST_SOFTEN_LOCAL_CONTRAST_SPEC,
    adjust_soften_local_contrast,
)
from app.tools.adjust_temperature_tint import ADJUST_TEMPERATURE_TINT_SPEC, adjust_temperature_tint
from app.tools.adjust_texture import ADJUST_TEXTURE_SPEC, adjust_texture
from app.tools.adjust_vibrance_saturation import (
    ADJUST_VIBRANCE_SATURATION_SPEC,
    adjust_vibrance_saturation,
)
from app.tools.adjust_vignette import ADJUST_VIGNETTE_SPEC, adjust_vignette
from app.tools.adjust_whites_blacks import ADJUST_WHITES_BLACKS_SPEC, adjust_whites_blacks
from app.tools.apply_color_lookup import APPLY_COLOR_LOOKUP_SPEC, apply_color_lookup
from app.tools.apply_photo_filter import APPLY_PHOTO_FILTER_SPEC, apply_photo_filter
from app.tools.common.contracts import ToolSpec, execution_modes_for_spec


# Tone 工具负责基础亮度、对比和全局光影基线。
TONE_NATIVE_TOOLS: tuple[BaseTool, ...] = (
    adjust_exposure,
    adjust_brightness,
    adjust_contrast,
    adjust_local_contrast,
    adjust_highlights_shadows,
    adjust_whites_blacks,
    adjust_levels,
    adjust_curves,
    adjust_midtones,
    adjust_temperature_tint,
)

TONE_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ADJUST_EXPOSURE_SPEC,
    ADJUST_BRIGHTNESS_SPEC,
    ADJUST_CONTRAST_SPEC,
    ADJUST_LOCAL_CONTRAST_SPEC,
    ADJUST_HIGHLIGHTS_SHADOWS_SPEC,
    ADJUST_WHITES_BLACKS_SPEC,
    ADJUST_LEVELS_SPEC,
    ADJUST_CURVES_SPEC,
    ADJUST_MIDTONES_SPEC,
    ADJUST_TEMPERATURE_TINT_SPEC,
)

# Color 工具优先覆盖精确调色、颜色带选择、风格化收尾。
COLOR_NATIVE_TOOLS: tuple[BaseTool, ...] = (
    adjust_vibrance_saturation,
    adjust_hue_saturation,
    adjust_color_balance,
    adjust_black_white,
    adjust_color_mixer,
    adjust_point_color,
    adjust_selective_color,
    adjust_single_color_shift,
    adjust_neutral_clean_tone,
    adjust_skin_tone_balance,
    adjust_color_cleanup,
    adjust_color_overlay,
    adjust_channel_mixer,
    adjust_color_grading,
    apply_photo_filter,
    apply_color_lookup,
)

COLOR_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ADJUST_VIBRANCE_SATURATION_SPEC,
    ADJUST_HUE_SATURATION_SPEC,
    ADJUST_COLOR_BALANCE_SPEC,
    ADJUST_BLACK_WHITE_SPEC,
    ADJUST_COLOR_MIXER_SPEC,
    ADJUST_POINT_COLOR_SPEC,
    ADJUST_SELECTIVE_COLOR_SPEC,
    ADJUST_SINGLE_COLOR_SHIFT_SPEC,
    ADJUST_NEUTRAL_CLEAN_TONE_SPEC,
    ADJUST_SKIN_TONE_BALANCE_SPEC,
    ADJUST_COLOR_CLEANUP_SPEC,
    ADJUST_COLOR_OVERLAY_SPEC,
    ADJUST_CHANNEL_MIXER_SPEC,
    ADJUST_COLOR_GRADING_SPEC,
    APPLY_PHOTO_FILTER_SPEC,
    APPLY_COLOR_LOOKUP_SPEC,
)

# Detail 工具负责质感、降噪、去雾、锐化和局部清晰度控制。
DETAIL_NATIVE_TOOLS: tuple[BaseTool, ...] = (
    adjust_texture,
    adjust_clarity,
    adjust_dehaze,
    adjust_sharpness,
    adjust_noise_reduction,
    adjust_moire_reduction,
    adjust_defringe,
    adjust_skin_texture_reduce,
    adjust_soften_local_contrast,
    adjust_color_noise_reduction,
)

DETAIL_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ADJUST_TEXTURE_SPEC,
    ADJUST_CLARITY_SPEC,
    ADJUST_DEHAZE_SPEC,
    ADJUST_SHARPNESS_SPEC,
    ADJUST_NOISE_REDUCTION_SPEC,
    ADJUST_MOIRE_REDUCTION_SPEC,
    ADJUST_DEFRINGE_SPEC,
    ADJUST_SKIN_TEXTURE_REDUCE_SPEC,
    ADJUST_SOFTEN_LOCAL_CONTRAST_SPEC,
    ADJUST_COLOR_NOISE_REDUCTION_SPEC,
)

# Effects 工具负责轻风格化效果，不改变画面结构。
EFFECT_NATIVE_TOOLS: tuple[BaseTool, ...] = (
    adjust_chromatic_aberration,
    adjust_vignette,
    adjust_grain,
    adjust_glow_highlights,
    adjust_soft_glow,
)

EFFECT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ADJUST_CHROMATIC_ABERRATION_SPEC,
    ADJUST_VIGNETTE_SPEC,
    ADJUST_GRAIN_SPEC,
    ADJUST_GLOW_HIGHLIGHTS_SPEC,
    ADJUST_SOFT_GLOW_SPEC,
)

# Portrait 工具默认面向主体精修，后续只在 subject_cleanup 暴露。
PORTRAIT_NATIVE_TOOLS: tuple[BaseTool, ...] = (
    adjust_skin_smooth,
    adjust_skin_brightness,
    adjust_hair_enhance,
    adjust_face_color_cleanup,
)

PORTRAIT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ADJUST_SKIN_SMOOTH_SPEC,
    ADJUST_SKIN_BRIGHTNESS_SPEC,
    ADJUST_HAIR_ENHANCE_SPEC,
    ADJUST_FACE_COLOR_CLEANUP_SPEC,
)

NATIVE_TOOLS: tuple[BaseTool, ...] = (
    TONE_NATIVE_TOOLS
    + COLOR_NATIVE_TOOLS
    + DETAIL_NATIVE_TOOLS
    + EFFECT_NATIVE_TOOLS
    + PORTRAIT_NATIVE_TOOLS
)

TOOL_SPECS: tuple[ToolSpec, ...] = (
    TONE_TOOL_SPECS
    + COLOR_TOOL_SPECS
    + DETAIL_TOOL_SPECS
    + EFFECT_TOOL_SPECS
    + PORTRAIT_TOOL_SPECS
)

TONE_TOOL_NAMES = tuple(spec.name for spec in TONE_TOOL_SPECS)
COLOR_TOOL_NAMES = tuple(spec.name for spec in COLOR_TOOL_SPECS)
DETAIL_TOOL_NAMES = tuple(spec.name for spec in DETAIL_TOOL_SPECS)
EFFECT_TOOL_NAMES = tuple(spec.name for spec in EFFECT_TOOL_SPECS)
PORTRAIT_TOOL_NAMES = tuple(spec.name for spec in PORTRAIT_TOOL_SPECS)

TOOL_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}
TOOLS_BY_NAME = {tool.name: tool for tool in NATIVE_TOOLS}


_CONFLICT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("adjust_exposure", "adjust_brightness", "adjust_whites_blacks", "adjust_glow_highlights", "adjust_soft_glow"),
    ("adjust_vibrance_saturation", "adjust_hue_saturation", "adjust_color_mixer", "adjust_single_color_shift"),
    ("adjust_sharpness", "adjust_clarity", "adjust_texture", "adjust_local_contrast"),
    ("adjust_skin_smooth", "adjust_skin_texture_reduce", "adjust_soften_local_contrast"),
)


def _derived_conflict_tools(spec: ToolSpec) -> list[str]:
    """Return tools that should not be stacked casually with this tool."""

    conflicts = set(spec.conflict_tools)
    for group in _CONFLICT_GROUPS:
        if spec.name in group:
            conflicts.update(name for name in group if name != spec.name)
    return sorted(conflicts)


def _derived_selection_guidance(spec: ToolSpec) -> str:
    """Build a short model-facing tool selection hint."""

    if spec.selection_guidance:
        return spec.selection_guidance
    if spec.family == "tone":
        return "Use for tonal correction; avoid stacking several tone tools unless the prior result still has a measured issue."
    if spec.family == "color":
        return "Use for color correction or color styling; prefer targeted color tools when only one hue range is wrong."
    if spec.family == "detail":
        return "Use for texture, sharpness, haze, or noise issues; keep amounts conservative on portraits."
    if spec.family == "portrait":
        return "Use only when the matching human region is visible and a suitable mask is available."
    if spec.family == "effects":
        return "Use only as a finishing effect after tone and color are already stable."
    return "Use when the request and image analysis match this tool's documented purpose."


def require_tool(name: str) -> BaseTool:
    """Return a concrete tool or raise a clear error."""

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise KeyError(f"Unknown tool: {name}")
    return tool


def require_tool_spec(name: str) -> ToolSpec:
    """Return a concrete tool spec or raise a clear error."""

    spec = TOOL_SPECS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"Unknown tool spec: {name}")
    return spec


def export_tool_catalog() -> list[dict[str, Any]]:
    """Return the planner/API-visible tool catalog."""

    items: list[dict[str, Any]] = []
    for spec in TOOL_SPECS:
        if spec.supports_mask:
            mask_policy = "required" if spec.requires_mask or not spec.supports_whole_image else "optional"
        else:
            mask_policy = "none"
        items.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "family": spec.family,
                "focus_affinity": list(spec.focus_affinity),
                "supports_mask": spec.supports_mask,
                "requires_mask": spec.requires_mask,
                "supports_whole_image": spec.supports_whole_image,
                "recommended_mask_prompt": spec.recommended_mask_prompt,
                "recommended_mask_prompts": list(spec.recommended_mask_prompts),
                "selection_guidance": _derived_selection_guidance(spec),
                "conflict_tools": _derived_conflict_tools(spec),
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
def build_default_tool_node() -> ToolNode:
    """Build the shared ToolNode backed by the native tools."""

    return ToolNode(list(NATIVE_TOOLS))


def exported_tool_names() -> tuple[str, ...]:
    """Return tool names in stable definition order."""

    return tuple(spec.name for spec in TOOL_SPECS)


__all__ = [
    "COLOR_TOOL_NAMES",
    "COLOR_TOOL_SPECS",
    "COLOR_NATIVE_TOOLS",
    "DETAIL_TOOL_NAMES",
    "DETAIL_TOOL_SPECS",
    "DETAIL_NATIVE_TOOLS",
    "EFFECT_TOOL_NAMES",
    "EFFECT_TOOL_SPECS",
    "EFFECT_NATIVE_TOOLS",
    "NATIVE_TOOLS",
    "PORTRAIT_TOOL_NAMES",
    "PORTRAIT_TOOL_SPECS",
    "PORTRAIT_NATIVE_TOOLS",
    "TOOLS_BY_NAME",
    "TOOL_SPECS",
    "TOOL_SPECS_BY_NAME",
    "TONE_TOOL_NAMES",
    "TONE_TOOL_SPECS",
    "TONE_NATIVE_TOOLS",
    "build_default_tool_node",
    "export_tool_catalog",
    "exported_tool_names",
    "require_tool",
    "require_tool_spec",
]
