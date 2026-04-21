"""Native adjust_lip_enhance tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_lip_enhance_adjustment


@tool
def adjust_lip_enhance(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    hue_shift: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    saturation_boost: Annotated[float, Field(default=0.12, ge=-1.0, le=1.0)] = 0.12,
    brightness_shift: Annotated[float, Field(default=0.05, ge=-1.0, le=1.0)] = 0.05,
    texture_preserve: Annotated[float, Field(default=0.7, ge=0.0, le=1.0)] = 0.7,
    gloss_boost: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    feather_radius: Annotated[float, Field(default=10.0, ge=0.0, le=64.0)] = 10.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when lips need stronger color presence, slightly brighter tone, or a touch of gloss while preserving texture. It is a lip-region portrait enhancement tool and should be used with a lips mask rather than globally."""

    output_path = temp_output_path("psagent_lip_enhance_")
    saved_path = apply_lip_enhance_adjustment(
        image_path,
        output_path,
        hue_shift=hue_shift,
        saturation_boost=saturation_boost,
        brightness_shift=brightness_shift,
        texture_preserve=texture_preserve,
        gloss_boost=gloss_boost,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_lip_enhance",
        output_image=saved_path,
        applied_params={
            "hue_shift": hue_shift,
            "saturation_boost": saturation_boost,
            "brightness_shift": brightness_shift,
            "texture_preserve": texture_preserve,
            "gloss_boost": gloss_boost,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_LIP_ENHANCE_SPEC = ToolSpec(
    name="adjust_lip_enhance",
    label="唇色增强",
    description="Enhance lip color and gloss while preserving texture.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="lips",
    default_params={
        "hue_shift": 0.0,
        "saturation_boost": 0.12,
        "brightness_shift": 0.05,
        "texture_preserve": 0.7,
        "gloss_boost": 0.0,
        "feather_radius": 10.0,
    },
    planner_schema=build_planner_schema(
        adjust_lip_enhance,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="saturation_boost",
    risk_level="medium",
    status_label="正在增强唇色",
    keywords=("唇色", "口红", "嘴唇"),
)


__all__ = ["ADJUST_LIP_ENHANCE_SPEC", "adjust_lip_enhance"]
