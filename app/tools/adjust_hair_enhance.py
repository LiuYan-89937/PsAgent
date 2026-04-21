"""Native adjust_hair_enhance tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_regional_enhancement


@tool
def adjust_hair_enhance(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    texture_boost: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    clarity_boost: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    highlight_control: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    saturation_balance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    feather_radius: Annotated[float, Field(default=14.0, ge=0.0, le=64.0)] = 14.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when hair should have more texture, separation, and controlled shine without affecting skin or background. It is a hair-region portrait tool and should be used with a hair mask instead of across the full frame."""

    output_path = temp_output_path("psagent_hair_enhance_")
    saved_path = apply_regional_enhancement(
        image_path,
        output_path,
        exposure_boost=0.0,
        saturation_boost=saturation_balance,
        warmth_shift=0.0,
        clarity_boost=clarity_boost,
        smooth_amount=0.0,
        sharpen_amount=texture_boost,
        highlight_protection=highlight_control,
        shadow_lift=0.0,
        yellow_suppression=0.0,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_hair_enhance",
        output_image=saved_path,
        applied_params={
            "texture_boost": texture_boost,
            "clarity_boost": clarity_boost,
            "highlight_control": highlight_control,
            "saturation_balance": saturation_balance,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_HAIR_ENHANCE_SPEC = ToolSpec(
    name="adjust_hair_enhance",
    label="发丝质感增强",
    description="Enhance hair texture and clarity while keeping highlights controlled.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="hair",
    default_params={
        "texture_boost": 0.3,
        "clarity_boost": 0.2,
        "highlight_control": 0.2,
        "saturation_balance": 0.0,
        "feather_radius": 14.0,
    },
    planner_schema=build_planner_schema(
        adjust_hair_enhance,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="texture_boost",
    risk_level="medium",
    status_label="正在增强发丝质感",
    keywords=("发丝质感", "头发增强", "头发层次"),
)


__all__ = ["ADJUST_HAIR_ENHANCE_SPEC", "adjust_hair_enhance"]
