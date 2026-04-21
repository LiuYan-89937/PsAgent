"""Native adjust_skin_brightness tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_regional_enhancement


@tool
def adjust_skin_brightness(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    brightness_shift: Annotated[float, Field(default=0.1, ge=-1.0, le=1.0)] = 0.1,
    saturation_shift: Annotated[float, Field(default=-0.05, ge=-1.0, le=1.0)] = -0.05,
    highlight_protection: Annotated[float, Field(default=0.25, ge=0.0, le=1.0)] = 0.25,
    preserve_texture: Annotated[float, Field(default=0.6, ge=0.0, le=1.0)] = 0.6,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when skin should look cleaner, brighter, and more flattering without losing too much texture. It is a skin-region local retouch tool and should be used with a skin mask rather than as a whole-image brightening step."""

    output_path = temp_output_path("psagent_skin_brightness_")
    saved_path = apply_regional_enhancement(
        image_path,
        output_path,
        exposure_boost=brightness_shift,
        saturation_boost=saturation_shift,
        warmth_shift=0.0,
        clarity_boost=0.0,
        smooth_amount=max(0.0, (1.0 - preserve_texture) * 0.16),
        sharpen_amount=preserve_texture * 0.08,
        highlight_protection=highlight_protection,
        shadow_lift=max(0.0, brightness_shift) * 0.08,
        yellow_suppression=max(0.0, -saturation_shift) * 0.45,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_skin_brightness",
        output_image=saved_path,
        applied_params={
            "brightness_shift": brightness_shift,
            "saturation_shift": saturation_shift,
            "highlight_protection": highlight_protection,
            "preserve_texture": preserve_texture,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SKIN_BRIGHTNESS_SPEC = ToolSpec(
    name="adjust_skin_brightness",
    label="皮肤亮度提纯",
    description="Brighten skin gently while suppressing dirty yellow and preserving texture.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="skin",
    default_params={
        "brightness_shift": 0.1,
        "saturation_shift": -0.05,
        "highlight_protection": 0.25,
        "preserve_texture": 0.6,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_skin_brightness,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="brightness_shift",
    risk_level="medium",
    status_label="正在提纯皮肤亮度",
    keywords=("皮肤亮度", "提亮脸", "肤色提亮"),
)


__all__ = ["ADJUST_SKIN_BRIGHTNESS_SPEC", "adjust_skin_brightness"]
