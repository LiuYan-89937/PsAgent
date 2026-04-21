"""Native adjust_under_eye_brighten tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_regional_enhancement


@tool
def adjust_under_eye_brighten(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.25, ge=0.0, le=1.0)] = 0.25,
    contrast_soften: Annotated[float, Field(default=0.15, ge=0.0, le=1.0)] = 0.15,
    saturation_shift: Annotated[float, Field(default=-0.05, ge=-1.0, le=1.0)] = -0.05,
    shadow_lift: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    feather_radius: Annotated[float, Field(default=12.0, ge=0.0, le=64.0)] = 12.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the under-eye area looks too dark, tired, or hollow and needs gentle lifting with softer contrast. It is a very localized portrait retouch tool and should be used only with an under-eye mask."""

    output_path = temp_output_path("psagent_under_eye_brighten_")
    saved_path = apply_regional_enhancement(
        image_path,
        output_path,
        exposure_boost=amount * 0.9,
        saturation_boost=saturation_shift,
        warmth_shift=0.0,
        clarity_boost=0.0,
        smooth_amount=contrast_soften * 0.6,
        sharpen_amount=0.0,
        highlight_protection=0.42,
        shadow_lift=shadow_lift,
        yellow_suppression=max(0.0, -saturation_shift) * 0.25,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_under_eye_brighten",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "contrast_soften": contrast_soften,
            "saturation_shift": saturation_shift,
            "shadow_lift": shadow_lift,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_UNDER_EYE_BRIGHTEN_SPEC = ToolSpec(
    name="adjust_under_eye_brighten",
    label="眼下提亮",
    description="Brighten under-eye shadows and soften harsh contrast in a restrained way.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="under eye",
    default_params={
        "amount": 0.25,
        "contrast_soften": 0.15,
        "saturation_shift": -0.05,
        "shadow_lift": 0.2,
        "feather_radius": 12.0,
    },
    planner_schema=build_planner_schema(
        adjust_under_eye_brighten,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="amount",
    risk_level="medium",
    status_label="正在提亮眼下",
    keywords=("眼下提亮", "黑眼圈", "眼袋"),
)


__all__ = ["ADJUST_UNDER_EYE_BRIGHTEN_SPEC", "adjust_under_eye_brighten"]
