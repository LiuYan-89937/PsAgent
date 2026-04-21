"""Native adjust_skin_tone_balance tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_point_color_adjustment


@tool
def adjust_skin_tone_balance(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    skin_hue_shift: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    skin_saturation_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    skin_luminance_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    protection: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    softness: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when skin tone specifically needs a more balanced hue, saturation, or luminance without changing the rest of the image. It is a skin-focused color correction tool and should be used with a skin mask rather than as a global color adjustment."""

    output_path = temp_output_path("psagent_skin_tone_balance_")
    preserve_neutrals = 0.15 + protection * 0.65
    range_width = 18.0 + softness * 24.0
    saved_path = apply_point_color_adjustment(
        image_path,
        output_path,
        target_color="skin",
        target_hue=None,
        range_width=range_width,
        hue_shift=skin_hue_shift * (1.0 - protection * 0.35),
        saturation_shift=skin_saturation_shift * (1.0 - protection * 0.25),
        luminance_shift=skin_luminance_shift * (1.0 - protection * 0.2),
        preserve_neutrals=preserve_neutrals,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_skin_tone_balance",
        output_image=saved_path,
        applied_params={
            "skin_hue_shift": skin_hue_shift,
            "skin_saturation_shift": skin_saturation_shift,
            "skin_luminance_shift": skin_luminance_shift,
            "protection": protection,
            "softness": softness,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SKIN_TONE_BALANCE_SPEC = ToolSpec(
    name="adjust_skin_tone_balance",
    label="肤色校正",
    description="Fine-tune skin hue, saturation, and luminance in a protected skin band.",
    family="color",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="skin",
    default_params={
        "skin_hue_shift": 0.0,
        "skin_saturation_shift": 0.0,
        "skin_luminance_shift": 0.0,
        "protection": 0.5,
        "softness": 0.5,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_skin_tone_balance,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="skin_hue_shift",
    risk_level="medium",
    status_label="正在校正肤色",
    keywords=("肤色校正", "肤色平衡", "脸色", "skin tone"),
)


__all__ = ["ADJUST_SKIN_TONE_BALANCE_SPEC", "adjust_skin_tone_balance"]
