"""Native adjust_single_color_shift tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_single_color_shift_adjustment


@tool
def adjust_single_color_shift(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    target_hue: Annotated[float, Field(default=0.0, ge=0.0, le=360.0)] = 0.0,
    hue_width: Annotated[float, Field(default=15.0, ge=8.0, le=90.0)] = 15.0,
    hue_shift: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    saturation_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    luminance_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    softness: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you explicitly want to move one narrow hue band around a chosen target hue without affecting the whole image. It is useful for technical color nudging when you know the target hue range, and is more surgical but less semantic than point-color presets."""

    output_path = temp_output_path("psagent_single_color_shift_")
    saved_path = apply_single_color_shift_adjustment(
        image_path,
        output_path,
        target_hue=target_hue,
        hue_width=hue_width,
        hue_shift=hue_shift,
        saturation_shift=saturation_shift,
        luminance_shift=luminance_shift,
        softness=softness,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_single_color_shift",
        output_image=saved_path,
        applied_params={
            "target_hue": target_hue,
            "hue_width": hue_width,
            "hue_shift": hue_shift,
            "saturation_shift": saturation_shift,
            "luminance_shift": luminance_shift,
            "softness": softness,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SINGLE_COLOR_SHIFT_SPEC = ToolSpec(
    name="adjust_single_color_shift",
    label="单色偏移",
    description="Shift a single narrow hue band with soft falloff.",
    family="color",
    focus_affinity=["subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "target_hue": 0.0,
        "hue_width": 15.0,
        "hue_shift": 0.0,
        "saturation_shift": 0.0,
        "luminance_shift": 0.0,
        "softness": 0.5,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_single_color_shift,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="saturation_shift",
    risk_level="medium",
    status_label="正在调整单色偏移",
    keywords=("单色偏移", "单个颜色", "某个颜色", "颜色微调"),
)


__all__ = ["ADJUST_SINGLE_COLOR_SHIFT_SPEC", "adjust_single_color_shift"]
