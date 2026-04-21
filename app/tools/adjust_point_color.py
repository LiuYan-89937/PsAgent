"""Native adjust_point_color tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_point_color_adjustment


@tool
def adjust_point_color(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    target_color: Annotated[str, Field(default="skin", description="Target color preset name.")] = "skin",
    target_hue: Annotated[float | None, Field(default=None, ge=0.0, le=360.0, description="Optional explicit target hue.")] = None,
    range_width: Annotated[float, Field(default=24.0, ge=8.0, le=80.0, description="Hue range width.")] = 24.0,
    hue_shift: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    saturation_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    luminance_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    preserve_neutrals: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when one very specific color value or a narrow color neighborhood needs correction, such as a slightly yellow white dress or a small dirty skin patch. It is more precise than color mixer, so prefer it when broad color-band edits would cause unwanted spill into nearby colors."""

    output_path = temp_output_path("psagent_point_color_")
    saved_path = apply_point_color_adjustment(
        image_path,
        output_path,
        target_color=target_color,
        target_hue=target_hue,
        range_width=range_width,
        hue_shift=hue_shift,
        saturation_shift=saturation_shift,
        luminance_shift=luminance_shift,
        preserve_neutrals=preserve_neutrals,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_point_color",
        output_image=saved_path,
        applied_params={
            "target_color": target_color,
            "target_hue": target_hue,
            "range_width": range_width,
            "hue_shift": hue_shift,
            "saturation_shift": saturation_shift,
            "luminance_shift": luminance_shift,
            "preserve_neutrals": preserve_neutrals,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_POINT_COLOR_SPEC = ToolSpec(
    name="adjust_point_color",
    label="精准点颜色",
    description="Adjust a narrow target color band with hue, saturation, and luminance shifts.",
    family="color",
    stage_affinity=["local_balance", "subject_refine"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "target_color": "skin",
        "target_hue": None,
        "range_width": 24.0,
        "hue_shift": 0.0,
        "saturation_shift": 0.0,
        "luminance_shift": 0.0,
        "preserve_neutrals": 0.2,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_point_color,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="saturation_shift",
    risk_level="medium",
    status_label="正在精准调色",
    keywords=("点颜色", "精准颜色", "白裙去黄", "肤色微调"),
)


__all__ = ["ADJUST_POINT_COLOR_SPEC", "adjust_point_color"]
