"""Native adjust_color_grading tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_grading


@tool
def adjust_color_grading(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    shadow_hue: Annotated[float, Field(default=0.0, ge=0.0, le=360.0)] = 0.0,
    shadow_saturation: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    midtone_hue: Annotated[float, Field(default=0.0, ge=0.0, le=360.0)] = 0.0,
    midtone_saturation: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    highlight_hue: Annotated[float, Field(default=0.0, ge=0.0, le=360.0)] = 0.0,
    highlight_saturation: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    balance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    blending: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want creative split-toning or color grading across shadows, midtones, and highlights. It is intended for final look development and mood building, not for fixing narrow local color mistakes or technical white-balance issues."""

    output_path = temp_output_path("psagent_color_grading_")
    saved_path = apply_color_grading(
        image_path,
        output_path,
        shadow_hue=shadow_hue,
        shadow_saturation=shadow_saturation,
        midtone_hue=midtone_hue,
        midtone_saturation=midtone_saturation,
        highlight_hue=highlight_hue,
        highlight_saturation=highlight_saturation,
        balance=balance,
        blending=blending,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_grading",
        output_image=saved_path,
        applied_params={
            "shadow_hue": shadow_hue,
            "shadow_saturation": shadow_saturation,
            "midtone_hue": midtone_hue,
            "midtone_saturation": midtone_saturation,
            "highlight_hue": highlight_hue,
            "highlight_saturation": highlight_saturation,
            "balance": balance,
            "blending": blending,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_GRADING_SPEC = ToolSpec(
    name="adjust_color_grading",
    label="色彩分级",
    description="Apply split-toning across shadows, midtones, and highlights.",
    family="color",
    stage_affinity=["finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "shadow_hue": 0.0,
        "shadow_saturation": 0.0,
        "midtone_hue": 0.0,
        "midtone_saturation": 0.0,
        "highlight_hue": 0.0,
        "highlight_saturation": 0.0,
        "balance": 0.0,
        "blending": 0.5,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_color_grading,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="blending",
    risk_level="medium",
    status_label="正在进行色彩分级",
    keywords=("色彩分级", "split tone", "调色轮"),
)


__all__ = ["ADJUST_COLOR_GRADING_SPEC", "adjust_color_grading"]
