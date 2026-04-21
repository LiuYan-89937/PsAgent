"""Native adjust_black_white tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_black_white_mix_adjustment


@tool
def adjust_black_white(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    red_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    green_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    blue_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    yellow_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    cyan_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    magenta_weight: Annotated[float, Field(default=0.5, ge=0.0, le=2.0)] = 0.5,
    contrast_boost: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want a black-and-white conversion with control over how different source colors map into grayscale brightness. It is useful for monochrome styling and tonal separation, not for subtle color cleanup while staying in color."""

    output_path = temp_output_path("psagent_black_white_")
    saved_path = apply_black_white_mix_adjustment(
        image_path,
        output_path,
        red_weight=red_weight,
        green_weight=green_weight,
        blue_weight=blue_weight,
        yellow_weight=yellow_weight,
        cyan_weight=cyan_weight,
        magenta_weight=magenta_weight,
        contrast_boost=contrast_boost,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_black_white",
        output_image=saved_path,
        applied_params={
            "red_weight": red_weight,
            "green_weight": green_weight,
            "blue_weight": blue_weight,
            "yellow_weight": yellow_weight,
            "cyan_weight": cyan_weight,
            "magenta_weight": magenta_weight,
            "contrast_boost": contrast_boost,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_BLACK_WHITE_SPEC = ToolSpec(
    name="adjust_black_white",
    label="黑白转换",
    description="Convert to black and white using weighted color-band mixing.",
    family="color",
    stage_affinity=["finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "red_weight": 0.5,
        "green_weight": 0.5,
        "blue_weight": 0.5,
        "yellow_weight": 0.5,
        "cyan_weight": 0.5,
        "magenta_weight": 0.5,
        "contrast_boost": 0.2,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_black_white,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="contrast_boost",
    risk_level="low",
    status_label="正在转换黑白",
    keywords=("黑白", "black and white", "单色"),
)


__all__ = ["ADJUST_BLACK_WHITE_SPEC", "adjust_black_white"]
