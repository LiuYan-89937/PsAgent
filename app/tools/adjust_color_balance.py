"""Native adjust_color_balance tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_balance_adjustment


@tool
def adjust_color_balance(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    shadow_cyan_red: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    shadow_magenta_green: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    shadow_yellow_blue: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    midtone_cyan_red: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    midtone_magenta_green: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    midtone_yellow_blue: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    highlight_cyan_red: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    highlight_magenta_green: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    highlight_yellow_blue: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    preserve_luminosity: Annotated[bool, Field(default=True)] = True,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when shadows, midtones, or highlights each need different color bias corrections such as cooler shadows or warmer highlights. It is best for tonal-range-aware color balancing and is more structured than a simple global hue or temperature shift."""

    output_path = temp_output_path("psagent_color_balance_")
    saved_path = apply_color_balance_adjustment(
        image_path,
        output_path,
        shadow_cyan_red=shadow_cyan_red,
        shadow_magenta_green=shadow_magenta_green,
        shadow_yellow_blue=shadow_yellow_blue,
        midtone_cyan_red=midtone_cyan_red,
        midtone_magenta_green=midtone_magenta_green,
        midtone_yellow_blue=midtone_yellow_blue,
        highlight_cyan_red=highlight_cyan_red,
        highlight_magenta_green=highlight_magenta_green,
        highlight_yellow_blue=highlight_yellow_blue,
        preserve_luminosity=preserve_luminosity,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_balance",
        output_image=saved_path,
        applied_params={
            "shadow_cyan_red": shadow_cyan_red,
            "shadow_magenta_green": shadow_magenta_green,
            "shadow_yellow_blue": shadow_yellow_blue,
            "midtone_cyan_red": midtone_cyan_red,
            "midtone_magenta_green": midtone_magenta_green,
            "midtone_yellow_blue": midtone_yellow_blue,
            "highlight_cyan_red": highlight_cyan_red,
            "highlight_magenta_green": highlight_magenta_green,
            "highlight_yellow_blue": highlight_yellow_blue,
            "preserve_luminosity": preserve_luminosity,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_BALANCE_SPEC = ToolSpec(
    name="adjust_color_balance",
    label="色彩平衡",
    description="Adjust color balance across shadows, midtones, and highlights.",
    family="color",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "shadow_cyan_red": 0.0,
        "shadow_magenta_green": 0.0,
        "shadow_yellow_blue": 0.0,
        "midtone_cyan_red": 0.0,
        "midtone_magenta_green": 0.0,
        "midtone_yellow_blue": 0.0,
        "highlight_cyan_red": 0.0,
        "highlight_magenta_green": 0.0,
        "highlight_yellow_blue": 0.0,
        "preserve_luminosity": True,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_color_balance,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="midtone_yellow_blue",
    risk_level="low",
    status_label="正在调整色彩平衡",
    keywords=("色彩平衡", "冷暖平衡", "阴影调色", "高光调色"),
)


__all__ = ["ADJUST_COLOR_BALANCE_SPEC", "adjust_color_balance"]
