"""Native adjust_levels tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_levels_adjustment


@tool
def adjust_levels(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    input_black: Annotated[float, Field(default=0.0, ge=0.0, le=0.9, description="Input black point.")] = 0.0,
    input_white: Annotated[float, Field(default=1.0, ge=0.02, le=1.0, description="Input white point.")] = 1.0,
    gamma: Annotated[float, Field(default=1.0, ge=0.2, le=3.5, description="Gamma / midtone response.")] = 1.0,
    output_black: Annotated[float, Field(default=0.0, ge=0.0, le=0.9, description="Output black point.")] = 0.0,
    output_white: Annotated[float, Field(default=1.0, ge=0.02, le=1.0, description="Output white point.")] = 1.0,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you need a classic levels-style remap of input black, white, gamma, and output range. It is useful for deliberate tonal remapping and technical cleanup, especially when you want more explicit control than simple brightness or contrast tools."""

    output_path = temp_output_path("psagent_levels_")
    saved_path = apply_levels_adjustment(
        image_path,
        output_path,
        input_black=input_black,
        input_white=input_white,
        gamma=gamma,
        output_black=output_black,
        output_white=output_white,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_levels",
        output_image=saved_path,
        applied_params={
            "input_black": input_black,
            "input_white": input_white,
            "gamma": gamma,
            "output_black": output_black,
            "output_white": output_white,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_LEVELS_SPEC = ToolSpec(
    name="adjust_levels",
    label="色阶",
    description="Adjust input and output levels on the luminance channel.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "input_black": 0.0,
        "input_white": 1.0,
        "gamma": 1.0,
        "output_black": 0.0,
        "output_white": 1.0,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_levels,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="gamma",
    risk_level="low",
    status_label="正在调整色阶",
    keywords=("色阶", "黑白点", "levels"),
)


__all__ = ["ADJUST_LEVELS_SPEC", "adjust_levels"]
