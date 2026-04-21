"""Native adjust_channel_mixer tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_channel_mixer_adjustment


@tool
def adjust_channel_mixer(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    red_from_red: Annotated[float, Field(default=1.0, ge=-2.0, le=2.0)] = 1.0,
    red_from_green: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    red_from_blue: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    green_from_red: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    green_from_green: Annotated[float, Field(default=1.0, ge=-2.0, le=2.0)] = 1.0,
    green_from_blue: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    blue_from_red: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    blue_from_green: Annotated[float, Field(default=0.0, ge=-2.0, le=2.0)] = 0.0,
    blue_from_blue: Annotated[float, Field(default=1.0, ge=-2.0, le=2.0)] = 1.0,
    monochrome: Annotated[bool, Field(default=False)] = False,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you need explicit per-channel mixing for advanced color reconstruction, calibration-like moves, or custom monochrome conversion. It is a technical tool with broad side effects, so prefer simpler color tools unless you truly need channel-level control."""

    output_path = temp_output_path("psagent_channel_mixer_")
    saved_path = apply_channel_mixer_adjustment(
        image_path,
        output_path,
        red_from_red=red_from_red,
        red_from_green=red_from_green,
        red_from_blue=red_from_blue,
        green_from_red=green_from_red,
        green_from_green=green_from_green,
        green_from_blue=green_from_blue,
        blue_from_red=blue_from_red,
        blue_from_green=blue_from_green,
        blue_from_blue=blue_from_blue,
        monochrome=monochrome,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_channel_mixer",
        output_image=saved_path,
        applied_params={
            "red_from_red": red_from_red,
            "red_from_green": red_from_green,
            "red_from_blue": red_from_blue,
            "green_from_red": green_from_red,
            "green_from_green": green_from_green,
            "green_from_blue": green_from_blue,
            "blue_from_red": blue_from_red,
            "blue_from_green": blue_from_green,
            "blue_from_blue": blue_from_blue,
            "monochrome": monochrome,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_CHANNEL_MIXER_SPEC = ToolSpec(
    name="adjust_channel_mixer",
    label="通道混合器",
    description="Mix output channels from weighted RGB input channels.",
    family="color",
    stage_affinity=["global_base", "finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "red_from_red": 1.0,
        "red_from_green": 0.0,
        "red_from_blue": 0.0,
        "green_from_red": 0.0,
        "green_from_green": 1.0,
        "green_from_blue": 0.0,
        "blue_from_red": 0.0,
        "blue_from_green": 0.0,
        "blue_from_blue": 1.0,
        "monochrome": False,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_channel_mixer,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="red_from_red",
    risk_level="medium",
    status_label="正在调整通道混合器",
    keywords=("通道混合器", "channel mixer"),
)


__all__ = ["ADJUST_CHANNEL_MIXER_SPEC", "adjust_channel_mixer"]
