"""Native adjust_neutral_clean_tone tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_neutral_clean_tone_adjustment


@tool
def adjust_neutral_clean_tone(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    neutral_range: Annotated[float, Field(default=0.2, ge=0.05, le=0.5)] = 0.2,
    yellow_blue_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    green_magenta_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    brightness_shift: Annotated[float, Field(default=0.0, ge=-0.4, le=0.4)] = 0.0,
    protect_skin: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when whites, grays, low-saturation fabrics, or neutral surfaces feel slightly dirty, yellow, green, or magenta. It focuses on neutral-toned areas, so it is good for cleaning cast contamination without broadly restyling saturated colors."""

    output_path = temp_output_path("psagent_neutral_clean_tone_")
    saved_path = apply_neutral_clean_tone_adjustment(
        image_path,
        output_path,
        neutral_range=neutral_range,
        yellow_blue_shift=yellow_blue_shift,
        green_magenta_shift=green_magenta_shift,
        brightness_shift=brightness_shift,
        protect_skin=protect_skin,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_neutral_clean_tone",
        output_image=saved_path,
        applied_params={
            "neutral_range": neutral_range,
            "yellow_blue_shift": yellow_blue_shift,
            "green_magenta_shift": green_magenta_shift,
            "brightness_shift": brightness_shift,
            "protect_skin": protect_skin,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_NEUTRAL_CLEAN_TONE_SPEC = ToolSpec(
    name="adjust_neutral_clean_tone",
    label="中性色净化",
    description="Clean low-saturation neutrals while protecting skin tones.",
    family="color",
    focus_affinity=["subject_separation", "subject_cleanup"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "neutral_range": 0.2,
        "yellow_blue_shift": 0.0,
        "green_magenta_shift": 0.0,
        "brightness_shift": 0.0,
        "protect_skin": 0.3,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_neutral_clean_tone,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="yellow_blue_shift",
    risk_level="low",
    status_label="正在净化中性色",
    keywords=("中性色净化", "灰色净化", "去脏灰", "去偏色"),
)


__all__ = ["ADJUST_NEUTRAL_CLEAN_TONE_SPEC", "adjust_neutral_clean_tone"]
