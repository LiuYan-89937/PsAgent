"""Native adjust_brightness tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_brightness_adjustment


@tool
def adjust_brightness(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    brightness_offset: Annotated[
        float,
        Field(
            default=0.0,
            ge=-0.6,
            le=0.6,
            description="Tone-mapped brightness shift focused on midtones. Positive values brighten, negative values darken.",
        ),
    ] = 0.0,
    highlight_protection: Annotated[float, Field(default=0.2, ge=0.0, le=0.85, description="Highlight protection strength.")] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image needs a tone-mapped brightness lift or darkening without behaving like raw exposure gain. It focuses more on midtones than on pure exposure, so use it for perceived brightness shaping and not when you want a strong stop-like exposure move."""

    output_path = temp_output_path("psagent_brightness_")
    saved_path = apply_brightness_adjustment(
        image_path,
        output_path,
        brightness_offset=brightness_offset,
        protect_highlights=highlight_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_brightness",
        output_image=saved_path,
        applied_params={
            "brightness_offset": brightness_offset,
            "highlight_protection": highlight_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_BRIGHTNESS_SPEC = ToolSpec(
    name="adjust_brightness",
    label="亮度",
    description="Adjust luminance with restrained highlight protection.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "brightness_offset": 0.0,
        "highlight_protection": 0.2,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_brightness,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="brightness_offset",
    risk_level="low",
    status_label="正在调整亮度",
    keywords=("亮度", "更亮", "更暗", "提亮", "压暗"),
)


__all__ = ["ADJUST_BRIGHTNESS_SPEC", "adjust_brightness"]
