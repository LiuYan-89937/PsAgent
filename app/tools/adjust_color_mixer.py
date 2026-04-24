"""Native adjust_color_mixer tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_mixer_adjustment


@tool
def adjust_color_mixer(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    red_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    red_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    red_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    orange_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    orange_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    orange_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    yellow_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    yellow_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    yellow_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    green_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    green_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    green_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    cyan_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    cyan_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    cyan_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    blue_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    blue_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    blue_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    purple_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    purple_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    purple_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    magenta_hue: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0)] = 0.0,
    magenta_saturation: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    magenta_luminance: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    saturation_protection: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    luminance_protection: Annotated[float, Field(default=0.22, ge=0.0, le=1.0)] = 0.22,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when an entire color band such as blues, greens, yellows, or oranges needs coordinated hue, saturation, or luminance changes. It affects broad color families, so it is better for sky, grass, foliage, or skin-band tuning than for fixing one very specific local color value."""

    output_path = temp_output_path("psagent_color_mixer_")
    channel_settings = {
        "red": {"hue_shift_deg": red_hue, "saturation_shift": red_saturation, "luminance_shift": red_luminance},
        "orange": {"hue_shift_deg": orange_hue, "saturation_shift": orange_saturation, "luminance_shift": orange_luminance},
        "yellow": {"hue_shift_deg": yellow_hue, "saturation_shift": yellow_saturation, "luminance_shift": yellow_luminance},
        "green": {"hue_shift_deg": green_hue, "saturation_shift": green_saturation, "luminance_shift": green_luminance},
        "aqua": {"hue_shift_deg": cyan_hue, "saturation_shift": cyan_saturation, "luminance_shift": cyan_luminance},
        "blue": {"hue_shift_deg": blue_hue, "saturation_shift": blue_saturation, "luminance_shift": blue_luminance},
        "purple": {"hue_shift_deg": purple_hue, "saturation_shift": purple_saturation, "luminance_shift": purple_luminance},
        "magenta": {"hue_shift_deg": magenta_hue, "saturation_shift": magenta_saturation, "luminance_shift": magenta_luminance},
    }
    saved_path = apply_color_mixer_adjustment(
        image_path,
        output_path,
        channel_settings=channel_settings,
        saturation_protection=saturation_protection,
        luminance_protection=luminance_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_mixer",
        output_image=saved_path,
        applied_params={
            **channel_settings,
            "saturation_protection": saturation_protection,
            "luminance_protection": luminance_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_MIXER_SPEC = ToolSpec(
    name="adjust_color_mixer",
    label="颜色混合",
    description="Adjust hue, saturation, and luminance per color band.",
    family="color",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "red_hue": 0.0,
        "red_saturation": 0.0,
        "red_luminance": 0.0,
        "orange_hue": 0.0,
        "orange_saturation": 0.0,
        "orange_luminance": 0.0,
        "yellow_hue": 0.0,
        "yellow_saturation": 0.0,
        "yellow_luminance": 0.0,
        "green_hue": 0.0,
        "green_saturation": 0.0,
        "green_luminance": 0.0,
        "cyan_hue": 0.0,
        "cyan_saturation": 0.0,
        "cyan_luminance": 0.0,
        "blue_hue": 0.0,
        "blue_saturation": 0.0,
        "blue_luminance": 0.0,
        "purple_hue": 0.0,
        "purple_saturation": 0.0,
        "purple_luminance": 0.0,
        "magenta_hue": 0.0,
        "magenta_saturation": 0.0,
        "magenta_luminance": 0.0,
        "saturation_protection": 0.3,
        "luminance_protection": 0.22,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_color_mixer,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="blue_saturation",
    risk_level="medium",
    status_label="正在调整颜色混合",
    keywords=("颜色混合", "HSL", "颜色带", "单色调节"),
)


__all__ = ["ADJUST_COLOR_MIXER_SPEC", "adjust_color_mixer"]
