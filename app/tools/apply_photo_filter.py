"""Native apply_photo_filter tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_overlay_adjustment


@tool
def apply_photo_filter(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    filter_hue: Annotated[float, Field(default=35.0, ge=0.0, le=360.0)] = 35.0,
    filter_saturation: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    density: Annotated[float, Field(default=0.25, ge=0.0, le=1.0)] = 0.25,
    preserve_luminosity: Annotated[bool, Field(default=True)] = True,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want a restrained warming, cooling, or stylistic color filter over an image or masked area. It is a simple look-building tint and is better for finishing mood than for detailed color correction."""

    output_path = temp_output_path("psagent_photo_filter_")
    saved_path = apply_color_overlay_adjustment(
        image_path,
        output_path,
        overlay_hue=filter_hue,
        overlay_saturation=filter_saturation,
        overlay_luminance=0.92 if preserve_luminosity else 0.7,
        opacity=density,
        blend_mode="soft_light" if preserve_luminosity else "color",
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="apply_photo_filter",
        output_image=saved_path,
        applied_params={
            "filter_hue": filter_hue,
            "filter_saturation": filter_saturation,
            "density": density,
            "preserve_luminosity": preserve_luminosity,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


APPLY_PHOTO_FILTER_SPEC = ToolSpec(
    name="apply_photo_filter",
    label="照片滤镜",
    description="Apply a restrained photo-filter style tint.",
    family="color",
    stage_affinity=["finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "filter_hue": 35.0,
        "filter_saturation": 0.4,
        "density": 0.25,
        "preserve_luminosity": True,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        apply_photo_filter,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="density",
    risk_level="low",
    status_label="正在应用照片滤镜",
    keywords=("照片滤镜", "冷暖滤镜", "暖调", "冷调"),
)


__all__ = ["APPLY_PHOTO_FILTER_SPEC", "apply_photo_filter"]
