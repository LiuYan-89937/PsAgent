"""Native adjust_color_overlay tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_overlay_adjustment


@tool
def adjust_color_overlay(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    overlay_hue: Annotated[float, Field(default=0.0, ge=0.0, le=360.0)] = 0.0,
    overlay_saturation: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    overlay_luminance: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    opacity: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    blend_mode: Annotated[str, Field(default="soft_light", description="soft_light | color | overlay")] = "soft_light",
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want to tint an area or the whole image with a controlled overlay color and blend mode. It is useful for stylized color washes or creative local tinting, not for technical white-balance or dirty-color repair."""

    output_path = temp_output_path("psagent_color_overlay_")
    saved_path = apply_color_overlay_adjustment(
        image_path,
        output_path,
        overlay_hue=overlay_hue,
        overlay_saturation=overlay_saturation,
        overlay_luminance=overlay_luminance,
        opacity=opacity,
        blend_mode=blend_mode,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_overlay",
        output_image=saved_path,
        applied_params={
            "overlay_hue": overlay_hue,
            "overlay_saturation": overlay_saturation,
            "overlay_luminance": overlay_luminance,
            "opacity": opacity,
            "blend_mode": blend_mode,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_OVERLAY_SPEC = ToolSpec(
    name="adjust_color_overlay",
    label="局部染色",
    description="Overlay a restrained tint color using a simple blend mode.",
    family="color",
    stage_affinity=["local_balance", "finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "overlay_hue": 0.0,
        "overlay_saturation": 0.0,
        "overlay_luminance": 0.5,
        "opacity": 0.2,
        "blend_mode": "soft_light",
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_color_overlay,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="opacity",
    risk_level="medium",
    status_label="正在进行局部染色",
    keywords=("局部染色", "叠色", "染色", "overlay"),
)


__all__ = ["ADJUST_COLOR_OVERLAY_SPEC", "adjust_color_overlay"]
