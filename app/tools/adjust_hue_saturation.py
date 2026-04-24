"""Native adjust_hue_saturation tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_hue_saturation_adjustment


@tool
def adjust_hue_saturation(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    hue_shift: Annotated[float, Field(default=0.0, ge=-180.0, le=180.0, description="Hue shift in degrees.")] = 0.0,
    saturation_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Saturation shift.")] = 0.0,
    lightness_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Lightness/value shift.")] = 0.0,
    protect_skin: Annotated[float, Field(default=0.3, ge=0.0, le=1.0, description="Skin-tone protection.")] = 0.3,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want a broad global color shift, stronger or weaker overall saturation, or a gentle global lightness move. It affects the full image more broadly than color-band tools, so avoid it when only one narrow color family needs correction."""

    output_path = temp_output_path("psagent_hue_saturation_")
    saved_path = apply_hue_saturation_adjustment(
        image_path,
        output_path,
        hue_shift=hue_shift,
        saturation_shift=saturation_shift,
        lightness_shift=lightness_shift,
        protect_skin=protect_skin,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_hue_saturation",
        output_image=saved_path,
        applied_params={
            "hue_shift": hue_shift,
            "saturation_shift": saturation_shift,
            "lightness_shift": lightness_shift,
            "protect_skin": protect_skin,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_HUE_SATURATION_SPEC = ToolSpec(
    name="adjust_hue_saturation",
    label="色相饱和度",
    description="Adjust hue, saturation, and lightness in HSV/HSL space.",
    family="color",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "hue_shift": 0.0,
        "saturation_shift": 0.0,
        "lightness_shift": 0.0,
        "protect_skin": 0.3,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_hue_saturation,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="saturation_shift",
    risk_level="low",
    status_label="正在调整色相和饱和度",
    keywords=("色相", "饱和度", "颜色偏移", "hue", "saturation"),
)


__all__ = ["ADJUST_HUE_SATURATION_SPEC", "adjust_hue_saturation"]
